#!/usr/bin/env python3
"""gpushare TUI monitor -- real-time terminal dashboard for remote GPU sessions.

Works on macOS and Linux with Python 3.7+ using only the standard library.
"""

import argparse
import collections
import curses
import os
import platform
import socket
import struct
import subprocess
import sys
import threading
import time

# ── Protocol constants (mirror include/gpushare/protocol.h) ─────────────
GPUSHARE_MAGIC = 0x47505553
GPUSHARE_HEADER_SIZE = 16
GPUSHARE_DEFAULT_PORT = 9847

GS_OP_INIT = 0x0001
GS_OP_PING = 0x0003
GS_OP_GET_DEVICE_PROPS = 0x0011
GS_OP_GET_STATS = 0x0060

GS_FLAG_RESPONSE = 0x0001
GS_FLAG_ERROR = 0x0002

HEADER_FMT = "<IIIHH"  # magic(4) length(4) req_id(4) opcode(2) flags(2)

# ── Color pair IDs ───────────────────────────────────────────────────────
CP_GREEN = 1
CP_RED = 2
CP_YELLOW = 3
CP_CYAN = 4
CP_STATUS_BAR = 5

# ── Unicode box-drawing helpers ──────────────────────────────────────────
BOX_H = "\u2500"   # ─
BOX_V = "\u2502"   # │
BOX_TL = "\u250c"  # ┌
BOX_TR = "\u2510"  # ┐
BOX_BL = "\u2514"  # └
BOX_BR = "\u2518"  # ┘

SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"  # ▁▂▃▄▅▆▇█

# ── Platform detection ───────────────────────────────────────────────────
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"


# ── Utility ──────────────────────────────────────────────────────────────

def format_bytes(n):
    """Return a human-readable byte string."""
    if n < 1024:
        return "{} B".format(n)
    elif n < 1024 * 1024:
        return "{:.1f} KB".format(n / 1024)
    elif n < 1024 * 1024 * 1024:
        return "{:.1f} MB".format(n / (1024 * 1024))
    else:
        return "{:.2f} GB".format(n / (1024 * 1024 * 1024))


def format_uptime(secs):
    """Return e.g. '2h 34m 12s'."""
    h = int(secs) // 3600
    m = (int(secs) % 3600) // 60
    s = int(secs) % 60
    parts = []
    if h:
        parts.append("{}h".format(h))
    if h or m:
        parts.append("{}m".format(m))
    parts.append("{}s".format(s))
    return " ".join(parts)


def format_number(n):
    """Return number with thousands separators."""
    return "{:,}".format(int(n))


def sparkline(values, width):
    """Build a sparkline string from *values* scaled to *width* characters."""
    if not values:
        return " " * width
    recent = list(values)[-width:]
    mx = max(recent) if max(recent) > 0 else 1.0
    out = []
    for v in recent:
        idx = int((v / mx) * (len(SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        out.append(SPARK_CHARS[idx])
    # pad to width
    line = "".join(out)
    if len(line) < width:
        line += " " * (width - len(line))
    return line


# ── Service management ───────────────────────────────────────────────────

MACOS_PLIST = os.path.expanduser(
    "~/Library/LaunchAgents/com.gpushare.dashboard.plist"
)


def service_start():
    """Start the gpushare service. Returns (success: bool, message: str)."""
    try:
        if IS_LINUX:
            subprocess.check_output(
                ["systemctl", "start", "gpushare-server"],
                stderr=subprocess.STDOUT,
            )
            return True, "Service started (systemctl)"
        elif IS_MACOS:
            subprocess.check_output(
                ["launchctl", "load", MACOS_PLIST],
                stderr=subprocess.STDOUT,
            )
            return True, "Service started (launchctl)"
        else:
            return False, "Unsupported platform: {}".format(platform.system())
    except FileNotFoundError as exc:
        return False, "Command not found: {}".format(exc)
    except subprocess.CalledProcessError as exc:
        out = exc.output.decode("utf-8", errors="replace").strip()
        return False, "Service start failed: {}".format(out or exc)
    except PermissionError:
        return False, "Permission denied -- try running with sudo"


def service_stop():
    """Stop the gpushare service. Returns (success: bool, message: str)."""
    try:
        if IS_LINUX:
            subprocess.check_output(
                ["systemctl", "stop", "gpushare-server"],
                stderr=subprocess.STDOUT,
            )
            return True, "Service stopped (systemctl)"
        elif IS_MACOS:
            subprocess.check_output(
                ["launchctl", "unload", MACOS_PLIST],
                stderr=subprocess.STDOUT,
            )
            return True, "Service stopped (launchctl)"
        else:
            return False, "Unsupported platform: {}".format(platform.system())
    except FileNotFoundError as exc:
        return False, "Command not found: {}".format(exc)
    except subprocess.CalledProcessError as exc:
        out = exc.output.decode("utf-8", errors="replace").strip()
        return False, "Service stop failed: {}".format(out or exc)
    except PermissionError:
        return False, "Permission denied -- try running with sudo"


# ── Config file writing ──────────────────────────────────────────────────

CONFIG_SEARCH_PATHS = [
    os.path.join(os.getcwd(), "gpushare-client.conf"),
    os.path.expanduser("~/.config/gpushare/client.conf"),
    "/etc/gpushare/client.conf",
]

# Preferred path when creating a new config
CONFIG_DEFAULT_PATH = os.path.expanduser("~/.config/gpushare/client.conf")


def load_config(path=None):
    """Return dict of key=value pairs from the first config found."""
    paths = [path] if path else CONFIG_SEARCH_PATHS
    for p in paths:
        if p and os.path.isfile(p):
            cfg = {}
            try:
                with open(p, "r") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            cfg[k.strip()] = v.strip()
            except OSError:
                continue
            return cfg
    return {}


def _find_config_path():
    """Return the path of the first existing config, or the default path."""
    for p in CONFIG_SEARCH_PATHS:
        if p and os.path.isfile(p):
            return p
    return CONFIG_DEFAULT_PATH


def write_config_server(new_addr):
    """Update the server= line in the config file.

    Creates the file and parent directories if they don't exist.
    Returns (success: bool, message: str).
    """
    conf_path = _find_config_path()
    lines = []
    found = False

    if os.path.isfile(conf_path):
        try:
            with open(conf_path, "r") as fh:
                lines = fh.readlines()
        except OSError as exc:
            return False, "Cannot read {}: {}".format(conf_path, exc)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("server") and "=" in stripped:
                lines[i] = "server={}\n".format(new_addr)
                found = True
                break

    if not found:
        lines.append("server={}\n".format(new_addr))

    try:
        parent = os.path.dirname(conf_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(conf_path, "w") as fh:
            fh.writelines(lines)
        return True, "Config updated: {}".format(conf_path)
    except OSError as exc:
        return False, "Cannot write {}: {}".format(conf_path, exc)


# ── Network layer ────────────────────────────────────────────────────────

def build_header(opcode, req_id, payload_len):
    total = GPUSHARE_HEADER_SIZE + payload_len
    return struct.pack(HEADER_FMT, GPUSHARE_MAGIC, total, req_id, opcode, 0)


def recv_exact(sock, n, timeout=5.0):
    """Receive exactly *n* bytes, raising on timeout or disconnect."""
    sock.settimeout(timeout)
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf.extend(chunk)
    return bytes(buf)


def recv_response(sock, timeout=5.0):
    """Read one full gpushare response; returns (opcode, flags, req_id, payload)."""
    hdr_bytes = recv_exact(sock, GPUSHARE_HEADER_SIZE, timeout)
    magic, length, req_id, opcode, flags = struct.unpack(HEADER_FMT, hdr_bytes)
    if magic != GPUSHARE_MAGIC:
        raise ValueError("bad magic 0x{:08X}".format(magic))
    payload_len = length - GPUSHARE_HEADER_SIZE
    payload = b""
    if payload_len > 0:
        payload = recv_exact(sock, payload_len, timeout)
    return opcode, flags, req_id, payload


def parse_address(addr_str):
    """Return (host, port) from 'host:port' or 'host'."""
    if ":" in addr_str:
        host, port_s = addr_str.rsplit(":", 1)
        return host, int(port_s)
    return addr_str, GPUSHARE_DEFAULT_PORT


# ── State container ──────────────────────────────────────────────────────

class MonitorState:
    """Thread-safe container for all data shown in the TUI."""

    def __init__(self, server_addr):
        self.server_addr = server_addr
        self.lock = threading.Lock()

        # connection
        self.connected = False
        self.connect_error = ""
        self.latency_ms = 0.0
        self.session_uptime = 0

        # device props
        self.gpu_name = ""
        self.vram_total = 0     # bytes
        self.vram_used = 0      # bytes  (approximated from alloc)
        self.sm_major = 0
        self.sm_minor = 0
        self.sm_count = 0
        self.max_threads_per_block = 0

        # stats
        self.total_ops = 0
        self.ops_per_sec = 0.0
        self.bytes_in = 0
        self.bytes_out = 0
        self.bw_up = 0.0        # bytes/sec
        self.bw_down = 0.0
        self.alloc_mem = 0

        # bandwidth history (one sample per second, last 120s)
        self.bw_history_up = collections.deque(maxlen=120)
        self.bw_history_down = collections.deque(maxlen=120)

        # log
        self.log_lines = collections.deque(maxlen=200)

        # auto-stop
        self.auto_stop = False
        self.consecutive_ping_failures = 0
        self.last_successful_ping = time.monotonic()

        # internal
        self._prev_ops = 0
        self._prev_bytes_in = 0
        self._prev_bytes_out = 0
        self._prev_time = time.monotonic()

    def add_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        with self.lock:
            self.log_lines.append("[{}] {}".format(ts, msg))

    def clear_log(self):
        with self.lock:
            self.log_lines.clear()


# ── Network worker thread ───────────────────────────────────────────────

class NetworkWorker(threading.Thread):
    """Background thread that connects, polls, and updates *state*."""

    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self._stop_event = threading.Event()
        self._reconnect_event = threading.Event()
        self._req_id = 0

    def stop(self):
        self._stop_event.set()

    def reconnect(self):
        self._reconnect_event.set()

    def update_target(self, new_addr):
        """Update the server address and trigger a reconnect."""
        with self.state.lock:
            self.state.server_addr = new_addr
        self._reconnect_event.set()

    def _next_id(self):
        self._req_id += 1
        return self._req_id

    # ---- high-level ops ----

    def _send_recv(self, sock, opcode, payload=b"", timeout=5.0):
        rid = self._next_id()
        hdr = build_header(opcode, rid, len(payload))
        sock.sendall(hdr + payload)
        r_op, r_flags, r_rid, r_payload = recv_response(sock, timeout)
        if r_flags & GS_FLAG_ERROR:
            raise RuntimeError("server returned error for opcode 0x{:04X}".format(opcode))
        return r_payload

    def _do_init(self, sock):
        # version=1, client_type=1 (python)
        payload = struct.pack("<II", 1, 1)
        resp = self._send_recv(sock, GS_OP_INIT, payload)
        if len(resp) >= 12:
            ver, sid, max_xfer = struct.unpack("<III", resp[:12])
            self.state.add_log("Session {} established (v{})".format(sid, ver))

    def _do_ping(self, sock):
        t0 = time.monotonic()
        self._send_recv(sock, GS_OP_PING, timeout=3.0)
        latency = (time.monotonic() - t0) * 1000.0
        with self.state.lock:
            self.state.latency_ms = latency
            self.state.consecutive_ping_failures = 0
            self.state.last_successful_ping = time.monotonic()

    def _do_get_device_props(self, sock):
        payload = struct.pack("<i", 0)  # device 0
        resp = self._send_recv(sock, GS_OP_GET_DEVICE_PROPS, payload)
        if len(resp) < 256 + 8:
            return
        # Parse gs_device_props_t
        name_raw = resp[:256]
        name = name_raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        off = 256
        (total_global_mem, shared_mem, regs, warp, max_tpb) = struct.unpack_from("<QQiiI", resp, off)
        off += 8 + 8 + 4 + 4 + 4
        # max_threads_dim[3]
        off += 4 * 3
        # max_grid_size[3]
        off += 4 * 3
        # clock_rate
        off += 4
        # major, minor, multi_processor_count
        if off + 12 <= len(resp):
            major, minor, mp_count = struct.unpack_from("<iii", resp, off)
        else:
            major = minor = mp_count = 0

        with self.state.lock:
            self.state.gpu_name = name
            self.state.vram_total = total_global_mem
            self.state.sm_major = major
            self.state.sm_minor = minor
            self.state.sm_count = mp_count
            self.state.max_threads_per_block = max_tpb

        self.state.add_log("GPU: {} ({})".format(name, format_bytes(total_global_mem)))

    def _do_get_stats(self, sock):
        resp = self._send_recv(sock, GS_OP_GET_STATS)
        # gs_stats_header_t: 5 * uint64 + 3 * uint32 = 52 bytes
        if len(resp) < 44:
            return
        (uptime, total_ops, bytes_in, bytes_out, alloc_bytes,
         active_clients, total_conns, num_clients) = struct.unpack_from("<QQQQQIIi", resp, 0)

        now = time.monotonic()
        with self.state.lock:
            dt = now - self.state._prev_time
            if dt > 0:
                self.state.ops_per_sec = (total_ops - self.state._prev_ops) / dt
                self.state.bw_up = (bytes_in - self.state._prev_bytes_in) / dt
                self.state.bw_down = (bytes_out - self.state._prev_bytes_out) / dt
            self.state._prev_ops = total_ops
            self.state._prev_bytes_in = bytes_in
            self.state._prev_bytes_out = bytes_out
            self.state._prev_time = now

            self.state.session_uptime = uptime
            self.state.total_ops = total_ops
            self.state.bytes_in = bytes_in
            self.state.bytes_out = bytes_out
            self.state.alloc_mem = alloc_bytes
            self.state.vram_used = alloc_bytes

            self.state.bw_history_up.append(self.state.bw_up)
            self.state.bw_history_down.append(self.state.bw_down)

    def _handle_ping_failure(self):
        """Track consecutive ping failures and trigger auto-disconnect / auto-stop."""
        with self.state.lock:
            self.state.consecutive_ping_failures += 1
            failures = self.state.consecutive_ping_failures
            auto_stop = self.state.auto_stop
            last_ok = self.state.last_successful_ping

        if failures >= 3:
            with self.state.lock:
                if self.state.connected:
                    self.state.connected = False
            self.state.add_log("Connection lost -- server unreachable")

            # auto-stop: if enabled and unreachable for 30s
            if auto_stop and (time.monotonic() - last_ok) >= 30.0:
                self.state.add_log("Auto-stop: stopping local service")
                ok, msg = service_stop()
                self.state.add_log(msg)

    # ---- main loop ----

    def run(self):
        while not self._stop_event.is_set():
            self._reconnect_event.clear()
            sock = None
            try:
                host, port = parse_address(self.state.server_addr)
                self.state.add_log("Connecting to {}:{}...".format(host, port))
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((host, port))
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                with self.state.lock:
                    self.state.connected = True
                    self.state.connect_error = ""
                    self.state.consecutive_ping_failures = 0
                    self.state.last_successful_ping = time.monotonic()

                self.state.add_log("Connected to server")

                # handshake
                self._do_init(sock)
                self._do_get_device_props(sock)

                # poll loop
                while not self._stop_event.is_set() and not self._reconnect_event.is_set():
                    try:
                        self._do_ping(sock)
                        self._do_get_stats(sock)
                    except (OSError, ConnectionError, struct.error) as exc:
                        self.state.add_log("Poll error: {}".format(exc))
                        self._handle_ping_failure()
                        break
                    # wait 1s between polls, but wake on stop/reconnect
                    self._stop_event.wait(1.0)

            except (OSError, ConnectionError, ValueError, RuntimeError) as exc:
                with self.state.lock:
                    self.state.connected = False
                    self.state.connect_error = str(exc)
                    self.state.consecutive_ping_failures += 1
                self.state.add_log("Connection failed: {}".format(exc))
                self._handle_ping_failure()

            finally:
                if sock:
                    try:
                        sock.close()
                    except OSError:
                        pass
                with self.state.lock:
                    self.state.connected = False

            if not self._stop_event.is_set():
                self.state.add_log("Reconnecting in 3s...")
                self._stop_event.wait(3.0)


# ── Drawing helpers ──────────────────────────────────────────────────────

def draw_box(win, y, x, w, h, title="", color_pair=0):
    """Draw a Unicode box at (y, x) of size w*h with optional title."""
    maxh, maxw = win.getmaxyx()
    if y + h > maxh or x + w > maxw:
        return  # silently clip

    attr = curses.color_pair(color_pair)
    # top
    top_line = BOX_TL + BOX_H
    if title:
        top_line += " " + title + " "
    remaining = w - len(top_line) - 1
    if remaining > 0:
        top_line += BOX_H * remaining
    top_line += BOX_TR
    try:
        win.addstr(y, x, top_line[:w], attr)
    except curses.error:
        pass
    # sides
    for row in range(1, h - 1):
        try:
            win.addstr(y + row, x, BOX_V, attr)
            win.addstr(y + row, x + w - 1, BOX_V, attr)
        except curses.error:
            pass
    # bottom
    bot_line = BOX_BL + BOX_H * (w - 2) + BOX_BR
    try:
        win.addstr(y + h - 1, x, bot_line[:w], attr)
    except curses.error:
        pass


def safe_addstr(win, y, x, text, attr=0):
    """Write text, silently ignoring out-of-bounds errors."""
    maxh, maxw = win.getmaxyx()
    if y < 0 or y >= maxh or x < 0:
        return
    avail = maxw - x - 1
    if avail <= 0:
        return
    try:
        win.addstr(y, x, text[:avail], attr)
    except curses.error:
        pass


def draw_progress_bar(win, y, x, width, fraction, label=""):
    """Draw [████░░░░░] style progress bar."""
    inner = width - 2  # exclude [ and ]
    if inner < 1:
        return
    filled = int(fraction * inner)
    filled = max(0, min(filled, inner))
    empty = inner - filled
    bar = "[" + "\u2588" * filled + "\u2591" * empty + "]"
    if label:
        bar += " " + label
    maxh, maxw = win.getmaxyx()
    avail = maxw - x - 1
    try:
        win.addstr(y, x, bar[:avail])
    except curses.error:
        pass


# ── Inline text input ────────────────────────────────────────────────────

def inline_edit(stdscr, prompt, initial=""):
    """Show a prompt at the bottom of the screen, collect input.

    Returns the entered string, or None if the user pressed Escape.
    """
    curses.curs_set(1)
    stdscr.nodelay(False)  # blocking getch for text input
    h, w = stdscr.getmaxyx()
    row = h - 1
    buf = list(initial)

    while True:
        # draw prompt line
        line = prompt + "".join(buf)
        try:
            stdscr.move(row, 0)
            stdscr.clrtoeol()
            stdscr.addstr(row, 0, line[:w - 1], curses.color_pair(CP_STATUS_BAR))
            stdscr.move(row, min(len(line), w - 2))
        except curses.error:
            pass
        stdscr.refresh()

        ch = stdscr.getch()

        if ch == 27:  # Escape
            curses.curs_set(0)
            stdscr.nodelay(True)
            return None
        elif ch in (curses.KEY_ENTER, 10, 13):
            curses.curs_set(0)
            stdscr.nodelay(True)
            return "".join(buf)
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            if buf:
                buf.pop()
        elif 32 <= ch <= 126:
            buf.append(chr(ch))


# ── Main TUI ─────────────────────────────────────────────────────────────

def tui_main(stdscr, state):
    """curses main loop -- called via curses.wrapper."""
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(250)  # getch timeout in ms for 4 redraws/sec

    # colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(CP_GREEN, curses.COLOR_GREEN, -1)
    curses.init_pair(CP_RED, curses.COLOR_RED, -1)
    curses.init_pair(CP_YELLOW, curses.COLOR_YELLOW, -1)
    curses.init_pair(CP_CYAN, curses.COLOR_CYAN, -1)
    curses.init_pair(CP_STATUS_BAR, curses.COLOR_WHITE, curses.COLOR_BLUE)

    worker = NetworkWorker(state)
    worker.start()

    try:
        while True:
            # input
            ch = stdscr.getch()
            if ch == ord("q") or ch == ord("Q"):
                break
            elif ch == ord("r") or ch == ord("R"):
                state.add_log("Manual reconnect requested")
                worker.reconnect()
            elif ch == ord("c") or ch == ord("C"):
                state.clear_log()
            elif ch == ord("s") or ch == ord("S"):
                state.add_log("Starting service...")
                ok, msg = service_start()
                state.add_log(msg)
            elif ch == ord("x") or ch == ord("X"):
                state.add_log("Stopping service...")
                ok, msg = service_stop()
                state.add_log(msg)
            elif ch == ord("a") or ch == ord("A"):
                with state.lock:
                    state.auto_stop = not state.auto_stop
                    new_val = state.auto_stop
                state.add_log("Auto-stop {}".format(
                    "enabled" if new_val else "disabled"
                ))
            elif ch == ord("e") or ch == ord("E"):
                # enter inline edit mode for server IP
                current_addr = state.server_addr
                new_addr = inline_edit(
                    stdscr, "Server address (host:port): ", current_addr
                )
                stdscr.timeout(250)  # restore after inline_edit
                if new_addr is not None and new_addr.strip():
                    new_addr = new_addr.strip()
                    # update config file
                    ok, msg = write_config_server(new_addr)
                    state.add_log(msg)
                    # update worker target and reconnect
                    state.add_log(
                        "Server address changed to {}".format(new_addr)
                    )
                    worker.update_target(new_addr)
                else:
                    state.add_log("Edit IP cancelled")
            elif ch == curses.KEY_RESIZE:
                stdscr.clear()

            # draw
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            if h < 10 or w < 40:
                safe_addstr(stdscr, 0, 0, "Terminal too small (need 40x10+)")
                stdscr.refresh()
                continue

            with state.lock:
                _draw_all(stdscr, h, w, state)

            stdscr.refresh()
    finally:
        worker.stop()
        worker.join(timeout=2)


def _draw_all(win, h, w, st):
    """Render every panel.  Caller holds st.lock."""
    pw = min(w, 80)  # panel width cap for readability
    y = 0

    # ── Panel 1: Connection info ─────────────────────────────
    box_h = 4
    if y + box_h <= h:
        draw_box(win, y, 0, pw, box_h, "gpushare Monitor", CP_CYAN)
        # row 1
        server_str = "Server: {}".format(st.server_addr)
        if st.connected:
            status_str = "\u25cf Connected"
            status_cp = CP_GREEN
        else:
            status_str = "\u25cf Disconnected"
            status_cp = CP_RED
        safe_addstr(win, y + 1, 2, server_str)
        status_x = pw - len(status_str) - 3
        if status_x > len(server_str) + 4:
            safe_addstr(win, y + 1, status_x, status_str, curses.color_pair(status_cp) | curses.A_BOLD)
        # row 2
        uptime_str = "Uptime: {}".format(format_uptime(st.session_uptime))
        latency_str = "Latency: {:.1f}ms".format(st.latency_ms)
        safe_addstr(win, y + 2, 2, uptime_str)
        lat_x = pw - len(latency_str) - 3
        if lat_x > len(uptime_str) + 4:
            cp = CP_GREEN if st.latency_ms < 5 else (CP_YELLOW if st.latency_ms < 50 else CP_RED)
            safe_addstr(win, y + 2, lat_x, latency_str, curses.color_pair(cp))
        y += box_h

    # ── Panel 2: Remote GPU ──────────────────────────────────
    box_h = 6
    if y + box_h <= h:
        draw_box(win, y, 0, pw, box_h, "Remote GPU", CP_CYAN)
        gpu_name = st.gpu_name if st.gpu_name else "(unknown)"
        safe_addstr(win, y + 1, 2, gpu_name, curses.A_BOLD)

        # VRAM bar
        vram_total_mb = st.vram_total / (1024 * 1024) if st.vram_total else 0
        vram_used_mb = st.vram_used / (1024 * 1024) if st.vram_used else 0
        frac = (vram_used_mb / vram_total_mb) if vram_total_mb > 0 else 0
        bar_w = min(20, pw - 40)
        label = "{:.0f} / {:.0f} MB ({:.0f}%)".format(vram_used_mb, vram_total_mb, frac * 100)
        safe_addstr(win, y + 2, 2, "VRAM: ")
        if bar_w > 4:
            draw_progress_bar(win, y + 2, 8, bar_w, frac, label)

        # SM info
        sm_str = "SM: {}.{}    SMs: {}    Max Threads/Block: {}".format(
            st.sm_major, st.sm_minor, st.sm_count, st.max_threads_per_block
        )
        safe_addstr(win, y + 3, 2, sm_str)
        y += box_h

    # ── Panel 3: Transfer Stats ──────────────────────────────
    box_h = 6
    if y + box_h <= h:
        draw_box(win, y, 0, pw, box_h, "Transfer Stats", CP_CYAN)

        col2 = pw // 2
        safe_addstr(win, y + 1, 2, "Operations:  {}".format(format_number(st.total_ops)))
        safe_addstr(win, y + 1, col2, "Ops/sec: {:.0f}".format(st.ops_per_sec))

        safe_addstr(win, y + 2, 2, "Bytes Sent:  {}".format(format_bytes(st.bytes_in)))
        bw_up_str = "Bandwidth \u2191: {}/s".format(format_bytes(st.bw_up))
        safe_addstr(win, y + 2, col2, bw_up_str)

        safe_addstr(win, y + 3, 2, "Bytes Recv:  {}".format(format_bytes(st.bytes_out)))
        bw_dn_str = "Bandwidth \u2193: {}/s".format(format_bytes(st.bw_down))
        safe_addstr(win, y + 3, col2, bw_dn_str)

        safe_addstr(win, y + 4, 2, "Alloc Mem:   {}".format(format_bytes(st.alloc_mem)))
        y += box_h

    # ── Panel 4: Bandwidth sparkline ─────────────────────────
    box_h = 3
    if y + box_h <= h:
        draw_box(win, y, 0, pw, box_h, "Bandwidth History (last 60s)", CP_CYAN)
        spark_w = pw - 4
        if spark_w > 0:
            combined = [u + d for u, d in zip(st.bw_history_up, st.bw_history_down)]
            line = sparkline(combined, spark_w)
            safe_addstr(win, y + 1, 2, line, curses.color_pair(CP_GREEN))
        y += box_h

    # ── Panel 5: Log ─────────────────────────────────────────
    footer_h = 1
    log_box_h = max(h - y - footer_h, 4)
    if y + log_box_h <= h:
        draw_box(win, y, 0, pw, log_box_h, "Log", CP_CYAN)
        visible = log_box_h - 2
        log_list = list(st.log_lines)
        shown = log_list[-visible:] if len(log_list) > visible else log_list
        for i, line in enumerate(shown):
            safe_addstr(win, y + 1 + i, 2, line[:pw - 4])
        y += log_box_h

    # ── Footer / keybindings ─────────────────────────────────
    if y < h:
        auto_tag = " [AUTO-STOP]" if st.auto_stop else ""
        footer = "  [q]Quit [r]Reconnect [e]Edit IP [s]Start [x]Stop [a]Auto-stop [c]Clear{}  ".format(
            auto_tag
        )
        safe_addstr(win, h - 1, 0, footer.ljust(w - 1), curses.color_pair(CP_STATUS_BAR))


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="gpushare TUI monitor -- real-time remote GPU dashboard"
    )
    parser.add_argument(
        "server", nargs="?", default=None,
        help="Server address as host:port (default: from config or localhost:9847)"
    )
    parser.add_argument(
        "--server", dest="server_flag", default=None, metavar="ADDR",
        help="Server address as host:port"
    )
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="Path to config file"
    )
    args = parser.parse_args()

    # resolve server address
    addr = args.server_flag or args.server
    if not addr:
        cfg = load_config(args.config)
        addr = cfg.get("server", "localhost:{}".format(GPUSHARE_DEFAULT_PORT))

    state = MonitorState(addr)
    state.add_log("gpushare monitor starting")

    try:
        curses.wrapper(lambda stdscr: tui_main(stdscr, state))
    except KeyboardInterrupt:
        pass
    except curses.error as e:
        print(f"Terminal error: {e}", file=sys.stderr)
        print("Try resizing your terminal to at least 80x24.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
