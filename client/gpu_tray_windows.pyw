"""
gpushare GPU System Tray Widget for Windows.

Shows remote GPU stats in the system tray — temperature, utilization,
VRAM usage — with hover tooltip and right-click menu. Provides the
"Task Manager GPU experience" that Windows can't natively show for
remote GPUs.

Requires: pip install pystray pillow

Run: pythonw gpu_tray_windows.pyw
  (pythonw = no console window)
"""

import ctypes
import ctypes.util
import os
import sys
import threading
import time
import struct
import socket

# ---------------------------------------------------------------------------
# GPU data fetching via gpushare protocol (no DLL needed, pure sockets)
# ---------------------------------------------------------------------------

MAGIC = 0x47505553
HEADER_SIZE = 16

def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return buf
        buf += chunk
    return buf

def _send_msg(sock, opcode, req_id, payload=b""):
    total = HEADER_SIZE + len(payload)
    hdr = struct.pack("<IIIHH", MAGIC, total, req_id, opcode, 0)
    sock.sendall(hdr + payload)

def _recv_msg(sock):
    raw = _recv_exact(sock, HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        return None, None
    magic, length, rid, opcode, flags = struct.unpack("<IIIHH", raw)
    if magic != MAGIC:
        return None, None
    pl = length - HEADER_SIZE
    payload = _recv_exact(sock, pl) if pl > 0 else b""
    return opcode, payload


def read_server_from_config():
    paths = []
    appdata = os.environ.get("LOCALAPPDATA", "")
    if appdata:
        paths.append(os.path.join(appdata, "gpushare", "client.conf"))
    paths.append(os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"),
                              "gpushare", "client.conf"))
    home = os.path.expanduser("~")
    paths.append(os.path.join(home, ".config", "gpushare", "client.conf"))

    for p in paths:
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    for line in f:
                        if line.strip().startswith("server="):
                            return line.strip()[7:].strip()
            except Exception:
                pass
    return None


def fetch_gpu_status(host, port):
    """Connect, query device props + live status, return dict."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))

        # Init
        _send_msg(sock, 0x0001, 1, struct.pack("<II", 1, 1))
        _recv_msg(sock)

        # Device props
        _send_msg(sock, 0x0011, 2, struct.pack("<i", 0))
        _, props_data = _recv_msg(sock)

        name = ""
        vram_total = 0
        sm_major = 0
        sm_minor = 0
        sms = 0
        if props_data and len(props_data) >= 352:
            name = props_data[:256].split(b"\x00")[0].decode("utf-8", errors="replace")
            fmt = "<QQ" + "i" * 14 + "QQQ"
            vals = struct.unpack_from(fmt, props_data, 256)
            vram_total = vals[0]
            sm_major = vals[12]
            sm_minor = vals[13]
            sms = vals[14]

        # Live GPU status
        _send_msg(sock, 0x0070, 3)
        _, status_data = _recv_msg(sock)

        mem_total = mem_used = mem_free = 0
        gpu_util = mem_util = temp = power = power_limit = fan = 0
        if status_data and len(status_data) >= 48:
            vals = struct.unpack_from("<QQQIIIIIIii", status_data, 0)
            mem_total = vals[0]
            mem_used = vals[1]
            mem_free = vals[2]
            gpu_util = vals[3]
            mem_util = vals[4]
            temp = vals[5]
            power = vals[6]
            power_limit = vals[7]
            fan = vals[8]

        # Close
        _send_msg(sock, 0x0002, 4)
        sock.close()

        return {
            "connected": True,
            "name": name,
            "vram_total_mb": mem_total / (1024 * 1024) if mem_total else vram_total / (1024 * 1024),
            "vram_used_mb": mem_used / (1024 * 1024),
            "vram_free_mb": mem_free / (1024 * 1024),
            "gpu_util": gpu_util,
            "mem_util": mem_util,
            "temp": temp,
            "power_w": power / 1000,
            "power_limit_w": power_limit / 1000,
            "fan": fan,
            "sm": f"{sm_major}.{sm_minor}",
            "sms": sms,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


# ---------------------------------------------------------------------------
# System Tray Icon
# ---------------------------------------------------------------------------

def create_gpu_icon(temp, util, connected):
    """Create a small icon showing GPU temp/util. Returns PIL Image."""
    from PIL import Image, ImageDraw, ImageFont

    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if not connected:
        # Red circle with X
        draw.ellipse([4, 4, 60, 60], fill=(200, 30, 30, 230))
        draw.line([18, 18, 46, 46], fill="white", width=4)
        draw.line([46, 18, 18, 46], fill="white", width=4)
        return img

    # Background — color based on temp
    if temp > 80:
        bg = (200, 40, 40, 230)    # red hot
    elif temp > 65:
        bg = (200, 150, 30, 230)   # warm yellow
    else:
        bg = (40, 160, 60, 230)    # cool green

    draw.rounded_rectangle([2, 2, 62, 62], radius=10, fill=bg)

    # Draw temperature text
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    temp_text = f"{temp}"
    draw.text((size // 2, 8), temp_text, fill="white", font=font, anchor="mt")
    draw.text((size // 2, 38), f"{util}%", fill="white", font=font_small, anchor="mt")

    # Utilization bar at bottom
    bar_y = 54
    bar_h = 6
    draw.rectangle([6, bar_y, 58, bar_y + bar_h], fill=(255, 255, 255, 80))
    bar_w = int(52 * util / 100) if util > 0 else 0
    if bar_w > 0:
        draw.rectangle([6, bar_y, 6 + bar_w, bar_y + bar_h], fill=(255, 255, 255, 220))

    return img


def main():
    try:
        import pystray
        from PIL import Image
    except ImportError:
        print("Required packages missing. Install with:")
        print("  pip install pystray pillow")
        print("\nThen run: pythonw gpu_tray_windows.pyw")
        input("Press Enter to exit...")
        sys.exit(1)

    # Resolve server
    addr = read_server_from_config()
    if not addr:
        addr = "localhost:9847"
    if ":" in addr:
        host, port = addr.rsplit(":", 1)
        port = int(port)
    else:
        host, port = addr, 9847

    # State
    gpu_data = {"connected": False}
    lock = threading.Lock()
    running = True

    def update_loop():
        nonlocal gpu_data, running
        while running:
            data = fetch_gpu_status(host, port)
            with lock:
                gpu_data = data
            # Update icon
            try:
                connected = data.get("connected", False)
                temp = data.get("temp", 0)
                util = data.get("gpu_util", 0)
                icon.icon = create_gpu_icon(temp, util, connected)

                if connected:
                    icon.title = (
                        f"gpushare: {data.get('name', 'GPU')}\n"
                        f"Temp: {temp}C  |  GPU: {util}%\n"
                        f"VRAM: {data.get('vram_used_mb', 0):.0f}/{data.get('vram_total_mb', 0):.0f} MB\n"
                        f"Power: {data.get('power_w', 0):.0f}W/{data.get('power_limit_w', 0):.0f}W"
                    )
                else:
                    icon.title = f"gpushare: Disconnected\n{data.get('error', '')}"
            except Exception:
                pass
            time.sleep(2)

    # ── Service management ────────────────────────────────────
    def find_config_path():
        paths = [
            os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"), "gpushare", "client.conf"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "gpushare", "client.conf"),
            os.path.join(os.path.expanduser("~"), ".config", "gpushare", "client.conf"),
        ]
        for p in paths:
            if os.path.isfile(p):
                return p
        return paths[0]  # default to ProgramData

    def update_config_server(new_addr):
        conf = find_config_path()
        lines = []
        found = False
        if os.path.isfile(conf):
            with open(conf, "r") as f:
                lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("server="):
                lines[i] = f"server={new_addr}\n"
                found = True
                break
        if not found:
            lines.append(f"server={new_addr}\n")
        os.makedirs(os.path.dirname(conf), exist_ok=True)
        with open(conf, "w") as f:
            f.writelines(lines)
        return conf

    def svc_start():
        try:
            import subprocess
            subprocess.run(
                ["schtasks", "/Run", "/TN", "gpushare-dashboard"],
                capture_output=True, timeout=10
            )
            return True, "Dashboard service started"
        except Exception as e:
            return False, str(e)

    def svc_stop():
        try:
            import subprocess
            subprocess.run(
                ["schtasks", "/End", "/TN", "gpushare-dashboard"],
                capture_output=True, timeout=10
            )
            return True, "Dashboard service stopped"
        except Exception as e:
            return False, str(e)

    # ── Menu callbacks ────────────────────────────────────────
    def on_status(icon, item):
        with lock:
            d = dict(gpu_data)
        if d.get("connected"):
            msg = (
                f"GPU: {d.get('name', '?')}\n"
                f"VRAM: {d.get('vram_used_mb', 0):.0f} / {d.get('vram_total_mb', 0):.0f} MB\n"
                f"GPU Utilization: {d.get('gpu_util', 0)}%\n"
                f"Temperature: {d.get('temp', 0)}C\n"
                f"Power: {d.get('power_w', 0):.1f}W / {d.get('power_limit_w', 0):.1f}W\n"
                f"Fan: {d.get('fan', 0)}%\n"
                f"Compute: SM {d.get('sm', '?')}, {d.get('sms', 0)} SMs\n"
                f"\nServer: {host}:{port}"
            )
        else:
            msg = f"Not connected to gpushare server\n\nServer: {host}:{port}\nError: {d.get('error', '?')}"
        ctypes.windll.user32.MessageBoxW(0, msg, "gpushare GPU Status", 0x40)

    def on_change_server(icon, item):
        nonlocal host, port
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        new_addr = simpledialog.askstring(
            "gpushare - Change Server",
            "Enter server address (host:port):",
            initialvalue=f"{host}:{port}",
            parent=root
        )
        root.destroy()

        if new_addr and new_addr.strip():
            new_addr = new_addr.strip()
            if ":" in new_addr:
                host, port_str = new_addr.rsplit(":", 1)
                port = int(port_str)
            else:
                host = new_addr
                port = 9847
            try:
                conf = update_config_server(f"{host}:{port}")
                ctypes.windll.user32.MessageBoxW(
                    0,
                    f"Server changed to {host}:{port}\nConfig saved: {conf}\n\nReconnecting...",
                    "gpushare", 0x40
                )
            except Exception as e:
                ctypes.windll.user32.MessageBoxW(
                    0,
                    f"Server updated to {host}:{port}\nConfig save failed: {e}",
                    "gpushare", 0x30
                )
            # Force immediate reconnect
            with lock:
                gpu_data["connected"] = False

    def on_start_service(icon, item):
        ok, msg = svc_start()
        flag = 0x40 if ok else 0x10
        ctypes.windll.user32.MessageBoxW(0, msg, "gpushare Service", flag)

    def on_stop_service(icon, item):
        ok, msg = svc_stop()
        flag = 0x40 if ok else 0x10
        ctypes.windll.user32.MessageBoxW(0, msg, "gpushare Service", flag)

    def on_restart_service(icon, item):
        svc_stop()
        time.sleep(1)
        ok, msg = svc_start()
        msg = "Service restarted" if ok else f"Restart failed: {msg}"
        flag = 0x40 if ok else 0x10
        ctypes.windll.user32.MessageBoxW(0, msg, "gpushare Service", flag)

    def on_nvidia_smi(icon, item):
        os.system('start cmd /k "nvidia-smi"')

    def on_dashboard(icon, item):
        import webbrowser
        webbrowser.open(f"http://{host}:9848")

    def on_open_config(icon, item):
        conf = find_config_path()
        if os.path.isfile(conf):
            os.system(f'notepad "{conf}"')
        else:
            ctypes.windll.user32.MessageBoxW(0, f"Config not found:\n{conf}", "gpushare", 0x30)

    def on_quit(icon, item):
        nonlocal running
        running = False
        icon.stop()

    # ── Connection status for menu label ──────────────────────
    def get_connection_text(item):
        with lock:
            d = dict(gpu_data)
        if d.get("connected"):
            return f"Connected: {d.get('name', 'GPU')}"
        return "Disconnected"

    def get_server_text(item):
        return f"Server: {host}:{port}"

    def is_connected(item):
        with lock:
            return gpu_data.get("connected", False)

    # ── Build menu ────────────────────────────────────────────
    initial_icon = create_gpu_icon(0, 0, False)

    menu = pystray.Menu(
        pystray.MenuItem(get_connection_text, on_status, default=True),
        pystray.MenuItem(get_server_text, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("GPU Status", on_status),
        pystray.MenuItem("nvidia-smi", on_nvidia_smi),
        pystray.MenuItem("Web Dashboard", on_dashboard),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Change Server...", on_change_server),
        pystray.MenuItem("Edit Config", on_open_config),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Service", pystray.Menu(
            pystray.MenuItem("Start", on_start_service),
            pystray.MenuItem("Stop", on_stop_service),
            pystray.MenuItem("Restart", on_restart_service),
        )),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )

    icon = pystray.Icon("gpushare", initial_icon, "gpushare: Starting...", menu)

    # Start updater thread
    updater = threading.Thread(target=update_loop, daemon=True)
    updater.start()

    # Run (blocks)
    icon.run()


if __name__ == "__main__":
    main()
