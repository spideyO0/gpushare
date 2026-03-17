#!/usr/bin/env python3
"""
gpushare Dashboard — zero-dependency web UI for GPU sharing monitoring.

Uses only Python stdlib: http.server, json, subprocess, threading, struct, socket.
Serves a single HTML page with all CSS/JS embedded.

Modes:
  Server (default) — shows local GPU stats via nvidia-smi, connected clients
  Client (--client) — shows connection status to a remote gpushare server

Usage:
  python app.py                        # server mode, port 9848
  python app.py --client               # client mode, port 9849
  python app.py --port 8080            # custom port
  python app.py --server 10.0.0.5:9847 # custom gpushare server address
"""

import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GS_MAGIC = 0x47505553  # 'GPUS'
GS_OP_INIT = 0x0001
GS_OP_CLOSE = 0x0002
GS_OP_GET_STATS = 0x0060
# Wire header: magic(4) + length(4) + req_id(4) + opcode(2) + flags(2) = 16 bytes
GS_HEADER_FMT = "<IIIHH"
GS_HEADER_SIZE = 16
GS_FLAG_RESPONSE = 0x0001

DEFAULT_SERVER_PORT = 9848
DEFAULT_CLIENT_PORT = 9849
DEFAULT_GS_SERVER = "localhost:9847"

REFRESH_INTERVAL_SEC = 2

# ---------------------------------------------------------------------------
# State (shared across threads)
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_state = {
    "mode": "server",
    "start_time": time.time(),
    "gpu": None,
    "clients": [],
    "bandwidth_history": [],   # last 60 samples of total bandwidth (bytes/s)
    "memory_history": [],      # last 60 samples of memory used (MiB)
    "ops_total": 0,
    "connected": False,
    "transfer_bytes_in": 0,
    "transfer_bytes_out": 0,
    "error": None,
}

MAX_HISTORY = 60

# ---------------------------------------------------------------------------
# nvidia-smi helper
# ---------------------------------------------------------------------------

def _parse_nvidia_smi():
    """Run nvidia-smi and return a dict of GPU info, or None on failure."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,"
        "utilization.gpu,utilization.memory,temperature.gpu,"
        "power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
        line = out.decode().strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            return None
        return {
            "name": parts[0],
            "memory_total": _safe_float(parts[1]),
            "memory_used": _safe_float(parts[2]),
            "memory_free": _safe_float(parts[3]),
            "utilization_gpu": _safe_float(parts[4]),
            "utilization_memory": _safe_float(parts[5]),
            "temperature": _safe_float(parts[6]),
            "power_draw": _safe_float(parts[7]),
            "power_limit": _safe_float(parts[8]),
        }
    except Exception:
        return None


def _safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0

# ---------------------------------------------------------------------------
# gpushare protocol helper
# ---------------------------------------------------------------------------

def _gs_send_msg(sock, opcode, req_id, payload=b""):
    total = GS_HEADER_SIZE + len(payload)
    hdr = struct.pack(GS_HEADER_FMT, GS_MAGIC, total, req_id, opcode, 0)
    sock.sendall(hdr + payload)

def _gs_recv_msg(sock):
    raw = _recv_exact(sock, GS_HEADER_SIZE)
    if raw is None:
        return None, None, None
    magic, length, req_id, opcode, flags = struct.unpack(GS_HEADER_FMT, raw)
    if magic != GS_MAGIC:
        return None, None, None
    payload_len = length - GS_HEADER_SIZE
    payload = _recv_exact(sock, payload_len) if payload_len > 0 else b""
    return opcode, flags, payload

def _query_gs_server(host, port):
    """Connect, init, query stats, close. Returns a dict or None."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect((host, port))
        # Init handshake
        _gs_send_msg(sock, GS_OP_INIT, 1, struct.pack("<II", 1, 1))
        _gs_recv_msg(sock)  # init response
        # Query stats
        _gs_send_msg(sock, GS_OP_GET_STATS, 2)
        opcode, flags, payload = _gs_recv_msg(sock)
        # Close
        _gs_send_msg(sock, GS_OP_CLOSE, 3)
        sock.close()
        if payload is None or len(payload) < 44:
            return None
        # Parse gs_stats_header_t (packed)
        hdr_fmt = "<QQQQQIIi"  # uptime, total_ops, bytes_in, bytes_out, alloc, active, total_conn, num_clients
        # Actually: Q uptime, Q total_ops, Q bytes_in, Q bytes_out, Q alloc, I active, I total_conn, I num_clients
        hdr_size = struct.calcsize("<QQQQQIII")
        uptime, total_ops, bytes_in, bytes_out, alloc, active, total_conn, num_clients = \
            struct.unpack_from("<QQQQQIII", payload, 0)
        clients = []
        cl_fmt = "<I64sQQQQQ"
        cl_size = struct.calcsize(cl_fmt)
        for i in range(min(num_clients, 32)):
            off = hdr_size + i * cl_size
            if off + cl_size > len(payload):
                break
            sid, addr_raw, mem, ops, bi, bo, conn_s = struct.unpack_from(cl_fmt, payload, off)
            addr = addr_raw.split(b"\x00")[0].decode("utf-8", errors="replace")
            clients.append({
                "session_id": sid, "addr": addr,
                "mem_allocated_mb": mem / (1024*1024),
                "ops": ops, "bytes_in": bi, "bytes_out": bo,
                "connected_secs": conn_s,
            })
        return {
            "uptime": uptime, "total_ops": total_ops,
            "bytes_in": bytes_in, "bytes_out": bytes_out,
            "alloc_mb": alloc / (1024*1024),
            "active_clients": active, "total_connections": total_conn,
            "clients": clients,
        }
    except Exception as e:
        return None


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

# ---------------------------------------------------------------------------
# Background poller
# ---------------------------------------------------------------------------

class Poller(threading.Thread):
    def __init__(self, mode, gs_host, gs_port):
        super().__init__(daemon=True)
        self.mode = mode
        self.gs_host = gs_host
        self.gs_port = gs_port

    def run(self):
        while True:
            try:
                self._tick()
            except Exception:
                pass
            time.sleep(REFRESH_INTERVAL_SEC)

    def _tick(self):
        global _state
        with _state_lock:
            if self.mode == "server":
                self._tick_server()
            else:
                self._tick_client()

    # -- server mode --------------------------------------------------------

    def _tick_server(self):
        gpu = _parse_nvidia_smi()
        _state["gpu"] = gpu
        _state["connected"] = gpu is not None

        # Try to get client info from the gpushare server
        stats = _query_gs_server(self.gs_host, self.gs_port)
        if stats and isinstance(stats, dict):
            _state["clients"] = stats.get("clients", [])
            _state["ops_total"] = stats.get("total_ops", _state["ops_total"])
            _state["transfer_bytes_in"] = stats.get("bytes_in", 0)
            _state["transfer_bytes_out"] = stats.get("bytes_out", 0)
            # Bandwidth: delta from last sample
            prev_in = getattr(self, "_prev_bytes_in", 0)
            prev_out = getattr(self, "_prev_bytes_out", 0)
            cur_in = stats.get("bytes_in", 0)
            cur_out = stats.get("bytes_out", 0)
            bw = (cur_in - prev_in + cur_out - prev_out) / max(REFRESH_INTERVAL_SEC, 1)
            self._prev_bytes_in = cur_in
            self._prev_bytes_out = cur_out
        else:
            _state["clients"] = _state.get("clients", [])
            bw = 0

        _state["bandwidth_history"].append(bw)
        if len(_state["bandwidth_history"]) > MAX_HISTORY:
            _state["bandwidth_history"] = _state["bandwidth_history"][-MAX_HISTORY:]

        mem_used = gpu["memory_used"] if gpu else 0
        _state["memory_history"].append(mem_used)
        if len(_state["memory_history"]) > MAX_HISTORY:
            _state["memory_history"] = _state["memory_history"][-MAX_HISTORY:]

        _state["error"] = None if gpu else "nvidia-smi not available"

    # -- client mode --------------------------------------------------------

    def _tick_client(self):
        stats = _query_gs_server(self.gs_host, self.gs_port)
        if stats and isinstance(stats, dict):
            _state["connected"] = True
            _state["transfer_bytes_in"] = stats.get("bytes_in", _state["transfer_bytes_in"])
            _state["transfer_bytes_out"] = stats.get("bytes_out", _state["transfer_bytes_out"])
            _state["ops_total"] = stats.get("total_ops", _state["ops_total"])
            _state["clients"] = stats.get("clients", [])
            # Bandwidth: delta from last sample
            prev_in = getattr(self, "_prev_bytes_in", 0)
            prev_out = getattr(self, "_prev_bytes_out", 0)
            cur_in = stats.get("bytes_in", 0)
            cur_out = stats.get("bytes_out", 0)
            bw = (cur_in - prev_in + cur_out - prev_out) / max(REFRESH_INTERVAL_SEC, 1)
            self._prev_bytes_in = cur_in
            self._prev_bytes_out = cur_out
            _state["error"] = None
        else:
            _state["connected"] = False
            bw = 0
            _state["error"] = f"Cannot reach gpushare server at {self.gs_host}:{self.gs_port}"

        _state["bandwidth_history"].append(bw)
        if len(_state["bandwidth_history"]) > MAX_HISTORY:
            _state["bandwidth_history"] = _state["bandwidth_history"][-MAX_HISTORY:]

        mem_used = _state["gpu"]["memory_used"] if _state.get("gpu") else 0
        _state["memory_history"].append(mem_used)
        if len(_state["memory_history"]) > MAX_HISTORY:
            _state["memory_history"] = _state["memory_history"][-MAX_HISTORY:]

# ---------------------------------------------------------------------------
# HTML / CSS / JS  (all embedded)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>gpushare Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0d1117;--card:#161b22;--border:#30363d;
  --accent:#58a6ff;--green:#3fb950;--red:#f85149;--yellow:#d29922;
  --text:#c9d1d9;--text-dim:#8b949e;--text-bright:#f0f6fc;
}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:14px;line-height:1.5}
a{color:var(--accent);text-decoration:none}

/* header */
.header{display:flex;align-items:center;justify-content:space-between;padding:16px 24px;border-bottom:1px solid var(--border);background:var(--card)}
.header h1{font-size:20px;font-weight:600;color:var(--text-bright);display:flex;align-items:center;gap:10px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;flex-shrink:0}
.dot.green{background:var(--green);box-shadow:0 0 8px var(--green)}
.dot.red{background:var(--red);box-shadow:0 0 8px var(--red)}
.header .meta{font-size:12px;color:var(--text-dim)}

/* layout */
.container{max-width:1280px;margin:0 auto;padding:20px 24px}
.grid{display:grid;gap:16px}
.grid-2{grid-template-columns:1fr 1fr}
.grid-3{grid-template-columns:1fr 1fr 1fr}
@media(max-width:900px){.grid-2,.grid-3{grid-template-columns:1fr}}

/* cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:20px;position:relative;overflow:hidden}
.card h2{font-size:14px;font-weight:600;color:var(--text-dim);text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}
.card .big{font-size:32px;font-weight:700;color:var(--text-bright)}
.card .unit{font-size:14px;color:var(--text-dim);margin-left:4px}

/* progress bar */
.bar-wrap{height:8px;background:var(--border);border-radius:4px;margin-top:8px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;transition:width .4s ease}
.bar-fill.accent{background:var(--accent)}
.bar-fill.green{background:var(--green)}
.bar-fill.yellow{background:var(--yellow)}
.bar-fill.red{background:var(--red)}

/* stat rows */
.stat-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border)}
.stat-row:last-child{border-bottom:none}
.stat-label{color:var(--text-dim)}
.stat-value{color:var(--text-bright);font-weight:500}

/* table */
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl th{text-align:left;padding:8px 10px;border-bottom:2px solid var(--border);color:var(--text-dim);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.5px}
.tbl td{padding:8px 10px;border-bottom:1px solid var(--border)}
.tbl tr:hover td{background:rgba(88,166,255,.04)}

/* canvas chart */
.chart-wrap{position:relative;width:100%;height:160px}
.chart-wrap canvas{width:100%;height:100%}

/* error banner */
.error-banner{background:rgba(248,81,73,.12);border:1px solid var(--red);border-radius:6px;padding:10px 16px;margin-bottom:16px;color:var(--red);font-size:13px;display:none}
.error-banner.show{display:block}

/* mode badge */
.badge{font-size:11px;padding:2px 8px;border-radius:12px;font-weight:600;text-transform:uppercase}
.badge.server{background:rgba(63,185,80,.15);color:var(--green)}
.badge.client{background:rgba(88,166,255,.15);color:var(--accent)}

/* service buttons */
.svc-btn{padding:8px 16px;border:1px solid var(--border);border-radius:6px;cursor:pointer;font-weight:600;font-size:13px;background:var(--bg);color:var(--text);transition:all .2s}
.svc-btn:hover{border-color:var(--accent);color:var(--text-bright)}
.svc-start:hover{border-color:var(--green);color:var(--green)}
.svc-stop:hover{border-color:var(--red);color:var(--red)}
.svc-restart:hover{border-color:var(--yellow);color:var(--yellow)}
.svc-btn:disabled{opacity:.4;cursor:not-allowed}
</style>
</head>
<body>

<div class="header">
  <h1>
    <span class="dot" id="statusDot"></span>
    gpushare Dashboard
    <span class="badge" id="modeBadge"></span>
  </h1>
  <div class="meta">
    <span id="uptimeLabel">Uptime: —</span> &nbsp;|&nbsp; Refresh: 2 s
  </div>
</div>

<div class="container">

<div class="error-banner" id="errorBanner"></div>

<!-- GPU info -->
<div class="grid grid-3" style="margin-bottom:16px">
  <div class="card" id="gpuCard">
    <h2>GPU</h2>
    <div id="gpuName" class="big" style="font-size:18px">—</div>
    <div style="margin-top:12px">
      <div class="stat-row"><span class="stat-label">VRAM</span><span class="stat-value" id="vramText">—</span></div>
      <div class="bar-wrap"><div class="bar-fill accent" id="vramBar" style="width:0%"></div></div>
    </div>
    <div style="margin-top:10px">
      <div class="stat-row"><span class="stat-label">GPU Util</span><span class="stat-value" id="gpuUtil">—</span></div>
      <div class="bar-wrap"><div class="bar-fill green" id="gpuUtilBar" style="width:0%"></div></div>
    </div>
  </div>

  <div class="card">
    <h2>Temperature &amp; Power</h2>
    <div style="margin-top:4px">
      <div class="stat-row"><span class="stat-label">Temperature</span><span class="stat-value" id="tempVal">—</span></div>
      <div class="stat-row"><span class="stat-label">Power Draw</span><span class="stat-value" id="powerVal">—</span></div>
      <div class="stat-row"><span class="stat-label">Power Limit</span><span class="stat-value" id="powerLimitVal">—</span></div>
      <div class="stat-row"><span class="stat-label">Mem Util</span><span class="stat-value" id="memUtilVal">—</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Operations</h2>
    <div class="big" id="opsTotal">0</div>
    <div class="unit">total ops</div>
    <div style="margin-top:16px">
      <div class="stat-row"><span class="stat-label">Bytes In</span><span class="stat-value" id="bytesIn">0 B</span></div>
      <div class="stat-row"><span class="stat-label">Bytes Out</span><span class="stat-value" id="bytesOut">0 B</span></div>
    </div>
  </div>
</div>

<!-- Charts -->
<div class="grid grid-2" style="margin-bottom:16px">
  <div class="card">
    <h2>Bandwidth (last 60 samples)</h2>
    <div class="chart-wrap"><canvas id="bwChart"></canvas></div>
  </div>
  <div class="card">
    <h2>Memory Usage (last 60 samples)</h2>
    <div class="chart-wrap"><canvas id="memChart"></canvas></div>
  </div>
</div>

<!-- Clients table (server mode) -->
<div class="card" id="clientsCard">
  <h2>Connected Clients</h2>
  <table class="tbl">
    <thead><tr>
      <th>IP Address</th><th>Session ID</th><th>Memory (MiB)</th><th>Ops</th><th>Bandwidth</th><th>Connected</th>
    </tr></thead>
    <tbody id="clientsBody">
      <tr><td colspan="6" style="text-align:center;color:var(--text-dim)">No clients connected</td></tr>
    </tbody>
  </table>
</div>

<!-- Client mode: connection info + controls -->
<div class="card" id="connectionCard" style="display:none">
  <h2>Connection to Remote GPU</h2>
  <div class="stat-row"><span class="stat-label">Server</span><span class="stat-value" id="serverAddr">—</span></div>
  <div class="stat-row"><span class="stat-label">Status</span><span class="stat-value" id="connStatus">Disconnected</span></div>
  <div class="stat-row"><span class="stat-label">Transferred In</span><span class="stat-value" id="xferIn">0 B</span></div>
  <div class="stat-row"><span class="stat-label">Transferred Out</span><span class="stat-value" id="xferOut">0 B</span></div>
</div>

<!-- Client mode: control panel -->
<div class="grid grid-2" id="controlPanel" style="display:none;margin-top:16px">
  <div class="card">
    <h2>Server Address</h2>
    <div style="display:flex;gap:8px;margin-top:8px">
      <input type="text" id="serverInput" placeholder="192.168.1.100:9847"
        style="flex:1;padding:8px 12px;background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--text-bright);font-size:14px;outline:none"
        onfocus="this.style.borderColor='var(--accent)'" onblur="this.style.borderColor='var(--border)'"/>
      <button onclick="changeServer()" style="padding:8px 16px;background:var(--accent);color:#fff;border:none;border-radius:6px;cursor:pointer;font-weight:600;font-size:13px;white-space:nowrap">
        Apply
      </button>
    </div>
    <div id="serverMsg" style="margin-top:8px;font-size:12px;color:var(--text-dim);min-height:18px"></div>
    <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border)">
      <div class="stat-row"><span class="stat-label">Config File</span><span class="stat-value" id="configPath" style="font-size:12px">—</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Service Control</h2>
    <div id="serviceStatus" style="margin-bottom:12px">
      <div class="stat-row"><span class="stat-label">Service</span><span class="stat-value" id="svcState">Unknown</span></div>
      <div class="stat-row"><span class="stat-label">Platform</span><span class="stat-value" id="svcPlatform">—</span></div>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <button onclick="svcAction('start')" class="svc-btn svc-start">Start</button>
      <button onclick="svcAction('stop')" class="svc-btn svc-stop">Stop</button>
      <button onclick="svcAction('restart')" class="svc-btn svc-restart">Restart</button>
      <button onclick="svcAction('status')" class="svc-btn svc-status">Refresh</button>
    </div>
    <div id="svcMsg" style="margin-top:8px;font-size:12px;color:var(--text-dim);min-height:18px"></div>
  </div>
</div>

</div><!-- /container -->

<script>
// ---- Config injected by server ----
const MODE = "{{MODE}}";
const SERVER_ADDR = "{{SERVER_ADDR}}";

// ---- Helpers ----
function fmtBytes(b){
  if(b===0)return"0 B";
  const u=["B","KB","MB","GB","TB"];
  const i=Math.min(Math.floor(Math.log(b)/Math.log(1024)),u.length-1);
  return (b/Math.pow(1024,i)).toFixed(i?1:0)+" "+u[i];
}
function fmtUptime(s){
  const d=Math.floor(s/86400),h=Math.floor((s%86400)/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60);
  let r="";if(d)r+=d+"d ";if(h)r+=h+"h ";if(m)r+=m+"m ";r+=sec+"s";return r;
}

// ---- Chart drawing ----
function drawChart(canvas,data,color,maxLabel,unit){
  const ctx=canvas.getContext("2d");
  const dpr=window.devicePixelRatio||1;
  const rect=canvas.getBoundingClientRect();
  canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
  ctx.scale(dpr,dpr);
  const W=rect.width,H=rect.height;
  ctx.clearRect(0,0,W,H);

  if(!data||data.length===0){
    ctx.fillStyle="#8b949e";ctx.font="12px sans-serif";ctx.textAlign="center";
    ctx.fillText("No data yet",W/2,H/2);return;
  }

  const maxV=Math.max(...data,1);
  const padL=50,padR=10,padT=10,padB=24;
  const gW=W-padL-padR,gH=H-padT-padB;

  // grid lines
  ctx.strokeStyle="#30363d";ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y=padT+gH*(1-i/4);
    ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(padL+gW,y);ctx.stroke();
    ctx.fillStyle="#8b949e";ctx.font="10px sans-serif";ctx.textAlign="right";
    ctx.fillText((maxV*i/4).toFixed(0)+(unit||""),padL-4,y+3);
  }

  // area fill
  ctx.beginPath();
  const step=gW/Math.max(data.length-1,1);
  ctx.moveTo(padL,padT+gH);
  for(let i=0;i<data.length;i++){
    const x=padL+i*step;
    const y=padT+gH*(1-data[i]/maxV);
    ctx.lineTo(x,y);
  }
  ctx.lineTo(padL+(data.length-1)*step,padT+gH);
  ctx.closePath();
  const grad=ctx.createLinearGradient(0,padT,0,padT+gH);
  grad.addColorStop(0,color+"44");grad.addColorStop(1,color+"05");
  ctx.fillStyle=grad;ctx.fill();

  // line
  ctx.beginPath();
  for(let i=0;i<data.length;i++){
    const x=padL+i*step;
    const y=padT+gH*(1-data[i]/maxV);
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  }
  ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();

  // label
  if(maxLabel){
    ctx.fillStyle="#8b949e";ctx.font="10px sans-serif";ctx.textAlign="right";
    ctx.fillText(maxLabel,W-padR,padT+10);
  }
}

// ---- DOM refs ----
const $=id=>document.getElementById(id);
const dot=$("statusDot"),badge=$("modeBadge"),errBanner=$("errorBanner");
const gpuName=$("gpuName"),vramText=$("vramText"),vramBar=$("vramBar");
const gpuUtil=$("gpuUtil"),gpuUtilBar=$("gpuUtilBar");
const tempVal=$("tempVal"),powerVal=$("powerVal"),powerLimitVal=$("powerLimitVal"),memUtilVal=$("memUtilVal");
const opsTotal=$("opsTotal"),bytesIn=$("bytesIn"),bytesOut=$("bytesOut");
const clientsBody=$("clientsBody"),clientsCard=$("clientsCard"),connectionCard=$("connectionCard");
const uptimeLabel=$("uptimeLabel");
const bwCanvas=$("bwChart"),memCanvas=$("memChart");
const serverAddr=$("serverAddr"),connStatus=$("connStatus"),xferIn=$("xferIn"),xferOut=$("xferOut");

// ---- Init ----
badge.textContent=MODE;
badge.className="badge "+MODE;
if(MODE==="client"){
  clientsCard.style.display="none";
  connectionCard.style.display="block";
  $("controlPanel").style.display="grid";
  serverAddr.textContent=SERVER_ADDR;
  $("serverInput").value=SERVER_ADDR;
  // Fetch initial service status
  svcAction("status");
}

// ---- Fetch loop ----
async function poll(){
  try{
    const r=await fetch("/api/stats");
    const d=await r.json();
    update(d);
  }catch(e){
    dot.className="dot red";
    errBanner.textContent="Dashboard fetch error: "+e.message;
    errBanner.classList.add("show");
  }
}

function update(d){
  // connection status
  dot.className=d.connected?"dot green":"dot red";
  if(d.error){errBanner.textContent=d.error;errBanner.classList.add("show");}
  else{errBanner.classList.remove("show");}

  // uptime
  const up=Date.now()/1000-d.start_time;
  uptimeLabel.textContent="Uptime: "+fmtUptime(up);

  // GPU
  const g=d.gpu;
  if(g){
    gpuName.textContent=g.name||"Unknown GPU";
    const pct=g.memory_total?((g.memory_used/g.memory_total)*100):0;
    vramText.textContent=g.memory_used.toFixed(0)+" / "+g.memory_total.toFixed(0)+" MiB";
    vramBar.style.width=pct.toFixed(1)+"%";
    vramBar.className="bar-fill "+(pct>90?"red":pct>70?"yellow":"accent");
    gpuUtil.textContent=g.utilization_gpu.toFixed(0)+"%";
    gpuUtilBar.style.width=g.utilization_gpu.toFixed(1)+"%";
    gpuUtilBar.className="bar-fill "+(g.utilization_gpu>90?"red":g.utilization_gpu>70?"yellow":"green");
    tempVal.textContent=g.temperature.toFixed(0)+" C";
    powerVal.textContent=g.power_draw.toFixed(1)+" W";
    powerLimitVal.textContent=g.power_limit.toFixed(1)+" W";
    memUtilVal.textContent=g.utilization_memory.toFixed(0)+"%";
  }else{
    gpuName.textContent="No GPU detected";
    vramText.textContent="—";vramBar.style.width="0%";
    gpuUtil.textContent="—";gpuUtilBar.style.width="0%";
    tempVal.textContent="—";powerVal.textContent="—";powerLimitVal.textContent="—";memUtilVal.textContent="—";
  }

  // ops
  opsTotal.textContent=d.ops_total.toLocaleString();
  bytesIn.textContent=fmtBytes(d.transfer_bytes_in||0);
  bytesOut.textContent=fmtBytes(d.transfer_bytes_out||0);

  // clients (server mode)
  if(MODE==="server"){
    const clients=d.clients||[];
    if(clients.length===0){
      clientsBody.innerHTML='<tr><td colspan="6" style="text-align:center;color:var(--text-dim)">No clients connected</td></tr>';
    }else{
      clientsBody.innerHTML=clients.map(c=>{
        const bw=(c.bytes_in||0)+(c.bytes_out||0);
        const conn=c.connected_secs?fmtUptime(c.connected_secs):"—";
        return `<tr><td>${esc(c.addr||c.ip||"—")}</td><td>${c.session_id||c.session||"—"}</td><td>${(c.mem_allocated_mb||c.memory||0).toFixed(0)}</td><td>${(c.ops||0).toLocaleString()}</td><td>${fmtBytes(bw)}</td><td>${conn}</td></tr>`;
      }).join("");
    }
  }

  // client mode connection info
  if(MODE==="client"){
    connStatus.textContent=d.connected?"Connected":"Disconnected";
    connStatus.style.color=d.connected?"var(--green)":"var(--red)";
    xferIn.textContent=fmtBytes(d.transfer_bytes_in||0);
    xferOut.textContent=fmtBytes(d.transfer_bytes_out||0);
  }

  // charts
  const bwData=d.bandwidth_history||[];
  const memData=d.memory_history||[];
  const memTotal=g?g.memory_total:0;
  drawChart(bwCanvas,bwData,"#58a6ff",bwData.length?"Peak: "+fmtBytes(Math.max(...bwData))+"/s":"",""  );
  drawChart(memCanvas,memData,"#3fb950",memTotal?"Total: "+memTotal.toFixed(0)+" MiB":"", " MiB");
}

function esc(s){const d=document.createElement("div");d.textContent=s;return d.innerHTML;}

// ---- Server address change ----
async function changeServer(){
  const addr=$("serverInput").value.trim();
  const msg=$("serverMsg");
  if(!addr){msg.textContent="Enter a server address";msg.style.color="var(--red)";return;}
  msg.textContent="Applying...";msg.style.color="var(--text-dim)";
  try{
    const r=await fetch("/api/server",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({server:addr})});
    const d=await r.json();
    if(d.ok){
      msg.textContent="Server changed to "+addr+(d.config_updated?" (config saved)":"");
      msg.style.color="var(--green)";
      serverAddr.textContent=addr;
    }else{
      msg.textContent="Error: "+(d.error||"unknown");msg.style.color="var(--red)";
    }
  }catch(e){msg.textContent="Request failed: "+e.message;msg.style.color="var(--red)";}
}
$("serverInput").addEventListener("keydown",e=>{if(e.key==="Enter")changeServer();});

// ---- Service control ----
async function svcAction(action){
  const msg=$("svcMsg");
  const state=$("svcState");
  if(action!=="status"){msg.textContent=action+"ing...";msg.style.color="var(--text-dim)";}
  try{
    const r=await fetch("/api/service",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action:action})});
    const d=await r.json();
    state.textContent=d.status||"Unknown";
    state.style.color=d.running?"var(--green)":"var(--red)";
    $("svcPlatform").textContent=d.platform||"—";
    $("configPath").textContent=d.config_path||"—";
    if(action!=="status"){
      msg.textContent=d.message||("Service "+action+" done");
      msg.style.color=d.ok?"var(--green)":"var(--red)";
    }
  }catch(e){
    if(action!=="status"){msg.textContent="Error: "+e.message;msg.style.color="var(--red)";}
    state.textContent="Unknown";state.style.color="var(--text-dim)";
  }
}

poll();
setInterval(poll,2000);

// redraw charts on resize
window.addEventListener("resize",()=>{poll();});
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Config update + Service management helpers
# ---------------------------------------------------------------------------

def _find_config_file():
    """Find the client config file path."""
    import platform
    system = platform.system()
    candidates = []
    if system == "Windows":
        appdata = os.environ.get("LOCALAPPDATA", "")
        if appdata:
            candidates.append(os.path.join(appdata, "gpushare", "client.conf"))
        candidates.append(r"C:\ProgramData\gpushare\client.conf")
    else:
        home = os.path.expanduser("~")
        candidates.append(os.path.join(home, ".config", "gpushare", "client.conf"))
        candidates.append("/etc/gpushare/client.conf")

    for path in candidates:
        if os.path.isfile(path):
            return path
    # Return first writable candidate for creation
    for path in candidates:
        parent = os.path.dirname(path)
        if os.path.isdir(parent) and os.access(parent, os.W_OK):
            return path
    return candidates[0] if candidates else None


def _update_config_server(new_addr):
    """Update the server= line in the config file. Returns True if written."""
    path = _find_config_file()
    if not path:
        return False
    try:
        lines = []
        found = False
        if os.path.isfile(path):
            with open(path, "r") as f:
                for line in f:
                    if line.strip().startswith("server="):
                        lines.append(f"server={new_addr}\n")
                        found = True
                    else:
                        lines.append(line)
        if not found:
            lines.append(f"server={new_addr}\n")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.writelines(lines)
        return True
    except Exception:
        return False


def _manage_service(action):
    """Start/stop/restart/status the gpushare-related service. Cross-platform."""
    import platform
    system = platform.system()
    result = {"ok": True, "platform": system, "config_path": _find_config_file() or "not found"}

    if system == "Linux":
        result.update(_manage_linux_service(action))
    elif system == "Darwin":
        result.update(_manage_macos_service(action))
    elif system == "Windows":
        result.update(_manage_windows_service(action))
    else:
        result.update({"ok": False, "status": "Unsupported", "running": False,
                        "message": f"Unknown platform: {system}"})
    return result


def _manage_linux_service(action):
    """Manage gpushare via systemd (Linux)."""
    # Could be server or client dashboard service
    # Check which services exist
    svc_names = ["gpushare-server", "gpushare-dashboard"]
    active_svc = None
    for svc in svc_names:
        try:
            out = subprocess.check_output(
                ["systemctl", "is-active", svc], stderr=subprocess.DEVNULL, timeout=5
            ).decode().strip()
            if out == "active":
                active_svc = svc
                break
        except Exception:
            pass

    if active_svc is None:
        # Check if any service unit exists
        for svc in svc_names:
            try:
                subprocess.check_output(
                    ["systemctl", "cat", svc], stderr=subprocess.DEVNULL, timeout=5
                )
                active_svc = svc
                break
            except Exception:
                pass

    if active_svc is None:
        return {"status": "Not installed", "running": False,
                "message": "No gpushare systemd service found"}

    if action == "status":
        try:
            out = subprocess.check_output(
                ["systemctl", "is-active", active_svc], stderr=subprocess.DEVNULL, timeout=5
            ).decode().strip()
            running = out == "active"
        except Exception:
            running = False
        return {"status": "Running" if running else "Stopped", "running": running,
                "service": active_svc}

    cmd_map = {"start": "start", "stop": "stop", "restart": "restart"}
    if action not in cmd_map:
        return {"ok": False, "message": f"Unknown action: {action}"}

    try:
        subprocess.check_call(
            ["systemctl", cmd_map[action], active_svc],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=10
        )
        # Check new status
        try:
            out = subprocess.check_output(
                ["systemctl", "is-active", active_svc], stderr=subprocess.DEVNULL, timeout=5
            ).decode().strip()
            running = out == "active"
        except Exception:
            running = action == "start"
        return {"ok": True, "status": "Running" if running else "Stopped",
                "running": running, "service": active_svc,
                "message": f"Service {active_svc} {action}ed"}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "status": "Error", "running": False,
                "message": f"systemctl {action} failed (exit {e.returncode}). May need sudo."}
    except Exception as e:
        return {"ok": False, "status": "Error", "running": False,
                "message": str(e)}


def _manage_macos_service(action):
    """Manage gpushare via launchctl (macOS)."""
    plist = os.path.expanduser("~/Library/LaunchAgents/com.gpushare.dashboard.plist")
    label = "com.gpushare.dashboard"

    if not os.path.isfile(plist):
        return {"status": "Not installed", "running": False,
                "message": "No launchd plist found"}

    if action == "status":
        try:
            out = subprocess.check_output(
                ["launchctl", "list"], stderr=subprocess.DEVNULL, timeout=5
            ).decode()
            running = label in out
        except Exception:
            running = False
        return {"status": "Running" if running else "Stopped", "running": running}

    try:
        if action == "start":
            subprocess.check_call(["launchctl", "load", plist],
                                   stderr=subprocess.DEVNULL, timeout=10)
        elif action == "stop":
            subprocess.check_call(["launchctl", "unload", plist],
                                   stderr=subprocess.DEVNULL, timeout=10)
        elif action == "restart":
            subprocess.call(["launchctl", "unload", plist],
                             stderr=subprocess.DEVNULL, timeout=10)
            subprocess.check_call(["launchctl", "load", plist],
                                   stderr=subprocess.DEVNULL, timeout=10)

        running = action != "stop"
        return {"ok": True, "status": "Running" if running else "Stopped",
                "running": running, "message": f"Service {action}ed"}
    except Exception as e:
        return {"ok": False, "status": "Error", "running": False, "message": str(e)}


def _manage_windows_service(action):
    """Manage gpushare on Windows via scheduled tasks."""
    task_name = "gpushare-dashboard"
    if action == "status":
        try:
            out = subprocess.check_output(
                ["schtasks", "/Query", "/TN", task_name, "/FO", "CSV"],
                stderr=subprocess.DEVNULL, timeout=5
            ).decode()
            running = "Running" in out
            return {"status": "Running" if running else "Stopped", "running": running}
        except Exception:
            return {"status": "Not installed", "running": False}

    cmd_map = {"start": "/Run", "stop": "/End", "restart": None}
    if action == "restart":
        subprocess.call(["schtasks", "/End", "/TN", task_name],
                         stderr=subprocess.DEVNULL, timeout=5)
        subprocess.call(["schtasks", "/Run", "/TN", task_name],
                         stderr=subprocess.DEVNULL, timeout=5)
        return {"ok": True, "status": "Running", "running": True, "message": "Restarted"}

    try:
        subprocess.check_call(
            ["schtasks", cmd_map[action], "/TN", task_name],
            stderr=subprocess.DEVNULL, timeout=10
        )
        running = action == "start"
        return {"ok": True, "status": "Running" if running else "Stopped",
                "running": running, "message": f"Service {action}ed"}
    except Exception as e:
        return {"ok": False, "status": "Error", "running": False, "message": str(e)}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):
    """Serve the dashboard page and JSON API."""

    server_version = "gpushare-dashboard/1.0"

    def log_message(self, fmt, *args):
        # quieter logging — single line
        sys.stderr.write("[dashboard] %s %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._serve_html()
        elif path == "/api/stats":
            self._serve_stats()
        elif path == "/api/gpu":
            self._serve_gpu()
        else:
            self.send_error(404)

    def do_POST(self):
        path = self.path.split("?")[0]
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b""
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        if path == "/api/server":
            self._handle_change_server(data)
        elif path == "/api/service":
            self._handle_service(data)
        else:
            self.send_error(404)

    # -- responses ----------------------------------------------------------

    def _serve_html(self):
        html = DASHBOARD_HTML.replace("{{MODE}}", _state["mode"])
        html = html.replace("{{SERVER_ADDR}}", _gs_server_addr)
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_stats(self):
        with _state_lock:
            data = {
                "mode": _state["mode"],
                "start_time": _state["start_time"],
                "connected": _state["connected"],
                "gpu": _state["gpu"],
                "clients": _state["clients"],
                "bandwidth_history": list(_state["bandwidth_history"]),
                "memory_history": list(_state["memory_history"]),
                "ops_total": _state["ops_total"],
                "transfer_bytes_in": _state["transfer_bytes_in"],
                "transfer_bytes_out": _state["transfer_bytes_out"],
                "error": _state["error"],
                "server_addr": _gs_server_addr,
            }
        self._json_response(data)

    def _serve_gpu(self):
        with _state_lock:
            data = _state["gpu"] or {}
        self._json_response(data)

    def _json_response(self, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    # -- POST: change server address -----------------------------------------

    def _handle_change_server(self, data):
        global _gs_server_addr
        addr = data.get("server", "").strip()
        if not addr:
            self._json_response({"ok": False, "error": "No server address provided"})
            return

        # Parse host:port
        if ":" in addr:
            host, port_str = addr.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                self._json_response({"ok": False, "error": "Invalid port"})
                return
        else:
            host, port = addr, 9847
            addr = f"{host}:{port}"

        # Update global state
        _gs_server_addr = addr

        # Update the poller's target
        for t in threading.enumerate():
            if isinstance(t, Poller):
                t.gs_host = host
                t.gs_port = port
                break

        # Try to update the config file
        config_updated = _update_config_server(addr)

        # Reset connection state
        with _state_lock:
            _state["connected"] = False
            _state["error"] = f"Reconnecting to {addr}..."

        self._json_response({"ok": True, "server": addr, "config_updated": config_updated})

    # -- POST: service control -----------------------------------------------

    def _handle_service(self, data):
        action = data.get("action", "status")
        result = _manage_service(action)
        self._json_response(result)

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

_gs_server_addr = DEFAULT_GS_SERVER  # module-level for template injection


def main():
    global _gs_server_addr

    parser = argparse.ArgumentParser(
        description="gpushare Dashboard — zero-dependency web UI"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="HTTP port for the dashboard (default: 9848 server, 9849 client)",
    )
    parser.add_argument(
        "--server", type=str, default=DEFAULT_GS_SERVER,
        help="gpushare server address HOST:PORT (default: localhost:9847)",
    )
    parser.add_argument(
        "--client", action="store_true",
        help="Run in client mode",
    )
    parser.add_argument(
        "--mode", type=str, choices=["server", "client"], default=None,
        help="Explicit mode selection (alternative to --client flag)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to gpushare client.conf (reads server= address from it)",
    )
    args = parser.parse_args()

    # Read server address from config file if --config is given
    config_parsed_server = None
    if args.config and os.path.isfile(args.config):
        try:
            with open(args.config) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("server="):
                        addr = line[7:].strip()
                        if addr:
                            config_parsed_server = addr
        except Exception:
            pass
    if config_parsed_server and args.server == DEFAULT_GS_SERVER:
        args.server = config_parsed_server

    # In server mode: also try reading server.conf to find the bind address
    # (the server might bind to a specific IP, not localhost)
    if args.server == DEFAULT_GS_SERVER:
        for cfg_path in ["/etc/gpushare/server.conf"]:
            if os.path.isfile(cfg_path):
                try:
                    bind_addr = None
                    srv_port = "9847"
                    with open(cfg_path) as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("bind_address="):
                                bind_addr = line[13:].strip()
                            elif line.startswith("port="):
                                srv_port = line[5:].strip()
                    if bind_addr and bind_addr not in ("", "0.0.0.0"):
                        args.server = f"{bind_addr}:{srv_port}"
                except Exception:
                    pass

    # Determine mode
    mode = "client" if args.client else (args.mode or "server")
    # Auto-detect: if --config is given, default to client mode
    if args.config and not args.client and args.mode is None:
        mode = "client"
    _state["mode"] = mode

    # Parse server address
    _gs_server_addr = args.server
    if ":" in args.server:
        gs_host, gs_port_str = args.server.rsplit(":", 1)
        gs_port = int(gs_port_str)
    else:
        gs_host = args.server
        gs_port = 9847

    # Determine dashboard port
    if args.port is not None:
        port = args.port
    else:
        port = DEFAULT_CLIENT_PORT if mode == "client" else DEFAULT_SERVER_PORT

    # Start background poller
    poller = Poller(mode, gs_host, gs_port)
    poller.start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"gpushare dashboard ({mode} mode) listening on http://0.0.0.0:{port}")
    print(f"  gpushare server: {gs_host}:{gs_port}")
    print(f"  Refresh interval: {REFRESH_INTERVAL_SEC}s")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard.")
        server.shutdown()


if __name__ == "__main__":
    main()
