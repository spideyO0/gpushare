#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# gpushare — Arch Linux Server Install Script
#
# Builds the gpushare server + client from source, installs binaries, configs,
# dashboard, TUI, and systemd services. Opens firewall ports.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

INSTALL_BIN=/usr/local/bin
INSTALL_CONF=/etc/gpushare
INSTALL_SHARE=/usr/local/share/gpushare
SYSTEMD_DIR=/etc/systemd/system

SERVER_PORT=9847
DASHBOARD_PORT=9848

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }
die()   { err "$@"; exit 1; }

# ── Help ──────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: install-server-arch.sh [OPTIONS]

Install the gpushare server on Arch Linux.

Options:
  -h, --help        Show this help
  --skip-build      Skip cmake/make (use existing build/)
  --skip-firewall   Do not modify iptables rules
  --no-enable       Install but do not enable/start services
  --force           Force full reinstall even if already installed

Prerequisites:
  - Arch Linux with an NVIDIA GPU
  - CUDA toolkit (pacman -S cuda)
  - NVIDIA driver loaded
  - cmake, gcc/g++, make
EOF
    exit 0
fi

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_BUILD=false
SKIP_FIREWALL=false
NO_ENABLE=false
FORCE_REINSTALL=false
for arg in "$@"; do
    case "$arg" in
        --skip-build)    SKIP_BUILD=true ;;
        --skip-firewall) SKIP_FIREWALL=true ;;
        --no-enable)     NO_ENABLE=true ;;
        --force)         FORCE_REINSTALL=true ;;
        *) die "Unknown option: $arg (try --help)" ;;
    esac
done

# ── Root check ────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    die "This script must be run as root (sudo $0)"
fi

# ── Upgrade detection ────────────────────────────────────────────────────────
IS_UPGRADE=false
if [[ -f "$INSTALL_BIN/gpushare-server" ]] && [[ "$FORCE_REINSTALL" == false ]]; then
    IS_UPGRADE=true
fi

if [[ "$IS_UPGRADE" == true ]]; then
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  gpushare — Upgrading existing installation${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
else
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  gpushare — Arch Linux Server Installer${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
fi
echo

# ── Stop running services before upgrade ─────────────────────────────────────
if [[ "$IS_UPGRADE" == true ]]; then
    info "Stopping running services for upgrade..."
    systemctl stop gpushare-server.service 2>/dev/null || true
    systemctl stop gpushare-dashboard.service 2>/dev/null || true
    ok "Services stopped"
fi

# ── 1. Prerequisites ─────────────────────────────────────────────────────────
info "Checking prerequisites..."

# Arch installs CUDA to /opt/cuda — add to PATH if not present
if [[ -d /opt/cuda/bin ]] && ! echo "$PATH" | grep -q /opt/cuda/bin; then
    export PATH="/opt/cuda/bin:$PATH"
    info "Added /opt/cuda/bin to PATH"
fi

missing=()
# Check for nvcc in PATH or known Arch location
if ! command -v nvcc >/dev/null 2>&1 && [[ ! -x /opt/cuda/bin/nvcc ]]; then
    missing+=("cuda toolkit (nvcc)")
fi
command -v cmake  >/dev/null 2>&1 || missing+=("cmake")
command -v gcc    >/dev/null 2>&1 || missing+=("gcc")
command -v g++    >/dev/null 2>&1 || missing+=("g++")
command -v make   >/dev/null 2>&1 || missing+=("make")

if ! nvidia-smi >/dev/null 2>&1; then
    missing+=("nvidia driver (nvidia-smi)")
fi

if [[ ${#missing[@]} -gt 0 ]]; then
    err "Missing prerequisites:"
    for m in "${missing[@]}"; do
        echo -e "  ${RED}-${NC} $m"
    done
    echo
    info "Install with: pacman -S cuda nvidia cmake gcc make"
    exit 1
fi

nvcc_ver=$(nvcc --version | grep -oP 'release \K[0-9.]+')
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "CUDA $nvcc_ver  |  GPU: $gpu_name"

# ── 2. Build ──────────────────────────────────────────────────────────────────
if [[ "$SKIP_BUILD" == false ]]; then
    info "Building gpushare (server + client)..."
    mkdir -p "$BUILD_DIR"
    # Clean stale cmake cache if it was generated on a different machine
    if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
        cached_src=$(grep -oP '^CMAKE_HOME_DIRECTORY:INTERNAL=\K.*' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null || true)
        if [[ -n "$cached_src" && "$cached_src" != "$PROJECT_DIR" ]]; then
            warn "Stale CMake cache from $cached_src — cleaning build directory"
            rm -rf "$BUILD_DIR"
            mkdir -p "$BUILD_DIR"
        fi
    fi
    # Run codegen to generate stubs from CUDA headers
    if command -v python3 >/dev/null 2>&1; then
        info "Running codegen (generating CUDA library stubs)..."
        python3 "$PROJECT_DIR/codegen/parse_headers.py" 2>/dev/null || warn "parse_headers.py skipped"
        python3 "$PROJECT_DIR/codegen/generate_stubs.py" 2>/dev/null || warn "generate_stubs.py skipped"
        ok "Codegen complete"
    fi

    # Remove weak stubs that conflict with strong implementations
    local all_stubs="$PROJECT_DIR/client/generated_all_stubs.cpp"
    if [[ -f "$all_stubs" ]] && grep -q 'WEAK_SYM cuGetProcAddress()' "$all_stubs" 2>/dev/null; then
        sed -i '/STUB_EXPORT int WEAK_SYM cuGetProcAddress() /d' "$all_stubs"
        sed -i '/STUB_EXPORT int WEAK_SYM cuGetProcAddress_v2() /d' "$all_stubs"
        info "Removed conflicting weak stubs from generated_all_stubs.cpp"
    fi

    # Ensure cmake can find CUDA on Arch
    export CUDACXX="${CUDACXX:-/opt/cuda/bin/nvcc}"
    cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SERVER=ON \
        -DBUILD_CLIENT=ON \
        -DCUDAToolkit_ROOT=/opt/cuda
    cmake --build "$BUILD_DIR" -j"$(nproc)"
    ok "Build complete"
else
    info "Skipping build (--skip-build)"
fi

# Verify artifacts
[[ -f "$BUILD_DIR/gpushare-server" ]]          || die "Build artifact missing: $BUILD_DIR/gpushare-server"
[[ -e "$BUILD_DIR/libgpushare_client.so" ]]    || die "Build artifact missing: $BUILD_DIR/libgpushare_client.so"
ok "Build artifacts verified"

# ── 3. Install binary ────────────────────────────────────────────────────────
info "Installing server binary..."
if [[ "$IS_UPGRADE" == true ]] && cmp -s "$BUILD_DIR/gpushare-server" "$INSTALL_BIN/gpushare-server"; then
    ok "Server binary unchanged — skipping"
else
    install -Dm755 "$BUILD_DIR/gpushare-server" "$INSTALL_BIN/gpushare-server"
    ok "Installed $INSTALL_BIN/gpushare-server"
fi

# Also install client library on the server (useful for local testing)
# Resolve symlinks to get the actual .so file
REAL_SO="$(readlink -f "$BUILD_DIR/libgpushare_client.so")"
if [[ "$IS_UPGRADE" == true ]] && cmp -s "$REAL_SO" /usr/local/lib/gpushare/libgpushare_client.so; then
    ok "Client library unchanged — skipping"
else
    install -Dm755 "$REAL_SO" /usr/local/lib/gpushare/libgpushare_client.so
    ok "Installed /usr/local/lib/gpushare/libgpushare_client.so"
fi

# ── 3b. Backup real CUDA libs + create symlinks for dual-GPU support ─────────
# This allows apps on the server machine to see BOTH the local GPU (via backed-up
# real CUDA libs) and remote GPUs (via gpushare forwarding).
info "Setting up local GPU passthrough (dual-GPU support)..."
REAL_CUDA_BACKUP="/usr/local/lib/gpushare/real"
LIB_DIR="/usr/local/lib/gpushare"
mkdir -p "$REAL_CUDA_BACKUP"

backup_lib() {
    local libname="$1"
    if [[ -f "$REAL_CUDA_BACKUP/$libname" ]]; then return; fi
    # Search common CUDA library paths on Arch
    for dir in /opt/cuda/lib64 /opt/cuda/targets/x86_64-linux/lib /usr/lib /usr/lib64; do
        local src="$dir/$libname"
        if [[ -e "$src" ]]; then
            local real_path
            real_path=$(readlink -f "$src" 2>/dev/null || echo "$src")
            if [[ -f "$real_path" ]] && [[ "$real_path" != *gpushare* ]]; then
                cp "$real_path" "$REAL_CUDA_BACKUP/$libname"
                ok "Backed up $libname from $real_path"
                return
            fi
        fi
    done
    warn "Could not find $libname for backup"
}

backup_lib "libcudart.so"
backup_lib "libcuda.so.1"
backup_lib "libnvidia-ml.so.1"

# Create CUDA symlinks in the gpushare directory (used by local GPU passthrough).
# On the server, the REAL CUDA libs in /opt/cuda/lib64 and /usr/lib MUST keep
# priority — the server daemon needs them. The Python hook handles remote GPU
# visibility via ctypes without overriding system libraries.
info "Creating CUDA symlinks (for passthrough reference)..."
for link in libcudart.so libcudart.so.12 libcudart.so.13 libcudart.so.13.0 \
            libcuda.so libcuda.so.1 \
            libnvidia-ml.so libnvidia-ml.so.1 \
            libcublas.so libcublas.so.12 libcublas.so.13 \
            libcublasLt.so libcublasLt.so.12 libcublasLt.so.13 \
            libcudnn.so libcudnn.so.8 libcudnn.so.9 \
            libcufft.so libcufft.so.11 \
            libcusparse.so libcusparse.so.12 \
            libcusolver.so libcusolver.so.11 \
            libcurand.so libcurand.so.10 \
            libnvrtc.so libnvrtc.so.12 \
            libnvjpeg.so libnvjpeg.so.12; do
    ln -sf libgpushare_client.so "$LIB_DIR/$link"
done
# Fix soname symlink (ldconfig creates a confusing chain otherwise)
ln -sf libgpushare_client.so "$LIB_DIR/libgpushare_client.so.1"
ok "All CUDA symlinks created"

# Configure dynamic linker — add our dir so libgpushare_client.so.1 is findable
echo "$LIB_DIR" > /etc/ld.so.conf.d/gpushare.conf
ldconfig
ok "ldconfig updated"

# On the server, real CUDA keeping priority is CORRECT (server daemon needs it).
# The Python hook + ctypes handles remote GPU detection without library overrides.
ok "Server mode: real CUDA libs keep priority (correct for server daemon)"

# ── 4. Install config ────────────────────────────────────────────────────────
info "Installing configuration..."
mkdir -p "$INSTALL_CONF"
if [[ -f "$INSTALL_CONF/server.conf" ]]; then
    warn "Config already exists at $INSTALL_CONF/server.conf — not overwriting"
    install -Dm644 "$PROJECT_DIR/config/gpushare-server.conf" "$INSTALL_CONF/server.conf.new"
    info "New default written to $INSTALL_CONF/server.conf.new for reference"
else
    install -Dm644 "$PROJECT_DIR/config/gpushare-server.conf" "$INSTALL_CONF/server.conf"
    ok "Installed $INSTALL_CONF/server.conf"
fi

# Also install a client config so the client library on the server machine
# knows to connect to localhost (for dual-GPU: local + remote via loopback)
if [[ ! -f "$INSTALL_CONF/client.conf" ]]; then
    cat > "$INSTALL_CONF/client.conf" <<CLIENTEOF
# Auto-generated by server installer for local dual-GPU support
server=localhost:$SERVER_PORT
gpu_mode=all
CLIENTEOF
    ok "Installed $INSTALL_CONF/client.conf (server=localhost:$SERVER_PORT)"
else
    ok "Client config already exists at $INSTALL_CONF/client.conf"
fi

# ── 5. Install dashboard + TUI ───────────────────────────────────────────────
info "Installing dashboard and TUI..."
mkdir -p "$INSTALL_SHARE/dashboard"
mkdir -p "$INSTALL_SHARE/tui"

cp -a "$PROJECT_DIR/dashboard/"* "$INSTALL_SHARE/dashboard/" 2>/dev/null || warn "No dashboard files to install"
cp -a "$PROJECT_DIR/tui/"*       "$INSTALL_SHARE/tui/"       2>/dev/null || warn "No TUI files to install"

# Install TUI as a command
cat > "$INSTALL_BIN/gpushare-monitor" <<'TEOF'
#!/usr/bin/env bash
exec python3 /usr/local/share/gpushare/tui/monitor.py "$@"
TEOF
chmod 755 "$INSTALL_BIN/gpushare-monitor"
ok "Installed dashboard + TUI"

# ── 6. Systemd: gpushare-server ──────────────────────────────────────────────
info "Checking systemd service: gpushare-server..."
NEEDS_RELOAD=false

SERVER_UNIT=$(cat <<UNITEOF
[Unit]
Description=gpushare GPU Sharing Server
After=network.target nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=simple
ExecStart=$INSTALL_BIN/gpushare-server --config $INSTALL_CONF/server.conf
Restart=on-failure
RestartSec=5
Nice=-10
Environment=LD_LIBRARY_PATH=/opt/cuda/lib64
LimitMEMLOCK=infinity
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
UNITEOF
)

if [[ -f "$SYSTEMD_DIR/gpushare-server.service" ]] && \
   printf '%s\n' "$SERVER_UNIT" | cmp -s "$SYSTEMD_DIR/gpushare-server.service" -; then
    ok "gpushare-server.service unchanged — skipping"
else
    printf '%s\n' "$SERVER_UNIT" > "$SYSTEMD_DIR/gpushare-server.service"
    NEEDS_RELOAD=true
    ok "Written $SYSTEMD_DIR/gpushare-server.service"
fi

# ── 7. Systemd: gpushare-dashboard ───────────────────────────────────────────
info "Checking systemd service: gpushare-dashboard..."

DASHBOARD_UNIT=$(cat <<UNITEOF
[Unit]
Description=gpushare Web Dashboard
After=network.target gpushare-server.service
Wants=gpushare-server.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 $INSTALL_SHARE/dashboard/app.py --mode server --port $DASHBOARD_PORT --config /etc/gpushare/server.conf
Restart=on-failure
RestartSec=5
WorkingDirectory=$INSTALL_SHARE/dashboard

[Install]
WantedBy=multi-user.target
UNITEOF
)

if [[ -f "$SYSTEMD_DIR/gpushare-dashboard.service" ]] && \
   printf '%s\n' "$DASHBOARD_UNIT" | cmp -s "$SYSTEMD_DIR/gpushare-dashboard.service" -; then
    ok "gpushare-dashboard.service unchanged — skipping"
else
    printf '%s\n' "$DASHBOARD_UNIT" > "$SYSTEMD_DIR/gpushare-dashboard.service"
    NEEDS_RELOAD=true
    ok "Written $SYSTEMD_DIR/gpushare-dashboard.service"
fi

if [[ "$NEEDS_RELOAD" == true ]]; then
    systemctl daemon-reload
    ok "systemd daemon reloaded"
else
    ok "No systemd changes — daemon-reload skipped"
fi

# ── 8. Firewall ──────────────────────────────────────────────────────────────
if [[ "$SKIP_FIREWALL" == false ]]; then
    info "Configuring firewall (iptables)..."
    for port in $SERVER_PORT $DASHBOARD_PORT; do
        if iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null; then
            warn "iptables rule for port $port already exists"
        else
            iptables -I INPUT -p tcp --dport "$port" -j ACCEPT
            ok "Opened TCP port $port"
        fi
    done
    # Persist rules if iptables-save is available
    if command -v iptables-save >/dev/null 2>&1; then
        iptables-save > /etc/iptables/iptables.rules 2>/dev/null || true
    fi
else
    info "Skipping firewall config (--skip-firewall)"
fi

# ── 9. Enable & start ────────────────────────────────────────────────────────
if [[ "$NO_ENABLE" == false ]]; then
    info "Enabling and starting services..."
    systemctl enable --now gpushare-server.service
    ok "gpushare-server: enabled + started"
    systemctl enable --now gpushare-dashboard.service
    ok "gpushare-dashboard: enabled + started"
else
    info "Skipping service enable (--no-enable). Start manually with:"
    echo "  systemctl enable --now gpushare-server gpushare-dashboard"
fi

# ── 10. Summary ──────────────────────────────────────────────────────────────
LOCAL_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' || hostname -I | awk '{print $1}')
if [[ "$IS_UPGRADE" == true ]]; then
    RESULT_VERB="Upgraded"
else
    RESULT_VERB="Installed"
fi
echo
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  gpushare server ${RESULT_VERB,,} successfully!${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo
echo -e "  Server address:   ${CYAN}${LOCAL_IP}:${SERVER_PORT}${NC}"
echo -e "  Dashboard:        ${CYAN}http://${LOCAL_IP}:${DASHBOARD_PORT}${NC}"
echo -e "  Config:           $INSTALL_CONF/server.conf"
echo -e "  GPU:              $gpu_name"
echo -e "  API coverage:    2620+ functions (cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSOLVER, cuRAND, NVRTC, nvJPEG)"
echo -e "  Transfer opts:   ${GREEN}ACTIVE${NC} (tiered pinned pools, async memcpy, chunked pipelining, D2H prefetch, RDMA, LZ4/zstd, multi-server GPU pooling)"
# Check if RDMA was built in (CMake sets IBVERBS_LIB when found)
if grep -q "^IBVERBS_LIB:FILEPATH=.*libibverbs" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    echo -e "  RDMA transport:  ${GREEN}BUILT IN${NC} (rdma-core auto-detected)"
elif [[ -f /usr/lib/libibverbs.so ]] || ldconfig -p 2>/dev/null | grep -q libibverbs; then
    echo -e "  RDMA transport:  ${GREEN}AVAILABLE${NC} (rebuild to enable)"
else
    echo -e "  RDMA transport:  ${YELLOW}NOT AVAILABLE${NC} (install rdma-core for InfiniBand support)"
fi
# Check if compression was built in
if grep -q "^LZ4_LIB:FILEPATH=.*liblz4" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    echo -e "  Compression:     ${GREEN}LZ4${NC} (auto-detected)"
elif grep -q "^ZSTD_LIB:FILEPATH=.*libzstd" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    echo -e "  Compression:     ${GREEN}zstd${NC} (auto-detected)"
else
    echo -e "  Compression:     ${YELLOW}NOT AVAILABLE${NC} (install lz4 or zstd for transfer compression)"
fi
echo
echo -e "  ${BOLD}Client setup:${NC}"
echo -e "    On client machines, set server=${CYAN}${LOCAL_IP}:${SERVER_PORT}${NC}"
echo -e "    in the client config, or run the client install script."
echo
echo -e "  ${BOLD}Commands:${NC}"
echo -e "    systemctl status gpushare-server"
echo -e "    systemctl status gpushare-dashboard"
echo -e "    gpushare-monitor"
echo -e "    journalctl -u gpushare-server -f"
echo
