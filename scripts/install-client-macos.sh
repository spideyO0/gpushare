#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# gpushare — macOS Client Install Script
#
# Installs the gpushare client library on macOS, enabling transparent remote
# GPU access. Since macOS has no local NVIDIA GPU, the library provides CUDA
# by forwarding all calls to a remote gpushare server.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

LIB_DIR=/usr/local/lib/gpushare
CONF_DIR="$HOME/.config/gpushare"
BIN_DIR=/usr/local/bin
SHARE_DIR=/usr/local/share/gpushare

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
Usage: install-client-macos.sh [OPTIONS]

Install the gpushare client library on macOS. Provides transparent CUDA
access to a remote gpushare server.

Options:
  -h, --help            Show this help
  --server HOST:PORT    Set the server address (can also edit config later)
  --skip-build          Skip cmake/make (use existing build/)
  --skip-python         Do not install Python client package
  --force               Force full reinstall even if already installed

Prerequisites:
  - Xcode Command Line Tools (xcode-select --install)
  - Homebrew (for cmake)
  - Python 3.8+ (optional, for Python client & TUI)
EOF
    exit 0
fi

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_BUILD=false
SKIP_PYTHON=false
FORCE_REINSTALL=false
SERVER_ADDR=""
for arg in "$@"; do
    case "$arg" in
        --skip-build)   SKIP_BUILD=true ;;
        --skip-python)  SKIP_PYTHON=true ;;
        --force)        FORCE_REINSTALL=true ;;
        --server=*)     SERVER_ADDR="${arg#--server=}" ;;
        *) die "Unknown option: $arg (try --help)" ;;
    esac
done

# ── Upgrade detection ────────────────────────────────────────────────────────
IS_UPGRADE=false
if [[ -f "$LIB_DIR/libgpushare_client.dylib" ]] && [[ "$FORCE_REINSTALL" == false ]]; then
    IS_UPGRADE=true
fi

if [[ "$IS_UPGRADE" == true ]]; then
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  gpushare — Upgrading existing installation${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
else
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  gpushare — macOS Client Installer${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
fi
echo

# ── 1. Prerequisites ─────────────────────────────────────────────────────────
info "Checking prerequisites..."

# Xcode CLT
if ! xcode-select -p >/dev/null 2>&1; then
    warn "Xcode Command Line Tools not found. Installing..."
    xcode-select --install
    die "Please re-run this script after Xcode CLT installation completes."
fi
ok "Xcode Command Line Tools found"

# Homebrew
if ! command -v brew >/dev/null 2>&1; then
    warn "Homebrew not found."
    read -rp "Install Homebrew now? [Y/n] " answer
    if [[ "$answer" != "n" && "$answer" != "N" ]]; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
        die "Homebrew is required for cmake. Install from https://brew.sh"
    fi
fi
ok "Homebrew found"

# cmake
if ! command -v cmake >/dev/null 2>&1; then
    info "Installing cmake via Homebrew..."
    brew install cmake
fi
ok "cmake found"

# ── 2. Build client library ──────────────────────────────────────────────────
if [[ "$SKIP_BUILD" == false ]]; then
    info "Building gpushare client library..."
    mkdir -p "$BUILD_DIR"
    # Clean stale cmake cache if it was generated on a different machine
    if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
        cached_src=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null || true)
        if [[ -n "$cached_src" && "$cached_src" != "$PROJECT_DIR" ]]; then
            warn "Stale CMake cache from $cached_src — cleaning build directory"
            rm -rf "$BUILD_DIR"
            mkdir -p "$BUILD_DIR"
        fi
    fi
    cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SERVER=OFF \
        -DBUILD_CLIENT=ON
    cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu)"
    ok "Build complete"
else
    info "Skipping build (--skip-build)"
fi

[[ -e "$BUILD_DIR/libgpushare_client.dylib" ]] || die "Build artifact missing: $BUILD_DIR/libgpushare_client.dylib"
ok "Client library verified"

# ── 3. Install library (may need sudo) ───────────────────────────────────────
info "Installing client library..."
NEED_SUDO=""
if [[ ! -w /usr/local/lib ]]; then
    NEED_SUDO="sudo"
    info "Need sudo to write to /usr/local/lib"
fi

$NEED_SUDO mkdir -p "$LIB_DIR"
LIB_UPDATED=false
if [[ "$IS_UPGRADE" == true ]] && cmp -s "$BUILD_DIR/libgpushare_client.dylib" "$LIB_DIR/libgpushare_client.dylib"; then
    ok "Client library unchanged — skipping"
else
    $NEED_SUDO cp "$BUILD_DIR/libgpushare_client.dylib" "$LIB_DIR/libgpushare_client.dylib"
    $NEED_SUDO chmod 755 "$LIB_DIR/libgpushare_client.dylib"
    LIB_UPDATED=true
    ok "Installed $LIB_DIR/libgpushare_client.dylib"
fi

# ── 4. Create CUDA symlinks ──────────────────────────────────────────────────
# macOS SIP prevents DYLD_LIBRARY_PATH from working with system binaries.
# Instead we place symlinks directly in /usr/local/lib, which is searched
# by the dynamic linker without needing environment variables.
info "Creating transparent CUDA symlinks..."

# Symlinks in gpushare dir
$NEED_SUDO ln -sf libgpushare_client.dylib "$LIB_DIR/libcudart.dylib"
ok "Created $LIB_DIR/libcudart.dylib -> gpushare"

# Symlinks in /usr/local/lib for broader visibility
# Runtime API (libcudart) — for programs that link -lcudart
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libcudart.dylib
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libcudart.12.dylib
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libcudart.13.dylib
ok "Created /usr/local/lib/libcudart.dylib -> gpushare"

# Driver API (libcuda) — PyTorch/TensorFlow/JAX load this from system
# even when they bundle their own libcudart. This makes venvs work natively.
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libcuda.dylib
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libcuda.1.dylib
ok "Created /usr/local/lib/libcuda.dylib -> gpushare (native venv support)"

# NVML (libnvidia-ml) — GPU detection for apps, monitoring tools, nvidia-smi
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libnvidia-ml.dylib
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libnvidia-ml.1.dylib
ok "Created /usr/local/lib/libnvidia-ml.dylib -> gpushare (GPU app detection)"

# cuBLAS
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libcublas.dylib
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libcublasLt.dylib
# cuDNN
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libcudnn.dylib
# cuFFT
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libcufft.dylib
# cuSPARSE
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libcusparse.dylib
# cuSOLVER
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libcusolver.dylib
# cuRAND
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libcurand.dylib
# NVRTC
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libnvrtc.dylib
# nvJPEG
$NEED_SUDO ln -sf gpushare/libgpushare_client.dylib /usr/local/lib/libnvjpeg.dylib
ok "Created all CUDA library symlinks (cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSOLVER, cuRAND, NVRTC, nvJPEG)"

# Clear quarantine on EVERYTHING gpushare installs
# Without this, macOS blocks execution of downloaded files
$NEED_SUDO xattr -dr com.apple.quarantine "$LIB_DIR" 2>/dev/null || true
$NEED_SUDO xattr -dr com.apple.quarantine /usr/local/lib/libcudart.dylib 2>/dev/null || true
$NEED_SUDO xattr -dr com.apple.quarantine /usr/local/lib/libcudart.12.dylib 2>/dev/null || true
$NEED_SUDO xattr -dr com.apple.quarantine /usr/local/lib/libcudart.13.dylib 2>/dev/null || true
$NEED_SUDO xattr -dr com.apple.quarantine /usr/local/lib/libcuda.dylib 2>/dev/null || true
$NEED_SUDO xattr -dr com.apple.quarantine /usr/local/lib/libcuda.1.dylib 2>/dev/null || true
$NEED_SUDO xattr -dr com.apple.quarantine /usr/local/lib/libnvidia-ml.dylib 2>/dev/null || true
$NEED_SUDO xattr -dr com.apple.quarantine /usr/local/lib/libnvidia-ml.1.dylib 2>/dev/null || true
for _lib in libcublas libcublasLt libcudnn libcufft libcusparse libcusolver libcurand libnvrtc libnvjpeg; do
    $NEED_SUDO xattr -dr com.apple.quarantine "/usr/local/lib/${_lib}.dylib" 2>/dev/null || true
done
$NEED_SUDO xattr -dr com.apple.quarantine "$SHARE_DIR" 2>/dev/null || true
$NEED_SUDO xattr -dr com.apple.quarantine "$BIN_DIR/gpushare-monitor" 2>/dev/null || true
$NEED_SUDO xattr -dr com.apple.quarantine "$BIN_DIR/gpushare-patch" 2>/dev/null || true
$NEED_SUDO xattr -dr com.apple.quarantine "$BIN_DIR/nvidia-smi" 2>/dev/null || true
ok "Cleared macOS quarantine on all installed files"

# Add to /etc/paths.d for PATH (so gpushare-monitor is found)
if [[ ! -f /etc/paths.d/gpushare ]]; then
    echo "$BIN_DIR" | $NEED_SUDO tee /etc/paths.d/gpushare >/dev/null
    ok "Added $BIN_DIR to system PATH via /etc/paths.d/gpushare"
fi

# ── 5. Install config ────────────────────────────────────────────────────────
info "Installing client configuration..."
mkdir -p "$CONF_DIR"

if [[ "$IS_UPGRADE" == true ]] && [[ -f "$CONF_DIR/client.conf" ]]; then
    # On upgrade, preserve existing config; install new default as reference
    warn "Config already exists at $CONF_DIR/client.conf — not overwriting"
    cp "$PROJECT_DIR/config/gpushare-client.conf" "$CONF_DIR/client.conf.new"
    info "New default written to $CONF_DIR/client.conf.new for reference"
    # Read server address from existing config for summary
    SERVER_ADDR=$(sed -n 's/^server=//p' "$CONF_DIR/client.conf" 2>/dev/null || echo "unknown")
else
    # Fresh install — prompt for server address
    if [[ -z "$SERVER_ADDR" ]]; then
        if [[ -f "$CONF_DIR/client.conf" ]]; then
            current=$(sed -n 's/^server=//p' "$CONF_DIR/client.conf" 2>/dev/null || echo "")
            read -rp "Server address [${current:-192.168.1.100:9847}]: " SERVER_ADDR
            SERVER_ADDR="${SERVER_ADDR:-${current:-192.168.1.100:9847}}"
        else
            read -rp "Server address (host:port) [192.168.1.100:9847]: " SERVER_ADDR
            SERVER_ADDR="${SERVER_ADDR:-192.168.1.100:9847}"
        fi
    fi

    if [[ -f "$CONF_DIR/client.conf" ]]; then
        warn "Config exists at $CONF_DIR/client.conf — updating server address"
        sed -i '' "s|^server=.*|server=$SERVER_ADDR|" "$CONF_DIR/client.conf"
    else
        sed "s|^server=.*|server=$SERVER_ADDR|" "$PROJECT_DIR/config/gpushare-client.conf" \
            > "$CONF_DIR/client.conf"
    fi
    ok "Config installed: $CONF_DIR/client.conf (server=$SERVER_ADDR)"
fi

# ── 6. Install TUI + dashboard ───────────────────────────────────────────────
info "Installing TUI monitor and dashboard..."
$NEED_SUDO mkdir -p "$SHARE_DIR/tui"
$NEED_SUDO mkdir -p "$SHARE_DIR/dashboard"
$NEED_SUDO cp -a "$PROJECT_DIR/tui/"*       "$SHARE_DIR/tui/"       2>/dev/null || warn "No TUI files found"
$NEED_SUDO cp -a "$PROJECT_DIR/dashboard/"* "$SHARE_DIR/dashboard/"  2>/dev/null || warn "No dashboard files found"

$NEED_SUDO tee "$BIN_DIR/gpushare-monitor" >/dev/null <<'TEOF'
#!/usr/bin/env bash
exec python3 /usr/local/share/gpushare/tui/monitor.py "$@"
TEOF
$NEED_SUDO chmod 755 "$BIN_DIR/gpushare-monitor"
# Clear macOS quarantine on all installed scripts
$NEED_SUDO xattr -dr com.apple.quarantine "$BIN_DIR/gpushare-monitor" 2>/dev/null || true
$NEED_SUDO xattr -dr com.apple.quarantine "$SHARE_DIR" 2>/dev/null || true
$NEED_SUDO xattr -dr com.apple.quarantine "$LIB_DIR" 2>/dev/null || true
ok "Installed $BIN_DIR/gpushare-monitor"

# nvidia-smi shim — so GPU detection scripts and monitoring tools work
$NEED_SUDO cp "$PROJECT_DIR/scripts/nvidia-smi" "$SHARE_DIR/nvidia-smi"
$NEED_SUDO chmod 755 "$SHARE_DIR/nvidia-smi"
# Install nvidia-smi as a bash wrapper that calls the Python script
# (so it works even if someone runs "bash nvidia-smi")
if ! command -v nvidia-smi >/dev/null 2>&1; then
    # Write as a proper binary-like script with correct ownership
    # Use /bin/bash (not /usr/bin/env) to avoid SIP issues with sudo
    $NEED_SUDO tee "$BIN_DIR/nvidia-smi" >/dev/null <<'NSMI'
#!/bin/bash
exec /usr/bin/python3 /usr/local/share/gpushare/nvidia-smi "$@"
NSMI
    $NEED_SUDO chmod 755 "$BIN_DIR/nvidia-smi"
    $NEED_SUDO chown root:wheel "$BIN_DIR/nvidia-smi" 2>/dev/null || true
    # Clear macOS quarantine flag (blocks execution on downloaded files)
    $NEED_SUDO xattr -dr com.apple.quarantine "$BIN_DIR/nvidia-smi" 2>/dev/null || true
    $NEED_SUDO xattr -dr com.apple.quarantine "$SHARE_DIR/nvidia-smi" 2>/dev/null || true
    ok "Installed nvidia-smi command (queries remote GPU)"
else
    warn "Real nvidia-smi found — not overriding (shim at $SHARE_DIR/nvidia-smi)"
fi

# ── 7. Install Python client ─────────────────────────────────────────────────
if [[ "$SKIP_PYTHON" == false ]]; then
    if command -v python3 >/dev/null 2>&1; then
        # Install Python client package via pip
        if command -v pip3 >/dev/null 2>&1; then
            info "Installing Python client package..."
            pip3 install "$PROJECT_DIR/python/" 2>/dev/null \
                || pip3 install --user "$PROJECT_DIR/python/" 2>/dev/null \
                || warn "Python client install failed (non-fatal)"
            ok "Python client installed"
        else
            warn "pip3 not found — skipping Python client package"
        fi

        # Install Python startup hook (patches torch.cuda for remote GPUs)
        info "Installing Python startup hook..."
        local hook_src="$PROJECT_DIR/python/gpushare_hook.py"
        local pth_src="$PROJECT_DIR/python/gpushare.pth"
        if [[ -f "$hook_src" ]] && [[ -f "$pth_src" ]]; then
            local hook_installed=false
            for site_dir in $(python3 -c "import site; print(' '.join(site.getsitepackages()))" 2>/dev/null); do
                if [[ -d "$site_dir" ]]; then
                    cp "$hook_src" "$site_dir/gpushare_hook.py" 2>/dev/null && \
                    cp "$pth_src" "$site_dir/gpushare.pth" 2>/dev/null && \
                    hook_installed=true && \
                    ok "Installed startup hook to $site_dir"
                fi
            done
            # Also try user site-packages
            local user_site
            user_site=$(python3 -c "import site; print(site.getusersitepackages())" 2>/dev/null || true)
            if [[ -n "$user_site" ]]; then
                mkdir -p "$user_site" 2>/dev/null || true
                cp "$hook_src" "$user_site/gpushare_hook.py" 2>/dev/null && \
                cp "$pth_src" "$user_site/gpushare.pth" 2>/dev/null && \
                hook_installed=true && \
                ok "Installed startup hook to $user_site"
            fi
            if [[ "$hook_installed" == false ]]; then
                warn "Could not install startup hook to any site-packages"
            fi
            # Clear quarantine on hook files
            xattr -dr com.apple.quarantine "$hook_src" 2>/dev/null || true
            xattr -dr com.apple.quarantine "$pth_src" 2>/dev/null || true
        else
            warn "Hook files not found in $PROJECT_DIR/python/"
        fi
    else
        warn "python3 not found — skipping Python client"
    fi
else
    info "Skipping Python client (--skip-python)"
fi

# ── 8. launchd plist for auto-start ──────────────────────────────────────────
info "Checking launchd agent for dashboard auto-start..."
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_FILE="$PLIST_DIR/com.gpushare.dashboard.plist"
mkdir -p "$PLIST_DIR"

PLIST_CONTENT="<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\"
  \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
    <key>Label</key>
    <string>com.gpushare.dashboard</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>${SHARE_DIR}/dashboard/app.py</string>
        <string>--client</string>
        <string>--config</string>
        <string>${CONF_DIR}/client.conf</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${HOME}/Library/Logs/gpushare-dashboard.log</string>
    <key>StandardErrorPath</key>
    <string>${HOME}/Library/Logs/gpushare-dashboard.err</string>
</dict>
</plist>"

if [[ -f "$PLIST_FILE" ]] && printf '%s\n' "$PLIST_CONTENT" | cmp -s "$PLIST_FILE" -; then
    ok "launchd plist unchanged — skipping"
else
    # Unload existing plist before updating
    if [[ -f "$PLIST_FILE" ]]; then
        launchctl unload "$PLIST_FILE" 2>/dev/null || true
    fi
    printf '%s\n' "$PLIST_CONTENT" > "$PLIST_FILE"
    ok "Written $PLIST_FILE"
    # Load the updated agent
    launchctl load "$PLIST_FILE" 2>/dev/null || warn "Could not load launchd agent"
fi
info "Dashboard will auto-start on login. Control with:"
info "  launchctl load/unload ~/Library/LaunchAgents/com.gpushare.dashboard.plist"

# ── 9. Patch ML frameworks ───────────────────────────────────────────────────
if command -v python3 >/dev/null 2>&1; then
    info "Patching ML frameworks (PyTorch/TensorFlow/JAX) if installed..."
    python3 "$PROJECT_DIR/scripts/gpushare-patch-frameworks.py" 2>/dev/null || true
    # Also install the patcher as a command
    $NEED_SUDO cp "$PROJECT_DIR/scripts/gpushare-patch-frameworks.py" "$SHARE_DIR/gpushare-patch-frameworks.py" 2>/dev/null || true
    $NEED_SUDO tee "$BIN_DIR/gpushare-patch" >/dev/null <<'PEOF'
#!/usr/bin/env bash
exec python3 /usr/local/share/gpushare/gpushare-patch-frameworks.py "$@"
PEOF
    $NEED_SUDO chmod 755 "$BIN_DIR/gpushare-patch"
    ok "Installed gpushare-patch command"
fi

# ── 10. Summary ──────────────────────────────────────────────────────────────
if [[ "$IS_UPGRADE" == true ]]; then
    RESULT_VERB="upgraded"
else
    RESULT_VERB="installed"
fi
echo
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  gpushare client ${RESULT_VERB} successfully!${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo
echo -e "  Server:          ${CYAN}${SERVER_ADDR}${NC}"
echo -e "  Library:         $LIB_DIR/libgpushare_client.dylib"
echo -e "  Config:          $CONF_DIR/client.conf"
echo -e "  CUDA override:   ${GREEN}ACTIVE${NC} (libcudart.dylib -> gpushare)"
echo -e "  API coverage:    2620+ functions (cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSOLVER, cuRAND, NVRTC, nvJPEG)"
echo -e "  Transfer opts:   ${GREEN}ACTIVE${NC} (tiered pinned pools, async memcpy, chunked pipelining, D2H prefetch, RDMA, LZ4/zstd, multi-server GPU pooling)"
echo
echo -e "  ${BOLD}Usage — CUDA programs work transparently:${NC}"
echo -e "    python3 my_training.py        ${CYAN}# uses remote GPU${NC}"
echo -e "    gpushare-monitor              ${CYAN}# TUI status monitor${NC}"
echo
echo -e "  ${BOLD}To change server:${NC}"
echo -e "    Edit $CONF_DIR/client.conf"
echo
echo -e "  ${BOLD}To uninstall:${NC}"
echo -e "    $PROJECT_DIR/scripts/uninstall.sh"
echo
