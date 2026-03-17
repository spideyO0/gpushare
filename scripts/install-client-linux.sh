#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# gpushare — Linux Client Install Script
#
# Installs the gpushare client library as a TRANSPARENT CUDA replacement.
# Applications will find libcudart.so pointing to gpushare, so any CUDA
# program automatically uses the remote GPU with zero code changes.
#
# Supports: Debian/Ubuntu, Fedora/RHEL/CentOS, Arch/Manjaro, openSUSE,
#           Alpine, Void, Gentoo, Clear Linux, and more.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

LIB_DIR=/usr/local/lib/gpushare
CONF_DIR=/etc/gpushare
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
Usage: install-client-linux.sh [OPTIONS]

Install the gpushare client library on Linux. This makes remote GPU access
transparent — programs link against libcudart.so which is actually gpushare.

Options:
  -h, --help            Show this help
  --server HOST:PORT    Set the server address (can also edit config later)
  --skip-build          Skip cmake/make (use existing build/)
  --skip-python         Do not install Python client package
  --no-symlinks         Install library but do not create CUDA symlinks
  --force               Force full reinstall even if already installed
  --auto-deps           Install missing build deps without prompting

Prerequisites (auto-detected and installable):
  - cmake, gcc/g++, make
  - Python 3.8+ (optional, for Python client & TUI)

Supported distros:
  Debian/Ubuntu, Fedora/RHEL/CentOS, Arch/Manjaro, openSUSE,
  Alpine, Void, Gentoo, Clear Linux, and others
EOF
    exit 0
fi

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_BUILD=false
SKIP_PYTHON=false
NO_SYMLINKS=false
FORCE_REINSTALL=false
AUTO_DEPS=false
SERVER_ADDR=""
for arg in "$@"; do
    case "$arg" in
        --skip-build)   SKIP_BUILD=true ;;
        --skip-python)  SKIP_PYTHON=true ;;
        --no-symlinks)  NO_SYMLINKS=true ;;
        --force)        FORCE_REINSTALL=true ;;
        --auto-deps)    AUTO_DEPS=true ;;
        --server=*)     SERVER_ADDR="${arg#--server=}" ;;
        --server)       shift_next=true ;;
        *) if [[ "${shift_next:-}" == true ]]; then SERVER_ADDR="$arg"; shift_next=false;
           else die "Unknown option: $arg (try --help)"; fi ;;
    esac
done

# ── Root check ────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    die "This script must be run as root (sudo $0)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Distro Detection
# ══════════════════════════════════════════════════════════════════════════════
DISTRO_FAMILY=""   # debian, fedora, arch, suse, alpine, void, gentoo, clearlinux, unknown
DISTRO_ID=""       # exact ID from os-release (e.g., ubuntu, fedora, manjaro)

detect_distro() {
    # Method 1: /etc/os-release (most reliable, present on nearly all modern distros)
    if [[ -f /etc/os-release ]]; then
        # shellcheck source=/dev/null
        . /etc/os-release
        DISTRO_ID="${ID:-unknown}"
        local id_like="${ID_LIKE:-}"

        case "$DISTRO_ID" in
            ubuntu|debian|linuxmint|pop|elementary|zorin|kali|raspbian|neon)
                DISTRO_FAMILY="debian" ;;
            fedora|rhel|centos|rocky|alma|ol|nobara)
                DISTRO_FAMILY="fedora" ;;
            arch|manjaro|endeavouros|garuda|artix|cachyos)
                DISTRO_FAMILY="arch" ;;
            opensuse*|sles)
                DISTRO_FAMILY="suse" ;;
            alpine)
                DISTRO_FAMILY="alpine" ;;
            void)
                DISTRO_FAMILY="void" ;;
            gentoo|funtoo)
                DISTRO_FAMILY="gentoo" ;;
            clear-linux-os|clearlinux)
                DISTRO_FAMILY="clearlinux" ;;
            *)
                # Fall back to ID_LIKE
                case "$id_like" in
                    *debian*|*ubuntu*)   DISTRO_FAMILY="debian" ;;
                    *fedora*|*rhel*)     DISTRO_FAMILY="fedora" ;;
                    *arch*)              DISTRO_FAMILY="arch" ;;
                    *suse*)              DISTRO_FAMILY="suse" ;;
                    *)                   DISTRO_FAMILY="unknown" ;;
                esac
                ;;
        esac
        return
    fi

    # Method 2: lsb_release fallback
    if command -v lsb_release >/dev/null 2>&1; then
        local lsb_id
        lsb_id="$(lsb_release -i -s 2>/dev/null | tr '[:upper:]' '[:lower:]')"
        DISTRO_ID="$lsb_id"
        case "$lsb_id" in
            ubuntu|debian|linuxmint|pop)  DISTRO_FAMILY="debian" ;;
            fedora|centos|redhat*)        DISTRO_FAMILY="fedora" ;;
            arch|manjaro)                 DISTRO_FAMILY="arch" ;;
            opensuse*)                    DISTRO_FAMILY="suse" ;;
            *)                            DISTRO_FAMILY="unknown" ;;
        esac
        return
    fi

    # Method 3: distro-specific files
    if [[ -f /etc/debian_version ]]; then
        DISTRO_ID="debian"; DISTRO_FAMILY="debian"; return
    elif [[ -f /etc/arch-release ]]; then
        DISTRO_ID="arch"; DISTRO_FAMILY="arch"; return
    elif [[ -f /etc/fedora-release ]]; then
        DISTRO_ID="fedora"; DISTRO_FAMILY="fedora"; return
    elif [[ -f /etc/centos-release ]] || [[ -f /etc/redhat-release ]]; then
        DISTRO_ID="centos"; DISTRO_FAMILY="fedora"; return
    elif [[ -f /etc/SuSE-release ]]; then
        DISTRO_ID="opensuse"; DISTRO_FAMILY="suse"; return
    elif [[ -f /etc/alpine-release ]]; then
        DISTRO_ID="alpine"; DISTRO_FAMILY="alpine"; return
    elif [[ -f /etc/gentoo-release ]]; then
        DISTRO_ID="gentoo"; DISTRO_FAMILY="gentoo"; return
    fi

    DISTRO_ID="unknown"
    DISTRO_FAMILY="unknown"
}

detect_distro
info "Detected distro: ${DISTRO_ID} (family: ${DISTRO_FAMILY})"

# ══════════════════════════════════════════════════════════════════════════════
# Package Installation Helpers
# ══════════════════════════════════════════════════════════════════════════════

# Map generic tool names to distro-specific package names
get_pkg_names() {
    local family="$1"
    shift
    local tools=("$@")
    local pkgs=()

    for tool in "${tools[@]}"; do
        case "$family" in
            debian)
                case "$tool" in
                    cmake) pkgs+=("cmake") ;;
                    gcc)   pkgs+=("gcc") ;;
                    g++)   pkgs+=("g++") ;;
                    make)  pkgs+=("make") ;;
                esac
                ;;
            fedora)
                case "$tool" in
                    cmake) pkgs+=("cmake") ;;
                    gcc)   pkgs+=("gcc") ;;
                    g++)   pkgs+=("gcc-c++") ;;
                    make)  pkgs+=("make") ;;
                esac
                ;;
            arch)
                case "$tool" in
                    cmake) pkgs+=("cmake") ;;
                    gcc)   pkgs+=("gcc") ;;
                    g++)   pkgs+=("gcc") ;;  # g++ is part of gcc package on Arch
                    make)  pkgs+=("make") ;;
                esac
                ;;
            suse)
                case "$tool" in
                    cmake) pkgs+=("cmake") ;;
                    gcc)   pkgs+=("gcc") ;;
                    g++)   pkgs+=("gcc-c++") ;;
                    make)  pkgs+=("make") ;;
                esac
                ;;
            alpine)
                case "$tool" in
                    cmake) pkgs+=("cmake") ;;
                    gcc)   pkgs+=("gcc") ;;
                    g++)   pkgs+=("g++") ;;
                    make)  pkgs+=("make") ;;
                esac
                # Alpine also needs musl-dev and libc-dev for compilation
                pkgs+=("musl-dev")
                ;;
            void)
                case "$tool" in
                    cmake) pkgs+=("cmake") ;;
                    gcc)   pkgs+=("gcc") ;;
                    g++)   pkgs+=("gcc") ;;  # g++ included in gcc on Void
                    make)  pkgs+=("make") ;;
                esac
                ;;
            gentoo)
                case "$tool" in
                    cmake) pkgs+=("dev-build/cmake") ;;
                    gcc)   pkgs+=("sys-devel/gcc") ;;
                    g++)   pkgs+=("sys-devel/gcc") ;;
                    make)  pkgs+=("dev-build/make") ;;
                esac
                ;;
            *)
                pkgs+=("$tool")
                ;;
        esac
    done

    # Deduplicate
    local seen=()
    local unique=()
    for p in "${pkgs[@]}"; do
        local found=false
        for s in "${seen[@]+"${seen[@]}"}"; do
            if [[ "$s" == "$p" ]]; then found=true; break; fi
        done
        if [[ "$found" == false ]]; then
            unique+=("$p")
            seen+=("$p")
        fi
    done
    echo "${unique[*]}"
}

# Build the install command for the detected distro
build_install_cmd() {
    local family="$1"
    shift
    local pkgs=("$@")

    case "$family" in
        debian)
            echo "apt-get update && apt-get install -y ${pkgs[*]}"
            ;;
        fedora)
            if command -v dnf >/dev/null 2>&1; then
                echo "dnf install -y ${pkgs[*]}"
            else
                echo "yum install -y ${pkgs[*]}"
            fi
            ;;
        arch)
            echo "pacman -S --noconfirm --needed ${pkgs[*]}"
            ;;
        suse)
            echo "zypper install -y ${pkgs[*]}"
            ;;
        alpine)
            echo "apk add ${pkgs[*]}"
            ;;
        void)
            echo "xbps-install -y ${pkgs[*]}"
            ;;
        gentoo)
            echo "emerge --noreplace ${pkgs[*]}"
            ;;
        clearlinux)
            echo "swupd bundle-add ${pkgs[*]}"
            ;;
        *)
            echo ""
            ;;
    esac
}

install_missing_deps() {
    local missing_tools=("$@")
    local pkg_names
    pkg_names="$(get_pkg_names "$DISTRO_FAMILY" "${missing_tools[@]}")"

    if [[ -z "$pkg_names" ]]; then
        return 1
    fi

    # Split pkg_names string back into array
    local pkgs
    read -ra pkgs <<< "$pkg_names"

    local install_cmd
    install_cmd="$(build_install_cmd "$DISTRO_FAMILY" "${pkgs[@]}")"

    if [[ -z "$install_cmd" ]]; then
        # Unknown distro — print manual instructions
        err "Could not determine package manager for distro '${DISTRO_ID}'"
        info "Please install the following packages manually:"
        info "  ${missing_tools[*]}"
        info "Then re-run this script with --skip-build or after installing deps."
        return 1
    fi

    info "Packages to install: ${pkgs[*]}"
    info "Command: $install_cmd"

    if [[ "$AUTO_DEPS" == true ]]; then
        info "Auto-installing dependencies (--auto-deps)..."
        eval "$install_cmd"
    else
        echo -en "${YELLOW}Missing: ${missing_tools[*]}. Install now? [Y/n] ${NC}"
        read -r answer
        answer="${answer:-y}"
        if [[ "${answer,,}" == "y" || "${answer,,}" == "yes" ]]; then
            info "Installing dependencies..."
            eval "$install_cmd"
        else
            err "Cannot proceed without build dependencies."
            info "Install manually: ${pkgs[*]}"
            return 1
        fi
    fi
}

# ── Upgrade detection ────────────────────────────────────────────────────────
IS_UPGRADE=false
if [[ -f "$LIB_DIR/libgpushare_client.so" ]] && [[ "$FORCE_REINSTALL" == false ]]; then
    IS_UPGRADE=true
fi

if [[ "$IS_UPGRADE" == true ]]; then
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  gpushare — Upgrading existing installation${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
else
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  gpushare — Linux Client Installer${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
fi
echo

# ── CUDA conflict warning ────────────────────────────────────────────────────
if command -v nvcc >/dev/null 2>&1 || [[ -d /usr/local/cuda ]] || [[ -d /opt/cuda ]]; then
    echo -e "${YELLOW}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  WARNING: Local CUDA installation detected!          ║${NC}"
    echo -e "${YELLOW}║                                                      ║${NC}"
    echo -e "${YELLOW}║  gpushare installs libcudart symlinks that override  ║${NC}"
    echo -e "${YELLOW}║  the real CUDA runtime. Real CUDA libraries will be  ║${NC}"
    echo -e "${YELLOW}║  backed up so both local and remote GPUs can be used.║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════╝${NC}"
    echo
    HAS_LOCAL_CUDA=true
    if [[ "$NO_SYMLINKS" == false ]] && [[ "$IS_UPGRADE" == false ]]; then
        read -rp "Continue? Real CUDA libs will be backed up for dual-GPU support. [Y/n] " answer
        if [[ "${answer,,}" == "n" ]]; then
            info "Aborting. Re-run with --no-symlinks to install without overriding CUDA."
            exit 0
        fi
    fi
fi

# ── 1. Prerequisites ─────────────────────────────────────────────────────────
info "Checking prerequisites..."
missing=()
command -v cmake >/dev/null 2>&1 || missing+=("cmake")
command -v gcc   >/dev/null 2>&1 || missing+=("gcc")
command -v g++   >/dev/null 2>&1 || missing+=("g++")
command -v make  >/dev/null 2>&1 || missing+=("make")

if [[ ${#missing[@]} -gt 0 ]]; then
    warn "Missing prerequisites: ${missing[*]}"
    if install_missing_deps "${missing[@]}"; then
        # Verify they are now available
        still_missing=()
        command -v cmake >/dev/null 2>&1 || still_missing+=("cmake")
        command -v gcc   >/dev/null 2>&1 || still_missing+=("gcc")
        command -v g++   >/dev/null 2>&1 || still_missing+=("g++")
        command -v make  >/dev/null 2>&1 || still_missing+=("make")
        if [[ ${#still_missing[@]} -gt 0 ]]; then
            die "Still missing after install: ${still_missing[*]}"
        fi
        ok "Dependencies installed successfully"
    else
        die "Cannot proceed without build dependencies: ${missing[*]}"
    fi
fi
ok "Build tools found"

# ── 2. Build client library ──────────────────────────────────────────────────
if [[ "$SKIP_BUILD" == false ]]; then
    info "Building gpushare client library..."
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
    cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SERVER=OFF \
        -DBUILD_CLIENT=ON
    cmake --build "$BUILD_DIR" -j"$(nproc)"
    ok "Build complete"
else
    info "Skipping build (--skip-build)"
fi

[[ -e "$BUILD_DIR/libgpushare_client.so" ]] || die "Build artifact missing: $BUILD_DIR/libgpushare_client.so"
ok "Client library verified"

# ── 3. Install library ───────────────────────────────────────────────────────
info "Installing client library..."
mkdir -p "$LIB_DIR"
LIB_UPDATED=false
if [[ "$IS_UPGRADE" == true ]] && cmp -s "$BUILD_DIR/libgpushare_client.so" "$LIB_DIR/libgpushare_client.so"; then
    ok "Client library unchanged — skipping"
else
    install -Dm755 "$BUILD_DIR/libgpushare_client.so" "$LIB_DIR/libgpushare_client.so"
    LIB_UPDATED=true
    ok "Installed $LIB_DIR/libgpushare_client.so"
fi

# ── 3b. Backup real CUDA libraries for local GPU passthrough ─────────────────
REAL_CUDA_BACKUP="$LIB_DIR/real"
if [[ "${HAS_LOCAL_CUDA:-false}" == true ]] && [[ "$NO_SYMLINKS" == false ]]; then
    info "Backing up real CUDA libraries for dual-GPU support..."
    mkdir -p "$REAL_CUDA_BACKUP"

    # Search for real CUDA libraries in standard paths
    CUDA_SEARCH_PATHS=(
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/lib
        /usr/local/cuda/lib64
        /opt/cuda/lib64
        /opt/cuda/lib
    )

    backup_lib() {
        local libname="$1"
        # Skip if already backed up
        if [[ -f "$REAL_CUDA_BACKUP/$libname" ]]; then
            return
        fi
        for dir in "${CUDA_SEARCH_PATHS[@]}"; do
            local src="$dir/$libname"
            # Follow symlinks to find the real file
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
    }

    backup_lib "libcudart.so"
    backup_lib "libcuda.so.1"
    backup_lib "libnvidia-ml.so.1"

    if [[ -f "$REAL_CUDA_BACKUP/libcudart.so" ]] && [[ -f "$REAL_CUDA_BACKUP/libcuda.so.1" ]]; then
        ok "Real CUDA libraries backed up to $REAL_CUDA_BACKUP"
        info "Both local and remote GPUs will be available (gpu_mode=all)"
    else
        warn "Could not find all real CUDA libraries to backup"
        warn "Only remote GPU will be available"
    fi
fi

# ── 4. Create CUDA symlinks (transparent replacement) ────────────────────────
if [[ "$NO_SYMLINKS" == false ]]; then
    info "Creating transparent CUDA symlinks..."

    # libcudart.so variants — applications looking for any of these will
    # find our client library instead, transparently forwarding to remote GPU.
    ln -sf libgpushare_client.so "$LIB_DIR/libcudart.so"
    ln -sf libgpushare_client.so "$LIB_DIR/libcudart.so.11"
    ln -sf libgpushare_client.so "$LIB_DIR/libcudart.so.11.0"
    ln -sf libgpushare_client.so "$LIB_DIR/libcudart.so.12"
    ln -sf libgpushare_client.so "$LIB_DIR/libcudart.so.12.0"
    ln -sf libgpushare_client.so "$LIB_DIR/libcudart.so.13"
    ln -sf libgpushare_client.so "$LIB_DIR/libcudart.so.13.0"

    # Driver API (libcuda.so.1) — PyTorch/TF/JAX load this from system
    # even when they bundle their own libcudart. Provides native venv support.
    ln -sf libgpushare_client.so "$LIB_DIR/libcuda.so"
    ln -sf libgpushare_client.so "$LIB_DIR/libcuda.so.1"

    # NVML (libnvidia-ml) — GPU detection for apps, monitoring tools
    ln -sf libgpushare_client.so "$LIB_DIR/libnvidia-ml.so"
    ln -sf libgpushare_client.so "$LIB_DIR/libnvidia-ml.so.1"

    # cuBLAS
    ln -sf libgpushare_client.so "$LIB_DIR/libcublas.so"
    ln -sf libgpushare_client.so "$LIB_DIR/libcublas.so.11"
    ln -sf libgpushare_client.so "$LIB_DIR/libcublas.so.12"
    ln -sf libgpushare_client.so "$LIB_DIR/libcublas.so.13"
    ln -sf libgpushare_client.so "$LIB_DIR/libcublasLt.so"
    ln -sf libgpushare_client.so "$LIB_DIR/libcublasLt.so.11"
    ln -sf libgpushare_client.so "$LIB_DIR/libcublasLt.so.12"
    ln -sf libgpushare_client.so "$LIB_DIR/libcublasLt.so.13"
    # cuDNN
    ln -sf libgpushare_client.so "$LIB_DIR/libcudnn.so"
    ln -sf libgpushare_client.so "$LIB_DIR/libcudnn.so.8"
    ln -sf libgpushare_client.so "$LIB_DIR/libcudnn.so.9"
    # cuFFT
    ln -sf libgpushare_client.so "$LIB_DIR/libcufft.so"
    ln -sf libgpushare_client.so "$LIB_DIR/libcufft.so.11"
    # cuSPARSE
    ln -sf libgpushare_client.so "$LIB_DIR/libcusparse.so"
    ln -sf libgpushare_client.so "$LIB_DIR/libcusparse.so.12"
    # cuSOLVER
    ln -sf libgpushare_client.so "$LIB_DIR/libcusolver.so"
    ln -sf libgpushare_client.so "$LIB_DIR/libcusolver.so.11"
    # cuRAND
    ln -sf libgpushare_client.so "$LIB_DIR/libcurand.so"
    ln -sf libgpushare_client.so "$LIB_DIR/libcurand.so.10"
    # NVRTC
    ln -sf libgpushare_client.so "$LIB_DIR/libnvrtc.so"
    ln -sf libgpushare_client.so "$LIB_DIR/libnvrtc.so.12"
    # nvJPEG
    ln -sf libgpushare_client.so "$LIB_DIR/libnvjpeg.so"
    ln -sf libgpushare_client.so "$LIB_DIR/libnvjpeg.so.12"

    ok "All CUDA library symlinks created (runtime + driver + NVML + cuBLAS + cuDNN + cuFFT + cuSPARSE + cuSOLVER + cuRAND + NVRTC)"

    # ── 5. Configure dynamic linker (distro-specific) ─────────────────────────
    info "Configuring dynamic linker..."

    case "$DISTRO_FAMILY" in
        alpine)
            # Alpine uses musl libc — no /etc/ld.so.conf.d support
            # musl uses /etc/ld-musl-<arch>.path
            local_arch="$(uname -m)"
            MUSL_PATH_FILE="/etc/ld-musl-${local_arch}.path"
            if [[ -f "$MUSL_PATH_FILE" ]]; then
                if ! grep -qxF "$LIB_DIR" "$MUSL_PATH_FILE" 2>/dev/null; then
                    echo "$LIB_DIR" >> "$MUSL_PATH_FILE"
                    ok "Added $LIB_DIR to $MUSL_PATH_FILE"
                else
                    ok "$LIB_DIR already in $MUSL_PATH_FILE"
                fi
            else
                # Create the path file with default paths + ours
                cat > "$MUSL_PATH_FILE" <<MUSLEOF
/lib
/usr/local/lib
/usr/lib
$LIB_DIR
MUSLEOF
                ok "Created $MUSL_PATH_FILE with $LIB_DIR"
            fi
            ;;
        *)
            # Standard glibc distros: use ld.so.conf.d + ldconfig
            echo "$LIB_DIR" > /etc/ld.so.conf.d/gpushare.conf
            if [[ "$LIB_UPDATED" == true ]]; then
                ldconfig
                ok "ldconfig updated — $LIB_DIR is now in the library search path"
            else
                ok "Library unchanged — ldconfig skipped"
            fi

            # Verify it takes priority
            resolved=$(ldconfig -p 2>/dev/null | grep "libcudart.so " | head -1 || true)
            if echo "$resolved" | grep -q gpushare; then
                ok "Verified: libcudart.so resolves to gpushare"
            else
                warn "libcudart.so may not resolve to gpushare (another CUDA path has priority)"
                warn "Check: ldconfig -p | grep libcudart"
            fi
            ;;
    esac
else
    info "Skipping CUDA symlinks (--no-symlinks)"
    info "Set LD_LIBRARY_PATH=$LIB_DIR before your CUDA application to use gpushare."
fi

# ── 6. Install config ────────────────────────────────────────────────────────
info "Installing client configuration..."
mkdir -p "$CONF_DIR"

if [[ "$IS_UPGRADE" == true ]] && [[ -f "$CONF_DIR/client.conf" ]]; then
    # On upgrade, preserve existing config; install new default as reference
    warn "Config already exists at $CONF_DIR/client.conf — not overwriting"
    install -Dm644 "$PROJECT_DIR/config/gpushare-client.conf" "$CONF_DIR/client.conf.new"
    info "New default written to $CONF_DIR/client.conf.new for reference"
    # Read server address from existing config for summary
    SERVER_ADDR=$(grep -oP '^server=\K.*' "$CONF_DIR/client.conf" 2>/dev/null || echo "unknown")
else
    # Fresh install — prompt for server address
    if [[ -z "$SERVER_ADDR" ]]; then
        if [[ -f "$CONF_DIR/client.conf" ]]; then
            current=$(grep -oP '^server=\K.*' "$CONF_DIR/client.conf" 2>/dev/null || echo "")
            read -rp "Server address [${current:-192.168.1.100:9847}]: " SERVER_ADDR
            SERVER_ADDR="${SERVER_ADDR:-${current:-192.168.1.100:9847}}"
        else
            read -rp "Server address (host:port) [192.168.1.100:9847]: " SERVER_ADDR
            SERVER_ADDR="${SERVER_ADDR:-192.168.1.100:9847}"
        fi
    fi

    if [[ -f "$CONF_DIR/client.conf" ]]; then
        warn "Config already exists at $CONF_DIR/client.conf — updating server address"
        sed -i "s|^server=.*|server=$SERVER_ADDR|" "$CONF_DIR/client.conf"
    else
        sed "s|^server=.*|server=$SERVER_ADDR|" "$PROJECT_DIR/config/gpushare-client.conf" \
            > "$CONF_DIR/client.conf"
    fi
    chmod 644 "$CONF_DIR/client.conf"
    ok "Config installed: $CONF_DIR/client.conf (server=$SERVER_ADDR)"
fi

# ── 7. Install TUI + dashboard ───────────────────────────────────────────────
info "Installing TUI monitor and dashboard..."
mkdir -p "$SHARE_DIR/tui"
mkdir -p "$SHARE_DIR/dashboard"
cp -a "$PROJECT_DIR/tui/"*       "$SHARE_DIR/tui/"       2>/dev/null || warn "No TUI files found"
cp -a "$PROJECT_DIR/dashboard/"* "$SHARE_DIR/dashboard/"  2>/dev/null || warn "No dashboard files found"

cat > "$BIN_DIR/gpushare-monitor" <<'TEOF'
#!/usr/bin/env bash
exec python3 /usr/local/share/gpushare/tui/monitor.py "$@"
TEOF
chmod 755 "$BIN_DIR/gpushare-monitor"
ok "Installed $BIN_DIR/gpushare-monitor"

# nvidia-smi shim — so GPU detection scripts and monitoring tools work
cp "$PROJECT_DIR/scripts/nvidia-smi" "$SHARE_DIR/nvidia-smi"
chmod 755 "$SHARE_DIR/nvidia-smi"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    cat > "$BIN_DIR/nvidia-smi" <<'NSMI'
#!/usr/bin/env bash
exec python3 /usr/local/share/gpushare/nvidia-smi "$@"
NSMI
    chmod 755 "$BIN_DIR/nvidia-smi"
    # Fix SELinux context if enforcing (Fedora, RHEL, CentOS)
    if command -v restorecon >/dev/null 2>&1; then
        restorecon -v "$BIN_DIR/nvidia-smi" 2>/dev/null || true
        restorecon -Rv "$SHARE_DIR" 2>/dev/null || true
        restorecon -Rv "$LIB_DIR" 2>/dev/null || true
    fi
    # Fix AppArmor if present (Ubuntu, SUSE)
    if command -v aa-complain >/dev/null 2>&1 && [[ -d /etc/apparmor.d ]]; then
        # Ensure our paths aren't blocked
        if [[ ! -f /etc/apparmor.d/local/gpushare ]]; then
            mkdir -p /etc/apparmor.d/local
            cat > /etc/apparmor.d/local/gpushare <<'AAEOF'
# gpushare — allow execution of nvidia-smi shim and client library
/usr/local/bin/nvidia-smi rix,
/usr/local/bin/gpushare-* rix,
/usr/local/share/gpushare/** rix,
/usr/local/lib/gpushare/** rm,
AAEOF
            ok "Created AppArmor local profile for gpushare"
        fi
    fi
    ok "Installed nvidia-smi command (queries remote GPU)"
else
    warn "Real nvidia-smi found — not overriding (shim at $SHARE_DIR/nvidia-smi)"
fi

# ── 8. Install Python client ─────────────────────────────────────────────────
if [[ "$SKIP_PYTHON" == false ]]; then
    if command -v python3 >/dev/null 2>&1 && command -v pip3 >/dev/null 2>&1; then
        info "Installing Python client package..."
        pip3 install --break-system-packages "$PROJECT_DIR/python/" 2>/dev/null \
            || pip3 install "$PROJECT_DIR/python/" 2>/dev/null \
            || warn "Python client install failed (non-fatal)"
        ok "Python client installed"
    else
        warn "python3/pip3 not found — skipping Python client"
    fi
else
    info "Skipping Python client (--skip-python)"
fi

# ── 9. Systemd user service for dashboard ────────────────────────────────────
# Systemd is not available on all distros (e.g., Alpine with OpenRC, Void with runit)
if command -v systemctl >/dev/null 2>&1; then
    info "Checking systemd user service for dashboard..."
    SUSER_DIR=/etc/systemd/user
    mkdir -p "$SUSER_DIR"

    DASHBOARD_UNIT=$(cat <<UNITEOF
[Unit]
Description=gpushare Client Dashboard
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 $SHARE_DIR/dashboard/app.py --client --config $CONF_DIR/client.conf
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNITEOF
)

    if [[ -f "$SUSER_DIR/gpushare-dashboard.service" ]] && \
       printf '%s\n' "$DASHBOARD_UNIT" | cmp -s "$SUSER_DIR/gpushare-dashboard.service" -; then
        ok "gpushare-dashboard.service unchanged — skipping"
    else
        printf '%s\n' "$DASHBOARD_UNIT" > "$SUSER_DIR/gpushare-dashboard.service"
        ok "Written $SUSER_DIR/gpushare-dashboard.service"
    fi
    info "Enable with: systemctl --user enable --now gpushare-dashboard"
else
    info "systemd not detected — skipping dashboard service installation"
    info "You can run the dashboard manually: python3 $SHARE_DIR/dashboard/app.py"
fi

# ── 10. Patch ML frameworks ──────────────────────────────────────────────────
if command -v python3 >/dev/null 2>&1; then
    info "Patching ML frameworks (PyTorch/TensorFlow/JAX) if installed..."
    python3 "$PROJECT_DIR/scripts/gpushare-patch-frameworks.py" 2>/dev/null || true
    cp "$PROJECT_DIR/scripts/gpushare-patch-frameworks.py" "$SHARE_DIR/gpushare-patch-frameworks.py" 2>/dev/null || true
    cat > "$BIN_DIR/gpushare-patch" <<'PEOF'
#!/usr/bin/env bash
exec python3 /usr/local/share/gpushare/gpushare-patch-frameworks.py "$@"
PEOF
    chmod 755 "$BIN_DIR/gpushare-patch"
    ok "Installed gpushare-patch command"
fi

# ── 11. Summary ──────────────────────────────────────────────────────────────
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
echo -e "  Distro:          ${CYAN}${DISTRO_ID}${NC} (${DISTRO_FAMILY})"
echo -e "  Server:          ${CYAN}${SERVER_ADDR}${NC}"
echo -e "  Library:         $LIB_DIR/libgpushare_client.so"
echo -e "  Config:          $CONF_DIR/client.conf"
if [[ "$NO_SYMLINKS" == false ]]; then
echo -e "  CUDA override:   ${GREEN}ACTIVE${NC} (libcudart.so -> gpushare)"
echo -e "  API coverage:    2600+ functions (cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSOLVER, cuRAND, NVRTC, nvJPEG)"
echo -e "  Transfer opts:   ${GREEN}ACTIVE${NC} (tiered pinned pools, async memcpy, chunked pipelining, D2H prefetch, RDMA auto-detect)"
if grep -q "^IBVERBS_LIB:FILEPATH=.*libibverbs" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
echo -e "  RDMA transport:  ${GREEN}BUILT IN${NC} (rdma-core auto-detected)"
elif [[ -f /usr/lib/libibverbs.so ]] || [[ -f /usr/lib64/libibverbs.so ]] || ldconfig -p 2>/dev/null | grep -q libibverbs; then
echo -e "  RDMA transport:  ${GREEN}AVAILABLE${NC} (rebuild to enable)"
else
echo -e "  RDMA transport:  ${YELLOW}NOT AVAILABLE${NC} (install rdma-core for InfiniBand support)"
fi
else
echo -e "  CUDA override:   ${YELLOW}DISABLED${NC} (use LD_LIBRARY_PATH)"
fi
echo
echo -e "  ${BOLD}Usage — any CUDA program works transparently:${NC}"
echo -e "    python3 my_training.py        ${CYAN}# uses remote GPU${NC}"
echo -e "    ./my_cuda_app                 ${CYAN}# uses remote GPU${NC}"
echo -e "    gpushare-monitor              ${CYAN}# TUI status monitor${NC}"
echo
echo -e "  ${BOLD}To change server:${NC}"
echo -e "    Edit $CONF_DIR/client.conf"
echo
echo -e "  ${BOLD}To uninstall:${NC}"
echo -e "    $PROJECT_DIR/scripts/uninstall.sh"
echo
