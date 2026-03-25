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
  --remote-first        Prioritize remote GPUs (Remote=Device 0)
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
REMOTE_FIRST=false
SERVER_ADDR=""
for arg in "$@"; do
    case "$arg" in
        --skip-build)   SKIP_BUILD=true ;;
        --skip-python)  SKIP_PYTHON=true ;;
        --no-symlinks)  NO_SYMLINKS=true ;;
        --force)        FORCE_REINSTALL=true ;;
        --auto-deps)    AUTO_DEPS=true ;;
        --remote-first) REMOTE_FIRST=true ;;
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
    # Remove weak stubs that conflict with strong implementations.
    # On some toolchains (MinGW, older GCC), weak symbol override doesn't work
    # and causes "multiple definition" linker errors.
    all_stubs="$PROJECT_DIR/client/generated_all_stubs.cpp"
    if [[ -f "$all_stubs" ]]; then
        CONFLICTING_STUBS=(
            cuGetProcAddress cuGetProcAddress_v2
            cudaLaunchKernel cudaHostAlloc cudaHostRegister cudaHostUnregister
            cudaMemcpyPeerAsync cudaFuncSetAttribute cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
            cudaLaunchKernelExC cudaDeviceGetPCIBusId cudaLaunchHostFunc
            cudaStreamBeginCapture cudaStreamEndCapture cudaStreamIsCapturing
            cudaGraphInstantiateWithFlags cudaGraphGetNodes cudaGraphDebugDotPrint
            cudaGraphNodeGetDependencies cudaMemPoolSetAttribute cudaMemPoolGetAttribute
            cudaMemPoolSetAccess cudaIpcGetMemHandle cudaIpcOpenMemHandle cudaIpcCloseMemHandle
            cudaIpcGetEventHandle cudaIpcOpenEventHandle cudaMemcpyToSymbol
            cudaEventRecordWithFlags cudaStreamGetPriority cudaMemcpy2DAsync
            cudaGetDriverEntryPoint cudaGetDriverEntryPointByVersion
            cudaGetDeviceProperties_v2
            cudaThreadExchangeStreamCaptureMode
        )
        stub_removed=false
        for stub in "${CONFLICTING_STUBS[@]}"; do
            if grep -q "WEAK_SYM ${stub}()" "$all_stubs" 2>/dev/null; then
                sed -i "/STUB_EXPORT int WEAK_SYM ${stub}() /d" "$all_stubs"
                stub_removed=true
            fi
        done
        if [[ "$stub_removed" == true ]]; then
            info "Removed conflicting weak stubs from generated_all_stubs.cpp"
        fi
    fi

    # Force clean build on upgrade to pick up all source changes
    if [[ "$FORCE_REINSTALL" == true ]] && [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
        info "Force rebuild: cleaning build directory"
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"
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

# Find the actual built library (could be .so.1.0.0 or just .so)
BUILD_LIB=""
for f in "$BUILD_DIR/libgpushare_client.so.1.0.0" "$BUILD_DIR/libgpushare_client.so.1" "$BUILD_DIR/libgpushare_client.so"; do
    if [[ -f "$f" ]]; then
        BUILD_LIB="$f"
        break
    fi
done
[[ -n "$BUILD_LIB" ]] || die "Build artifact missing: $BUILD_DIR/libgpushare_client.so*"
ok "Client library verified: $BUILD_LIB"

# ── 3. Install library ───────────────────────────────────────────────────────
info "Installing client library..."
mkdir -p "$LIB_DIR"
LIB_UPDATED=false
INSTALLED_LIB="$LIB_DIR/libgpushare_client.so.1.0.0"

if [[ "$IS_UPGRADE" == true ]] && cmp -s "$BUILD_LIB" "$INSTALLED_LIB"; then
    ok "Client library unchanged — skipping"
else
    # Install the actual library file with version suffix
    install -Dm755 "$BUILD_LIB" "$INSTALLED_LIB"
    LIB_UPDATED=true
    ok "Installed $INSTALLED_LIB"
    
    # Create versioned symlinks
    ln -sf "libgpushare_client.so.1.0.0" "$LIB_DIR/libgpushare_client.so.1"
    ln -sf "libgpushare_client.so.1.0.0" "$LIB_DIR/libgpushare_client.so"
    ok "Created versioned symlinks"
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

# ── 4. Configure dynamic linker (must run BEFORE symlinks so ldconfig
#       doesn't overwrite our symlinks with SONAME-based ones) ────────────────
if [[ "$NO_SYMLINKS" == false ]]; then
    info "Configuring dynamic linker..."

    case "$DISTRO_FAMILY" in
        alpine)
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
            echo "$LIB_DIR" > /etc/ld.so.conf.d/gpushare.conf
            ldconfig
            ok "ldconfig updated"
            ;;
    esac
fi

# ── 5. Create CUDA symlinks (transparent replacement) ────────────────────────
if [[ "$NO_SYMLINKS" == false ]]; then
    info "Creating transparent CUDA symlinks..."

    # Determine the system library directory where the dynamic linker looks.
    # Symlinks MUST go in /usr/lib (or /usr/lib64) because:
    # - ldconfig ignores symlinks whose target has a different SONAME
    # - The dynamic linker only searches /lib, /usr/lib, and the ldconfig cache
    # - Putting symlinks in a subdirectory does NOT make them findable
    SYS_LIB_DIR="/usr/lib"
    if [[ -d /usr/lib64 ]] && [[ ! -L /usr/lib64 ]]; then
        SYS_LIB_DIR="/usr/lib64"
    fi

    # All CUDA library names that should resolve to gpushare
    CUDA_SYMLINKS=(
        libcudart.so libcudart.so.11 libcudart.so.11.0
        libcudart.so.12 libcudart.so.12.0 libcudart.so.13 libcudart.so.13.0
        libcuda.so libcuda.so.1
        libnvidia-ml.so libnvidia-ml.so.1
        libcublas.so libcublas.so.11 libcublas.so.12 libcublas.so.13
        libcublasLt.so libcublasLt.so.11 libcublasLt.so.12 libcublasLt.so.13
        libcudnn.so libcudnn.so.8 libcudnn.so.9
        libcufft.so libcufft.so.11
        libcusparse.so libcusparse.so.12
        libcusolver.so libcusolver.so.11
        libcurand.so libcurand.so.10
        libnvrtc.so libnvrtc.so.12
        libnvjpeg.so libnvjpeg.so.12
    )

    # Create symlinks in the gpushare directory (for reference/organization)
    for link in "${CUDA_SYMLINKS[@]}"; do
        ln -sf libgpushare_client.so "$LIB_DIR/$link"
    done
    # Fix soname symlink (prevent ldconfig from creating a confusing chain)
    ln -sf libgpushare_client.so "$LIB_DIR/libgpushare_client.so.1"

    # Create symlinks in the SYSTEM library directory so the dynamic linker
    # can actually find them. These point to the absolute path of our library.
    target="$LIB_DIR/libgpushare_client.so"
    created=0
    skipped=0
    for link in "${CUDA_SYMLINKS[@]}"; do
        sys_path="$SYS_LIB_DIR/$link"
        # Back up any existing real CUDA library before overriding
        if [[ -e "$sys_path" ]] && [[ ! -L "$sys_path" ]]; then
            # It's a real file (not a symlink) — back it up
            mkdir -p "$LIB_DIR/real"
            if [[ ! -f "$LIB_DIR/real/$link" ]]; then
                cp "$sys_path" "$LIB_DIR/real/$link"
                info "Backed up $sys_path to $LIB_DIR/real/$link"
            fi
        elif [[ -L "$sys_path" ]]; then
            existing_target=$(readlink -f "$sys_path" 2>/dev/null || true)
            if [[ "$existing_target" == *gpushare* ]]; then
                skipped=$((skipped + 1))
                continue  # already points to gpushare
            fi
            # Symlink to something else (e.g., real NVIDIA driver) — back up target
            if [[ -f "$existing_target" ]] && [[ "$existing_target" != *gpushare* ]]; then
                mkdir -p "$LIB_DIR/real"
                if [[ ! -f "$LIB_DIR/real/$link" ]]; then
                    cp "$existing_target" "$LIB_DIR/real/$link"
                    info "Backed up $link (from $existing_target)"
                fi
            fi
        fi
        ln -sf "$target" "$sys_path"
        created=$((created + 1))
    done

    if [[ $skipped -gt 0 ]]; then
        ok "CUDA symlinks: $created created, $skipped already correct"
    else
        ok "All $created CUDA library symlinks created in $SYS_LIB_DIR"
    fi

    # Verify the critical symlink resolves correctly
    if [[ -L "$SYS_LIB_DIR/libcuda.so.1" ]]; then
        target_check=$(readlink -f "$SYS_LIB_DIR/libcuda.so.1" 2>/dev/null || true)
        if [[ "$target_check" == *gpushare* ]]; then
            ok "Verified: $SYS_LIB_DIR/libcuda.so.1 -> gpushare"
        else
            warn "libcuda.so.1 in $SYS_LIB_DIR does not point to gpushare"
        fi
    else
        warn "libcuda.so.1 symlink not found in $SYS_LIB_DIR"
    fi
else
    info "Skipping CUDA symlinks (--no-symlinks)"
    info "Set LD_LIBRARY_PATH=$LIB_DIR before your CUDA application to use gpushare."
fi

# ── 6. Install config ────────────────────────────────────────────────────────
info "Installing client configuration..."
mkdir -p "$CONF_DIR"

if [[ "$IS_UPGRADE" == true ]] && [[ -f "$CONF_DIR/client.conf" ]]; then
    # On upgrade, update server address and remote_first if requested
    if [[ -z "$SERVER_ADDR" ]]; then
        current=$(grep -oP '^server=\K.*' "$CONF_DIR/client.conf" 2>/dev/null || echo "")
        read -rp "Server address [${current:-192.168.1.100:9847}]: " SERVER_ADDR
        SERVER_ADDR="${SERVER_ADDR:-${current:-192.168.1.100:9847}}"
    fi
    sed -i "s|^server=.*|server=$SERVER_ADDR|" "$CONF_DIR/client.conf"
    if [[ "$REMOTE_FIRST" == true ]]; then
        if grep -q '^remote_first=' "$CONF_DIR/client.conf"; then
            sed -i "s|^remote_first=.*|remote_first=true|" "$CONF_DIR/client.conf"
        elif grep -q '^# remote_first=' "$CONF_DIR/client.conf"; then
            sed -i "s|^# remote_first=.*|remote_first=true|" "$CONF_DIR/client.conf"
        else
            echo "remote_first=true" >> "$CONF_DIR/client.conf"
        fi
    fi
    ok "Config updated: $CONF_DIR/client.conf (server=$SERVER_ADDR)"
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
        if [[ "$REMOTE_FIRST" == true ]]; then
            sed -i "s|^# remote_first=.*|remote_first=true|" "$CONF_DIR/client.conf"
        fi
    else
        sed "s|^server=.*|server=$SERVER_ADDR|" "$PROJECT_DIR/config/gpushare-client.conf" \
            > "$CONF_DIR/client.conf"
        if [[ "$REMOTE_FIRST" == true ]]; then
            sed -i "s|^# remote_first=.*|remote_first=true|" "$CONF_DIR/client.conf"
        fi
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

# nvidia-smi wrapper for dual-GPU systems (has local GPU + remote GPU)
# On systems with a local GPU, install a wrapper that uses the real nvidia-smi
# On systems without a local GPU, install the shim that queries remote GPU

# First, backup the real nvidia-smi if it exists
NVIDIA_SMI_REAL="/usr/bin/nvidia-smi.real"
if [[ -f /usr/bin/nvidia-smi ]] && [[ ! -L /usr/bin/nvidia-smi ]]; then
    cp /usr/bin/nvidia-smi "$NVIDIA_SMI_REAL"
    info "Backed up real nvidia-smi to $NVIDIA_SMI_REAL"
fi

# Copy our tools to share directory
cp "$PROJECT_DIR/scripts/nvidia-smi" "$SHARE_DIR/nvidia-smi"
cp "$PROJECT_DIR/scripts/nvidia-smi-wrapper.sh" "$SHARE_DIR/nvidia-smi-wrapper.sh"
chmod 755 "$SHARE_DIR/nvidia-smi"
chmod 755 "$SHARE_DIR/nvidia-smi-wrapper.sh"

# Install the appropriate nvidia-smi command
if [[ -f "$NVIDIA_SMI_REAL" ]] && [[ "$HAS_LOCAL_CUDA" == true ]]; then
    # Dual-GPU system: install wrapper that uses real nvidia-smi with correct library path
    cat > "$BIN_DIR/nvidia-smi" <<'WRAPPER'
#!/usr/bin/env bash
# nvidia-smi wrapper for gpushare clients
#
# This script ensures nvidia-smi works correctly on systems with both
# local and remote GPUs by using the real NVIDIA tools with the proper
# library path.

if [ -d /usr/local/lib/gpushare/real ]; then
    export LD_LIBRARY_PATH="/usr/local/lib/gpushare/real:$LD_LIBRARY_PATH"
fi

# Find real nvidia-smi
if [ -f /usr/bin/nvidia-smi.real ]; then
    exec /usr/bin/nvidia-smi.real "$@"
elif [ -f /opt/cuda/bin/nvidia-smi ]; then
    exec /opt/cuda/bin/nvidia-smi "@@"
else
    # Fallback to remote GPU query
    exec python3 /usr/local/share/gpushare/nvidia-smi-wrapper.sh "$@"
fi
WRAPPER
    chmod 755 "$BIN_DIR/nvidia-smi"
    ok "Installed nvidia-smi command (wrapper -> real nvidia-smi for local GPUs)"
else
    # No local GPU: install shim that queries remote GPU
    if ! command -v nvidia-smi >/dev/null 2>&1 || [[ -f "$NVIDIA_SMI_REAL" ]]; then
        cat > "$BIN_DIR/nvidia-smi" <<'SHIM'
#!/usr/bin/env bash
# nvidia-smi shim for remote GPU queries
exec python3 /usr/local/share/gpushare/nvidia-smi "$@"
SHIM
        chmod 755 "$BIN_DIR/nvidia-smi"
    fi
    ok "Installed nvidia-smi command (queries remote GPU)"
fi

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
# gpushare — allow execution of nvidia-smi and client library
/usr/local/bin/nvidia-smi rix,
/usr/local/bin/gpushare-* rix,
/usr/local/share/gpushare/** rix,
/usr/local/lib/gpushare/** rm,
AAEOF
        ok "Created AppArmor local profile for gpushare"
    fi
fi

# ── Detect and manage Python environments ─────────────────────────────────────
detect_python_environments() {
    local envs=()
    local active_env=""

    # Check if a Python environment is currently activated
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        active_env="$VIRTUAL_ENV"
        envs+=("VIRTUAL_ENV:$VIRTUAL_ENV")
    fi
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        active_env="$CONDA_PREFIX"
        envs+=("CONDA:$CONDA_PREFIX")
    fi

    # pyenv
    if [[ -d "$HOME/.pyenv" ]]; then
        local pyenv_path="$HOME/.pyenv/bin"
        if [[ ":$PATH:" != *":$pyenv_path:"* ]]; then
            envs+=("pyenv:$pyenv_path")
        fi
    fi

    # conda
    local conda_paths=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "$HOME/opt/miniconda3"
        "$HOME/opt/anaconda3"
        "/opt/miniconda3"
        "/opt/anaconda3"
    )
    for conda_path in "${conda_paths[@]}"; do
        if [[ -d "$conda_path" ]]; then
            local conda_bin="$conda_path/bin"
            if [[ ":$PATH:" != *":$conda_bin:"* ]]; then
                envs+=("conda:$conda_bin")
            fi
            break
        fi
    done

    # virtualenvwrapper
    if [[ -d "$HOME/.virtualenvs" ]]; then
        envs+=("virtualenvwrapper:$HOME/.virtualenvs")
    fi

    # venv
    for venv_dir in "$HOME/*/venv" "$HOME/*/.venv" "$HOME/venv" "$HOME/.venv"; do
        if [[ -d "$venv_dir" ]]; then
            local venv_bin="$venv_dir/bin"
            if [[ -d "$venv_bin" ]]; then
                envs+=("venv:$venv_bin")
            fi
        fi
    done

    printf '%s\n' "${envs[@]}"
}

manage_ld_preload() {
    local preload_lib="$LIB_DIR/libgpushare_client.so"
    local preload_path="$preload_lib"
    local added=0
    local corrected=0
    local skipped=0
    local p
    local f

    info "Configuring LD_PRELOAD in shell profiles..."

    for f in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile" "$HOME/.zshrc" "$HOME/.zprofile"; do
        if [[ -f "$f" ]]; then
            if grep -q 'LD_PRELOAD.*'"$preload_path" "$f" 2>/dev/null; then
                skipped=$((skipped + 1))
            else
                echo "" >> "$f"
                echo "# gpushare - set LD_PRELOAD to use remote GPU" >> "$f"
                echo "export LD_PRELOAD=\"$preload_path\"" >> "$f"
                added=$((added + 1))
            fi
        fi
    done

    if [[ -f /etc/environment ]]; then
        if grep -q "$preload_path" /etc/environment 2>/dev/null; then
            skipped=$((skipped + 1))
        else
            echo "LD_PRELOAD=\"$preload_path\"" >> /etc/environment
            added=$((added + 1))
        fi
    fi

    ok "LD_PRELOAD configured: $added added, $skipped already configured"
}

manage_python_path() {
    info "Checking PATH for Python environments..."
    local modified=0
    local env_str
    local env_type
    local env_path
    local p

    env_str=$(detect_python_environments)
    if [[ -z "$env_str" ]]; then
        info "No additional Python environments detected"
        return
    fi

    for env_info in $env_str; do
        env_type="${env_info%%:*}"
        env_path="${env_info#*:}"

        case "$env_type" in
            pyenv)
                p="$HOME/.bashrc"
                if [[ -f "$p" ]] && ! grep -q 'pyenv init' "$p" 2>/dev/null; then
                    echo "" >> "$p"
                    echo "# gpushare - pyenv initialization" >> "$p"
                    echo 'export PYENV_ROOT="'"$env_path"'/../"' >> "$p"
                    echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> "$p"
                    ok "  Added pyenv to $p"
                    modified=1
                fi
                ;;
            conda)
                p="$HOME/.bashrc"
                if [[ -f "$p" ]] && ! grep -q "$env_path" "$p" 2>/dev/null; then
                    echo "" >> "$p"
                    echo "# gpushare - conda initialization" >> "$p"
                    echo 'export PATH="'"$env_path"':$PATH"' >> "$p"
                    ok "  Added conda to $p"
                    modified=1
                fi
                ;;
            virtualenvwrapper)
                p="$HOME/.bashrc"
                if [[ -f "$p" ]] && ! grep -q 'virtualenvwrapper' "$p" 2>/dev/null; then
                    echo "" >> "$p"
                    echo "# gpushare - virtualenvwrapper initialization" >> "$p"
                    echo 'export WORKON_HOME="$HOME/.virtualenvs"' >> "$p"
                    ok "  Added virtualenvwrapper to $p"
                    modified=1
                fi
                ;;
            VIRTUAL_ENV|venv)
                info "  $env_type detected (use 'source venv/bin/activate')"
                ;;
        esac
    done

    if [[ $modified -gt 0 ]]; then
        ok "PATH configuration updated for Python environments"
        warn "Please restart your shell or run: source ~/.bashrc"
    else
        ok "Python environment PATH already configured"
    fi
}

# ── 8. Manage PATH and LD_PRELOAD for Python environments ────────────────────
info "Configuring PATH and LD_PRELOAD for Python environments..."
manage_python_path
manage_ld_preload

# ── 9. Install Python client + startup hook ────────────────────────────────────
if [[ "$SKIP_PYTHON" == false ]]; then
    if command -v python3 >/dev/null 2>&1 && command -v pip3 >/dev/null 2>&1; then
        info "Installing Python client package..."
        pip3 install --break-system-packages "$PROJECT_DIR/python/" 2>/dev/null \
            || pip3 install "$PROJECT_DIR/python/" 2>/dev/null \
            || warn "Python client install failed (non-fatal)"
        ok "Python client installed"

        # Install PyTorch startup hook for transparent remote GPU detection
        info "Installing PyTorch startup hook..."
        SITE_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
        if [[ -n "$SITE_DIR" && -d "$SITE_DIR" ]]; then
            cp "$PROJECT_DIR/python/gpushare_hook.py" "$SITE_DIR/gpushare_hook.py" 2>/dev/null || true
            cp "$PROJECT_DIR/python/gpushare.pth" "$SITE_DIR/gpushare.pth" 2>/dev/null || true
            ok "Startup hook installed (remote GPUs auto-detected by PyTorch)"
        else
            warn "Could not find Python site-packages directory"
        fi
    else
        warn "python3/pip3 not found — skipping Python client"
    fi
else
    info "Skipping Python client (--skip-python)"
fi

# ── 10. Replace PyTorch's bundled CUDA libraries ───────────────────────────────
# PyTorch bundles multiple CUDA libraries that need replacement:
# - libcudart.so.12 (CUDA Runtime)
# - libcuda.so.1 (CUDA Driver)
# - libcublas.so.11, libcublas.so.12 (cuBLAS)
# - libcudnn.so.8, libcudnn.so.9 (cuDNN)
# And other library-specific libraries
if [[ "$NO_SYMLINKS" == false ]] && [[ "$LIB_UPDATED" == true || "$FORCE_REINSTALL" == true ]]; then
    info "Checking for PyTorch bundled CUDA libraries to replace..."

    TORCH_LIBS_REPLACED=0

    # Search paths: system, pyenv, venv, conda, virtualenvwrapper, user installs
    SEARCH_PATHS=(
        # System Python
        /usr/lib/python3*/site-packages
        /usr/local/lib/python3*/site-packages
        /usr/lib/python3*/dist-packages
        /usr/local/lib/python3*/dist-packages
        # pyenv installations
        $HOME/.pyenv/versions/*/lib/python3*/site-packages
        $HOME/.pyenv/shims/*/lib/python*/site-packages
        # User installs (explicit paths for pip --user installs)
        $HOME/.local/lib/python3.*/site-packages
        $HOME/.local/lib/python3*/site-packages
        # conda environments
        $HOME/miniconda3/envs/*/lib/python*/site-packages
        $HOME/anaconda3/envs/*/lib/python*/site-packages
        # virtualenvwrapper
        $HOME/.virtualenvs/*/lib/python*/site-packages
        # Common venv locations
        $HOME/*/venv/lib/python*/site-packages
        $HOME/*/.venv/lib/python*/site-packages
        # Google Colab-style installations
        $HOME/.local/share/jupyter/runtime/*/site-packages
    )

    # PyTorch CUDA library patterns to replace (newer PyTorch uses nvidia/cuXX)
    TORCH_LIB_PATTERNS=(
        "nvidia/cuda_runtime/lib/libcudart.so.12"
        "nvidia/cuda_runtime/lib/libcudart.so.13"
        "nvidia/cuda_runtime/lib/libcuda.so.1"
        "nvidia/cu12/lib/libcudart.so.12"
        "nvidia/cu13/lib/libcudart.so.13"
        "nvidia/cudnn/lib/libcudnn.so.8"
        "nvidia/cudnn/lib/libcudnn.so.9"
        "torch/lib/cudart/cuda-runtime/lib/libcudart.so.12"
        "torch/lib/libcuda.so.1"
        "torch/lib/libcudnn.so.8"
        "torch/lib/libcudnn.so.9"
    )

    for sp_dir in "${SEARCH_PATHS[@]}"; do
        [[ -d "$sp_dir" ]] || continue

        for lib_pattern in "${TORCH_LIB_PATTERNS[@]}"; do
            torch_lib="$sp_dir/$lib_pattern"
            [[ -f "$torch_lib" ]] || continue

            # Check if already replaced (has __cudaRegisterFatBinary or __cudaPushCallConfiguration)
            if nm -D "$torch_lib" 2>/dev/null | grep -q '__cudaRegisterFatBinary\|__cudaPushCallConfiguration'; then
                info "  Already replaced: $torch_lib"
                continue
            fi

            # Back up the original
            [[ -f "${torch_lib}.real" ]] || cp "$torch_lib" "${torch_lib}.real"

            # Replace with gpushare library
            cp "$LIB_DIR/libgpushare_client.so.1.0.0" "$torch_lib"
            chmod 644 "$torch_lib"
            TORCH_LIBS_REPLACED=$((TORCH_LIBS_REPLACED + 1))
            ok "  Replaced: $torch_lib"
        done
    done

    if [[ $TORCH_LIBS_REPLACED -gt 0 ]]; then
        ok "Replaced $TORCH_LIBS_REPLACED PyTorch bundled CUDA libraries with gpushare"
    else
        info "No PyTorch installations found (non-fatal)"
    fi
fi

# ── 11. Systemd user service for dashboard ────────────────────────────────────
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

# ── 12. Patch ML frameworks ──────────────────────────────────────────────────
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

# ── 13. Summary ──────────────────────────────────────────────────────────────
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
if [[ "$REMOTE_FIRST" == true ]]; then
echo -e "  Remote Priority: ${GREEN}ACTIVE${NC} (Remote=Device 0)"
fi
echo -e "  Library:         $LIB_DIR/libgpushare_client.so"
echo -e "  Config:          $CONF_DIR/client.conf"
if [[ "$NO_SYMLINKS" == false ]]; then
echo -e "  CUDA override:   ${GREEN}ACTIVE${NC} (libcudart.so -> gpushare)"
echo -e "  API coverage:    2620+ functions (cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSOLVER, cuRAND, NVRTC, nvJPEG)"
echo -e "  Transfer opts:   ${GREEN}ACTIVE${NC} (tiered pinned pools, async memcpy, chunked pipelining, D2H prefetch, RDMA, LZ4/zstd, multi-server GPU pooling)"
if grep -q "^IBVERBS_LIB:FILEPATH=.*libibverbs" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
echo -e "  RDMA transport:  ${GREEN}BUILT IN${NC} (rdma-core auto-detected)"
elif [[ -f /usr/lib/libibverbs.so ]] || [[ -f /usr/lib64/libibverbs.so ]] || ldconfig -p 2>/dev/null | grep -q libibverbs; then
echo -e "  RDMA transport:  ${GREEN}AVAILABLE${NC} (rebuild to enable)"
else
echo -e "  RDMA transport:  ${YELLOW}NOT AVAILABLE${NC} (install rdma-core for InfiniBand support)"
fi
if grep -q "^LZ4_LIB:FILEPATH=.*liblz4" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
echo -e "  Compression:     ${GREEN}LZ4${NC} (auto-detected)"
elif grep -q "^ZSTD_LIB:FILEPATH=.*libzstd" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
echo -e "  Compression:     ${GREEN}zstd${NC} (auto-detected)"
else
echo -e "  Compression:     ${YELLOW}NOT AVAILABLE${NC} (install lz4 or zstd for transfer compression)"
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
