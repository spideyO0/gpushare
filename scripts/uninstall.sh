#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# gpushare — Cross-platform Uninstall Script
#
# Detects the OS and removes all gpushare components. Use --purge to also
# remove configuration files.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

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

# ── Help ──────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: uninstall.sh [OPTIONS]

Remove gpushare from this machine (server and/or client components).

Options:
  -h, --help    Show this help
  --purge       Also remove configuration files
  --yes         Skip confirmation prompt
  --dry-run     Show what would be removed without deleting anything
EOF
    exit 0
fi

# ── Parse flags ───────────────────────────────────────────────────────────────
PURGE=false
YES=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --purge)   PURGE=true ;;
        --yes|-y)  YES=true ;;
        --dry-run) DRY_RUN=true ;;
        *) err "Unknown option: $arg (try --help)"; exit 1 ;;
    esac
done

# ── Detect OS ─────────────────────────────────────────────────────────────────
OS="unknown"
case "$(uname -s)" in
    Linux*)  OS="linux" ;;
    Darwin*) OS="macos" ;;
    MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
esac

echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  gpushare — Uninstaller (${OS})${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo

# ── Root check (Linux) ───────────────────────────────────────────────────────
if [[ "$OS" == "linux" && $EUID -ne 0 ]]; then
    err "This script must be run as root on Linux (sudo $0)"
    exit 1
fi

NEED_SUDO=""
if [[ "$OS" == "macos" && ! -w /usr/local/lib ]]; then
    NEED_SUDO="sudo"
fi

# ── Build removal list ───────────────────────────────────────────────────────
FILES_TO_REMOVE=()
DIRS_TO_REMOVE=()
COMMANDS_TO_RUN=()

remove_file() {
    if [[ -f "$1" || -L "$1" ]]; then
        FILES_TO_REMOVE+=("$1")
    fi
}

remove_dir() {
    if [[ -d "$1" ]]; then
        DIRS_TO_REMOVE+=("$1")
    fi
}

# ── Linux components ─────────────────────────────────────────────────────────
if [[ "$OS" == "linux" ]]; then
    # Server binary
    remove_file /usr/local/bin/gpushare-server

    # Client library + all symlinks (CUDA runtime, driver API, NVML)
    remove_dir /usr/local/lib/gpushare

    # Commands
    remove_file /usr/local/bin/gpushare-monitor
    remove_file /usr/local/bin/gpushare-patch

    # Shared files
    remove_dir /usr/local/share/gpushare

    # ldconfig
    remove_file /etc/ld.so.conf.d/gpushare.conf
    COMMANDS_TO_RUN+=("ldconfig")

    # Systemd services
    for svc in gpushare-server gpushare-dashboard; do
        if systemctl is-active "$svc" >/dev/null 2>&1; then
            COMMANDS_TO_RUN+=("systemctl stop $svc")
        fi
        if systemctl is-enabled "$svc" >/dev/null 2>&1; then
            COMMANDS_TO_RUN+=("systemctl disable $svc")
        fi
        remove_file "/etc/systemd/system/${svc}.service"
    done

    # User service
    remove_file /etc/systemd/user/gpushare-dashboard.service
    COMMANDS_TO_RUN+=("systemctl daemon-reload")

    # Python client
    if command -v pip3 >/dev/null 2>&1; then
        if pip3 show gpushare >/dev/null 2>&1; then
            COMMANDS_TO_RUN+=("pip3 uninstall -y gpushare")
        fi
    fi

    # Config (only with --purge)
    if [[ "$PURGE" == true ]]; then
        remove_dir /etc/gpushare
    fi
fi

# ── macOS components ─────────────────────────────────────────────────────────
if [[ "$OS" == "macos" ]]; then
    # Client library + all symlinks
    remove_dir /usr/local/lib/gpushare
    remove_file /usr/local/lib/libcudart.dylib
    remove_file /usr/local/lib/libcudart.12.dylib
    remove_file /usr/local/lib/libcudart.13.dylib
    remove_file /usr/local/lib/libcuda.dylib
    remove_file /usr/local/lib/libcuda.1.dylib
    remove_file /usr/local/lib/libnvidia-ml.dylib
    remove_file /usr/local/lib/libnvidia-ml.1.dylib
    remove_file /usr/local/lib/libcublas.dylib
    remove_file /usr/local/lib/libcublasLt.dylib
    remove_file /usr/local/lib/libcudnn.dylib
    remove_file /usr/local/lib/libcufft.dylib
    remove_file /usr/local/lib/libcusparse.dylib
    remove_file /usr/local/lib/libcusolver.dylib
    remove_file /usr/local/lib/libcurand.dylib
    remove_file /usr/local/lib/libnvrtc.dylib
    remove_file /usr/local/lib/libnvjpeg.dylib

    # Commands
    remove_file /usr/local/bin/gpushare-monitor
    remove_file /usr/local/bin/gpushare-patch

    # Shared files
    remove_dir /usr/local/share/gpushare

    # PATH entry
    remove_file /etc/paths.d/gpushare

    # launchd agent
    PLIST="$HOME/Library/LaunchAgents/com.gpushare.dashboard.plist"
    if [[ -f "$PLIST" ]]; then
        COMMANDS_TO_RUN+=("launchctl unload '$PLIST' 2>/dev/null || true")
        FILES_TO_REMOVE+=("$PLIST")
    fi

    # Python client
    if command -v pip3 >/dev/null 2>&1; then
        if pip3 show gpushare >/dev/null 2>&1; then
            COMMANDS_TO_RUN+=("pip3 uninstall -y gpushare")
        fi
    fi

    # Config (only with --purge)
    if [[ "$PURGE" == true ]]; then
        remove_dir "$HOME/.config/gpushare"
    fi
fi

# ── Windows (via Git Bash / MSYS2) ───────────────────────────────────────────
if [[ "$OS" == "windows" ]]; then
    info "For Windows, run the following in an Administrator PowerShell:"
    echo
    cat <<'EOF'
    # Stop and remove scheduled task
    Unregister-ScheduledTask -TaskName "gpushare-dashboard" -Confirm:$false -ErrorAction SilentlyContinue

    # Remove virtual GPU from registry (Task Manager entry)
    $classPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    Get-ChildItem -Path $classPath -ErrorAction SilentlyContinue | ForEach-Object {
        $managed = (Get-ItemProperty -Path $_.PSPath -Name "gpushare_managed" -EA SilentlyContinue).gpushare_managed
        if ($managed -eq 1) { Remove-Item -Path $_.PSPath -Recurse -Force }
    }

    # Remove Defender exclusion
    Remove-MpPreference -ExclusionPath "C:\Program Files\gpushare" -ErrorAction SilentlyContinue

    # Remove files (includes cudart64_*.dll, nvcuda.dll, nvml.dll, nvidia-smi.bat)
    Remove-Item -Recurse -Force "C:\Program Files\gpushare" -ErrorAction SilentlyContinue

    # Remove from PATH
    $path = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $path = ($path -split ";" | Where-Object { $_ -ne "C:\Program Files\gpushare" }) -join ";"
    [Environment]::SetEnvironmentVariable("Path", $path, "Machine")

    # Remove config (--purge equivalent)
    # Remove-Item -Recurse -Force "C:\ProgramData\gpushare" -ErrorAction SilentlyContinue

    # Remove Python client
    # python -m pip uninstall -y gpushare
EOF
    echo
    info "Copy and paste the commands above into PowerShell."
    exit 0
fi

# ── Show plan ─────────────────────────────────────────────────────────────────
if [[ ${#FILES_TO_REMOVE[@]} -eq 0 && ${#DIRS_TO_REMOVE[@]} -eq 0 && ${#COMMANDS_TO_RUN[@]} -eq 0 ]]; then
    info "Nothing to uninstall. gpushare does not appear to be installed."
    exit 0
fi

echo -e "${BOLD}The following will be removed:${NC}"
echo

for f in "${FILES_TO_REMOVE[@]}"; do
    echo -e "  ${RED}rm${NC}  $f"
done
for d in "${DIRS_TO_REMOVE[@]}"; do
    echo -e "  ${RED}rm -r${NC}  $d"
done
for c in "${COMMANDS_TO_RUN[@]}"; do
    echo -e "  ${CYAN}run${NC}  $c"
done

if [[ "$PURGE" == false ]]; then
    echo
    if [[ "$OS" == "linux" ]]; then
        echo -e "  ${YELLOW}KEPT${NC}  /etc/gpushare/ (use --purge to remove)"
    elif [[ "$OS" == "macos" ]]; then
        echo -e "  ${YELLOW}KEPT${NC}  ~/.config/gpushare/ (use --purge to remove)"
    fi
fi
echo

# ── Confirm ───────────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == true ]]; then
    info "Dry run — nothing was removed."
    exit 0
fi

if [[ "$YES" == false ]]; then
    read -rp "Proceed with uninstall? [y/N] " answer
    if [[ "${answer,,}" != "y" ]]; then
        info "Aborted."
        exit 0
    fi
fi

# ── Execute removal ──────────────────────────────────────────────────────────
info "Removing gpushare..."

for c in "${COMMANDS_TO_RUN[@]}"; do
    info "Running: $c"
    eval $NEED_SUDO $c 2>/dev/null || warn "Command had non-zero exit: $c"
done

for f in "${FILES_TO_REMOVE[@]}"; do
    $NEED_SUDO rm -f "$f" && ok "Removed $f" || warn "Could not remove $f"
done

for d in "${DIRS_TO_REMOVE[@]}"; do
    $NEED_SUDO rm -rf "$d" && ok "Removed $d" || warn "Could not remove $d"
done

# ── Done ──────────────────────────────────────────────────────────────────────
echo
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  gpushare uninstalled successfully.${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo
if [[ "$PURGE" == false ]]; then
    info "Configuration files were kept. Use --purge to remove them too."
fi
echo
