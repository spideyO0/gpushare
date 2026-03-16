#!/bin/bash
# Legacy wrapper — detects OS and redirects to the correct installer
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$(uname -s)" in
    Darwin*)  exec "$SCRIPT_DIR/install-client-macos.sh" "$@" ;;
    Linux*)   exec "$SCRIPT_DIR/install-client-linux.sh" "$@" ;;
    MINGW*|MSYS*|CYGWIN*)
        echo "On Windows, run in PowerShell as Administrator:"
        echo "  .\\install-client-windows.ps1"
        exit 1 ;;
    *)
        echo "Unknown OS: $(uname -s)"
        exit 1 ;;
esac
