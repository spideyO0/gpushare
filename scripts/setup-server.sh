#!/bin/bash
# Legacy wrapper — redirects to the full installer
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Redirecting to install-server-arch.sh..."
exec "$SCRIPT_DIR/install-server-arch.sh" "$@"
