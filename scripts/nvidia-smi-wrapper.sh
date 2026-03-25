#!/bin/bash
# nvidia-smi wrapper for gpushare clients with local GPUs
#
# This script runs the real nvidia-smi while avoiding gpushare libraries

# Export environment to avoid gpushare libraries
if [ -d /usr/local/lib/gpushare/real ]; then
    export LD_LIBRARY_PATH="/usr/local/lib/gpushare/real:$LD_LIBRARY_PATH"
fi

# Find real nvidia-smi
if [ -f /usr/bin/nvidia-smi.real ]; then
    exec /usr/bin/nvidia-smi.real "$@"
elif [ -f /opt/cuda/bin/nvidia-smi ]; then
    exec /opt/cuda/bin/nvidia-smi "$@"
else
    # Use the system nvidia-smi with updated library path
    exec /usr/bin/nvidia-smi "$@"
fi
