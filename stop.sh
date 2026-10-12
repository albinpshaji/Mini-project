#!/usr/bin/env bash
# ==============================================================================
#  OmniSpec-Triage Workspace Stop Entrypoint
#  Forwards commands to triage-agent/stop.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="$SCRIPT_DIR/triage-agent/stop.sh"

if [ -f "$TARGET_SCRIPT" ]; then
    chmod +x "$TARGET_SCRIPT"
    exec "$TARGET_SCRIPT" "$@"
else
    echo "❌ Error: Could not find $TARGET_SCRIPT"
    exit 1
fi
