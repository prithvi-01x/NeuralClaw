#!/usr/bin/env bash
# run.sh — NeuralClaw launcher
# Always uses the project .venv so you never need to activate it manually.
#
# Usage:
#   ./run.sh              → CLI interface (default)
#   ./run.sh telegram     → Telegram bot
#   ./run.sh --help       → Show main.py help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "❌  .venv not found. Run first:"
    echo "    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Default to CLI interface if no args given
if [[ $# -eq 0 ]]; then
    exec "$VENV_PYTHON" "$SCRIPT_DIR/main.py" --interface cli
fi

# If first arg is a bare interface name (no --), prepend --interface
if [[ "$1" != --* && "$1" != -* ]]; then
    INTERFACE="$1"
    shift
    exec "$VENV_PYTHON" "$SCRIPT_DIR/main.py" --interface "$INTERFACE" "$@"
fi

# Pass everything through as-is
exec "$VENV_PYTHON" "$SCRIPT_DIR/main.py" "$@"
