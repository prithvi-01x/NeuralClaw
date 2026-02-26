#!/usr/bin/env bash
# run.sh — NeuralClaw launcher
# Always uses the project .venv so you never need to activate it manually.
#
# Usage:
#   ./run.sh              → CLI interface (default)
#   ./run.sh onboard      → Interactive setup wizard
#   ./run.sh install news → Install the 'news' skill
#   ./run.sh skills       → List available skills
#   ./run.sh telegram     → Telegram bot
#   ./run.sh --help       → Show main.py help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "❌  .venv not found. Run first:"
    echo "    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Subcommands passed directly through to main.py
_SUBCOMMANDS="onboard install skills"

if [[ $# -eq 0 ]]; then
    exec "$VENV_PYTHON" "$SCRIPT_DIR/main.py" --interface cli
fi

# Check if the first arg is a known subcommand
FIRST="$1"
for sub in $_SUBCOMMANDS; do
    if [[ "$FIRST" == "$sub" ]]; then
        exec "$VENV_PYTHON" "$SCRIPT_DIR/main.py" "$@"
    fi
done

# If first arg is a bare interface name (no --), prepend --interface
if [[ "$FIRST" != --* && "$FIRST" != -* ]]; then
    INTERFACE="$FIRST"
    shift
    exec "$VENV_PYTHON" "$SCRIPT_DIR/main.py" --interface "$INTERFACE" "$@"
fi

# Pass everything through as-is
exec "$VENV_PYTHON" "$SCRIPT_DIR/main.py" "$@"
