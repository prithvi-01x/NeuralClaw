#!/usr/bin/env bash
# Symlinks neuralclaw_cli.py to ~/bin/neuralclaw so it's available everywhere
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLI_FILE="$SCRIPT_DIR/neuralclaw_cli.py"

if [ ! -f "$CLI_FILE" ]; then
    echo "✗  neuralclaw_cli.py not found at $CLI_FILE"
    exit 1
fi

chmod +x "$CLI_FILE"
mkdir -p ~/bin
ln -sf "$CLI_FILE" ~/bin/neuralclaw
echo "✓ neuralclaw CLI installed. Make sure ~/bin is in your PATH."
