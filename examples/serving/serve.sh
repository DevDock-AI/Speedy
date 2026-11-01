#!/bin/bash
# serve.sh <config.json> <port>: one Speedy server, started by llama-swap (one model loaded at a time).
# The API key is read from a file at start (never stored in the config or this script).
ulimit -l unlimited 2>/dev/null   # pinned host memory for the expert cache / KV stream
SPEEDY_ROOT=${SPEEDY_ROOT:-/opt/speedy}
KEY_FILE=${STRATA_API_KEY_FILE:-$HOME/.config/strata-llm/api_key}
[ -z "$STRATA_API_KEY" ] && [ -r "$KEY_FILE" ] && export STRATA_API_KEY=$(cat "$KEY_FILE")
cd "$SPEEDY_ROOT"
exec "${SPEEDY_PYTHON:-$SPEEDY_ROOT/.venv/bin/python}" serve/server.py --engine strata --config "$1" --host 127.0.0.1 --port "$2"
