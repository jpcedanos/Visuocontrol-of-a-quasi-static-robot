#!/usr/bin/env bash
# macOS / Linux: start the hand tracker and the Processing sketch together.
# Usage: ./launch.sh [--check] [--camera N]
cd "$(dirname "$0")" || exit 1
PY=$(command -v python3 || command -v python) || { echo "[launcher] ERROR: Python 3 not found"; exit 1; }
exec "$PY" launcher.py "$@"
