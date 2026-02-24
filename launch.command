#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Weather Trader desktop launcher (macOS/Linux shell)
# Backend is auto-started by app_new.py if ../weather-engine is present.

if command -v uv >/dev/null 2>&1; then
  exec uv run python app_new.py
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 app_new.py
fi

if command -v python >/dev/null 2>&1; then
  exec python app_new.py
fi

echo "[ERROR] Neither 'uv' nor 'python' found in PATH."
echo "Install Python 3.12+ and uv, then retry."
exit 1
