#!/usr/bin/env bash
# Prithivi launcher.
#
# Runs main.py from the project folder so `.env` is found regardless of where
# you call it from. Examples:
#
#     ./run.sh                 # interactive chat
#     ./run.sh voice           # voice mode
#     ./run.sh "what time is it"
#     ./run.sh doctor
#     ./run.sh --demo
#
# If the shebang line does not work on your device, run: bash run.sh voice
set -euo pipefail

cd "$(dirname "$0")"

if [ -n "${PRITHIVI_PYTHON:-}" ]; then
  PY="$PRITHIVI_PYTHON"
elif command -v python >/dev/null 2>&1; then
  PY="python"
elif [ -x /data/data/com.termux/files/usr/bin/python ]; then
  PY="/data/data/com.termux/files/usr/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "run.sh: no python interpreter found (set PRITHIVI_PYTHON to its path)" >&2
  exit 1
fi

exec "$PY" main.py "$@"
