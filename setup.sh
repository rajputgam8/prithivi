#!/usr/bin/env bash
# Prithivi setup check for Termux.
#
# Run from NATIVE Termux (the app's own shell), not inside proot:
#     ~/prithivi-app/setup.sh
#
# It only CHECKS by default. To also install what is missing, pass --install:
#     ~/prithivi-app/setup.sh --install
set -eu

cd "$(dirname "$0")"

cyan() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
ok()   { printf '  \033[32mOK\033[0m  %s\n' "$1"; }
warn() { printf '  \033[33m!!\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mXX\033[0m  %s\n' "$1"; }

INSTALL=0
[ "${1:-}" = "--install" ] && INSTALL=1

is_termux() { [ -d /data/data/com.termux/files/usr ]; }

cyan "Python"
PY="${PRITHIVI_PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python >/dev/null 2>&1; then PY="$(command -v python)"
  elif [ -x /data/data/com.termux/files/usr/bin/python ]; then PY="/data/data/com.termux/files/usr/bin/python"
  elif command -v python3 >/dev/null 2>&1; then PY="$(command -v python3)"
  fi
fi
if [ -z "$PY" ]; then
  bad "python not found"
  echo "      fix:  pkg install python"
  exit 1
fi
ok "$PY  ($($PY --version 2>&1))"

cyan "Python packages"
if "$PY" -c "import httpx, rich" >/dev/null 2>&1; then
  ok "httpx and rich are installed"
else
  warn "httpx and/or rich are missing"
  if [ "$INSTALL" = 1 ]; then
    cyan "Installing requirements"
    "$PY" -m pip install -r requirements.txt
    ok "installed"
  else
    echo "      fix:  $PY -m pip install -r requirements.txt"
    echo "      or:   ~/prithivi-app/setup.sh --install"
  fi
fi

cyan "Voice (Termux:API)"
if is_termux; then
  if command -v termux-tts-speak >/dev/null 2>&1 && command -v termux-speech-to-text >/dev/null 2>&1; then
    ok "termux-api commands present"
  else
    warn "termux-api not installed"
    if [ "$INSTALL" = 1 ]; then
      pkg install -y termux-api
      ok "installed termux-api"
    else
      echo "      fix:  pkg install termux-api"
    fi
  fi
  echo "      also install the 'Termux:API' app from F-Droid, then allow microphone access."
else
  warn "not running in native Termux shell — voice needs native Termux"
fi

cyan "Config"
if [ -f .env ]; then
  ok ".env found"
else
  warn ".env missing"
  echo "      fix:  cp .env.example .env  # then set PRITHIVI_LLM_API_KEY"
fi

cyan "Readiness (doctor)"
"$PY" main.py doctor || true

printf '\n\033[1;32mDone.\033[0m Run Prithivi with:\n'
echo "  ./run.sh                 # text chat"
echo "  ./run.sh voice           # voice mode"
echo "  ./run.sh \"what time is it\""
