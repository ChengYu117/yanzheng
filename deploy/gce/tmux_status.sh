#!/usr/bin/env bash
set -euo pipefail

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed." >&2
  exit 1
fi

echo "=== tmux sessions ==="
tmux ls || true
echo
echo "=== recent logs ==="
find "${1:-.}" -maxdepth 3 -type f -name "run.log" 2>/dev/null | sort
