#!/usr/bin/env bash
# Standalone RViz on WSL2 + WSLg (when not using launch env injection).
set -euo pipefail
if [[ -d /mnt/wslg ]]; then
  export DISPLAY="${DISPLAY:-:0}"
  export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
fi
exec rviz2 "$@"
