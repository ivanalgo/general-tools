#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-7471}"
ADDR="${ADDR:-192.168.130.2}"
MODE="${MODE:-all}"
SIZE="${SIZE:-1024}"
ITERS="${ITERS:-3}"
DEV_A="${DEV_A:-rxe_demo0}"
DEV_B="${DEV_B:-rxe_demo1}"
GID_INDEX="${GID_INDEX:-0}"
USE_CM="${USE_CM:-0}"

make -C "$DIR"

if [[ "$USE_CM" == "1" ]]; then
  "$DIR/rdma_demo" --server --port "$PORT" --size "$SIZE" --iters "$ITERS" &
  server_pid=$!
  trap 'kill "$server_pid" >/dev/null 2>&1 || true' EXIT
  sleep 1
  "$DIR/rdma_demo" --client --addr "$ADDR" --port "$PORT" --mode "$MODE" --size "$SIZE" --iters "$ITERS"
  wait "$server_pid"
  trap - EXIT
else
  # RXE 设备刚创建后 GID/neighbor 状态可能需要极短时间稳定。
  sleep "${SETTLE_SECONDS:-2}"
  "$DIR/rdma_demo" --selftest --dev-a "$DEV_A" --dev-b "$DEV_B" \
    --mode "$MODE" --size "$SIZE" --iters "$ITERS" --gid-index "$GID_INDEX"
fi
