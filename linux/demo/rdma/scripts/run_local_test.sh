#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-7471}"
ADDR="${ADDR:-192.168.130.2}"
SERVER_BIND_ADDR="${SERVER_BIND_ADDR:-}"
CLIENT_BIND_ADDR="${CLIENT_BIND_ADDR:-${BIND_ADDR:-}}"
CASES="${CASES:-${MODE:-send write read all}}"
SIZE="${SIZE:-1024}"
ITERS="${ITERS:-3}"
DEV_A="${DEV_A:-rxe_demo0}"
DEV_B="${DEV_B:-rxe_demo1}"
GID_INDEX="${GID_INDEX:-0}"
USE_CM="${USE_CM:-0}"
SHOW_IBV_WARNINGS="${SHOW_IBV_WARNINGS:-0}"

run_demo() {
  if [[ "$SHOW_IBV_WARNINGS" == "1" ]]; then
    "$DIR/rdma_demo" "$@"
  else
    # Some OFED installs leave provider config files for HCAs that are not
    # installed on this host. libibverbs prints one warning per missing provider
    # before it successfully opens the RXE provider; filter only that known noise.
    "$DIR/rdma_demo" "$@" \
      2> >(grep -v -E "^libibverbs: Warning: couldn't load driver 'lib.*-rdmav[0-9]+\.so'" >&2)
  fi
}

make -C "$DIR"

case_list=($CASES)
if [[ ${#case_list[@]} -eq 0 ]]; then
  echo "no test case selected; set CASES='send write read all' or MODE=send" >&2
  exit 2
fi

run_one_selftest_case() {
  local mode="$1"
  echo "[RUN] selftest mode=$mode dev-a=$DEV_A dev-b=$DEV_B size=$SIZE iters=$ITERS gid-index=$GID_INDEX"
  run_demo --selftest --dev-a "$DEV_A" --dev-b "$DEV_B" \
    --mode "$mode" --size "$SIZE" --iters "$ITERS" --gid-index "$GID_INDEX"
  echo "[PASS] selftest mode=$mode verified data movement"
}

run_one_cm_case() {
  local mode="$1"
  local server_args=(--server --port "$PORT" --size "$SIZE" --iters "$ITERS")
  local client_args=(--client --addr "$ADDR" --port "$PORT" --mode "$mode" --size "$SIZE" --iters "$ITERS")
  if [[ -n "$SERVER_BIND_ADDR" ]]; then
    server_args+=(--bind-addr "$SERVER_BIND_ADDR")
  fi
  if [[ -n "$CLIENT_BIND_ADDR" ]]; then
    client_args+=(--bind-addr "$CLIENT_BIND_ADDR")
  fi

  echo "[RUN] RDMA-CM mode=$mode server-bind=${SERVER_BIND_ADDR:-<any>} client-bind=${CLIENT_BIND_ADDR:-<route>} addr=$ADDR port=$PORT"
  run_demo "${server_args[@]}" &
  local server_pid=$!
  trap 'kill "$server_pid" >/dev/null 2>&1 || true' EXIT
  sleep 1
  run_demo "${client_args[@]}"
  wait "$server_pid"
  trap - EXIT
  echo "[PASS] RDMA-CM mode=$mode verified data movement"
}

if [[ "$USE_CM" == "1" ]]; then
  for mode in "${case_list[@]}"; do
    run_one_cm_case "$mode"
  done
else
  # RXE 设备刚创建后 GID/neighbor 状态可能需要极短时间稳定。
  sleep "${SETTLE_SECONDS:-2}"
  for mode in "${case_list[@]}"; do
    run_one_selftest_case "$mode"
  done
fi

echo "[PASS] all selected RDMA cases completed: ${case_list[*]}"
