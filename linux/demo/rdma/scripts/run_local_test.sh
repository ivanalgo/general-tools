#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-7471}"
CLIENT_IP="${CLIENT_IP:-192.168.130.1}"
SERVER_IP="${SERVER_IP:-192.168.130.2}"
ADDR="${ADDR:-$SERVER_IP}"
SERVER_BIND_ADDR="${SERVER_BIND_ADDR:-$SERVER_IP}"
CLIENT_BIND_ADDR="${CLIENT_BIND_ADDR:-${BIND_ADDR:-$CLIENT_IP}}"
CASES="${CASES:-${MODE:-send write read all}}"
SIZE="${SIZE:-1024}"
ITERS="${ITERS:-3}"
CLIENT_NETDEV="${CLIENT_NETDEV:-rdma-veth0}"
SERVER_NETDEV="${SERVER_NETDEV:-rdma-veth1}"
CLIENT_RDMA_DEV="${CLIENT_RDMA_DEV:-${DEV_A:-rxe_demo0}}"
SERVER_RDMA_DEV="${SERVER_RDMA_DEV:-${DEV_B:-rxe_demo1}}"
GID_INDEX="${GID_INDEX:-0}"
USE_CM="${USE_CM:-0}"
SELFTEST="${SELFTEST:-0}"
SHOW_IBV_WARNINGS="${SHOW_IBV_WARNINGS:-0}"
ALLOW_SAME_DEV="${ALLOW_SAME_DEV:-0}"

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

require_different_endpoints() {
  if [[ "$CLIENT_IP" == "$SERVER_IP" ]]; then
    echo "client/server IP must be different: CLIENT_IP=$CLIENT_IP SERVER_IP=$SERVER_IP" >&2
    exit 2
  fi
  if [[ "$CLIENT_RDMA_DEV" == "$SERVER_RDMA_DEV" && "$ALLOW_SAME_DEV" != "1" ]]; then
    echo "client/server RDMA devices must be different: CLIENT_RDMA_DEV=$CLIENT_RDMA_DEV SERVER_RDMA_DEV=$SERVER_RDMA_DEV" >&2
    echo "set ALLOW_SAME_DEV=1 only if you intentionally want loopback-on-one-RDMA-device testing" >&2
    exit 2
  fi
}

require_ip_on_netdev() {
  local ip="$1"
  local netdev="$2"
  if ! ip -o -4 addr show dev "$netdev" | grep -qw "$ip"; then
    echo "expected IP $ip on netdev $netdev, but it was not found" >&2
    ip -o -4 addr show dev "$netdev" >&2 || true
    exit 2
  fi
}

require_rdma_on_netdev() {
  local rdma_dev="$1"
  local netdev="$2"
  if ! rdma link show 2>/dev/null | grep -F "link $rdma_dev/" | grep -Fq "netdev $netdev"; then
    echo "expected RDMA device $rdma_dev to be bound to netdev $netdev" >&2
    rdma link show >&2 || true
    exit 2
  fi
}

verify_local_topology() {
  require_different_endpoints
  require_ip_on_netdev "$CLIENT_IP" "$CLIENT_NETDEV"
  require_ip_on_netdev "$SERVER_IP" "$SERVER_NETDEV"
  require_rdma_on_netdev "$CLIENT_RDMA_DEV" "$CLIENT_NETDEV"
  require_rdma_on_netdev "$SERVER_RDMA_DEV" "$SERVER_NETDEV"
  echo "[TOPOLOGY] client: ip=$CLIENT_IP netdev=$CLIENT_NETDEV rdma=$CLIENT_RDMA_DEV"
  echo "[TOPOLOGY] server: ip=$SERVER_IP netdev=$SERVER_NETDEV rdma=$SERVER_RDMA_DEV"
}

verify_local_topology

case_list=($CASES)
if [[ ${#case_list[@]} -eq 0 ]]; then
  echo "no test case selected; set CASES='send write read all' or MODE=send" >&2
  exit 2
fi

run_one_selftest_case() {
  local mode="$1"
  echo "[RUN] selftest mode=$mode client-dev=$CLIENT_RDMA_DEV server-dev=$SERVER_RDMA_DEV size=$SIZE iters=$ITERS gid-index=$GID_INDEX"
  run_demo --selftest --client-dev "$CLIENT_RDMA_DEV" --server-dev "$SERVER_RDMA_DEV" \
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

run_one_verbs_socket_case() {
  local mode="$1"
  local server_args=(--server --bind-addr "$SERVER_BIND_ADDR" --server-dev "$SERVER_RDMA_DEV" \
    --port "$PORT" --size "$SIZE" --iters "$ITERS" --gid-index "$GID_INDEX")
  local client_args=(--client --bind-addr "$CLIENT_BIND_ADDR" --addr "$ADDR" \
    --client-dev "$CLIENT_RDMA_DEV" --port "$PORT" --mode "$mode" --size "$SIZE" \
    --iters "$ITERS" --gid-index "$GID_INDEX")

  echo "[RUN] raw-verbs mode=$mode client=$CLIENT_BIND_ADDR/$CLIENT_RDMA_DEV server=$SERVER_BIND_ADDR/$SERVER_RDMA_DEV port=$PORT"
  run_demo "${server_args[@]}" &
  local server_pid=$!
  trap "kill $server_pid >/dev/null 2>&1 || true" EXIT
  sleep 1
  run_demo "${client_args[@]}"
  wait "$server_pid"
  trap - EXIT
  echo "[PASS] raw-verbs mode=$mode verified data movement"
}

if [[ "$USE_CM" == "1" ]]; then
  for mode in "${case_list[@]}"; do
    run_one_cm_case "$mode"
  done
elif [[ "$SELFTEST" == "1" ]]; then
  # RXE 设备刚创建后 GID/neighbor 状态可能需要极短时间稳定。
  sleep "${SETTLE_SECONDS:-2}"
  for mode in "${case_list[@]}"; do
    run_one_selftest_case "$mode"
  done
else
  # RXE 设备刚创建后 GID/neighbor 状态可能需要极短时间稳定。
  sleep "${SETTLE_SECONDS:-2}"
  for mode in "${case_list[@]}"; do
    run_one_verbs_socket_case "$mode"
  done
fi

echo "[PASS] all selected RDMA cases completed: ${case_list[*]}"
