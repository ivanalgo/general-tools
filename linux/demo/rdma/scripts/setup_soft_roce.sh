#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-up}"
DEV0="rdma-veth0"
DEV1="rdma-veth1"
RXE0="rxe_demo0"
RXE1="rxe_demo1"
IP0="192.168.130.1/24"
IP1="192.168.130.2/24"

need_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "please run as root" >&2
    exit 1
  fi
}

del_rxe() {
  local name="$1"
  if rdma link show "$name" >/dev/null 2>&1; then
    rdma link delete "$name" || true
  fi
}

case "$ACTION" in
  up)
    need_root
    modprobe rdma_rxe
    "$0" down >/dev/null 2>&1 || true
    ip link add "$DEV0" type veth peer name "$DEV1"
    ip addr replace "$IP0" dev "$DEV0"
    ip addr replace "$IP1" dev "$DEV1"
    ip link set "$DEV0" up
    ip link set "$DEV1" up
    rdma link add "$RXE0" type rxe netdev "$DEV0"
    rdma link add "$RXE1" type rxe netdev "$DEV1"
    echo "created $RXE0 on $DEV0 (${IP0%/*})"
    echo "created $RXE1 on $DEV1 (${IP1%/*})"
    rdma link show
    ;;
  down)
    need_root
    del_rxe "$RXE0"
    del_rxe "$RXE1"
    ip link delete "$DEV0" >/dev/null 2>&1 || true
    ip link delete "$DEV1" >/dev/null 2>&1 || true
    ;;
  *)
    echo "usage: $0 [up|down]" >&2
    exit 2
    ;;
esac
