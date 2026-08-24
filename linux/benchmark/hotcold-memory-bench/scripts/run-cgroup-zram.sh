#!/bin/sh
set -eu
if [ "$#" -lt 3 ]; then
	echo "usage: $0 TOTAL_SIZE HOT_SIZE TARGET_MEMORY_MAX [options...]" >&2
	exit 2
fi
total=$1 hot=$2 target=$3
shift 3
bench=./hotcold-memory-bench
test -z "\${BENCH_BIN+x}" || bench=$BENCH_BIN
root=/sys/fs/cgroup
test -z "\${CGROUP_ROOT+x}" || root=$CGROUP_ROOT
group=$root/hotcold-memory-bench.$$
[ "$(id -u)" -eq 0 ] || { echo "error: run as root" >&2; exit 1; }
[ -f "$root/cgroup.controllers" ] || { echo "error: cgroup v2 required" >&2; exit 1; }
[ "$(wc -l </proc/swaps)" -gt 1 ] || { echo "error: configure zram swap first" >&2; exit 1; }
mkdir "$group"
cleanup() {
	printf 'max\n' >"$group/memory.max" 2>/dev/null || true
	rmdir "$group" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
sh -c 'printf "%s\n" $$ >"$1/cgroup.procs"; shift; exec "$@"' \
	sh "$group" "$bench" --total-size "$total" --hot-size "$hot" \
	--assess eviction --cgroup "$group" --memory-max "$target" "$@"
