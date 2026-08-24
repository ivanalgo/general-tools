#!/bin/sh
set -eu
if [ "$#" -lt 4 ]; then
	echo "usage: $0 TOTAL_SIZE HOT_SIZE FAST_NODES SLOW_NODES [options...]" >&2
	exit 2
fi
total=$1 hot=$2 fast=$3 slow=$4
shift 4
bench=./hotcold-memory-bench
test -z "\${BENCH_BIN+x}" || bench=$BENCH_BIN
command -v numactl >/dev/null || { echo "error: numactl required" >&2; exit 1; }
exec numactl --cpunodebind="$fast" --membind="$fast" "$bench" \
	--total-size "$total" --hot-size "$hot" --assess tiering \
	--fast-nodes "$fast" --slow-nodes "$slow" "$@"
