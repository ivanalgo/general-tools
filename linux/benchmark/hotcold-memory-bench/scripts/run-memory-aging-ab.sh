#!/bin/bash
set -euo pipefail

# Run one side of the native versus memory.aging comparison. Invoke this
# script under the stock kernel with "native" and under the patched kernel
# with "aging". An aging result is invalid unless the requested number of
# kernel-reported rounds completes before memory.max is lowered.

mode=${1:?usage: $0 native|aging [anon|file] [result-directory]}
backing=${2:-anon}
result=${3:-/tmp/hotcold-aging-ab}

case "$mode" in
	native|aging) ;;
	*) echo "invalid mode: $mode" >&2; exit 2 ;;
esac
case "$backing" in
	anon|file) ;;
	*) echo "invalid backing: $backing" >&2; exit 2 ;;
esac

script_dir=$(cd "$(dirname "$0")" && pwd)
bench=${BENCH:-$script_dir/../hotcold-memory-bench}
cgroot=${CGROOT:-/sys/fs/cgroup}
group=$cgroot/hotcold-aging-$mode-$backing
rounds_required=${ROUNDS:-4}
aging_budget=${AGING_BUDGET:-1G}
aging_period=${AGING_PERIOD:-1}
total_size=${TOTAL_SIZE:-10G}
hot_size=${HOT_SIZE:-2G}
memory_max=${MEMORY_MAX:-6G}
assess_at=${ASSESS_AT:-240}
duration=${DURATION:-270}
settle=${SETTLE:-10}
deadline=${AGING_DEADLINE:-210}
mkdir -p "$result"

log=$result/$mode-$backing.log
samples=$result/$mode-$backing-memory.csv
aging_log=$result/$mode-$backing-aging.csv

cleanup_group()
{
	if [ -d "$group" ]; then
		local pids
		pids=$(sudo cat "$group/cgroup.procs" 2>/dev/null || true)
		for pid in $pids; do
			sudo kill -KILL "$pid" 2>/dev/null || true
		done
		sudo sh -c "echo max > '$group/memory.max'" 2>/dev/null || true
		sudo rmdir "$group" 2>/dev/null || true
	fi
}

sample_memory()
{
	local phase=$1 now key value
	now=$(date +%s%N)
	for key in anon file inactive_anon active_anon inactive_file active_file \
		workingset_refault_anon workingset_refault_file pgscan pgsteal; do
		value=$(sudo awk -v key="$key" '$1 == key { print $2; found = 1 }
			END { if (!found) print 0 }' "$group/memory.stat")
		printf '%s,%s,%s,%s\n' "$now" "$phase" "$key" "$value" >>"$samples"
	done
	printf '%s,%s,%s,%s\n' "$now" "$phase" memory_current \
		"$(sudo cat "$group/memory.current")" >>"$samples"
	printf '%s,%s,%s,%s\n' "$now" "$phase" memory_swap_current \
		"$(sudo cat "$group/memory.swap.current")" >>"$samples"
}

cleanup()
{
	if [ -n "${sampler:-}" ]; then
		kill "$sampler" 2>/dev/null || true
		wait "$sampler" 2>/dev/null || true
	fi
	cleanup_group
}
trap cleanup EXIT

cleanup_group
sudo mkdir "$group"
sudo sh -c "echo max > '$group/memory.max'; echo max > '$group/memory.swap.max'"

if [ "$mode" = aging ] && [ ! -e "$group/memory.aging" ]; then
	echo "memory.aging is unavailable on $(uname -r)" >&2
	exit 1
fi

printf 'timestamp_ns,phase,key,value\n' >"$samples"
printf 'index,start_ns,end_ns,duration_ms,rounds,scanned_pages\n' >"$aging_log"
: >"$log"

backing_arg=()
if [ "$backing" = file ]; then
	backing_arg=(--backing-file "/data00/hotcold-aging-$mode.tmp")
fi

sudo sh -c "echo \$\$ > '$group/cgroup.procs'; exec taskset -c 0 '$bench' \
	--total-size '$total_size' --hot-size '$hot_size' --duration '$duration' --interval 10 \
	--seed 20260824 --assess eviction --assess-at '$assess_at' --settle '$settle' \
	--cgroup '$group' --memory-max '$memory_max' ${backing_arg[*]}" >"$log" 2>&1 &
runner=$!

until grep -q '^CONFIG ' "$log" 2>/dev/null; do
	# sudo changes the child owner to root, so unprivileged kill -0 reports
	# EPERM even while the benchmark is alive.
	if ! ps -p "$runner" -o pid= >/dev/null; then
		wait "$runner" || true
		echo "benchmark failed during initialization" >&2
		tail -n 20 "$log" >&2 || true
		exit 1
	fi
	sleep 0.2
done
start_epoch=$(date +%s)

(
	while ps -p "$runner" -o pid= >/dev/null; do
		sample_memory periodic
		sleep 10
	done
) &
sampler=$!

sample_memory before_aging

if [ "$mode" = aging ]; then
	base_rounds=$(sudo awk '$1 == "rounds" { print $2 }' "$group/memory.aging")
	target_rounds=$((base_rounds + rounds_required))
	index=0
	while :; do
		read -r rounds scanned < <(sudo awk '
			$1 == "rounds" { rounds = $2 }
			$1 == "scanned" { scanned = $2 }
			END { print rounds + 0, scanned + 0 }
		' "$group/memory.aging")
		if [ "$rounds" -ge "$target_rounds" ]; then
			break
		fi
		if [ $(( $(date +%s) - start_epoch )) -ge "$deadline" ]; then
			echo "aging failed to finish $rounds_required rounds before deadline" >&2
			exit 1
		fi
		index=$((index + 1))
		start_ns=$(date +%s%N)
		printf '%s\n' "$aging_budget" | sudo tee "$group/memory.aging" >/dev/null
		end_ns=$(date +%s%N)
		read -r rounds scanned < <(sudo awk '
			$1 == "rounds" { rounds = $2 }
			$1 == "scanned" { scanned = $2 }
			END { print rounds + 0, scanned + 0 }
		' "$group/memory.aging")
		printf '%s,%s,%s,%s,%s,%s\n' "$index" "$start_ns" "$end_ns" \
			$(((end_ns - start_ns) / 1000000)) "$rounds" "$scanned" \
			>>"$aging_log"
		sleep "$aging_period"
	done
	elapsed=$(( $(date +%s) - start_epoch ))
	remaining=$((assess_at - elapsed))
	echo "AGING_COMPLETE rounds=$((rounds - base_rounds)) scanned_pages=$scanned elapsed_s=$elapsed remaining_s=$remaining" | tee -a "$log"
	if [ "$remaining" -lt 20 ]; then
		echo "less than 20 seconds remain before memory.max assessment" >&2
		exit 1
	fi
	sample_memory after_aging
fi

wait "$runner"
runner=
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
sampler=
sample_memory after_reclaim

grep -q '^ACCURACY ' "$log"
grep -E '^(CONFIG|AGING_COMPLETE|EVENT|ACCURACY)' "$log"
