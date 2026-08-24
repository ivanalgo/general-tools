#!/bin/sh
set -eu
bin=$1
tmp=/tmp/hotcold-memory-bench.$$
trap 'rm -f "$tmp"' EXIT INT TERM
"$bin" --total-size 16M --hot-size 4M --duration 1 >"$tmp"
grep -q '^CONFIG ' "$tmp"
grep -q '^PERF ' "$tmp"
"$bin" --total-size 16M --hot-size 4M --duration 1 --output csv >"$tmp"
grep -q '^PERF,' "$tmp"
echo "smoke tests passed"
