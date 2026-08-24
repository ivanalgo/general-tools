# hotcold-memory-bench

hotcold-memory-bench is a page-granularity Linux benchmark for evaluating
hot/cold memory decisions. It creates a randomized pointer-chase working set,
records every page's access count, and compares that ground truth with:

- resident versus evicted pages after cgroup reclaim, including zram swap; or
- fast versus slow NUMA placement, including DRAM/CXL memory tiering.

The primary results are deliberately small:

- pointer chases per second and nanoseconds per chase;
- cold-placement purity;
- hot-page misplacement rate; and
- access-weighted hotness loss.

## Build

    make
    make test

The core binary uses Linux system calls directly and does not require libnuma.
The optional tiering wrapper uses numactl for initial CPU and memory binding.

## Workload

The program allocates --total-size of private anonymous memory and populates
every base page. The first --hot-size forms a randomly shuffled, page-level
pointer ring. Only this ring is accessed after initialization; all other pages
are known cold pages. Warm-up accesses are not counted.

Transparent huge pages are disabled for the mapping so one counter and placement
observation corresponds to one base page. Use a fixed --seed for repeatability.

Use --backing-file PATH to create an exclusive, temporary file-backed mapping.
The file is unlinked immediately after mapping and initialization is flushed
with msync(), leaving clean cache pages that can be reclaimed without swap.
This mode tests LRU accuracy independently of anonymous swap configuration.

## Performance

    ./hotcold-memory-bench \
      --total-size 10G --hot-size 2G \
      --duration 120 --interval 1

Example:

    PERF time=1.001s chases_per_sec=21342012 ns_per_chase=46.86 major_faults=0

Use --output csv for machine-readable output. --no-page-counters measures the
lowest pointer-chase overhead but cannot be combined with assessment.

## Cgroup and zram eviction accuracy

Configure zram swap using the distribution's normal mechanism first. The helper
does not reset or modify an existing swap device.

    sudo scripts/run-cgroup-zram.sh 10G 2G 6G \
      --duration 180 --assess-at 120 --settle 10

At assessment time the benchmark pauses pointer chasing, lowers memory.max,
waits for reclaim, and queries residency with mincore() without touching the
workload pages. Pausing prevents an incorrectly evicted hot page from faulting
back in before it can be measured.

For a clean page-cache eviction test, add:

    --backing-file /data00/hotcold-memory-bench.tmp

Do not set memory.max equal to the hot-data size. Page tables, access counters,
executable pages, shared libraries, and runtime data also need memory.

## NUMA and CXL tiering accuracy

Tiering mode pauses and queries every page's NUMA node with move_pages(2). It
observes placement but does not migrate pages. Automatic NUMA balancing, DAMON,
a userspace policy, or another mechanism must promote and demote pages.

For DRAM nodes 0-1 and CXL node 2:

    scripts/run-numa-tiering.sh 10G 2G 0-1 2 \
      --duration 180 --assess-at 120

The helper initially allocates on fast nodes. The tiering setup must create
pressure or perform demotion before assessment. Pages outside both node sets are
reported as unknown_pages and excluded from the score.

## Accuracy metrics

Pages with at most --cold-threshold accesses are cold; the default is zero.
The selected set means nonresident pages in eviction mode and slow-node pages
in tiering mode.

    cold_placement_purity = selected cold pages / all selected pages
    hot_misplacement_rate = selected hot pages / all hot pages
    hotness_loss          = accesses to selected pages / all page accesses

Ideal results are 100% purity, 0% misplacement, and 0% hotness loss.
hotness_loss weights errors: placing a frequently accessed page incorrectly
costs more than placing a page touched once.

Example:

    ACCURACY kind=evicted selected_pages=1048576 cold_placement_purity=99.820000% hot_misplacement_rate=0.130000% hotness_loss=0.004700%

## Methodology

- Run performance and accuracy cases separately for precise overhead results.
- Keep seed, pressure, duration, and CPU binding identical across policies.
- Repeat comparisons at least five times.
- The current workload has a stationary binary hot set. Moving hot sets and
  multiple access-frequency bands can be added without changing the metrics.

Run ./hotcold-memory-bench --help for all options.
