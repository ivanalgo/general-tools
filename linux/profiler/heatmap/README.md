# memheat_profiler

`memheat_profiler` is a Linux memory-access heatmap profiler built on top of `perf_event_open`. It samples memory access events, aggregates them at page granularity, and reports which pages are hot, warm, or cold.

The tool supports two backend families:

- **PEBS** on Intel systems
- **IBS** on AMD systems

The binary name is `memheat_profiler` as defined in `Makefile:5`.

## What the tool does

At a high level, the profiler:

1. Selects a backend automatically or by `--backend`.
2. Opens perf sampling events for a process or for the whole system.
3. Reads perf samples from the mmap ring buffer.
4. Resolves each sample to a page key.
5. Increments the page heat score.
6. Applies hot/warm/cold classification.
7. Prints detail and/or summary reports.

The current implementation adds `1.0` heat per accepted sample, optionally with cooling over time. The relevant logic is in `heatmap.c:481` and `heatmap.c:504`.

## Build

```bash
make
```

This builds:

```bash
./memheat_profiler
```

## Basic usage

```bash
./memheat_profiler [options]
```

Main command-line options come from `main.c:92`.

### Common examples

Profile the current process for 5 seconds:

```bash
./memheat_profiler
```

Profile a specific process:

```bash
./memheat_profiler --pid 12345
```

Profile system-wide:

```bash
./memheat_profiler --system
```

Force a backend:

```bash
./memheat_profiler --backend pebs
./memheat_profiler --backend ibs
```

Write JSON output to a file:

```bash
./memheat_profiler --output json --output-file report.json
```

Show summary only:

```bash
./memheat_profiler --report-mode summary
```

Use samples instead of pages as the summary metric:

```bash
./memheat_profiler --summary-metric samples
```

Track more pages and report more entries:

```bash
./memheat_profiler --max-pages 262144 --top 50 --process-top 20
```

## Command-line options

### Target selection

- `--pid <pid>`: profile a specific process. If omitted, the tool profiles itself by default.
- `--system`: profile all online CPUs system-wide.
- `--user-only`: exclude kernel samples.

### Backend selection

- `--backend auto|pebs|ibs`

Automatic backend selection is implemented in `backend.c:47`:

- Intel (`GenuineIntel`) prefers **PEBS**.
- AMD (`AuthenticAMD`) prefers **IBS**.
- If the preferred backend is unavailable, the tool falls back to the first supported backend.

### Sampling controls

- `--duration <sec>`: profiling duration, default `5`
- `--sample-period <n>`: PMU sample period, default `4000`
- `--mmap-pages <n>`: perf ring pages, default `128`
- `--max-pages <n>`: max tracked pages, default `65536`

### Report controls

- `--top <n>`: top N pages in the detail view, default `20`
- `--process-top <n>`: top N processes in the summary, default `10`
- `--report-mode detail|summary|both`: default `both`
- `--summary-metric pages|heat|samples`: default `pages`
- `--output text|json|csv`: default `text`
- `--output-file <path>`: write output to a file instead of stdout

### Address mode

- `--addr-mode auto|virtual|physical`

Behavior:

- `virtual`: use sampled virtual addresses.
- `physical`: use physical addresses when available, otherwise try `/proc/<pid>/pagemap` translation.
- `auto`: prefer physical addresses when available, otherwise fall back to backend-specific page selection.

The resolution logic is implemented in `heatmap.c:293`.

### Cooling controls

- `--cooling none|step|exp`: default `exp`
- `--cooling-interval-ms <n>`
- `--cooling-decay <f>`: exponential decay factor, default `0.80`
- `--cooling-step <f>`: decrement per interval in step mode, default `1.0`

Cooling is implemented in `heatmap.c:404`.

## Heat calculation

Each accepted sample updates exactly one page entry.

Current heat update logic:

```text
page heat = cooled previous heat + 1.0
```

Additional fields are also tracked:

- `samples`: total sample count on the page
- `total_weight`: sum of sampled weights when the PMU provides them
- `last_ip`, `last_data_src`
- owner PID/TID statistics

The sample parsing path is implemented in `perf_sampler.c:78`, and heat accounting is in `heatmap.c:481`.

## Hot / warm / cold classification

The profiler supports two classification policies:

- **absolute**
- **percentile**

The policy parser is in `main.c:75`, and the classifier is in `heatmap.c:114`.

### 1. Absolute thresholds

Default values are defined in `main.c:24` and `main.c:25`:

- `hot_threshold = 20.0`
- `cold_threshold = 3.0`

The current absolute classification logic is:

- **hot**: `heat >= hot_threshold`
- **cold**: `heat < cold_threshold`
- **warm**: everything else

This logic is implemented in `heatmap.c:28`.

With the current defaults, the classes are:

- **hot**: `heat >= 20`
- **cold**: `heat < 3`
- **warm**: `3 <= heat < 20`

Examples:

- `heat = 20.00` → hot
- `heat = 19.99` → warm
- `heat = 3.00` → warm
- `heat = 2.99` → cold

Use custom thresholds like this:

```bash
./memheat_profiler --hot-threshold 20 --cold-threshold 3
```

### 2. Percentile-based classification

If `--heat-policy percentile` is selected, the tool first sorts pages by descending heat and then assigns classes by rank.

Defaults:

- `hot_percent = 10.0`
- `cold_percent = 50.0`

This means:

- top 10% of ranked pages → hot
- bottom 50% of ranked pages → cold
- the middle section → warm

The percentile logic is implemented in `heatmap.c:118`.

Example:

```bash
./memheat_profiler --heat-policy percentile --hot-percent 5 --cold-percent 30
```

## Summary report

The summary aggregates all tracked pages into:

- total pages / bytes / heat / samples
- hot pages / bytes / heat / samples
- warm pages / bytes / heat / samples
- cold pages / bytes / heat / samples

Aggregation happens in `heatmap.c:155`.

The summary metric controls how the ratio column is computed:

- `pages`: ratio based on page counts
- `heat`: ratio based on summed heat
- `samples`: ratio based on sample counts

The summary metric helpers are implemented in `heatmap.c:69` and `heatmap.c:82`.

Example summary-oriented run:

```bash
./memheat_profiler \
  --report-mode summary \
  --summary-metric heat \
  --hot-threshold 20 \
  --cold-threshold 3
```

## Output formats

### Text

Human-readable summary and top pages/processes.

### JSON

Structured output suitable for scripting or post-processing.

### CSV

Compact tabular output for spreadsheet or pipeline use.

## Backend principles

## Intel PEBS

**PEBS** stands for **Precise Event-Based Sampling**.

High-level idea:

- The CPU monitors a configured microarchitectural event.
- When the sampling period is reached, hardware records a precise sample.
- Linux perf exposes that sample to user space through the perf ring buffer.
- This tool reads the sample and maps it to a memory page.

In this project, the PEBS backend:

- requires vendor `GenuineIntel` (`backend_pebs.c:6`)
- requires the `cpu` PMU to exist (`backend_pebs.c:9`)
- requests precise sampling with `precise_ip = 2` (`backend_pebs.c:49`)
- samples memory-load related information (`backend_pebs.c:59`)
- falls back to raw event encoding `event=0xcd,umask=0x1,ldlat=3` if `mem-loads` is not available (`backend_pebs.c:65`)
- requests sample fields including IP, TID, time, address, CPU, weight, data source, and physical address when supported (`backend_pebs.c:52`)

Practical meaning:

- PEBS is used here to obtain relatively precise memory load samples.
- The sampled address is then aggregated into a page heatmap.

## AMD IBS

**IBS** stands for **Instruction-Based Sampling**.

High-level idea:

- AMD hardware periodically captures detailed execution samples at the instruction level.
- These samples can include data-address information for memory operations.
- Linux perf exposes the samples via the `ibs_op` PMU.
- This tool reads those samples and aggregates them by page.

In this project, the IBS backend:

- requires vendor `AuthenticAMD` (`backend_ibs.c:6`)
- requires the `ibs_op` PMU to exist (`backend_ibs.c:9`)
- uses `precise_ip = 2` (`backend_ibs.c:49`)
- requests IP, TID, time, address, CPU, weight, data source, and physical address when available (`backend_ibs.c:52`)
- tries to encode an `ibs_op` event named `cycles`, but tolerates failure and still uses the PMU type (`backend_ibs.c:59`)

Practical meaning:

- IBS provides the low-level sampled execution/memory information used to identify hot pages on AMD systems.

## PEBS vs IBS in this tool

Both backends serve the same purpose here:

- collect precise-enough memory-related samples
- recover an address for each sample
- convert the address to a page key
- accumulate page heat
- classify the page as hot, warm, or cold

The backend choice affects how the samples are obtained from hardware, but the later heatmap pipeline is shared.

## Permissions and runtime requirements

You may need sufficient perf privileges, for example:

- `CAP_PERFMON`
- a permissive `kernel.perf_event_paranoid`

The process-open error path explicitly points this out in `perf_sampler.c:266`.

Physical address translation may depend on kernel support and permissions for `/proc/<pid>/pagemap`.

## Notes and limitations

- Heat is sample-based, not direct bandwidth or latency.
- A page becomes hot because it is sampled frequently, especially after cooling is applied.
- Output quality depends on PMU support, privilege level, sampling period, and workload behavior.
- Physical address availability depends on kernel and PMU support.

## Quick recipes

Absolute thresholds, using the requested rule `hot >= 20` and `cold < 3`:

```bash
./memheat_profiler --hot-threshold 20 --cold-threshold 3
```

Summary only, classified by absolute thresholds:

```bash
./memheat_profiler --report-mode summary --hot-threshold 20 --cold-threshold 3
```

Percentile classification:

```bash
./memheat_profiler --heat-policy percentile --hot-percent 10 --cold-percent 50
```

System-wide JSON report:

```bash
./memheat_profiler --system --output json --output-file system-report.json
```
