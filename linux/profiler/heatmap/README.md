# memheat_profiler

`memheat_profiler` is a Linux memory-access heatmap profiler built on top of `perf_event_open`. It samples memory access events, aggregates them at page granularity, and reports which pages are hot, warm, or cold.

The tool supports two backend families:

- **PEBS** on Intel systems
- **IBS** on AMD systems

The binary name is `memheat_profiler`.

## What the tool does

At a high level, the profiler:

1. Selects a backend automatically or by `--backend`.
2. Opens perf sampling events for a process or for the whole system.
3. Reads perf samples from the mmap ring buffer.
4. Resolves each sample to a page key.
5. Increments the page heat score.
6. Applies hot/warm/cold classification.
7. Prints detail and/or summary reports.

The current implementation adds `1.0` heat per accepted sample, optionally with cooling over time.

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

The main command-line options are summarized below.

### Common examples

Profile system-wide for 5 seconds:

```bash
./memheat_profiler
```

Profile a specific process:

```bash
./memheat_profiler --pid 12345
```

The same command using short options:

```bash
./memheat_profiler -p 12345 -d 10 -t 50
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

The same idea using short options:

```bash
./memheat_profiler -o json -f report.json -r summary
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

- `-p, --pid <pid>`: profile a specific process and switch from the default system-wide mode to process mode.
- `-s, --system`: profile all online CPUs system-wide. If omitted, the tool now profiles system-wide by default.
- `-u, --user-only`: exclude kernel samples. If omitted, both user-space and kernel-space samples may be included.

Target selection notes:

- Use either `--pid` or `--system` for clarity.
- If both are provided, the last one on the command line takes effect.

### Backend selection

- `-b, --backend auto|pebs|ibs`: default `auto`

Automatic backend selection works as follows:

- Intel (`GenuineIntel`) prefers **PEBS**.
- AMD (`AuthenticAMD`) prefers **IBS**.
- If the preferred backend is unavailable, the tool falls back to the first supported backend.

### Sampling controls

- `-d, --duration <sec>`: profiling duration, default `5`
- `-P, --sample-period <n>`: PMU sample period, default `4000`
- `-m, --mmap-pages <n>`: perf ring pages, default `128`
- `-M, --max-pages <n>`: max tracked pages, default `65536`

### Report controls

- `-t, --top <n>`: top N pages in the detail view, default `20`
- `-T, --process-top <n>`: top N processes in the summary, default `10`
- `-r, --report-mode detail|summary|both`: default `both`
- `-S, --summary-metric pages|heat|samples`: default `pages`
- `-o, --output text|json|csv`: default `text`
- `-f, --output-file <path>`: write output to a file instead of stdout

### Address mode

- `-a, --addr-mode auto|virtual|physical`: default `auto`

Behavior:

- `virtual`: use sampled virtual addresses.
- `physical`: use physical addresses when available, otherwise try `/proc/<pid>/pagemap` translation.
- `auto`: prefer physical addresses when available, otherwise fall back to backend-specific page selection.

In `auto` mode, the profiler tries to use physical addresses when they are available; otherwise it falls back to the backend's normal page-resolution path.

### Heat and classification controls

- `-H, --heat-policy absolute|percentile`: default `absolute`
- `--hot-threshold <f>`: used in `absolute` mode, default `20.0`
- `--cold-threshold <f>`: used in `absolute` mode, default `3.0`
- `--hot-percent <f>`: used in `percentile` mode, default `10.0`
- `--cold-percent <f>`: used in `percentile` mode, default `50.0`

Classification notes:

- In `absolute` mode, `hot_threshold` and `cold_threshold` determine hot/warm/cold.
- In `percentile` mode, `hot_percent` and `cold_percent` determine hot/warm/cold by rank after sorting pages by heat.
- Changing `--summary-metric` does not affect classification; it only changes how summary ratios are computed.

### Cooling controls

- `-c, --cooling none|step|exp`: default `exp`
- `-I, --cooling-interval-ms <n>`: default `500`
- `--cooling-decay <f>`: exponential decay factor, default `0.80`
- `--cooling-step <f>`: decrement per interval in step mode, default `1.0`

#### Cooling principle

Cooling is used to make the heatmap emphasize **recently active pages** instead of letting old hotspots dominate forever.

The implementation follows this sequence:

1. Before a new sample is accounted, the profiler checks how much time has passed since the last cooling pass.
2. If one or more cooling intervals have elapsed, it walks every tracked page and reduces the existing heat.
3. In `step` mode, each elapsed interval subtracts a fixed amount from the page heat.
4. In `exp` mode, each elapsed interval multiplies the page heat by `cooling_decay`.
5. After cooling is applied, the incoming sample contributes `+1.0` heat.

So the effective update rule is:

```text
page heat = cool(previous heat, elapsed time) + 1.0
```

Key values and knobs:

- Cooling mode: `--cooling none|step|exp`, default `exp`
- Cooling interval: `--cooling-interval-ms <n>`, default `500` ms
- Step decrement: `--cooling-step <f>`, default `1.0`
- Exponential decay factor: `--cooling-decay <f>`, default `0.80`

Practical effect:

- pages that were hot only in the past gradually cool down
- pages that continue to receive samples remain hot
- the reported hot/warm/cold state becomes more sensitive to phase changes in workload behavior

Examples:

- `--cooling none`: disable cooling completely; heat only grows as new samples arrive
- `--cooling step --cooling-interval-ms 500 --cooling-step 1.0`: every 500 ms, subtract `1.0` from each tracked page heat, but never below `0`
- `--cooling exp --cooling-interval-ms 500 --cooling-decay 0.80`: every 500 ms, keep 80% of the previous heat and decay 20% away

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

In other words, each accepted sample contributes to exactly one page entry, and the profiler tracks both heat and supporting metadata for that page.

## Hot / warm / cold classification

The profiler supports two classification policies:

- **absolute**
- **percentile**

By default, the tool uses **absolute** classification.

### 1. Absolute thresholds

Default values:

- `hot_threshold = 20.0`
- `cold_threshold = 3.0`

The current absolute classification logic is:

- **hot**: `heat >= hot_threshold`
- **cold**: `heat < cold_threshold`
- **warm**: everything else

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

If a percentile would otherwise round down to zero while the configured percentage is greater than zero, the tool still classifies at least one page into that bucket when pages exist.

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

This aggregation is done over all tracked pages collected during the run.

The summary metric controls how the ratio column is computed:

- `pages`: ratio based on page counts
- `heat`: ratio based on summed heat
- `samples`: ratio based on sample counts

So changing `--summary-metric` changes the ratio basis in the summary, but does not change how pages are classified.

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

Typical `report-mode=both` text output looks like this:

```text
backend=pebs pages=15278 dropped_pages=0 dropped_samples=0 lost_samples=0 report_mode=both summary_metric=pages heat_policy=absolute addr_mode=auto output=text cooling=exp interval_ms=500.00
phys_translate_attempts=49582 phys_translate_failures=22114
summary policy=absolute metric=pages total_pages=15278 total_bytes=62578688 total_heat=1468.80 total_samples=49582
summary thresholds hot>=20.00 cold<3.00
class    pages        bytes              metric_value     ratio
hot      7            28672              7.00             0.05%
warm     48           196608             48.00            0.31%
cold     15223        62353408           15223.00         99.64%

rank   kind      page_base          state   heat    avg_weight owner_pid owner_tid owner_samples last_ip
1      virtual   0xffffa55441abb000 hot     194.00  8.61       4025963   4025963   194           0xffffffffc0bb0e2f
2      virtual   0xffffffffc0bcc000 hot      80.00  7.20       4025963   4025963    80           0xffffffffc0bb0b73
3      physical  0x000000026a369000 hot      20.40  8.49       4069528   4069528   176           0x0000561524408755

rank   pid        heat       pages      samples    hot_pages  warm_pages cold_pages
1      4025963    335.02     16         377        5          3          8
2      4069528     81.44     42         554        1         10         31
```

The text report has three sections when `report-mode=both`:

1. **Run metadata and overall summary**
2. **Top page results**
3. **Top process summary**

Important fields in the metadata and summary section:

- `backend`: selected sampling backend, typically `pebs` or `ibs`
- `pages`: total tracked page entries in the final report
- `dropped_pages`: pages that could not be inserted because the tracking table hit its configured limit
- `dropped_samples`: samples discarded because no valid page key could be produced
- `lost_samples`: perf samples lost by the kernel/perf ring path
- `report_mode`: `detail`, `summary`, or `both`
- `summary_metric`: the basis used for the `ratio` column in the summary table
- `heat_policy`: `absolute` or `percentile`
- `addr_mode`: `virtual`, `physical`, or `auto`
- `cooling`: `none`, `step`, or `exp`
- `interval_ms`: cooling interval in milliseconds
- `phys_translate_attempts`: number of times the profiler tried to obtain a physical page, either directly from the sample or by pagemap fallback
- `phys_translate_failures`: number of those physical-address attempts that failed and had to fall back or be dropped
- `total_pages` / `total_bytes` / `total_heat` / `total_samples`: totals across all tracked pages
- `summary thresholds hot>=... cold<...`: shown in `absolute` mode to describe the classification rule actually used
- `class`: one of `hot`, `warm`, or `cold`
- `metric_value`: the value used to compute `ratio`; it means pages, heat, or samples depending on `summary_metric`
- `ratio`: percentage contribution of each class under the selected `summary_metric`

Important fields in the top page results section:

- `rank`: page order after sorting by descending heat
- `kind`: whether the page key is `virtual` or `physical`
- `page_base`: base address of the page after page alignment
- `state`: page class after applying the selected heat policy
- `heat`: current heat score after cooling has been applied over time
- `avg_weight`: average PMU weight for samples mapped to this page; `0` means the PMU did not provide useful weight data for those samples
- `owner_pid` / `owner_tid`: the process/thread that contributed the most samples to this page
- `owner_samples`: number of samples contributed by that dominant owner
- `last_ip`: instruction pointer of the most recent sample mapped to the page

Text mode keeps the page table compact, so it does not show a separate `samples` column there. If you need per-page `samples` explicitly, use JSON or CSV output.

Important fields in the top process summary section:

- `pid`: process ID ranked by accumulated heat
- `heat`: total heat summed from pages owned by that process in the summary view
- `pages`: number of tracked pages attributed to that process
- `samples`: total samples attributed to that process
- `hot_pages` / `warm_pages` / `cold_pages`: page counts by class for that process

### JSON

Structured output suitable for scripting or post-processing.

JSON contains the same information as text mode, but exposes it as machine-readable objects:

- top-level run metadata such as `backend`, `pages`, `report_mode`, `heat_policy`, `addr_mode`, `cooling`
- a `summary` object with total and class-specific counters
- a `results` array for page-level detail entries
- a `process_results` array for process-level summary entries

Compared with text mode, each page entry in JSON also includes `samples`, which is the total sample count accumulated on that page.

### CSV

Compact tabular output for spreadsheet or pipeline use.

CSV follows the same logical structure as text mode:

- one metadata line
- optional physical-translation statistics line
- summary rows
- a page detail table
- a process summary table

Compared with text mode, the CSV page detail table also includes a `samples` column.

## Backend principles

## Intel PEBS

**PEBS** stands for **Precise Event-Based Sampling**.

High-level idea:

- The CPU monitors a configured microarchitectural event.
- When the sampling period is reached, hardware records a precise sample.
- Linux perf exposes that sample to user space through the perf ring buffer.
- This tool reads the sample and maps it to a memory page.

In this project, the PEBS backend:

- requires vendor `GenuineIntel`
- requires the `cpu` PMU to exist
- requests precise sampling with `precise_ip = 2`
- samples memory-load related information
- falls back to raw event encoding `event=0xcd,umask=0x1,ldlat=3` if `mem-loads` is not available
- requests sample fields including IP, TID, time, address, CPU, weight, data source, and physical address when supported

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

- requires vendor `AuthenticAMD`
- requires the `ibs_op` PMU to exist
- uses `precise_ip = 2`
- requests IP, TID, time, address, CPU, weight, data source, and physical address when available
- tries to encode an `ibs_op` event named `cycles`, but tolerates failure and still uses the PMU type

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

Physical address translation may depend on kernel support and permissions for `/proc/<pid>/pagemap`.

## Notes and limitations

- Heat is sample-based, not direct bandwidth or latency.
- A page becomes hot because it is sampled frequently and still retains high heat after cooling.
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

Specific-process report:

```bash
./memheat_profiler --pid 12345 --output json --output-file process-report.json
```
