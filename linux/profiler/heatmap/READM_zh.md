# memheat_profiler

`memheat_profiler` 是一个基于 Linux `perf_event_open` 的内存访问热度分析工具。它对内存访问事件进行采样，按页粒度聚合，并将页面划分为 hot、warm、cold 三类。

该工具支持两类后端：

- **PEBS**，面向 Intel 平台
- **IBS**，面向 AMD 平台

二进制名称为 `memheat_profiler`。

## 工具做了什么

从整体流程看，这个工具会：

1. 自动选择后端，或者按 `--backend` 指定后端。
2. 为目标进程或全系统打开 perf 采样事件。
3. 从 mmap ring buffer 中读取 perf sample。
4. 将每个 sample 解析并映射到某个 page。
5. 累加该 page 的 heat。
6. 按规则将 page 分类为 hot / warm / cold。
7. 输出 detail 和/或 summary 报告。

当前实现中，每接收一个有效 sample，页面 heat 增加 `1.0`，同时支持随时间 cooling。

## 编译

```bash
make
```

生成的可执行文件为：

```bash
./memheat_profiler
```

## 基本用法

```bash
./memheat_profiler [options]
```

主要命令行选项如下。

### 常见示例

默认方式运行 5 秒，分析整机系统范围：

```bash
./memheat_profiler
```

分析指定进程：

```bash
./memheat_profiler --pid 12345
```

使用短参数的等价写法：

```bash
./memheat_profiler -p 12345 -d 10 -t 50
```

全系统分析：

```bash
./memheat_profiler --system
```

强制指定后端：

```bash
./memheat_profiler --backend pebs
./memheat_profiler --backend ibs
```

将 JSON 输出写入文件：

```bash
./memheat_profiler --output json --output-file report.json
```

使用短参数的等价写法：

```bash
./memheat_profiler -o json -f report.json -r summary
```

只看 summary：

```bash
./memheat_profiler --report-mode summary
```

让 summary 的 ratio 按 samples 计算：

```bash
./memheat_profiler --summary-metric samples
```

增大 page 跟踪上限和输出数量：

```bash
./memheat_profiler --max-pages 262144 --top 50 --process-top 20
```

## 命令行参数说明

### 目标选择

- `-p, --pid <pid>`：分析指定进程，并从默认的整系统模式切换到进程模式。
- `-s, --system`：按系统范围分析所有在线 CPU；如果不传，现在默认就是整系统采样。
- `-u, --user-only`：排除内核态 sample；如果不传，则可能同时包含用户态和内核态 sample。

目标选择补充说明：

- 建议在 `--pid` 和 `--system` 之间二选一，避免歧义。
- 如果两者都传，以命令行中最后出现的那个为准。

### 后端选择

- `-b, --backend auto|pebs|ibs`：默认 `auto`

自动后端选择行为如下：

- Intel（`GenuineIntel`）优先选 **PEBS**。
- AMD（`AuthenticAMD`）优先选 **IBS**。
- 如果首选后端不可用，则退化到第一个可用后端。

### 采样控制

- `-d, --duration <sec>`：采样时长，默认 `5`
- `-P, --sample-period <n>`：PMU 采样周期，默认 `4000`
- `-m, --mmap-pages <n>`：perf ring 页数，默认 `128`
- `-M, --max-pages <n>`：最多跟踪的页面数，默认 `65536`

### 报告控制

- `-t, --top <n>`：detail 模式中输出前 N 个 page，默认 `20`
- `-T, --process-top <n>`：summary 中输出前 N 个进程，默认 `10`
- `-r, --report-mode detail|summary|both`：默认 `both`
- `-S, --summary-metric pages|heat|samples`：默认 `pages`
- `-o, --output text|json|csv`：默认 `text`
- `-f, --output-file <path>`：输出到文件而不是 stdout

### 地址模式

- `-a, --addr-mode auto|virtual|physical`：默认 `auto`

行为如下：

- `virtual`：直接使用采样到的虚拟地址。
- `physical`：优先使用物理地址；如果没有，则尝试通过 `/proc/<pid>/pagemap` 做转换。
- `auto`：如果能拿到物理地址则优先使用，否则退回到后端自己的 page 解析逻辑。

在 `auto` 模式下，如果能拿到物理地址就优先使用；否则退回到后端默认的 page 解析路径。

### Heat 与分类控制

- `-H, --heat-policy absolute|percentile`：默认 `absolute`
- `--hot-threshold <f>`：在 `absolute` 模式下生效，默认 `20.0`
- `--cold-threshold <f>`：在 `absolute` 模式下生效，默认 `3.0`
- `--hot-percent <f>`：在 `percentile` 模式下生效，默认 `10.0`
- `--cold-percent <f>`：在 `percentile` 模式下生效，默认 `50.0`

分类补充说明：

- `absolute` 模式下，hot / warm / cold 由 `hot_threshold` 和 `cold_threshold` 决定。
- `percentile` 模式下，hot / warm / cold 由 page 按 heat 排序后的排名区间决定。
- 修改 `--summary-metric` 不会改变分类结果，只会改变 summary 里的 ratio 计算口径。

### Cooling 控制

- `-c, --cooling none|step|exp`：默认 `exp`
- `-I, --cooling-interval-ms <n>`：默认 `500`
- `--cooling-decay <f>`：指数衰减因子，默认 `0.80`
- `--cooling-step <f>`：step 模式下每个周期减少多少，默认 `1.0`

#### Cooling 机制原理

Cooling 的目的，是让热图更偏向反映**最近一段时间仍然活跃的 page**，避免历史上的热点长期“挂住”高 heat。

整体流程如下：

1. 每次新 sample 到来前，先判断距离上一次 cooling 已经过了多久。
2. 如果已经跨过一个或多个 cooling 周期，就遍历所有已跟踪 page，并对旧 heat 做衰减。
3. `step` 模式下，每经过一个周期，就按固定值减少 heat。
4. `exp` 模式下，每经过一个周期，就把 heat 乘以 `cooling_decay`。
5. 完成 cooling 后，再把当前 sample 贡献的 `+1.0` heat 加到 page 上。

因此可以把 heat 更新理解为：

```text
page heat = cooling 之后的旧 heat + 1.0
```

关键参数和默认值如下：

- Cooling 模式：`--cooling none|step|exp`，默认 `exp`
- Cooling 周期：`--cooling-interval-ms <n>`，默认 `500` 毫秒
- 固定值衰减：`--cooling-step <f>`，默认 `1.0`
- 指数衰减因子：`--cooling-decay <f>`，默认 `0.80`

它带来的实际效果是：

- 只在过去短暂变热的 page，会随着时间逐渐降温
- 持续被采样命中的 page，会保持较高 heat
- hot / warm / cold 的结果会更敏感地反映 workload 的阶段性变化

例如：

- `--cooling none`：完全关闭 cooling，heat 只会随着新 sample 持续累加
- `--cooling step --cooling-interval-ms 500 --cooling-step 1.0`：每 500 毫秒，对每个已跟踪 page 的 heat 固定减 `1.0`，最低不会小于 `0`
- `--cooling exp --cooling-interval-ms 500 --cooling-decay 0.80`：每 500 毫秒，把旧 heat 保留 80%，衰减 20%

## Heat 是怎么计算的

每个有效 sample 只会更新一个 page 条目。

当前 heat 更新公式可以理解为：

```text
page heat = cooling 之后的旧 heat + 1.0
```

此外还会额外记录：

- `samples`：这个 page 上累计的 sample 数
- `total_weight`：如果 PMU 提供 weight，则累计 weight
- `last_ip`、`last_data_src`
- owner PID/TID 统计信息

换句话说，每个有效 sample 只会落到一个 page 条目上，同时工具会为该 page 记录 heat 以及相关辅助统计信息。

## 热冷判断规则

工具支持两种分类策略：

- **absolute**
- **percentile**

默认使用 **absolute** 分类。

### 1. absolute 阈值分类

默认值如下：

- `hot_threshold = 20.0`
- `cold_threshold = 3.0`

当前 absolute 模式的分类规则是：

- **hot**：`heat >= hot_threshold`
- **cold**：`heat < cold_threshold`
- **warm**：其他情况

按当前默认值展开后就是：

- **hot**：`heat >= 20`
- **cold**：`heat < 3`
- **warm**：`3 <= heat < 20`

例子：

- `heat = 20.00` → hot
- `heat = 19.99` → warm
- `heat = 3.00` → warm
- `heat = 2.99` → cold

自定义阈值可以这样传：

```bash
./memheat_profiler --hot-threshold 20 --cold-threshold 3
```

### 2. percentile 百分位分类

如果选择 `--heat-policy percentile`，工具会先按 heat 从高到低排序，再按排名分配 hot/cold。

默认值：

- `hot_percent = 10.0`
- `cold_percent = 50.0`

含义是：

- heat 排名前 10% 的 page → hot
- heat 排名后 50% 的 page → cold
- 中间部分 → warm

如果百分比对应的 page 数在四舍五入后会变成 0，但配置值本身又大于 0，那么只要存在 page，工具仍然会至少分出 1 个 page 到该桶里。

示例：

```bash
./memheat_profiler --heat-policy percentile --hot-percent 5 --cold-percent 30
```

## Summary 报告说明

summary 会把所有 page 聚合成：

- total pages / bytes / heat / samples
- hot pages / bytes / heat / samples
- warm pages / bytes / heat / samples
- cold pages / bytes / heat / samples

这些统计都是基于本次运行中跟踪到的全部 page 做聚合。

`summary-metric` 会影响 ratio 列怎么计算：

- `pages`：按页面数占比
- `heat`：按 heat 总和占比
- `samples`：按 sample 数占比

因此，修改 `--summary-metric` 只会改变 summary 里的 ratio 计算口径，不会改变 hot / warm / cold 的分类结果。

一个典型的 summary 示例：

```bash
./memheat_profiler \
  --report-mode summary \
  --summary-metric heat \
  --hot-threshold 20 \
  --cold-threshold 3
```

## 输出格式

### Text

适合人工阅读，包含 summary 和 top pages/processes。

`report-mode=both` 时，典型文本输出大致如下：

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

在 `report-mode=both` 下，文本输出通常分成三块：

1. **运行元信息与整体 summary**
2. **page 级 detail 排名**
3. **process 级汇总排名**

第一块里建议重点关注这些字段：

- `backend`：实际使用的采样后端，通常是 `pebs` 或 `ibs`
- `pages`：最终报告中跟踪到的 page 条目总数
- `dropped_pages`：由于 page 跟踪表达到上限而无法插入的 page 数量
- `dropped_samples`：因为无法得到有效 page key 而被丢弃的 sample 数量
- `lost_samples`：内核/perf ring 路径里丢失的 sample 数量
- `report_mode`：`detail`、`summary` 或 `both`
- `summary_metric`：summary 表中 `ratio` 的计算口径
- `heat_policy`：`absolute` 或 `percentile`
- `addr_mode`：`virtual`、`physical` 或 `auto`
- `cooling`：`none`、`step` 或 `exp`
- `interval_ms`：cooling 周期，单位毫秒
- `phys_translate_attempts`：尝试获得 physical page 的次数，包括 sample 直接给出 physical address，以及通过 pagemap 回退翻译的情况
- `phys_translate_failures`：上述 physical-address 尝试失败的次数；失败后会回退成 virtual 或直接丢弃
- `total_pages` / `total_bytes` / `total_heat` / `total_samples`：本次运行所有跟踪 page 的总体统计
- `summary thresholds hot>=... cold<...`：`absolute` 模式下实际使用的分类阈值
- `class`：`hot`、`warm`、`cold` 三种类别
- `metric_value`：用于计算 `ratio` 的值；它的含义取决于 `summary_metric`，可能是 pages、heat 或 samples
- `ratio`：在当前 `summary_metric` 口径下，该类别占总量的百分比

第二块 page 级 detail 里，关键字段含义如下：

- `rank`：按 heat 从高到低排序后的名次
- `kind`：当前 page key 是 `virtual` 还是 `physical`
- `page_base`：按页对齐后的起始地址
- `state`：按当前 heat policy 计算出的 page 分类结果
- `heat`：这个 page 当前的 heat 分数，已经包含 cooling 的影响
- `avg_weight`：落到该 page 上的 sample 的平均 PMU weight；如果是 `0`，通常表示 PMU 没有提供有意义的 weight
- `owner_pid` / `owner_tid`：对该 page 贡献 sample 最多的进程/线程
- `owner_samples`：这个主导 owner 在该 page 上贡献的 sample 数量
- `last_ip`：最近一次命中该 page 的 sample 对应的指令地址

为了让 text 输出更紧凑，text 模式的 page 明细表没有单独展示 `samples` 列；如果你需要逐页查看总 sample 数，建议使用 JSON 或 CSV 输出。

第三块 process 汇总里，关键字段含义如下：

- `pid`：按累计 heat 排名的进程 ID
- `heat`：该进程在 summary 视角下累计到的总 heat
- `pages`：归属到该进程的 page 数量
- `samples`：归属到该进程的 sample 数量
- `hot_pages` / `warm_pages` / `cold_pages`：该进程名下 page 在三种分类中的数量

### JSON

结构化输出，方便脚本或后处理使用。

JSON 和 text 模式承载的是同一批信息，只是改成了机器更容易处理的对象结构：

- 顶层运行元信息，例如 `backend`、`pages`、`report_mode`、`heat_policy`、`addr_mode`、`cooling`
- `summary` 对象，包含总体统计和 hot/warm/cold 三类统计
- `results` 数组，对应 page 级明细
- `process_results` 数组，对应 process 级汇总

相比 text 模式，JSON 的每条 page 明细里还包含 `samples` 字段，表示这个 page 累积到的总 sample 数。

### CSV

紧凑表格格式，适合导入表格工具或管道处理。

CSV 在逻辑上与 text 模式一致，通常依次包含：

- 一行运行元信息
- 一行可选的 physical-address 翻译统计
- summary 表
- page 级明细表
- process 级汇总表

相比 text 模式，CSV 的 page 明细表也额外包含 `samples` 列。

## 后端工作原理

## Intel PEBS

**PEBS** 全称是 **Precise Event-Based Sampling**。

高层原理可以理解为：

- CPU 监控某个微架构事件。
- 当采样周期达到时，硬件生成一条更精确的 sample。
- Linux perf 通过 perf ring buffer 将 sample 暴露给用户态。
- 这个工具读取 sample，并把地址聚合到 page 上。

在本项目里，PEBS 后端：

- 要求 CPU vendor 是 `GenuineIntel`
- 要求系统暴露 `cpu` PMU
- 设置 `precise_ip = 2`
- 优先使用 `mem-loads` 事件做采样
- 如果没有 `mem-loads`，则回退到原始事件编码 `event=0xcd,umask=0x1,ldlat=3`
- 请求采样字段包括 IP、TID、time、address、CPU、weight、data source，以及内核支持时的 physical address

实际意义：

- 这里的 PEBS 用来获取相对精确的内存 load sample。
- 拿到 sample 地址后，再聚合成 page heatmap。

## AMD IBS

**IBS** 全称是 **Instruction-Based Sampling**。

高层原理可以理解为：

- AMD 硬件按一定周期抓取指令级采样信息。
- 对于内存相关操作，sample 中可以带出数据访问地址等信息。
- Linux perf 通过 `ibs_op` PMU 暴露这些 sample。
- 这个工具读取 sample 并按 page 聚合。

在本项目里，IBS 后端：

- 要求 CPU vendor 是 `AuthenticAMD`
- 要求系统暴露 `ibs_op` PMU
- 同样设置 `precise_ip = 2`
- 请求 IP、TID、time、address、CPU、weight、data source，以及可用时的 physical address
- 尝试编码 `ibs_op` 上的 `cycles` 事件；如果编码失败，也不会中止，而是继续使用该 PMU 类型

实际意义：

- IBS 提供 AMD 平台上底层执行/访存采样能力，供本工具识别 hot page。

## 在本工具里 PEBS 和 IBS 的关系

在这个项目里，两种后端的目标是一致的：

- 获取足够精确的内存相关 sample
- 为每个 sample 恢复一个地址
- 将地址映射到 page key
- 累加 page heat
- 按规则划分 hot / warm / cold

后端差异主要体现在“硬件如何产出 sample”，而后面的热度聚合和报告逻辑是共享的。

## 权限与运行要求

你可能需要足够的 perf 权限，例如：

- `CAP_PERFMON`
- 较宽松的 `kernel.perf_event_paranoid`

如果使用物理地址或 pagemap 转换，还依赖内核能力以及对 `/proc/<pid>/pagemap` 的访问权限。

## 注意事项与限制

- heat 反映的是采样热度，不是直接的带宽值或延迟值。
- 一个 page 之所以变 hot，本质上是因为它被频繁采样，且在 cooling 后依然保持较高 heat。
- 输出质量受 PMU 支持、权限、采样周期以及 workload 行为影响。
- physical address 能否获取取决于 PMU 和内核支持情况。

## 快速命令示例

使用你要求的 absolute 规则：`hot >= 20`、`cold < 3`：

```bash
./memheat_profiler --hot-threshold 20 --cold-threshold 3
```

只输出 summary，并按 absolute 阈值分类：

```bash
./memheat_profiler --report-mode summary --hot-threshold 20 --cold-threshold 3
```

按百分位分类：

```bash
./memheat_profiler --heat-policy percentile --hot-percent 10 --cold-percent 50
```

输出系统级 JSON 报告：

```bash
./memheat_profiler --system --output json --output-file system-report.json
```

输出指定进程的 JSON 报告：

```bash
./memheat_profiler --pid 12345 --output json --output-file process-report.json
```
