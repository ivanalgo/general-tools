# memheat_profiler

`memheat_profiler` 是一个基于 Linux `perf_event_open` 的内存访问热度分析工具。它对内存访问事件进行采样，按页粒度聚合，并将页面划分为 hot、warm、cold 三类。

该工具支持两类后端：

- **PEBS**，面向 Intel 平台
- **IBS**，面向 AMD 平台

二进制名称为 `memheat_profiler`，定义在 `Makefile:5`。

## 工具做了什么

从整体流程看，这个工具会：

1. 自动选择后端，或者按 `--backend` 指定后端。
2. 为目标进程或全系统打开 perf 采样事件。
3. 从 mmap ring buffer 中读取 perf sample。
4. 将每个 sample 解析并映射到某个 page。
5. 累加该 page 的 heat。
6. 按规则将 page 分类为 hot / warm / cold。
7. 输出 detail 和/或 summary 报告。

当前实现中，每接收一个有效 sample，页面 heat 增加 `1.0`，同时支持随时间 cooling。核心逻辑见 `heatmap.c:481` 和 `heatmap.c:504`。

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

主要命令行选项定义在 `main.c:92`。

### 常见示例

默认方式运行 5 秒，分析整机系统范围：

```bash
./memheat_profiler
```

分析指定进程：

```bash
./memheat_profiler --pid 12345
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

- `--pid <pid>`：分析指定进程。
- `--system`：按系统范围分析所有在线 CPU；如果不传，现在默认就是整系统采样。
- `--user-only`：排除内核态 sample。

### 后端选择

- `--backend auto|pebs|ibs`

自动后端选择逻辑在 `backend.c:47`：

- Intel（`GenuineIntel`）优先选 **PEBS**。
- AMD（`AuthenticAMD`）优先选 **IBS**。
- 如果首选后端不可用，则退化到第一个可用后端。

### 采样控制

- `--duration <sec>`：采样时长，默认 `5`
- `--sample-period <n>`：PMU 采样周期，默认 `4000`
- `--mmap-pages <n>`：perf ring 页数，默认 `128`
- `--max-pages <n>`：最多跟踪的页面数，默认 `65536`

### 报告控制

- `--top <n>`：detail 模式中输出前 N 个 page，默认 `20`
- `--process-top <n>`：summary 中输出前 N 个进程，默认 `10`
- `--report-mode detail|summary|both`：默认 `both`
- `--summary-metric pages|heat|samples`：默认 `pages`
- `--output text|json|csv`：默认 `text`
- `--output-file <path>`：输出到文件而不是 stdout

### 地址模式

- `--addr-mode auto|virtual|physical`

行为如下：

- `virtual`：直接使用采样到的虚拟地址。
- `physical`：优先使用物理地址；如果没有，则尝试通过 `/proc/<pid>/pagemap` 做转换。
- `auto`：如果能拿到物理地址则优先使用，否则退回到后端自己的 page 解析逻辑。

地址解析逻辑位于 `heatmap.c:293`。

### Cooling 控制

- `--cooling none|step|exp`：默认 `exp`
- `--cooling-interval-ms <n>`
- `--cooling-decay <f>`：指数衰减因子，默认 `0.80`
- `--cooling-step <f>`：step 模式下每个周期减少多少，默认 `1.0`

cooling 实现在 `heatmap.c:404`。

#### Cooling 机制原理

Cooling 的目的，是让热图更偏向反映**最近一段时间仍然活跃的 page**，避免历史上的热点长期“挂住”高 heat。

整体流程如下：

1. 每次新 sample 到来前，先用 sample 时间戳与上一次 cooling 时间做比较，逻辑在 `heatmap.c:415`。
2. 如果已经跨过一个或多个 cooling 周期，就遍历所有已跟踪 page，并在 `heatmap.c:430` 对旧 heat 做衰减。
3. `step` 模式下，每经过一个周期，就按固定值减少 heat，逻辑在 `heatmap.c:437`。
4. `exp` 模式下，每经过一个周期，就把 heat 乘以 `cooling_decay`，逻辑在 `heatmap.c:440`。
5. 完成 cooling 后，再把当前 sample 贡献的 `+1.0` heat 加到 page 上，逻辑在 `heatmap.c:504`。

因此可以把 heat 更新理解为：

```text
page heat = cooling 之后的旧 heat + 1.0
```

它带来的实际效果是：

- 只在过去短暂变热的 page，会随着时间逐渐降温
- 持续被采样命中的 page，会保持较高 heat
- hot / warm / cold 的结果会更敏感地反映 workload 的阶段性变化

按默认配置看，`exp` 模式配合 `cooling_decay = 0.80`，表示每过一个 cooling 周期，旧 heat 保留 80%，衰减 20%。

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

sample 解析逻辑在 `perf_sampler.c:78`，heat 统计逻辑在 `heatmap.c:481`。

## 热冷判断规则

工具支持两种分类策略：

- **absolute**
- **percentile**

策略解析在 `main.c:75`，分类入口在 `heatmap.c:114`。

### 1. absolute 阈值分类

默认值定义在 `main.c:24` 和 `main.c:25`：

- `hot_threshold = 20.0`
- `cold_threshold = 3.0`

当前 absolute 模式的分类规则是：

- **hot**：`heat >= hot_threshold`
- **cold**：`heat < cold_threshold`
- **warm**：其他情况

对应实现位于 `heatmap.c:28`。

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

percentile 分类逻辑在 `heatmap.c:118`。

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

聚合逻辑在 `heatmap.c:155`。

`summary-metric` 会影响 ratio 列怎么计算：

- `pages`：按页面数占比
- `heat`：按 heat 总和占比
- `samples`：按 sample 数占比

相关辅助函数在 `heatmap.c:69` 和 `heatmap.c:82`。

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

### JSON

结构化输出，方便脚本或后处理使用。

### CSV

紧凑表格格式，适合导入表格工具或管道处理。

## 后端工作原理

## Intel PEBS

**PEBS** 全称是 **Precise Event-Based Sampling**。

高层原理可以理解为：

- CPU 监控某个微架构事件。
- 当采样周期达到时，硬件生成一条更精确的 sample。
- Linux perf 通过 perf ring buffer 将 sample 暴露给用户态。
- 这个工具读取 sample，并把地址聚合到 page 上。

在本项目里，PEBS 后端：

- 要求 CPU vendor 是 `GenuineIntel`，见 `backend_pebs.c:6`
- 要求系统暴露 `cpu` PMU，见 `backend_pebs.c:9`
- 设置 `precise_ip = 2`，见 `backend_pebs.c:49`
- 优先使用 `mem-loads` 事件做采样，见 `backend_pebs.c:59`
- 如果没有 `mem-loads`，则回退到原始事件编码 `event=0xcd,umask=0x1,ldlat=3`，见 `backend_pebs.c:65`
- 请求采样字段包括 IP、TID、time、address、CPU、weight、data source，以及内核支持时的 physical address，见 `backend_pebs.c:52`

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

- 要求 CPU vendor 是 `AuthenticAMD`，见 `backend_ibs.c:6`
- 要求系统暴露 `ibs_op` PMU，见 `backend_ibs.c:9`
- 同样设置 `precise_ip = 2`，见 `backend_ibs.c:49`
- 请求 IP、TID、time、address、CPU、weight、data source，以及可用时的 physical address，见 `backend_ibs.c:52`
- 尝试编码 `ibs_op` 上的 `cycles` 事件；如果编码失败，也不会中止，而是继续使用该 PMU 类型，见 `backend_ibs.c:59`

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

相关错误提示路径在 `perf_sampler.c:266`。

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
