# 舍入方案原理、示例与评估程序使用说明 / Rounding Schemes Guide and Benchmark Usage

本文介绍常见舍入方案的原理、优缺点、示例，以及本目录下 `rounding_scheme_benchmark.py` 的使用方法。

This guide explains common rounding schemes, their principles, trade-offs, examples, and how to use `rounding_scheme_benchmark.py` in this directory.

---

## 1. 背景：什么是舍入？ / Background: What Is Rounding?

舍入（rounding）是把一个连续或高精度数值映射到某个离散网格上的过程。例如，把实数舍入到整数，或者把金额舍入到分、角、元。

Rounding maps a continuous or high-precision value onto a discrete grid. For example, rounding a real number to an integer, or rounding money to cents, tenths, or whole currency units.

如果舍入粒度为 `q`，则所有可表示值都是：

If the rounding quantum is `q`, all representable values are:

```text
..., -2q, -q, 0, q, 2q, ...
```

例如 `q = 1` 表示舍入到整数，`q = 0.01` 表示舍入到两位小数。

For example, `q = 1` means rounding to integers; `q = 0.01` means rounding to two decimal places.

舍入的关键问题是：当一个数落在两个可表示值之间时，应该选择哪一个？尤其是当它恰好在中点时，例如 `2.5` 位于 `2` 和 `3` 正中间，不同方案会给出不同答案。

The key question is: when a value lies between two representable values, which one should be chosen? This is especially important for ties, such as `2.5`, which is exactly halfway between `2` and `3`.

---

## 2. 术语 / Terminology

### 2.1 量化粒度 / Quantum

量化粒度 `q` 是舍入目标网格的间距。

The quantum `q` is the spacing of the target rounding grid.

示例 / Example:

```text
q = 1      -> ..., 1, 2, 3, ...
q = 0.1    -> ..., 1.1, 1.2, 1.3, ...
q = 0.01   -> ..., 1.23, 1.24, 1.25, ...
```

### 2.2 舍入误差 / Rounding Error

设原始值为 `x`，舍入后为 `r(x)`，则舍入误差为：

Let the original value be `x` and the rounded value be `r(x)`. The rounding error is:

```text
error = r(x) - x
```

如果误差长期为正，说明舍入结果系统性偏大；如果长期为负，说明舍入结果系统性偏小。

If the error is positive on average, rounded values are systematically too large. If it is negative on average, rounded values are systematically too small.

### 2.3 tie / 正中间值

当一个数恰好落在两个网格点中间时，称为 tie。例如舍入到整数时：

A tie occurs when a value is exactly halfway between two grid points. For integer rounding:

```text
1.5 is halfway between 1 and 2
2.5 is halfway between 2 and 3
-1.5 is halfway between -2 and -1
```

很多舍入规则的差异只在 tie 上体现；对于连续随机分布，恰好出现 tie 的概率通常接近 0，所以这些规则在连续分布上看起来会非常相似。

Many rounding rules differ only on ties. For continuous random distributions, exact ties are rare, so these rules often look almost identical.

---

## 3. 常见舍入方案 / Common Rounding Schemes

下面示例默认舍入到整数，即 `q = 1`。

The examples below assume integer rounding, i.e. `q = 1`.

### 3.1 向下取整：`floor` / Round Toward Negative Infinity

**规则 / Rule**：总是选择不大于 `x` 的最大整数。

Always choose the greatest integer less than or equal to `x`.

```text
floor( 2.1) =  2
floor( 2.9) =  2
floor(-2.1) = -3
floor(-2.9) = -3
```

**特点 / Characteristics**：

- 对正数总是向小方向偏。
- 对负数会向更负方向偏。
- 在非负数据上，平均误差通常接近 `-q/2`。

- For positive numbers, it always rounds downward.
- For negative numbers, it rounds toward more negative values.
- For non-negative data, the mean error is often close to `-q/2`.

**适用场景 / Use cases**：

- 需要保守下界，例如“最多不超过”的安全估计。
- 离散桶编号，例如把时间戳映射到过去的时间窗口。

- Conservative lower bounds.
- Bucket indexing, such as mapping timestamps to previous time windows.

### 3.2 向上取整：`ceiling` / Round Toward Positive Infinity

**规则 / Rule**：总是选择不小于 `x` 的最小整数。

Always choose the smallest integer greater than or equal to `x`.

```text
ceil( 2.1) =  3
ceil( 2.9) =  3
ceil(-2.1) = -2
ceil(-2.9) = -2
```

**特点 / Characteristics**：

- 对正数总是向大方向偏。
- 在非负数据上，平均误差通常接近 `+q/2`。

- For positive numbers, it always rounds upward.
- For non-negative data, the mean error is often close to `+q/2`.

**适用场景 / Use cases**：

- 资源容量估计，例如需要至少多少页、多少块、多少容器。
- 费用、配额、分页等不能低估的场景。

- Capacity estimation, such as required pages, blocks, or containers.
- Billing, quota, and pagination scenarios where underestimation is unacceptable.

### 3.3 朝 0 取整：`toward_zero` / Round Toward Zero

**规则 / Rule**：直接截断小数部分。

Drop the fractional part.

```text
toward_zero( 2.9) =  2
toward_zero( 2.1) =  2
toward_zero(-2.1) = -2
toward_zero(-2.9) = -2
```

**特点 / Characteristics**：

- 正数方向类似 `floor`。
- 负数方向类似 `ceiling`。
- 如果数据正负对称，正负偏差可能部分抵消。

- For positive numbers, it behaves like `floor`.
- For negative numbers, it behaves like `ceiling`.
- For symmetric data, positive and negative biases may partly cancel.

**适用场景 / Use cases**：

- 很多编程语言中浮点转整数的默认语义。
- 只关心整数部分，不希望绝对值增大。

- Default float-to-integer conversion in many programming languages.
- Situations where only the integer part matters and magnitude should not increase.

### 3.4 远离 0 取整：`away_from_zero` / Round Away From Zero

**规则 / Rule**：只要不是整数，就向绝对值更大的方向取整。

If the value is not already an integer, round toward larger magnitude.

```text
away_from_zero( 2.1) =  3
away_from_zero( 2.9) =  3
away_from_zero(-2.1) = -3
away_from_zero(-2.9) = -3
```

**特点 / Characteristics**：

- 会系统性增大绝对值。
- 在正负对称数据上，均值偏差可能小，但绝对误差和方差通常较大。

- It systematically increases magnitudes.
- For symmetric data, mean bias may be small, but absolute error and variance are often larger.

**适用场景 / Use cases**：

- 需要保守地扩大边界或误差范围。
- 对安全裕量有要求的场景。

- Conservative expansion of bounds or error margins.
- Safety-margin-sensitive applications.

### 3.5 最近取整，tie 向 `+∞`：`half_up` / Round Half Up Toward Positive Infinity

**规则 / Rule**：选择最近的整数；如果恰好 `.5`，向 `+∞` 方向取。

Choose the nearest integer; if exactly halfway, round toward positive infinity.

```text
half_up( 2.4) =  2
half_up( 2.5) =  3
half_up( 2.6) =  3
half_up(-2.4) = -2
half_up(-2.5) = -2
half_up(-2.6) = -3
```

**中文语境说明 / Chinese-context note**：

日常口语里常说“五舍五入”，经常指“遇到 5 往上进”。但这里的“上”如果严格理解为数轴上的 `+∞`，则 `-2.5` 会变成 `-2`；如果理解为“绝对值进位”，则 `-2.5` 会变成 `-3`，那就是下面的 `half_away_from_zero`。

In everyday Chinese, people often say “五舍五入” to mean “5 rounds up”. If “up” means toward `+∞`, then `-2.5` becomes `-2`. If “up” means increasing magnitude, then `-2.5` becomes `-3`, which is `half_away_from_zero`.

**特点 / Characteristics**：

- 对全部为正、且 tie 很多的数据，会产生正偏差。
- 对连续分布，tie 很少，偏差通常不明显。

- For positive tie-heavy data, it creates positive bias.
- For continuous distributions, ties are rare, so the bias is often small.

### 3.6 最近取整，tie 向 `-∞`：`half_down` / Round Half Down Toward Negative Infinity

**规则 / Rule**：选择最近的整数；如果恰好 `.5`，向 `-∞` 方向取。

Choose the nearest integer; if exactly halfway, round toward negative infinity.

```text
half_down( 2.5) =  2
half_down(-2.5) = -3
```

**特点 / Characteristics**：

- 与 `half_up` 相反，tie 多时可能产生负偏差。
- 在实际业务中较少作为默认规则使用。

- It is the opposite of `half_up`; tie-heavy data may produce negative bias.
- Less commonly used as a default business rule.

### 3.7 最近取整，tie 远离 0：`half_away_from_zero` / Round Half Away From Zero

**规则 / Rule**：选择最近的整数；如果恰好 `.5`，选择绝对值更大的那个。

Choose the nearest integer; if exactly halfway, choose the value with larger magnitude.

```text
half_away_from_zero( 2.5) =  3
half_away_from_zero(-2.5) = -3
```

**特点 / Characteristics**：

- 很多商业、财务、用户直觉中的“四舍五入”是这个规则。
- 对正数而言，它和 `half_up` 一样。
- 对正负对称数据，tie 的正负偏差可能抵消。

- Often matches business or user intuition for “round half up”.
- For positive values, it is identical to `half_up`.
- For symmetric positive/negative tie data, biases may cancel.

**注意 / Note**：

如果数据几乎都是正数，并且存在大量 `.5`，它仍然会向上偏。

If data is mostly positive and contains many `.5` ties, it still has upward bias.

### 3.8 最近取整，tie 朝 0：`half_toward_zero` / Round Half Toward Zero

**规则 / Rule**：选择最近的整数；如果恰好 `.5`，选择绝对值更小的那个。

Choose the nearest integer; if exactly halfway, choose the value with smaller magnitude.

```text
half_toward_zero( 2.5) =  2
half_toward_zero(-2.5) = -2
```

**特点 / Characteristics**：

- 与 `half_away_from_zero` 相反。
- 对正数 tie 偏小，对负数 tie 偏大。

- Opposite of `half_away_from_zero`.
- Positive ties round downward; negative ties round upward.

### 3.9 最近取整，tie 到偶数：`half_even` / Round Half To Even

**规则 / Rule**：选择最近的整数；如果恰好 `.5`，选择偶数。

Choose the nearest integer; if exactly halfway, choose the even integer.

```text
half_even(1.5) = 2
half_even(2.5) = 2
half_even(3.5) = 4
half_even(4.5) = 4
```

**别名 / Aliases**：

- 银行家舍入 / Banker's rounding
- 统计学舍入 / Statistician's rounding
- IEEE 754 默认最近舍入思想常用 tie 策略 / Common tie-breaking strategy in IEEE 754 round-to-nearest mode

**为什么常用 / Why it is common**：

如果大量 tie 均匀分布在奇偶整数之间，`half_even` 不会总是向上或向下，因此长期偏差通常更小。

If many ties are evenly distributed between odd and even integers, `half_even` does not always round up or down, so long-term bias is often smaller.

**示例 / Example**：

```text
Data: 1.5, 2.5, 3.5, 4.5

half_up result:   2, 3, 4, 5  -> sum = 14
half_even result: 2, 2, 4, 4  -> sum = 12
original sum: 12
```

在这个例子里，`half_even` 完全保留了总和，而 `half_up` 产生了系统性正偏差。

In this example, `half_even` preserves the sum exactly, while `half_up` creates systematic positive bias.

### 3.10 最近取整，tie 到奇数：`half_odd` / Round Half To Odd

**规则 / Rule**：选择最近的整数；如果恰好 `.5`，选择奇数。

Choose the nearest integer; if exactly halfway, choose the odd integer.

```text
half_odd(1.5) = 1
half_odd(2.5) = 3
half_odd(3.5) = 3
half_odd(4.5) = 5
```

**特点 / Characteristics**：

- 和 `half_even` 类似，也可以避免总是向同一方向偏。
- 但在通用科学计算和金融系统中不如 `half_even` 常见。

- Similar to `half_even`, it avoids always rounding in one direction.
- Less common than `half_even` in scientific computing and financial systems.

### 3.11 最近取整，tie 随机：`random_tie` / Random Tie Breaking

**规则 / Rule**：非 tie 时选择最近整数；恰好 `.5` 时随机向上或向下。

Choose the nearest integer; if exactly halfway, randomly choose one of the two neighbors.

```text
random_tie(2.5) -> 2 with probability 0.5, 3 with probability 0.5
```

**特点 / Characteristics**：

- 对 tie 的期望值无偏。
- 只在 tie 上引入随机性。
- 可复现性依赖随机种子。

- Unbiased in expectation for ties.
- Adds randomness only at ties.
- Reproducibility depends on the random seed.

### 3.12 随机舍入：`stochastic` / Stochastic Rounding

**规则 / Rule**：如果 `x` 位于两个相邻网格点 `a` 和 `b` 之间，则按距离决定概率：

If `x` lies between neighboring grid points `a` and `b`, choose probabilistically according to distance:

```text
P(round to b) = (x - a) / (b - a)
P(round to a) = (b - x) / (b - a)
```

例如 `q = 1`：

For `q = 1`:

```text
x = 2.1 -> 10% probability to 3, 90% probability to 2
x = 2.7 -> 70% probability to 3, 30% probability to 2
x = 2.5 -> 50% probability to 3, 50% probability to 2
```

**核心性质 / Key property**：

随机舍入在数学期望上是无偏的：

Stochastic rounding is unbiased in expectation:

```text
E[round(x)] = x
```

以 `x = 2.7` 为例：

For `x = 2.7`:

```text
E[round(2.7)] = 0.3 * 2 + 0.7 * 3 = 2.7
```

**特点 / Characteristics**：

- 平均偏差通常很小。
- 误差方差通常比确定性最近取整更大。
- 在低精度数值计算、机器学习训练、迭代算法中有研究价值。
- 输出存在随机性，需要固定 seed 才方便复现实验。

- Mean bias is often small.
- Error variance is usually larger than deterministic nearest rounding.
- Useful in low-precision numerical computing, machine learning training, and iterative algorithms.
- Results are random; a fixed seed is needed for reproducibility.

---

## 4. 示例对比表 / Example Comparison Table

假设舍入到整数，以下是几个代表值的结果：

Assume integer rounding. The following table compares representative values:

| scheme | 2.4 | 2.5 | 2.6 | -2.4 | -2.5 | -2.6 |
|---|---:|---:|---:|---:|---:|---:|
| floor | 2 | 2 | 2 | -3 | -3 | -3 |
| ceiling | 3 | 3 | 3 | -2 | -2 | -2 |
| toward_zero | 2 | 2 | 2 | -2 | -2 | -2 |
| away_from_zero | 3 | 3 | 3 | -3 | -3 | -3 |
| half_up | 2 | 3 | 3 | -2 | -2 | -3 |
| half_down | 2 | 2 | 3 | -2 | -3 | -3 |
| half_away_from_zero | 2 | 3 | 3 | -2 | -3 | -3 |
| half_toward_zero | 2 | 2 | 3 | -2 | -2 | -3 |
| half_even | 2 | 2 | 3 | -2 | -2 | -3 |
| half_odd | 2 | 3 | 3 | -2 | -3 | -3 |
| random_tie | 2 | 2 or 3 | 3 | -2 | -3 or -2 | -3 |
| stochastic | random | random | random | random | random | random |

注意：`stochastic` 对非 tie 也随机，例如 `2.4` 有 40% 概率到 `3`、60% 概率到 `2`。

Note: `stochastic` is random even for non-ties. For example, `2.4` has a 40% probability of rounding to `3` and a 60% probability of rounding to `2`.

---

## 5. 如何评价一个舍入方案？ / How to Evaluate a Rounding Scheme?

不同应用关心的指标不同。下面是程序中使用的指标。

Different applications care about different metrics. The benchmark program reports the following metrics.

### 5.1 平均误差：`mean_error` / Mean Error, Bias

```text
mean_error = average(round(x) - x)
```

它衡量长期系统性偏差。如果接近 0，说明整体上没有明显向上或向下偏。

This measures long-term systematic bias. A value near zero means there is no obvious upward or downward drift.

### 5.2 ULP 归一化平均误差：`mean_error_in_ulps` / Mean Error in ULPs

```text
mean_error_in_ulps = mean_error / quantum
```

这里的 ULP 可以近似理解为一个舍入单位。它方便比较不同 `quantum` 下的偏差。

Here ULP roughly means one rounding unit. It makes bias comparable across different `quantum` values.

### 5.3 误差方差：`error_variance` / Error Variance

误差方差衡量舍入噪声的波动程度。随机舍入通常 bias 小，但 variance 更大。

Error variance measures the spread of rounding noise. Stochastic rounding usually has low bias but higher variance.

### 5.4 平均绝对误差：`mean_abs_error` / Mean Absolute Error

```text
mean_abs_error = average(abs(round(x) - x))
```

它衡量典型误差大小，不区分正负方向。

This measures typical error magnitude without considering sign.

### 5.5 均方根误差：`rmse` / Root Mean Squared Error

```text
rmse = sqrt(average(error^2))
```

RMSE 对较大误差更敏感。

RMSE is more sensitive to large errors.

### 5.6 最大绝对误差：`max_abs_error` / Maximum Absolute Error

它表示最坏情况下误差有多大。

This measures the worst observed error.

### 5.7 95 分位绝对误差：`p95_abs_error` / 95th Percentile Absolute Error

它表示 95% 样本的误差不超过这个值，比最大值更不容易被极端样本支配。

This says 95% of samples have absolute error no larger than this value. It is less dominated by outliers than the maximum.

### 5.8 平均相对绝对误差：`mean_relative_abs_error` / Mean Relative Absolute Error

```text
mean_relative_abs_error = average(abs(error) / max(abs(x), quantum))
```

程序用 `max(abs(x), quantum)` 避免 `x` 接近 0 时相对误差爆炸。

The program uses `max(abs(x), quantum)` to avoid exploding relative errors when `x` is close to zero.

### 5.9 总和误差：`sum_error` / Sum Error

```text
sum_error = sum(round(x)) - sum(x)
```

这对聚合计算非常重要。例如财务总账、统计报表、科学模拟中的守恒量。

This is important for aggregation, such as financial ledgers, statistical reports, or conserved quantities in simulations.

### 5.10 总和相对误差：`sum_relative_error` / Relative Sum Error

```text
sum_relative_error = sum_error / sum(x)
```

它把总和误差归一化，便于比较不同规模的数据集。

It normalizes the sum error, making datasets of different scales easier to compare.

---

## 6. 程序说明 / Program Overview

程序文件：

Program file:

```text
math/rounding_scheme_benchmark.py
```

它会：

It will:

1. 生成多种模拟数据分布。
2. 对每种分布应用多个舍入方案。
3. 计算误差指标。
4. 输出对齐后的表格。
5. 可选导出 CSV。

1. Generate multiple synthetic data distributions.
2. Apply multiple rounding schemes to each distribution.
3. Compute error metrics.
4. Print aligned tables.
5. Optionally export CSV.

### 6.1 支持的舍入方案 / Supported Schemes

```text
floor
ceiling
toward_zero
away_from_zero
half_up
half_down
half_away_from_zero
half_toward_zero
half_even
half_odd
random_tie
stochastic
```

### 6.2 支持的数据分布 / Supported Distributions

```text
uniform_positive   U(0, 100), positive uniform distribution
uniform_centered   U(-50, 50), symmetric uniform distribution
normal             N(0, 20^2), symmetric normal distribution
lognormal          right-skewed positive distribution
exponential        right-skewed positive distribution
beta_skewed        bounded skewed distribution, 100 * Beta(2, 8)
half_ties          artificial tie-heavy distribution, values like k + 0.5
retail_prices      toy retail-price distribution, many .99 endings
```

---

## 7. 如何运行 / How to Run

在仓库根目录运行：

Run from the repository root:

```bash
python3 math/rounding_scheme_benchmark.py
```

指定样本数量：

Specify sample count:

```bash
python3 math/rounding_scheme_benchmark.py --n 200000
```

指定舍入粒度，例如舍入到两位小数：

Specify quantum, for example rounding to two decimal places:

```bash
python3 math/rounding_scheme_benchmark.py --quantum 0.01
```

只运行某几个分布：

Run selected distributions only:

```bash
python3 math/rounding_scheme_benchmark.py \
  --distributions normal,half_ties,retail_prices
```

只比较某几个舍入方案：

Compare selected rounding schemes only:

```bash
python3 math/rounding_scheme_benchmark.py \
  --schemes half_up,half_away_from_zero,half_even,stochastic
```

按某个指标排序：

Sort by a metric:

```bash
python3 math/rounding_scheme_benchmark.py --sort-by mean_error
```

默认是升序。如果希望大的值排前面：

Ascending order is used by default. To sort descending:

```bash
python3 math/rounding_scheme_benchmark.py --sort-by sum_error --descending
```

只显示每个分布下前 N 个方案：

Show only the first N schemes per distribution:

```bash
python3 math/rounding_scheme_benchmark.py --top 5
```

导出 CSV：

Export CSV:

```bash
python3 math/rounding_scheme_benchmark.py --csv rounding_results.csv
```

固定随机种子以便复现：

Fix the random seed for reproducibility:

```bash
python3 math/rounding_scheme_benchmark.py --seed 12345
```

---

## 8. 输出如何解读 / How to Read the Output

输出开头类似：

The output starts like this:

```text
Rounding scheme benchmark
sample_count=50000, quantum=1.0, seed=20260710
Lower mean_error/mean_error_in_ulps indicates lower bias; lower rmse/mae/variance indicates lower noise.
```

含义：

Meaning:

- `sample_count`：每个分布生成多少个样本。
- `quantum`：舍入粒度。
- `seed`：随机种子。

- `sample_count`: number of samples per distribution.
- `quantum`: rounding quantum.
- `seed`: random seed.

每个分布会先显示原始数据统计：

Each distribution first shows input data metrics:

```text
Input data metrics: input_mean=..., input_variance=..., input_stddev=..., input_sum=...
```

这些指标描述的是“原始数据本身”，不是舍入误差。

These metrics describe the original input data itself, not rounding error.

然后表格第一行是：

Then the first table row is:

```text
original/no_round
```

这表示“不做舍入”的误差基线，所以误差指标应该都是 0。

This is the no-rounding baseline, so error metrics should be zero.

后续每一行是一个舍入方案。常见阅读方式：

Each following row is a rounding scheme. Common ways to read it:

- 看 `mean_error`：判断是否系统性偏大或偏小。
- 看 `rmse` / `mean_abs_error`：判断单个值的典型误差。
- 看 `error_variance`：判断舍入噪声波动。
- 看 `sum_error`：判断大量数据聚合后的总漂移。
- 看 `half_ties` 分布：专门观察 tie-breaking 规则差异。
- 看 `retail_prices` 分布：观察实际业务尾数结构可能带来的偏差。

- Check `mean_error` for systematic upward or downward bias.
- Check `rmse` / `mean_abs_error` for typical per-value error.
- Check `error_variance` for rounding-noise spread.
- Check `sum_error` for aggregate drift.
- Check `half_ties` to emphasize tie-breaking differences.
- Check `retail_prices` to see bias caused by business-like price endings.

---

## 9. 推荐实验 / Suggested Experiments

### 9.1 观察 tie-breaking 差异 / Observe Tie-Breaking Differences

```bash
python3 math/rounding_scheme_benchmark.py \
  --distributions half_ties \
  --schemes half_up,half_down,half_away_from_zero,half_even,half_odd,random_tie,stochastic \
  --n 100000
```

预期现象：

Expected observation:

- `half_up` 会明显正偏。
- `half_down` 会明显负偏。
- `half_even`、`random_tie`、`stochastic` 的平均偏差通常更接近 0。

- `half_up` has clear positive bias.
- `half_down` has clear negative bias.
- `half_even`, `random_tie`, and `stochastic` usually have mean bias closer to zero.

### 9.2 观察随机舍入的 bias 与 variance / Bias and Variance of Stochastic Rounding

```bash
python3 math/rounding_scheme_benchmark.py \
  --distributions uniform_positive,normal,retail_prices \
  --schemes half_even,stochastic \
  --n 200000
```

预期现象：

Expected observation:

- `stochastic` 的 `mean_error` 往往很小。
- `stochastic` 的 `error_variance` 和 `rmse` 往往比 `half_even` 大。

- `stochastic` often has small `mean_error`.
- `stochastic` often has larger `error_variance` and `rmse` than `half_even`.

### 9.3 观察舍入到两位小数 / Rounding to Two Decimal Places

```bash
python3 math/rounding_scheme_benchmark.py \
  --quantum 0.01 \
  --distributions retail_prices \
  --schemes half_up,half_even,stochastic \
  --n 100000
```

---

## 10. 实践建议 / Practical Recommendations

### 10.1 日常展示 / Display Formatting

如果只是显示给用户看，通常选择符合用户直觉和业务规则的方案，例如 `half_away_from_zero` 或特定语言/数据库默认规则。

For user-facing display, choose a rule matching user expectation and business requirements, such as `half_away_from_zero` or the default of your language/database.

### 10.2 财务和报表 / Finance and Reporting

不要只说“使用四舍五入”，而要明确：

Do not merely say “round normally”; specify:

```text
rounding quantum: 0.01
tie rule: half_even or half_away_from_zero
aggregation rule: round before sum or sum before round
```

特别注意“先逐项舍入再求和”和“先求和再舍入”可能得到不同结果。

Especially note that “round each item then sum” and “sum then round” may produce different results.

### 10.3 科学计算 / Scientific Computing

如果关心长期无偏性，优先关注：

If long-term unbiasedness matters, focus on:

```text
mean_error
sum_error
sum_relative_error
```

如果关心数值噪声，关注：

If numerical noise matters, focus on:

```text
error_variance
rmse
p95_abs_error
max_abs_error
```

### 10.4 随机舍入 / Stochastic Rounding

随机舍入不是“误差更小”的万能方案。它通常是“偏差更小，但噪声更大”。是否合适取决于应用是否能接受随机性和更大的单点误差。

Stochastic rounding is not a universal way to make errors smaller. It usually means “lower bias but higher noise”. Its suitability depends on whether the application can tolerate randomness and larger per-value errors.

---

## 11. 小结 / Summary

- `floor`、`ceiling`、`toward_zero`、`away_from_zero` 是方向性舍入，通常 bias 明显。
- `half_up`、`half_down`、`half_away_from_zero` 等最近取整方案在连续分布上相似，但在 tie-heavy 数据上差异很大。
- `half_even` 的目标是减少长期 tie 偏差，因此在统计和科学计算中常见。
- `stochastic` 在期望上无偏，但会引入更大的随机噪声。
- 评价舍入方案不能只看单个例子，要结合数据分布和指标，例如 `mean_error`、`error_variance`、`rmse`、`sum_error`。

- `floor`, `ceiling`, `toward_zero`, and `away_from_zero` are directional and often biased.
- Nearest rounding schemes such as `half_up`, `half_down`, and `half_away_from_zero` look similar on continuous distributions but differ greatly on tie-heavy data.
- `half_even` aims to reduce long-term tie bias and is common in statistics and scientific computing.
- `stochastic` is unbiased in expectation but introduces more random noise.
- A rounding scheme should be evaluated with respect to data distribution and metrics such as `mean_error`, `error_variance`, `rmse`, and `sum_error`, not just isolated examples.
