#!/usr/bin/env python3
"""Benchmark common rounding schemes on synthetic distributions.

The script compares deterministic and stochastic rounding rules by measuring
their rounding errors on several data-generating models.  It intentionally uses
only the Python standard library so it can be copied into small toolboxes and run
without dependency installation.

Examples:
    python3 math/rounding_scheme_benchmark.py
    python3 math/rounding_scheme_benchmark.py --n 200000 --quantum 0.01 --csv out.csv
    python3 math/rounding_scheme_benchmark.py --distributions normal,half_ties --schemes half_even,half_away_from_zero,stochastic
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, TypeVar


EPS = 1e-12
T = TypeVar("T")


@dataclass(frozen=True)
class Scheme:
    """A rounding scheme under test."""

    name: str
    description: str
    func: Callable[[float, float, random.Random], float]


@dataclass(frozen=True)
class Distribution:
    """A synthetic data distribution."""

    name: str
    description: str
    generator: Callable[[int, random.Random], list[float]]


def _round_scaled_nearest(scaled: float, tie_rule: str, rng: random.Random) -> int:
    """Round a scaled value to an integer using the given tie rule.

    Python's modulo/floor decomposition is used, so this function works for
    negative values as well: scaled = floor(scaled) + frac, frac in [0, 1).
    """

    lower = math.floor(scaled)
    frac = scaled - lower

    if frac < 0.5 - EPS:
        return lower
    if frac > 0.5 + EPS:
        return lower + 1

    if tie_rule == "half_up":
        return lower + 1
    if tie_rule == "half_down":
        return lower
    if tie_rule == "half_even":
        return lower if lower % 2 == 0 else lower + 1
    if tie_rule == "half_odd":
        return lower if lower % 2 != 0 else lower + 1
    if tie_rule == "half_away_from_zero":
        # For ties, choose the candidate with larger absolute magnitude.
        return lower if abs(lower) > abs(lower + 1) else lower + 1
    if tie_rule == "half_toward_zero":
        return lower if abs(lower) < abs(lower + 1) else lower + 1
    if tie_rule == "random_tie":
        return lower + rng.randrange(2)
    raise ValueError(f"unknown tie rule: {tie_rule}")


def round_floor(x: float, quantum: float, rng: random.Random) -> float:
    return math.floor(x / quantum) * quantum


def round_ceiling(x: float, quantum: float, rng: random.Random) -> float:
    return math.ceil(x / quantum) * quantum


def round_toward_zero(x: float, quantum: float, rng: random.Random) -> float:
    scaled = x / quantum
    return (math.floor(scaled) if scaled >= 0 else math.ceil(scaled)) * quantum


def round_away_from_zero(x: float, quantum: float, rng: random.Random) -> float:
    scaled = x / quantum
    return (math.ceil(scaled) if scaled >= 0 else math.floor(scaled)) * quantum


def make_nearest(tie_rule: str) -> Callable[[float, float, random.Random], float]:
    def _round(x: float, quantum: float, rng: random.Random) -> float:
        return _round_scaled_nearest(x / quantum, tie_rule, rng) * quantum

    return _round


def round_stochastic(x: float, quantum: float, rng: random.Random) -> float:
    """Unbiased stochastic rounding to the adjacent grid points.

    If x lies between lower and upper multiples of quantum, return upper with
    probability (x-lower)/(upper-lower), otherwise lower.  The expectation of the
    rounded value equals x up to floating-point noise.
    """

    scaled = x / quantum
    lower_i = math.floor(scaled)
    frac = scaled - lower_i
    if frac <= EPS:
        return lower_i * quantum
    return (lower_i + 1 if rng.random() < frac else lower_i) * quantum


SCHEMES: tuple[Scheme, ...] = (
    Scheme("floor", "向下取整：总是取不大于 x 的量化格点", round_floor),
    Scheme("ceiling", "向上取整：总是取不小于 x 的量化格点", round_ceiling),
    Scheme("toward_zero", "朝 0 取整：截断小数部分", round_toward_zero),
    Scheme("away_from_zero", "远离 0 取整：绝对值方向进位", round_away_from_zero),
    Scheme("half_up", "最近取整，恰好 0.5 时向 +∞；中国日常口径常说的“五舍五入”", make_nearest("half_up")),
    Scheme("half_down", "最近取整，恰好 0.5 时向 -∞", make_nearest("half_down")),
    Scheme("half_away_from_zero", "最近取整，恰好 0.5 时远离 0；许多语言/库称为商业四舍五入", make_nearest("half_away_from_zero")),
    Scheme("half_toward_zero", "最近取整，恰好 0.5 时朝 0", make_nearest("half_toward_zero")),
    Scheme("half_even", "最近取整，恰好 0.5 时取偶数；银行家舍入/IEEE 754 默认近似思想", make_nearest("half_even")),
    Scheme("half_odd", "最近取整，恰好 0.5 时取奇数", make_nearest("half_odd")),
    Scheme("random_tie", "最近取整，仅在恰好 0.5 时随机上下", make_nearest("random_tie")),
    Scheme("stochastic", "随机舍入：按距离概率随机到相邻上下格点，理论上无偏", round_stochastic),
)


def gen_uniform_positive(n: int, rng: random.Random) -> list[float]:
    return [rng.uniform(0.0, 100.0) for _ in range(n)]


def gen_uniform_centered(n: int, rng: random.Random) -> list[float]:
    return [rng.uniform(-50.0, 50.0) for _ in range(n)]


def gen_normal(n: int, rng: random.Random) -> list[float]:
    return [rng.gauss(0.0, 20.0) for _ in range(n)]


def gen_lognormal(n: int, rng: random.Random) -> list[float]:
    return [rng.lognormvariate(2.0, 0.8) for _ in range(n)]


def gen_exponential(n: int, rng: random.Random) -> list[float]:
    return [rng.expovariate(1.0 / 20.0) for _ in range(n)]


def gen_beta_skewed(n: int, rng: random.Random) -> list[float]:
    # Skewed and bounded; useful for seeing how one-sided distributions amplify
    # directed rounding bias.
    return [100.0 * rng.betavariate(2.0, 8.0) for _ in range(n)]


def gen_half_ties(n: int, rng: random.Random) -> list[float]:
    # Heavy mass exactly on k + 0.5, where tie-breaking rules differ maximally.
    return [rng.randint(-50, 50) + 0.5 for _ in range(n)]


def gen_retail_prices(n: int, rng: random.Random) -> list[float]:
    # A toy retail-like distribution: many .99 values plus small random cents.
    values: list[float] = []
    for _ in range(n):
        base = rng.randint(1, 200)
        if rng.random() < 0.65:
            values.append(base + 0.99)
        else:
            values.append(base + rng.randrange(100) / 100.0)
    return values


DISTRIBUTIONS: tuple[Distribution, ...] = (
    Distribution("uniform_positive", "U(0, 100)，正数连续均匀分布", gen_uniform_positive),
    Distribution("uniform_centered", "U(-50, 50)，正负对称连续均匀分布", gen_uniform_centered),
    Distribution("normal", "N(0, 20^2)，正负对称正态分布", gen_normal),
    Distribution("lognormal", "对数正态，右偏且只含正数", gen_lognormal),
    Distribution("exponential", "指数分布，右偏且只含正数", gen_exponential),
    Distribution("beta_skewed", "100*Beta(2,8)，有界右偏分布", gen_beta_skewed),
    Distribution("half_ties", "大量恰好落在 k+0.5 的人工分布", gen_half_ties),
    Distribution("retail_prices", "零售价格玩具模型，.99 尾数较多", gen_retail_prices),
)


def get_by_names(items: Sequence[T], names: str, item_name: str) -> list[T]:
    lookup = {getattr(item, "name"): item for item in items}
    if names == "all":
        return list(items)
    selected: list[T] = []
    for raw_name in names.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in lookup:
            valid = ", ".join(lookup)
            raise SystemExit(f"Unknown {item_name}: {name}. Valid values: all,{valid}")
        selected.append(lookup[name])
    return selected


def percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def evaluate(values: Sequence[float], scheme: Scheme, quantum: float, seed: int) -> dict[str, float | str | int]:
    rng = random.Random(seed)
    rounded = [scheme.func(x, quantum, rng) for x in values]
    errors = [r - x for x, r in zip(values, rounded)]
    abs_errors = [abs(e) for e in errors]
    squared_errors = [e * e for e in errors]
    rel_errors = [abs(e) / max(abs(x), quantum) for x, e in zip(values, errors)]
    sorted_abs = sorted(abs_errors)

    sum_original = math.fsum(values)
    sum_rounded = math.fsum(rounded)
    mean_error = statistics.fmean(errors)
    mse = statistics.fmean(squared_errors)

    return {
        "scheme": scheme.name,
        "mean_error": mean_error,
        "mean_error_in_ulps": mean_error / quantum,
        "error_variance": statistics.pvariance(errors),
        "mean_abs_error": statistics.fmean(abs_errors),
        "rmse": math.sqrt(mse),
        "max_abs_error": max(abs_errors),
        "p95_abs_error": percentile(sorted_abs, 0.95),
        "mean_relative_abs_error": statistics.fmean(rel_errors),
        "sum_error": sum_rounded - sum_original,
        "sum_relative_error": (sum_rounded - sum_original) / sum_original if abs(sum_original) > EPS else float("nan"),
        "rounded_mean_shift": statistics.fmean(rounded) - statistics.fmean(values),
    }


METRIC_COLUMNS = (
    "mean_error",
    "mean_error_in_ulps",
    "error_variance",
    "mean_abs_error",
    "rmse",
    "max_abs_error",
    "p95_abs_error",
    "mean_relative_abs_error",
    "sum_error",
    "sum_relative_error",
)

INPUT_METRIC_COLUMNS = (
    "input_mean",
    "input_variance",
    "input_stddev",
    "input_mean_abs",
    "input_rms",
    "input_min",
    "input_p05",
    "input_median",
    "input_p95",
    "input_max",
    "input_sum",
)

FIXED_DECIMALS = 6
SCI_DECIMALS = 6


def original_error_row(values: Sequence[float]) -> dict[str, float | str | int]:
    """Return the no-rounding baseline row.

    The table primarily reports error metrics.  For the original input data,
    the rounding error is exactly zero, so this row is a useful visual baseline
    before the actual rounding schemes.
    """

    return {
        "scheme": "original/no_round",
        "mean_error": 0.0,
        "mean_error_in_ulps": 0.0,
        "error_variance": 0.0,
        "mean_abs_error": 0.0,
        "rmse": 0.0,
        "max_abs_error": 0.0,
        "p95_abs_error": 0.0,
        "mean_relative_abs_error": 0.0,
        "sum_error": 0.0,
        "sum_relative_error": 0.0,
        "rounded_mean_shift": 0.0,
    }


def input_metrics(values: Sequence[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    squares = [x * x for x in values]
    return {
        "input_mean": statistics.fmean(values),
        "input_variance": statistics.pvariance(values),
        "input_stddev": statistics.pstdev(values),
        "input_mean_abs": statistics.fmean(abs(x) for x in values),
        "input_rms": math.sqrt(statistics.fmean(squares)),
        "input_min": sorted_values[0],
        "input_p05": percentile(sorted_values, 0.05),
        "input_median": percentile(sorted_values, 0.50),
        "input_p95": percentile(sorted_values, 0.95),
        "input_max": sorted_values[-1],
        "input_sum": math.fsum(values),
    }


def should_use_scientific(values: Sequence[float]) -> bool:
    finite_nonzero = [abs(v) for v in values if math.isfinite(v) and abs(v) > 0.0]
    if not finite_nonzero:
        return False
    max_abs = max(finite_nonzero)
    min_abs = min(finite_nonzero)
    return max_abs >= 10_000_000 or (min_abs < 1e-6 and max_abs < 10_000)


def common_exponent(values: Sequence[float]) -> int:
    finite_nonzero = [abs(v) for v in values if math.isfinite(v) and abs(v) > 0.0]
    if not finite_nonzero:
        return 0
    return int(math.floor(math.log10(max(finite_nonzero))))


def format_fixed(value: float) -> str:
    return f"{value:.{FIXED_DECIMALS}f}"


def format_scientific_with_exponent(value: float, exponent: int) -> str:
    scaled = value / (10.0 ** exponent)
    return f"{scaled:.{SCI_DECIMALS}f}e{exponent:+04d}"


def align_decimal_strings(strings: Sequence[str]) -> list[str]:
    decimal_positions = [s.find(".") for s in strings]
    left_width = max((pos if pos >= 0 else len(s)) for s, pos in zip(strings, decimal_positions))
    right_width = max((len(s) - pos - 1 if pos >= 0 else 0) for s, pos in zip(strings, decimal_positions))
    aligned: list[str] = []
    for s, pos in zip(strings, decimal_positions):
        if pos >= 0:
            left = s[:pos]
            right = s[pos + 1 :]
            aligned.append(f"{left.rjust(left_width)}.{right.ljust(right_width)}")
        else:
            aligned.append(s.rjust(left_width + 1 + right_width))
    return aligned


def format_numeric_column(values: Sequence[float | str | int]) -> list[str]:
    numeric_values = [float(v) for v in values if not isinstance(v, str)]
    use_scientific = should_use_scientific(numeric_values)
    exponent = common_exponent(numeric_values) if use_scientific else 0

    raw_strings: list[str] = []
    for value in values:
        if isinstance(value, str):
            raw_strings.append(value)
            continue
        numeric = float(value)
        if math.isnan(numeric):
            raw_strings.append("nan")
        elif use_scientific:
            raw_strings.append(format_scientific_with_exponent(numeric, exponent))
        else:
            raw_strings.append(format_fixed(numeric))
    return align_decimal_strings(raw_strings)


def format_float(value: float | str | int) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "nan"
    return format_fixed(value)


def print_metric_line(label: str, metrics: dict[str, float]) -> None:
    formatted_values = format_numeric_column([metrics[col] for col in INPUT_METRIC_COLUMNS])
    parts = [f"{col}={value}" for col, value in zip(INPUT_METRIC_COLUMNS, formatted_values)]
    print(f"{label}: " + ", ".join(parts))


def print_table(
    distribution: Distribution,
    input_stats: dict[str, float],
    original_row: dict[str, float | str | int],
    rows: list[dict[str, float | str | int]],
    top: int | None,
) -> None:
    print(f"\n=== Distribution: {distribution.name} ({distribution.description}) ===")
    print_metric_line("Input data metrics", input_stats)
    printable_rows = rows if top is None else rows[:top]
    printable_rows = [original_row] + printable_rows
    columns = ("scheme",) + METRIC_COLUMNS
    formatted_columns: dict[str, list[str]] = {}
    for col in columns:
        if col == "scheme":
            formatted_columns[col] = [str(row[col]) for row in printable_rows]
        else:
            formatted_columns[col] = format_numeric_column([row[col] for row in printable_rows])
    widths = {col: max(len(col), *(len(value) for value in formatted_columns[col])) for col in columns}
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row_index in range(len(printable_rows)):
        cells: list[str] = []
        for col in columns:
            value = formatted_columns[col][row_index]
            cells.append(value.ljust(widths[col]) if col == "scheme" else value.rjust(widths[col]))
        print("  ".join(cells))


def write_csv(path: str, all_rows: Iterable[dict[str, float | str | int]]) -> None:
    fieldnames = ("distribution", "scheme") + METRIC_COLUMNS + ("rounded_mean_shift",) + INPUT_METRIC_COLUMNS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rounding schemes on multiple synthetic distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=50_000, help="sample count per distribution")
    parser.add_argument("--quantum", type=float, default=1.0, help="rounding step, e.g. 1, 0.1, 0.01")
    parser.add_argument("--seed", type=int, default=20260710, help="random seed")
    parser.add_argument("--distributions", default="all", help="comma-separated distribution names or 'all'")
    parser.add_argument("--schemes", default="all", help="comma-separated scheme names or 'all'")
    parser.add_argument("--sort-by", default="rmse", choices=METRIC_COLUMNS, help="metric used to sort each table")
    parser.add_argument("--descending", action="store_true", help="sort larger metric values first")
    parser.add_argument("--top", type=int, default=None, help="show only the first N schemes per distribution")
    parser.add_argument("--csv", default=None, help="optional output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise SystemExit("--n must be positive")
    if args.quantum <= 0:
        raise SystemExit("--quantum must be positive")
    if args.top is not None and args.top <= 0:
        raise SystemExit("--top must be positive when provided")

    distributions = get_by_names(DISTRIBUTIONS, args.distributions, "distribution")
    schemes = get_by_names(SCHEMES, args.schemes, "scheme")
    all_rows: list[dict[str, float | str | int]] = []

    print("Rounding scheme benchmark")
    print(f"sample_count={args.n}, quantum={args.quantum}, seed={args.seed}")
    print("Lower mean_error/mean_error_in_ulps indicates lower bias; lower rmse/mae/variance indicates lower noise.")

    for dist_index, distribution in enumerate(distributions):
        # Use a stable but distinct seed per distribution, independent of the
        # number/order of schemes.
        dist_rng = random.Random(args.seed + 10_000 * dist_index)
        values = distribution.generator(args.n, dist_rng)
        input_stats = input_metrics(values)
        baseline_row = original_error_row(values)
        baseline_row["distribution"] = distribution.name
        baseline_row.update(input_stats)
        rows: list[dict[str, float | str | int]] = []
        for scheme_index, scheme in enumerate(schemes):
            row = evaluate(values, scheme, args.quantum, args.seed + 1_000_000 * dist_index + scheme_index)
            row["distribution"] = distribution.name
            row.update(input_stats)
            rows.append(row)
        rows.sort(key=lambda item: float(item[args.sort_by]), reverse=args.descending)
        all_rows.append(baseline_row)
        all_rows.extend(rows)
        print_table(distribution, input_stats, baseline_row, rows, args.top)

    if args.csv:
        write_csv(args.csv, all_rows)
        print(f"\nCSV written to: {args.csv}")


if __name__ == "__main__":
    main()
