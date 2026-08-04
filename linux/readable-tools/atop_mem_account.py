#!/usr/bin/env python3

"""从 atop 原始日志中提取内存记账信息。

默认读取最新 atop 原始日志中的最新完整样本（通过 `atop -r` 回放），展示：
1. 整机物理内存/Swap 概览
2. 内核/系统相关内存分项（直接基于 atop 的 MEM/SWP 行）
3. 进程内存聚合与全部进程列表

说明：
- 进程侧优先使用 PSS（需要 atop 支持 `-R`），否则退化到 RSS。
- 内核分项和进程分项来自不同视角，不应简单逐项相加。
- 本脚本只依赖 atop 自身的 parseable 输出，不直接读 /proc。
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import signal
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 atop 原始日志中生成内存记账摘要",
    )
    parser.add_argument(
        "rawfile",
        nargs="?",
        help="atop 原始日志路径；省略时默认自动选择最新完整日志",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="不读取原始日志，直接执行 atop 采集当前一帧样本",
    )
    parser.add_argument(
        "--begin",
        help="开始时间，透传给 atop -b，格式如 202608041030 或 10:30",
    )
    parser.add_argument(
        "--end",
        help="结束时间，透传给 atop -e，格式如 202608041130 或 11:30",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=-1,
        help="选择第几个样本；默认 -1 表示最后一个样本",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="展示前多少个进程；默认 0 表示全部进程",
    )
    parser.add_argument(
        "--sort",
        choices=["auto", "pss", "rss", "vsize", "swap"],
        default="auto",
        help="Top 进程排序字段，默认 auto（优先 pss，否则 rss）",
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Top 进程中保留占用为 0 的条目",
    )
    args = parser.parse_args()
    if args.live and (args.begin or args.end):
        parser.error("--live 模式下不支持 --begin/--end")
    return args


def find_latest_rawfile() -> str:
    candidates: List[Path] = []
    patterns = [
        (Path("/var/log/atop"), r"atop_\d{8}"),
        (Path("/var/log"), r"atop_\d{8}"),
    ]

    for base, name_pattern in patterns:
        if not base.exists() or not base.is_dir():
            continue
        for path in sorted(base.iterdir()):
            if path.is_file() and re.fullmatch(name_pattern, path.name):
                candidates.append(path)

    if not candidates:
        print(
            "错误：未找到 atop 原始日志。请显式传入 rawfile，或确认 /var/log/atop/atop_YYYYMMDD 存在。",
            file=sys.stderr,
        )
        raise SystemExit(2)

    dated_candidates = []
    for path in candidates:
        m = re.fullmatch(r"atop_(\d{8})", path.name)
        if not m:
            continue
        dated_candidates.append((dt.datetime.strptime(m.group(1), "%Y%m%d").date(), path))

    if not dated_candidates:
        return str(sorted(candidates)[-1])

    today = dt.date.today()
    completed = [item for item in dated_candidates if item[0] < today]
    selected = max(completed or dated_candidates, key=lambda item: item[0])[1]
    return str(selected)


def rawfile_date(rawfile: str) -> Optional[dt.date]:
    m = re.fullmatch(r"atop_(\d{8})", Path(rawfile).name)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y%m%d").date()


def default_begin_for_latest_sample(rawfile: str) -> Optional[str]:
    """Return an atop -b value that avoids replaying a full day by default.

    Atop logs are commonly one file per day. To get the newest complete sample
    from a completed day, replaying from 23:50 is enough on the usual 10-minute
    log interval and avoids producing millions of PRM rows.
    """

    date = rawfile_date(rawfile)
    if not date:
        return None

    if date < dt.date.today():
        return date.strftime("%Y%m%d") + "2350"

    # If the only available file is today's still-growing file, use a short
    # recent window instead of replaying from midnight.
    begin = dt.datetime.now() - dt.timedelta(minutes=15)
    return begin.strftime("%Y%m%d%H%M")


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{value} B"


def display_width(text: object) -> int:
    width = 0
    for ch in str(text):
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in {"F", "W"} else 1
    return width


def ljust_display(text: object, width: int) -> str:
    text = str(text)
    return text + " " * max(width - display_width(text), 0)


def rjust_display(text: object, width: int) -> str:
    text = str(text)
    return " " * max(width - display_width(text), 0) + text


def fit_display(text: object, width: int) -> str:
    text = str(text)
    out = []
    used = 0
    for ch in text:
        ch_width = 2 if unicodedata.east_asian_width(ch) in {"F", "W"} else 1
        if used + ch_width > width:
            break
        out.append(ch)
        used += ch_width
    return "".join(out)


def kib_to_bytes(value: int) -> int:
    return value * 1024


def pages_to_bytes(pages: int, page_size: int) -> int:
    return pages * page_size


def to_int(value: str) -> int:
    if value == "?":
        return 0
    return int(value)


@dataclass
class ProcessMem:
    pid: int
    name: str
    state: str
    page_size: int
    vsize_kib: int
    rss_kib: int
    text_kib: int
    vgrow_kib: int
    rgrow_kib: int
    minflt: int
    majflt: int
    vlibs_kib: int
    vdata_kib: int
    vstack_kib: int
    swap_kib: int
    tgid: int
    is_process: bool
    pss_kib: int
    locked_kib: int
    mem_max_kib: int
    mem_max_eff_kib: int
    swap_max_kib: int
    swap_max_eff_kib: int

    @property
    def sort_pss(self) -> int:
        return self.pss_kib if self.pss_kib > 0 else self.rss_kib


@dataclass
class Sample:
    host: str
    epoch: int
    date: str
    time: str
    interval: int
    mem: Optional[Dict[str, int]] = None
    swp: Optional[Dict[str, int]] = None
    procs: List[ProcessMem] = field(default_factory=list)


def run_atop(args: argparse.Namespace) -> str:
    if args.live:
        cmd = ["atop", "-P", "MEM,SWP,PRM", "-R", "-Z", "1", "1"]
    else:
        rawfile = args.rawfile or find_latest_rawfile()
        cmd = ["atop", "-r", rawfile, "-P", "MEM,SWP,PRM", "-R", "-Z"]
        begin = args.begin
        if begin is None and args.end is None and args.sample_index == -1:
            begin = default_begin_for_latest_sample(rawfile)
        if begin:
            cmd.extend(["-b", begin])
        if args.end:
            cmd.extend(["-e", args.end])

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        print("错误：未找到 atop 命令，请先安装 atop。", file=sys.stderr)
        raise SystemExit(127)
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.strip() or exc.stdout.strip() or "未知错误"
        print(f"错误：执行 atop 失败：{err}", file=sys.stderr)
        raise SystemExit(exc.returncode or 1)

    return result.stdout


def ensure_sample(samples: List[Sample], host: str, epoch: int, date: str, time_: str, interval: int) -> Sample:
    if samples and samples[-1].epoch == epoch:
        return samples[-1]
    sample = Sample(host=host, epoch=epoch, date=date, time=time_, interval=interval)
    samples.append(sample)
    return sample


def parse_mem(tokens: List[str]) -> Dict[str, int]:
    page_size = to_int(tokens[6])
    return {
        "page_size": page_size,
        "phys_total": pages_to_bytes(to_int(tokens[7]), page_size),
        "free": pages_to_bytes(to_int(tokens[8]), page_size),
        "cache": pages_to_bytes(to_int(tokens[9]), page_size),
        "buffers": pages_to_bytes(to_int(tokens[10]), page_size),
        "slab": pages_to_bytes(to_int(tokens[11]), page_size),
        "dirty": pages_to_bytes(to_int(tokens[12]), page_size),
        "slab_reclaimable": pages_to_bytes(to_int(tokens[13]), page_size),
        "balloon": pages_to_bytes(to_int(tokens[14]), page_size),
        "shmem": pages_to_bytes(to_int(tokens[15]), page_size),
        "shmrss": pages_to_bytes(to_int(tokens[16]), page_size),
        "shmswp": pages_to_bytes(to_int(tokens[17]), page_size),
        "huge_page_size": to_int(tokens[18]),
        "huge_total": to_int(tokens[19]) * to_int(tokens[18]),
        "huge_free": to_int(tokens[20]) * to_int(tokens[18]),
        "zfs_arc": pages_to_bytes(to_int(tokens[21]), page_size),
        "ksm_sharing": pages_to_bytes(to_int(tokens[22]), page_size),
        "ksm_shared": pages_to_bytes(to_int(tokens[23]), page_size),
        "tcp": pages_to_bytes(to_int(tokens[24]), page_size),
        "udp": pages_to_bytes(to_int(tokens[25]), page_size),
        "pagetables": pages_to_bytes(to_int(tokens[26]), page_size),
    }


def parse_swp(tokens: List[str]) -> Dict[str, int]:
    page_size = to_int(tokens[6])
    return {
        "page_size": page_size,
        "swap_total": pages_to_bytes(to_int(tokens[7]), page_size),
        "swap_free": pages_to_bytes(to_int(tokens[8]), page_size),
        "swap_cache": pages_to_bytes(to_int(tokens[9]), page_size),
        "committed": pages_to_bytes(to_int(tokens[10]), page_size),
        "commit_limit": pages_to_bytes(to_int(tokens[11]), page_size),
        "swap_cache_dup": pages_to_bytes(to_int(tokens[12]), page_size),
        "zswap_compressed": pages_to_bytes(to_int(tokens[13]), page_size),
        "zswap_pool": pages_to_bytes(to_int(tokens[14]), page_size),
    }


def parse_prm(tokens: List[str]) -> ProcessMem:
    return ProcessMem(
        pid=to_int(tokens[6]),
        name=tokens[7],
        state=tokens[8],
        page_size=to_int(tokens[9]),
        vsize_kib=to_int(tokens[10]),
        rss_kib=to_int(tokens[11]),
        text_kib=to_int(tokens[12]),
        vgrow_kib=to_int(tokens[13]),
        rgrow_kib=to_int(tokens[14]),
        minflt=to_int(tokens[15]),
        majflt=to_int(tokens[16]),
        vlibs_kib=to_int(tokens[17]),
        vdata_kib=to_int(tokens[18]),
        vstack_kib=to_int(tokens[19]),
        swap_kib=to_int(tokens[20]),
        tgid=to_int(tokens[21]),
        is_process=tokens[22].lower() == "y",
        pss_kib=to_int(tokens[23]),
        locked_kib=to_int(tokens[24]),
        mem_max_kib=to_int(tokens[25]),
        mem_max_eff_kib=to_int(tokens[26]),
        swap_max_kib=to_int(tokens[27]),
        swap_max_eff_kib=to_int(tokens[28]),
    )


def parse_samples(raw_text: str) -> List[Sample]:
    samples: List[Sample] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line in {"SEP", "RESET"}:
            continue

        tokens = line.split()
        if len(tokens) < 7:
            continue

        label, host, epoch, date, time_, interval = tokens[:6]
        sample = ensure_sample(samples, host, to_int(epoch), date, time_, to_int(interval))

        if label == "MEM":
            sample.mem = parse_mem(tokens)
        elif label == "SWP":
            sample.swp = parse_swp(tokens)
        elif label == "PRM":
            sample.procs.append(parse_prm(tokens))

    return [s for s in samples if s.mem is not None or s.swp is not None or s.procs]


def choose_sample(samples: List[Sample], sample_index: int) -> Sample:
    if not samples:
        print("错误：没有从 atop 输出中解析到任何样本。", file=sys.stderr)
        raise SystemExit(2)

    try:
        return samples[sample_index]
    except IndexError:
        print(
            f"错误：样本索引 {sample_index} 超出范围，当前共有 {len(samples)} 个样本。",
            file=sys.stderr,
        )
        raise SystemExit(2)


def metric_value(proc: ProcessMem, sort_key: str) -> int:
    if sort_key == "pss":
        return proc.pss_kib
    if sort_key == "rss":
        return proc.rss_kib
    if sort_key == "vsize":
        return proc.vsize_kib
    if sort_key == "swap":
        return proc.swap_kib
    return proc.sort_pss


def print_kv(name: str, value: int, total: Optional[int] = None) -> None:
    label = ljust_display(name, 24)
    value_text = rjust_display(human_bytes(value), 12)
    if total and total > 0:
        ratio = value / total * 100.0
        print(f"  {label} {value_text}  {ratio:6.2f}%")
    else:
        print(f"  {label} {value_text}")


def print_report(sample: Sample, args: argparse.Namespace) -> None:
    mem = sample.mem or {}
    swp = sample.swp or {}
    page_total = mem.get("phys_total", 0)
    free = mem.get("free", 0)
    used = max(page_total - free, 0)
    reclaimable = mem.get("cache", 0) + mem.get("buffers", 0) + mem.get("slab_reclaimable", 0)
    non_reclaimable = max(used - reclaimable, 0)
    slab_unreclaimable = max(mem.get("slab", 0) - mem.get("slab_reclaimable", 0), 0)
    huge_used = max(mem.get("huge_total", 0) - mem.get("huge_free", 0), 0)
    swap_total = swp.get("swap_total", 0)
    swap_used = max(swap_total - swp.get("swap_free", 0), 0)

    processes = [p for p in sample.procs if p.is_process]
    total_rss = sum(kib_to_bytes(p.rss_kib) for p in processes)
    total_pss = sum(kib_to_bytes(p.pss_kib) for p in processes if p.pss_kib > 0)
    total_swap = sum(kib_to_bytes(p.swap_kib) for p in processes)
    total_locked = sum(kib_to_bytes(p.locked_kib) for p in processes)
    pss_available = any(p.pss_kib > 0 for p in processes)

    sort_key = args.sort
    if sort_key == "auto":
        sort_key = "pss" if pss_available else "rss"

    ranked = sorted(processes, key=lambda p: metric_value(p, sort_key), reverse=True)
    if args.include_zero:
        pass
    elif args.top > 0:
        ranked = [p for p in ranked if metric_value(p, sort_key) > 0]
    if args.top > 0:
        ranked = ranked[: args.top]

    print(f"样本时间: {sample.date} {sample.time}  host={sample.host}  interval={sample.interval}s")
    print()

    print("=== 整机内存概览 ===")
    print_kv("物理内存总量", page_total)
    print_kv("空闲内存", free, page_total)
    print_kv("已用内存", used, page_total)
    print_kv("可回收内存估计", reclaimable, page_total)
    print_kv("不可回收已用估计", non_reclaimable, page_total)
    print()

    print("=== 内核/系统分项（来自 atop MEM/SWP，部分字段存在交叉，不宜直接求和）===")
    print_kv("Page Cache", mem.get("cache", 0), page_total)
    print_kv("Buffer Cache", mem.get("buffers", 0), page_total)
    print_kv("Slab 总量", mem.get("slab", 0), page_total)
    print_kv("Slab 可回收", mem.get("slab_reclaimable", 0), page_total)
    print_kv("Slab 不可回收", slab_unreclaimable, page_total)
    print_kv("共享内存/SHMEM", mem.get("shmem", 0), page_total)
    print_kv("共享内存驻留", mem.get("shmrss", 0), page_total)
    print_kv("共享内存换出", mem.get("shmswp", 0))
    print_kv("脏页", mem.get("dirty", 0), page_total)
    print_kv("HugePages 已用", huge_used, page_total)
    print_kv("HugePages 空闲", mem.get("huge_free", 0), page_total)
    print_kv("TCP Socket 内存", mem.get("tcp", 0), page_total)
    print_kv("UDP Socket 内存", mem.get("udp", 0), page_total)
    print_kv("页表", mem.get("pagetables", 0), page_total)
    print_kv("ZFS ARC", mem.get("zfs_arc", 0), page_total)
    print_kv("KSM sharing", mem.get("ksm_sharing", 0), page_total)
    print_kv("KSM shared", mem.get("ksm_shared", 0), page_total)
    print_kv("VM Balloon", mem.get("balloon", 0), page_total)
    print_kv("Swap 总量", swap_total)
    print_kv("Swap 已用", swap_used, swap_total)
    print_kv("Swap Cache", swp.get("swap_cache", 0), swap_total)
    print_kv("Committed_AS", swp.get("committed", 0))
    print_kv("Commit Limit", swp.get("commit_limit", 0))
    print_kv("zswap 压缩页", swp.get("zswap_compressed", 0), swap_total)
    print_kv("zswap Pool", swp.get("zswap_pool", 0), swap_total)
    print()

    print("=== 进程侧聚合（来自 atop PRM）===")
    print(f"  {ljust_display('进程数量', 24)} {rjust_display(len(processes), 12)}")
    print_kv("进程 RSS 总和", total_rss, page_total)
    if pss_available:
        print_kv("进程 PSS 总和", total_pss, page_total)
    else:
        print(f"  {ljust_display('进程 PSS 总和', 24)} {rjust_display('N/A', 12)}  该 atop 样本未记录 PSS")
    print_kv("进程 Swap 总和", total_swap)
    print_kv("进程 Locked 总和", total_locked)
    print()

    title = f"Top {len(ranked)} 进程" if args.top > 0 else f"全部 {len(ranked)} 个进程"
    print(f"=== {title}（按 {sort_key.upper()} 排序）===")
    print(f"{rjust_display('PID', 8)}  {ljust_display('NAME', 24)} {rjust_display('RSS', 10)} {rjust_display('PSS', 10)} {rjust_display('SWAP', 10)} {rjust_display('VSIZE', 10)}")
    for proc in ranked:
        pss_text = human_bytes(kib_to_bytes(proc.pss_kib)) if pss_available else "N/A"
        print(
            f"{rjust_display(proc.pid, 8)}  "
            f"{ljust_display(fit_display(proc.name, 24), 24)} "
            f"{rjust_display(human_bytes(kib_to_bytes(proc.rss_kib)), 10)} "
            f"{rjust_display(pss_text, 10)} "
            f"{rjust_display(human_bytes(kib_to_bytes(proc.swap_kib)), 10)} "
            f"{rjust_display(human_bytes(kib_to_bytes(proc.vsize_kib)), 10)}"
        )

    print()
    print("说明:")
    print("  1) 进程 RSS 会重复计算共享页；PSS 更适合做总量记账。")
    if not pss_available:
        print("  2) 当前 atop 样本未记录 PSS；历史 raw 日志只有录制时启用 atop -R 才能保留 PSS。")
        print("  3) atop 回放时加 -R 不能为旧 raw 日志重新计算 PSS，因为历史进程的 smaps 已不存在。")
        print("  4) atop 的内核分项与进程分项属于不同观察维度，不能逐项简单相加。")
        print("  5) 可回收内存估计 = cache + buffers + slab_reclaimable。")
        print("  6) 不可回收已用估计 = 已用内存 - 可回收内存估计。")
    else:
        print("  2) atop 的内核分项与进程分项属于不同观察维度，不能逐项简单相加。")
        print("  3) 可回收内存估计 = cache + buffers + slab_reclaimable。")
        print("  4) 不可回收已用估计 = 已用内存 - 可回收内存估计。")


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    args = parse_args()
    raw_text = run_atop(args)
    samples = parse_samples(raw_text)
    sample = choose_sample(samples, args.sample_index)
    print_report(sample, args)


if __name__ == "__main__":
    main()
