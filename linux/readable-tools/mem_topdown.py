#!/usr/bin/env python3

"""
机器内存 topdown 分解。

设计目标：
1. 用统一的树结构表达“包含关系”。
2. 每个节点都可以通过“source 表”或“自定义函数”取值，便于扩展。
3. 新增数据来源时，只需要补一个 reader / resolver，或者补一个定义表项。
4. 同时提供：
   - 系统视角：内核专有内存、空闲/可回收内存。
    - 进程视角：列出所有进程，每进程一行输出 total / anon / stack / file / shmem；Java进程额外显示heap/metaspace/codecache/offheap细分。
   - 内核模块视角：列出所有内核模块，每模块一行输出占用大小。
   - slab 视角：列出所有 slab cache，每 cache 一行输出占用大小。

说明：
- 默认进程维度的总量使用 PSS 视角，shared 页会在多进程之间按比例分摊；可通过参数切换回 RSS。
- 为兼顾准确性与性能，total / anon / file / shmem 优先来自 /proc/<pid>/smaps_rollup，stack / heap 的细分来自 /proc/<pid>/smaps。
- Java进程内存细分通过jcmd获取，需JDK环境支持。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional


PROC_ROOT = "/proc"
MEMINFO_PATH = "/proc/meminfo"
SMAPS_HEADER_RE = re.compile(r"^[0-9a-fA-F]+-[0-9a-fA-F]+\s")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
SLAB_SYSFS_ROOT = "/sys/kernel/slab"
JCMD_PATH = shutil.which("jcmd")


def clamp_non_negative(value: int) -> int:
    return max(int(value), 0)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as file_obj:
        return file_obj.read()


def parse_meminfo(path: str = MEMINFO_PATH) -> Dict[str, int]:
    meminfo: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as file_obj:
        for line in file_obj:
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].rstrip(":")
            try:
                meminfo[key] = int(parts[1])
            except ValueError:
                continue
    return meminfo


def parse_status_file(path: str) -> Dict[str, object]:
    data: Dict[str, object] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as file_obj:
        for line in file_obj:
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            value = raw_value.strip()
            if value.endswith(" kB"):
                number = value.split()[0]
                try:
                    data[key] = int(number)
                    continue
                except ValueError:
                    pass
            data[key] = value
    return data


def parse_cmdline(path: str) -> str:
    try:
        raw = read_text(path).replace("\x00", " ").strip()
        return raw
    except OSError:
        return ""


def parse_smaps_vma_categories(path: str, value_key: str) -> Dict[str, int]:
    """
    只做最需要的 anon 子类细分：stack / heap。
    value_key 可取 Rss / Pss。
    其余 anon 统一通过 anon_total - stack - heap 推导。
    """
    stack_kb = 0
    heap_kb = 0
    current_name = ""
    target_prefix = f"{value_key}:"

    with open(path, "r", encoding="utf-8", errors="replace") as file_obj:
        for line in file_obj:
            if SMAPS_HEADER_RE.match(line):
                parts = line.rstrip("\n").split(None, 5)
                current_name = parts[5] if len(parts) >= 6 else ""
                continue

            if not line.startswith(target_prefix):
                continue

            pieces = line.split()
            if len(pieces) < 2:
                continue

            try:
                value_kb = int(pieces[1])
            except ValueError:
                continue

            if current_name.startswith("[stack"):
                stack_kb += value_kb
            elif current_name == "[heap]":
                heap_kb += value_kb

    return {
        "stack": stack_kb,
        "heap": heap_kb,
    }


def parse_smaps_rollup_pss(path: str) -> Dict[str, int]:
    data = parse_status_file(path)
    metrics: Dict[str, int] = {}

    for source_key, target_key in (
        ("Pss", "total_kb"),
        ("Pss_Anon", "anon"),
        ("Pss_File", "file"),
        ("Pss_Shmem", "shmem"),
    ):
        value = data.get(source_key)
        if isinstance(value, int):
            metrics[target_key] = clamp_non_negative(value)

    return metrics


def accounting_label(accounting_mode: str) -> str:
    return "PSS" if accounting_mode == "pss" else "RSS"


def get_java_memory_metrics(pid: int) -> Dict[str, int]:
    """
    尝试通过jcmd获取Java进程内存细分指标，返回单位为KB的字典。
    需JVM开启NMT(-XX:NativeMemoryTracking=summary)才能获取完整信息，否则仅返回heap/metaspace基础信息。
    """
    if JCMD_PATH is None:
        return {}

    metrics: Dict[str, int] = {}

    try:
        result = subprocess.run(
            [JCMD_PATH, str(pid), "VM.native_memory", "summary"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            output = result.stdout
            heap_match = re.search(r"Java Heap\s+\(reserved=\d+KB,\s*committed=(\d+)KB\)", output)
            if heap_match:
                metrics["java_heap_kb"] = int(heap_match.group(1))

            class_match = re.search(r"-\s+Class\s+\(reserved=\d+KB,\s*committed=(\d+)KB\)", output)
            if class_match:
                metrics["java_metaspace_kb"] = int(class_match.group(1))

            code_match = re.search(r"-\s+Code\s+\(reserved=\d+KB,\s*committed=(\d+)KB\)", output)
            if code_match:
                metrics["java_code_cache_kb"] = int(code_match.group(1))

            total_match = re.search(r"Total:\s+reserved=\d+KB,\s*committed=(\d+)KB", output)
            if total_match and "java_heap_kb" in metrics:
                total_committed = int(total_match.group(1))
                metrics["java_off_heap_kb"] = max(total_committed - metrics["java_heap_kb"], 0)
            return metrics
    except (subprocess.SubprocessError, OSError, ValueError):
        pass

    try:
        result = subprocess.run(
            [JCMD_PATH, str(pid), "GC.heap_info"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            output = result.stdout
            heap_match = re.search(r"heap\s+.*?total\s+(\d+)K,\s*used\s+\d+K", output)
            if heap_match:
                metrics["java_heap_kb"] = int(heap_match.group(1))

            metaspace_match = re.search(r"Metaspace\s+.*?committed\s*=\s*(\d+)KB?", output)
            if not metaspace_match:
                metaspace_match = re.search(r"Metaspace\s+.*?committed\s+(\d+)K", output)
            if metaspace_match:
                metrics["java_metaspace_kb"] = int(metaspace_match.group(1))
            return metrics
    except (subprocess.SubprocessError, OSError, ValueError):
        pass

    return metrics


def parse_proc_modules(path: str = "/proc/modules") -> List[Dict[str, object]]:
    modules: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as file_obj:
        for line in file_obj:
            parts = line.split()
            if len(parts) < 6:
                continue

            try:
                size_bytes = int(parts[1])
                ref_count = int(parts[2])
            except ValueError:
                continue

            used_by = [] if parts[3] == "-" else [item for item in parts[3].split(",") if item]
            modules.append(
                {
                    "name": parts[0],
                    "size_bytes": size_bytes,
                    "size_kb": clamp_non_negative((size_bytes + 1023) // 1024),
                    "ref_count": ref_count,
                    "used_by": used_by,
                    "state": parts[4],
                    "address": parts[5],
                }
            )

    modules.sort(key=lambda module: (-int(module["size_kb"]), str(module["name"])))
    return modules


def parse_slabinfo(path: str = "/proc/slabinfo") -> List[Dict[str, object]]:
    slabs: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as file_obj:
        for line in file_obj:
            stripped = line.strip()
            if not stripped or stripped.startswith("slabinfo -") or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 16:
                continue

            try:
                name = parts[0]
                active_objs = int(parts[1])
                num_objs = int(parts[2])
                objsize = int(parts[3])
                objperslab = int(parts[4])
                pagesperslab = int(parts[5])
                active_slabs = int(parts[13])
                num_slabs = int(parts[14])
            except ValueError:
                continue

            total_bytes = num_slabs * pagesperslab * PAGE_SIZE
            active_bytes = active_objs * objsize
            reclaim_account_path = os.path.join(SLAB_SYSFS_ROOT, name, "reclaim_account")
            reclaimable = False
            if os.path.exists(reclaim_account_path):
                try:
                    reclaimable = read_text(reclaim_account_path).strip() == "1"
                except OSError:
                    reclaimable = False

            active_kb = clamp_non_negative((active_bytes + 1023) // 1024)
            size_kb = clamp_non_negative((total_bytes + 1023) // 1024)
            if size_kb <= 0:
                continue

            slabs.append(
                {
                    "name": name,
                    "active_objs": active_objs,
                    "num_objs": num_objs,
                    "objsize": objsize,
                    "objperslab": objperslab,
                    "pagesperslab": pagesperslab,
                    "active_slabs": active_slabs,
                    "num_slabs": num_slabs,
                    "reclaimable": reclaimable,
                    "category": "reclaimable" if reclaimable else "unreclaimable",
                    "active_kb": active_kb,
                    "size_kb": size_kb,
                }
            )

    slabs.sort(key=lambda slab: (-int(slab["size_kb"]), str(slab["name"])))
    return slabs


@dataclass
class ProcessSnapshot:
    pid: int
    name: str
    command: str
    metrics: Dict[str, int]
    accounting_mode: str = "rss"

    @property
    def display_name(self) -> str:
        base = self.command if self.command else self.name
        return f"{base} [{self.pid}]"

    @property
    def total_kb(self) -> int:
        return self.metrics.get("total_kb", self.metrics.get("rss_total", 0))

    def to_row_dict(self) -> Dict[str, object]:
        return {
            "pid": self.pid,
            "name": self.name,
            "command": self.command,
            "display_name": self.display_name,
            "accounting_mode": self.accounting_mode,
            "total_kb": self.total_kb,
            "anon_kb": self.metrics.get("anon", 0),
            "stack_kb": self.metrics.get("stack", 0),
            "file_kb": self.metrics.get("file", 0),
            "shmem_kb": self.metrics.get("shmem", 0),
            "java_heap_kb": self.metrics.get("java_heap_kb"),
            "java_metaspace_kb": self.metrics.get("java_metaspace_kb"),
            "java_code_cache_kb": self.metrics.get("java_code_cache_kb"),
            "java_off_heap_kb": self.metrics.get("java_off_heap_kb"),
        }


@dataclass
class SystemSnapshot:
    meminfo: Dict[str, int]
    processes: List[ProcessSnapshot]
    modules: List[Dict[str, object]]
    slabs: List[Dict[str, object]]
    accounting_mode: str = "rss"
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SourceRef:
    namespace: str
    key: str


@dataclass
class MetricSpec:
    key: str
    label: str
    source: Optional[SourceRef] = None
    value_func: Optional[Callable[[SystemSnapshot, Optional[ProcessSnapshot]], int]] = None
    children: List["MetricSpec"] = field(default_factory=list)
    children_factory: Optional[Callable[[SystemSnapshot, Optional[ProcessSnapshot]], List["MetricSpec"]]] = None
    children_sum_func: Optional[Callable[[SystemSnapshot, Optional[ProcessSnapshot]], int]] = None
    description: str = ""
    bound_process: Optional[ProcessSnapshot] = None


@dataclass
class MetricNode:
    key: str
    label: str
    value_kb: int
    description: str = ""
    children: List["MetricNode"] = field(default_factory=list)
    children_sum_kb: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "value_kb": self.value_kb,
            "children_sum_kb": self.children_sum_kb,
            "description": self.description,
            "children": [child.to_dict() for child in self.children],
        }


def resolve_meminfo_value(snapshot: SystemSnapshot, _process: Optional[ProcessSnapshot], key: str) -> int:
    return clamp_non_negative(snapshot.meminfo.get(key, 0))


def resolve_process_metric(_snapshot: SystemSnapshot, process: Optional[ProcessSnapshot], key: str) -> int:
    if process is None:
        return 0
    return clamp_non_negative(process.metrics.get(key, 0))


SOURCE_RESOLVERS: Dict[str, Callable[[SystemSnapshot, Optional[ProcessSnapshot], str], int]] = {
    "meminfo": resolve_meminfo_value,
    "process_metric": resolve_process_metric,
}


def resolve_source(snapshot: SystemSnapshot, process: Optional[ProcessSnapshot], source: SourceRef) -> int:
    resolver = SOURCE_RESOLVERS.get(source.namespace)
    if resolver is None:
        raise KeyError(f"unknown source namespace: {source.namespace}")
    return clamp_non_negative(resolver(snapshot, process, source.key))


def build_metric_tree(
    spec: MetricSpec,
    snapshot: SystemSnapshot,
    inherited_process: Optional[ProcessSnapshot] = None,
) -> MetricNode:
    process = spec.bound_process or inherited_process
    child_specs = list(spec.children)
    if spec.children_factory is not None:
        child_specs.extend(spec.children_factory(snapshot, process))

    child_nodes = [build_metric_tree(child, snapshot, process) for child in child_specs]

    if spec.value_func is not None:
        value_kb = clamp_non_negative(spec.value_func(snapshot, process))
    elif spec.source is not None:
        value_kb = resolve_source(snapshot, process, spec.source)
    else:
        value_kb = sum(child.value_kb for child in child_nodes)

    if spec.children_sum_func is not None:
        children_sum_kb: Optional[int] = clamp_non_negative(spec.children_sum_func(snapshot, process))
    elif child_nodes:
        children_sum_kb = sum(child.value_kb for child in child_nodes)
    else:
        children_sum_kb = None

    return MetricNode(
        key=spec.key,
        label=spec.label,
        value_kb=value_kb,
        description=spec.description,
        children=child_nodes,
        children_sum_kb=children_sum_kb,
    )


def meminfo_leaf(key: str, field_name: Optional[str] = None, description: str = "") -> MetricSpec:
    return MetricSpec(
        key=key,
        label=key,
        source=SourceRef("meminfo", field_name or key),
        description=description,
    )


def process_leaf(key: str, field_name: Optional[str] = None, description: str = "") -> MetricSpec:
    return MetricSpec(
        key=key,
        label=key,
        source=SourceRef("process_metric", field_name or key),
        description=description,
    )


def build_process_metric_specs(snapshot: SystemSnapshot, _process: Optional[ProcessSnapshot]) -> List[MetricSpec]:
    return [
        MetricSpec(
            key=f"process_{proc.pid}",
            label=proc.display_name,
            value_func=lambda _snapshot, _proc, proc=proc: proc.total_kb,
            description=f"单个进程的 {accounting_label(proc.accounting_mode)}。",
        )
        for proc in snapshot.processes
    ]


def read_single_process(pid: int, include_smaps: bool = True, accounting_mode: str = "pss") -> Optional[ProcessSnapshot]:
    proc_dir = os.path.join(PROC_ROOT, str(pid))
    status_path = os.path.join(proc_dir, "status")
    cmdline_path = os.path.join(proc_dir, "cmdline")
    smaps_path = os.path.join(proc_dir, "smaps")
    smaps_rollup_path = os.path.join(proc_dir, "smaps_rollup")

    try:
        status = parse_status_file(status_path)
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None

    name = str(status.get("Name", str(pid)))
    command = parse_cmdline(cmdline_path)

    rss_anon = clamp_non_negative(int(status.get("RssAnon", 0)))
    rss_file = clamp_non_negative(int(status.get("RssFile", 0)))
    rss_shmem = clamp_non_negative(int(status.get("RssShmem", 0)))
    rss_total = clamp_non_negative(int(status.get("VmRSS", rss_anon + rss_file + rss_shmem)))

    metrics = {
        "rss_total": rss_total,
        "total_kb": rss_total,
        "anon": rss_anon,
        "file": rss_file,
        "shmem": rss_shmem,
        "stack": 0,
        "heap": 0,
        "other_anon": rss_anon,
        "swap": clamp_non_negative(int(status.get("VmSwap", 0))),
        "page_tables": clamp_non_negative(int(status.get("VmPTE", 0))) + clamp_non_negative(int(status.get("VmPMD", 0))),
        "hugetlb": clamp_non_negative(int(status.get("HugetlbPages", 0))),
    }

    detail_value_key = "Rss"

    if accounting_mode == "pss":
        try:
            rollup_metrics = parse_smaps_rollup_pss(smaps_rollup_path)
            required_keys = {"total_kb", "anon", "file", "shmem"}
            if required_keys.issubset(rollup_metrics):
                metrics.update({key: rollup_metrics[key] for key in required_keys})
                detail_value_key = "Pss"
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            pass

    if include_smaps:
        try:
            smaps = parse_smaps_vma_categories(smaps_path, detail_value_key)
            stack_kb = min(clamp_non_negative(smaps.get("stack", 0)), metrics["anon"])
            heap_kb = min(clamp_non_negative(smaps.get("heap", 0)), max(metrics["anon"] - stack_kb, 0))
            metrics["stack"] = stack_kb
            metrics["heap"] = heap_kb
            metrics["other_anon"] = max(metrics["anon"] - stack_kb - heap_kb, 0)
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            pass
    else:
        metrics["other_anon"] = metrics["anon"]

    if name == "java" or "java" in command.lower():
        java_metrics = get_java_memory_metrics(pid)
        metrics.update(java_metrics)

    return ProcessSnapshot(pid=pid, name=name, command=command, metrics=metrics, accounting_mode=accounting_mode)


def collect_processes(
    include_smaps: bool = True,
    include_zero_process: bool = False,
    accounting_mode: str = "pss",
) -> List[ProcessSnapshot]:
    processes: List[ProcessSnapshot] = []

    for entry in os.listdir(PROC_ROOT):
        if not entry.isdigit():
            continue

        process = read_single_process(int(entry), include_smaps=include_smaps, accounting_mode=accounting_mode)
        if process is None:
            continue

        if not include_zero_process and process.total_kb == 0:
            continue

        processes.append(process)

    processes.sort(key=lambda proc: (-proc.total_kb, proc.pid))
    return processes


def collect_snapshot(
    include_smaps: bool = True,
    include_zero_process: bool = False,
    accounting_mode: str = "pss",
) -> SystemSnapshot:
    notes: List[str] = []
    meminfo = parse_meminfo()
    try:
        modules = parse_proc_modules()
        notes.append("内核模块明细来自 /proc/modules，大小字段使用模块对象占用字节数。")
    except (FileNotFoundError, PermissionError, OSError):
        modules = []
        notes.append("无法读取 /proc/modules，内核模块明细将为空。")

    try:
        slabs = parse_slabinfo()
        notes.append("slab 明细来自 /proc/slabinfo，大小字段按 num_slabs * pagesperslab * PAGE_SIZE 计算。")
        notes.append("slab 明细按 /sys/kernel/slab/<name>/reclaim_account 分为 reclaimable / unreclaimable 两类。")
        notes.append("slab 明细总量更接近 meminfo 中的 Slab；其中可回收/不可回收总量仍分别以 SReclaimable / SUnreclaim 表示。")
    except (FileNotFoundError, PermissionError, OSError):
        slabs = []
        notes.append("无法读取 /proc/slabinfo，slab 明细将为空。")

    if accounting_mode == "pss":
        notes.append("进程 total/anon/file/shmem 默认使用 PSS 口径，优先读取 /proc/<pid>/smaps_rollup 中的 Pss/Pss_Anon/Pss_File/Pss_Shmem，以避免跨进程共享页重复计数；无法读取时会按进程退化为 RSS。")
        if include_smaps:
            notes.append("PSS 模式下仅为 [heap]/[stack] 细分扫描 /proc/<pid>/smaps 的 Pss 字段，以兼顾准确性与性能。")
        else:
            notes.append("已禁用 smaps，PSS 模式下 stack/heap 细分将为 0；但 total/anon/file/shmem 仍会优先读取 smaps_rollup。")
    else:
        notes.append("进程 total/anon/file/shmem 使用 RSS 口径；shared 页在多进程求和时可能重复计数。")
        if include_smaps:
            notes.append("RSS 模式下 [heap]/[stack] 细分使用 /proc/<pid>/smaps 的 Rss 字段。")
        else:
            notes.append("已禁用 smaps，RSS 模式下 stack/heap 细分将为 0。")

    if JCMD_PATH is not None:
        notes.append("Java进程内存细分通过jcmd获取：JVM启动时添加-XX:NativeMemoryTracking=summary可获取完整heap/metaspace/codecache/offheap细分，未开启NMT时仅能获取heap/metaspace基础信息，权限不足时自动降级。")
    else:
        notes.append("未检测到jcmd命令，Java进程内存细分功能不可用。")

    processes = collect_processes(
        include_smaps=include_smaps,
        include_zero_process=include_zero_process,
        accounting_mode=accounting_mode,
    )

    return SystemSnapshot(
        meminfo=meminfo,
        processes=processes,
        modules=modules,
        slabs=slabs,
        accounting_mode=accounting_mode,
        notes=notes,
    )


def build_spec_root(accounting_mode: str) -> MetricSpec:
    process_accounting = accounting_label(accounting_mode)
    kernel_dedicated = MetricSpec(
        key="kernel_dedicated",
        label="内核专有内存",
        description="不含 page cache / buffers 等用户可回收页，只统计内核自身持有。",
        children=[
            meminfo_leaf("KernelStack", description="内核线程/进程内核栈。"),
            meminfo_leaf("PageTables", description="页表页。"),
            meminfo_leaf("SecPageTables", description="二级页表，如 IOMMU 页表。"),
            meminfo_leaf("Percpu", description="per-cpu 内存。"),
            meminfo_leaf("SUnreclaim", description="不可回收 slab。"),
            MetricSpec(
                key="kernel_modules",
                label="KernelModules",
                source=SourceRef("meminfo", "KernelModules"),
                children_sum_func=lambda snapshot, _proc: sum(int(module["size_kb"]) for module in snapshot.modules),
                description="数值来自 meminfo 的 KernelModules；子项合计为 /proc/modules 中所有模块对象大小汇总。",
            ),
            meminfo_leaf("Hugetlb", description="hugetlb 预留页。"),
        ],
    )

    free_reclaimable = MetricSpec(
        key="free_reclaimable",
        label="空闲与可回收",
        description="free + cache + buffers + reclaimable slab。",
        children=[
            meminfo_leaf("MemFree", description="完全空闲页。"),
            meminfo_leaf("Buffers", description="buffer cache。"),
            meminfo_leaf("Cached", description="page cache。"),
            meminfo_leaf("SReclaimable", description="可回收 slab。"),
        ],
    )

    processes = MetricSpec(
        key="processes",
        label=f"进程占用({process_accounting}汇总)",
        value_func=lambda snapshot, _proc: sum(proc.total_kb for proc in snapshot.processes),
        children_sum_func=lambda snapshot, _proc: sum(proc.total_kb for proc in snapshot.processes),
        description=(
            f"数值为所有进程 {process_accounting} 汇总；子项合计同样为逐进程 {process_accounting} 加总。"
            + ("PSS 会对共享页按 mapcount 比例分摊。" if accounting_mode == "pss" else "RSS 在跨进程求和时可能重复计数共享页。")
        ),
    )

    return MetricSpec(
        key="MemTotal",
        label="MemTotal",
        source=SourceRef("meminfo", "MemTotal"),
        description="系统物理内存总量。下面同时给出系统视角与进程视角。",
        children=[kernel_dedicated, processes, free_reclaimable],
    )


def format_size(value_kb: int, unit: str) -> str:
    if unit == "kb":
        return f"{value_kb:,} kB"
    if unit == "mb":
        return f"{value_kb / 1024:,.2f} MB"
    if unit == "gb":
        return f"{value_kb / 1024 / 1024:,.2f} GB"
    raise ValueError(f"unknown unit: {unit}")


def display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def pad_display(text: str, width: int, align: str = "left") -> str:
    padding = max(width - display_width(text), 0)
    if align == "right":
        return " " * padding + text
    return text + " " * padding


def collect_tree_rows(
    node: MetricNode,
    unit: str,
    depth: int = 0,
    parent_value: Optional[int] = None,
) -> List[Dict[str, str]]:
    indent = "  " * depth
    if parent_value and parent_value > 0:
        ratio = f"{node.value_kb * 100.0 / parent_value:6.2f}%"
    else:
        ratio = "-"

    children_sum = format_size(node.children_sum_kb, unit) if node.children_sum_kb is not None else "-"

    rows = [
        {
            "label": f"{indent}{node.label}",
            "value": format_size(node.value_kb, unit),
            "children_sum": children_sum,
            "ratio": ratio,
        }
    ]
    for child in node.children:
        rows.extend(collect_tree_rows(child, unit, depth + 1, node.value_kb))
    return rows


def render_tree_lines(node: MetricNode, unit: str) -> List[str]:
    rows = collect_tree_rows(node, unit)
    label_width = max(display_width("名称"), max(display_width(row["label"]) for row in rows))
    value_width = max(display_width("数值"), max(display_width(row["value"]) for row in rows))
    children_sum_width = max(display_width("子项合计"), max(display_width(row["children_sum"]) for row in rows))
    ratio_width = max(display_width("占父节点比例"), max(display_width(row["ratio"]) for row in rows))

    header = "  ".join(
        [
            pad_display("名称", label_width, "left"),
            pad_display("数值", value_width, "right"),
            pad_display("子项合计", children_sum_width, "right"),
            pad_display("占父节点比例", ratio_width, "right"),
        ]
    )
    separator = "-" * (label_width + 2 + value_width + 2 + children_sum_width + 2 + ratio_width)
    lines = [header, separator]
    for row in rows:
        lines.append(
            "  ".join(
                [
                    pad_display(row["label"], label_width, "left"),
                    pad_display(row["value"], value_width, "right"),
                    pad_display(row["children_sum"], children_sum_width, "right"),
                    pad_display(row["ratio"], ratio_width, "right"),
                ]
            )
        )
    return lines


def print_tree(node: MetricNode, unit: str) -> None:
    for line in render_tree_lines(node, unit):
        print(line)


def truncate_text(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def render_process_table(snapshot: SystemSnapshot, unit: str) -> List[str]:
    suffix = accounting_label(snapshot.accounting_mode)
    total_header = f"TOTAL_{suffix}"
    anon_header = f"ANON_{suffix}"
    stack_header = f"STACK_{suffix}"
    file_header = f"FILE_{suffix}"
    shmem_header = f"SHMEM_{suffix}"
    has_java = any(proc.metrics.get("java_heap_kb") is not None for proc in snapshot.processes)
    if has_java:
        header = (
            f"{'PID':>8}  {'COMM':<32}  {total_header:>12}  {anon_header:>12}  {stack_header:>12}  {file_header:>12}  {shmem_header:>12}  "
            f"{'JAVA_HEAP':>12}  {'JAVA_META':>12}  {'JAVA_CODE':>12}  {'JAVA_OFFHEAP':>12}"
        )
    else:
        header = (
            f"{'PID':>8}  {'COMM':<32}  {total_header:>12}  {anon_header:>12}  {stack_header:>12}  {file_header:>12}  {shmem_header:>12}"
        )
    lines = [header, "-" * len(header)]

    for process in snapshot.processes:
        name = process.command if process.command else process.name
        name = truncate_text(name, 32)
        base_line = (
            f"{process.pid:>8}  "
            f"{name:<32}  "
            f"{format_size(process.total_kb, unit):>12}  "
            f"{format_size(process.metrics.get('anon', 0), unit):>12}  "
            f"{format_size(process.metrics.get('stack', 0), unit):>12}  "
            f"{format_size(process.metrics.get('file', 0), unit):>12}  "
            f"{format_size(process.metrics.get('shmem', 0), unit):>12}"
        )
        if has_java:
            def fmt_java(key: str) -> str:
                val = process.metrics.get(key)
                return format_size(val, unit) if val is not None else "-"

            java_cols = (
                f"  {fmt_java('java_heap_kb'):>12}  "
                f"{fmt_java('java_metaspace_kb'):>12}  "
                f"{fmt_java('java_code_cache_kb'):>12}  "
                f"{fmt_java('java_off_heap_kb'):>12}"
            )
            base_line += java_cols
        lines.append(base_line)

    return lines


def print_process_table(snapshot: SystemSnapshot, unit: str) -> None:
    print(f"\n进程明细(每进程一行, {accounting_label(snapshot.accounting_mode)}口径):")
    for line in render_process_table(snapshot, unit):
        print(line)


def render_module_table(snapshot: SystemSnapshot, unit: str) -> List[str]:
    header = f"{'MODULE':<32}  {'SIZE':>12}"
    lines = [header, "-" * len(header)]

    for module in snapshot.modules:
        lines.append(
            f"{truncate_text(str(module['name']), 32):<32}  "
            f"{format_size(int(module['size_kb']), unit):>12}"
        )

    return lines


def print_module_table(snapshot: SystemSnapshot, unit: str) -> None:
    print("\n内核模块明细(每模块一行):")
    for line in render_module_table(snapshot, unit):
        print(line)


def render_slab_table(slabs: List[Dict[str, object]], unit: str) -> List[str]:
    header = f"{'SLAB':<32}  {'SIZE':>12}"
    lines = [header, "-" * len(header)]

    for slab in slabs:
        lines.append(
            f"{truncate_text(str(slab['name']), 32):<32}  "
            f"{format_size(int(slab['size_kb']), unit):>12}"
        )

    return lines


def print_slab_table(snapshot: SystemSnapshot, unit: str) -> None:
    reclaimable_slabs = [slab for slab in snapshot.slabs if bool(slab.get("reclaimable", False))]
    unreclaimable_slabs = [slab for slab in snapshot.slabs if not bool(slab.get("reclaimable", False))]

    print("\nreclaimable slab 明细(每 cache 一行):")
    for line in render_slab_table(reclaimable_slabs, unit):
        print(line)

    print("\nunreclaimable slab 明细(每 cache 一行):")
    for line in render_slab_table(unreclaimable_slabs, unit):
        print(line)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="结构化输出机器内存 topdown 分解")
    parser.add_argument(
        "--accounting",
        choices=["pss", "rss"],
        default="pss",
        help="进程内存核算口径：pss(默认，适合整机汇总) 或 rss(适合看单进程resident)",
    )
    parser.add_argument(
        "--unit",
        choices=["kb", "mb", "gb"],
        default="mb",
        help="输出单位，默认 mb",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出树结构，便于程序消费",
    )
    parser.add_argument(
        "--no-smaps",
        action="store_true",
        help="不读取 /proc/<pid>/smaps；stack/heap 细分将不可用，pss 模式下 total/anon/file/shmem 仍会优先读取 smaps_rollup",
    )
    parser.add_argument(
        "--show-zero-processes",
        action="store_true",
        help="是否显示当前核算口径下 total 为 0 的进程",
    )
    parser.add_argument(
        "-p",
        "--processes",
        action="store_true",
        help="输出进程明细表",
    )
    parser.add_argument(
        "-m",
        "--modules",
        action="store_true",
        help="输出内核模块明细表",
    )
    parser.add_argument(
        "-s",
        "--slabs",
        action="store_true",
        help="输出 slab 明细表",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="输出总表、全部明细表和说明",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    snapshot = collect_snapshot(
        include_smaps=not args.no_smaps,
        include_zero_process=args.show_zero_processes,
        accounting_mode=args.accounting,
    )
    root_spec = build_spec_root(snapshot.accounting_mode)
    root_node = build_metric_tree(root_spec, snapshot)

    if args.json:
        payload = {
            "accounting_mode": snapshot.accounting_mode,
            "notes": snapshot.notes,
            "tree": root_node.to_dict(),
            "processes": [process.to_row_dict() for process in snapshot.processes],
            "kernel_modules": snapshot.modules,
            "slabs": snapshot.slabs,
        }
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print_tree(root_node, args.unit)
        show_processes = args.all or args.processes
        show_modules = args.all or args.modules
        show_slabs = args.all or args.slabs

        if show_modules:
            print_module_table(snapshot, args.unit)
        if show_slabs:
            print_slab_table(snapshot, args.unit)
        if show_processes:
            print_process_table(snapshot, args.unit)
        if args.all and snapshot.notes:
            print("\n说明:")
            for note in snapshot.notes:
                print(f"- {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
