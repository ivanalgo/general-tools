#!/usr/bin/env python3

import re
import argparse

# 解析 meminfo 文件，生成最基本的原始数据
def parse_meminfo():
    meminfo = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, val, *_ = line.strip().split()
            meminfo[key.rstrip(":")] = int(val)
    return meminfo

# 树结构定义，每个节点可以有计算函数或自动聚合子节点
class MemTreeNode:
    def __init__(self, name, calc_func=None):
        self.name = name
        self.calc_func = calc_func  # 可以为 lambda meminfo: ...
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        return child

    def evaluate(self, meminfo):
        if self.calc_func:
            return self.calc_func(meminfo)
        return sum(child.evaluate(meminfo) for child in self.children)

def collect_tree_lines(node, meminfo, indent=0):
    """收集所有树节点的 (prefix_text, value) 元组"""
    value = node.evaluate(meminfo)
    name_prefix = " " * indent + node.name
    lines = [(name_prefix, value)]
    for child in node.children:
        lines.extend(collect_tree_lines(child, meminfo, indent + 2))
    return lines

def print_tree(root, meminfo, unit="kB"):
    lines = collect_tree_lines(root, meminfo)
    max_name_len = max(len(name) for name, _ in lines)

    if unit == "MB":
        scale = 1 / 1024
        suffix = " MB"
    elif unit == "GB":
        scale = 1 / (1024 * 1024)
        suffix = " GB"
    else:
        scale = 1
        suffix = " kB"

    for name, value in lines:
        converted = value * scale
        if scale == 1:
            display = f"{int(converted):>10}"
        else:
            display = f"{converted:>10.2f}"
        print(f"{name:<{max_name_len}} : {display}{suffix}")

def build_memory_tree():
    root = MemTreeNode("MemTotal", lambda m: m["MemTotal"])

    free = MemTreeNode("MemFree", lambda m: m["MemFree"])
    available = MemTreeNode("MemAvailable", lambda m: m["MemAvailable"])
    root.add_child(free)
    root.add_child(available)

    cached = MemTreeNode("Cached", lambda m: m["Cached"])
    cached.add_child(MemTreeNode("Dirty", lambda m: m["Dirty"]))
    cached.add_child(MemTreeNode("Writeback", lambda m: m["Writeback"]))

    Shmem = MemTreeNode("Shmem", lambda m: m["Shmem"])
    Shmem.add_child(MemTreeNode("ShmemHugePages", lambda m: m.get("ShmemHugePages", 0)))

    Slab = MemTreeNode("Slab", lambda m: m["Slab"])
    Slab.add_child(MemTreeNode("SReclaimable", lambda m: m["SReclaimable"]))
    Slab.add_child(MemTreeNode("SUnreclaim", lambda m: m["SUnreclaim"]))

    Kstack = MemTreeNode("KernelStack", lambda m: m["KernelStack"])
    Pagetable = MemTreeNode("PageTables", lambda m: m["PageTables"])
    Buffer = MemTreeNode("Buffers", lambda m: m["Buffers"])

    Anon = MemTreeNode("AnonPages", lambda m: m["AnonPages"])
    Anon.add_child(MemTreeNode("AnonHugePages", lambda m: m["AnonHugePages"]))

    root.add_child(Anon)
    root.add_child(cached)
    root.add_child(Shmem)
    root.add_child(Slab)
    root.add_child(Kstack)
    root.add_child(Buffer)

    return root

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display /proc/meminfo as a structured memory tree.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-m", "--megabytes", action="store_true", help="Display memory in MB")
    group.add_argument("-g", "--gigabytes", action="store_true", help="Display memory in GB")
    args = parser.parse_args()

    if args.megabytes:
        unit = "MB"
    elif args.gigabytes:
        unit = "GB"
    else:
        unit = "kB"

    meminfo_dict = parse_meminfo()
    tree = build_memory_tree()
    print_tree(tree, meminfo_dict, unit=unit)

