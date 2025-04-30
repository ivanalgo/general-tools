#!/usr/bin/env python3

import re
import os

class SchedDomain:
    def __init__(self, name, level, cpumask, stype):
        self.name = name
        self.level = level
        self.cpumask = cpumask
        self.type = stype

        self.load = 0

    def has_cpu(self, cpu):
        if (1 << cpu) & self.cpumask:
            return True

        return False

    def add_cpu_load(self, load):
        self.load += load

    def __eq__(self, other):
        return self.level == other.level and self.cpumask == other.cpumask

    def __hash__(self):
        return hash((self.cpumask, self.level))

    def __repr__(self):
        return f"domain (name: {self.name}, level: {self.level} cpumask: {self.cpumask}, load: {self.load})\n"

class CPU(SchedDomain):
    def __init__(self, cpu, level):
        super().__init__(f"cpu{cpu}", level, 1 << cpu, "CPU")

class DomainTree:
    def __init__(self):
        self.children = {}

    def add_child(self, sd):
        if sd in self.children:
            return self.children[sd]
        else:
            self.children[sd] = DomainTree()
            return self.children[sd]

    def add_cpu_load(self, cpu, load):
        for sd in self.children:
            if sd.has_cpu(cpu):
                sd.add_cpu_load(load)
                child = self.children[sd]
                return child.add_cpu_load(cpu, load)

    @staticmethod
    def print_ident(sd, ident):
        s = " " * ident
        s += str(sd)

        return s

    def __repr__(self):
        def _repr_helper(tree, ident):
            s = ""
            for k, v in tree.children.items():
                s += DomainTree.print_ident(k, ident)
                s += _repr_helper(v, ident + 2)
            return s
        return f"DomainTree:\n{_repr_helper(self, 2)}"

    def show_load(self):
        def _load_helper(tree, ident):
            for k, v in tree.children.items():
                print(" " * ident, k.name, k.type, k.load)
                _load_helper(v, ident + 2)

        for k, v in self.children.items():
            print(k.name, k.type, k.load)
            _load_helper(v, 2)

def get_sched_domain_names():
    base_path = "/proc/sys/kernel/sched_domain/cpu0"
    result = {}

    # 遍历 cpu0 下的所有 domain* 目录
    for entry in os.listdir(base_path):
        if entry.startswith("domain") and os.path.isdir(os.path.join(base_path, entry)):
            domain_name = entry  # 如 "domain0"
            name_file = os.path.join(base_path, entry, "name")

            # 读取 name 文件内容
            try:
                with open(name_file, 'r') as f:
                    content = f.read().strip()  # 读取并去除空白字符
                    result[domain_name] = content
            except IOError:
                result[domain_name] = None  # 如果读取失败，设为 None

    return result

domain_names = get_sched_domain_names()

def build_schedstat_hierarchy():
    sd_tree = DomainTree()

    with open("/proc/schedstat", "r") as file:
        prev_cpu = -1
        domain_list = []
        level = 0

        for line in file:
            if line.startswith("cpu"):
                match = re.match(r'cpu(\d+)', line)
                if not match:
                    print("Invalid cpu id in /proc/schedstat")
                    return None
                cpu_id = int(match.group(1))

                # meet a new cpu
                if len(domain_list) > 0:
                    cur = sd_tree
                    for d in domain_list:
                        cur = cur.add_child(d)

                # prepare for the next cpu
                domain_list = []
                level = 0
                domain_list.append(CPU(cpu_id, level))
            elif line.startswith("domain"):
                domainText = line.split()[0]
                cpumask = line.split()[1]
                match = re.match(r'domain(\d+)', domainText)
                if not match:
                    print("Invalid domain id in /proc/schedstat")
                    return None
                domain_id = int(match.group(1))
                cpumask = int(cpumask.replace(",", ""), 16)
                level = level + 1
                domain_list.insert(0, SchedDomain(f"domain{domain_id}", level, cpumask, domain_names[domainText]))

    return sd_tree

def get_cpu_ids():
    with open("/proc/cpuinfo", "r") as f:
        return [int(line.split(":")[1].strip()) for line in f if line.strip().startswith("processor")]


def get_cpu_cfs_loads():
    cfs_loads = {}
    cpu_ids = get_cpu_ids()
    # ensure all cfs has load with default as 0 if /proc/sched_debug doesn't contain its entry
    for cpu in cpu_ids:
        cfs_loads.setdefault(cpu, 0)

    with open("/proc/sched_debug", "r") as file:
        current_cfs = None
        for line in file:
            match = re.match(r"^cfs_rq\[(\d+)\]:/", line)
            if match:
                current_cfs = int(match.group(1))
                cfs_loads[current_cfs] = None
            elif current_cfs is not None:
                match = re.match(r"^\s+\.load\s+:\s+(\d+)", line)
                if match:
                    cfs_loads[current_cfs] = int(match.group(1))
                    current_cfs = None

    return cfs_loads


domain_tree = build_schedstat_hierarchy()
cfs_loads = get_cpu_cfs_loads()

# Add cpu load to the domain_tree
for cpu, load in cfs_loads.items():
    domain_tree.add_cpu_load(cpu, load)


domain_tree.show_load()
