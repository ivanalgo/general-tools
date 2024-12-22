#!/usr/bin/env python3

import re


def get_cpu_ids():
    with open("/proc/cpuinfo", "r") as f:
        return [int(line.split(":")[1].strip()) for line in f if line.strip().startswith("processor")]


def parse_bitmap(bitmap_str):
    # remove ',' and covert hex to dec
    return [i for i in range(1024) if (int(bitmap_str.replace(',', ''), 16) >> i) & 1]


def read_schedstat_and_parse():
    sched_domains = {}
    with open("/proc/schedstat", "r") as file:
        for line in file:
            for domain in ['domain0', 'domain1', 'domain2']:
                if line.startswith(domain):
                    bitmap_str = line.split()[1]
                    set_bits = parse_bitmap(bitmap_str)
                    sched_domains.setdefault(domain, set()).add(tuple(set_bits))
    return sched_domains


# get all cpu ids
cpu_ids = get_cpu_ids()

# read cfs load for all CPU(s)
cfs_loads = {}
with open("/proc/sched_debug", "r") as file:
    current_cfs = None
    for line in file:
        if (match := re.match(r"^cfs_rq\[(\d+)\]:/", line)):
            current_cfs = int(match.group(1))
            cfs_loads[current_cfs] = None
        elif current_cfs is not None and (match := re.match(r"^\s+\.load\s+:\s+(\d+)", line)):
            cfs_loads[current_cfs] = int(match.group(1))
            current_cfs = None

# ensure all cfs has load with default as 0 if /proc/sched_debug doesn't contain its entry
for cpu in cpu_ids:
    cfs_loads.setdefault(cpu, 0)

# Parse /proc/schedstat and generate sched domain information
sched_domains = read_schedstat_and_parse()

# output load of sched domains in each level
for domain, bit_sets in sched_domains.items():
    print(f"{domain}", end=" ")
    for bit_set in bit_sets:
        load_sum = sum(cfs_loads.get(bit, 0) for bit in bit_set)
        print(f"{load_sum} ", end="")
    print()

# output load of CPU(s) which are organized by sched domain
for bit_sets in sched_domains["domain1"]:
    print("{", end="")
    printed_list = []
    for bit in bit_sets:
        if bit not in printed_list:
            for domain0_bit_sets in sched_domains["domain0"]:
                if bit in domain0_bit_sets:
                    print("(", end="")
                    for d0_bit in domain0_bit_sets:
                        print(f"{cfs_loads.get(d0_bit, 0):3}", end=" ")
                        printed_list.append(d0_bit)
                    print(")", end="")
    print("}")
