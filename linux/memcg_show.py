#!/usr/bin/env python3

import os
import re

cgroup_dir = "/sys/fs/cgroup/"
is_cgroup_v2 = os.path.exists(os.path.join(cgroup_dir, "cgroup.controllers"))

class Item:
    def __init__(self, show_name, file, field_in_file = None):
       self.show_name = show_name
       self.file = file
       self.field_in_file = field_in_file

    def get_file_value(file):
    	with open(file, "r") as f:
            return f.read().strip()

    def get_field_in_file(file, field):
    	with open(file, "r") as f:
            for line in f:
                # 按空格拆分行
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] == field:
                    return parts[1]

    def get_field_vale(self, dir_path):
    	if self.field_in_file is None:
    		return Item.get_file_value(os.path.join(dir_path, self.file))
    	else:
    		return Item.get_field_in_file(os.path.join(dir_path, self.file), self.field_in_file)

    def get_showname(self):
    	return self.show_name;
        

if not is_cgroup_v2:
    items = [
    	Item('limit', 'limit_in_bytes'),
    	Item('usage', 'usage_in_bytes'),
    	Item('softlimit', 'soft_limit_in_bytes'),
    	Item('cache', 'memory.stat', 'cache'),
    	Item('rss', 'memory.stat', 'rss')
    ]
else:
    items = [
    	Item('max', 'memory.max'),
    	Item('high', 'memory.high'),
    	Item('low', 'memory.low'),
    	Item('min', 'memory.min'),
    	Item('file', 'memory.stat', 'file'),
    	Item('anon', 'memory.stat', 'anon')
    ]

def conver_num_to_readable(bytes):
    if bytes == "max":
        return "max"

    bytes = int(bytes)

    if bytes >= 1024 * 1024 * 1024:
        return f"{bytes/1024/1024/1024:.2f}G"

    if bytes >= 1024 * 1024:
        return f"{bytes/1024/1024:.2f}M"

    if bytes >= 1024:
        return f"{bytes/1024:.2f}K"

    return f"{bytes}"

def calculate_max_width(base_path, prefix=""):
    """计算最长的 prefix + entry 的宽度"""
    max_width = 0
    try:
        entries = os.listdir(base_path)
    except FileNotFoundError:
        return max_width

    for entry in entries:
        entry_path = os.path.join(base_path, entry)
        if os.path.isdir(entry_path):
            entry_width = len(prefix) + len(entry)
            max_width = max(max_width, entry_width)
            # 递归计算子目录的宽度
            max_width = max(max_width, calculate_max_width(entry_path, prefix + "  "))
    return max_width

def analyze_cgroup_memory(base_path, prefix, max_width, cgroup_v2):
    """递归分析 cgroup 目录，输出内存信息"""
    try:
        entries = os.listdir(base_path)
    except FileNotFoundError:
        print(f"Path {base_path} not found.")
        return

    for entry in entries:
        entry_path = os.path.join(base_path, entry)
        if os.path.isdir(entry_path):
            memory_info = {}
            for item in items:
                value = item.get_field_vale(entry_path)
                name = item.get_showname()

                memory_info[name] = value 

            # 打印 cgroup 信息
            entry = f"{prefix}{entry}"
            print(f"{entry:{max_width}} | ", end="")
            for key, value in memory_info.items():
                print(f" {conver_num_to_readable(value):10}", end="")
            print('')

            # 递归处理子目录
            analyze_cgroup_memory(entry_path, prefix + "  ", max_width, cgroup_v2)

if __name__ == "__main__":
    # 默认基路径
    max_width =  calculate_max_width(cgroup_dir)
    item = "group name"
    print(f"{item:{max_width}} | ", end = '')
    for item in items:
        name = item.get_showname()
        print(f" {name:10}", end = '')
    print('')

    if not is_cgroup_v2:
        cgroup_dir = cgroup_dir + "memory/"

    analyze_cgroup_memory(cgroup_dir, "", max_width, is_cgroup_v2)

