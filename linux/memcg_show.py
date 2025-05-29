#!/usr/bin/env python3

import os
import re
import csv
import sys

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

class Htable:
    def __init__(self, name, dict_value):
        self.name = name
        self.dict_value = dict_value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def print_csv(self, ident=0):
        writer = csv.writer(sys.stdout)  # 直接输出到控制台

        # 准备当前行数据
        row_data = [ident * " " + self.name]  # 节点名（带缩进）
        for item in items:
            row_data.append(self.dict_value.get(item.show_name, ""))

        # 写入 CSV 行（自动处理特殊字符）
        writer.writerow(row_data)

        # 递归处理子节点
        for child in self.children:
            child.print_csv(ident + 2)

if not is_cgroup_v2:
    items = [
    	Item('limit', 'memory.limit_in_bytes'),
    	Item('softlimit', 'memory.soft_limit_in_bytes'),
    	Item('usage', 'memory.usage_in_bytes'),
    	Item('cache', 'memory.stat', 'cache'),
    	Item('rss', 'memory.stat', 'rss')
    ]
else:
    items = [
    	Item('cpu.max', 'cpu.max'),
    	Item('cpu.weigh', 'cpu.weigh'),
    	Item('cpuset.cpus', 'cpuset.cpus'),
    	Item('cpuset.mems', 'cpuset.meme'),
    	Item('memory.max', 'memory.max'),
    	Item('memory.high','memory.high'),
    ]

def conver_num_to_readable(bytes):
    if bytes == "max":
        return "max"

    if bytes == '':
        return '<empty>'

    # handle none-numbers
    try:
        bytes = int(bytes)
    except ValueError:
        return bytes

    if bytes == 9223372036854771712:
        return "max"

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

def build_cgroup_one(current_dir, gname):
    result = {}
    for item in items:
        try:
            name = item.get_showname()
            value = item.get_field_vale(current_dir)
        except:
            value = ""

        result[name] = value
    return Htable(gname, result)

def build_cgroup_htable(parent_dir, gname, has_attribut = False):
    if has_attribut:
        parent = build_cgroup_one(parent_dir, gname)
    else:
        result = {}
        for item in items:
            name = item.get_showname()
            result[name] = name

            parent = Htable(gname, result)

    try:
    	entries = os.listdir(parent_dir)
    except FileNotFoundError:
    	print(f"Path {base_path} not found.")
    	return

    for entry in entries:
    	entry_path = os.path.join(parent_dir, entry)
    	if os.path.isdir(entry_path):
            child = build_cgroup_htable(entry_path, entry, True)
            parent.add_child(child)

    return parent

if __name__ == "__main__":
    if not is_cgroup_v2:
    	cgroup_dir = cgroup_dir + "memory/"

    htable = build_cgroup_htable(cgroup_dir, "<root>")
    htable.print_csv()
