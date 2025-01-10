import os
import re

cgroup_dir = "/sys/fs/cgroup/memory/"
is_cgroup_v2 = os.path.exists(os.path.join(cgroup_dir, "cgroup.controllers"))

if not is_cgroup_v2:
    files = ['limit_in_bytes', 'usage_in_bytes', 'soft_limit_in_bytes']
    names = ['limit', 'usage', 'softlimit', 'cache', 'rss']
else:
    files = ['max', "high", "low", "min", "current"]
    names = ['max', 'high', 'low', 'min', 'usage', 'cache', 'rss']

def parse_memory_stat_file(file_path):
    """
    解析 memory.stat 文件，提取 'cache' 和 'rss' 数据。

    参数:
        file_path (str): memory.stat 文件的路径。

    返回:
        dict: 包含 'cache' 和 'rss' 数据的字典，例如:
              {"cache": 123456, "rss": 789012}
    """
    result = {"cache": None, "rss": None}
    
    try:
        with open(file_path, "r") as file:
            for line in file:
                # 按空格拆分行
                parts = line.strip().split()
                if len(parts) == 2:
                    key, value = parts
                    if key in result:
                        result[key] = int(value)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except ValueError as e:
        print(f"Error: Failed to parse file {file_path}: {e}")
    
    return result

def parse_memory_info_v1(cgroup_path):
    """解析 cgroup v1 的内存信息"""
    memory_stat = {}
    #files = ['limit_in_bytes', 'usage_in_bytes', 'soft_limit_in_bytes']
    #names = ['limit', 'usage', 'softlimit']

    for key in range(len(files)):
        try:
            with open(os.path.join(cgroup_path, f"memory.{files[key]}"), "r") as f:
                memory_stat[names[key]] = int(f.read().strip())
        except FileNotFoundError:
            memory_stat[key] = None
    return memory_stat

def parse_memory_info_v2(cgroup_path):
    """解析 cgroup v2 的内存信息"""
    memory_stat = {}
    #files = ['max', "high", "low", "min", "current"]
    #names = ['max', 'high', 'low', 'min', 'usage']

    for key in range(len(files)):
        try:
            with open(os.path.join(cgroup_path, f"memory.{files[key]}"), "r") as f:
                memory_stat[names[key]] = int(f.read().strip())
        except FileNotFoundError:
            memory_stat[key] = None
    return memory_stat

def conver_num_to_readable(bytes):
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

def analyze_cgroup_memory(base_path, prefix, max_width):
    """递归分析 cgroup 目录，输出内存信息"""
    try:
        entries = os.listdir(base_path)
    except FileNotFoundError:
        print(f"Path {base_path} not found.")
        return

    is_cgroup_v2 = os.path.exists(os.path.join(base_path, "cgroup.controllers"))
    for entry in entries:
        entry_path = os.path.join(base_path, entry)
        if os.path.isdir(entry_path):
            # 读取 cgroup 的内存信息
            if is_cgroup_v2:
                memory_info = parse_memory_info_v2(entry_path)
            else:
                memory_info = parse_memory_info_v1(entry_path)

            memory_stat = parse_memory_stat_file(entry_path + "/memory.stat")
            # Append memoyr_stat to memory_info
            for key, value in memory_stat.items():
                memory_info[key] = value

            # 打印 cgroup 信息
            entry = f"{prefix}{entry}"
            print(f"{entry:{max_width}} | ", end="")
            for key, value in memory_info.items():
                print(f" {conver_num_to_readable(value):10}", end="")
            print('')

            # 递归处理子目录
            analyze_cgroup_memory(entry_path, prefix + "  ", max_width)

if __name__ == "__main__":
    # 默认基路径
    max_width =  calculate_max_width(cgroup_dir, "")
    item = "name"
    print(f"{item:{max_width}} | ", end = '')
    for name in names:
        print(f" {name:10}", end = '')
    print('')
    analyze_cgroup_memory(cgroup_dir, "", max_width)

