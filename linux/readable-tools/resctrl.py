#!/usr/bin/env python3
import os
import sys
import time

INTERVAL = 1.0
MIN_MB = 1.0  # 小于 1 MB/s 不显示

def discover_groups(root):
    groups = []
    if os.path.isdir(os.path.join(root, "mon_data")):
        groups.append((root, os.path.basename(root)))
        return groups

    for name in os.listdir(root):
        g = os.path.join(root, name)
        if not os.path.isdir(g):
            continue
        if not os.path.isdir(os.path.join(g, "mon_data")):
            continue
        groups.append((g, name))
    return groups


def read_int(path):
    try:
        with open(path, "r") as f:
            v = f.read().strip()
        if v == "Unavailable":
            return None
        return int(v)
    except Exception:
        return None


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <resctrl_mon_groups_or_group>")
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print("Invalid path")
        sys.exit(1)

    groups = discover_groups(root)
    if not groups:
        print("No valid resctrl groups found")
        sys.exit(1)

    print(f"Discovered {len(groups)} groups")

    last_local = {}
    last_total = {}
    last_time = {}

    while True:
        now = time.time()
        print(time.strftime("[%F %T]"))

        sum_local = 0.0
        sum_total = 0.0

        for gpath, gname in groups:
            mon_data = os.path.join(gpath, "mon_data")
            try:
                domains = os.listdir(mon_data)
            except Exception:
                continue

            for dname in domains:
                if not dname.startswith("mon_L3_"):
                    continue

                dpath = os.path.join(mon_data, dname)
                key = f"{gname}/{dname}"

                local_v = read_int(os.path.join(dpath, "mbm_local_bytes"))
                total_v = read_int(os.path.join(dpath, "mbm_total_bytes"))

                if local_v is None or total_v is None:
                    continue

                if key in last_time:
                    dt = now - last_time[key]
                    if dt <= 0:
                        continue

                    dl = local_v - last_local[key]
                    dtot = total_v - last_total[key]

                    local_bw = dl / dt / 1024 / 1024
                    total_bw = dtot / dt / 1024 / 1024

                    if local_bw >= MIN_MB or total_bw >= MIN_MB:
                        print(f"{gname:<12} {dname:<10} "
                              f"local: {local_bw:8.2f} MB/s  "
                              f"total: {total_bw:8.2f} MB/s")

                        sum_local += local_bw
                        sum_total += total_bw

                last_local[key] = local_v
                last_total[key] = total_v
                last_time[key] = now

        print(f"{'SUM':<24} local: {sum_local:8.2f} MB/s  "
              f"total: {sum_total:8.2f} MB/s\n")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()

