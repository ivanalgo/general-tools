#!/usr/bin/env python3

import argparse

def set_value(_dict, name, value):
    if _dict[name] == 0:
        _dict[name] = value * 4096  # page size in bytes

def parse_zoneinfo():
    water_mark = {}
    with open("/proc/zoneinfo", "r") as f:
        node_id = None
        zone_name = None

        for line in f:
            line = line.strip()
            if line.startswith("Node"):
                parts = line.split(',')
                node_id = int(parts[0].split()[1])
                zone_name = parts[1].split()[1]

                water_mark[(node_id, zone_name)] = {
                    "min": 0,
                    "low": 0,
                    "high": 0,
                    "free": 0,
                }

            elif line.startswith("pages free"):
                free_val = int(line.split()[2])
                set_value(water_mark[(node_id, zone_name)], 'free', free_val)
            elif line.startswith("min"):
                min_val = int(line.split()[1])
                set_value(water_mark[(node_id, zone_name)], 'min',  min_val)
            elif line.startswith("low"):
                low_val = int(line.split()[1])
                set_value(water_mark[(node_id, zone_name)], 'low', low_val)
            elif line.startswith("high"):
                high_val = int(line.split()[1])
                set_value(water_mark[(node_id, zone_name)], 'high', high_val)

    return water_mark

def format_value(value, unit):
    if unit == "k":
        return value // 1024
    elif unit == "m":
        return value // (1024 ** 2)
    elif unit == "g":
        return value // (1024 ** 3)
    else:
        return value  # fallback, should not happen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show zone watermarks from /proc/zoneinfo")
    parser.add_argument("-k", action="store_const", dest="unit", const="k", help="Display in KB")
    parser.add_argument("-m", action="store_const", dest="unit", const="m", help="Display in MB")
    parser.add_argument("-g", action="store_const", dest="unit", const="g", help="Display in GB")
    args = parser.parse_args()
    unit = args.unit or "k"  # default to KB

    water_mark = parse_zoneinfo()

    print("{:<6} {:<10} {:>16} {:>16} {:>16} {:>16}".format(
        "Node", "Zone", f"min({unit.upper()})", f"low({unit.upper()})", f"high({unit.upper()})", f"free({unit.upper()})"
    ))
    print("-" * 100)
    for (node_id, zone_name), values in water_mark.items():
        print("{:<6} {:<10} {:>16} {:>16} {:>16} {:>16}".format(
            node_id, zone_name,
            format_value(values["min"], unit),
            format_value(values["low"], unit),
            format_value(values["high"], unit),
            format_value(values["free"], unit)
        ))

