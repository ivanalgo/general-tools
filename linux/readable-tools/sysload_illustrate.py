#!/usr/bin/python

import time

def read_run_block_proc():
    run = 0
    block = 0

    for line in open("/proc/stat", "r"):
        if line.startswith("procs_running"):
            run = int(line.split()[1])
        if line.startswith("procs_blocked"):
            block = int(line.split()[1])

    return (run, block)

def read_sysload():
    load1m = 0.0
    load5m = 0.0
    load15m = 0.0

    f = open("/proc/loadavg", "r")
    line = f.readline()
    tokens = line.split()

    load1m = float(tokens[0])
    load5m = float(tokens[1])
    load15m = float(tokens[2])

    return (load1m, load5m, load15m)


load1m = 0.0
load5m = 0.0
load15m = 0.0

(load1m, load5m, load15m) = read_sysload()
while True:
    # sleep 5 seconds and 10 jiffes
    time.sleep(5.01)
    (run, block) = read_run_block_proc()
    (sys1, sys5, sys15) = read_sysload()

    # according kernel's rule to calculate the load step by step
    load1m = load1m * 1884 / 2048 + (run + block) * (2048 - 1884) / 2048
    load5m = load5m * 2014 / 2048 + (run + block) * (2048 - 2014) / 2048
    load15m = load15m * 2037 / 2048 + (run + block) * (2048 - 2037) / 2048

    print("run: %4d, block: %4d ;   sysload: %6.2f  %6.2f  %6.2f ;     calc load: %6.2f  %6.2f  %6.2f" %
              (run, block, sys1, sys5, sys15, load1m, load5m, load15m))
