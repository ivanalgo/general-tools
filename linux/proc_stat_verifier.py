#!/usr/bin/env python3

#
# As we know, linux diagnostic tools such as top and atop use /proc/stat pseudo file to
# reprot the CPU usage for the whole system.
# But we met a problem that /proc/stat does not always report the accurate CPU
# usage for each CPU. So I write the program the verify the correctness of /proc/stat
# file.
#
# Layout of /proc/stat for the part of CPU usage.
#
# cpu  34932333723 14885525220 14043387310 462769183392 103874067 0 191913423 0 0 0
# cpu0 759208772 369572667 352158538 7789583400 2504737 0 5513162 0 0 0
# cpu1 691994174 350285959 326233847 7996648891 2483028 0 4590576 0 0 0
# cpu2 727777199 334772867 338500088 7978292073 2382235 0 4814811 0 0 0
# cpu3 738631535 338167834 348899513 7956051058 2386989 0 5105868 0 0 0
# cpu4 737611715 337044965 351556453 7956195981 2388532 0 5251331 0 0 0
# cpu5 711146154 338943196 336672211 7996663252 2379988 0 5334267 0 0 0
#
# field 0: "cpu" means this line records the cpu time of all CPU(s),  while "cpu<n>" means
#          this line records the cpu time for CPU-n
# field 1:  user        time in the unit of 10ms
# field 2:  nice        time in the unit of 10ms
# field 3:  system      time in the unit of 10ms
# field 4:  idle        time in the unit of 10ms
# field 5:  iowait      time in the unit of 10ms
# field 6:  irq         time in the unit of 10ms
# field 7:  softirq     time in the unit of 10ms
# field 8:  steal       time in the unit of 10ms
# field 9:  guest       time in the unit of 10ms
# field 10: guest_nice  time in the unit of 10ms
#
# Note:
#  1. user time contains the guest time
#  2. nice time contains the guest_nice time
#
#
# Checking algorithm:
# 1. Read the line of all CPU(s) time which starts with the "cpu" tokens in an INTERVAL time
# 2. Read fields from the first to the eighth (because user contains guest and nice contains guest_nice)
# 3. sum all fields into a SUM result
# 4. The ideal SUM should be 100 (1 second has 100 units of 10ms) * CPU * INTERVAL
# 5. Check whether the SUM within an error of 5%
#

import time
from datetime import datetime
import multiprocessing
import os

PROC_STAT = "/proc/stat"
CPUS = multiprocessing.cpu_count()
SLEEP = 10 # in second
SUM_FIELDS = 8

def read_proc_stat():
    f = open(PROC_STAT)

    while True:
        line = f.readline()
        tokens = line.split()
        if tokens[0] == "cpu":
            return tokens[1:]

    raise

previous_times = []
next_times = []
diff_times = {}

previous_times = read_proc_stat()
while True:
    time.sleep(SLEEP)
    next_times = read_proc_stat()

    for i in range(0, SUM_FIELDS):
        diff_times[i] = int(next_times[i]) - int(previous_times[i])

    diffs = 0
    for i in range(0, SUM_FIELDS):
        diffs += diff_times[i]

    if diffs < (CPUS * SLEEP * 100 * 0.95) or diffs > (CPUS * SLEEP * 100 * 1.05):
        print("{} /proc/stat error: expected time: {} but real time: {}".format(datetime.now(), CPUS * SLEEP * 100, diffs))

    previous_times = next_times
