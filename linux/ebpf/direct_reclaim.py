#!/usr/bin/python3
import datetime

from bcc import BPF
from time import sleep
from bcc.utils import printb

bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

BPF_HASH(start_time, u32, u64);
BPF_PERF_OUTPUT(events);

struct data_t {
    u32 pid;
    char comm[16];
    u64 delta_ns;
    u64 pages_reclaimed;
    char source[16];
};

static int handle_reclaim_begin(void *ctx, const char *source) {
    u32 pid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    start_time.update(&pid, &ts);
    return 0;
}

static int handle_reclaim_end(void *ctx, unsigned long nr_reclaimed, const char *source) {
    u32 pid = bpf_get_current_pid_tgid();
    u64 *tsp = start_time.lookup(&pid);
    if (!tsp) return 0;

    u64 delta = bpf_ktime_get_ns() - *tsp;
    start_time.delete(&pid);

    struct data_t data = {};
    data.pid = pid;
	bpf_get_current_comm(data.comm, sizeof(data.comm));
    data.delta_ns = delta;
    data.pages_reclaimed = nr_reclaimed;
    __builtin_memcpy(&data.source, source, sizeof(data.source));

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

TRACEPOINT_PROBE(vmscan, mm_vmscan_direct_reclaim_begin) {
    return handle_reclaim_begin(args, "direct");
}

TRACEPOINT_PROBE(vmscan, mm_vmscan_direct_reclaim_end) {
    return handle_reclaim_end(args, args->nr_reclaimed, "direct");
}

TRACEPOINT_PROBE(vmscan, mm_vmscan_memcg_reclaim_begin) {
    return handle_reclaim_begin(args, "memcg");
}

TRACEPOINT_PROBE(vmscan, mm_vmscan_memcg_reclaim_end) {
    return handle_reclaim_end(args, args->nr_reclaimed, "memcg");
}
"""

b = BPF(text=bpf_text)

def print_event(cpu, data, size):
    event = b["events"].event(data)
    now = datetime.datetime.now()
    print(" %-10s %-8s %-6d %-16s %-8d %.3f ms" % (
        now.strftime("%H:%M:%S"),
        event.source.decode('utf-8', errors='replace'),
        event.pid,
        event.comm.decode('utf-8', errors='replace'),
        event.pages_reclaimed,
        float(event.delta_ns) / 1e6))

b["events"].open_perf_buffer(print_event)
print("Tracing direct and memcg reclaim... Press Ctrl-C to end.")

try:
    while True:
        b.perf_buffer_poll()
except KeyboardInterrupt:
    pass

