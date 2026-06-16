#include "profiler.h"

#include <dirent.h>
#include <poll.h>
#include <sched.h>

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>

static void perf_rmb(void) {
    __sync_synchronize();
}

static void perf_mbw(void) {
    __sync_synchronize();
}

static uint64_t monotonic_time_ns(void) {
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void ring_copy(const struct perf_event_mmap_page *metadata,
                      uint64_t absolute_offset, void *dst, size_t len) {
    const char *data = ((const char *)metadata) + metadata->data_offset;
    size_t data_size = metadata->data_size;
    size_t begin = (size_t)(absolute_offset % data_size);
    size_t first = len;

    if (begin + len > data_size) {
        first = data_size - begin;
    }

    memcpy(dst, data + begin, first);
    if (first < len) {
        memcpy((char *)dst + first, data, len - first);
    }
}

static uint64_t ring_read_u64(const struct perf_event_mmap_page *metadata,
                              uint64_t *cursor) {
    uint64_t value;

    ring_copy(metadata, *cursor, &value, sizeof(value));
    *cursor += sizeof(value);
    return value;
}

static void ring_read_pid_tid(const struct perf_event_mmap_page *metadata,
                              uint64_t *cursor, uint32_t *pid,
                              uint32_t *tid) {
    struct {
        uint32_t pid;
        uint32_t tid;
    } value;

    ring_copy(metadata, *cursor, &value, sizeof(value));
    *cursor += sizeof(value);
    *pid = value.pid;
    *tid = value.tid;
}

static void ring_read_cpu(const struct perf_event_mmap_page *metadata,
                          uint64_t *cursor, uint32_t *cpu) {
    struct {
        uint32_t cpu;
        uint32_t reserved;
    } value;

    ring_copy(metadata, *cursor, &value, sizeof(value));
    *cursor += sizeof(value);
    *cpu = value.cpu;
}

static void parse_sample(const struct perf_event_mmap_page *metadata,
                         uint64_t payload_offset, uint64_t sample_type,
                         struct sample_record *sample) {
    uint64_t cursor = payload_offset;

    memset(sample, 0, sizeof(*sample));

    if (sample_type & PERF_SAMPLE_IP) {
        sample->ip = ring_read_u64(metadata, &cursor);
    }
    if (sample_type & PERF_SAMPLE_TID) {
        ring_read_pid_tid(metadata, &cursor, &sample->pid, &sample->tid);
    }
    if (sample_type & PERF_SAMPLE_TIME) {
        sample->time_ns = ring_read_u64(metadata, &cursor);
    }
    if (sample_type & PERF_SAMPLE_ADDR) {
        sample->addr = ring_read_u64(metadata, &cursor);
        sample->has_addr = true;
    }
    if (sample_type & PERF_SAMPLE_CPU) {
        ring_read_cpu(metadata, &cursor, &sample->cpu);
    }
    if (sample_type & PERF_SAMPLE_WEIGHT) {
        sample->weight = ring_read_u64(metadata, &cursor);
        sample->has_weight = true;
    }
    if (sample_type & PERF_SAMPLE_DATA_SRC) {
        sample->data_src = ring_read_u64(metadata, &cursor);
        sample->has_data_src = true;
    }
#ifdef PERF_SAMPLE_PHYS_ADDR
    if (sample_type & PERF_SAMPLE_PHYS_ADDR) {
        sample->phys_addr = ring_read_u64(metadata, &cursor);
        sample->has_phys_addr = true;
    }
#endif
}

static int count_online_cpus(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

static int list_thread_ids(pid_t pid, int **tids_out, size_t *count_out,
                           char *reason, size_t reason_len) {
    char path[PATH_BUFFER_SIZE];
    DIR *dir;
    struct dirent *entry;
    int *tids = NULL;
    size_t count = 0;
    size_t capacity = 16;

    snprintf(path, sizeof(path), "/proc/%d/task", pid);
    dir = opendir(path);
    if (!dir) {
        snprintf(reason, reason_len, "failed to open %s: %s", path,
                 strerror(errno));
        return -errno;
    }

    tids = calloc(capacity, sizeof(*tids));
    if (!tids) {
        closedir(dir);
        return -ENOMEM;
    }

    while ((entry = readdir(dir)) != NULL) {
        char *endptr;
        long tid;

        if (entry->d_name[0] == '.') {
            continue;
        }

        tid = strtol(entry->d_name, &endptr, 10);
        if (*endptr != '\0' || tid <= 0) {
            continue;
        }

        if (count == capacity) {
            int *new_tids;
            capacity *= 2;
            new_tids = realloc(tids, capacity * sizeof(*tids));
            if (!new_tids) {
                free(tids);
                closedir(dir);
                return -ENOMEM;
            }
            tids = new_tids;
        }

        tids[count++] = (int)tid;
    }

    closedir(dir);
    *tids_out = tids;
    *count_out = count;
    return 0;
}

static int perf_enable_all(struct perf_session *session) {
    size_t i;

    for (i = 0; i < session->nr_opened; i++) {
        if (ioctl(session->handles[i].fd, PERF_EVENT_IOC_RESET, 0) != 0) {
            return -errno;
        }
    }
    for (i = 0; i < session->nr_opened; i++) {
        if (ioctl(session->handles[i].fd, PERF_EVENT_IOC_ENABLE, 0) != 0) {
            return -errno;
        }
    }
    return 0;
}

static void perf_disable_all(struct perf_session *session) {
    size_t i;

    for (i = 0; i < session->nr_opened; i++) {
        if (session->handles[i].fd >= 0) {
            ioctl(session->handles[i].fd, PERF_EVENT_IOC_DISABLE, 0);
        }
    }
}

int perf_session_open(struct perf_session *session,
                      const struct profiler_options *options,
                      const struct profiler_backend *backend,
                      char *reason,
                      size_t reason_len) {
    struct perf_event_attr attr;
    int *tids = NULL;
    pid_t target_pid = options->pid > 0 ? options->pid : getpid();
    int ret;
    int nr_targets;
    int i;

    memset(session, 0, sizeof(*session));
    session->page_size = (size_t)sysconf(_SC_PAGESIZE);

    ret = backend->prepare_attr(options, &attr, reason, reason_len);
    if (ret != 0) {
        return ret;
    }

    session->sample_type = attr.sample_type;
    if (options->system_wide) {
        nr_targets = count_online_cpus();
    } else {
        size_t tid_count = 0;
        ret = list_thread_ids(target_pid, &tids, &tid_count, reason, reason_len);
        if (ret != 0) {
            return ret;
        }
        if (tid_count == 0) {
            free(tids);
            snprintf(reason, reason_len,
                     "no threads found under /proc/%d/task", target_pid);
            return -ESRCH;
        }
        nr_targets = (int)tid_count;
    }
    session->nr_handles = (size_t)nr_targets;
    session->handles = calloc(session->nr_handles, sizeof(*session->handles));
    if (!session->handles) {
        free(tids);
        snprintf(reason, reason_len, "failed to allocate perf handles");
        return -ENOMEM;
    }

    for (i = 0; i < nr_targets; i++) {
        session->handles[i].fd = -1;
    }

    for (i = 0; i < nr_targets; i++) {
        struct perf_handle *handle = &session->handles[i];
        pid_t pid = options->system_wide ? -1 : tids[i];
        int cpu = options->system_wide ? i : -1;

        /*
         * perf_event_open target selection convention:
         * - system-wide profiling: pid=-1 and cpu=<online cpu index>
         * - per-thread profiling : pid=<tid> and cpu=-1
         *
         * The code never mixes pid>=0 with cpu>=0 because the profiler chooses
         * either "one event per CPU" or "one event per thread".
         */
        handle->cpu = cpu;
        handle->map_len = (options->mmap_pages + 1) * session->page_size;

        /*
         * group_fd=-1 means this event is opened standalone rather than as part
         * of a perf event group. flags=0 keeps the default perf semantics.
         */
        handle->fd = perf_event_open_syscall(&attr, pid, cpu, -1, 0);
        if (handle->fd < 0) {
            if (!options->system_wide) {
                snprintf(reason, reason_len,
                         "perf_event_open failed for backend=%s pid=%d: %s. "
                         "Check CAP_PERFMON or /proc/sys/kernel/perf_event_paranoid",
                         backend->name, pid, strerror(errno));
                free(tids);
                return -errno;
            }
            continue;
        }

        handle->base = mmap(NULL, handle->map_len, PROT_READ | PROT_WRITE,
                            MAP_SHARED, handle->fd, 0);
        if (handle->base == MAP_FAILED) {
            close(handle->fd);
            handle->fd = -1;
            handle->base = NULL;
            if (!options->system_wide) {
                snprintf(reason, reason_len, "mmap perf ring failed: %s",
                         strerror(errno));
                free(tids);
                return -errno;
            }
            continue;
        }

        session->nr_opened++;
    }

    if (session->nr_opened == 0) {
        free(tids);
        snprintf(reason, reason_len,
                 "no perf event opened successfully for backend=%s",
                 backend->name);
        return -EINVAL;
    }

    ret = perf_enable_all(session);
    if (ret != 0) {
        free(tids);
        snprintf(reason, reason_len, "failed to enable perf events: %s",
                 strerror(-ret));
        return ret;
    }

    free(tids);

    return 0;
}

static void drain_perf_ring(struct perf_session *session,
                            struct perf_handle *handle,
                            const struct profiler_options *options,
                            const struct profiler_backend *backend,
                            struct heatmap *heatmap) {
    struct perf_event_mmap_page *metadata = handle->base;
    uint64_t head;
    uint64_t tail;

    if (!metadata) {
        return;
    }

    head = metadata->data_head;
    perf_rmb();
    tail = metadata->data_tail;

    while (tail < head) {
        struct perf_event_header header;

        ring_copy(metadata, tail, &header, sizeof(header));
        if (header.size < sizeof(header)) {
            break;
        }

        if (header.type == PERF_RECORD_SAMPLE) {
            struct sample_record sample;

            parse_sample(metadata, tail + sizeof(header), session->sample_type,
                         &sample);
            heatmap_record(heatmap, options, backend, &sample);
        } else if (header.type == PERF_RECORD_LOST) {
            uint64_t cursor = tail + sizeof(header);
            (void)ring_read_u64(metadata, &cursor);
            session->lost_samples += ring_read_u64(metadata, &cursor);
        }

        tail += header.size;
    }

    metadata->data_tail = tail;
    perf_mbw();
}

int perf_session_run(struct perf_session *session,
                     const struct profiler_options *options,
                     const struct profiler_backend *backend,
                     struct heatmap *heatmap,
                     char *reason,
                     size_t reason_len) {
    struct pollfd *pfds;
    uint64_t start_ns = monotonic_time_ns();
    int ret = 0;
    size_t i;

    pfds = calloc(session->nr_opened, sizeof(*pfds));
    if (!pfds) {
        snprintf(reason, reason_len, "failed to allocate pollfd array");
        return -ENOMEM;
    }

    for (i = 0; i < session->nr_opened; i++) {
        pfds[i].fd = session->handles[i].fd;
        pfds[i].events = POLLIN;
    }

    while ((monotonic_time_ns() - start_ns) <
           (uint64_t)options->duration_sec * 1000000000ULL) {
        int ready = poll(pfds, session->nr_opened, options->poll_timeout_ms);

        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            snprintf(reason, reason_len, "poll failed: %s", strerror(errno));
            ret = -errno;
            break;
        }

        for (i = 0; i < session->nr_opened; i++) {
            if (pfds[i].revents & (POLLIN | POLLHUP)) {
                drain_perf_ring(session, &session->handles[i], options, backend,
                                heatmap);
            }
        }
    }

    perf_disable_all(session);

    for (i = 0; i < session->nr_opened; i++) {
        drain_perf_ring(session, &session->handles[i], options, backend,
                        heatmap);
    }

    free(pfds);
    return ret;
}

void perf_session_close(struct perf_session *session) {
    size_t i;

    if (!session) {
        return;
    }

    for (i = 0; i < session->nr_handles; i++) {
        if (session->handles[i].base) {
            munmap(session->handles[i].base, session->handles[i].map_len);
        }
        if (session->handles[i].fd >= 0) {
            close(session->handles[i].fd);
        }
    }

    free(session->handles);
    memset(session, 0, sizeof(*session));
}
