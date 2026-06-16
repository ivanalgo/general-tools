#ifndef MEMHEAT_PROFILER_H
#define MEMHEAT_PROFILER_H

#define _GNU_SOURCE


#include <errno.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define PATH_BUFFER_SIZE 512
#define REASON_BUFFER_SIZE 256

enum address_kind {
    ADDR_KIND_VIRTUAL = 0,
    ADDR_KIND_PHYSICAL = 1,
};

enum cooling_mode {
    COOLING_NONE = 0,
    COOLING_STEP = 1,
    COOLING_EXP = 2,
};

enum stats_address_mode {
    STATS_ADDR_AUTO = 0,
    STATS_ADDR_VIRTUAL = 1,
    STATS_ADDR_PHYSICAL = 2,
};

enum output_format {
    OUTPUT_TEXT = 0,
    OUTPUT_JSON = 1,
    OUTPUT_CSV = 2,
};

enum report_mode {
    REPORT_DETAIL = 0,
    REPORT_SUMMARY = 1,
    REPORT_BOTH = 2,
};

enum heat_classification_policy {
    HEAT_POLICY_ABSOLUTE = 0,
    HEAT_POLICY_PERCENTILE = 1,
};

enum summary_metric {
    SUMMARY_PAGES = 0,
    SUMMARY_HEAT = 1,
    SUMMARY_SAMPLES = 2,
};

#define PAGEMAP_CACHE_SIZE 32
#define PAGE_OWNER_SLOTS 4

struct profiler_options {
    pid_t pid;
    bool system_wide;
    bool user_only;
    unsigned duration_sec;
    unsigned poll_timeout_ms;
    uint64_t sample_period;
    size_t mmap_pages;
    size_t max_pages;
    unsigned top_n;
    unsigned process_top_n;
    enum report_mode report_mode;
    enum summary_metric summary_metric;
    enum cooling_mode cooling_mode;
    double cooling_decay;
    double cooling_step;
    uint64_t cooling_interval_ns;
    double hot_threshold;
    double cold_threshold;
    double hot_percent;
    double cold_percent;
    enum heat_classification_policy heat_policy;
    enum stats_address_mode stats_address_mode;
    enum output_format output_format;
    const char *output_path;
    const char *backend_name;
};

struct heat_owner {
    uint32_t pid;
    uint32_t tid;
    uint64_t samples;
    bool used;
};

struct sample_record {
    uint64_t ip;
    uint64_t addr;
    uint64_t phys_addr;
    uint64_t time_ns;
    uint64_t data_src;
    uint64_t weight;
    uint32_t pid;
    uint32_t tid;
    uint32_t cpu;
    bool has_addr;
    bool has_phys_addr;
    bool has_weight;
    bool has_data_src;
};

struct format_field {
    char name[64];
    int reg_index;
    unsigned lo;
    unsigned hi;
};

struct heat_page {
    uint64_t page;
    double heat;
    double total_weight;
    uint64_t samples;
    uint64_t last_ip;
    uint64_t last_time_ns;
    uint64_t last_data_src;
    uint32_t owner_pid;
    uint32_t owner_tid;
    uint64_t owner_samples;
    enum address_kind kind;
    struct heat_owner owners[PAGE_OWNER_SLOTS];
    bool used;
};

struct heatmap {
    struct heat_page *pages;
    size_t capacity;
    size_t count;
    size_t dropped_pages;
    size_t dropped_samples;
    size_t phys_translate_attempts;
    size_t phys_translate_failures;
    size_t page_shift;
    uint64_t last_cooling_ns;
    struct {
        pid_t pid;
        int fd;
        bool used;
    } pagemap_cache[PAGEMAP_CACHE_SIZE];
    size_t pagemap_cache_victim;
};

struct profiler_backend {
    const char *name;
    const char *pmu_name;
    bool (*supported)(char *reason, size_t reason_len);
    int (*prepare_attr)(const struct profiler_options *options,
                        struct perf_event_attr *attr,
                        char *reason,
                        size_t reason_len);
    uint64_t (*page_key)(const struct sample_record *sample, size_t page_shift,
                         enum address_kind *kind);
};

struct perf_handle {
    int fd;
    int cpu;
    void *base;
    size_t map_len;
};

struct perf_session {
    struct perf_handle *handles;
    size_t nr_handles;
    size_t nr_opened;
    size_t page_size;
    uint64_t sample_type;
    uint64_t lost_samples;
};

const struct profiler_backend *profiler_select_backend(const char *name,
                                                       char *reason,
                                                       size_t reason_len);
const char *detect_cpu_vendor(void);

bool pmu_exists(const char *pmu_name);
int pmu_read_type(const char *pmu_name, uint32_t *type_out);
int pmu_encode_event(const char *pmu_name, const char *alias_or_expr,
                     uint64_t *config, uint64_t *config1, uint64_t *config2,
                     char *reason, size_t reason_len);

void heatmap_init(struct heatmap *heatmap, size_t max_pages, size_t page_shift);
void heatmap_destroy(struct heatmap *heatmap);
void heatmap_record(struct heatmap *heatmap,
                    const struct profiler_options *options,
                    const struct profiler_backend *backend,
                    const struct sample_record *sample);
void heatmap_report(const struct heatmap *heatmap,
                    const struct profiler_options *options,
                    const struct profiler_backend *backend,
                    uint64_t lost_samples,
                    FILE *out);

int perf_session_open(struct perf_session *session,
                      const struct profiler_options *options,
                      const struct profiler_backend *backend,
                      char *reason,
                      size_t reason_len);
int perf_session_run(struct perf_session *session,
                     const struct profiler_options *options,
                     const struct profiler_backend *backend,
                     struct heatmap *heatmap,
                     char *reason,
                     size_t reason_len);
void perf_session_close(struct perf_session *session);

static inline int perf_event_open_syscall(struct perf_event_attr *attr,
                                          pid_t pid, int cpu, int group_fd,
                                          unsigned long flags) {
    return (int)syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static inline uint64_t read_u64_file(const char *path, int *err) {
    FILE *fp = fopen(path, "r");
    uint64_t value = 0;

    if (!fp) {
        if (err) {
            *err = errno;
        }
        return 0;
    }

    if (fscanf(fp, "%" SCNu64, &value) != 1) {
        if (err) {
            *err = EINVAL;
        }
        fclose(fp);
        return 0;
    }

    fclose(fp);
    if (err) {
        *err = 0;
    }
    return value;
}

static inline const char *cooling_mode_name(enum cooling_mode mode) {
    switch (mode) {
    case COOLING_NONE:
        return "none";
    case COOLING_STEP:
        return "step";
    case COOLING_EXP:
        return "exp";
    default:
        return "unknown";
    }
}

static inline const char *stats_address_mode_name(enum stats_address_mode mode) {
    switch (mode) {
    case STATS_ADDR_AUTO:
        return "auto";
    case STATS_ADDR_VIRTUAL:
        return "virtual";
    case STATS_ADDR_PHYSICAL:
        return "physical";
    default:
        return "unknown";
    }
}

static inline const char *output_format_name(enum output_format format) {
    switch (format) {
    case OUTPUT_TEXT:
        return "text";
    case OUTPUT_JSON:
        return "json";
    case OUTPUT_CSV:
        return "csv";
    default:
        return "unknown";
    }
}

static inline const char *report_mode_name(enum report_mode mode) {
    switch (mode) {
    case REPORT_DETAIL:
        return "detail";
    case REPORT_SUMMARY:
        return "summary";
    case REPORT_BOTH:
        return "both";
    default:
        return "unknown";
    }
}

static inline const char *heat_policy_name(enum heat_classification_policy policy) {
    switch (policy) {
    case HEAT_POLICY_ABSOLUTE:
        return "absolute";
    case HEAT_POLICY_PERCENTILE:
        return "percentile";
    default:
        return "unknown";
    }
}

static inline const char *summary_metric_name(enum summary_metric metric) {
    switch (metric) {
    case SUMMARY_PAGES:
        return "pages";
    case SUMMARY_HEAT:
        return "heat";
    case SUMMARY_SAMPLES:
        return "samples";
    default:
        return "unknown";
    }
}

#endif
