#include "profiler.h"

#include <math.h>
#include <fcntl.h>
#include <sys/stat.h>


static size_t next_power_of_two(size_t value) {
    size_t power = 1;

    while (power < value) {
        power <<= 1;
    }
    return power;
}

static uint64_t hash_page(uint64_t page, enum address_kind kind) {
    uint64_t x = page ^ ((uint64_t)kind << 61);

    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static const char *page_state_name(const struct heat_page *page,
                                   const struct profiler_options *options) {
    if (page->heat >= options->hot_threshold) {
        return "hot";
    }
    if (page->heat < options->cold_threshold) {
        return "cold";
    }
    return "warm";
}

struct process_summary {
    uint32_t pid;
    double heat;
    double total_weight;
    uint64_t samples;
    uint64_t pages;
    uint64_t hot_pages;
    uint64_t warm_pages;
    uint64_t cold_pages;
};

struct overall_summary {
    uint64_t total_pages;
    uint64_t hot_pages;
    uint64_t warm_pages;
    uint64_t cold_pages;
    uint64_t total_bytes;
    uint64_t hot_bytes;
    uint64_t warm_bytes;
    uint64_t cold_bytes;
    double total_heat;
    double hot_heat;
    double warm_heat;
    double cold_heat;
    uint64_t total_samples;
    uint64_t hot_samples;
    uint64_t warm_samples;
    uint64_t cold_samples;
};

static double summary_metric_total(const struct overall_summary *summary,
                                   enum summary_metric metric) {
    switch (metric) {
    case SUMMARY_HEAT:
        return summary->total_heat;
    case SUMMARY_SAMPLES:
        return (double)summary->total_samples;
    case SUMMARY_PAGES:
    default:
        return (double)summary->total_pages;
    }
}

static double summary_metric_value(const struct overall_summary *summary,
                                   enum summary_metric metric,
                                   const char *state) {
    if (metric == SUMMARY_HEAT) {
        if (strcmp(state, "hot") == 0) {
            return summary->hot_heat;
        }
        if (strcmp(state, "cold") == 0) {
            return summary->cold_heat;
        }
        return summary->warm_heat;
    }

    if (metric == SUMMARY_SAMPLES) {
        if (strcmp(state, "hot") == 0) {
            return (double)summary->hot_samples;
        }
        if (strcmp(state, "cold") == 0) {
            return (double)summary->cold_samples;
        }
        return (double)summary->warm_samples;
    }

    if (strcmp(state, "hot") == 0) {
        return (double)summary->hot_pages;
    }
    if (strcmp(state, "cold") == 0) {
        return (double)summary->cold_pages;
    }
    return (double)summary->warm_pages;
}

static const char *classify_page_state(const struct heat_page *page,
                                       const struct profiler_options *options,
                                       size_t rank,
                                       size_t total_count) {
    if (options->heat_policy == HEAT_POLICY_PERCENTILE) {
        size_t hot_cutoff;
        size_t cold_cutoff;
        size_t cold_start;

        hot_cutoff = (size_t)((options->hot_percent / 100.0) * (double)total_count);
        cold_cutoff = (size_t)((options->cold_percent / 100.0) * (double)total_count);
        if (options->hot_percent > 0.0 && hot_cutoff == 0 && total_count > 0) {
            hot_cutoff = 1;
        }
        if (options->cold_percent > 0.0 && cold_cutoff == 0 && total_count > 0) {
            cold_cutoff = 1;
        }
        if (hot_cutoff > total_count) {
            hot_cutoff = total_count;
        }
        if (cold_cutoff > total_count) {
            cold_cutoff = total_count;
        }

        if (rank < hot_cutoff) {
            return "hot";
        }

        cold_start = cold_cutoff >= total_count ? 0 : total_count - cold_cutoff;
        if (cold_start < hot_cutoff) {
            cold_start = hot_cutoff;
        }
        if (rank >= cold_start) {
            return "cold";
        }
        return "warm";
    }

    return page_state_name(page, options);
}

static struct overall_summary build_overall_summary(
    struct heat_page **ordered,
    size_t count,
    const struct profiler_options *options,
    size_t page_shift) {
    struct overall_summary summary;
    size_t i;
    uint64_t page_bytes = 1ULL << page_shift;

    memset(&summary, 0, sizeof(summary));
    summary.total_pages = count;
    summary.total_bytes = count * page_bytes;

    for (i = 0; i < count; i++) {
        const struct heat_page *page = ordered[i];
        const char *state = classify_page_state(ordered[i], options, i, count);

        summary.total_heat += page->heat;
        summary.total_samples += page->samples;

        if (strcmp(state, "hot") == 0) {
            summary.hot_pages++;
            summary.hot_bytes += page_bytes;
            summary.hot_heat += page->heat;
            summary.hot_samples += page->samples;
        } else if (strcmp(state, "cold") == 0) {
            summary.cold_pages++;
            summary.cold_bytes += page_bytes;
            summary.cold_heat += page->heat;
            summary.cold_samples += page->samples;
        } else {
            summary.warm_pages++;
            summary.warm_bytes += page_bytes;
            summary.warm_heat += page->heat;
            summary.warm_samples += page->samples;
        }
    }

    return summary;
}

void heatmap_init(struct heatmap *heatmap, size_t max_pages, size_t page_shift) {
    size_t capacity = next_power_of_two(max_pages * 2);

    memset(heatmap, 0, sizeof(*heatmap));
    heatmap->capacity = capacity < 1024 ? 1024 : capacity;
    heatmap->pages = calloc(heatmap->capacity, sizeof(*heatmap->pages));
    heatmap->page_shift = page_shift;
}

void heatmap_destroy(struct heatmap *heatmap) {
    size_t i;

    for (i = 0; i < ARRAY_SIZE(heatmap->pagemap_cache); i++) {
        if (heatmap->pagemap_cache[i].used && heatmap->pagemap_cache[i].fd >= 0) {
            close(heatmap->pagemap_cache[i].fd);
        }
    }
    free(heatmap->pages);
    memset(heatmap, 0, sizeof(*heatmap));
}

static int heatmap_get_pagemap_fd(struct heatmap *heatmap, pid_t pid) {
    char path[PATH_BUFFER_SIZE];
    size_t i;
    size_t slot;
    int fd;

    for (i = 0; i < ARRAY_SIZE(heatmap->pagemap_cache); i++) {
        if (heatmap->pagemap_cache[i].used && heatmap->pagemap_cache[i].pid == pid) {
            return heatmap->pagemap_cache[i].fd;
        }
    }

    snprintf(path, sizeof(path), "/proc/%d/pagemap", pid);
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return -errno;
    }

    for (i = 0; i < ARRAY_SIZE(heatmap->pagemap_cache); i++) {
        if (!heatmap->pagemap_cache[i].used) {
            heatmap->pagemap_cache[i].used = true;
            heatmap->pagemap_cache[i].pid = pid;
            heatmap->pagemap_cache[i].fd = fd;
            return fd;
        }
    }

    slot = heatmap->pagemap_cache_victim++ % ARRAY_SIZE(heatmap->pagemap_cache);
    if (heatmap->pagemap_cache[slot].used && heatmap->pagemap_cache[slot].fd >= 0) {
        close(heatmap->pagemap_cache[slot].fd);
    }
    heatmap->pagemap_cache[slot].used = true;
    heatmap->pagemap_cache[slot].pid = pid;
    heatmap->pagemap_cache[slot].fd = fd;
    return fd;
}

static bool translate_user_vaddr_to_phys_page(struct heatmap *heatmap,
                                              pid_t pid,
                                              uint64_t vaddr,
                                              uint64_t *phys_page) {
    uint64_t entry;
    uint64_t page_index = vaddr >> heatmap->page_shift;
    off_t offset = (off_t)(page_index * sizeof(entry));
    int fd;
    ssize_t nread;
    uint64_t pfn_mask = ((1ULL << 55) - 1ULL);

    if (pid <= 0) {
        return false;
    }
    if ((vaddr >> 63) != 0) {
        return false;
    }

    fd = heatmap_get_pagemap_fd(heatmap, pid);
    if (fd < 0) {
        return false;
    }

    nread = pread(fd, &entry, sizeof(entry), offset);
    if (nread != (ssize_t)sizeof(entry)) {
        return false;
    }

    if (((entry >> 63) & 0x1) == 0) {
        return false;
    }
    if (((entry >> 62) & 0x1) != 0) {
        return false;
    }

    *phys_page = entry & pfn_mask;
    return *phys_page != 0;
}

static uint64_t resolve_page_key(struct heatmap *heatmap,
                                 const struct profiler_options *options,
                                 const struct profiler_backend *backend,
                                 const struct sample_record *sample,
                                 enum address_kind *kind) {
    uint64_t phys_page;

    if (options->stats_address_mode == STATS_ADDR_VIRTUAL) {
        if (!sample->has_addr) {
            return UINT64_MAX;
        }
        *kind = ADDR_KIND_VIRTUAL;
        return sample->addr >> heatmap->page_shift;
    }

    if (options->stats_address_mode == STATS_ADDR_PHYSICAL) {
        heatmap->phys_translate_attempts++;
        if (sample->has_phys_addr) {
            *kind = ADDR_KIND_PHYSICAL;
            return sample->phys_addr >> heatmap->page_shift;
        }
        if (sample->has_addr &&
            translate_user_vaddr_to_phys_page(heatmap, (pid_t)sample->pid,
                                              sample->addr, &phys_page)) {
            *kind = ADDR_KIND_PHYSICAL;
            return phys_page;
        }
        heatmap->phys_translate_failures++;
        return UINT64_MAX;
    }

    if (sample->has_phys_addr) {
        heatmap->phys_translate_attempts++;
        *kind = ADDR_KIND_PHYSICAL;
        return sample->phys_addr >> heatmap->page_shift;
    }

    if (sample->has_addr &&
        translate_user_vaddr_to_phys_page(heatmap, (pid_t)sample->pid,
                                          sample->addr, &phys_page)) {
        heatmap->phys_translate_attempts++;
        *kind = ADDR_KIND_PHYSICAL;
        return phys_page;
    }

    if (sample->has_addr) {
        heatmap->phys_translate_attempts++;
        heatmap->phys_translate_failures++;
    }

    return backend->page_key(sample, heatmap->page_shift, kind);
}

static void heat_page_refresh_owner(struct heat_page *page) {
    size_t i;

    page->owner_pid = 0;
    page->owner_tid = 0;
    page->owner_samples = 0;
    for (i = 0; i < ARRAY_SIZE(page->owners); i++) {
        if (!page->owners[i].used) {
            continue;
        }
        if (page->owners[i].samples > page->owner_samples) {
            page->owner_pid = page->owners[i].pid;
            page->owner_tid = page->owners[i].tid;
            page->owner_samples = page->owners[i].samples;
        }
    }
}

static void heat_page_track_owner(struct heat_page *page,
                                  const struct sample_record *sample) {
    size_t i;
    size_t min_index = 0;
    uint64_t min_samples = UINT64_MAX;

    for (i = 0; i < ARRAY_SIZE(page->owners); i++) {
        if (page->owners[i].used && page->owners[i].pid == sample->pid &&
            page->owners[i].tid == sample->tid) {
            page->owners[i].samples++;
            heat_page_refresh_owner(page);
            return;
        }
    }

    for (i = 0; i < ARRAY_SIZE(page->owners); i++) {
        if (!page->owners[i].used) {
            page->owners[i].used = true;
            page->owners[i].pid = sample->pid;
            page->owners[i].tid = sample->tid;
            page->owners[i].samples = 1;
            heat_page_refresh_owner(page);
            return;
        }
    }

    for (i = 0; i < ARRAY_SIZE(page->owners); i++) {
        if (page->owners[i].samples < min_samples) {
            min_samples = page->owners[i].samples;
            min_index = i;
        }
    }

    page->owners[min_index].pid = sample->pid;
    page->owners[min_index].tid = sample->tid;
    page->owners[min_index].samples = 1;
    page->owners[min_index].used = true;
    heat_page_refresh_owner(page);
}

static void heatmap_apply_cooling(struct heatmap *heatmap,
                                  const struct profiler_options *options,
                                  uint64_t now_ns) {
    size_t i;
    uint64_t elapsed_intervals;

    if (options->cooling_mode == COOLING_NONE ||
        options->cooling_interval_ns == 0 || now_ns == 0) {
        return;
    }

    if (heatmap->last_cooling_ns == 0) {
        heatmap->last_cooling_ns = now_ns;
        return;
    }

    if (now_ns <= heatmap->last_cooling_ns) {
        return;
    }

    elapsed_intervals = (now_ns - heatmap->last_cooling_ns) /
                        options->cooling_interval_ns;
    if (elapsed_intervals == 0) {
        return;
    }

    for (i = 0; i < heatmap->capacity; i++) {
        struct heat_page *page = &heatmap->pages[i];

        if (!page->used || page->heat <= 0.0) {
            continue;
        }

        if (options->cooling_mode == COOLING_STEP) {
            double delta = options->cooling_step * (double)elapsed_intervals;
            page->heat = page->heat > delta ? page->heat - delta : 0.0;
        } else if (options->cooling_mode == COOLING_EXP) {
            page->heat *= pow(options->cooling_decay, (double)elapsed_intervals);
        }
    }

    heatmap->last_cooling_ns += elapsed_intervals * options->cooling_interval_ns;
}

static struct heat_page *heatmap_lookup(struct heatmap *heatmap, uint64_t page,
                                        enum address_kind kind,
                                        size_t max_pages) {
    size_t index = (size_t)hash_page(page, kind) & (heatmap->capacity - 1);
    size_t start = index;

    do {
        struct heat_page *slot = &heatmap->pages[index];

        if (!slot->used) {
            if (heatmap->count >= max_pages) {
                heatmap->dropped_pages++;
                return NULL;
            }

            slot->used = true;
            slot->page = page;
            slot->kind = kind;
            heatmap->count++;
            return slot;
        }

        if (slot->page == page && slot->kind == kind) {
            return slot;
        }

        index = (index + 1) & (heatmap->capacity - 1);
    } while (index != start);

    heatmap->dropped_pages++;
    return NULL;
}

void heatmap_record(struct heatmap *heatmap,
                    const struct profiler_options *options,
                    const struct profiler_backend *backend,
                    const struct sample_record *sample) {
    struct heat_page *page;
    enum address_kind kind = ADDR_KIND_VIRTUAL;
    uint64_t page_key;
    double weight;

    heatmap_apply_cooling(heatmap, options, sample->time_ns);
    page_key = resolve_page_key(heatmap, options, backend, sample, &kind);
    if (page_key == UINT64_MAX) {
        heatmap->dropped_samples++;
        return;
    }

    page = heatmap_lookup(heatmap, page_key, kind, options->max_pages);
    if (!page) {
        return;
    }

    weight = sample->has_weight && sample->weight != 0 ?
             (double)sample->weight : 0.0;
    page->heat += 1.0;
    page->total_weight += weight;
    page->samples++;
    page->last_ip = sample->ip;
    page->last_time_ns = sample->time_ns;
    page->last_data_src = sample->has_data_src ? sample->data_src : 0;
    heat_page_track_owner(page, sample);
}

static int compare_heat_page_desc(const void *lhs, const void *rhs) {
    const struct heat_page *const *a = lhs;
    const struct heat_page *const *b = rhs;

    if ((*a)->heat < (*b)->heat) {
        return 1;
    }
    if ((*a)->heat > (*b)->heat) {
        return -1;
    }
    if ((*a)->samples < (*b)->samples) {
        return 1;
    }
    if ((*a)->samples > (*b)->samples) {
        return -1;
    }
    return 0;
}

static struct heat_page **heatmap_build_sorted_pages(const struct heatmap *heatmap,
                                                     size_t *count_out) {
    struct heat_page **ordered;
    size_t i;
    size_t count = 0;

    ordered = calloc(heatmap->count ? heatmap->count : 1, sizeof(*ordered));
    if (!ordered) {
        return NULL;
    }

    for (i = 0; i < heatmap->capacity; i++) {
        if (heatmap->pages[i].used) {
            ordered[count++] = &heatmap->pages[i];
        }
    }

    qsort(ordered, count, sizeof(*ordered), compare_heat_page_desc);
    *count_out = count;
    return ordered;
}

static int compare_process_summary_desc(const void *lhs, const void *rhs) {
    const struct process_summary *a = lhs;
    const struct process_summary *b = rhs;

    if (a->heat < b->heat) {
        return 1;
    }
    if (a->heat > b->heat) {
        return -1;
    }
    if (a->samples < b->samples) {
        return 1;
    }
    if (a->samples > b->samples) {
        return -1;
    }
    if (a->pid < b->pid) {
        return -1;
    }
    if (a->pid > b->pid) {
        return 1;
    }
    return 0;
}

static struct process_summary *build_process_summaries(
    struct heat_page **ordered,
    size_t count,
    const struct profiler_options *options,
    size_t *summary_count_out) {
    struct process_summary *summaries;
    size_t summary_count = 0;
    size_t i;

    summaries = calloc(count ? count : 1, sizeof(*summaries));
    if (!summaries) {
        return NULL;
    }

    for (i = 0; i < count; i++) {
        const struct heat_page *page = ordered[i];
        size_t j;
        const char *state;

        for (j = 0; j < summary_count; j++) {
            if (summaries[j].pid == page->owner_pid) {
                break;
            }
        }
        if (j == summary_count) {
            summaries[j].pid = page->owner_pid;
            summary_count++;
        }

        summaries[j].heat += page->heat;
        summaries[j].total_weight += page->total_weight;
        summaries[j].samples += page->samples;
        summaries[j].pages++;

        state = classify_page_state(page, options, i, count);
        if (strcmp(state, "hot") == 0) {
            summaries[j].hot_pages++;
        } else if (strcmp(state, "cold") == 0) {
            summaries[j].cold_pages++;
        } else {
            summaries[j].warm_pages++;
        }
    }

    qsort(summaries, summary_count, sizeof(*summaries), compare_process_summary_desc);
    *summary_count_out = summary_count;
    return summaries;
}

static void report_text_summary(const struct overall_summary *summary,
                                const struct profiler_options *options,
                                FILE *out) {
    double metric_total = summary_metric_total(summary, options->summary_metric);
    double hot_metric = summary_metric_value(summary, options->summary_metric, "hot");
    double warm_metric = summary_metric_value(summary, options->summary_metric, "warm");
    double cold_metric = summary_metric_value(summary, options->summary_metric, "cold");

    fprintf(out,
            "summary policy=%s metric=%s total_pages=%" PRIu64
            " total_bytes=%" PRIu64 " total_heat=%.2f total_samples=%" PRIu64 "\n",
            heat_policy_name(options->heat_policy),
            summary_metric_name(options->summary_metric), summary->total_pages,
            summary->total_bytes, summary->total_heat, summary->total_samples);
    if (options->heat_policy == HEAT_POLICY_ABSOLUTE) {
        fprintf(out,
                "summary thresholds hot>=%.2f cold<%.2f\n",
                options->hot_threshold, options->cold_threshold);
    } else {
        fprintf(out,
                "summary percentiles hot_top=%.2f%% cold_bottom=%.2f%%\n",
                options->hot_percent, options->cold_percent);
    }
    fprintf(out,
            "%-8s %-12s %-18s %-16s %-12s\n",
            "class", "pages", "bytes", "metric_value", "ratio");
    fprintf(out, "%-8s %-12" PRIu64 " %-18" PRIu64 " %-16.2f %8.2f%%\n",
            "hot", summary->hot_pages, summary->hot_bytes, hot_metric,
            metric_total ? (100.0 * hot_metric / metric_total) : 0.0);
    fprintf(out, "%-8s %-12" PRIu64 " %-18" PRIu64 " %-16.2f %8.2f%%\n",
            "warm", summary->warm_pages, summary->warm_bytes, warm_metric,
            metric_total ? (100.0 * warm_metric / metric_total) : 0.0);
    fprintf(out, "%-8s %-12" PRIu64 " %-18" PRIu64 " %-16.2f %8.2f%%\n",
            "cold", summary->cold_pages, summary->cold_bytes, cold_metric,
            metric_total ? (100.0 * cold_metric / metric_total) : 0.0);
}

static void report_csv_summary(const struct overall_summary *summary,
                               const struct profiler_options *options,
                               FILE *out) {
    double metric_total = summary_metric_total(summary, options->summary_metric);
    double hot_metric = summary_metric_value(summary, options->summary_metric, "hot");
    double warm_metric = summary_metric_value(summary, options->summary_metric, "warm");
    double cold_metric = summary_metric_value(summary, options->summary_metric, "cold");

    fprintf(out,
            "summary_policy=%s,summary_metric=%s,total_pages=%" PRIu64
            ",total_bytes=%" PRIu64 ",total_heat=%.2f,total_samples=%" PRIu64,
            heat_policy_name(options->heat_policy),
            summary_metric_name(options->summary_metric), summary->total_pages,
            summary->total_bytes, summary->total_heat, summary->total_samples);
    if (options->heat_policy == HEAT_POLICY_ABSOLUTE) {
        fprintf(out, ",hot_threshold=%.2f,cold_threshold=%.2f\n",
                options->hot_threshold, options->cold_threshold);
    } else {
        fprintf(out, ",hot_percent=%.2f,cold_percent=%.2f\n",
                options->hot_percent, options->cold_percent);
    }
    fprintf(out, "summary_class,pages,bytes,metric_value,ratio\n");
    fprintf(out, "hot,%" PRIu64 ",%" PRIu64 ",%.2f,%.2f\n",
            summary->hot_pages, summary->hot_bytes, hot_metric,
            metric_total ? (100.0 * hot_metric / metric_total) : 0.0);
    fprintf(out, "warm,%" PRIu64 ",%" PRIu64 ",%.2f,%.2f\n",
            summary->warm_pages, summary->warm_bytes, warm_metric,
            metric_total ? (100.0 * warm_metric / metric_total) : 0.0);
    fprintf(out, "cold,%" PRIu64 ",%" PRIu64 ",%.2f,%.2f\n",
            summary->cold_pages, summary->cold_bytes, cold_metric,
            metric_total ? (100.0 * cold_metric / metric_total) : 0.0);
}

static void heatmap_report_text(const struct heatmap *heatmap,
                                const struct profiler_options *options,
                                const struct profiler_backend *backend,
                                uint64_t lost_samples,
                                struct heat_page **ordered,
                                size_t count,
                                FILE *out) {
    size_t i;
    size_t limit = options->top_n < count ? options->top_n : count;
    struct process_summary *summaries;
    size_t summary_count = 0;
    size_t summary_limit;
    struct overall_summary overall_summary;

    summaries = build_process_summaries(ordered, count, options, &summary_count);
    overall_summary = build_overall_summary(ordered, count, options,
                                            heatmap->page_shift);

    fprintf(out,
            "backend=%s pages=%zu dropped_pages=%zu dropped_samples=%zu lost_samples=%" PRIu64 " report_mode=%s summary_metric=%s heat_policy=%s addr_mode=%s output=%s cooling=%s interval_ms=%.2f\n",
            backend->name, heatmap->count, heatmap->dropped_pages,
            heatmap->dropped_samples, lost_samples,
            report_mode_name(options->report_mode),
            summary_metric_name(options->summary_metric),
            heat_policy_name(options->heat_policy),
            stats_address_mode_name(options->stats_address_mode),
            output_format_name(options->output_format),
            cooling_mode_name(options->cooling_mode),
            options->cooling_interval_ns / 1000000.0);
    if (options->stats_address_mode == STATS_ADDR_PHYSICAL ||
        options->stats_address_mode == STATS_ADDR_AUTO) {
        fprintf(out,
                "phys_translate_attempts=%zu phys_translate_failures=%zu\n",
                heatmap->phys_translate_attempts,
                heatmap->phys_translate_failures);
    }
    report_text_summary(&overall_summary, options, out);

    if (options->report_mode == REPORT_SUMMARY) {
        free(summaries);
        return;
    }

    fprintf(out, "\n");
    fprintf(out,
            "%-6s %-18s %-18s %-10s %-12s %-12s %-12s %-12s %-14s %-18s\n",
            "rank", "kind", "page_base", "state", "heat",
            "avg_weight", "owner_pid", "owner_tid", "owner_samples",
            "last_ip");

    for (i = 0; i < limit; i++) {
        const struct heat_page *page = ordered[i];
        uint64_t base = page->page << heatmap->page_shift;
        double avg_weight = page->samples ?
                            page->total_weight / (double)page->samples : 0.0;

        fprintf(out,
                "%-6zu %-18s 0x%016" PRIx64 " %-10s %-12.2f %-12.2f %-12u %-12u %-14" PRIu64
                " 0x%016" PRIx64 "\n",
                i + 1,
                page->kind == ADDR_KIND_PHYSICAL ? "physical" : "virtual",
                base, classify_page_state(page, options, i, count), page->heat, avg_weight,
                page->owner_pid, page->owner_tid, page->owner_samples,
                page->last_ip);
    }

    if (summaries) {
        summary_limit = options->process_top_n < summary_count ?
                        options->process_top_n : summary_count;
        fprintf(out,
                "\n%-6s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
                "rank", "pid", "heat", "pages", "samples",
                "hot_pages", "warm_pages", "cold_pages");
        for (i = 0; i < summary_limit; i++) {
            fprintf(out,
                    "%-6zu %-12u %-12.2f %-12" PRIu64 " %-12" PRIu64 " %-12" PRIu64 " %-12" PRIu64 " %-12" PRIu64 "\n",
                    i + 1, summaries[i].pid, summaries[i].heat,
                    summaries[i].pages, summaries[i].samples,
                    summaries[i].hot_pages, summaries[i].warm_pages,
                    summaries[i].cold_pages);
        }
        free(summaries);
    }
}

static void heatmap_report_csv(const struct heatmap *heatmap,
                               const struct profiler_options *options,
                               const struct profiler_backend *backend,
                               uint64_t lost_samples,
                               struct heat_page **ordered,
                               size_t count,
                               FILE *out) {
    size_t i;
    size_t limit = options->top_n < count ? options->top_n : count;
    struct process_summary *summaries;
    size_t summary_count = 0;
    size_t summary_limit;
    struct overall_summary overall_summary;

    summaries = build_process_summaries(ordered, count, options, &summary_count);
    overall_summary = build_overall_summary(ordered, count, options,
                                            heatmap->page_shift);

    fprintf(out,
            "backend=%s,pages=%zu,dropped_pages=%zu,dropped_samples=%zu,lost_samples=%" PRIu64 ",report_mode=%s,summary_metric=%s,heat_policy=%s,addr_mode=%s,output=%s,cooling=%s,interval_ms=%.2f\n",
            backend->name, heatmap->count, heatmap->dropped_pages,
            heatmap->dropped_samples, lost_samples,
            report_mode_name(options->report_mode),
            summary_metric_name(options->summary_metric),
            heat_policy_name(options->heat_policy),
            stats_address_mode_name(options->stats_address_mode),
            output_format_name(options->output_format),
            cooling_mode_name(options->cooling_mode),
            options->cooling_interval_ns / 1000000.0);
    if (options->stats_address_mode == STATS_ADDR_PHYSICAL ||
        options->stats_address_mode == STATS_ADDR_AUTO) {
        fprintf(out, "phys_translate_attempts=%zu,phys_translate_failures=%zu\n",
                heatmap->phys_translate_attempts,
                heatmap->phys_translate_failures);
    }
    report_csv_summary(&overall_summary, options, out);

    if (options->report_mode == REPORT_SUMMARY) {
        free(summaries);
        return;
    }

    fprintf(out, "\n");
    fprintf(out,
            "rank,kind,page_base,state,heat,avg_weight,owner_pid,owner_tid,owner_samples,samples,last_ip\n");
    for (i = 0; i < limit; i++) {
        const struct heat_page *page = ordered[i];
        uint64_t base = page->page << heatmap->page_shift;
        double avg_weight = page->samples ?
                            page->total_weight / (double)page->samples : 0.0;

        fprintf(out,
                "%zu,%s,0x%016" PRIx64 ",%s,%.2f,%.2f,%u,%u,%" PRIu64 ",%" PRIu64 ",0x%016" PRIx64 "\n",
                i + 1,
                page->kind == ADDR_KIND_PHYSICAL ? "physical" : "virtual",
                base, classify_page_state(page, options, i, count), page->heat, avg_weight,
                page->owner_pid, page->owner_tid, page->owner_samples,
                page->samples, page->last_ip);
    }

    if (summaries) {
        summary_limit = options->process_top_n < summary_count ?
                        options->process_top_n : summary_count;
        fprintf(out,
                "\nprocess_rank,pid,heat,pages,samples,hot_pages,warm_pages,cold_pages\n");
        for (i = 0; i < summary_limit; i++) {
            fprintf(out,
                    "%zu,%u,%.2f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                    i + 1, summaries[i].pid, summaries[i].heat,
                    summaries[i].pages, summaries[i].samples,
                    summaries[i].hot_pages, summaries[i].warm_pages,
                    summaries[i].cold_pages);
        }
        free(summaries);
    }
}

static void heatmap_report_json(const struct heatmap *heatmap,
                                const struct profiler_options *options,
                                const struct profiler_backend *backend,
                                uint64_t lost_samples,
                                struct heat_page **ordered,
                                size_t count,
                                FILE *out) {
    size_t i;
    size_t limit = options->top_n < count ? options->top_n : count;
    struct process_summary *summaries;
    size_t summary_count = 0;
    size_t summary_limit;
    struct overall_summary overall_summary;

    summaries = build_process_summaries(ordered, count, options, &summary_count);
    overall_summary = build_overall_summary(ordered, count, options,
                                            heatmap->page_shift);
    summary_limit = options->process_top_n < summary_count ?
                    options->process_top_n : summary_count;

    fprintf(out,
            "{\n  \"backend\": \"%s\",\n  \"pages\": %zu,\n  \"dropped_pages\": %zu,\n  \"dropped_samples\": %zu,\n  \"lost_samples\": %" PRIu64 ",\n  \"report_mode\": \"%s\",\n  \"summary_metric\": \"%s\",\n  \"heat_policy\": \"%s\",\n  \"addr_mode\": \"%s\",\n  \"output\": \"%s\",\n  \"cooling\": \"%s\",\n  \"interval_ms\": %.2f,\n  \"phys_translate_attempts\": %zu,\n  \"phys_translate_failures\": %zu,\n  \"summary\": {\n    \"total_pages\": %" PRIu64 ",\n    \"total_bytes\": %" PRIu64 ",\n    \"total_heat\": %.2f,\n    \"total_samples\": %" PRIu64 ",\n    \"hot_pages\": %" PRIu64 ",\n    \"hot_bytes\": %" PRIu64 ",\n    \"hot_heat\": %.2f,\n    \"hot_samples\": %" PRIu64 ",\n    \"warm_pages\": %" PRIu64 ",\n    \"warm_bytes\": %" PRIu64 ",\n    \"warm_heat\": %.2f,\n    \"warm_samples\": %" PRIu64 ",\n    \"cold_pages\": %" PRIu64 ",\n    \"cold_bytes\": %" PRIu64 ",\n    \"cold_heat\": %.2f,\n    \"cold_samples\": %" PRIu64 ",\n    \"hot_ratio\": %.2f,\n    \"warm_ratio\": %.2f,\n    \"cold_ratio\": %.2f",
            backend->name, heatmap->count, heatmap->dropped_pages,
            heatmap->dropped_samples, lost_samples,
            report_mode_name(options->report_mode),
            summary_metric_name(options->summary_metric),
            heat_policy_name(options->heat_policy),
            stats_address_mode_name(options->stats_address_mode),
            output_format_name(options->output_format),
            cooling_mode_name(options->cooling_mode),
            options->cooling_interval_ns / 1000000.0,
            heatmap->phys_translate_attempts,
            heatmap->phys_translate_failures,
            overall_summary.total_pages, overall_summary.total_bytes,
            overall_summary.total_heat, overall_summary.total_samples,
            overall_summary.hot_pages, overall_summary.hot_bytes,
            overall_summary.hot_heat, overall_summary.hot_samples,
            overall_summary.warm_pages, overall_summary.warm_bytes,
            overall_summary.warm_heat, overall_summary.warm_samples,
            overall_summary.cold_pages, overall_summary.cold_bytes,
            overall_summary.cold_heat, overall_summary.cold_samples,
            summary_metric_total(&overall_summary, options->summary_metric) ?
            (100.0 * summary_metric_value(&overall_summary, options->summary_metric, "hot") /
             summary_metric_total(&overall_summary, options->summary_metric)) : 0.0,
            summary_metric_total(&overall_summary, options->summary_metric) ?
            (100.0 * summary_metric_value(&overall_summary, options->summary_metric, "warm") /
             summary_metric_total(&overall_summary, options->summary_metric)) : 0.0,
            summary_metric_total(&overall_summary, options->summary_metric) ?
            (100.0 * summary_metric_value(&overall_summary, options->summary_metric, "cold") /
             summary_metric_total(&overall_summary, options->summary_metric)) : 0.0);
    if (options->heat_policy == HEAT_POLICY_ABSOLUTE) {
        fprintf(out,
                ",\n    \"hot_threshold\": %.2f,\n    \"cold_threshold\": %.2f\n  }",
                options->hot_threshold, options->cold_threshold);
    } else {
        fprintf(out,
                ",\n    \"hot_percent\": %.2f,\n    \"cold_percent\": %.2f\n  }",
                options->hot_percent, options->cold_percent);
    }

    if (options->report_mode == REPORT_SUMMARY) {
        fprintf(out, "\n}\n");
        free(summaries);
        return;
    }

    fprintf(out, ",\n  \"results\": [\n");

    for (i = 0; i < limit; i++) {
        const struct heat_page *page = ordered[i];
        uint64_t base = page->page << heatmap->page_shift;
        double avg_weight = page->samples ?
                            page->total_weight / (double)page->samples : 0.0;

        fprintf(out,
                "    {\"rank\": %zu, \"kind\": \"%s\", \"page_base\": \"0x%016" PRIx64 "\", \"state\": \"%s\", \"heat\": %.2f, \"avg_weight\": %.2f, \"owner_pid\": %u, \"owner_tid\": %u, \"owner_samples\": %" PRIu64 ", \"samples\": %" PRIu64 ", \"last_ip\": \"0x%016" PRIx64 "\"}%s\n",
                i + 1,
                page->kind == ADDR_KIND_PHYSICAL ? "physical" : "virtual",
                base, classify_page_state(page, options, i, count), page->heat, avg_weight,
                page->owner_pid, page->owner_tid, page->owner_samples,
                page->samples, page->last_ip,
                i + 1 == limit ? "" : ",");
    }

    fprintf(out, "  ],\n  \"process_results\": [\n");
    for (i = 0; i < summary_limit; i++) {
        fprintf(out,
                "    {\"rank\": %zu, \"pid\": %u, \"heat\": %.2f, \"pages\": %" PRIu64 ", \"samples\": %" PRIu64 ", \"hot_pages\": %" PRIu64 ", \"warm_pages\": %" PRIu64 ", \"cold_pages\": %" PRIu64 "}%s\n",
                i + 1, summaries[i].pid, summaries[i].heat,
                summaries[i].pages, summaries[i].samples,
                summaries[i].hot_pages, summaries[i].warm_pages,
                summaries[i].cold_pages,
                i + 1 == summary_limit ? "" : ",");
    }

    fprintf(out, "  ]\n}\n");
    free(summaries);
}

void heatmap_report(const struct heatmap *heatmap,
                    const struct profiler_options *options,
                    const struct profiler_backend *backend,
                    uint64_t lost_samples,
                    FILE *out) {
    struct heat_page **ordered;
    size_t count = 0;

    ordered = heatmap_build_sorted_pages(heatmap, &count);
    if (!ordered) {
        fprintf(out, "failed to allocate report buffer\n");
        return;
    }

    switch (options->output_format) {
    case OUTPUT_JSON:
        heatmap_report_json(heatmap, options, backend, lost_samples, ordered,
                            count, out);
        break;
    case OUTPUT_CSV:
        heatmap_report_csv(heatmap, options, backend, lost_samples, ordered,
                           count, out);
        break;
    case OUTPUT_TEXT:
    default:
        heatmap_report_text(heatmap, options, backend, lost_samples, ordered,
                            count, out);
        break;
    }

    free(ordered);
}
