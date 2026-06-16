#include "backend.h"

#include <strings.h>


static bool ibs_supported(char *reason, size_t reason_len) {
    const char *vendor = detect_cpu_vendor();

    if (!pmu_exists("ibs_op")) {
        snprintf(reason, reason_len, "ibs_op PMU is not exposed in sysfs");
        return false;
    }

    if (strcasecmp(vendor, "AuthenticAMD") != 0) {
        snprintf(reason, reason_len,
                 "IBS backend expects AuthenticAMD, found %s", vendor);
        return false;
    }

    return true;
}

static int ibs_prepare_attr(const struct profiler_options *options,
                            struct perf_event_attr *attr,
                            char *reason,
                            size_t reason_len) {
    uint64_t config = 0;
    uint64_t config1 = 0;
    uint64_t config2 = 0;
    uint32_t pmu_type = 0;
    int ret = pmu_read_type("ibs_op", &pmu_type);

    if (ret != 0) {
        snprintf(reason, reason_len, "failed to read ibs_op PMU type: %s",
                 strerror(-ret));
        return ret;
    }

    memset(attr, 0, sizeof(*attr));
    attr->size = sizeof(*attr);
    attr->type = pmu_type;
    attr->sample_period = options->sample_period;
    attr->disabled = 1;
    attr->inherit = 0;
    attr->exclude_guest = 1;
    attr->exclude_hv = 1;
    attr->exclude_kernel = options->user_only ? 1 : 0;
    attr->exclude_callchain_kernel = options->user_only ? 1 : 0;
    attr->precise_ip = 2;
    attr->sample_id_all = 1;
    attr->wakeup_events = 1;
    attr->sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME |
                        PERF_SAMPLE_ADDR | PERF_SAMPLE_CPU |
                        PERF_SAMPLE_WEIGHT | PERF_SAMPLE_DATA_SRC;
#ifdef PERF_SAMPLE_PHYS_ADDR
    attr->sample_type |= PERF_SAMPLE_PHYS_ADDR;
#endif

    ret = pmu_encode_event("ibs_op", "cycles", &config, &config1, &config2,
                           reason, reason_len);
    if (ret != 0) {
        config = 0;
        config1 = 0;
        config2 = 0;
    }

    attr->config = config;
    attr->config1 = config1;
    attr->config2 = config2;

    return 0;
}

static uint64_t ibs_page_key(const struct sample_record *sample,
                             size_t page_shift,
                             enum address_kind *kind) {
    if (sample->has_phys_addr) {
        *kind = ADDR_KIND_PHYSICAL;
        return sample->phys_addr >> page_shift;
    }
    if (sample->has_addr) {
        *kind = ADDR_KIND_VIRTUAL;
        return sample->addr >> page_shift;
    }
    return UINT64_MAX;
}

const struct profiler_backend ibs_backend = {
    .name = "ibs",
    .pmu_name = "ibs_op",
    .supported = ibs_supported,
    .prepare_attr = ibs_prepare_attr,
    .page_key = ibs_page_key,
};
