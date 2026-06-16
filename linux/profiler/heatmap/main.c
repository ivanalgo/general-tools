#include "backend.h"

#include <getopt.h>


static void set_default_options(struct profiler_options *options) {
    memset(options, 0, sizeof(*options));
    options->pid = -1;
    options->system_wide = true;
    options->user_only = false;
    options->duration_sec = 5;
    options->poll_timeout_ms = 250;
    options->sample_period = 4000;
    options->mmap_pages = 128;
    options->max_pages = 65536;
    options->top_n = 20;
    options->process_top_n = 10;
    options->report_mode = REPORT_BOTH;
    options->summary_metric = SUMMARY_PAGES;
    options->cooling_mode = COOLING_EXP;
    options->cooling_decay = 0.80;
    options->cooling_step = 1.0;
    options->cooling_interval_ns = 500ULL * 1000ULL * 1000ULL;
    options->hot_threshold = 20.0;
    options->cold_threshold = 3.0;
    options->hot_percent = 10.0;
    options->cold_percent = 50.0;
    options->heat_policy = HEAT_POLICY_ABSOLUTE;
    options->stats_address_mode = STATS_ADDR_AUTO;
    options->output_format = OUTPUT_TEXT;
    options->output_path = NULL;
    options->backend_name = "auto";
}

static enum cooling_mode parse_cooling_mode(const char *text) {
    if (strcmp(text, "none") == 0) {
        return COOLING_NONE;
    }
    if (strcmp(text, "step") == 0) {
        return COOLING_STEP;
    }
    return COOLING_EXP;
}

static enum stats_address_mode parse_stats_address_mode(const char *text) {
    if (strcmp(text, "virtual") == 0) {
        return STATS_ADDR_VIRTUAL;
    }
    if (strcmp(text, "physical") == 0) {
        return STATS_ADDR_PHYSICAL;
    }
    return STATS_ADDR_AUTO;
}

static enum output_format parse_output_format(const char *text) {
    if (strcmp(text, "json") == 0) {
        return OUTPUT_JSON;
    }
    if (strcmp(text, "csv") == 0) {
        return OUTPUT_CSV;
    }
    return OUTPUT_TEXT;
}

static enum report_mode parse_report_mode(const char *text) {
    if (strcmp(text, "summary") == 0) {
        return REPORT_SUMMARY;
    }
    if (strcmp(text, "both") == 0) {
        return REPORT_BOTH;
    }
    return REPORT_DETAIL;
}

static enum heat_classification_policy parse_heat_policy(const char *text) {
    if (strcmp(text, "percentile") == 0) {
        return HEAT_POLICY_PERCENTILE;
    }
    return HEAT_POLICY_ABSOLUTE;
}

static enum summary_metric parse_summary_metric(const char *text) {
    if (strcmp(text, "heat") == 0) {
        return SUMMARY_HEAT;
    }
    if (strcmp(text, "samples") == 0) {
        return SUMMARY_SAMPLES;
    }
    return SUMMARY_PAGES;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --pid <pid>              profile a specific process\n"
            "  --system                 profile system-wide on all online CPUs (default)\n"
            "  --backend <auto|pebs|ibs>\n"
            "  --duration <sec>         profiling duration, default 5\n"
            "  --sample-period <n>      PMU sample period, default 4000\n"
            "  --mmap-pages <n>         perf ring pages, default 128\n"
            "  --max-pages <n>          max tracked pages, default 65536\n"
            "  --top <n>                report top N pages, default 20\n"
            "  --process-top <n>        report top N processes, default 10\n"
            "  --report-mode <detail|summary|both>\n"
            "  --summary-metric <pages|heat|samples>\n"
            "  --user-only              exclude kernel samples\n"
            "  --heat-policy <absolute|percentile>\n"
            "  --hot-percent <f>        top percentile marked hot, default 10\n"
            "  --cold-percent <f>       bottom percentile marked cold, default 50\n"
            "  --addr-mode <auto|virtual|physical>\n"
            "  --output <text|json|csv>\n"
            "  --output-file <path>     write report to file instead of stdout\n"
            "  --cooling <none|step|exp>\n"
            "  --cooling-interval-ms <n>\n"
            "  --cooling-decay <f>      exp cooling factor, default 0.80\n"
            "  --cooling-step <f>       step cooling decrement, default 1.0\n"
            "  --hot-threshold <f>\n"
            "  --cold-threshold <f>\n",
            prog);
}

int main(int argc, char **argv) {
    struct profiler_options options;
    struct heatmap heatmap;
    struct perf_session session;
    const struct profiler_backend *backend;
    char reason[REASON_BUFFER_SIZE];
    int ret;
    int opt;
    size_t page_shift;
    FILE *report_out = stdout;
    static const struct option long_options[] = {
        {"pid", required_argument, NULL, 'p'},
        {"system", no_argument, NULL, 's'},
        {"backend", required_argument, NULL, 'b'},
        {"duration", required_argument, NULL, 'd'},
        {"sample-period", required_argument, NULL, 'P'},
        {"mmap-pages", required_argument, NULL, 'm'},
        {"max-pages", required_argument, NULL, 'M'},
        {"top", required_argument, NULL, 't'},
        {"process-top", required_argument, NULL, 1008},
        {"report-mode", required_argument, NULL, 1010},
        {"summary-metric", required_argument, NULL, 1014},
        {"user-only", no_argument, NULL, 'u'},
        {"heat-policy", required_argument, NULL, 1011},
        {"hot-percent", required_argument, NULL, 1012},
        {"cold-percent", required_argument, NULL, 1013},
        {"addr-mode", required_argument, NULL, 1006},
        {"output", required_argument, NULL, 1007},
        {"output-file", required_argument, NULL, 1009},
        {"cooling", required_argument, NULL, 1000},
        {"cooling-interval-ms", required_argument, NULL, 1001},
        {"cooling-decay", required_argument, NULL, 1002},
        {"cooling-step", required_argument, NULL, 1003},
        {"hot-threshold", required_argument, NULL, 1004},
        {"cold-threshold", required_argument, NULL, 1005},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0},
    };

    set_default_options(&options);

    while ((opt = getopt_long(argc, argv, "p:sb:d:P:m:M:t:uh",
                              long_options, NULL)) != -1) {
        switch (opt) {
        case 'p':
            options.pid = (pid_t)atoi(optarg);
            options.system_wide = false;
            break;
        case 's':
            options.system_wide = true;
            options.pid = -1;
            break;
        case 'b':
            options.backend_name = optarg;
            break;
        case 'd':
            options.duration_sec = (unsigned)strtoul(optarg, NULL, 0);
            break;
        case 'P':
            options.sample_period = strtoull(optarg, NULL, 0);
            break;
        case 'm':
            options.mmap_pages = strtoull(optarg, NULL, 0);
            break;
        case 'M':
            options.max_pages = strtoull(optarg, NULL, 0);
            break;
        case 't':
            options.top_n = (unsigned)strtoul(optarg, NULL, 0);
            break;
        case 1008:
            options.process_top_n = (unsigned)strtoul(optarg, NULL, 0);
            break;
        case 1010:
            options.report_mode = parse_report_mode(optarg);
            break;
        case 1014:
            options.summary_metric = parse_summary_metric(optarg);
            break;
        case 'u':
            options.user_only = true;
            break;
        case 1011:
            options.heat_policy = parse_heat_policy(optarg);
            break;
        case 1012:
            options.hot_percent = strtod(optarg, NULL);
            break;
        case 1013:
            options.cold_percent = strtod(optarg, NULL);
            break;
        case 1006:
            options.stats_address_mode = parse_stats_address_mode(optarg);
            break;
        case 1007:
            options.output_format = parse_output_format(optarg);
            break;
        case 1009:
            options.output_path = optarg;
            break;
        case 1000:
            options.cooling_mode = parse_cooling_mode(optarg);
            break;
        case 1001:
            options.cooling_interval_ns =
                strtoull(optarg, NULL, 0) * 1000ULL * 1000ULL;
            break;
        case 1002:
            options.cooling_decay = strtod(optarg, NULL);
            break;
        case 1003:
            options.cooling_step = strtod(optarg, NULL);
            break;
        case 1004:
            options.hot_threshold = strtod(optarg, NULL);
            break;
        case 1005:
            options.cold_threshold = strtod(optarg, NULL);
            break;
        case 'h':
            usage(argv[0]);
            return 0;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    backend = profiler_select_backend(options.backend_name, reason,
                                      sizeof(reason));
    if (!backend) {
        fprintf(stderr, "backend selection failed: %s\n", reason);
        return 1;
    }

    page_shift = (size_t)__builtin_ctzl((unsigned long)sysconf(_SC_PAGESIZE));
    heatmap_init(&heatmap, options.max_pages, page_shift);

    ret = perf_session_open(&session, &options, backend, reason, sizeof(reason));
    if (ret != 0) {
        fprintf(stderr, "open session failed: %s\n", reason);
        heatmap_destroy(&heatmap);
        return 1;
    }

    fprintf(stderr,
            "profiling backend=%s vendor=%s target=%s duration=%us period=%" PRIu64
            " report_mode=%s summary_metric=%s heat_policy=%s addr_mode=%s output=%s cooling=%s\n",
            backend->name, detect_cpu_vendor(),
            options.system_wide ? "system" : "process", options.duration_sec,
            options.sample_period,
            report_mode_name(options.report_mode),
            summary_metric_name(options.summary_metric),
            heat_policy_name(options.heat_policy),
            stats_address_mode_name(options.stats_address_mode),
            output_format_name(options.output_format),
            cooling_mode_name(options.cooling_mode));
    fflush(stderr);

    ret = perf_session_run(&session, &options, backend, &heatmap, reason,
                           sizeof(reason));
    if (ret != 0) {
        fprintf(stderr, "profiling failed: %s\n", reason);
        perf_session_close(&session);
        heatmap_destroy(&heatmap);
        return 1;
    }

    if (options.output_path) {
        report_out = fopen(options.output_path, "w");
        if (!report_out) {
            fprintf(stderr, "failed to open output file %s: %s\n",
                    options.output_path, strerror(errno));
            perf_session_close(&session);
            heatmap_destroy(&heatmap);
            return 1;
        }
    }

    heatmap_report(&heatmap, &options, backend, session.lost_samples, report_out);

    if (report_out != stdout) {
        fclose(report_out);
        fprintf(stderr, "report written to %s\n", options.output_path);
    }

    perf_session_close(&session);
    heatmap_destroy(&heatmap);
    return 0;
}
