#include "backend.h"

#include <strings.h>


static const struct profiler_backend *all_backends[] = {
    &pebs_backend,
    &ibs_backend,
};

const char *detect_cpu_vendor(void) {
    static char vendor[64];
    static bool initialized = false;
    FILE *fp;
    char line[256];

    if (initialized) {
        return vendor[0] ? vendor : "unknown";
    }

    initialized = true;
    fp = fopen("/proc/cpuinfo", "r");
    if (!fp) {
        return "unknown";
    }

    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "vendor_id", 9) == 0) {
            char *colon = strchr(line, ':');
            if (!colon) {
                continue;
            }
            colon++;
            while (*colon == ' ' || *colon == '\t') {
                colon++;
            }
            snprintf(vendor, sizeof(vendor), "%s", colon);
            vendor[strcspn(vendor, "\r\n")] = '\0';
            break;
        }
    }

    fclose(fp);
    return vendor[0] ? vendor : "unknown";
}

const struct profiler_backend *profiler_select_backend(const char *name,
                                                       char *reason,
                                                       size_t reason_len) {
    size_t i;
    char backend_reason[REASON_BUFFER_SIZE];
    const char *vendor = detect_cpu_vendor();

    if (name && strcasecmp(name, "auto") != 0) {
        for (i = 0; i < ARRAY_SIZE(all_backends); i++) {
            if (strcasecmp(all_backends[i]->name, name) == 0) {
                if (all_backends[i]->supported(backend_reason,
                                               sizeof(backend_reason))) {
                    return all_backends[i];
                }
                snprintf(reason, reason_len,
                         "backend %s is not supported: %s",
                         all_backends[i]->name, backend_reason);
                return NULL;
            }
        }

        snprintf(reason, reason_len,
                 "unknown backend '%s', expected auto|pebs|ibs", name);
        return NULL;
    }

    if (strcasecmp(vendor, "GenuineIntel") == 0 &&
        pebs_backend.supported(backend_reason, sizeof(backend_reason))) {
        return &pebs_backend;
    }

    if (strcasecmp(vendor, "AuthenticAMD") == 0 &&
        ibs_backend.supported(backend_reason, sizeof(backend_reason))) {
        return &ibs_backend;
    }

    for (i = 0; i < ARRAY_SIZE(all_backends); i++) {
        if (all_backends[i]->supported(backend_reason, sizeof(backend_reason))) {
            return all_backends[i];
        }
    }

    snprintf(reason, reason_len,
             "no supported backend found on vendor=%s", vendor);
    return NULL;
}
