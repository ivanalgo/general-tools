#include "profiler.h"

#include <ctype.h>
#include <dirent.h>
#include <limits.h>


static int read_text_file(const char *path, char *buf, size_t buf_len) {
    FILE *fp = fopen(path, "r");
    size_t n;

    if (!fp) {
        return -errno;
    }

    n = fread(buf, 1, buf_len - 1, fp);
    if (ferror(fp)) {
        int err = errno;
        fclose(fp);
        return -err;
    }
    fclose(fp);

    buf[n] = '\0';
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) {
        buf[--n] = '\0';
    }
    return 0;
}

bool pmu_exists(const char *pmu_name) {
    char path[PATH_BUFFER_SIZE];
    snprintf(path, sizeof(path), "/sys/bus/event_source/devices/%s", pmu_name);
    return access(path, F_OK) == 0;
}

int pmu_read_type(const char *pmu_name, uint32_t *type_out) {
    char path[PATH_BUFFER_SIZE];
    int err = 0;
    uint64_t type;

    snprintf(path, sizeof(path), "/sys/bus/event_source/devices/%s/type",
             pmu_name);
    type = read_u64_file(path, &err);
    if (err != 0) {
        return -err;
    }

    *type_out = (uint32_t)type;
    return 0;
}

static int parse_format_spec(const char *text, int *reg_index,
                             unsigned *lo, unsigned *hi) {
    char reg_name[32];

    if (sscanf(text, "%31[^:]:%u-%u", reg_name, lo, hi) == 3) {
        /* parsed as range */
    } else if (sscanf(text, "%31[^:]:%u", reg_name, lo) == 2) {
        *hi = *lo;
    } else {
        return -EINVAL;
    }

    if (strcmp(reg_name, "config") == 0) {
        *reg_index = 0;
    } else if (strcmp(reg_name, "config1") == 0) {
        *reg_index = 1;
    } else if (strcmp(reg_name, "config2") == 0) {
        *reg_index = 2;
    } else {
        return -EINVAL;
    }

    return 0;
}

static int load_format_fields(const char *pmu_name, struct format_field *fields,
                              size_t max_fields, size_t *count_out,
                              char *reason, size_t reason_len) {
    char path[PATH_BUFFER_SIZE];
    DIR *dir;
    struct dirent *entry;
    size_t count = 0;

    snprintf(path, sizeof(path), "/sys/bus/event_source/devices/%s/format",
             pmu_name);
    dir = opendir(path);
    if (!dir) {
        snprintf(reason, reason_len, "failed to open %s: %s", path,
                 strerror(errno));
        return -errno;
    }

    while ((entry = readdir(dir)) != NULL) {
        char spec[128];
        char field_path[PATH_BUFFER_SIZE];
        int ret;

        if (entry->d_name[0] == '.') {
            continue;
        }
        if (count >= max_fields) {
            closedir(dir);
            return -E2BIG;
        }

        if (strlen(path) + 1 + strlen(entry->d_name) >= sizeof(field_path)) {
            snprintf(reason, reason_len,
                     "format path for PMU %s exceeds buffer: %s/%s", pmu_name,
                     path, entry->d_name);
            closedir(dir);
            return -ENAMETOOLONG;
        }
        memcpy(field_path, path, strlen(path));
        field_path[strlen(path)] = '/';
        memcpy(field_path + strlen(path) + 1, entry->d_name,
               strlen(entry->d_name) + 1);
        ret = read_text_file(field_path, spec, sizeof(spec));
        if (ret != 0) {
            closedir(dir);
            return ret;
        }

        if (strlen(entry->d_name) >= sizeof(fields[count].name)) {
            snprintf(reason, reason_len,
                     "format field name '%s' is too long for PMU %s",
                     entry->d_name, pmu_name);
            closedir(dir);
            return -ENAMETOOLONG;
        }
        strcpy(fields[count].name, entry->d_name);
        ret = parse_format_spec(spec, &fields[count].reg_index,
                                &fields[count].lo, &fields[count].hi);
        if (ret != 0) {
            snprintf(reason, reason_len,
                     "failed to parse format %s for PMU %s", entry->d_name,
                     pmu_name);
            closedir(dir);
            return ret;
        }
        count++;
    }

    closedir(dir);
    *count_out = count;
    return 0;
}

static const struct format_field *find_field(const struct format_field *fields,
                                             size_t count,
                                             const char *name) {
    size_t i;

    for (i = 0; i < count; i++) {
        if (strcmp(fields[i].name, name) == 0) {
            return &fields[i];
        }
    }
    return NULL;
}

static int set_field_value(uint64_t *config, uint64_t *config1, uint64_t *config2,
                           const struct format_field *field, uint64_t value,
                           char *reason, size_t reason_len) {
    uint64_t *target;
    unsigned width = field->hi - field->lo + 1;
    uint64_t limit_mask;
    uint64_t shifted;

    if (field->reg_index == 0) {
        target = config;
    } else if (field->reg_index == 1) {
        target = config1;
    } else {
        target = config2;
    }

    if (width >= 64) {
        limit_mask = UINT64_MAX;
    } else {
        limit_mask = (1ULL << width) - 1ULL;
    }

    if ((value & ~limit_mask) != 0) {
        snprintf(reason, reason_len,
                 "field %s only supports %u bits but got value 0x%" PRIx64,
                 field->name, width, value);
        return -ERANGE;
    }

    shifted = value << field->lo;
    *target |= shifted;
    return 0;
}

static int parse_expr_into_configs(const char *pmu_name, const char *expr,
                                   uint64_t *config, uint64_t *config1,
                                   uint64_t *config2, char *reason,
                                   size_t reason_len) {
    struct format_field fields[64];
    char temp[256];
    char *saveptr = NULL;
    char *token;
    size_t field_count = 0;
    int ret;

    *config = 0;
    *config1 = 0;
    *config2 = 0;

    if (!expr || expr[0] == '\0') {
        return 0;
    }

    ret = load_format_fields(pmu_name, fields, ARRAY_SIZE(fields), &field_count,
                             reason, reason_len);
    if (ret != 0) {
        return ret;
    }

    snprintf(temp, sizeof(temp), "%s", expr);
    token = strtok_r(temp, ",", &saveptr);
    while (token) {
        char *eq = strchr(token, '=');
        const struct format_field *field;
        uint64_t value;
        char *endptr;

        while (*token == ' ' || *token == '\t') {
            token++;
        }

        if (!eq) {
            snprintf(reason, reason_len,
                     "invalid PMU token '%s' for PMU %s", token, pmu_name);
            return -EINVAL;
        }

        *eq = '\0';
        field = find_field(fields, field_count, token);
        if (!field) {
            snprintf(reason, reason_len,
                     "PMU %s does not expose format field '%s'", pmu_name,
                     token);
            return -ENOENT;
        }

        value = strtoull(eq + 1, &endptr, 0);
        if (*endptr != '\0' && !isspace((unsigned char)*endptr)) {
            snprintf(reason, reason_len,
                     "invalid numeric value '%s' in token '%s'", eq + 1,
                     token);
            return -EINVAL;
        }

        ret = set_field_value(config, config1, config2, field, value, reason,
                              reason_len);
        if (ret != 0) {
            return ret;
        }

        token = strtok_r(NULL, ",", &saveptr);
    }

    return 0;
}

static int read_event_alias(const char *pmu_name, const char *event_name,
                            char *buf, size_t buf_len) {
    char path[PATH_BUFFER_SIZE];

    snprintf(path, sizeof(path), "/sys/bus/event_source/devices/%s/events/%s",
             pmu_name, event_name);
    return read_text_file(path, buf, buf_len);
}

int pmu_encode_event(const char *pmu_name, const char *alias_or_expr,
                     uint64_t *config, uint64_t *config1, uint64_t *config2,
                     char *reason, size_t reason_len) {
    char expr[256];
    int ret;

    if (!alias_or_expr || alias_or_expr[0] == '\0') {
        *config = 0;
        *config1 = 0;
        *config2 = 0;
        return 0;
    }

    ret = read_event_alias(pmu_name, alias_or_expr, expr, sizeof(expr));
    if (ret == 0) {
        return parse_expr_into_configs(pmu_name, expr, config, config1, config2,
                                       reason, reason_len);
    }

    return parse_expr_into_configs(pmu_name, alias_or_expr, config, config1,
                                   config2, reason, reason_len);
}
