#include "rdma_demo.h"

#include <stdio.h>

const char *mode_name(enum demo_mode mode)
{
    switch (mode) {
    case MODE_SEND: return "send";
    case MODE_WRITE: return "write";
    case MODE_READ: return "read";
    case MODE_ALL: return "all";
    default: return "unknown";
    }
}

void fill_pattern(char *buf, size_t len, enum demo_mode mode, uint32_t seq)
{
    for (size_t i = 0; i < len; i++) {
        buf[i] = (char)('A' + ((i + seq + (unsigned)mode) % 26));
    }
}

int verify_pattern(const char *buf, size_t len, enum demo_mode mode, uint32_t seq,
                   char *err, size_t err_len)
{
    for (size_t i = 0; i < len; i++) {
        char expect = (char)('A' + ((i + seq + (unsigned)mode) % 26));
        if (buf[i] != expect) {
            snprintf(err, err_len, "offset=%zu expect=%c got=%c", i, expect, buf[i]);
            return -1;
        }
    }
    if (err_len) {
        err[0] = '\0';
    }
    return 0;
}

