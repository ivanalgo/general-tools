#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#define ONE_GB ( 200 * 1024 * 1024)
#define MAX_BUCKETS 32  // 最多 32 个桶，2^31ns = 2.1s，足够长了

int get_exp_bin_index(long ns) {
    if (ns == 0) return 0;
    int index = 0;
    while (ns >>= 1) {
        index++;
    }
    if (index >= MAX_BUCKETS)
        index = MAX_BUCKETS - 1;
    return index;
}

int main() {
    size_t size = ONE_GB;
    char *buffer = (char *)malloc(size);
    if (buffer == NULL) {
        perror("malloc failed");
        return 1;
    }

    size_t pagesize = sysconf(_SC_PAGESIZE);
    struct timespec start, end;

    int histogram[MAX_BUCKETS] = {0};

    for (size_t i = 0; i < size; i += pagesize) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        buffer[i] = (char)(i % 256);
        clock_gettime(CLOCK_MONOTONIC, &end);

        long elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000L +
                          (end.tv_nsec - start.tv_nsec);

        if (elapsed_ns / 1e9 > max)
            max = elapsed_ns / 1e9;

        int bin = get_exp_bin_index(elapsed_ns);
        histogram[bin]++;
    }

    printf("\nExponential histogram of per-page write latency (ns):\n");
    for (int i = 0; i < MAX_BUCKETS; i++) {
        long lower = (i == 0) ? 0 : (1L << (i));
        long upper = (1L << (i + 1));
        printf("[%10ld - %10ld) ns : %8d\n", lower, upper, histogram[i]);
    }

    free(buffer);
    return 0;
}
