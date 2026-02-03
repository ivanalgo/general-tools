#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

volatile uint64_t sink;

typedef enum {
    PAT_SEQ,
    PAT_STRIDE,
    PAT_REGION
} pattern_t;

int main(int argc, char **argv)
{
    size_t size = 256UL << 20;   // 256MB
    size_t loops = 10;
    size_t stride = 64;
    size_t region_count = 4;
	size_t region_size;
    pattern_t pattern = PAT_SEQ;

    int opt;
    while ((opt = getopt(argc, argv, "s:l:p:t:r:c:")) != -1) {
        switch (opt) {
        case 's': size = strtoull(optarg, NULL, 0); break;
        case 'l': loops = strtoull(optarg, NULL, 0); break;
        case 'p':
            if (!strcmp(optarg, "seq")) pattern = PAT_SEQ;
            else if (!strcmp(optarg, "stride")) pattern = PAT_STRIDE;
            else if (!strcmp(optarg, "region")) pattern = PAT_REGION;
            break;
        case 't': stride = strtoull(optarg, NULL, 0); break;
        case 'r': region_size = strtoull(optarg, NULL, 0); break;
        case 'c': region_count = strtoull(optarg, NULL, 0); break;
        default:
            fprintf(stderr,
              "-s size -l loops -p seq|stride|region "
              "-t stride -r region_size -c region_count\n");
            exit(1);
        }
    }

	region_size = size / region_count;
    size_t elems = size / sizeof(uint64_t);
    uint64_t *buf;

    if (posix_memalign((void **)&buf, 64, size)) {
		perror("posix_memalign");
		exit(1);
	}

    memset(buf, 1, size);

    size_t stride_elems = stride / sizeof(uint64_t);
    size_t region_elems = region_size / sizeof(uint64_t);

    for (size_t l = 0; l < loops; l++) {
        switch (pattern) {
        case PAT_SEQ:
            for (size_t i = 0; i < elems; i++)
                sink += buf[i];
            break;

        case PAT_STRIDE:
            for (size_t i = 0; i < elems; i++)
                sink += buf[(i * stride_elems) % elems];
            break;

        case PAT_REGION:
            for (size_t r = 0; r < region_count; r++) {
                size_t base = (r * region_elems) % elems;
                for (size_t i = 0; i < region_elems; i++)
                    sink += buf[base + i];
            }
            break;
        }
    }

    printf("sink=%lu\n", sink);
    return 0;
}

