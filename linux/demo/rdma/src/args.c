#include "rdma_demo.h"

#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_size(const char *s, size_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 0);
    if (errno || end == s) {
        return -1;
    }
    if (*end == 'k' || *end == 'K') {
        v *= 1024ull;
        end++;
    } else if (*end == 'm' || *end == 'M') {
        v *= 1024ull * 1024ull;
        end++;
    }
    if (*end != '\0' || v == 0) {
        return -1;
    }
    *out = (size_t)v;
    return 0;
}

static int parse_mode(const char *s, enum demo_mode *mode)
{
    if (strcmp(s, "send") == 0) {
        *mode = MODE_SEND;
    } else if (strcmp(s, "write") == 0) {
        *mode = MODE_WRITE;
    } else if (strcmp(s, "read") == 0) {
        *mode = MODE_READ;
    } else if (strcmp(s, "all") == 0) {
        *mode = MODE_ALL;
    } else {
        return -1;
    }
    return 0;
}

void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s --server [--bind-addr <local-ip>] [--port 7471] [--size 4096]\n"
            "  %s --client --addr <server-ip> [--bind-addr <local-ip>] [--port 7471]\n"
            "  %s --selftest [--dev-a rxe0 --dev-b rxe1] [--mode send|write|read|all]\n"
            "Options:\n"
            "  -s, --server          run as server; with --server-dev uses TCP control + raw verbs\n"
            "  -c, --client          run as client; with --client-dev uses TCP control + raw verbs\n"
            "      --selftest        run one-process verbs test without RDMA CM\n"
            "  -a, --addr ADDR       server IPv4/IPv6 address for client\n"
            "      --bind-addr ADDR  local IPv4/IPv6 address used by RDMA CM to select netdev/RDMA device\n"
            "  -p, --port PORT       TCP/RDMA-CM service port, default 7471\n"
            "  -m, --mode MODE       send/write/read/all, default all\n"
            "  -z, --size SIZE       bytes per operation, supports K/M suffix, default 4096\n"
            "  -n, --iters N         iterations per selected operation, default 10\n"
            "      --gid-index N     raw verbs/selftest GID index, default 0; RDMA CM uses route-selected GID\n"
            "      --cq-depth N      CQ/SQ/RQ depth, default 64\n"
            "      --cm-timeout-ms N RDMA CM event timeout, default 5000 ms\n"
            "      --client-dev NAME client/initiator RDMA device for raw verbs or selftest\n"
            "      --server-dev NAME server/responder RDMA device for raw verbs or selftest\n"
            "      --dev-a NAME      compatibility alias of --client-dev\n"
            "      --dev-b NAME      compatibility alias of --server-dev\n"
            "  -v, --verbose         print progress for every operation\n"
            "  -h, --help            show this help\n",
            prog, prog, prog);
}

int parse_options(int argc, char **argv, struct demo_options *opt)
{
    *opt = (struct demo_options){
        .role = 0,
        .mode = MODE_ALL,
        .addr = NULL,
        .bind_addr = NULL,
        .port = RDMA_DEMO_DEFAULT_PORT,
        .size = 4096,
        .iters = 10,
        .gid_index = -1,
        .cq_depth = 64,
        .verbose = 0,
        .cm_timeout_ms = RDMA_DEMO_DEFAULT_CM_TIMEOUT_MS,
        .dev_a = NULL,
        .dev_b = NULL,
    };

    static const struct option long_opts[] = {
        {"server", no_argument, NULL, 's'},
        {"client", no_argument, NULL, 'c'},
        {"selftest", no_argument, NULL, 1002},
        {"addr", required_argument, NULL, 'a'},
        {"bind-addr", required_argument, NULL, 1006},
        {"port", required_argument, NULL, 'p'},
        {"mode", required_argument, NULL, 'm'},
        {"size", required_argument, NULL, 'z'},
        {"iters", required_argument, NULL, 'n'},
        {"gid-index", required_argument, NULL, 1000},
        {"cq-depth", required_argument, NULL, 1001},
        {"dev-a", required_argument, NULL, 1003},
        {"dev-b", required_argument, NULL, 1004},
        {"cm-timeout-ms", required_argument, NULL, 1005},
        {"client-dev", required_argument, NULL, 1007},
        {"server-dev", required_argument, NULL, 1008},
        {"verbose", no_argument, NULL, 'v'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };

    int ch;
    while ((ch = getopt_long(argc, argv, "sca:p:m:z:n:vh", long_opts, NULL)) != -1) {
        switch (ch) {
        case 's': opt->role = ROLE_SERVER; break;
        case 'c': opt->role = ROLE_CLIENT; break;
        case 1002: opt->role = ROLE_SELFTEST; break;
        case 'a': opt->addr = optarg; break;
        case 1006: opt->bind_addr = optarg; break;
        case 'p': opt->port = optarg; break;
        case 'm':
            if (parse_mode(optarg, &opt->mode) != 0) {
                fprintf(stderr, "invalid mode: %s\n", optarg);
                return -1;
            }
            break;
        case 'z':
            if (parse_size(optarg, &opt->size) != 0) {
                fprintf(stderr, "invalid size: %s\n", optarg);
                return -1;
            }
            break;
        case 'n': opt->iters = atoi(optarg); break;
        case 1000: opt->gid_index = atoi(optarg); break;
        case 1001: opt->cq_depth = atoi(optarg); break;
        case 1003: opt->dev_a = optarg; break;
        case 1004: opt->dev_b = optarg; break;
        case 1005: opt->cm_timeout_ms = atoi(optarg); break;
        case 1007: opt->dev_a = optarg; break;
        case 1008: opt->dev_b = optarg; break;
        case 'v': opt->verbose++; break;
        case 'h': print_usage(argv[0]); exit(0);
        default: return -1;
        }
    }

    if (opt->role == 0) {
        fprintf(stderr, "must choose --server or --client\n");
        return -1;
    }
    if (opt->role == ROLE_CLIENT && !opt->addr) {
        fprintf(stderr, "client requires --addr\n");
        return -1;
    }
    if (opt->iters <= 0 || opt->cq_depth < 8) {
        fprintf(stderr, "--iters must be > 0 and --cq-depth must be >= 8\n");
        return -1;
    }
    if (opt->cm_timeout_ms <= 0) {
        fprintf(stderr, "--cm-timeout-ms must be > 0\n");
        return -1;
    }
    if (opt->size > RDMA_DEMO_MAX_PAYLOAD) {
        fprintf(stderr, "--size too large for SEND payload path; max %zu bytes\n",
                (size_t)RDMA_DEMO_MAX_PAYLOAD);
        return -1;
    }
    if (opt->gid_index >= 0 && opt->role != ROLE_SELFTEST && !opt->dev_a && !opt->dev_b) {
        fprintf(stderr,
                "note: --gid-index=%d is documented for ibv examples; this RDMA-CM demo "
                "keeps CM route-selected GID. Configure rxe/route instead.\n",
                opt->gid_index);
    }
    return 0;
}
