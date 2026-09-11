#include "rdma_demo.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int do_handshake(struct rdma_resources *res)
{
    if (rdma_post_recv(res) != 0) {
        return -1;
    }
    if (rdma_send_info(res) != 0) {
        return -1;
    }
    return rdma_recv_info(res);
}

static int send_ack(struct rdma_resources *res, uint32_t seq, uint32_t status, const char *text)
{
    struct demo_msg ack;
    memset(&ack, 0, sizeof(ack));
    ack.type = MSG_ACK;
    ack.seq = seq;
    ack.status = status;
    if (text) {
        snprintf(ack.text, sizeof(ack.text), "%s", text);
    }
    return rdma_send_msg(res, &ack, NULL, 0);
}

static int expect_ack(struct rdma_resources *res, uint32_t seq, const char *what)
{
    struct demo_msg ack;
    if (rdma_recv_msg(res, &ack, NULL, 0) != 0) {
        return -1;
    }
    if (ack.type != MSG_ACK || ack.seq != seq || ack.status != 0) {
        fprintf(stderr, "%s failed: type=%u seq=%u status=%u text=%s\n",
                what, ack.type, ack.seq, ack.status, ack.text);
        return -1;
    }
    return 0;
}

static int server_loop(struct rdma_resources *res)
{
    unsigned long send_ok = 0;
    unsigned long write_ok = 0;
    unsigned long read_ok = 0;

    for (;;) {
        if (rdma_post_recv(res) != 0) {
            return -1;
        }

        struct demo_msg msg;
        if (rdma_recv_msg(res, &msg, NULL, 0) != 0) {
            return -1;
        }

        if (msg.type == MSG_DONE) {
            printf("server done: send=%lu write=%lu read=%lu\n", send_ok, write_ok, read_ok);
            return 0;
        }

        char err[128];
        switch (msg.type) {
        case MSG_SEND_PAYLOAD:
            if (verify_pattern(res->recv_buf + sizeof(struct demo_msg), msg.size,
                               MODE_SEND, msg.seq, err, sizeof(err)) != 0) {
                send_ack(res, msg.seq, 1, err);
                return -1;
            }
            send_ok++;
            if (res->verbose) {
                printf("server verified SEND seq=%u size=%u\n", msg.seq, msg.size);
            }
            if (send_ack(res, msg.seq, 0, "send ok") != 0) {
                return -1;
            }
            break;
        case MSG_WRITE_VERIFY:
            if (verify_pattern(res->data_buf, msg.size, MODE_WRITE, msg.seq,
                               err, sizeof(err)) != 0) {
                send_ack(res, msg.seq, 1, err);
                return -1;
            }
            write_ok++;
            if (res->verbose) {
                printf("server verified RDMA WRITE seq=%u size=%u\n", msg.seq, msg.size);
            }
            if (send_ack(res, msg.seq, 0, "write ok") != 0) {
                return -1;
            }
            break;
        case MSG_READ_PREPARE:
            fill_pattern(res->data_buf, msg.size, MODE_READ, msg.seq);
            read_ok++;
            if (res->verbose) {
                printf("server prepared RDMA READ seq=%u size=%u\n", msg.seq, msg.size);
            }
            if (send_ack(res, msg.seq, 0, "read buffer ready") != 0) {
                return -1;
            }
            break;
        default:
            fprintf(stderr, "server got unknown message type=%u\n", msg.type);
            send_ack(res, msg.seq, 1, "unknown message");
            return -1;
        }
    }
}

static int client_send_test(struct rdma_resources *res, size_t size, int iters)
{
    char *payload = malloc(size);
    if (!payload) {
        perror("malloc");
        return -1;
    }
    for (int i = 0; i < iters; i++) {
        uint32_t seq = (uint32_t)i;
        fill_pattern(payload, size, MODE_SEND, seq);
        struct demo_msg msg;
        memset(&msg, 0, sizeof(msg));
        msg.type = MSG_SEND_PAYLOAD;
        msg.mode = MODE_SEND;
        msg.seq = seq;
        msg.size = (uint32_t)size;

        if (rdma_post_recv(res) != 0 ||
            rdma_send_msg(res, &msg, payload, size) != 0 ||
            expect_ack(res, seq, "SEND") != 0) {
            free(payload);
            return -1;
        }
        if (res->verbose) {
            printf("client SEND seq=%u size=%zu ok\n", seq, size);
        }
    }
    free(payload);
    printf("client SEND test passed: iters=%d size=%zu\n", iters, size);
    return 0;
}

static int client_write_test(struct rdma_resources *res, size_t size, int iters)
{
    for (int i = 0; i < iters; i++) {
        uint32_t seq = (uint32_t)i;
        if (rdma_write_peer(res, MODE_WRITE, seq, size) != 0) {
            return -1;
        }

        struct demo_msg msg;
        memset(&msg, 0, sizeof(msg));
        msg.type = MSG_WRITE_VERIFY;
        msg.mode = MODE_WRITE;
        msg.seq = seq;
        msg.size = (uint32_t)size;

        if (rdma_post_recv(res) != 0 ||
            rdma_send_msg(res, &msg, NULL, 0) != 0 ||
            expect_ack(res, seq, "RDMA WRITE") != 0) {
            return -1;
        }
        if (res->verbose) {
            printf("client RDMA WRITE seq=%u size=%zu ok\n", seq, size);
        }
    }
    printf("client RDMA WRITE test passed: iters=%d size=%zu\n", iters, size);
    return 0;
}

static int client_read_test(struct rdma_resources *res, size_t size, int iters)
{
    for (int i = 0; i < iters; i++) {
        uint32_t seq = (uint32_t)i;
        struct demo_msg msg;
        memset(&msg, 0, sizeof(msg));
        msg.type = MSG_READ_PREPARE;
        msg.mode = MODE_READ;
        msg.seq = seq;
        msg.size = (uint32_t)size;

        if (rdma_post_recv(res) != 0 ||
            rdma_send_msg(res, &msg, NULL, 0) != 0 ||
            expect_ack(res, seq, "RDMA READ prepare") != 0 ||
            rdma_read_peer(res, MODE_READ, seq, size) != 0) {
            return -1;
        }
        if (res->verbose) {
            printf("client RDMA READ seq=%u size=%zu ok\n", seq, size);
        }
    }
    printf("client RDMA READ test passed: iters=%d size=%zu\n", iters, size);
    return 0;
}

int run_server(const struct demo_options *opt)
{
    struct rdma_resources res;
    rdma_resources_init(&res, opt->verbose);
    rdma_resources_set_cm_timeout(&res, opt->cm_timeout_ms);

    int rc = -1;
    if (rdma_server_listen(&res, opt->port, 1) != 0) {
        goto out;
    }
    if (rdma_server_accept_one(&res, opt->size, opt->cq_depth) != 0) {
        goto out;
    }
    if (do_handshake(&res) != 0) {
        goto out;
    }
    rc = server_loop(&res);

out:
    rdma_disconnect_peer(&res);
    rdma_resources_cleanup(&res);
    return rc;
}

int run_client(const struct demo_options *opt)
{
    struct rdma_resources res;
    rdma_resources_init(&res, opt->verbose);
    rdma_resources_set_cm_timeout(&res, opt->cm_timeout_ms);

    int rc = -1;
    if (rdma_client_connect(&res, opt->addr, opt->port, opt->size, opt->cq_depth) != 0) {
        goto out;
    }
    if (do_handshake(&res) != 0) {
        goto out;
    }

    if ((opt->mode & MODE_SEND) && client_send_test(&res, opt->size, opt->iters) != 0) {
        goto out;
    }
    if ((opt->mode & MODE_WRITE) && client_write_test(&res, opt->size, opt->iters) != 0) {
        goto out;
    }
    if ((opt->mode & MODE_READ) && client_read_test(&res, opt->size, opt->iters) != 0) {
        goto out;
    }

    struct demo_msg done;
    memset(&done, 0, sizeof(done));
    done.type = MSG_DONE;
    if (rdma_send_msg(&res, &done, NULL, 0) != 0) {
        goto out;
    }
    rc = 0;

out:
    rdma_disconnect_peer(&res);
    rdma_resources_cleanup(&res);
    return rc;
}

int main(int argc, char **argv)
{
    struct demo_options opt;
    if (parse_options(argc, argv, &opt) != 0) {
        print_usage(argv[0]);
        return 2;
    }

    if (opt.role == ROLE_SERVER) {
        return run_server(&opt) == 0 ? 0 : 1;
    }
    if (opt.role == ROLE_SELFTEST) {
        return run_verbs_selftest(&opt) == 0 ? 0 : 1;
    }
    return run_client(&opt) == 0 ? 0 : 1;
}
