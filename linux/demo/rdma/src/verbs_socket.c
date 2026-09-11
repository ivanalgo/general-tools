#define _POSIX_C_SOURCE 200112L

#include "rdma_demo.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

enum sock_cmd_type {
    SOCK_CMD_SEND = 1,
    SOCK_CMD_WRITE = 2,
    SOCK_CMD_READ = 3,
    SOCK_CMD_DONE = 4,
    SOCK_CMD_ACK = 5,
};

struct sock_qp_info {
    uint32_t qpn;
    uint32_t psn;
    uint32_t rkey;
    uint32_t size;
    uint64_t addr;
    uint8_t gid[16];
};

struct sock_cmd {
    uint32_t type;
    uint32_t mode;
    uint32_t seq;
    uint32_t size;
    uint32_t status;
    char text[128];
};

struct sock_ep {
    const char *dev_name;
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    char *buf;
    size_t size;
    union ibv_gid gid;
    uint32_t psn;
};

static int write_full(int fd, const void *buf, size_t len)
{
    const char *p = buf;
    while (len) {
        ssize_t n = write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("write");
            return -1;
        }
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

static int read_full(int fd, void *buf, size_t len)
{
    char *p = buf;
    while (len) {
        ssize_t n = read(fd, p, len);
        if (n == 0) {
            fprintf(stderr, "peer closed TCP control connection\n");
            return -1;
        }
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("read");
            return -1;
        }
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

static int tcp_listen(const char *bind_addr, const char *port)
{
    struct addrinfo hints, *ai = NULL, *p;
    memset(&hints, 0, sizeof(hints));
    hints.ai_flags = bind_addr ? 0 : AI_PASSIVE;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    int rc = getaddrinfo(bind_addr, port, &hints, &ai);
    if (rc) {
        fprintf(stderr, "getaddrinfo(listen): %s\n", gai_strerror(rc));
        return -1;
    }
    int fd = -1;
    for (p = ai; p; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(fd, p->ai_addr, p->ai_addrlen) == 0 && listen(fd, 1) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(ai);
    if (fd < 0) perror("tcp listen");
    return fd;
}

static int tcp_connect(const char *addr, const char *bind_addr, const char *port)
{
    struct addrinfo hints, *dst = NULL, *p;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    int rc = getaddrinfo(addr, port, &hints, &dst);
    if (rc) {
        fprintf(stderr, "getaddrinfo(connect): %s\n", gai_strerror(rc));
        return -1;
    }
    int fd = -1;
    for (p = dst; p; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (bind_addr) {
            struct addrinfo sh, *src = NULL;
            memset(&sh, 0, sizeof(sh));
            sh.ai_family = p->ai_family;
            sh.ai_socktype = SOCK_STREAM;
            if (getaddrinfo(bind_addr, NULL, &sh, &src) == 0) {
                if (bind(fd, src->ai_addr, src->ai_addrlen) != 0) {
                    perror("tcp bind source");
                    freeaddrinfo(src);
                    close(fd);
                    fd = -1;
                    continue;
                }
                freeaddrinfo(src);
            }
        }
        if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(dst);
    if (fd < 0) perror("tcp connect");
    return fd;
}

static struct ibv_device *find_device(struct ibv_device **list, int n, const char *name)
{
    for (int i = 0; i < n; i++) {
        if (strcmp(ibv_get_device_name(list[i]), name) == 0) return list[i];
    }
    return NULL;
}

static int ep_init(struct sock_ep *ep, const char *dev_name, size_t size, int depth, int gid_index)
{
    memset(ep, 0, sizeof(*ep));
    ep->dev_name = dev_name;
    ep->size = size;
    ep->psn = 0xabc000u ^ (uint32_t)getpid();
    void *mem = NULL;
    if (posix_memalign(&mem, sysconf(_SC_PAGESIZE), size) != 0) {
        perror("posix_memalign");
        return -1;
    }
    ep->buf = mem;
    memset(ep->buf, 0, size);

    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list) {
        fprintf(stderr, "ibv_get_device_list failed\n");
        return -1;
    }
    struct ibv_device *dev = find_device(list, n, dev_name);
    if (!dev) {
        fprintf(stderr, "cannot find RDMA device %s\n", dev_name);
        ibv_free_device_list(list);
        return -1;
    }

    /* ibv_open_device() 显式打开用户指定的 RDMA NIC，避免 RDMA-CM 按路由自动选择。 */
    ep->ctx = ibv_open_device(dev);
    ibv_free_device_list(list);
    if (!ep->ctx) {
        fprintf(stderr, "ibv_open_device(%s) failed\n", dev_name);
        return -1;
    }
    /* ibv_alloc_pd()/ibv_create_cq()/ibv_create_qp() 在指定 RDMA NIC 上创建 PD/CQ/QPC/SQ/RQ。 */
    ep->pd = ibv_alloc_pd(ep->ctx);
    ep->cq = ep->pd ? ibv_create_cq(ep->ctx, depth, NULL, NULL, 0) : NULL;
    if (!ep->pd || !ep->cq) {
        perror("create pd/cq");
        return -1;
    }
    /* 注册 MR 并开放 rkey，供对端 RDMA READ/WRITE 访问。 */
    ep->mr = ibv_reg_mr(ep->pd, ep->buf, size,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
    if (!ep->mr) {
        perror("ibv_reg_mr");
        return -1;
    }
    if (ibv_query_gid(ep->ctx, 1, gid_index, &ep->gid) != 0) {
        perror("ibv_query_gid");
        return -1;
    }
    struct ibv_qp_init_attr qpia;
    memset(&qpia, 0, sizeof(qpia));
    qpia.qp_type = IBV_QPT_RC;
    qpia.send_cq = ep->cq;
    qpia.recv_cq = ep->cq;
    qpia.cap.max_send_wr = (uint32_t)depth;
    qpia.cap.max_recv_wr = (uint32_t)depth;
    qpia.cap.max_send_sge = 1;
    qpia.cap.max_recv_sge = 1;
    qpia.cap.max_inline_data = RDMA_DEMO_MAX_INLINE;
    ep->qp = ibv_create_qp(ep->pd, &qpia);
    if (!ep->qp) {
        perror("ibv_create_qp");
        return -1;
    }
    return 0;
}

static int ep_rtr_rts(struct sock_ep *ep, const struct sock_qp_info *peer, int gid_index)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
    if (ibv_modify_qp(ep->qp, &attr, IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS)) {
        perror("modify INIT");
        return -1;
    }
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = peer->qpn;
    attr.rq_psn = peer->psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    memcpy(&attr.ah_attr.grh.dgid, peer->gid, 16);
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.port_num = 1;
    if (ibv_modify_qp(ep->qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                      IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                      IBV_QP_MIN_RNR_TIMER)) {
        perror("modify RTR");
        return -1;
    }
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = ep->psn;
    attr.max_rd_atomic = 1;
    if (ibv_modify_qp(ep->qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                      IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
        perror("modify RTS");
        return -1;
    }
    return 0;
}

static void ep_info(const struct sock_ep *ep, struct sock_qp_info *info)
{
    memset(info, 0, sizeof(*info));
    info->qpn = ep->qp->qp_num;
    info->psn = ep->psn;
    info->rkey = ep->mr->rkey;
    info->size = (uint32_t)ep->size;
    info->addr = (uintptr_t)ep->buf;
    memcpy(info->gid, ep->gid.raw, 16);
}

static int ep_poll(struct sock_ep *ep, uint64_t wr_id)
{
    struct ibv_wc wc;
    for (;;) {
        int n = ibv_poll_cq(ep->cq, 1, &wc);
        if (n < 0) return -1;
        if (!n) continue;
        if (wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "%s CQE failed: %s\n", ep->dev_name, ibv_wc_status_str(wc.status));
            return -1;
        }
        if (wc.wr_id != wr_id) return -1;
        return 0;
    }
}

static int ep_post_recv(struct sock_ep *ep, size_t len)
{
    struct ibv_sge sge = { .addr = (uintptr_t)ep->buf, .length = (uint32_t)len, .lkey = ep->mr->lkey };
    struct ibv_recv_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 0x101;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    struct ibv_recv_wr *bad = NULL;
    /* ibv_post_recv() 提交 RQ WQE，server 等待 client 的 SEND 数据包。 */
    return ibv_post_recv(ep->qp, &wr, &bad);
}

static int ep_post_send(struct sock_ep *ep, enum ibv_wr_opcode op,
                        const struct sock_qp_info *peer, size_t len, uint64_t wr_id)
{
    struct ibv_sge sge = { .addr = (uintptr_t)ep->buf, .length = (uint32_t)len, .lkey = ep->mr->lkey };
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.opcode = op;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    if (op == IBV_WR_RDMA_WRITE || op == IBV_WR_RDMA_READ) {
        wr.wr.rdma.remote_addr = peer->addr;
        wr.wr.rdma.rkey = peer->rkey;
    }
    struct ibv_send_wr *bad = NULL;
    /* ibv_post_send() 提交 SQ WQE，真正的数据搬运由 HCA/RXE 执行。 */
    return ibv_post_send(ep->qp, &wr, &bad);
}

static void ep_cleanup(struct sock_ep *ep)
{
    if (ep->qp) ibv_destroy_qp(ep->qp);
    if (ep->mr) ibv_dereg_mr(ep->mr);
    if (ep->cq) ibv_destroy_cq(ep->cq);
    if (ep->pd) ibv_dealloc_pd(ep->pd);
    if (ep->ctx) ibv_close_device(ep->ctx);
    free(ep->buf);
}

static int send_ack(int fd, uint32_t seq, uint32_t status, const char *text)
{
    struct sock_cmd ack;
    memset(&ack, 0, sizeof(ack));
    ack.type = SOCK_CMD_ACK;
    ack.seq = seq;
    ack.status = status;
    if (text) snprintf(ack.text, sizeof(ack.text), "%s", text);
    return write_full(fd, &ack, sizeof(ack));
}

static int wait_ack(int fd, uint32_t seq, const char *what)
{
    struct sock_cmd ack;
    if (read_full(fd, &ack, sizeof(ack))) return -1;
    if (ack.type != SOCK_CMD_ACK || ack.seq != seq || ack.status) {
        fprintf(stderr, "%s failed: seq=%u status=%u text=%s\n", what, ack.seq, ack.status, ack.text);
        return -1;
    }
    return 0;
}

static int exchange_qp_info(int fd, struct sock_ep *ep, struct sock_qp_info *peer)
{
    struct sock_qp_info local;
    ep_info(ep, &local);
    return write_full(fd, &local, sizeof(local)) || read_full(fd, peer, sizeof(*peer));
}

static int server_loop(int fd, struct sock_ep *ep, const struct demo_options *opt)
{
    for (;;) {
        struct sock_cmd cmd;
        if (read_full(fd, &cmd, sizeof(cmd))) return -1;
        if (cmd.type == SOCK_CMD_DONE) return 0;
        char err[128];
        if (cmd.type == SOCK_CMD_SEND) {
            memset(ep->buf, 0, cmd.size);
            if (ep_post_recv(ep, cmd.size) || send_ack(fd, cmd.seq, 0, "recv ready")) return -1;
            if (ep_poll(ep, 0x101)) return -1;
            if (verify_pattern(ep->buf, cmd.size, MODE_SEND, cmd.seq, err, sizeof(err))) {
                send_ack(fd, cmd.seq, 1, err);
                return -1;
            }
            if (opt->verbose) printf("server verified SEND seq=%u\n", cmd.seq);
            if (send_ack(fd, cmd.seq, 0, "send ok")) return -1;
        } else if (cmd.type == SOCK_CMD_WRITE) {
            if (verify_pattern(ep->buf, cmd.size, MODE_WRITE, cmd.seq, err, sizeof(err))) {
                send_ack(fd, cmd.seq, 1, err);
                return -1;
            }
            if (opt->verbose) printf("server verified RDMA WRITE seq=%u\n", cmd.seq);
            if (send_ack(fd, cmd.seq, 0, "write ok")) return -1;
        } else if (cmd.type == SOCK_CMD_READ) {
            fill_pattern(ep->buf, cmd.size, MODE_READ, cmd.seq);
            if (opt->verbose) printf("server prepared RDMA READ seq=%u\n", cmd.seq);
            if (send_ack(fd, cmd.seq, 0, "read ready")) return -1;
        } else {
            send_ack(fd, cmd.seq, 1, "unknown command");
            return -1;
        }
    }
}

int run_verbs_socket_server(const struct demo_options *opt)
{
    const char *dev = opt->dev_b ? opt->dev_b : opt->dev_a;
    const char *bind_addr = opt->bind_addr;
    int gid_index = opt->gid_index >= 0 ? opt->gid_index : 0;
    int lfd = tcp_listen(bind_addr, opt->port);
    if (lfd < 0) return -1;
    printf("verbs server listening on %s:%s rdma-dev=%s\n",
           bind_addr ? bind_addr : "0.0.0.0", opt->port, dev);
    int fd = accept(lfd, NULL, NULL);
    close(lfd);
    if (fd < 0) {
        perror("accept");
        return -1;
    }
    struct sock_ep ep;
    struct sock_qp_info peer;
    int rc = -1;
    if (ep_init(&ep, dev, opt->size, opt->cq_depth, gid_index) == 0 &&
        exchange_qp_info(fd, &ep, &peer) == 0 &&
        ep_rtr_rts(&ep, &peer, gid_index) == 0) {
        printf("verbs server connected: rdma-dev=%s qpn=%u peer-qpn=%u\n",
               dev, ep.qp->qp_num, peer.qpn);
        rc = server_loop(fd, &ep, opt);
    }
    ep_cleanup(&ep);
    close(fd);
    return rc;
}

static int send_cmd(int fd, uint32_t type, enum demo_mode mode, uint32_t seq, size_t size)
{
    struct sock_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type = type;
    cmd.mode = mode;
    cmd.seq = seq;
    cmd.size = (uint32_t)size;
    return write_full(fd, &cmd, sizeof(cmd));
}

static int client_tests(int fd, struct sock_ep *ep, const struct sock_qp_info *peer,
                        const struct demo_options *opt)
{
    char err[128];
    for (int i = 0; i < opt->iters; i++) {
        uint32_t seq = (uint32_t)i;
        if (opt->mode & MODE_SEND) {
            fill_pattern(ep->buf, opt->size, MODE_SEND, seq);
            if (send_cmd(fd, SOCK_CMD_SEND, MODE_SEND, seq, opt->size) ||
                wait_ack(fd, seq, "server recv ready") ||
                ep_post_send(ep, IBV_WR_SEND, peer, opt->size, 0x201) || ep_poll(ep, 0x201) ||
                wait_ack(fd, seq, "SEND verify")) return -1;
            if (opt->verbose) printf("client SEND seq=%u ok\n", seq);
        }
        if (opt->mode & MODE_WRITE) {
            fill_pattern(ep->buf, opt->size, MODE_WRITE, seq);
            if (ep_post_send(ep, IBV_WR_RDMA_WRITE, peer, opt->size, 0x202) || ep_poll(ep, 0x202) ||
                send_cmd(fd, SOCK_CMD_WRITE, MODE_WRITE, seq, opt->size) ||
                wait_ack(fd, seq, "WRITE verify")) return -1;
            if (opt->verbose) printf("client RDMA WRITE seq=%u ok\n", seq);
        }
        if (opt->mode & MODE_READ) {
            if (send_cmd(fd, SOCK_CMD_READ, MODE_READ, seq, opt->size) ||
                wait_ack(fd, seq, "READ prepare")) return -1;
            memset(ep->buf, 0, opt->size);
            if (ep_post_send(ep, IBV_WR_RDMA_READ, peer, opt->size, 0x203) || ep_poll(ep, 0x203)) return -1;
            if (verify_pattern(ep->buf, opt->size, MODE_READ, seq, err, sizeof(err))) {
                fprintf(stderr, "client RDMA READ verify failed: %s\n", err);
                return -1;
            }
            if (opt->verbose) printf("client RDMA READ seq=%u ok\n", seq);
        }
    }
    return send_cmd(fd, SOCK_CMD_DONE, 0, 0, 0);
}

int run_verbs_socket_client(const struct demo_options *opt)
{
    const char *dev = opt->dev_a ? opt->dev_a : opt->dev_b;
    int gid_index = opt->gid_index >= 0 ? opt->gid_index : 0;
    int fd = tcp_connect(opt->addr, opt->bind_addr, opt->port);
    if (fd < 0) return -1;
    struct sock_ep ep;
    struct sock_qp_info peer;
    int rc = -1;
    if (ep_init(&ep, dev, opt->size, opt->cq_depth, gid_index) == 0 &&
        exchange_qp_info(fd, &ep, &peer) == 0 &&
        ep_rtr_rts(&ep, &peer, gid_index) == 0) {
        printf("verbs client connected: rdma-dev=%s qpn=%u peer-qpn=%u\n",
               dev, ep.qp->qp_num, peer.qpn);
        rc = client_tests(fd, &ep, &peer, opt);
        if (rc == 0) printf("verbs client tests passed: mode=%s iters=%d size=%zu\n",
                            mode_name(opt->mode), opt->iters, opt->size);
    }
    ep_cleanup(&ep);
    close(fd);
    return rc;
}

