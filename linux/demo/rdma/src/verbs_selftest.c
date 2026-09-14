#define _POSIX_C_SOURCE 200112L

#include "rdma_demo.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct qp_ep {
    const char *name;
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    char *buf;
    union ibv_gid gid;
    uint32_t psn;
};

static int alloc_page_aligned(char **p, size_t len)
{
    void *mem = NULL;
    int rc = posix_memalign(&mem, sysconf(_SC_PAGESIZE), len);
    if (rc != 0) {
        errno = rc;
        perror("posix_memalign");
        return -1;
    }
    memset(mem, 0, len);
    *p = mem;
    return 0;
}

static int poll_cq(struct qp_ep *ep, uint64_t wr_id)
{
    struct ibv_wc wc;
    for (;;) {
        int n = ibv_poll_cq(ep->cq, 1, &wc);
        if (n < 0) {
            fprintf(stderr, "%s: ibv_poll_cq failed\n", ep->name);
            return -1;
        }
        if (n == 0) {
            continue;
        }
        if (wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "%s: wc status=%s(%d)\n", ep->name,
                    ibv_wc_status_str(wc.status), wc.status);
            return -1;
        }
        if (wc.wr_id != wr_id) {
            fprintf(stderr, "%s: unexpected wr_id=0x%lx expect=0x%lx\n",
                    ep->name, (unsigned long)wc.wr_id, (unsigned long)wr_id);
            return -1;
        }
        return 0;
    }
}

static int init_ep(struct qp_ep *ep, struct ibv_device *dev, const char *name,
                   size_t size, int depth, int gid_index)
{
    memset(ep, 0, sizeof(*ep));
    ep->name = name;
    ep->psn = 0x1000 + (uint32_t)(name[0] << 4);
    if (alloc_page_aligned(&ep->buf, size) != 0) {
        return -1;
    }

    /* ibv_open_device() 打开 HCA/RXE 设备上下文；后续 PD/CQ/QP 都在该设备上创建。 */
    ep->ctx = ibv_open_device(dev);
    if (!ep->ctx) {
        fprintf(stderr, "%s: ibv_open_device failed\n", name);
        return -1;
    }
    /* ibv_alloc_pd() 创建保护域，硬件用 PD 隔离 QP 和 MR 访问权限。 */
    ep->pd = ibv_alloc_pd(ep->ctx);
    if (!ep->pd) {
        perror("ibv_alloc_pd");
        return -1;
    }
    /* ibv_create_cq() 创建 CQ/CQC，HCA 完成 WQE 后向 CQ 写 CQE。 */
    ep->cq = ibv_create_cq(ep->ctx, depth, NULL, NULL, 0);
    if (!ep->cq) {
        perror("ibv_create_cq");
        return -1;
    }
    /* ibv_reg_mr() pin 用户页并建立 MPT/MTT，返回 lkey/rkey 给 WQE 使用。 */
    ep->mr = ibv_reg_mr(ep->pd, ep->buf, size,
                        IBV_ACCESS_LOCAL_WRITE |
                        IBV_ACCESS_REMOTE_READ |
                        IBV_ACCESS_REMOTE_WRITE);
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
    /* ibv_create_qp() 创建 RC QP，对应硬件 QPC，并分配 SQ/RQ WQE ring。 */
    ep->qp = ibv_create_qp(ep->pd, &qpia);
    if (!ep->qp) {
        perror("ibv_create_qp");
        return -1;
    }
    return 0;
}

static int qp_to_init(struct qp_ep *ep)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
    return ibv_modify_qp(ep->qp, &attr,
                         IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS);
}

static int qp_to_rtr(struct qp_ep *ep, const struct qp_ep *peer, int gid_index)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = peer->qp->qp_num;
    attr.rq_psn = peer->psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = peer->gid;
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.port_num = 1;
    return ibv_modify_qp(ep->qp, &attr,
                         IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                         IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                         IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

static int qp_to_rts(struct qp_ep *ep)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = ep->psn;
    attr.max_rd_atomic = 1;
    return ibv_modify_qp(ep->qp, &attr,
                         IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                         IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

static int post_recv_ep(struct qp_ep *ep, size_t len)
{
    struct ibv_sge sge = { .addr = (uintptr_t)ep->buf, .length = (uint32_t)len, .lkey = ep->mr->lkey };
    struct ibv_recv_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 0x11;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    struct ibv_recv_wr *bad = NULL;
    /* ibv_post_recv() 向 RQ 写入 Receive WQE，等待对端 SEND 消费。 */
    return ibv_post_recv(ep->qp, &wr, &bad);
}

static int post_send_ep(struct qp_ep *ep, enum ibv_wr_opcode opcode,
                        struct qp_ep *peer, size_t len, uint64_t wr_id)
{
    struct ibv_sge sge = { .addr = (uintptr_t)ep->buf, .length = (uint32_t)len, .lkey = ep->mr->lkey };
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = wr_id;
    wr.opcode = opcode;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    if (opcode == IBV_WR_RDMA_WRITE || opcode == IBV_WR_RDMA_READ) {
        wr.wr.rdma.remote_addr = (uintptr_t)peer->buf;
        wr.wr.rdma.rkey = peer->mr->rkey;
    }
    struct ibv_send_wr *bad = NULL;
    /* ibv_post_send() 向 SQ 提交 SEND/RDMA READ/RDMA WRITE WQE 并敲 doorbell。 */
    return ibv_post_send(ep->qp, &wr, &bad);
}

static void cleanup_ep(struct qp_ep *ep)
{
    if (ep->qp) ibv_destroy_qp(ep->qp);
    if (ep->mr) ibv_dereg_mr(ep->mr);
    if (ep->cq) ibv_destroy_cq(ep->cq);
    if (ep->pd) ibv_dealloc_pd(ep->pd);
    if (ep->ctx) ibv_close_device(ep->ctx);
    free(ep->buf);
}

static struct ibv_device *pick_device(struct ibv_device **list, int n, const char *name, int index)
{
    if (name) {
        for (int i = 0; i < n; i++) {
            if (strcmp(ibv_get_device_name(list[i]), name) == 0) {
                return list[i];
            }
        }
        return NULL;
    }
    return index < n ? list[index] : NULL;
}

int run_verbs_selftest(const struct demo_options *opt)
{
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list || n == 0) {
        fprintf(stderr, "no verbs device found\n");
        return -1;
    }
    struct ibv_device *dev_a = pick_device(list, n, opt->dev_a, 0);
    struct ibv_device *dev_b = pick_device(list, n, opt->dev_b, n > 1 ? 1 : 0);
    if (!dev_a || !dev_b) {
        fprintf(stderr, "cannot find requested devices dev-a=%s dev-b=%s\n",
                opt->dev_a ? opt->dev_a : "<auto>", opt->dev_b ? opt->dev_b : "<auto>");
        ibv_free_device_list(list);
        return -1;
    }

    int gid_index = opt->gid_index >= 0 ? opt->gid_index : 0;
    struct qp_ep a, b;
    if (init_ep(&a, dev_a, ibv_get_device_name(dev_a), opt->size, opt->cq_depth, gid_index) != 0 ||
        init_ep(&b, dev_b, ibv_get_device_name(dev_b), opt->size, opt->cq_depth, gid_index) != 0) {
        ibv_free_device_list(list);
        return -1;
    }
    ibv_free_device_list(list);

    if (qp_to_init(&a) || qp_to_init(&b) ||
        qp_to_rtr(&a, &b, gid_index) || qp_to_rtr(&b, &a, gid_index) ||
        qp_to_rts(&a) || qp_to_rts(&b)) {
        perror("ibv_modify_qp");
        cleanup_ep(&b); cleanup_ep(&a);
        return -1;
    }

    for (int i = 0; i < opt->iters; i++) {
        if (opt->mode & MODE_SEND) {
            fill_pattern(a.buf, opt->size, MODE_SEND, (uint32_t)i);
            memset(b.buf, 0, opt->size);
            if (post_recv_ep(&b, opt->size) || post_send_ep(&a, IBV_WR_SEND, &b, opt->size, 0x21) ||
                poll_cq(&a, 0x21) || poll_cq(&b, 0x11)) {
                cleanup_ep(&b); cleanup_ep(&a); return -1;
            }
            char err[128];
            if (verify_pattern(b.buf, opt->size, MODE_SEND, (uint32_t)i, err, sizeof(err))) {
                fprintf(stderr, "selftest SEND verify failed: %s\n", err);
                cleanup_ep(&b); cleanup_ep(&a); return -1;
            }
        }
        if (opt->mode & MODE_WRITE) {
            fill_pattern(a.buf, opt->size, MODE_WRITE, (uint32_t)i);
            memset(b.buf, 0, opt->size);
            if (post_send_ep(&a, IBV_WR_RDMA_WRITE, &b, opt->size, 0x22) || poll_cq(&a, 0x22)) {
                cleanup_ep(&b); cleanup_ep(&a); return -1;
            }
            char err[128];
            if (verify_pattern(b.buf, opt->size, MODE_WRITE, (uint32_t)i, err, sizeof(err))) {
                fprintf(stderr, "selftest WRITE verify failed: %s\n", err);
                cleanup_ep(&b); cleanup_ep(&a); return -1;
            }
        }
        if (opt->mode & MODE_READ) {
            fill_pattern(b.buf, opt->size, MODE_READ, (uint32_t)i);
            memset(a.buf, 0, opt->size);
            if (post_send_ep(&a, IBV_WR_RDMA_READ, &b, opt->size, 0x23) || poll_cq(&a, 0x23)) {
                cleanup_ep(&b); cleanup_ep(&a); return -1;
            }
            char err[128];
            if (verify_pattern(a.buf, opt->size, MODE_READ, (uint32_t)i, err, sizeof(err))) {
                fprintf(stderr, "selftest READ verify failed: %s\n", err);
                cleanup_ep(&b); cleanup_ep(&a); return -1;
            }
        }
    }
    printf("verbs selftest passed: dev-a=%s dev-b=%s mode=%s iters=%d size=%zu gid-index=%d\n",
           a.name, b.name, mode_name(opt->mode), opt->iters, opt->size, gid_index);
    cleanup_ep(&b);
    cleanup_ep(&a);
    return 0;
}
