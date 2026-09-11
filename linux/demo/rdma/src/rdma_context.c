#define _POSIX_C_SOURCE 200112L

#include "rdma_demo.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum wr_id {
    WR_RECV = 0x100,
    WR_SEND = 0x200,
    WR_RDMA = 0x300,
};

static int die_errno(const char *what)
{
    fprintf(stderr, "%s: %s\n", what, strerror(errno));
    return -1;
}

static int die_wc(const char *what, enum ibv_wc_status status)
{
    fprintf(stderr, "%s: completion status=%s(%d)\n", what,
            ibv_wc_status_str(status), status);
    return -1;
}

void rdma_resources_init(struct rdma_resources *res, int verbose)
{
    memset(res, 0, sizeof(*res));
    res->verbose = verbose;
}

static int wait_cm_event(struct rdma_event_channel *ec,
                         enum rdma_cm_event_type expect,
                         struct rdma_cm_event **out)
{
    struct rdma_cm_event *ev = NULL;

    /*
     * rdma_get_cm_event() 从 RDMA CM event channel 取连接管理事件。
     * 硬件视角：这里通常还没有直接创建 WQE；内核 RDMA-CM/驱动在处理地址解析、
     * 路由解析、连接请求/响应等控制面事件，最终会驱动 QP 状态机迁移。
     */
    if (rdma_get_cm_event(ec, &ev) != 0) {
        return die_errno("rdma_get_cm_event");
    }
    if (ev->event != expect) {
        fprintf(stderr, "unexpected CM event: got %s, expect %s\n",
                rdma_event_str(ev->event), rdma_event_str(expect));
        rdma_ack_cm_event(ev);
        return -1;
    }
    *out = ev;
    return 0;
}

static int ack_cm_event(struct rdma_cm_event *ev)
{
    /* rdma_ack_cm_event() 归还事件对象；不创建硬件资源，但释放 CM 事件引用。 */
    if (rdma_ack_cm_event(ev) != 0) {
        return die_errno("rdma_ack_cm_event");
    }
    return 0;
}

static int alloc_aligned(char **p, size_t len)
{
    void *mem = NULL;
    int rc = posix_memalign(&mem, sysconf(_SC_PAGESIZE), len);
    if (rc != 0) {
        errno = rc;
        return die_errno("posix_memalign");
    }
    memset(mem, 0, len);
    *p = mem;
    return 0;
}

static int poll_one(struct rdma_resources *res, enum wr_id expect)
{
    struct ibv_wc wc;
    for (;;) {
        /*
         * ibv_poll_cq() 轮询 Completion Queue。
         * 硬件视角：HCA 完成 SQ/RQ 中的 WQE 后，会把 CQE 写入 CQ；
         * 此调用从 CQ 中取 CQE，应用据此知道 SEND/RECV/RDMA READ/WRITE 是否完成。
         */
        int n = ibv_poll_cq(res->cq, 1, &wc);
        if (n < 0) {
            fprintf(stderr, "ibv_poll_cq failed\n");
            return -1;
        }
        if (n == 0) {
            continue;
        }
        if (wc.status != IBV_WC_SUCCESS) {
            return die_wc("work completion failed", wc.status);
        }
        if ((enum wr_id)wc.wr_id != expect) {
            fprintf(stderr, "unexpected wr_id=0x%lx expect=0x%x\n",
                    (unsigned long)wc.wr_id, expect);
            return -1;
        }
        return 0;
    }
}

static int build_qp_and_memory(struct rdma_resources *res, size_t data_size, int cq_depth)
{
    res->data_size = data_size;
    if (alloc_aligned(&res->send_buf, RDMA_DEMO_CTRL_SIZE) != 0 ||
        alloc_aligned(&res->recv_buf, RDMA_DEMO_CTRL_SIZE) != 0 ||
        alloc_aligned(&res->data_buf, data_size) != 0) {
        return -1;
    }

    /*
     * ibv_alloc_pd() 分配 Protection Domain。
     * 硬件视角：PD 是 HCA 访问控制域，后续 MR/QP 都绑定到同一个 PD；
     * HCA 用 PD 编号校验 QP 是否有权限访问某个 lkey/rkey 对应的内存区域。
     */
    res->pd = ibv_alloc_pd(res->id->verbs);
    if (!res->pd) {
        return die_errno("ibv_alloc_pd");
    }

    /*
     * ibv_create_cq() 创建 Completion Queue。
     * 硬件视角：驱动在 HCA 上建立 CQ/CQC（Completion Queue Context）及 CQE ring；
     * SQ/RQ WQE 完成后，HCA 向这里写 CQE，本 demo 的 send/recv/rdma completion 共用一个 CQ。
     */
    res->cq = ibv_create_cq(res->id->verbs, cq_depth, NULL, NULL, 0);
    if (!res->cq) {
        return die_errno("ibv_create_cq");
    }

    /*
     * ibv_reg_mr() 注册控制面发送缓冲区。
     * 硬件视角：驱动 pin 住用户页，建立 MTT/MPT 等内存翻译/保护表项；
     * 返回 lkey，HCA 执行 SEND WQE 时用 lkey 校验并 DMA 读取该内存。
     */
    res->send_mr = ibv_reg_mr(res->pd, res->send_buf, RDMA_DEMO_CTRL_SIZE,
                              IBV_ACCESS_LOCAL_WRITE);
    if (!res->send_mr) {
        return die_errno("ibv_reg_mr(send)");
    }

    /*
     * ibv_reg_mr() 注册控制面接收缓冲区。
     * 硬件视角：HCA 收到对端 SEND 包并匹配到 RQ WQE 后，会 DMA 写入该 MR；
     * LOCAL_WRITE 权限允许本地 HCA 写入，lkey 会放入 RECV WQE 的 SGE。
     */
    res->recv_mr = ibv_reg_mr(res->pd, res->recv_buf, RDMA_DEMO_CTRL_SIZE,
                              IBV_ACCESS_LOCAL_WRITE);
    if (!res->recv_mr) {
        return die_errno("ibv_reg_mr(recv)");
    }

    /*
     * ibv_reg_mr() 注册数据缓冲区，并开放 REMOTE_READ/REMOTE_WRITE。
     * 硬件视角：除本地 lkey 外，驱动还生成 rkey；对端 RDMA READ/WRITE WQE
     * 携带 remote_addr+rkey 到达本 HCA 后，HCA 查询内存保护/翻译表，校验权限，
     * 然后直接 DMA 读/写该用户态页，不需要远端 CPU 参与数据搬运。
     */
    res->data_mr = ibv_reg_mr(res->pd, res->data_buf, data_size,
                              IBV_ACCESS_LOCAL_WRITE |
                              IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_WRITE);
    if (!res->data_mr) {
        return die_errno("ibv_reg_mr(data)");
    }

    struct ibv_qp_init_attr init_attr;
    memset(&init_attr, 0, sizeof(init_attr));
    init_attr.qp_type = IBV_QPT_RC;
    init_attr.send_cq = res->cq;
    init_attr.recv_cq = res->cq;
    init_attr.cap.max_send_wr = (uint32_t)cq_depth;
    init_attr.cap.max_recv_wr = (uint32_t)cq_depth;
    init_attr.cap.max_send_sge = 1;
    init_attr.cap.max_recv_sge = 1;
    init_attr.cap.max_inline_data = RDMA_DEMO_MAX_INLINE;

    /*
     * rdma_create_qp() 在 RDMA CM id 上创建 RC QP。
     * 硬件视角：驱动创建 QPC（Queue Pair Context），以及 SQ/RQ 两个 WQE ring；
     * QPC 记录 QPN、PSN、MTU、重传、访问标志、关联 CQ/PD 等字段。后续
     * ibv_post_send() 写 SQ doorbell，ibv_post_recv() 写 RQ doorbell。
     */
    if (rdma_create_qp(res->id, res->pd, &init_attr) != 0) {
        return die_errno("rdma_create_qp");
    }
    res->qp = res->id->qp;
    return 0;
}

int rdma_server_listen(struct rdma_resources *res, const char *port, int backlog)
{
    struct addrinfo hints;
    struct addrinfo *ai = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_flags = AI_PASSIVE;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    int rc = getaddrinfo(NULL, port, &hints, &ai);
    if (rc != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rc));
        return -1;
    }

    /*
     * rdma_create_event_channel() 创建 RDMA-CM 事件队列。
     * 硬件视角：这是用户态 CM 控制面 fd，不创建 QPC/CQ/WQE。
     */
    res->ec = rdma_create_event_channel();
    if (!res->ec) {
        freeaddrinfo(ai);
        return die_errno("rdma_create_event_channel");
    }

    /*
     * rdma_create_id() 创建 CM id，它是连接端点的控制面句柄。
     * 硬件视角：此时尚未创建 QP；id 后续会绑定地址、接收连接，并承载 QP。
     */
    if (rdma_create_id(res->ec, &res->listen_id, NULL, RDMA_PS_TCP) != 0) {
        freeaddrinfo(ai);
        return die_errno("rdma_create_id(listen)");
    }

    /* rdma_bind_addr() 绑定监听地址；控制面操作，不创建 SQ/RQ/CQ。 */
    if (rdma_bind_addr(res->listen_id, ai->ai_addr) != 0) {
        freeaddrinfo(ai);
        return die_errno("rdma_bind_addr");
    }
    freeaddrinfo(ai);

    /*
     * rdma_listen() 开始监听 RDMA CM 连接请求。
     * 硬件视角：仍属于连接管理控制面；收到请求后会产生 CONNECT_REQUEST 事件。
     */
    if (rdma_listen(res->listen_id, backlog) != 0) {
        return die_errno("rdma_listen");
    }
    printf("server listening on port %s\n", port);
    return 0;
}

int rdma_server_accept_one(struct rdma_resources *res, size_t data_size, int cq_depth)
{
    struct rdma_cm_event *ev = NULL;
    if (wait_cm_event(res->ec, RDMA_CM_EVENT_CONNECT_REQUEST, &ev) != 0) {
        return -1;
    }
    res->id = ev->id;
    if (ack_cm_event(ev) != 0) {
        return -1;
    }
    if (build_qp_and_memory(res, data_size, cq_depth) != 0) {
        return -1;
    }

    struct rdma_conn_param param;
    memset(&param, 0, sizeof(param));
    param.initiator_depth = 1;
    param.responder_resources = 1;
    param.retry_count = 7;
    param.rnr_retry_count = 7;

    /*
     * rdma_accept() 接受连接并推动 QP 状态机 INIT->RTR->RTS。
     * 硬件视角：内核/驱动把 QPC 写入对端 QPN、PSN、路径、ACK/重传等属性；
     * 成功后 QP 可收发 RC 报文，SQ/RQ/CQ 已可被硬件消费/产生 CQE。
     */
    if (rdma_accept(res->id, &param) != 0) {
        return die_errno("rdma_accept");
    }
    if (wait_cm_event(res->ec, RDMA_CM_EVENT_ESTABLISHED, &ev) != 0) {
        return -1;
    }
    if (ack_cm_event(ev) != 0) {
        return -1;
    }
    puts("server connection established");
    return 0;
}

int rdma_client_connect(struct rdma_resources *res, const char *addr, const char *port,
                        size_t data_size, int cq_depth)
{
    struct addrinfo hints;
    struct addrinfo *dst = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    int rc = getaddrinfo(addr, port, &hints, &dst);
    if (rc != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rc));
        return -1;
    }

    res->ec = rdma_create_event_channel();
    if (!res->ec) {
        freeaddrinfo(dst);
        return die_errno("rdma_create_event_channel");
    }
    if (rdma_create_id(res->ec, &res->id, NULL, RDMA_PS_TCP) != 0) {
        freeaddrinfo(dst);
        return die_errno("rdma_create_id(client)");
    }

    /*
     * rdma_resolve_addr() 根据目标 IP 选择本地 RDMA 设备/GID/源地址。
     * Soft-RoCE 场景会按 IP 路由映射到 rxe 设备；这是“同机两个 rxe 设备”测试的关键。
     */
    if (rdma_resolve_addr(res->id, NULL, dst->ai_addr, 2000) != 0) {
        freeaddrinfo(dst);
        return die_errno("rdma_resolve_addr");
    }
    freeaddrinfo(dst);

    struct rdma_cm_event *ev = NULL;
    if (wait_cm_event(res->ec, RDMA_CM_EVENT_ADDR_RESOLVED, &ev) != 0) {
        return -1;
    }
    if (ack_cm_event(ev) != 0) {
        return -1;
    }

    /* rdma_resolve_route() 解析 RDMA 路径；硬件相关路径属性将用于后续 QPC。 */
    if (rdma_resolve_route(res->id, 2000) != 0) {
        return die_errno("rdma_resolve_route");
    }
    if (wait_cm_event(res->ec, RDMA_CM_EVENT_ROUTE_RESOLVED, &ev) != 0) {
        return -1;
    }
    if (ack_cm_event(ev) != 0) {
        return -1;
    }

    if (build_qp_and_memory(res, data_size, cq_depth) != 0) {
        return -1;
    }

    struct rdma_conn_param param;
    memset(&param, 0, sizeof(param));
    param.initiator_depth = 1;
    param.responder_resources = 1;
    param.retry_count = 7;
    param.rnr_retry_count = 7;

    /*
     * rdma_connect() 发起连接并迁移 QP 状态。
     * 硬件视角：驱动根据 CM 路由信息填充 QPC，并在连接建立后让 RC QP 进入 RTS；
     * 此后 SQ doorbell 触发 HCA 发送包，RQ 接收对端 SEND。
     */
    if (rdma_connect(res->id, &param) != 0) {
        return die_errno("rdma_connect");
    }
    if (wait_cm_event(res->ec, RDMA_CM_EVENT_ESTABLISHED, &ev) != 0) {
        return -1;
    }
    if (ack_cm_event(ev) != 0) {
        return -1;
    }
    puts("client connection established");
    return 0;
}

int rdma_post_recv(struct rdma_resources *res)
{
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)res->recv_buf;
    sge.length = RDMA_DEMO_CTRL_SIZE;
    sge.lkey = res->recv_mr->lkey;

    struct ibv_recv_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = WR_RECV;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    struct ibv_recv_wr *bad = NULL;
    /*
     * ibv_post_recv() 向 RQ 提交一个 Receive WQE。
     * 硬件视角：驱动把包含 addr/len/lkey 的 WQE 写入 RQ ring，并敲 doorbell；
     * HCA 收到对端 SEND 后从 RQ 消费该 WQE，把 payload DMA 到 recv_buf，再写 CQE。
     */
    if (ibv_post_recv(res->qp, &wr, &bad) != 0) {
        return die_errno("ibv_post_recv");
    }
    return 0;
}

int rdma_send_msg(struct rdma_resources *res, const struct demo_msg *msg,
                  const void *payload, size_t payload_len)
{
    if (sizeof(*msg) + payload_len > RDMA_DEMO_CTRL_SIZE) {
        fprintf(stderr, "send message too large\n");
        return -1;
    }
    memcpy(res->send_buf, msg, sizeof(*msg));
    if (payload_len) {
        memcpy(res->send_buf + sizeof(*msg), payload, payload_len);
    }

    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)res->send_buf;
    sge.length = (uint32_t)(sizeof(*msg) + payload_len);
    sge.lkey = res->send_mr->lkey;

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = WR_SEND;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    if (sge.length <= RDMA_DEMO_MAX_INLINE) {
        wr.send_flags |= IBV_SEND_INLINE;
    }
    wr.sg_list = &sge;
    wr.num_sge = 1;

    struct ibv_send_wr *bad = NULL;
    /*
     * ibv_post_send(IBV_WR_SEND) 向 SQ 提交 Send WQE。
     * 硬件视角：WQE 进入 SQ，HCA 读取 SGE 指向的本地内存并发送 RC SEND 包；
     * 对端必须预先有 RQ WQE。完成后本地 CQ 产生 IBV_WC_SEND CQE，对端 CQ 产生 RECV CQE。
     */
    if (ibv_post_send(res->qp, &wr, &bad) != 0) {
        return die_errno("ibv_post_send(SEND)");
    }
    return poll_one(res, WR_SEND);
}

int rdma_recv_msg(struct rdma_resources *res, struct demo_msg *msg,
                  void *payload, size_t payload_len)
{
    if (poll_one(res, WR_RECV) != 0) {
        return -1;
    }
    memcpy(msg, res->recv_buf, sizeof(*msg));
    if (payload && payload_len) {
        size_t n = msg->size < payload_len ? msg->size : payload_len;
        memcpy(payload, res->recv_buf + sizeof(*msg), n);
    }
    return 0;
}

int rdma_send_info(struct rdma_resources *res)
{
    struct demo_msg msg;
    memset(&msg, 0, sizeof(msg));
    msg.type = MSG_INFO;
    msg.remote_addr = (uintptr_t)res->data_buf;
    msg.rkey = res->data_mr->rkey;
    msg.size = (uint32_t)res->data_size;
    snprintf(msg.text, sizeof(msg.text), "addr=0x%lx rkey=0x%x size=%u",
             (unsigned long)msg.remote_addr, msg.rkey, msg.size);
    return rdma_send_msg(res, &msg, NULL, 0);
}

int rdma_recv_info(struct rdma_resources *res)
{
    struct demo_msg msg;
    if (rdma_recv_msg(res, &msg, NULL, 0) != 0) {
        return -1;
    }
    if (msg.type != MSG_INFO) {
        fprintf(stderr, "expected MSG_INFO, got %u\n", msg.type);
        return -1;
    }
    res->peer_addr = msg.remote_addr;
    res->peer_rkey = msg.rkey;
    if (res->verbose) {
        printf("peer %s\n", msg.text);
    }
    return 0;
}

int rdma_write_peer(struct rdma_resources *res, enum demo_mode mode, uint32_t seq, size_t len)
{
    fill_pattern(res->data_buf, len, mode, seq);

    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)res->data_buf;
    sge.length = (uint32_t)len;
    sge.lkey = res->data_mr->lkey;

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = WR_RDMA;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.rdma.remote_addr = res->peer_addr;
    wr.wr.rdma.rkey = res->peer_rkey;

    struct ibv_send_wr *bad = NULL;
    /*
     * ibv_post_send(IBV_WR_RDMA_WRITE) 提交 RDMA Write WQE。
     * 硬件视角：本端 HCA 从本地 MR DMA 读取数据，携带 remote_addr+rkey 发出 RDMA WRITE；
     * 远端 HCA 查询 rkey 对应 MPT/MTT，校验 REMOTE_WRITE 权限后直接 DMA 写远端内存。
     * 远端 CPU/RQ/CQ 不参与数据面；本端收到 ACK 后在本端 CQ 生成 completion。
     */
    if (ibv_post_send(res->qp, &wr, &bad) != 0) {
        return die_errno("ibv_post_send(RDMA_WRITE)");
    }
    return poll_one(res, WR_RDMA);
}

int rdma_read_peer(struct rdma_resources *res, enum demo_mode mode, uint32_t seq, size_t len)
{
    memset(res->data_buf, 0, len);

    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)res->data_buf;
    sge.length = (uint32_t)len;
    sge.lkey = res->data_mr->lkey;

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = WR_RDMA;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.rdma.remote_addr = res->peer_addr;
    wr.wr.rdma.rkey = res->peer_rkey;

    struct ibv_send_wr *bad = NULL;
    /*
     * ibv_post_send(IBV_WR_RDMA_READ) 提交 RDMA Read WQE。
     * 硬件视角：本端 HCA 发起 READ 请求；远端 HCA 用 rkey 校验 REMOTE_READ 权限，
     * 从远端内存 DMA 读出并回包；本端 HCA 把响应数据 DMA 写入本地 SGE，最后写本端 CQE。
     */
    if (ibv_post_send(res->qp, &wr, &bad) != 0) {
        return die_errno("ibv_post_send(RDMA_READ)");
    }
    if (poll_one(res, WR_RDMA) != 0) {
        return -1;
    }

    char err[128];
    if (verify_pattern(res->data_buf, len, mode, seq, err, sizeof(err)) != 0) {
        fprintf(stderr, "RDMA READ verify failed: %s\n", err);
        return -1;
    }
    return 0;
}

int rdma_disconnect_peer(struct rdma_resources *res)
{
    if (!res->id) {
        return 0;
    }
    /* rdma_disconnect() 发起断连，驱动/CM 会将 QP 迁移到 ERROR/RESET 类状态。 */
    if (rdma_disconnect(res->id) != 0 && errno != ENOTCONN && errno != EINVAL) {
        return die_errno("rdma_disconnect");
    }
    return 0;
}

void rdma_resources_cleanup(struct rdma_resources *res)
{
    if (res->id && res->id->qp) {
        /* rdma_destroy_qp() 销毁 QPC/SQ/RQ 及相关硬件上下文。 */
        rdma_destroy_qp(res->id);
        res->qp = NULL;
    }
    if (res->data_mr) {
        /* ibv_dereg_mr() 注销 MR，释放 HCA 内存保护/翻译表项并解除 pin 页。 */
        ibv_dereg_mr(res->data_mr);
    }
    if (res->recv_mr) {
        ibv_dereg_mr(res->recv_mr);
    }
    if (res->send_mr) {
        ibv_dereg_mr(res->send_mr);
    }
    if (res->cq) {
        /* ibv_destroy_cq() 销毁 CQ/CQC 和 CQE ring。 */
        ibv_destroy_cq(res->cq);
    }
    if (res->pd) {
        ibv_dealloc_pd(res->pd);
    }
    if (res->id) {
        rdma_destroy_id(res->id);
    }
    if (res->listen_id) {
        rdma_destroy_id(res->listen_id);
    }
    if (res->ec) {
        rdma_destroy_event_channel(res->ec);
    }
    free(res->send_buf);
    free(res->recv_buf);
    free(res->data_buf);
}
