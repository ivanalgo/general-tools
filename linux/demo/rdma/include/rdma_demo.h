#ifndef RDMA_DEMO_H
#define RDMA_DEMO_H

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <stddef.h>
#include <stdint.h>

#define RDMA_DEMO_DEFAULT_PORT "7471"
#define RDMA_DEMO_DEFAULT_CM_TIMEOUT_MS 5000
#define RDMA_DEMO_MAX_INLINE 220
#define RDMA_DEMO_CTRL_SIZE 4096
#define RDMA_DEMO_MAX_PAYLOAD (RDMA_DEMO_CTRL_SIZE - sizeof(struct demo_msg))

enum demo_role {
    ROLE_SERVER = 1,
    ROLE_CLIENT = 2,
    ROLE_SELFTEST = 3,
};

enum demo_mode {
    MODE_SEND = 1u << 0,
    MODE_WRITE = 1u << 1,
    MODE_READ = 1u << 2,
    MODE_ALL = MODE_SEND | MODE_WRITE | MODE_READ,
};

enum demo_msg_type {
    MSG_INFO = 1,
    MSG_SEND_PAYLOAD = 2,
    MSG_WRITE_VERIFY = 3,
    MSG_READ_PREPARE = 4,
    MSG_ACK = 5,
    MSG_DONE = 6,
};

struct demo_options {
    enum demo_role role;
    enum demo_mode mode;
    const char *addr;
    const char *bind_addr;
    const char *port;
    size_t size;
    int iters;
    int gid_index;
    int cq_depth;
    int verbose;
    int cm_timeout_ms;
    const char *dev_a;
    const char *dev_b;
};

struct demo_msg {
    uint32_t type;
    uint32_t mode;
    uint32_t seq;
    uint32_t size;
    uint64_t remote_addr;
    uint32_t rkey;
    uint32_t status;
    char text[128];
};

struct rdma_resources {
    struct rdma_event_channel *ec;
    struct rdma_cm_id *listen_id;
    struct rdma_cm_id *id;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *send_mr;
    struct ibv_mr *recv_mr;
    struct ibv_mr *data_mr;
    char *send_buf;
    char *recv_buf;
    char *data_buf;
    size_t data_size;
    uint64_t peer_addr;
    uint32_t peer_rkey;
    int verbose;
    int cm_timeout_ms;
};

int parse_options(int argc, char **argv, struct demo_options *opt);
void print_usage(const char *prog);

void fill_pattern(char *buf, size_t len, enum demo_mode mode, uint32_t seq);
int verify_pattern(const char *buf, size_t len, enum demo_mode mode, uint32_t seq,
                   char *err, size_t err_len);
const char *mode_name(enum demo_mode mode);

int run_server(const struct demo_options *opt);
int run_client(const struct demo_options *opt);
int run_verbs_selftest(const struct demo_options *opt);

void rdma_resources_init(struct rdma_resources *res, int verbose);
void rdma_resources_set_cm_timeout(struct rdma_resources *res, int timeout_ms);
void rdma_resources_cleanup(struct rdma_resources *res);
int rdma_server_listen(struct rdma_resources *res, const char *bind_addr,
                       const char *port, int backlog);
int rdma_server_accept_one(struct rdma_resources *res, size_t data_size, int cq_depth);
int rdma_client_connect(struct rdma_resources *res, const char *addr, const char *bind_addr,
                        const char *port, size_t data_size, int cq_depth);
int rdma_send_info(struct rdma_resources *res);
int rdma_recv_info(struct rdma_resources *res);
int rdma_post_recv(struct rdma_resources *res);
int rdma_send_msg(struct rdma_resources *res, const struct demo_msg *msg,
                  const void *payload, size_t payload_len);
int rdma_recv_msg(struct rdma_resources *res, struct demo_msg *msg,
                  void *payload, size_t payload_len);
int rdma_write_peer(struct rdma_resources *res, enum demo_mode mode, uint32_t seq, size_t len);
int rdma_read_peer(struct rdma_resources *res, enum demo_mode mode, uint32_t seq, size_t len);
int rdma_disconnect_peer(struct rdma_resources *res);

#endif
