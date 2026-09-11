# RDMA 通信 demo（SEND / RDMA WRITE / RDMA READ）

本目录提供一个基于 `librdmacm` + `libibverbs` 的 RC QP 示例程序，用于演示：

- `IBV_WR_SEND`：双边 SEND/RECV，需要接收端预先提交 RQ WQE。
- `IBV_WR_RDMA_WRITE`：单边写，发起端 HCA 直接写入对端注册内存。
- `IBV_WR_RDMA_READ`：单边读，发起端 HCA 直接读取对端注册内存。
- 同机 Soft-RoCE（`rdma_rxe`）测试：脚本会创建一对 `veth` 网卡，并在其上创建两个 RXE RDMA 设备；默认用 `--selftest` 在一个进程中直连两个 RC QP，避免本机 RDMA-CM 路由受 `lo`/netns 模式影响。

代码中所有关键 RDMA lib 调用旁都写了硬件视角注释，例如 PD/MR/CQ/QP/QPC/SQ/RQ/CQE/WQE 的创建和消费过程。

## 目录结构

```text
rdma/
├── Makefile
├── README.md
├── include/rdma_demo.h      # 公共结构、协议、函数声明
├── src/args.c               # 命令行参数解析
├── src/main.c               # server/client 测试流程
├── src/pattern.c            # 测试数据生成与校验
├── src/rdma_context.c       # RDMA CM / verbs 资源管理与操作封装
└── scripts/
    ├── setup_soft_roce.sh   # 创建/删除同机双 RXE 设备
    └── run_local_test.sh    # 一键启动 server/client 本地测试
```

## 安装依赖

Ubuntu / Debian：

```bash
sudo apt-get update
sudo apt-get install -y build-essential rdma-core librdmacm-dev libibverbs-dev iproute2 kmod
```

如果系统安装的是厂商 OFED（例如 MLNX_OFED），但 `ibv_devinfo` 报 `couldn't load driver 'librxe-rdmavXX.so'`，需要确保安装了与当前 `libibverbs` ABI 匹配的 RXE userspace provider。否则 `rdma_rxe` 内核设备虽然能创建，用户态 verbs 仍无法打开 RXE 设备。

CentOS / RHEL / Fedora：

```bash
sudo dnf install -y gcc make rdma-core rdma-core-devel libibverbs-devel librdmacm-devel iproute kmod
```

内核需要支持 Soft-RoCE 模块：

```bash
sudo modprobe rdma_rxe
```

## 创建同机两个 Soft-RoCE 设备

```bash
cd linux/demo/rdma
sudo ./scripts/setup_soft_roce.sh up
```

脚本会创建两个 veth 网卡，并在每个网卡上挂一个 RXE 设备：

- `rdma-veth0`：`192.168.130.1/24`，对应 `rxe_demo0`
- `rdma-veth1`：`192.168.130.2/24`，对应 `rxe_demo1`

查看设备：

```bash
rdma link show
ibv_devices
ibv_devinfo -d rxe_demo0
ibv_devinfo -d rxe_demo1
```

清理环境：

```bash
sudo ./scripts/setup_soft_roce.sh down
```

## 编译

```bash
cd linux/demo/rdma
make
```

生成二进制：`./rdma_demo`。

## 运行测试

### 一键本地测试

先执行 Soft-RoCE 配置，然后运行：

```bash
cd linux/demo/rdma
./scripts/run_local_test.sh
```

默认一键测试使用 `--selftest`，即在同一个进程内创建两个 verbs 端点，分别打开 `rxe_demo0` / `rxe_demo1`，手工迁移 QP 到 RTS，然后完整测试 SEND / RDMA WRITE / RDMA READ。

可以通过环境变量调整参数：

```bash
MODE=write SIZE=2048 ITERS=5 DEV_A=rxe_demo0 DEV_B=rxe_demo1 GID_INDEX=0 ./scripts/run_local_test.sh
```

也可以直接运行：

```bash
./rdma_demo --selftest --dev-a rxe_demo0 --dev-b rxe_demo1 --mode all --size 1024 --iters 3 --gid-index 0
```

### 手工启动 server/client

终端 1：

```bash
./rdma_demo --server --port 7471 --size 1024 --iters 3 --verbose
```

终端 2：

```bash
./rdma_demo --client --addr 192.168.130.2 --port 7471 --mode all --size 1024 --iters 3 --verbose
```

注意：server/client 模式使用 RDMA CM。在一台机器的同一个 network namespace 内连接本机另一个 IP 时，Linux 可能通过 `local/lo` 路由处理地址，导致 RDMA CM 地址解析不选择 RXE 设备。此时推荐使用上面的 `--selftest` 完成本机双 RXE 设备验证；跨两台机器或具备正确 RDMA netns 隔离时再使用 server/client 模式。

预期输出类似：

```text
client SEND test passed: iters=3 size=1024
client RDMA WRITE test passed: iters=3 size=1024
client RDMA READ test passed: iters=3 size=1024
server done: send=3 write=3 read=3
```

## 命令行参数

```text
--server / --client        选择服务端或客户端
--addr ADDR               client 连接的 server IP
--port PORT               RDMA CM 端口，默认 7471
--mode send|write|read|all 测试模式，默认 all
--size SIZE               每次操作的数据长度，支持 K/M 后缀；当前 SEND 控制缓冲最大约 3.9KiB
--iters N                 每种模式迭代次数，默认 10
--cq-depth N              CQ/SQ/RQ 深度，默认 64
--gid-index N             selftest 使用的 GID index，默认 0；RDMA CM 模式按路由选择 GID
--selftest                单进程 verbs 自测模式，不依赖 RDMA CM
--dev-a NAME              selftest 发起端 RDMA 设备名
--dev-b NAME              selftest 响应端 RDMA 设备名
--verbose                 打印每次操作日志
```

## 测试流程说明

1. server 通过 RDMA CM 监听端口。
2. client 解析 `--addr` 路由并连接 server。
3. 双方创建 PD/CQ/QP，注册控制缓冲和数据缓冲 MR。
4. 双方用 SEND 交换各自数据 MR 的 `remote_addr + rkey`。
5. client 按 `--mode` 执行：
   - `send`：client `IBV_WR_SEND` 发送 payload，server 从 RQ WQE 对应缓冲区校验。
   - `write`：client `IBV_WR_RDMA_WRITE` 写 server 数据 MR，然后用 SEND 通知 server 校验。
   - `read`：server 准备数据并 ACK，client `IBV_WR_RDMA_READ` 读回本地并校验。
6. client 发送 `DONE`，server 输出统计并退出。

## 常见问题

- `rdma_resolve_addr: No such device`：没有创建 RXE 设备，或目标 IP 没有路由到对应 netdev。
- `ibv_reg_mr: Cannot allocate memory`：检查 `ulimit -l`，必要时执行 `ulimit -l unlimited` 或配置 memlock。
- `RDMA_CM_EVENT_REJECTED` / 连接超时：确认 server 已启动、端口一致、防火墙未拦截。
- `rdma link add ... Operation not supported`：内核缺少 `rdma_rxe`，或系统未安装完整 `rdma-core/iproute2` RDMA 支持。
