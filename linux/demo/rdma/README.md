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
    └── run_local_test.sh    # 一键启动本地 selftest/CM 测试
```

## 安装依赖

Ubuntu / Debian：

```bash
sudo apt-get update
sudo apt-get install -y build-essential rdma-core librdmacm-dev libibverbs-dev iproute2 kmod
```

如果系统安装的是厂商 OFED（例如 MLNX_OFED），但 `ibv_devinfo` 报 `couldn't load driver 'librxe-rdmavXX.so'`，需要确保安装了与当前 `libibverbs` ABI 匹配的 RXE userspace provider。否则 `rdma_rxe` 内核设备虽然能创建，用户态 verbs 仍无法打开 RXE 设备。

如果测试已经通过，但输出里有 `libibverbs: Warning: couldn't load driver 'libmthca-rdmav34.so'` 这类消息，通常只是 `/etc/libibverbs.d/` 里存在其它 HCA provider 配置，而机器上没有安装对应硬件/动态库；RXE provider 能正常打开时不影响 demo。`run_local_test.sh` 默认会过滤这类已知噪音；如需查看完整 libibverbs warning，可设置 `SHOW_IBV_WARNINGS=1`。

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

默认一键测试会启动一对独立的 server/client 进程，TCP 控制面分别绑定 `192.168.130.2` / `192.168.130.1`，verbs 数据面分别显式打开 `rxe_demo1` / `rxe_demo0`，不依赖 RDMA CM 路由选择，然后依次完整测试 `send`、`write`、`read`、`all` 四个 case。

脚本会显式指定并校验两端使用不同的 IP、eth netdev 和 RDMA NIC，默认映射为：

```text
client: ip=192.168.130.1 netdev=rdma-veth0 rdma=rxe_demo0
server: ip=192.168.130.2 netdev=rdma-veth1 rdma=rxe_demo1
```

如果 client/server IP 相同、RDMA NIC 相同，或 RDMA NIC 没有绑定到预期 netdev，脚本会直接失败退出，避免误共用同一个 RDMA 网卡。

每个 case 都不是“只创建资源”：

- `send`：A 端 `IBV_WR_SEND`，B 端预投递 RQ WQE 接收，并校验接收 buffer 的 pattern。
- `write`：A 端 `IBV_WR_RDMA_WRITE` 直接写 B 端 MR，随后检查 B 端内存 pattern。
- `read`：B 端准备 MR 数据，A 端 `IBV_WR_RDMA_READ` 读回本地，并校验本地 buffer pattern。
- `all`：在一个进程中连续执行上述三类操作。

只要任意 case 数据校验失败、CQE 状态异常或 WQE 提交失败，脚本会立即退出非 0。

可以通过环境变量调整参数：

```bash
SIZE=2048 ITERS=5 \
CLIENT_IP=192.168.130.1 CLIENT_NETDEV=rdma-veth0 CLIENT_RDMA_DEV=rxe_demo0 \
SERVER_IP=192.168.130.2 SERVER_NETDEV=rdma-veth1 SERVER_RDMA_DEV=rxe_demo1 \
GID_INDEX=0 ./scripts/run_local_test.sh
```

只运行部分 case：

```bash
CASES="send write" ./scripts/run_local_test.sh
# 兼容旧用法：MODE=write ./scripts/run_local_test.sh
```

显示完整 libibverbs provider warning：

```bash
SHOW_IBV_WARNINGS=1 ./scripts/run_local_test.sh
```

也可以直接运行：

```bash
./rdma_demo --selftest --client-dev rxe_demo0 --server-dev rxe_demo1 --mode all --size 1024 --iters 3 --gid-index 0
```

如果只想运行单进程自测而不启动 server/client 进程：

```bash
SELFTEST=1 ./scripts/run_local_test.sh
```

### 手工启动 server/client

终端 1：

```bash
./rdma_demo --server --bind-addr 192.168.130.2 --server-dev rxe_demo1 --port 7471 --size 1024 --iters 3 --verbose
```

终端 2：

```bash
./rdma_demo --client --bind-addr 192.168.130.1 --addr 192.168.130.2 --client-dev rxe_demo0 --port 7471 --mode all --size 1024 --iters 3 --verbose
```

上面这组命令会进入“TCP 控制面 + raw verbs 数据面”模式：TCP 连接只负责交换 QPN/PSN/GID/rkey/addr 元数据，RC QP 由程序手工从 `INIT` 迁移到 `RTR` / `RTS`，数据真正通过指定的 RDMA NIC 完成 SEND / WRITE / READ。因此同一台机器、同一个 network namespace 中，即使目标 IP 被内核识别为本地地址，也不会遇到 RDMA CM 地址解析选到 `lo` 的问题。

如果不传 `--client-dev` / `--server-dev`，server/client 会回退到 RDMA CM 模式。RDMA CM 模式不能直接传 `rxe_demo0` / `rxe_demo1` 这类 RDMA 设备名；它通过本地/目标 IP 地址和内核路由选择 eth 设备，再映射到该 eth 设备上的 RDMA 设备。因此：

- server 用 `--bind-addr` 绑定监听 IP，例如 `192.168.130.2`，让 CM 监听这个 IP 对应的 netdev/RDMA 设备。
- client 用 `--bind-addr` 绑定源 IP，例如 `192.168.130.1`，让 CM 从这个 IP 对应的 netdev/RDMA 设备发起连接。
- `--addr` 是 server 的目标 IP。

注意：RDMA CM 模式在一台机器的同一个 network namespace 内连接本机另一个 IP 时，Linux 仍可能通过 `local/lo` 路由处理地址，导致 RDMA CM 地址解析不选择 RXE 设备；即使加了 `--bind-addr`，目标地址如果显示为 `local ... dev lo`，也不适合用 CM 模式验证。此时请使用上面的 `--client-dev` / `--server-dev` raw verbs server/client 模式，或使用 `--selftest`。

如果只启动 client 而没有先启动 server，client 会在 RDMA CM 连接阶段等待/失败。可以用 `--cm-timeout-ms` 缩短等待时间并看到明确错误，例如：

```bash
./rdma_demo --client --addr 192.168.130.2 --port 7471 --mode all --cm-timeout-ms 1500
```

在同机当前 namespace 中可用下面命令确认 `192.168.130.2` 是否被内核识别成本地地址；如果输出包含 `local ... dev lo`，说明 RDMA CM server/client 模式不适合用这个拓扑验证，请使用 `--selftest`：

```bash
ip route get 192.168.130.2
```

预期输出类似：

```text
verbs client connected: rdma-dev=rxe_demo0 qpn=33 peer-qpn=33
client SEND seq=0 ok
client RDMA WRITE seq=0 ok
client RDMA READ seq=0 ok
...
verbs client tests passed: mode=all iters=3 size=1024
```

## 命令行参数

```text
--server / --client        选择服务端或客户端
--addr ADDR               client 连接的 server IP
--bind-addr ADDR          server/client 绑定的本地 TCP/RDMA-CM IP
--port PORT               TCP/RDMA-CM 端口，默认 7471
--mode send|write|read|all 测试模式，默认 all
--size SIZE               每次操作的数据长度，支持 K/M 后缀；当前 SEND 控制缓冲最大约 3.9KiB
--iters N                 每种模式迭代次数，默认 10
--cq-depth N              CQ/SQ/RQ 深度，默认 64
--cm-timeout-ms N         RDMA CM 事件等待超时，默认 5000 ms
--gid-index N             raw verbs/selftest 使用的 GID index，默认 0；RDMA CM 模式按路由选择 GID
--selftest                单进程 verbs 自测模式，不依赖 RDMA CM
--client-dev NAME         client/发起端 RDMA 设备名；server/client 模式中启用 raw verbs TCP 控制面
--server-dev NAME         server/响应端 RDMA 设备名；server/client 模式中启用 raw verbs TCP 控制面
--dev-a NAME              --client-dev 兼容别名
--dev-b NAME              --server-dev 兼容别名
--verbose                 打印每次操作日志
```

## 测试流程说明

1. raw verbs server/client 模式通过 TCP 控制连接监听/连接端口；RDMA CM 模式则通过 RDMA CM 监听/连接。
2. raw verbs 模式由 `--client-dev` / `--server-dev` 显式打开 RDMA 设备；RDMA CM 模式由内核路由选择设备。
3. 双方创建 PD/CQ/QP，注册控制缓冲和数据缓冲 MR。
4. 双方交换各自 QPN/PSN/GID 以及数据 MR 的 `remote_addr + rkey`；raw verbs 模式手工迁移 QP 到 `RTR` / `RTS`。
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
