# A 验证通过后迁移到 B（麒麟）

## 从 A 的成功修改中合入的内容

| A 的修改 | 内置 CANN 方案处理 |
| --- | --- |
| ACL 头文件查找增加 `aarch64-linux/include` | CMake 已支持；补齐 `build.sh` 前置检查，按 `uname -m` 识别架构 |
| ACL 库查找增加 `aarch64-linux/lib64` | 已支持，并保留 CANN 9 的 `lib64` |
| ACL 库查找增加 `fwkacllib/lib64` | 新增到 CMake 和运行时库路径；仍不搜索其他系统 CANN 或 `devlib` 桩库 |
| Ascend C 增加 `aarch64-linux/tikcpp/ascendc_kernel_cmake` | 新增为标准候选，并打印实际命中路径；保留唯一探测和显式路径覆盖 |
| `ascendc.cmake` 只 include 一次 | 保持原有做法；NV12 和 RGB 内核在同一文件声明，不重复 include |
| 不挂载 YOLO11 OM | 新的独立 Compose 已无此挂载，原 Compose 不修改 |
| 仅加载五个 YOLO26 NV12 模型 | `models_to_load.container.txt` 已与 A 的清单相同，无须删改原全模型清单 |

A 使用宿主机 Toolkit 成功，不代表 A 的驱动与 CANN 9 已匹配。新方案固定容器内 CANN 9.0.0，
必须重新编译和实际推理验证。CMake 能识别旧目录，不等于 Dockerfile 可以改用任意 CANN 镜像；
其复制路径仍固定为 `cann-9.0.0`。两台机器都不要升级/降级现有驱动作为默认操作。

## A：不替换原服务的验证

在 A 的更新后 Triton 仓库目录中进行。旧服务的 `.env` 和模型文件保留，不覆盖。

```bash
uname -m
npu-smi info
npu-smi info -m
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
```

当前 Dockerfile 只支持原生 ARM64（`aarch64`）。若 A 是 x86_64，不能直接按下面步骤在 A
构建并运行当前镜像，也不能将 A 的 x86 后端搬到 B；应换 ARM64 构建主机，或在 B 原生构建。

首次创建 A 的独立配置（已有同名文件时只合并，不覆盖）：

```bash
cp .env.cann.example .env.cann.a
```

编辑 `.env.cann.a`，至少调整以下值：

```dotenv
TRITON_CANN_CONTAINER_NAME=triton-ascend-cann9-a-test
TRITON_HTTP_PORT=55245
TRITON_GRPC_PORT=55246
TRITON_METRICS_PORT=55247
ASCEND_DEVICE_ID=0
YOLO26_MODEL_HOST_PATH=/A机器实际的绝对路径/yolo26s.om
```

设备 0 仅为示例，按 A 的映射与占用选择；不要沿用 B 的设备 2。若没有空闲 NPU，应在维护
窗口验证，不要默认让新旧服务挤在同一设备。`privileged: true` 不提供设备隔离。
端口 55245～55247 也应先确认未占用。实际配置中不能保留上述中文路径占位符。

旧 Compose 1.22 没有现代 `--env-file` 参数。用受信任的 shell 兼容环境文件导入独立配置，
同名 shell 变量会覆盖自动读取的旧 `.env`。以下示例必须在同一 Bash 终端执行，之后的每个
Compose 命令都保留相同的 `-p triton-cann-a-test`：

```bash
set -a
source .env.cann.a
set +a

test -f "$YOLO26_MODEL_HOST_PATH"
test -s "$YOLO26_MODEL_HOST_PATH"
docker-compose -p triton-cann-a-test -f docker-compose.cann.yml config
docker-compose -p triton-cann-a-test -f docker-compose.cann.yml build triton-ascend
```

先检查动态库，再检查 ACL（前者输出不得有 `not found`）：

```bash
docker-compose -p triton-cann-a-test -f docker-compose.cann.yml run --rm --no-deps triton-ascend \
  bash -c 'echo "$ASCEND_HOME_PATH"; ldd /models/YOLO26_DET_PRE/1/libtriton_yolo26_infer.so'

docker-compose -p triton-cann-a-test -f docker-compose.cann.yml run --rm --no-deps triton-ascend \
  python3 /opt/triton-ascend/check-device.py
```

检查成功后启动测试服务：

```bash
docker-compose -p triton-cann-a-test -f docker-compose.cann.yml up -d --no-deps triton-ascend
docker-compose -p triton-cann-a-test -f docker-compose.cann.yml logs --tail=200 triton-ascend
curl -f "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/models/YOLO26_DET_PRE_YUV_ENSEMBLE/ready"
```

确认模型 READY 后，在安装了客户端依赖的 A 环境中发送真实图片：

```bash
python3 workspace/triton_client/visualize.py --protocol http \
  --url "127.0.0.1:${TRITON_HTTP_PORT}" \
  --model YOLO26_DET_PRE_YUV_ENSEMBLE --image /A机器实际的测试图片.jpg
```

使用与旧服务相同的图片、OM、置信度比较框和类别，再测试连续多次请求、真实 NV12 帧、
耗时和 NPU 占用。客户端与现有 AIDetection 生产端点先不切换，不能用旧服务的响应冒充新服务验证。
新镜像的 CANN 9 编译会重新产生 `.so`，不会直接复用 A 原有的宿主机编译产物。

## A -> B：迁移已验证的镜像，不在 B 重编译

A 验证成功后，记录镜像 ID 并导出最终镜像：

```bash
docker image inspect "$TRITON_CANN_IMAGE" --format '{{.Id}} {{.Os}}/{{.Architecture}}'
docker save -o triton-ascend-cann9.docker.tar "$TRITON_CANN_IMAGE"
sha256sum triton-ascend-cann9.docker.tar
sha256sum "$YOLO26_MODEL_HOST_PATH"
```

将镜像归档、更新后的部署源码、测试图片和对应 OM 同步到 B，B 不复制 A 的 `.env`。
镜像不包含 OM；模型与归档分别校验。整个 CANN 和后端已在最终镜像中，B 无需再次拉取两个
基础镜像或在宿主机安装 Toolkit。不要使用 `docker export/import` 代替 `save/load`。

B 在新终端和自己的 Triton 项目目录中执行：

```bash
sha256sum triton-ascend-cann9.docker.tar
docker load -i triton-ascend-cann9.docker.tar
cp .env.cann.example .env.cann.b
```

首次创建时才复制。编辑 `.env.cann.b`：OM 改为 B 的实际路径；
`TRITON_CANN_IMAGE` 必须与导入标签一致；使用独立容器名；端口按 B 实际可用值配置。
B 的 `ASCEND_DEVICE_ID=2` 来自此前 NPU 5/Chip 0 的映射，启动前仍需确认未被占用。

```bash
set -a
source .env.cann.b
set +a

docker image inspect "$TRITON_CANN_IMAGE" --format '{{.Id}} {{.Os}}/{{.Architecture}}'
test -f "$YOLO26_MODEL_HOST_PATH"
test -s "$YOLO26_MODEL_HOST_PATH"
sha256sum "$YOLO26_MODEL_HOST_PATH"
docker-compose -p triton-cann-b -f docker-compose.cann.yml config
docker-compose -p triton-cann-b -f docker-compose.cann.yml run --rm --no-deps triton-ascend \
  bash -c 'echo "$ASCEND_HOME_PATH"; ldd /models/YOLO26_DET_PRE/1/libtriton_yolo26_infer.so'
docker-compose -p triton-cann-b -f docker-compose.cann.yml run --rm --no-deps triton-ascend \
  python3 /opt/triton-ascend/check-device.py
docker-compose -p triton-cann-b -f docker-compose.cann.yml up -d --no-deps triton-ascend
docker-compose -p triton-cann-b -f docker-compose.cann.yml logs --tail=200 triton-ascend
```

这里不执行 `build` 或 `--build`。先确认 B 的镜像 ID 与 A 一致，再重复模型 READY、相同图片的
真实推理和 NPU 位置验证。A 成功不能代替 B 的驱动/固件、内核、旧 Docker 兼容性验收。
B 全部通过后，才将 AIDetection 端点切换到 B 的 HTTP/gRPC 服务。
