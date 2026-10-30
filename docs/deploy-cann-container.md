# 麒麟 ARM64：内置 CANN 的 Triton 部署

本方案针对宿主机 `aarch64 + Ascend310P3 + HDK 25.5.1`，使用 Docker 18.09、
Compose 1.22 可解析的格式 2.4，不要求在宿主机安装 CANN Toolkit。
它是新增的独立部署入口，不覆盖原来的 `docker-compose.yml`，也不操作 SAM3。

需要先在 A 机器验证、再迁移到 B（麒麟）时，按
[A/B 分阶段验收与镜像迁移](validate-a-then-b.md) 操作，使用独立项目名、配置和测试端口。

## 软件边界

- 宿主机提供昇腾驱动、固件、设备节点，以及现有的 driver/develop 目录。
- 构建阶段使用与现有 SAM3 相同版本的官方 CANN 9.0.0、310P、Ubuntu 22.04 镜像。
- 后端和 Ascend C 内核在 CANN 镜像中重新编译；运行镜像基于 Triton 24.10。
- 最终镜像携带 `/usr/local/Ascend/cann-9.0.0` 和 Toolkit 兼容入口，不挂载宿主机 Toolkit。
- 不直接复制 SAM3 的业务代码、模型、NNAL 或宿主机驱动到镜像中。
- 构建阶段不需要访问 NPU；实际 ACL 初始化、OM 加载、推理必须在服务器验证。
- CANN 镜像的版本、架构、SOC 不能随意更换；这里只验证构建目录兼容，不承诺旧 OM 无需重新转换。

不要将本文件的 `-f docker-compose.cann.yml` 与原 Compose 叠加，否则旧挂载可能被保留。
不要挂载整个宿主机 `/usr/local/Ascend`，这会遮住镜像内的 CANN。
启动脚本始终优先查找宿主机 `driver/lib64/driver`、`driver/lib64/common`、`driver/lib64`，
再查找镜像 CANN 和已有环境路径；不要把 `develop/lib64` 放在真实驱动之前，以免加载另一份 HAL。
以下统一使用 `-p triton-cann9` 隔离旧服务的 Compose 项目；若已有服务占用默认端口，
还必须另外改端口和容器名。不同项目名不能隔离 NPU 计算资源。

## 1. 同步代码与配置

将修改后的本仓库同步到服务器的 Triton 项目目录。下面所有命令均在该目录执行，
不是在 rtspGrpcServer 或 SAM3 目录执行。

首次配置时才执行以下复制；已有 `.env` 时先合并配置，不覆盖原文件：

```bash
cp .env.cann.example .env
```

至少检查 `.env` 中以下项目：

```dotenv
CANN_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-310p-ubuntu22.04-py3.11
TRITON_BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.10-py3
TRITON_CANN_IMAGE=triton-ascend-cann:24.10-cann9.0.0
YOLO26_MODEL_HOST_PATH=/data/triton-inference-server-ascend/workspace/models/YOLO26_DET_PRE/1/yolo26s.om
ASCEND_DEVICE_ID=2
TRITON_HTTP_PORT=54245
TRITON_GRPC_PORT=54246
TRITON_METRICS_PORT=54247
```

模型路径只是示例，必须改为自己的现存非空文件。仅需要 YOLO26 NV12+AIPP 模型，
不需要为此准备 YOLO11 或 YOLO26 RGB 模型；模型应输出 `[1,300,6]`。
不要把 no-AIPP/RGB OM 放到这条 NV12 链路。OM 权重不包含在 Git 或构建上下文中。

```bash
test -s /你的实际路径/yolo26s.om
test -d /usr/local/Ascend/driver
test -d /usr/local/Ascend/develop
test -f /etc/ascend_install.info
npu-smi info -m
ls -l /dev/davinci*
```

当前示例用逻辑设备 2，源于用户提供的映射 NPU 5/Chip 0 -> Chip Logic ID 2。
这不是对服务器空闲状态的保证。容器保留原项目的 `privileged: true`，没有设备隔离；
默认假设所有设备可见且没有 Ascend Runtime/环境变量重映射。若实际做了重映射，
应按容器内映射改 `ASCEND_DEVICE_ID`。不要照抄 SAM3 容器名中的数字。
五个推理模型的默认配置不再写死 `device_id=0`，与预处理统一读取环境变量。
自行恢复模型级 `device_id` 时，必须与预处理使用同一设备。

## 2. 构建镜像

```bash
docker-compose -p triton-cann9 -f docker-compose.cann.yml config
docker-compose -p triton-cann9 -f docker-compose.cann.yml build triton-ascend
```

构建使用两个 ARM64/Ubuntu 22.04 基础镜像。可以预先拉取、离线导入，或通过
`.env` 的 `CANN_IMAGE`、`TRITON_BASE_IMAGE` 指向自己的镜像仓库。
不要使用 x86_64 镜像；旧 Docker 不支持的 BuildKit 指令和 `--platform` 均未使用。

默认 CANN 镜像与本机 ascend-sam3 Dockerfile 的基础镜像一致。如果服务器没有原始
CANN 镜像但保留了 `ascend-sam3-service:latest`，可以将 `CANN_IMAGE` 指向它作为
临时构建来源，前提是其 CANN 9.0.0 完整且包含头文件、Ascend C 工具和构建依赖。
只复制 CANN 目录到最终镜像，不会启动或修改已有 SAM3 容器。

若找不到 `acl.h`、`libascendcl.so` 或 `ascendc.cmake`，构建会失败并提示路径。
不要通过复制空目录或把 `devlib` 驱动桩库加入运行时搜索路径绕过。

## 3. 启动前检查

无设备初始化的环境检查（不发布端口、不启动 Triton）：

```bash
docker-compose -p triton-cann9 -f docker-compose.cann.yml run --rm --no-deps triton-ascend \
  bash -c 'echo "$ASCEND_HOME_PATH"; ldd /models/YOLO26_DET_PRE/1/libtriton_yolo26_infer.so'
```

`ldd` 输出中不能有 `not found`。首次创建容器之前一定先确认 OM 文件存在，否则
旧 Docker 可能将不存在的文件源创建为目录，启动脚本会拒绝这种挂载。

ACL 设备预检（只初始化所选设备，不加载模型）：

```bash
docker-compose -p triton-cann9 -f docker-compose.cann.yml run --rm --no-deps triton-ascend \
  python3 /opt/triton-ascend/check-device.py
```

预检成功不代表 OM 可运行，也不证明已经选中预期物理卡；结合容器可见设备映射与
宿主机 `npu-smi info -m` 核对。不要修改现有驱动、杀掉 SAM3 进程或复位整卡。

## 4. 启动与验证

```bash
docker-compose -p triton-cann9 -f docker-compose.cann.yml up -d --no-deps triton-ascend
docker-compose -p triton-cann9 -f docker-compose.cann.yml logs --tail=200 triton-ascend
curl -f http://127.0.0.1:54245/v2/health/ready
curl -f http://127.0.0.1:54245/v2/models/YOLO26_DET_PRE_YUV_ENSEMBLE/ready
```

默认加载清单为 `workspace/models_to_load.container.txt`，仅包括 `CUSTOM_LABELS`、
YUV 预处理、YOLO26 推理、后处理及 YUV ensemble。模型仓库和后端 `.so` 固化在镜像内，
只单独挂载 YOLO26 OM，不会被宿主机缺少 `.so` 的模型目录遮住。
修改 C++、模型配置或加载清单后须重新构建并创建容器；只改 `.env` 则重新创建：

```bash
docker-compose -p triton-cann9 -f docker-compose.cann.yml up -d --no-deps --force-recreate triton-ascend
```

模型 READY 后还需发起实际推理验证 NV12 输入、输出坐标和耗时。客户端安装依赖后：

```bash
cd workspace
python3 triton_client/visualize.py --protocol http --model YOLO26_DET_PRE_YUV_ENSEMBLE
```

示例默认访问 localhost:54245；如修改端口，按其 `--help` 调整。运行期间查看
`npu-smi info` 确认新增 tritonserver 进程出现在预期设备，且 SAM3 进程未受影响。

验证通过后，AIDetection 的 `TRITON_YOLO_URL`、`TRITON_YOLO_GRPC_URL` 分别指向
这台服务器的 HTTP 54245、gRPC 54246；模型名仍为 `YOLO26_DET_PRE_YUV_ENSEMBLE`。
不需要修改 AIDetection 的 NV12 请求协议。

## 已验证范围

本地可检查 Compose 展开、脚本语法、目录探测、模型默认设备配置及启动错误分支。
ARM64 镜像实际构建、CANN 9 Ascend C 编译、驱动访问、OM 加载与推理速度必须在
目标服务器上验收，不能以静态检查代替。旧内核/Docker 的兼容性也以实际启动结果为准。

部署回归测试只依赖 Python、PyYAML、Bash 和 CMake，不需要 NPU：

```bash
python3 -m unittest discover -s tests -v
```

缺少 Bash/CMake 时相应测试会跳过，不能把跳过视为通过。
