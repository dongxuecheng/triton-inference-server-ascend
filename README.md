# Ascend Triton YOLO 推理服务

运行在 ARM64 / Ascend 310P3 上的 Triton 自定义后端。CANN 9.0.0、Ascend C 内核和后端库均在容器镜像内构建、提供，宿主机无需安装 CANN Toolkit。

默认部署提供 `YOLO26_DET_PRE_YUV_ENSEMBLE`：原始 NV12 → NPU letterbox → YOLO26 AIPP 推理 → 检测框后处理。适用于 AIDetection 从 rtspGrpcServer 取得 NV12 后直接推理的链路。

A 机器已完成内置 CANN 镜像的实际验证：310P3、驱动 25.0.rc1.1、ACL 逻辑设备 1；1920×1080 测试图返回 9 个 person，单次客户端推理计时为 16.4 ms。该值不包含测试图转 NV12 的时间，也不是 B 机器或并发负载下的性能保证。

**A 机器迁移到 B（麒麟）的完整命令见 [镜像迁移指南](docs/validate-a-then-b.md)。可以直接导出 A 已验证的镜像，B 无需重新编译。**

## 部署文件

```text
Dockerfile                   CANN 编译阶段 + Triton 运行阶段
docker-compose.yml           唯一部署入口，兼容 Compose 1.22 格式 2.4
.env.example                 镜像、设备、OM 路径及端口配置
docker/                      CANN 环境、启动入口、ACL 设备预检
src/                         C++ 后端、Ascend C 预处理内核
cmake/                       CANN 头文件、运行库、Ascend C 工具定位
build.sh                     Dockerfile 内部使用的编译与库部署脚本
workspace/models/            模型配置和类别表；后端库在构建时生成
workspace/models_to_load.txt 唯一的默认模型加载清单
workspace/triton_client/     HTTP、gRPC、系统共享内存和可视化客户端
workspace/atc/               NV12 静态 AIPP 配置
scripts/                     启动、容器内模型转换、Triton 头文件提取
requirements-client.txt      宿主机图片测试所需 Python 依赖
```

旧的宿主机 Toolkit 部署已删除。原 `.cann` 部署文件并入上述默认文件；已有 `.env.cann.a` / `.env.cann.b` 可继续 source，镜像变量仍使用 `TRITON_CANN_IMAGE` / `TRITON_CANN_CONTAINER_NAME`。

## 环境要求与边界

- Linux `aarch64`、Ascend 310P3，宿主机已安装驱动和固件，`npu-smi info` 正常。
- A 使用 `docker compose`；B 的 Docker 18.09 / Compose 1.22 使用 `docker-compose`。以下普通部署命令以 A 为例，B 替换命令前缀即可。
- 宿主机提供 `/usr/local/Ascend/driver`、`/usr/local/Ascend/develop`、`/etc/ascend_install.info` 和设备节点。Compose 保留已验证的 `privileged: true`、`ipc: host`。
- CANN 来自镜像内 `/usr/local/Ascend/cann-9.0.0`。不要挂载宿主机 Toolkit 或整个 `/usr/local/Ascend`，否则可能遮住镜像内容。
- 默认链路只需要一个 YOLO26 NV12+AIPP `.om`，输出为 `[1,300,6]`；权重不在 Git 或镜像中，必须单独提供。

## 首次配置

```bash
git clone https://github.com/dongxuecheng/triton-inference-server-ascend.git
cd triton-inference-server-ascend
cp .env.example .env
```

已有仓库执行 `git pull --ff-only` 即可；已有 `.env` 时编辑原文件。至少设置：

```dotenv
TRITON_CANN_IMAGE=triton-ascend-cann:24.10-cann9.0.0
TRITON_CANN_CONTAINER_NAME=triton-ascend-cann9
YOLO26_MODEL_HOST_PATH=/实际路径/yolo26s.om
ASCEND_DEVICE_ID=2
TRITON_HTTP_PORT=54245
TRITON_GRPC_PORT=54246
TRITON_METRICS_PORT=54247
```

`ASCEND_DEVICE_ID` 是 ACL 逻辑设备号。A 当前使用 `1`；B 此前的 NPU 5 / Chip 0 对应逻辑设备 `2`，仍需以最新的 `npu-smi info -m` 和占用为准。`privileged` 使所有设备可见，并不隔离 NPU 资源。

在同一个 Bash 终端加载配置，确保下面的 curl、文件检查也能使用变量：

```bash
set -a
source .env
set +a

test -f "$YOLO26_MODEL_HOST_PATH" && test -s "$YOLO26_MODEL_HOST_PATH"
test -d /usr/local/Ascend/driver
test -d /usr/local/Ascend/develop
test -f /etc/ascend_install.info
npu-smi info -m
```

首次创建容器之前确认 OM 是非空文件，避免旧 Docker 把不存在的文件挂载源创建成目录。

## 构建与启动

```bash
docker compose -p triton-cann9 config
docker compose -p triton-cann9 build triton-ascend

docker compose -p triton-cann9 run --rm --no-deps triton-ascend \
  python3 /opt/triton-ascend/check-device.py

docker compose -p triton-cann9 up -d --no-deps triton-ascend
docker compose -p triton-cann9 logs --tail=200 triton-ascend
```

构建默认使用 ARM64 CANN 9.0.0 / 310P / Ubuntu 22.04 镜像编译，再将 CANN 和产物复制到 Triton 24.10 镜像。构建不需要 NPU 或宿主机 C++、CMake、Toolkit。`CANN_IMAGE` 和 `TRITON_BASE_IMAGE` 可改为相同内容的私有仓库地址；Dockerfile 的 CANN 版本目录固定为 9.0.0。

默认加载 `CUSTOM_LABELS`、`YUV_640_LETTERBOX_PREPROCESS`、`YOLO26_DET_PRE`、`YOLO26_DET_PRE_POSTPROCESS`、`YOLO26_DET_PRE_YUV_ENSEMBLE` 五个模型。模型配置、类别表、加载清单和 `.so` 固化在镜像里，宿主机只挂载单个 OM。

```bash
curl -sS -f -o /dev/null -w 'server HTTP %{http_code}\n' \
  "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready"
curl -sS -f -o /dev/null -w 'model HTTP %{http_code}\n' \
  "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/models/YOLO26_DET_PRE_YUV_ENSEMBLE/ready"
```

两项应返回 200；就绪接口的响应正文为空。接着执行真实图片测试，并用 `npu-smi info` 确认进程在预期芯片上。

## 图片测试

在客户端机器安装依赖。Ubuntu 的 Python 3.10 若缺少 `ensurepip`，先安装 `sudo apt-get install python3.10-venv`。

```bash
python3 -m venv .venv-triton-client && source .venv-triton-client/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-client.txt

python workspace/triton_client/visualize.py \
  --protocol http --url "127.0.0.1:${TRITON_HTTP_PORT}" \
  --image /实际路径/test.jpg --output /tmp/yolo26_result.jpg
```

客户端默认调用 YOLO26 YUV ensemble，将测试图先转换为 NV12。生产端已有 NV12 时直接提交即可。更详细的 Python API、gRPC 和 SHM 用法见 [客户端说明](workspace/triton_client/README.md)。

## 输入、输出及可选模型

| 接口 | 内容 |
| --- | --- |
| `YUV` | `UINT8 [H*3/2,W,1]`，先 Y 平面、后交错 UV 的 NV12；宽高必须为偶数，无 batch 维 |
| `NUM_DETS` | 有效检测数量 |
| `DETECTION_BOXES` | 原图像素坐标 `x1,y1,x2,y2` |
| `DETECTION_SCORES` | 检测分数 |
| `DETECTION_CLASSES` | 类别 ID；名称由 `CUSTOM_LABELS` 返回 |

预处理和推理在同一 NPU 上通过 `NV12_REF` 传递设备画布，避免画布回传主机。该句柄只能在同一请求链中使用。后处理利用 `[scale_x,scale_y,pad_left,pad_top,source_width,source_height]` 还原坐标。

保留以下可选模型及其源码，默认不加载：

| 模型 | 输入与用途 |
| --- | --- |
| `YOLO11_DET_PRE_ENSEMBLE` | BGR；YOLO11 检测 |
| `YOLO26_DET_PRE_ENSEMBLE` | BGR；YOLO26 AIPP 检测 |
| `YOLO26_DET_PRE_RGB_ENSEMBLE` | BGR；无 AIPP 的 RGB_CHW 推理 |
| `YOLO11_DET_PRE_HOST` / `YOLO26_DET_PRE_HOST` | 已 letterbox 的主机 NV12 画布 |

启用可选链路需提供对应 OM、调整模型配置和加载清单，并在 Compose 增加单个 OM 挂载，然后重建镜像。YOLO11 默认模型路径为 `/yolo/yolo11s.om`，RGB 模型为 `/models/YOLO26_DET_PRE_RGB/1/yolo26s_noaipp.om`。`benchmark.py` 会比较 BGR/YUV 链路，需先启用 BGR 预处理和 ensemble。

若需转换 YOLO26 ONNX，使用 CANN 容器中的 ATC：

```bash
docker run --rm -v "$PWD:/project" -w /project --entrypoint /bin/bash "$CANN_IMAGE" -lc \
  'source docker/cann-env.sh && python3 -m pip install onnx onnxsim && bash scripts/convert_yolo26.sh workspace/models/YOLO26_DET_PRE/1/yolo26s.onnx yolo26s'
```

无 AIPP 模型可给转换脚本设置 `AIPP_CFG=none`，并使用 RGB 对应路径及 ensemble。转换模型后仍需实际推理验证。

## 更新与排查

- 修改 C++、Ascend C、模型配置、类别表或加载清单：先 `docker compose -p triton-cann9 build triton-ascend`，再 `docker compose -p triton-cann9 up -d --no-deps --force-recreate triton-ascend`。不要挂载宿主机 `.so` 覆盖镜像。
- 修改 `.env` 或替换 OM：重新创建容器，无需重新编译后端。所有命令保持相同 Compose 项目名，A 现有测试项目名为 `triton-cann-a-test`。
- CANN 初始化报 `507033`：检查 `docker/cann-env.sh` 的真实 driver HAL 优先级，不能让 `develop/lib64` 中的 HAL 覆盖真实驱动。
- `ldd` 缺库：用 `docker compose -p triton-cann9 run --rm --no-deps triton-ascend ldd /models/YOLO26_DET_PRE/1/libtriton_yolo26_infer.so` 检查。
- CUDA / NVML / DCGM 初始化消息没有阻止 A 的实际 NV12 推理；换主机后仍以真实推理为准。
- 新 Compose 提示 `version` 过时是兼容 B 的旧 Compose 1.22 所致，保留格式 `2.4`。
- `CUSTOM_LABELS/names` 是类别表目录，不是模型版本目录，被 Triton 跳过版本扫描属于预期行为。

本地部署回归检查：`python3 -m unittest discover -s tests -v`，依赖 PyYAML、Bash、CMake；这些检查不能代替目标机器上的设备和模型推理验证。
