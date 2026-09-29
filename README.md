# Ascend Triton YOLO 推理服务

这是运行在 Ascend 310P3 上的 Triton 自定义后端项目，支持 YOLO11、YOLO26，以及 BGR、NV12/YUV420SP 和无 AIPP 的 RGB 输入路径。

项目提供 C++ Triton 后端、Ascend C 预处理内核、YOLO 推理与后处理后端、Triton 模型仓库、Docker Compose 服务配置，以及 Python 客户端和结果可视化工具。

## 目录结构

```text
src/                         Triton C++ 后端与 Ascend C 内核
  common/                    ACL 模型封装、画布池、配置解析
  preprocess/                BGR/NV12 -> NV12 letterbox
  rgb_preprocess/            BGR -> RGB_CHW letterbox
  yoloinfer/                 YOLO11 ACL 推理
  yolo26_infer/              YOLO26 ACL 推理
  yolo11_postprocess/        YOLO11 解码与 NMS
  yolo26_postprocess/        YOLO26 阈值过滤与坐标还原
workspace/models/            Triton 模型仓库；配置随代码管理，权重和 .so 不提交
workspace/triton_client/     HTTP、gRPC、SHM 客户端、基准和可视化脚本
workspace/atc/               ATC 静态 AIPP 配置
workspace/models_to_load.txt 服务启动时加载的模型清单
scripts/                     Triton 头文件获取、模型转换、服务启动脚本
build.sh                     编译并部署后端库
docker-compose.yml           Triton 容器、Ascend 设备和端口配置
```

## 支持的模型

| 模型 | 输入 | 用途 |
| --- | --- | --- |
| `YOLO11_DET_PRE_ENSEMBLE` | `IMAGE`：BGR `UINT8 [H,W,3]` | YOLO11 完整检测链 |
| `YOLO26_DET_PRE_ENSEMBLE` | `IMAGE`：BGR `UINT8 [H,W,3]` | YOLO26 AIPP 完整检测链 |
| `YOLO26_DET_PRE_RGB_ENSEMBLE` | `IMAGE`：BGR `UINT8 [H,W,3]` | YOLO26 无 AIPP/RGB 完整检测链 |
| `YOLO26_DET_PRE_YUV_ENSEMBLE` | `YUV`：打包 NV12 `UINT8 [rows,W,1]` | 直接输入 NV12，不做颜色转换 |
| `YOLO_640_LETTERBOX_PREPROCESS` | `IMAGE`：BGR | 单独执行 BGR 到 NV12 的 letterbox |
| `YUV_640_LETTERBOX_PREPROCESS` | `YUV`：打包 NV12 | 单独执行 NV12 到 NV12 的 letterbox |
| `YOLO_640_LETTERBOX_PREPROCESS_RGB` | `IMAGE`：BGR | 单独生成无 AIPP 模型使用的 RGB_CHW 张量 |
| `YOLO11_DET_PRE`、`YOLO26_DET_PRE` | 内部 `NV12_REF` 设备句柄 | Ensemble 使用的设备内存推理模型 |
| `YOLO11_DET_PRE_HOST`、`YOLO26_DET_PRE_HOST` | 已 letterbox 的 `NV12` 张量 | 调用方已有 640×640 NV12 画布时使用 |
| `YOLO26_DET_PRE_RGB` | `RGB_CHW` FP32 | 无 AIPP YOLO26 推理模型 |
| `YOLO11_DET_PRE_POSTPROCESS`、`YOLO26_DET_PRE_POSTPROCESS` | `PREDICTIONS` | 单独执行后处理 |
| `CUSTOM_LABELS` | 模型名 | 返回对应模型的类别名称 |

检测 ensemble 输出 `NUM_DETS`、`DETECTION_BOXES`、`DETECTION_SCORES` 和 `DETECTION_CLASSES`。`NUM_DETS` 是有效结果数量；框默认映射回输入图像的像素坐标。

## 环境要求

- Ascend 310P3 主机、已安装并可访问的 Ascend 驱动和 CANN/Ascend C Toolkit。此项目在 CANN 8.2 环境验证，目标 SOC 默认为 `Ascend310P3`。
- Docker Engine 和 Docker Compose 插件。
- C++17 编译器、CMake 3.16 或更新版本，以及 `nproc`。
- Triton 容器镜像默认使用 `nvcr.io/nvidia/tritonserver:24.10-py3`。镜像需要可拉取；若使用 NVIDIA NGC 私有镜像，请先按本机规定登录。
- 要运行推理，必须另外准备 YOLO11 和/或 YOLO26 的 Ascend `.om` 模型。模型权重不包含在仓库内。
- Python 客户端需要 Python 3、NumPy、OpenCV 和与服务兼容的 Triton 客户端 SDK。SHM 客户端另需 Linux System V 共享内存支持。

确认 CANN 工具链和 ATC：

```bash
export ASCEND_HOME_PATH="$HOME/Ascend/ascend-toolkit/latest"
test -f "$ASCEND_HOME_PATH/include/acl/acl.h"
test -x "$ASCEND_HOME_PATH/compiler/bin/atc"
```

如果 CANN 安装在其他位置，将 `ASCEND_HOME_PATH` 改为**包含 `include/acl/acl.h` 的 Toolkit 版本目录**，不要指向只包含版本目录的上级目录。

确认 Docker Compose 和设备可见：

```bash
docker compose version
npu-smi info
```

## 获取源码

```bash
git clone https://github.com/leon0514/triton-inference-server-ascend.git
cd triton-inference-server-ascend
```

如果当前目录已经是源码目录，则不需要再次 clone。首次克隆后先确认仓库中没有 `.om`、`.onnx`、本地 `.env` 或编译后的 `.so`；这些是本地生成/部署文件，不由 Git 管理。

## 编译并部署后端

### 1. 准备 Triton C++ 头文件

仓库附带用于编译的 Triton C API 头文件，`build.sh` 默认从 `third_party/triton/include` 查找。如果目录中缺少 `triton/core/tritonbackend.h`，可从 Triton 容器镜像提取：

```bash
scripts/fetch-triton-headers.sh
```

脚本默认使用 `nvcr.io/nvidia/tritonserver:24.10-py3`，因此需要 Docker 和可访问该镜像。也可以指定输出目录：

```bash
scripts/fetch-triton-headers.sh /path/to/triton/include
```

### 2. 编译和部署

```bash
export ASCEND_HOME_PATH="$HOME/Ascend/ascend-toolkit/latest"
export TMPDIR=/dev/shm
./build.sh
```

`build.sh` 会依次：

1. 配置 CMake，目标默认为 `Ascend310P3`，C++ 标准为 C++17。
2. 编译 Triton 后端、共享设备画布池和 Ascend C 预处理内核。
3. 安装到 `build/install/`，为依赖库设置 `$ORIGIN` RPATH。
4. 将每个后端 `.so` 复制到对应模型的 `workspace/models/<模型名>/1/` 目录。

常用可选环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | `/usr/local/Ascend/ascend-toolkit/latest` | CANN Toolkit 路径 |
| `SOC_VERSION` | `Ascend310P3` | ATC/Ascend C 目标 SOC |
| `TRITON_DIR` | `third_party/triton` 或自动探测 | 含 `include/triton/core/tritonbackend.h` 的 Triton 根目录 |
| `BUILD_DIR` | 仓库内 `build/` | CMake 构建目录 |
| `TMPDIR` | `/dev/shm` | Ascend C 编译临时目录；空间不足时请指定足够大的文件系统 |

例如使用非默认 Triton 头文件目录：

```bash
TRITON_DIR=/opt/tritonserver ASCEND_HOME_PATH="$ASCEND_HOME_PATH" ./build.sh
```

直接运行 `cmake --build` 只会构建，不会安装和部署后端库；常规使用请运行 `./build.sh`。若更新了共享内核库，必须重新运行整个脚本，让所有使用该 SONAME 的模型目录都获得相同版本，然后重启 Triton 进程。

## 准备 Ascend 模型

模型文件不在 Git 中，需从自己的训练/导出流程获得。YOLO11 配置默认读取容器内 `/yolo/yolo11s.om`；YOLO26 AIPP 配置默认读取仓库内 `workspace/models/YOLO26_DET_PRE/1/yolo26s.om`；无 AIPP 变体默认读取 `workspace/models/YOLO26_DET_PRE_RGB/1/yolo26s_noaipp.om`。如文件名或位置不同，修改相应模型配置中的 `model_path`，或使用 YOLO11 对应的 Compose 挂载变量。

### 转换 YOLO26 AIPP 模型

准备动态或较新 opset 的 YOLO26 ONNX 导出，并安装 `onnx`、`onnxsim` Python 包。转换脚本会固定输入形状、在需要时用 `onnxsim` 简化动态形状，并将 ATC 不支持的 opset 降至 17：

```bash
python3 -m pip install onnx onnxsim
export ASCEND_HOME_PATH="$HOME/Ascend/ascend-toolkit/latest"
export TMPDIR=/dev/shm
scripts/convert_yolo26.sh \
  workspace/models/YOLO26_DET_PRE/1/yolo26s.onnx yolo26s
```

默认插入 `workspace/atc/aipp_nv12_640.cfg`，生成 `workspace/models/YOLO26_DET_PRE/1/yolo26s.om`，输入为静态 NV12 + AIPP。输出由 YOLO26 模型提供，格式为 `[1,300,6]`。

### 转换 YOLO26 无 AIPP 模型

```bash
AIPP_CFG=none scripts/convert_yolo26.sh \
  workspace/models/YOLO26_DET_PRE_RGB/1/yolo26s.onnx yolo26s_noaipp
```

将生成的文件放在 `workspace/models/YOLO26_DET_PRE_RGB/1/yolo26s_noaipp.om`。无 AIPP 模型直接接收 FP32、RGB、CHW、范围 `[0,1]` 的输入，应配对 `YOLO_640_LETTERBOX_PREPROCESS_RGB` 和 `YOLO26_DET_PRE_RGB`，不要配 NV12/AIPP ensemble。

YOLO11 `.om` 不由本仓库的转换脚本生成。将已有模型复制到本机路径，并保证 `YOLO11_DET_PRE/config.pbtxt` 中的 `model_path` 与容器内挂载目标一致。

## 配置服务

### 1. 创建并编辑 `.env`

```bash
cp .env.example .env
```

编辑 `.env`，至少设置 Toolkit 路径和 YOLO11 `.om` 的**宿主机绝对路径**：

```dotenv
ASCEND_TOOLKIT_ROOT=/home/your-user/Ascend/ascend-toolkit
YOLO_MODEL_HOST_PATH=/absolute/path/to/yolo11s.om
YOLO_MODEL_PATH_IN_CONTAINER=/yolo/yolo11s.om
ASCEND_DEVICE_ID=0
ASCEND_CANVAS_SLOTS=8
TRITON_IMAGE=nvcr.io/nvidia/tritonserver:24.10-py3
TRITON_HTTP_PORT=54245
TRITON_GRPC_PORT=54246
TRITON_METRICS_PORT=54247
```

`.env` 是本机配置并被 `.gitignore` 忽略，不要提交真实机器路径或凭据。默认端口若已被占用，请改左侧宿主机端口，例如 `TRITON_HTTP_PORT=15425`；容器内端口仍为 `8000/8001/8002`。

YOLO11 `.om` 必须存在且是文件。Docker 对不存在的 bind mount 源可能创建目录，之后模型会加载失败。启动前检查：

```bash
test -f "$YOLO_MODEL_HOST_PATH"
test -f /etc/ascend_install.info
test -d "$ASCEND_TOOLKIT_ROOT"
```

Compose 会挂载整个 Toolkit（`latest` 通常是指向版本目录的符号链接）、Ascend 驱动、`/etc/ascend_install.info`、模型仓库和模型加载清单。此环境使用 `privileged: true` 供 ACL 访问设备；共享内存推理需要 `ipc: host`。如果系统没有 `/etc/ascend_install.info`，需按本机驱动安装情况调整挂载。

### 2. 设置启动时加载的模型

`workspace/models_to_load.txt` 每行一个模型，`#` 开头为注释；`scripts/serve.sh` 用它生成 Triton `--load-model` 参数，并自动追加 `CUSTOM_LABELS`。默认清单包含 YOLO11 和 YOLO26 模型。

在干净环境中启动前，清单中的推理模型必须有对应 `.om`，并且模型配置指向正确路径。若暂时没有 YOLO26 权重，可从清单中注释掉 `YOLO26_DET_PRE`、`YOLO26_DET_PRE_HOST`、`YOLO26_DET_PRE_ENSEMBLE`、`YOLO26_DET_PRE_RGB`、`YOLO26_DET_PRE_RGB_ENSEMBLE` 和 `YOLO26_DET_PRE_YUV_ENSEMBLE`，以及只用于这些链的独立模型；或者先完成相应模型转换。YOLO11 也必须先准备好。

显式模型控制模式下，未列入清单的模型不会在启动时加载，但可通过 Triton 仓库 API 加载。ensemble 的每个依赖模型也必须可用。

## 启动与停止

在仓库根目录执行：

```bash
docker compose config       # 展开变量并检查 Compose 配置
docker compose up -d        # 创建并后台启动 Triton
docker compose ps           # 查看容器与端口映射
docker compose logs -f      # 查看启动及模型加载日志，Ctrl-C 退出日志跟随
```

服务健康检查和模型状态：

```bash
curl -i http://localhost:54245/v2/health/ready
curl -s -X POST http://localhost:54245/v2/repository/index \
  -H 'Content-Type: application/json' -d '{}' | python3 -m json.tool
```

健康检查成功时返回 HTTP `200`。仓库索引中的每个模型应为 `READY`；模型未就绪时先查看：

```bash
docker compose logs --tail=200 triton-ascend
```

显式加载/卸载单个模型：

```bash
curl -s -X POST http://localhost:54245/v2/repository/models/YOLO26_DET_PRE_ENSEMBLE/load -d '{}'
curl -s -X POST http://localhost:54245/v2/repository/models/YOLO26_DET_PRE_ENSEMBLE/unload -d '{}'
```

停止服务但保留容器：

```bash
docker compose down
```

修改 `.env`、挂载、环境变量或模型加载清单后，用 `docker compose up -d` 重新创建容器；只替换已挂载的 `.so` 后则需重启 Triton，使动态加载器重新载入库：

```bash
docker compose restart
```

## 推理客户端

在客户端环境安装依赖：

```bash
python3 -m pip install numpy opencv-python 'tritonclient[all]'
```

HTTP、gRPC 和 SHM 默认端点分别为 `localhost:54245`、`localhost:54246` 和 `localhost:54246`（SHM 使用 gRPC 管理共享内存区域）。端口可由 `.env` 中的 `TRITON_HTTP_PORT`、`TRITON_GRPC_PORT`、`TRITON_METRICS_PORT` 修改。

在 `workspace/` 下运行示例：

```bash
cd workspace
python3 triton_client/example.py --protocol http
python3 triton_client/example.py --protocol grpc --model YOLO26_DET_PRE_ENSEMBLE
python3 triton_client/example.py --protocol shm --model YOLO26_DET_PRE_ENSEMBLE
```

命令行选项：

```bash
python3 triton_client/example.py --help
python3 triton_client/visualize.py --help
python3 triton_client/labels.py --help
python3 triton_client/benchmark.py --help
```

`visualize.py` 可对多个模型运行检测并把标注图写入 `workspace/detections/`：

```bash
python3 triton_client/visualize.py --protocol http \
  --model YOLO11_DET_PRE_ENSEMBLE,YOLO26_DET_PRE_ENSEMBLE
python3 triton_client/visualize.py --protocol http \
  --model YOLO26_DET_PRE_YUV_ENSEMBLE
```

类别表由 `CUSTOM_LABELS` 提供。每个 ensemble 对应 `workspace/models/CUSTOM_LABELS/names/<模型名>.txt`，每行一个类别名，行号就是类别 ID。更新标签文件后重新加载该 Python 模型：

```bash
curl -s -X POST http://localhost:54245/v2/repository/models/CUSTOM_LABELS/unload -d '{}'
curl -s -X POST http://localhost:54245/v2/repository/models/CUSTOM_LABELS/load -d '{}'
```

SHM 模式要求客户端和容器共享 IPC 命名空间；Compose 已配置 `ipc: host`。HTTP/gRPC 不依赖此设置。`NV12_REF` 是模型间传递的设备地址句柄，不是供客户端保存或重复提交的图像张量；普通客户端应调用 ensemble，或请求 `NV12` 主机张量。

## NV12/YUV 输入

`YOLO26_DET_PRE_YUV_ENSEMBLE` 接收打包 NV12：`UINT8 [rows, width, 1]`，其中 `rows = 图像高度 × 3/2`、宽度和图像高度均为偶数。布局是先完整 Y 平面，再接交错的 UV 平面（NV12）。例如 1568×1068 输入的张量形状为 `(1602, 1568, 1)`。

该路径直接缩放亮度和色度平面，不做 YUV↔RGB 颜色转换，然后输出与 BGR 预处理相同的 NV12 画布和变换元数据。摄像头或 DVPP 已经输出 NV12 时，可直接发送该帧；不要先转换为 BGR。

YUV 预处理与完整 ensemble 的基准：

```bash
cd workspace
python3 triton_client/benchmark.py --runs 30 --protocol http
python3 triton_client/benchmark.py --runs 30 --protocol grpc
```

该脚本使用同一张测试图生成对应 NV12，并比较 BGR 与 YUV 链路，也分别测预处理、回传画布和传输估算。实测结果会受设备负载、网络和图像大小影响；README 中的测量值仅作参考。

## 输出和变换元数据

后处理输出：

| 输出 | 内容 |
| --- | --- |
| `NUM_DETS` | 有效检测数 |
| `DETECTION_BOXES` | `x1,y1,x2,y2`，通常映射回源图像像素坐标 |
| `DETECTION_SCORES` | 检测分数 |
| `DETECTION_CLASSES` | 类别索引 |

预处理返回的 `TRANSFORM_METADATA` 为 6 个 FP32：`[scale_x, scale_y, pad_left, pad_top, source_width, source_height]`。后处理用它把 640×640 画布中的框映射回输入图像坐标。单独调用后处理而不提供 metadata 时，框保持在画布坐标中。

YOLO11 原始推理张量为 `PREDICTIONS FP32 [1,84,8400]`：前 4 个通道是 `cx,cy,w,h`，其余 80 个是 COCO 类别分数；YOLO11 后处理负责解码和 NMS。YOLO26 输出 `[1,300,6]` 检测行，后处理只负责阈值过滤和坐标还原。

## 配置项

所有后端参数都在对应模型 `config.pbtxt` 的 `parameters` 中配置。Triton 的 `ModelParameter` 使用字符串值，数字和布尔值也必须写成带引号的字符串。

常用参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| 预处理 `target_height`、`target_width` | `640`、`640` | 输出画布尺寸 |
| 预处理 `pad_luma`、`pad_chroma` | `114`、`128` | NV12 letterbox 边界的 Y、UV 值 |
| 预处理 `kernel_blocks` | `8` | Ascend C 内核 block 数 |
| 预处理 `even_align` | `true` | 将 paste 偏移对齐到偶数，满足 NV12 采样 |
| 推理 `model_path` | 各模型配置指定 | `.om` 文件路径；配置项优先于环境变量 |
| 推理 `device_id` | `ASCEND_DEVICE_ID`，默认 `0` | NPU 设备编号 |
| YOLO26 `input_format` | `nv12` | `nv12` 或 `rgb_chw`，必须与模型输入匹配 |
| YOLO11 `confidence_threshold` | `0.25` | 置信度阈值 |
| YOLO11 `iou_threshold` | `0.45` | NMS IoU 阈值 |
| 两代后处理 `max_detections` | `300` | 最大输出检测数 |

常用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `YOLO_MODEL_PATH` | 无 | YOLO11 模型配置未设置 `model_path` 时的回退路径 |
| `YOLO26_MODEL_PATH` | 无 | YOLO26 模型配置未设置 `model_path` 时的回退路径 |
| `ASCEND_DEVICE_ID` | `0` | NPU 设备编号回退值 |
| `ASCEND_CANVAS_SLOTS` | `8` | 并发推理可用的设备画布数量 |

通常修改模型配置后 Triton 会按其模型控制方式加载新配置，不需要重新编译。修改 C++/Ascend C 源码则需要重新运行 `./build.sh` 并重启服务。

## 重要实现说明

- BGR 预处理执行 letterbox、BT.601 limited-range BGR→YUV 转换并生成 NV12。
- YUV 预处理对 NV12 的 Y/UV 平面直接重采样，不执行颜色转换；UV 在半分辨率空间采样。
- RGB 预处理输出 FP32、RGB、CHW、`[0,1]` 张量，供无 AIPP 模型使用。
- Ensemble 在 NPU 上通过 `NV12_REF` 传递设备画布，避免把 614400 字节画布复制到主机再传回设备。句柄只应在同一 Triton 请求链内传递。
- ACL 初始化是进程级状态；后端允许重复初始化返回值，不调用 `aclFinalize`，避免一个后端影响同进程中的其他后端。
- `src/preprocess/CMakeLists.txt` 集中声明 Ascend C 内核库，因为 `ascendc.cmake` 在同一 CMake 项目中只能包含一次。
- `libyolo_preprocess_kernel.so` 有固定 SONAME。所有加载此库的模型版本目录必须部署同一构建版本；`build.sh` 会一并更新相关目录。
- 预处理内核以 128 字节输出块分块，将源行暂存到 UB，避免逐元素读取 GM；横向映射表减少重复浮点除法，块内部再拆成左边界/图像/右边界三段。
- 各后端按 Triton 的实例 stream 执行 ACL/Ascend C 工作，并在数据跨步骤读取前同步。

## 常见问题

### `CANN toolkit not found` 或找不到 `kernel_operator.h`

`ASCEND_HOME_PATH` 必须指向有 `include/acl/acl.h` 和 Ascend C 工具的 Toolkit 目录。确认路径正确后重新编译：

```bash
ASCEND_HOME_PATH="$HOME/Ascend/ascend-toolkit/latest" TMPDIR=/dev/shm ./build.sh
```

### 找不到 `tritonbackend.h`

检查 `third_party/triton/include/triton/core/tritonbackend.h` 是否存在；不存在时运行 `scripts/fetch-triton-headers.sh`，或设置 `TRITON_DIR` 指向 Triton 安装根目录。

### 模型不是 READY

检查 `.om` 文件是否存在、模型配置的 `model_path` 是否为**容器内路径**、`.env` 的宿主机 bind mount 是否有效，以及 `workspace/models_to_load.txt` 中的 ensemble 依赖是否都加载。用 `docker compose logs --tail=200` 查看 Triton 报错。

### `undefined symbol` 或加载到旧内核

不要只覆盖一个模型目录里的 `.so`。运行 `./build.sh` 更新所有部署位置，再执行 `docker compose restart`。同 SONAME 的库在进程中只加载一份，首个加载副本会决定实现版本。

### `address already in use`

修改 `.env` 的宿主机端口（例如将 `TRITON_HTTP_PORT` 改成 `15425`），再运行 `docker compose up -d`；容器内端口映射无需改动。

### `CUSTOM_LABELS` 缺少类别表

检查 `workspace/models/CUSTOM_LABELS/names/<ensemble 名>.txt` 是否存在，每行一个类别名。修改后 unload/load `CUSTOM_LABELS` 模型即可重新读取文件。

### SHM 报共享内存区域不可见

确认 `docker-compose.yml` 保留 `ipc: host`，并确认 Docker 和客户端位于同一台 Linux 主机。HTTP 或 gRPC 可用于排除 SHM 配置问题。

## 验证记录

项目在 Ascend 310P3、CANN 8.2 和 Triton 24.10 容器环境验证。示例图 `persons.jpg` 上，四条检测链分别产生 YOLO11 29 个、YOLO26 AIPP 28 个、YOLO26 RGB 27 个、YOLO26 YUV 30 个检测。BGR 与 YUV 路径采样方式不同，因此分数阈值附近的检测数可能不同；letterbox 边界和几何映射保持一致。

优化后的示例性能（20 次请求中位数，具体取决于设备负载和网络）：

| 路径 | 端到端耗时 | 预处理单独耗时 |
| --- | ---: | ---: |
| YOLO26 BGR/AIPP | 约 25.3 ms | 约 20.0 ms |
| YOLO26 NV12/YUV | 约 13.7 ms | 约 8.1 ms |

改动前后针对 1568×1068、640×480、333×200、334×200 等几何测试了预处理结果；优化版本与基线逐字节一致。NV12 输入的宽、高必须是偶数。