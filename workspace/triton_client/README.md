# Triton Python 客户端

封装 HTTP、gRPC、系统共享内存（SHM），默认检测模型为 `YOLO26_DET_PRE_YUV_ENSEMBLE`。服务部署见 [项目 README](../../README.md)。

## 安装与图片测试

在仓库根目录执行。Ubuntu Python 3.10 缺少 `ensurepip` 时，先执行 `sudo apt-get install python3.10-venv`。

```bash
python3 -m venv .venv-triton-client && source .venv-triton-client/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-client.txt

python workspace/triton_client/visualize.py --protocol http \
  --url localhost:54245 --image /实际路径/test.jpg --output /tmp/result.jpg
python workspace/triton_client/example.py --protocol grpc \
  --url localhost:54246 --image /实际路径/test.jpg
python workspace/triton_client/labels.py --protocol http --url localhost:54245 --count
```

当前封装会同时导入 HTTP 和 gRPC 客户端，因此两组依赖都需要。SHM 是系统共享内存，无需安装 CUDA 客户端依赖。`visualize.py` 的 `--conf` 仅过滤绘图结果，不修改服务端检测阈值。

图片路径通过 `--image` 或 `TRITON_TEST_IMAGE` 环境变量提供。图片测试先将 BGR 转为 NV12；生产端若已有 NV12，可以直接发送 `YUV` 张量，跳过这次转换。

## Python 调用

将仓库 `workspace/` 加入 `PYTHONPATH` 后：

```python
import numpy as np
from triton_client import TritonClient

# packed_nv12 来自实际 NV12 帧，形状为 [H*3/2,W,1]，H/W 为偶数。
inputs = {"YUV": np.ascontiguousarray(packed_nv12, dtype=np.uint8)}
outputs = ["NUM_DETS", "DETECTION_BOXES", "DETECTION_SCORES", "DETECTION_CLASSES"]
with TritonClient("localhost:54245", protocol="http") as client:
    result = client.infer("YOLO26_DET_PRE_YUV_ENSEMBLE", inputs, outputs)
count = int(result["NUM_DETS"][0])
boxes = result["DETECTION_BOXES"][:count]
```

gRPC 使用 `protocol="grpc"` 和端口 `54246`；可通过 `--url` 或构造函数指定其他地址。

SHM 使用 `protocol="shm"` 和 gRPC 端口，同次调用额外传 `output_specs`；可复用 `triton_client.yolo.SHM_OUTPUT_SPECS`。客户端与服务必须在同一主机、共享 IPC 命名空间，Compose 已设置 `ipc: host`。用 `control_protocol="http"` 可切换 SHM 管理接口为 HTTP。

## API 与其他工具

- `TritonClient(url, protocol="grpc", **kwargs)`：按协议创建客户端，支持 `with` 关闭连接。
- `infer(model_name, inputs, outputs, model_version="", output_specs=None)`：返回输出名到 NumPy 数组的字典；SHM 要预先声明输出缓冲区形状和类型。
- `labels.py`：查询 `CUSTOM_LABELS`；默认读取 `workspace/models_to_load.txt` 中的 ensemble。类别表位于服务端 `CUSTOM_LABELS/names/<模型名>.txt`。
- `visualize.py --model A,B`：分别生成多个模型的检测结果及对比图；使用前需在服务端启用相应模型。
- `benchmark.py`：比较 BGR 与 NV12 链路，需额外启用 BGR 预处理和 `YOLO26_DET_PRE_ENSEMBLE`；默认五模型配置仅提供 NV12 链路。

YOLO11、YOLO26 BGR 和 RGB 路径仍可通过 `--model` 指定，但需要相应服务端模型和权重。各工具可用 `--help` 查看参数。
