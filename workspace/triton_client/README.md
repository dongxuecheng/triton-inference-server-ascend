# Triton Client Wrapper

A concise Python wrapper around the official Triton client SDK that exposes the
same `infer()` API for HTTP, gRPC and system shared memory (SHM) inference.

## Installation

```bash
pip install tritonclient[all] numpy
```

## Usage

```python
import numpy as np
from triton_client import TritonClient

inputs = {"raw_image": np.random.randint(0, 256, (1, 640, 640, 3), dtype=np.uint8)}
outputs = ["num_dets", "detection_boxes", "detection_scores", "detection_classes"]

# gRPC
with TritonClient("localhost:8001", protocol="grpc") as client:
    result = client.infer("yolo", inputs, outputs)

# HTTP
with TritonClient("localhost:8000", protocol="http") as client:
    result = client.infer("yolo", inputs, outputs)

# Shared memory: output_specs describes every requested output tensor.
output_specs = {
    "num_dets": ([1, 1], "INT32"),
    "detection_boxes": ([1, 300, 4], "FP32"),
    "detection_scores": ([1, 300], "FP32"),
    "detection_classes": ([1, 300], "INT32"),
}
with TritonClient("localhost:8001", protocol="shm") as client:
    result = client.infer("yolo", inputs, outputs, output_specs=output_specs)
```

## API

### `TritonClient(url, *, protocol="grpc", **kwargs)`

Factory that returns one of:

- `TritonHttpClient`
- `TritonGrpcClient`
- `TritonSharedMemoryClient`

All clients support the context-manager protocol.

### `infer(model_name, inputs, outputs, *, model_version="", output_specs=None, **kwargs)`

Run inference and return a dict mapping output names to `np.ndarray`.

- `inputs`: `dict[str, np.ndarray]`
- `outputs`: `list[str]`
- `output_specs`: required for SHM, `dict[str, (shape, dtype)]` where `dtype`
  is a Triton datatype string such as `"FP32"`, `"INT32"`, `"UINT8"`, etc.

### Shared memory control protocol

`TritonSharedMemoryClient` uses gRPC by default to register/unregister SHM
regions with the server. Pass `control_protocol="http"` to switch the control
channel to HTTP:

```python
with TritonClient(
    "localhost:8000", protocol="shm", control_protocol="http"
) as client:
    result = client.infer(...)
```

## Notes

- The wrapper only deals with the Triton client itself. Thread pools,
  client pools and stream orchestration belong in the application layer.
- SHM regions are cached inside each `TritonSharedMemoryClient` instance and
  reused across requests with the same tensor name. They are released when
  the client is closed.

---

## Using this client with this project

The wrapper is used unchanged; only the tensor names and endpoints differ from
the example above, because the models here name the ensemble input `IMAGE`,
return upper-case detection tensors, and run with `max_batch_size: 0` (so the
image goes in as `[height, width, 3]`, without a batch dimension).

| | value |
| --- | --- |
| HTTP endpoint | `localhost:54245` |
| gRPC endpoint | `localhost:54246` |
| ensemble models | `YOLO11_DET_PRE_ENSEMBLE`, `YOLO26_DET_PRE_ENSEMBLE` |
| input | `IMAGE` `UINT8 [height, width, 3]` BGR |
| outputs | `NUM_DETS`, `DETECTION_BOXES`, `DETECTION_SCORES`, `DETECTION_CLASSES` |

`example.py` in this directory is the adapter:

```bash
python3 example.py --protocol grpc          # YOLO11 by default
python3 example.py --protocol http  --model YOLO26_DET_PRE_ENSEMBLE
python3 example.py --protocol shm
```

`DETECTION_BOXES`, `DETECTION_SCORES` and `DETECTION_CLASSES` have a variable
length, so they declare a maximum and `NUM_DETS` says how many entries are real.
For SHM, `example.py` therefore passes `output_specs` sized to the postprocess
cap (`max_detections`, 300 by default) and slices with `NUM_DETS`.

`visualize.py` runs the same call and draws the result onto the image. Pass
several models to get each one separately plus a side-by-side comparison:

```bash
python3 visualize.py                                  # works out of the box
python3 visualize.py --protocol shm --conf 0.5        # fewer, more certain boxes
python3 visualize.py --model YOLO11_DET_PRE_ENSEMBLE,YOLO26_DET_PRE_ENSEMBLE
python3 visualize.py --image other.jpg --output-dir /tmp
```

It writes `<image>_<model>.jpg` to `workspace/detections/` by default, drawing
boxes for every detection at or above `--conf` and labelling them best score
first, skipping a label when it would cover one already placed. `--conf` only
controls what is drawn: the threshold the models apply is their
`confidence_threshold` parameter, so lowering it here cannot reveal detections
the backend dropped.

`labels.py` asks the `CUSTOM_LABELS` model what a class id means, so a
custom-trained model needs no client change:

```bash
python3 labels.py                        # every ensemble in models_to_load.txt
python3 labels.py YOLO26_DET_PRE_ENSEMBLE
python3 labels.py --grep person
python3 labels.py --count
```

`CUSTOM_LABELS` is a Python backend model, not one of the compiled Ascend ones:
it reads `workspace/models/CUSTOM_LABELS/names/<model>.txt` at load time (one
name per line, line order is class id order) and returns the list as a string
tensor. Only the ensembles have a file, because a chain's steps expose the same
classes and are looked up through the ensemble name. `labels.py` prints the id
next to each name, since that id is what `DETECTION_CLASSES` carries. Only HTTP
and gRPC work for it: the output length varies and shared memory needs every
buffer sized before the call.

The three scripts share `coco.py` (class names and colours) and `yolo.py`
(tensor names, ports, `detect()`, `get_labels()`). `coco.py` keeps a built-in
COCO table as the fallback for when the service is not loaded.

**SHM needs `ipc: host`.** System shared memory is a host resource, so the
client's `/dev/shm` segment is invisible to a server running in a container with
a private IPC namespace, and inference fails with `Unable to open shared memory
region`. `docker-compose.yml` sets `ipc: host` for this reason. HTTP and gRPC do
not need it.
