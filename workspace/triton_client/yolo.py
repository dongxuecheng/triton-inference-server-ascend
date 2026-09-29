"""Project-specific glue for calling this project's YOLO detection ensembles.

The wrapper in this package is protocol-agnostic, so the only things that differ
from its own README are:

* the tensor names - the ensemble input is `IMAGE`, the outputs are
  `NUM_DETS` / `DETECTION_BOXES` / `DETECTION_SCORES` / `DETECTION_CLASSES`;
* the models run with `max_batch_size: 0`, so an image goes in as
  `[height, width, 3]` with no batch dimension;
* over shared memory every output buffer must be sized up front.
"""

import os
import time
from typing import NamedTuple

import numpy as np

from . import coco

DEFAULT_IMAGE = "/home/HwHiAiUser/project/ascend-sam3/workspace/persons.jpg"

OUTPUTS = ["NUM_DETS", "DETECTION_BOXES", "DETECTION_SCORES", "DETECTION_CLASSES"]

# The preprocess backend accepts either a decoded BGR frame or a packed NV12 one,
# and the model config decides which; `input_name` reads that off the model.
IMAGE_INPUT = "IMAGE"
YUV_INPUT = "YUV"

# The postprocess backend caps detections at `max_detections` (300 by default)
# and `NUM_DETS` says how many of those slots are real, so the shared memory
# buffers are declared at the cap and sliced afterwards.
SHM_OUTPUT_SPECS = {
    "NUM_DETS": ([1], "INT32"),
    "DETECTION_BOXES": ([300, 4], "FP32"),
    "DETECTION_SCORES": ([300], "FP32"),
    "DETECTION_CLASSES": ([300], "INT32"),
}

# Ports from docker-compose.yml; the default Triton ports are taken on this host.
HTTP_URL = "localhost:54245"
GRPC_URL = "localhost:54246"

# The Python backend model that maps class ids to names; see its config.pbtxt.
LABELS_MODEL = "CUSTOM_LABELS"

# workspace/models_to_load.txt, used by labels.py when no model is named.
MODELS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models_to_load.txt")


class Detection(NamedTuple):
    """One model's answer for one image, already trimmed to `count` rows."""

    count: int
    boxes: np.ndarray      # [count, 4], FP32, pixels in the source image
    scores: np.ndarray     # [count], FP32
    classes: np.ndarray    # [count], INT32
    elapsed_ms: float

    def histogram(self, names=None):
        """{label: number of detections} in the order classes were returned."""
        counts = {}
        for label in self.classes:
            key = coco.name(int(label), names)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def best(self, names=None):
        """(label, score, box) of the highest scoring detection, or None."""
        if not self.count:
            return None
        index = int(np.argmax(self.scores))
        return (coco.name(int(self.classes[index]), names),
                float(self.scores[index]),
                np.round(self.boxes[index], 1).tolist())


def default_url(protocol):
    return HTTP_URL if protocol == "http" else GRPC_URL


def client_kwargs(protocol):
    """Extra keyword arguments the wrapper needs for this protocol."""
    return {"output_specs": SHM_OUTPUT_SPECS} if protocol == "shm" else {}


def load_image(path):
    import cv2

    image = cv2.imread(path)
    if image is None:
        raise SystemExit("cannot read %s" % path)
    return image


def to_nv12(bgr):
    """A decoded frame as packed NV12, BT.601 limited range.

    This is the matrix the BGR preprocess kernel encodes with, so a caller that
    only holds a decoded image still reaches the same canvas the no-conversion
    path produces. A camera or DVPP pipeline would hand over its own frame
    instead and skip this entirely.
    """
    image = bgr.astype(np.float32)
    blue, green, red = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    luma = 16.0 + 0.257 * red + 0.504 * green + 0.098 * blue
    u_plane = 128.0 - 0.148 * red - 0.291 * green + 0.439 * blue
    v_plane = 128.0 + 0.439 * red - 0.368 * green - 0.071 * blue
    height, width = bgr.shape[:2]
    luma = np.clip(luma, 0, 255).astype(np.uint8)

    # Box average the two chroma planes onto the subsampled grid. NV12 subsamples
    # in both directions, so it needs even dimensions where the BGR path does not.
    if height % 2 or width % 2:
        raise SystemExit(
            "NV12 needs even dimensions, got %dx%d" % (width, height))
    u = u_plane.reshape(height // 2, 2, width // 2, 2)
    v = v_plane.reshape(height // 2, 2, width // 2, 2)
    u = np.clip(u.mean(axis=(1, 3)), 0, 255).astype(np.uint8)
    v = np.clip(v.mean(axis=(1, 3)), 0, 255).astype(np.uint8)

    chroma = np.empty((height // 2, width), dtype=np.uint8)
    chroma[:, 0::2] = u
    chroma[:, 1::2] = v
    return np.concatenate([luma, chroma], axis=0).reshape(height * 3 // 2,
                                                          width, 1)


def _input_declarations(metadata):
    """The input entries of a model metadata response, whatever shape it is.

    The three clients do not agree: gRPC hands back a protobuf message where the
    fields are attributes, HTTP a mapping with string keys.
    """
    declarations = getattr(metadata, "inputs", None)
    if declarations is None:
        declarations = metadata["inputs"]
    names = []
    for declaration in declarations:
        name = getattr(declaration, "name", None)
        if name is None:
            name = declaration["name"]
        names.append(str(name))
    return names


def input_name(client, model):
    """The image input `model` declares: `YUV` for packed NV12, else `IMAGE`."""
    try:
        if YUV_INPUT in _input_declarations(client.get_model_metadata(model)):
            return YUV_INPUT
    except Exception:  # noqa: BLE001 - not knowing just means "assume IMAGE",
        pass           # and assuming wrong produces a clear error from Triton
    return IMAGE_INPUT


def detect(client, model, image, protocol):
    """Runs one image through one model and returns a `Detection`.

    The tensor name follows the model: an ensemble built on the YUV preprocessor
    declares `YUV` and takes packed NV12, everything else declares `IMAGE` and
    takes the decoded frame. The conversion for the former uses the same matrix
    the BGR kernel encodes with, so a caller holding only a decoded image can
    still reach the no-conversion path.
    """
    field = input_name(client, model)
    payload = to_nv12(image) if field == YUV_INPUT else image
    start = time.time()
    result = client.infer(model, {field: payload}, OUTPUTS,
                          **client_kwargs(protocol))
    elapsed_ms = (time.time() - start) * 1000.0

    count = int(result["NUM_DETS"][0])
    return Detection(count,
                     np.asarray(result["DETECTION_BOXES"])[:count],
                     np.asarray(result["DETECTION_SCORES"])[:count],
                     np.asarray(result["DETECTION_CLASSES"])[:count],
                     elapsed_ms)


def split_models(value):
    """`--model A,B` -> ['A', 'B']."""
    return [name.strip() for name in value.split(",") if name.strip()]


def resolve_image(path):
    if not os.path.exists(path):
        raise SystemExit("no such image: %s" % path)
    return path


def configured_models():
    """Model names from `models_to_load.txt`, comments stripped, in order."""
    try:
        with open(MODELS_FILE, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return []

    names = []
    for line in lines:
        name = line.split("#", 1)[0].strip()
        if name and name not in names:
            names.append(name)
    return names


def get_labels(client, model, protocol="http"):
    """Class names for `model`, as served by the `CUSTOM_LABELS` model.

    Returns an empty list whenever the service cannot answer - not loaded, no
    names file for this model, or a protocol that cannot carry the response -
    so a caller can fall back to its own table. That also means an unknown model
    is not an error.
    """
    if protocol == "shm":
        # Shared memory buffers are sized before the call and the label count is
        # not known in advance, so this variable-length output needs HTTP/gRPC.
        return []

    request = np.array([model.encode("utf-8")], dtype=object)
    try:
        result = client.infer(LABELS_MODEL, {"model_name": request}, ["labels"])
    except Exception:  # noqa: BLE001 - any failure just means "no labels"
        return []

    return [label.decode("utf-8") if isinstance(label, (bytes, bytearray))
            else str(label) for label in result["labels"]]
