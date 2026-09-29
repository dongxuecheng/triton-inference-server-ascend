"""COCO class names and stable per-class colours.

The detection backends return integer class ids, so anything that wants to show
them to a human needs this table. Colours are derived from the id instead of
being assigned in detection order, so the same class keeps the same colour
across runs and across models, which is what makes two images comparable.
"""

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# BGR, as OpenCV wants them.
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def name(class_id, names=None):
    """Class id -> label, falling back to the bare id for unknown classes.

    `names` overrides the built-in COCO table, so a caller can pass the list a
    custom model reports through the `CUSTOM_LABELS` service.
    """
    table = CLASS_NAMES if names is None else names
    if 0 <= class_id < len(table):
        return table[class_id]
    return str(class_id)


def color(class_id):
    """Class id -> a BGR colour that is stable across runs."""
    return _PALETTE[class_id % len(_PALETTE)]
