"""Draws detection results onto the source image.

Runs one or more of this project's detection ensembles over an image and writes
an annotated copy, so a run can be judged by eye instead of by reading numbers.
Passing more than one model also writes a side-by-side comparison.

Usage:
    python3 visualize.py
    python3 visualize.py --protocol shm --conf 0.5
    python3 visualize.py --model YOLO11_DET_PRE_ENSEMBLE,YOLO26_DET_PRE_ENSEMBLE
    python3 visualize.py --image other.jpg --output out/annotated.jpg

The `--conf` filter only affects what is drawn; the threshold the models
actually apply is the `confidence_threshold` parameter in their config.pbtxt.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton_client import TritonClient, coco, yolo  # noqa: E402

FONT = None  # set in main(), cv2 is imported lazily to keep --help fast

# Annotated images land here by default, so a run never drops files next to
# whatever image happened to be passed in.
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "detections")


def overlaps(rect, placed, tolerance=0.35):
    """True if `rect` covers more than `tolerance` of an already placed rect."""
    x1, y1, x2, y2 = rect
    area = max(1, (x2 - x1) * (y2 - y1))
    for px1, py1, px2, py2 in placed:
        width = min(x2, px2) - max(x1, px1)
        height = min(y2, py2) - max(y1, py1)
        if width > 0 and height > 0 and width * height > tolerance * area:
            return True
    return False


def annotate(image, detection, *, conf, thickness, font_scale, show_labels,
             names=None):
    """Returns a copy of `image` with boxes, labels and scores drawn on it.

    Boxes are always drawn. Labels are placed best-score first and skipped when
    they would cover a label that is already there, so a crowded scene stays
    readable instead of turning into a wall of text.
    """
    import cv2

    canvas = image.copy()
    height, width = canvas.shape[:2]
    if thickness <= 0:
        thickness = max(1, int(round(min(height, width) / 400.0)))
    if font_scale <= 0:
        font_scale = max(0.4, min(1.2, min(height, width) / 1200.0))
    text_thickness = max(1, thickness - 1)
    placed = []
    drawn = 0
    labelled = 0

    order = np.argsort(-np.asarray(detection.scores, dtype=np.float32), kind="stable")
    for index in order:
        score = float(detection.scores[index])
        if score < conf:
            continue
        label = int(detection.classes[index])
        drawn += 1

        x1, y1, x2, y2 = (int(round(float(value)))
                          for value in detection.boxes[index])
        x1 = min(max(x1, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        x2 = min(max(x2, x1), width - 1)
        y2 = min(max(y2, y1), height - 1)
        color = coco.color(label)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if not show_labels:
            continue

        text = "%s %.2f" % (coco.name(label, names), score)
        (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale,
                                                     text_thickness)
        box_h = text_h + baseline + 4
        top = y1 - box_h
        if top < 0:                       # no room above, draw inside the box
            top = min(y1 + 2, height - box_h)
        top = max(top, 0)
        rect = (x1, top, min(x1 + text_w + 8, width - 1), top + box_h)
        if overlaps(rect, placed):
            continue
        placed.append(rect)
        labelled += 1

        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
        cv2.putText(canvas, text, (rect[0] + 4, rect[1] + text_h + 2), FONT,
                    font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return canvas, drawn, labelled


def banner(canvas, text):
    """Solid strip along the top naming the model that produced the image."""
    import cv2

    height, width = canvas.shape[:2]
    bar = max(26, int(round(height * 0.05)))
    cv2.rectangle(canvas, (0, 0), (width, bar), (24, 24, 24), -1)
    scale = bar / 46.0
    cv2.putText(canvas, text, (12, int(round(bar * 0.72))), FONT, scale,
                (255, 255, 255), max(1, int(round(scale * 2))), cv2.LINE_AA)
    return canvas


def stack(canvases):
    """Side by side when there are two, otherwise stacked top to bottom."""
    if len(canvases) == 2 and canvases[0].shape[0] == canvases[1].shape[0]:
        return np.hstack(canvases)

    width = max(canvas.shape[1] for canvas in canvases)
    padded = []
    for canvas in canvases:
        if canvas.shape[1] < width:
            filler = np.zeros((canvas.shape[0], width - canvas.shape[1], 3),
                              np.uint8)
            canvas = np.hstack([canvas, filler])
        padded.append(canvas)
    return np.vstack(padded)


def write_image(path, canvas, quality):
    import cv2

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    params = []
    if os.path.splitext(path)[1].lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if not cv2.imwrite(path, canvas, params):
        raise SystemExit("cannot write %s" % path)
    return path


def main():
    global FONT

    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="grpc", choices=["http", "grpc", "shm"])
    parser.add_argument("--model", default=yolo.DEFAULT_MODEL,
                        help="one model, or several separated by commas")
    parser.add_argument("--image", default=yolo.DEFAULT_IMAGE)
    parser.add_argument("--output", default=None,
                        help="output file; only valid with a single model")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="where to write images when --output is not given; "
                             "each is named <image>_<model>")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="do not draw detections below this score")
    parser.add_argument("--no-labels", dest="labels", action="store_false",
                        help="draw boxes without the class/score text")
    parser.add_argument("--font-scale", type=float, default=0.0,
                        help="0 picks a size from the image dimensions")
    parser.add_argument("--thickness", type=int, default=0,
                        help="0 picks a width from the image dimensions")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--url", default=None,
                        help="defaults to %s (grpc) or %s (http)"
                             % (yolo.GRPC_URL, yolo.HTTP_URL))
    args = parser.parse_args()

    models = yolo.split_models(args.model)
    if args.output and len(models) != 1:
        parser.error("--output needs exactly one model; got %d" % len(models))

    import cv2
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    image = yolo.load_image(yolo.resolve_image(args.image))
    height, width = image.shape[:2]
    url = args.url or yolo.default_url(args.protocol)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    extension = os.path.splitext(args.image)[1] or ".jpg"
    directory = args.output_dir

    print("%s %dx%d -> %s over %s (drawing at conf >= %.2f)"
          % (os.path.basename(args.image), width, height,
             ", ".join(models), args.protocol, args.conf))

    canvases = []
    with TritonClient(url, protocol=args.protocol) as client:
        for model in models:
            if not client.is_model_ready(model):
                raise SystemExit("model not ready: %s" % model)

            detection = yolo.detect(client, model, image, args.protocol)
            names = yolo.get_labels(client, model, args.protocol)
            source = yolo.LABELS_MODEL if names else "built-in COCO table"
            if not names:
                names = coco.CLASS_NAMES
            canvas, drawn, labelled = annotate(image, detection,
                                               conf=args.conf,
                                               thickness=args.thickness,
                                               font_scale=args.font_scale,
                                               show_labels=args.labels,
                                               names=names)
            if len(models) > 1:
                canvas = banner(canvas, "%s   %d detections   %.1f ms"
                                       % (model, detection.count,
                                          detection.elapsed_ms))
            canvases.append(canvas)

            path = args.output or os.path.join(
                directory, "%s_%s%s" % (stem, model, extension))
            write_image(path, canvas, args.quality)

            print("%-28s %3d detections in %6.1f ms  drew %3d boxes, %3d labels  -> %s"
                  % (model, detection.count, detection.elapsed_ms, drawn,
                     labelled, path))
            print("%-28s names from %s" % ("", source))
            print("%-28s classes: %s" % ("", detection.histogram(names)))
            best = detection.best(names)
            if best:
                print("%-28s best   : %s %.3f box=%s"
                      % ("", best[0], best[1], best[2]))

    if len(canvases) > 1:
        path = os.path.join(directory, "%s_compare%s" % (stem, extension))
        write_image(path, stack(canvases), args.quality)
        print("combined                     -> %s" % path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
