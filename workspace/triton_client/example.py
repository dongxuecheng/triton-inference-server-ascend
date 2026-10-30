"""Runs this project's YOLO ensembles through the client wrapper.

The wrapper is protocol-agnostic: the same call works over HTTP, gRPC and
system shared memory. Only the tensor names differ from the wrapper's own
README, because this project names the ensemble input `IMAGE` and returns
`NUM_DETS` / `DETECTION_BOXES` / `DETECTION_SCORES` / `DETECTION_CLASSES`, and
because the models run with `max_batch_size: 0`, so the image goes in as
`[height, width, 3]` without a batch dimension.

Usage:
    python3 example.py --protocol http|grpc|shm [--model NAME] [--image PATH]

See `visualize.py` for drawing the detections onto the image.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton_client import TritonClient, yolo  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="grpc", choices=["http", "grpc", "shm"])
    parser.add_argument("--model", default=yolo.DEFAULT_MODEL)
    parser.add_argument("--image", default=yolo.DEFAULT_IMAGE)
    parser.add_argument("--url", default=None,
                        help="defaults to %s (grpc) or %s (http)"
                             % (yolo.GRPC_URL, yolo.HTTP_URL))
    args = parser.parse_args()

    url = args.url or yolo.default_url(args.protocol)

    image = yolo.load_image(yolo.resolve_image(args.image))
    print("%s %dx%d -> %s over %s" %
          (os.path.basename(args.image), image.shape[1], image.shape[0],
           args.model, args.protocol))

    with TritonClient(url, protocol=args.protocol) as client:
        print("server ready:", client.is_server_ready())
        print("model ready :", client.is_model_ready(args.model))
        detection = yolo.detect(client, args.model, image, args.protocol)

    print("detections  : %d in %.1f ms" % (detection.count, detection.elapsed_ms))
    print("classes     : %s" % detection.histogram())
    best = detection.best()
    if best:
        print("best        : %s %.3f box=%s" % best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
