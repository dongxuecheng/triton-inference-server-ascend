"""Benchmarks the YUV (no colour conversion) path against the BGR one.

`YOLO26_DET_PRE_YUV_ENSEMBLE` and `YOLO26_DET_PRE_ENSEMBLE` end at the same
canvas by different routes: the BGR model converts every pixel to YUV before
resampling, the YUV one resamples the planes it is handed. They also take
different payloads - 1.5 bytes per pixel against 3 - so a single end-to-end
number mixes the transport difference with the kernel difference.

The two are separated the way `bench_upload.py` does it: the output is always
640x640, so the kernel samples roughly the same number of pixels whatever the
input size, and shrinking the input mostly removes upload time. The slope of
that line is the host-to-device bandwidth, which then converts the payload
difference into milliseconds.

Usage:
    python3 triton_client/benchmark.py [--image PATH] [--runs N] [--protocol http]
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton_client import TritonClient, yolo  # noqa: E402
from triton_client.yolo import to_nv12  # noqa: E402

DETS = ["NUM_DETS", "DETECTION_BOXES", "DETECTION_SCORES", "DETECTION_CLASSES"]

BGR_PRE = "YOLO_640_LETTERBOX_PREPROCESS"
YUV_PRE = "YUV_640_LETTERBOX_PREPROCESS"
BGR_CHAIN = "YOLO26_DET_PRE_ENSEMBLE"
YUV_CHAIN = "YOLO26_DET_PRE_YUV_ENSEMBLE"

# Input sizes for the bandwidth fit. The source is resized, not cropped, so the
# kernel still fills the whole canvas; only the upload changes.
BANDWIDTH_SIZES = [(1568, 1068), (640, 640), (320, 320)]


def measure(client, model, inputs, outputs, runs, warmup=5, **kwargs):
    for _ in range(warmup):
        client.infer(model, inputs, outputs, **kwargs)
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        client.infer(model, inputs, outputs, **kwargs)
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def report(label, samples):
    ordered = sorted(samples)
    median = statistics.median(samples)
    print("  %-50s %7.2f   %6.2f  %6.2f  %5.2f"
          % (label, median, min(samples), ordered[int(len(ordered) * 0.9)],
             statistics.pstdev(samples)))
    return median


def bandwidth(client, image_path, protocol, runs):
    """MB/s host->device from the slope of time against payload size.

    Every run produces the same 640x640 canvas, so the kernel cost is close to
    constant and the difference between two sizes is almost pure upload.
    """
    import cv2

    print("\n[upload bandwidth]  %s, 640x640 canvas either way" % YUV_PRE)
    print("  %-50s %7s   %6s  %6s  %5s" % ("input size", "median", "min", "p90",
                                           "sd"))
    points = []
    source = cv2.imread(image_path)
    for width, height in BANDWIDTH_SIZES:
        frame = cv2.resize(source, (width, height))
        nv12 = to_nv12(frame)
        megabytes = nv12.size / 1e6
        samples = measure(client, YUV_PRE, {"YUV": nv12}, ["TRANSFORM_METADATA"],
                          runs, **({"output_specs": SHM_OUTPUT_SPECS}
                                   if protocol == "shm" else {}))
        points.append((megabytes, report("NV12 %dx%d (%.2f MB)" % (width, height,
                                                                   megabytes),
                                         samples)))

    points.sort()
    smallest, largest = points[0], points[-1]
    delta_mb = largest[0] - smallest[0]
    delta_ms = largest[1] - smallest[1]
    rate = delta_mb / (delta_ms / 1000.0) if delta_ms > 0 else float("nan")
    print("\n  %.2f MB more costs %.2f ms -> %.0f MB/s host->device"
          % (delta_mb, delta_ms, rate))
    return rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=yolo.DEFAULT_IMAGE)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--protocol", default="http",
                        choices=["http", "grpc", "shm"])
    args = parser.parse_args()

    import cv2

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit("cannot read %s" % args.image)
    height, width = bgr.shape[:2]
    nv12 = to_nv12(bgr)
    bgr_bytes = bgr.size
    nv12_bytes = nv12.size
    print("%s  %dx%d, %d runs each, %s" % (args.image, width, height,
                                           args.runs, args.protocol))
    print("payload: BGR %d bytes (%.2f MB) vs NV12 %d bytes (%.2f MB), %.0f%%"
          % (bgr_bytes, bgr_bytes / 1e6, nv12_bytes, nv12_bytes / 1e6,
             100.0 * nv12_bytes / bgr_bytes))

    shm_kwargs = ({"output_specs": yolo.SHM_OUTPUT_SPECS}
                  if args.protocol == "shm" else {})
    with TritonClient(yolo.default_url(args.protocol),
                      protocol=args.protocol) as client:
        for model in (BGR_PRE, YUV_PRE, BGR_CHAIN, YUV_CHAIN):
            if not client.is_model_ready(model):
                raise SystemExit("model not ready: %s" % model)

        print("\n[end to end]  preprocess + infer + postprocess")
        print("  %-50s %7s   %6s  %6s  %5s" % ("chain", "median", "min", "p90",
                                               "sd"))
        bgr_chain = report("%s <- IMAGE" % BGR_CHAIN,
                           measure(client, BGR_CHAIN, {"IMAGE": bgr}, DETS,
                                   args.runs, **shm_kwargs))
        yuv_chain = report("%s <- YUV" % YUV_CHAIN,
                           measure(client, YUV_CHAIN, {"YUV": nv12}, DETS,
                                   args.runs, **shm_kwargs))

        print("\n[preprocess alone]  metadata only, kernel runs, nothing copied "
              "back")
        print("  %-50s %7s   %6s  %6s  %5s" % ("model", "median", "min", "p90",
                                               "sd"))
        bgr_pre = report("%s <- IMAGE" % BGR_PRE,
                         measure(client, BGR_PRE, {"IMAGE": bgr},
                                 ["TRANSFORM_METADATA"], args.runs, **shm_kwargs))
        yuv_pre = report("%s <- YUV" % YUV_PRE,
                         measure(client, YUV_PRE, {"YUV": nv12},
                                 ["TRANSFORM_METADATA"], args.runs, **shm_kwargs))

        print("\n[preprocess + 614 KB canvas out]")
        print("  %-50s %7s   %6s  %6s  %5s" % ("model", "median", "min", "p90",
                                               "sd"))
        bgr_out = report("%s -> NV12" % BGR_PRE,
                         measure(client, BGR_PRE, {"IMAGE": bgr}, ["NV12"],
                                 args.runs, **shm_kwargs))
        yuv_out = report("%s -> NV12" % YUV_PRE,
                         measure(client, YUV_PRE, {"YUV": nv12}, ["NV12"],
                                 args.runs, **shm_kwargs))

        print("\n[sanity]  the two chains must find the same objects")
        for model, field, payload in ((BGR_CHAIN, "IMAGE", bgr),
                                      (YUV_CHAIN, "YUV", nv12)):
            result = client.infer(model, {field: payload}, DETS, **shm_kwargs)
            count = int(result["NUM_DETS"][0])
            scores = np.asarray(result["DETECTION_SCORES"])[:count]
            print("  %-50s %d detections, best %.3f"
                  % (model, count, scores.max()))

        rate = bandwidth(client, args.image, args.protocol, args.runs)

    upload_bgr = bgr_bytes / 1e6 / rate * 1000.0
    upload_nv12 = nv12_bytes / 1e6 / rate * 1000.0
    transport = upload_bgr - upload_nv12
    saved = bgr_pre - yuv_pre
    print("\n[attribution]  the fit bounds the upload at %.0f MB/s, so every"
          "\n              upload figure below is an upper bound:" % rate)
    print("  upload:  BGR <= %6.2f ms   NV12 <= %6.2f ms   saved <= %6.2f ms"
          % (upload_bgr, upload_nv12, transport))
    print("  kernel:  BGR >= %6.2f ms   YUV  >= %6.2f ms   saved >= %6.2f ms"
          % (bgr_pre - upload_bgr, yuv_pre - upload_nv12,
             (bgr_pre - upload_bgr) - (yuv_pre - upload_nv12)))
    print("  copy out: %.2f ms either way (same 614 KB canvas)"
          % (yuv_out - yuv_pre))
    print("\n  the preprocess step saves %.2f ms: at most %.2f ms of that is the"
          "\n  smaller payload, at least %.2f ms is the colour conversion that no"
          "\n  longer happens. A wider picture also makes the kernel stage more"
          "\n  source per chunk, and that extra read sits inside the slope too, so"
          "\n  the real split leans further towards the kernel.\n"
          % (saved, transport, saved - transport))
    print("  chain total: BGR %6.2f ms   YUV %6.2f ms   saved %6.2f ms (%.2fx)"
          % (bgr_chain, yuv_chain, bgr_chain - yuv_chain,
             bgr_chain / yuv_chain))
    return 0


if __name__ == "__main__":
    sys.exit(main())
