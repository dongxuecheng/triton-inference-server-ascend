"""Prints the class names a model reports.

Asks the `CUSTOM_LABELS` Python backend model what class id N means, so the
list comes from the server rather than from a table baked into the client. A
custom-trained model only needs a `names/<model>.txt` next to that model.

Usage:
    python3 labels.py                       # every ensemble in models_to_load.txt
    python3 labels.py YOLO26_DET_PRE_ENSEMBLE
    python3 labels.py --protocol grpc --grep person
    python3 labels.py --count               # how many classes each has

Only the ensembles have a label file: the step models in a chain expose the same
classes, so they are looked up through the ensemble name. Models that share a
label file are printed as one group, so asking for the whole list does not repeat
the same names once per model.

Only HTTP and gRPC can be used: the response is variable length, and shared
memory needs every buffer sized before the call.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton_client import TritonClient, yolo  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="*",
                        help="models to look up; defaults to the ensembles listed "
                             "in models_to_load.txt")
    parser.add_argument("--protocol", default="http", choices=["http", "grpc"])
    parser.add_argument("--url", default=None,
                        help="defaults to %s (grpc) or %s (http)"
                             % (yolo.GRPC_URL, yolo.HTTP_URL))
    parser.add_argument("--grep", default=None,
                        help="only print labels containing this text")
    parser.add_argument("--count", action="store_true",
                        help="print the number of classes instead of the list")
    args = parser.parse_args()

    if args.model:
        models = list(args.model)
    else:
        # Ensemble names are the only ones with a label file here, and every one
        # of them has one, so the default lookup cannot come up empty-handed.
        models = [name for name in yolo.configured_models()
                  if name.endswith("_ENSEMBLE")]
    if not models:
        raise SystemExit("no models given and %s is unreadable" % yolo.MODELS_FILE)

    url = args.url or yolo.default_url(args.protocol)
    needle = args.grep.lower() if args.grep else None
    missing = []

    with TritonClient(url, protocol=args.protocol) as client:
        if not client.is_model_ready(yolo.LABELS_MODEL):
            raise SystemExit("%s is not loaded; scripts/serve.sh appends it to "
                             "the load list" % yolo.LABELS_MODEL)

        groups = []
        seen = {}
        for model in models:
            labels = yolo.get_labels(client, model, args.protocol)
            if not labels:
                missing.append(model)
                continue
            key = tuple(labels)
            if key in seen:
                seen[key].append(model)
            else:
                seen[key] = [model]
                groups.append((seen[key], labels))

    for group, labels in groups:
        # Keep the class id next to the name: it is what the postprocess
        # backends return, and --grep must not renumber what is printed.
        entries = list(enumerate(labels))
        if needle:
            entries = [(index, label) for index, label in entries
                       if needle in label.lower()]

        print("=== %s | %d classes%s ==="
              % (", ".join(group), len(labels),
                 ", %d matching '%s'" % (len(entries), args.grep)
                 if needle else ""))
        if not args.count:
            for index, label in entries:
                print("%3d  %s" % (index, label))
            print()

    if missing:
        print("no label file for: %s" % ", ".join(missing))
        print("add workspace/models/CUSTOM_LABELS/names/<model>.txt, "
              "one name per line")

    if args.count:
        print("%d label set(s) over %d model(s), %d classes each"
              % (len(groups), len(models) - len(missing),
                 len(groups[0][1]) if groups else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
