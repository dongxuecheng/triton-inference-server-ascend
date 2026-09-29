"""Serves the class names for a model.

Triton's Python backend rather than a compiled Ascend backend: at load time it
reads every `names/<model_name>.txt` and keeps the lines in memory, then hands
the requested set back as a single variable-length string tensor. Line order is
the class id order, so the index of a label is the class id the postprocess
backends return.
"""

import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils

DEFAULT_NAMES_DIR = "/models/CUSTOM_LABELS/names"


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        parameter = config.get("parameters", {}).get("names_directory", {})
        if isinstance(parameter, dict):
            names_dir = parameter.get("string_value", DEFAULT_NAMES_DIR)
        else:
            names_dir = str(parameter)

        self.labels = {}
        if not os.path.isdir(names_dir):
            print("[CUSTOM_LABELS] no such directory: %s" % names_dir)
            return

        for entry in sorted(os.listdir(names_dir)):
            if not entry.endswith(".txt"):
                continue
            with open(os.path.join(names_dir, entry), "r", encoding="utf-8") as handle:
                self.labels[entry[:-4]] = [
                    line.strip() for line in handle if line.strip()
                ]

        print("[CUSTOM_LABELS] %d label set(s) from %s: %s"
              % (len(self.labels), names_dir, ", ".join(sorted(self.labels))))

    def execute(self, requests):
        responses = []
        for request in requests:
            name = ""
            tensor = pb_utils.get_input_tensor_by_name(request, "model_name")
            if tensor is not None:
                values = tensor.as_numpy().reshape(-1)
                if values.size:
                    name = values[0]
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")

            labels = self.labels.get(name, [])
            encoded = np.array([label.encode("utf-8") for label in labels],
                               dtype=object)
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("labels", encoded)]))
        return responses

    def finalize(self):
        pass
