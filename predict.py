# Copyright 2022 Lunar Ring. All rights reserved.
# Written by Johannes Stelzer, email stelzer@lunar-ring.ai twitter @j_stelzer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

# from stable_diffusion_holder import StableDiffusionHolder
import torch
from tqdm import tqdm

torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from nanosam.utils.predictor import Predictor as nanoPredictor
import os

# Examples https://github.com/NVIDIA/TensorRT/blob/release/8.6/samples/python/detectron2/build_engine.py
import tensorrt as trt
import numpy as np
import os
import sys
import logging
from time import sleep

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

# Initialize TensorRT logger
VERBOSE = False
workspace = 8
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER.min_severity = trt.Logger.Severity.WARNING
trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")


def build_engine(onnx_path: str) -> None:
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * (2**30)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(f"/src/data/{onnx_path}", "rb") as f:
        if not parser.parse(f.read()):
            log.error(f"Failed to load ONNX file: {onnx_path}")
            for error in range(parser.num_errors):
                log.error(parser.get_error(error))
            sys.exit(1)

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    profile = builder.create_optimization_profile()

    # Get shapes here
    # https://github.com/NVIDIA-AI-IOT/nanosam/blob/653633614b2eb93b06ba3be9adb2aeffb117bd72/README.md?plain=1#L158
    if onnx_path.startswith("mobile"):
        # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/OptimizationProfile.html#tensorrt.IOptimizationProfile.set_shape
        profile.set_shape("point_coords", min=(1, 1, 2), opt=(1, 1, 2), max=(1, 10, 2))
        profile.set_shape("point_labels", min=(1, 1), opt=(1, 1), max=(1, 10))
        config.add_optimization_profile(profile)

    dynamic_inputs = False

    engine_path = os.path.realpath(f"{onnx_path.split('.onnx')[0]}.engine")
    engine_dir = os.path.dirname(engine_path)
    os.makedirs(engine_dir, exist_ok=True)
    precision = "fp16"
    log.info(f"Building {precision} Engine in {engine_path}")

    if precision in ["fp16", "int8"]:
        if not builder.platform_has_fast_fp16:
            log.warning(f"FP16 is not supported natively on this platform/device")
        config.set_flag(trt.BuilderFlag.FP16)

    engine_bytes = builder.build_serialized_network(network, config)

    with open(f"{engine_path}", "wb") as f:
        log.info(f"Serializing engine to file: {engine_path}")
        f.write(engine_bytes)


class Predictor(BasePredictor):
    def setup(self) -> None:
        # Load checkpoint from pre-downloaded location
        if not os.path.exists("/src/data/mobile_sam_mask_decoder.engine"):
            build_engine("mobile_sam_mask_decoder.onnx")

        if not os.path.exists("/src/data/resnet18_image_encoder.engine"):
            build_engine("resnet18_image_encoder.onnx")

        self.predictor = nanoPredictor(
            "/src/data/resnet18_image_encoder.engine",
            "/src/data/mobile_sam_mask_decoder.engine",
        )

    def predict(
        self,
        text: str = Input(description="What to dream"),
    ) -> Path:
        # Read image and run image encoder
        image = PIL.Image.open("assets/dogs.jpg")

        self.predictor.set_image(image)

        # Segment using bounding box
        bbox = [600, 600, 1050, 1059]  # x0, y0, x1, y1

        points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])

        # point_labels = np.array([2, 3])
        point_labels = np.array([0, 1])

        mask, _, _ = self.predictor.predict(points, point_labels)

        mask = (mask[0, 0] > 0).detach().cpu().numpy()

        # Draw resykts
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)
        x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
        y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
        plt.plot(x, y, "g-")
        plt.savefig("./basic_usage_out.jpg")

        return Path(f"./basic_usage_out.jpg")
