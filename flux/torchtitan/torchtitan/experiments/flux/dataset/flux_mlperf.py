# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import os
from typing import Any
from datasets import load_dataset, load_from_disk, Dataset
import torch

from torchtitan.experiments.flux.dataset.flux_dataset import (
    DATASETS,
    TextToImageDatasetConfig,
    _process_cc12m_image
)
from torchtitan.experiments.flux.dataset.tokenizer import FluxTokenizer
from torchvision.transforms import ToTensor
import PIL
import numpy as np

# Avoid PIL.Image.DecompressionBombError
PIL.Image.MAX_IMAGE_PIXELS = 933120000


def _laion_data_processor(
    sample: dict[str, Any],
    t5_tokenizer: FluxTokenizer,
    clip_tokenizer: FluxTokenizer,
    output_size: int = 256,
    pre_transformed: bool = False,
) -> dict[str, Any]:
    """
    Preprocess LAION dataset sample image and text for Flux model.

    Args:
        sample: A sample from dataset
        t5_encoder: T5 encoder
        clip_encoder: CLIP encoder
        output_size: The output image size

    """
    if pre_transformed:
        img = ToTensor()(sample["jpg"])
        img = img * 2.0 - 1.0
    else:
        img = _process_cc12m_image(sample["jpg"], output_size=output_size, skip_low_resolution=False)
    t5_tokens = t5_tokenizer.encode(sample["txt"])
    clip_tokens = clip_tokenizer.encode(sample["txt"])

    return {
        "image": img,
        "clip_tokens": clip_tokens,  # type: List[int]
        "t5_tokens": t5_tokens,  # type: List[int],
        "txt": sample["txt"],
    }


DATASETS["laion"] = TextToImageDatasetConfig(
    path="/dataset/laion",
    loader=lambda path: load_dataset(
        "webdataset",
        split="train",
        data_dir=os.path.join(path, "data"),
        cache_dir=os.path.join(path, "cache"),
        num_proc=8,
    ),
    data_processor=_laion_data_processor,
)

DATASETS["laion_pre_transformed"] = TextToImageDatasetConfig(
    path="/dataset/laion",
    loader=load_from_disk,
    data_processor=partial(_laion_data_processor, pre_transformed=True),
)


def _coco_data_processor(
    sample: dict[str, Any],
    t5_tokenizer: FluxTokenizer,
    clip_tokenizer: FluxTokenizer,
    output_size: int = 256,
    pre_transformed: bool = False,
) -> dict[str, Any]:
    """
    Preprocess COCO dataset sample image and text for Flux model.

    Args:
        sample: A sample from dataset
        t5_encoder: T5 encoder
        clip_encoder: CLIP encoder
        output_size: The output image size

    """
    if pre_transformed:
        img = ToTensor()(sample["jpg"])
        img = img * 2.0 - 1.0
    else:
        img = _process_cc12m_image(sample["jpg"], output_size=output_size, skip_low_resolution=False)
    t5_tokens = t5_tokenizer.encode(sample["txt"])
    clip_tokens = clip_tokenizer.encode(sample["txt"])

    return {
        "image": img,
        "clip_tokens": clip_tokens,  # type: List[int]
        "t5_tokens": t5_tokens,  # type: List[int]
        "txt": sample["txt"],
    }


DATASETS["coco"] = TextToImageDatasetConfig(
    path="/dataset/coco",
    loader=lambda path: load_dataset(
        "webdataset",
        split="train",
        data_dir=os.path.join(path, "data"),
        cache_dir=os.path.join(path, "cache"),
        num_proc=8,
    ),
    data_processor=_coco_data_processor,
)

DATASETS["coco_pre_transformed"] = TextToImageDatasetConfig(
    path="/dataset/coco",
    loader=load_from_disk,
    data_processor=partial(_coco_data_processor, pre_transformed=True),
)

def create_dummy_dataset(num_samples: int = 10000):
    """Create a dummy Hugging Face Dataset for testing."""
    # Create dummy images as PIL Images to ensure compatibility

    dummy_image = PIL.Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))

    data = {
        "txt": ["A photo of a cat"] * num_samples,
        "jpg": [dummy_image] * num_samples
    }
    return Dataset.from_dict(data)

DATASETS["dummy"] = TextToImageDatasetConfig(
    path="dummy",
    loader=lambda path: create_dummy_dataset(10000),
    data_processor=partial(_coco_data_processor, pre_transformed=True),
)
