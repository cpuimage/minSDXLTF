# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import tensorflow as tf

from .ckpt_loader import load_weights_from_file, CKPT_MAPPING, VAE_KEY_MAPPING
from .layers import GroupNormalization, VaeAttentionBlock, VaeResnetBlock, DownSampler


class ImageEncoder(tf.keras.Sequential):
    """ImageEncoder is the VAE Encoder for StableDiffusionXL."""

    def __init__(self, ckpt_path=None):
        super().__init__(
            [
                tf.keras.layers.Input((None, None, 3)),
                tf.keras.layers.ZeroPadding2D(padding=1),
                tf.keras.layers.Conv2D(128, 3, strides=1),
                VaeResnetBlock(128),
                VaeResnetBlock(128),
                DownSampler(128, padding=((0, 1), (0, 1))),
                VaeResnetBlock(256),
                VaeResnetBlock(256),
                DownSampler(256, padding=((0, 1), (0, 1))),
                VaeResnetBlock(512),
                VaeResnetBlock(512),
                DownSampler(512, padding=((0, 1), (0, 1))),
                VaeResnetBlock(512),
                VaeResnetBlock(512),
                VaeResnetBlock(512),
                VaeAttentionBlock(512),
                VaeResnetBlock(512),
                GroupNormalization(epsilon=1e-5),
                tf.keras.layers.Activation("swish"),
                tf.keras.layers.ZeroPadding2D(padding=1),
                tf.keras.layers.Conv2D(8, 3, strides=1),
                tf.keras.layers.Conv2D(8, 1, strides=1),
                tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1)[0] * 0.13025),
            ])
        origin = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae_1_0/diffusion_pytorch_model.fp16.safetensors"
        ckpt_mapping = CKPT_MAPPING["encoder"]
        if ckpt_path is not None:
            if os.path.exists(ckpt_path):
                load_weights_from_file(self, ckpt_path, ckpt_mapping=ckpt_mapping, key_mapping=VAE_KEY_MAPPING)
                return
            else:
                origin = ckpt_path
        model_weights_fpath = tf.keras.utils.get_file(origin=origin)
        if os.path.exists(model_weights_fpath):
            load_weights_from_file(self, model_weights_fpath, ckpt_mapping=ckpt_mapping, key_mapping=VAE_KEY_MAPPING)
