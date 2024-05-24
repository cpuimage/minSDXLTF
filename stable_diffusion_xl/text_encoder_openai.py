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

from stable_diffusion_xl.ckpt_loader import load_weights_from_file
from stable_diffusion_xl.layers import CLIPEmbedding, CLIPEncoderLayer


def quick_gelu(x):
    return x * tf.sigmoid(x * 1.702)


class TextEncoderOpenAi(tf.keras.Model):
    def __init__(self, max_length=77, embed_dim=768, vocab_size=49408, num_heads=12, num_layers=12, clip_skip=-2,
                 final_layer_norm=False,
                 name=None,
                 ckpt_path=None, lora_dict=None):
        tokens = tf.keras.layers.Input(shape=(max_length,), dtype="int32", name="tokens")
        positions = tf.keras.layers.Input(shape=(max_length,), dtype="int32", name="positions")
        clip_emb = CLIPEmbedding(vocab_size, embed_dim, max_length, name="embeddings")([tokens, positions])
        x = clip_emb
        out = []
        for idx in range(num_layers):
            x = CLIPEncoderLayer(embed_dim, num_heads, activation=quick_gelu,
                                 name="text_model.encoder.layers.{}".format(idx))(x)
            out.append(x)
        embedded = out[clip_skip]
        if final_layer_norm:
            embedded = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="text_model.final_layer_norm")(embedded)
        super().__init__([tokens, positions], embedded, name=name)
        origin = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.fp16.safetensors"
        ckpt_mapping = [('text_model.embeddings.token_embedding.weight', None),
                        ('text_model.embeddings.position_embedding.weight', None)]
        for idx in range(0, num_layers + clip_skip + 1):
            layers_name = 'text_model.encoder.layers.{}'.format(idx)
            ckpt_mapping.append(('{}.layer_norm1.weight'.format(layers_name), None))
            ckpt_mapping.append(('{}.layer_norm1.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.self_attn.q_proj.weight'.format(layers_name), (1, 0)))
            ckpt_mapping.append(('{}.self_attn.q_proj.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.self_attn.k_proj.weight'.format(layers_name), (1, 0)))
            ckpt_mapping.append(('{}.self_attn.k_proj.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.self_attn.v_proj.weight'.format(layers_name), (1, 0)))
            ckpt_mapping.append(('{}.self_attn.v_proj.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.self_attn.out_proj.weight'.format(layers_name), (1, 0)))
            ckpt_mapping.append(('{}.self_attn.out_proj.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.layer_norm2.weight'.format(layers_name), None))
            ckpt_mapping.append(('{}.layer_norm2.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.mlp.fc1.weight'.format(layers_name), (1, 0)))
            ckpt_mapping.append(('{}.mlp.fc1.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.mlp.fc2.weight'.format(layers_name), (1, 0)))
            ckpt_mapping.append(('{}.mlp.fc2.bias'.format(layers_name), None))
        if final_layer_norm:
            ckpt_mapping.append(('text_model.final_layer_norm.weight', None))
            ckpt_mapping.append(('text_model.final_layer_norm.bias', None))
        if ckpt_path is not None:
            if os.path.exists(ckpt_path):
                load_weights_from_file(self, ckpt_path, ckpt_mapping=ckpt_mapping, lora_dict=lora_dict)
                return
            else:
                origin = ckpt_path
        model_weights_fpath = tf.keras.utils.get_file(origin=origin, fname="text_encoder.fp16.safetensors")
        if os.path.exists(model_weights_fpath):
            load_weights_from_file(self, model_weights_fpath, ckpt_mapping=ckpt_mapping, lora_dict=lora_dict)
