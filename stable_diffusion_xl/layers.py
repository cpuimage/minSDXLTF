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
import numpy as np

from keras import layers, activations, ops


class VaeAttentionBlock(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm = layers.GroupNormalization(epsilon=1e-5)
        self.q = layers.Dense(output_dim, use_bias=True)
        self.k = layers.Dense(output_dim, use_bias=True)
        self.v = layers.Dense(output_dim, use_bias=True)
        self.proj_out = layers.Dense(output_dim, use_bias=True)

    def call(self, inputs):
        x = self.norm(inputs)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # Compute attention
        shape = ops.shape(q)
        h, w, c = shape[1], shape[2], shape[3]
        q = ops.reshape(q, (-1, h * w, c))  # b, hw, c
        k = ops.transpose(k, (0, 3, 1, 2))
        k = ops.reshape(k, (-1, c, h * w))  # b, c, hw
        y = q @ k
        y = y * 1 / ops.sqrt(ops.cast(c, self.compute_dtype))
        y = activations.softmax(y)

        # Attend to values
        v = ops.transpose(v, (0, 3, 1, 2))
        v = ops.reshape(v, (-1, c, h * w))
        y = ops.transpose(y, (0, 2, 1))
        x = v @ y
        x = ops.transpose(x, (0, 2, 1))
        x = ops.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + inputs


class VaeResnetBlock(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.padding = layers.ZeroPadding2D(padding=1)
        self.norm1 = layers.GroupNormalization(epsilon=1e-5)
        self.conv1 = layers.Conv2D(output_dim, 3, strides=1)
        self.norm2 = layers.GroupNormalization(epsilon=1e-5)
        self.conv2 = layers.Conv2D(output_dim, 3, strides=1)

    def build(self, input_shape):
        if input_shape[-1] != self.output_dim:
            self.residual_projection = layers.Conv2D(self.output_dim, 1, strides=1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(self.padding(activations.swish(self.norm1(inputs))))
        x = self.conv2(self.padding(activations.swish(self.norm2(x))))
        return x + self.residual_projection(inputs)


class CLIPEmbedding(layers.Layer):
    def __init__(self, input_dim=49408, output_dim=768, max_length=77, **kwargs):
        super().__init__(**kwargs)
        self.token_embedding = layers.Embedding(input_dim, output_dim, name="token_embedding")
        self.position_embedding = layers.Embedding(max_length, output_dim, name="position_embedding")

    def call(self, inputs):
        tokens, positions = inputs
        tokens = self.token_embedding(tokens)
        positions = self.position_embedding(positions)
        return tokens + positions


class CLIPEncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm1")
        self.clip_attn = CLIPAttention(embed_dim, num_heads, causal=True, name="self_attn")
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm2")
        self.fc1 = layers.Dense(embed_dim * 4, name="mlp.fc1")
        self.fc2 = layers.Dense(embed_dim, name="mlp.fc2")
        self.activation = activation

    def call(self, inputs):
        residual = inputs
        x = self.layer_norm1(inputs)
        x = self.clip_attn(x)
        x = residual + x
        residual = x
        x = self.layer_norm2(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + residual


class CLIPAttention(layers.Layer):
    def __init__(self, embed_dim=768, num_heads=12, causal=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = layers.Dense(self.embed_dim, name="q_proj")
        self.k_proj = layers.Dense(self.embed_dim, name="k_proj")
        self.v_proj = layers.Dense(self.embed_dim, name="v_proj")
        self.out_proj = layers.Dense(self.embed_dim, name="out_proj")

    def reshape_states(self, x, sequence_length, batch_size):
        x = ops.reshape(
            x, (batch_size, sequence_length, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))  # bs, heads, sequence_length, head_dim

    def call(self, inputs, attention_mask=None):
        if attention_mask is None and self.causal:
            length = inputs.get_shape().as_list()[1]
            attention_mask = ops.cast(np.triu(np.ones((1, 1, length, length), dtype="float32") * -np.inf, k=1),
                                      dtype=self.compute_dtype)
        _, tgt_len, embed_dim = inputs.shape
        query_states = self.q_proj(inputs) * self.scale
        key_states = self.reshape_states(self.k_proj(inputs), tgt_len, -1)
        value_states = self.reshape_states(self.v_proj(inputs), tgt_len, -1)
        proj_shape = (-1, tgt_len, self.head_dim)
        query_states = self.reshape_states(query_states, tgt_len, -1)
        query_states = ops.reshape(query_states, proj_shape)
        key_states = ops.reshape(key_states, proj_shape)
        src_len = tgt_len
        value_states = ops.reshape(value_states, proj_shape)
        attn_weights = query_states @ ops.transpose(key_states, (0, 2, 1))
        attn_weights = ops.reshape(attn_weights, (-1, self.num_heads, tgt_len, src_len))
        attn_weights = attn_weights + attention_mask
        attn_weights = ops.reshape(attn_weights, (-1, tgt_len, src_len))
        attn_weights = ops.softmax(attn_weights)
        attn_output = attn_weights @ value_states
        attn_output = ops.reshape(attn_output, (-1, self.num_heads, tgt_len, self.head_dim))
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (-1, tgt_len, embed_dim))
        return self.out_proj(attn_output)


class DownSampler(layers.Layer):
    def __init__(self, out_channels, padding=(1, 1), trainable=True, name=None, **kwargs):
        super(DownSampler, self).__init__(name=name, trainable=trainable, **kwargs)
        self.padding2d = layers.ZeroPadding2D(padding=padding)
        self.conv2d = layers.Conv2D(out_channels, 3, strides=2, name="conv")

    def call(self, inputs, *args, **kwargs):
        x = self.padding2d(inputs)
        return self.conv2d(x)


class UpSampler(layers.Layer):
    def __init__(self, out_channels, trainable=True, name=None, **kwargs):
        super(UpSampler, self).__init__(name=name, trainable=trainable, **kwargs)
        self.padding2d = layers.ZeroPadding2D(padding=1)
        self.conv2d = layers.Conv2D(out_channels, 3, strides=1, name="conv")
        self.up = layers.UpSampling2D(2)

    def call(self, inputs, *args, **kwargs):
        x = self.up(inputs)
        x = self.padding2d(x)
        return self.conv2d(x)


class Attention(layers.Layer):
    def __init__(self, query_dim, heads, dim_head, dropout=0.0, trainable=True, name=None, **kwargs):
        super(Attention, self).__init__(name=name, trainable=trainable, **kwargs)
        inner_dim = dim_head * heads
        self.num_heads = heads
        self.head_size = dim_head
        self.drop = layers.Dropout(rate=dropout)
        self.to_q = layers.Dense(inner_dim, use_bias=False, name="to_q")
        self.to_k = layers.Dense(inner_dim, use_bias=False, name="to_k")
        self.to_v = layers.Dense(inner_dim, use_bias=False, name="to_v")
        self.to_out = layers.Dense(query_dim, use_bias=True, name="to_out.0")
        self.scale = dim_head ** -0.5

    def call(self, hidden_states, encoder_hidden_states=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)
        batch_size = ops.shape(hidden_states)[0]
        q = ops.reshape(q, (batch_size, hidden_states.shape[1], self.num_heads, self.head_size))
        k = ops.reshape(k, (batch_size, -1, self.num_heads, self.head_size))
        v = ops.reshape(v, (batch_size, -1, self.num_heads, self.head_size))
        q = ops.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        k = ops.transpose(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
        v = ops.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        score = ops.einsum('bnqh,bnhk->bnqk', q, k) * self.scale
        weights = activations.softmax(score)  # (bs, num_heads, time, time)
        attn = ops.einsum('bnqk,bnkh->bnqh', weights, v)
        attn = ops.transpose(attn, (0, 2, 1, 3))  # (bs, time, num_heads, head_size)
        out = ops.reshape(attn, (-1, hidden_states.shape[1], self.num_heads * self.head_size))
        out = self.drop(out)
        return self.to_out(out)


class ResnetBlock(layers.Layer):
    def __init__(self, out_channels, groups=32, trainable=True, name=None, **kwargs):
        super(ResnetBlock, self).__init__(name=name, trainable=trainable, **kwargs)
        self.norm1 = layers.GroupNormalization(groups, epsilon=1e-05, center=True, scale=True, name="norm1")
        self.padding2d = layers.ZeroPadding2D(padding=1)
        self.conv1 = layers.Conv2D(out_channels, 3, strides=1, name="conv1")
        self.time_emb_proj = layers.Dense(out_channels, name="time_emb_proj")
        self.norm2 = layers.GroupNormalization(groups, epsilon=1e-05, center=True, scale=True, name="norm2")
        self.conv2 = layers.Conv2D(out_channels, 3, strides=1, name="conv2")
        self.nonlinearity = layers.Activation("swish")
        self.conv_shortcut = None
        self.out_channels = out_channels

    def build(self, input_shape):
        axis = -1
        if input_shape[0][axis] != self.out_channels:
            self.conv_shortcut = layers.Conv2D(self.out_channels, kernel_size=1, strides=1,
                                               name="conv_shortcut")

    def call(self, inputs, *args, **kwargs):
        hidden_states, time_emb = inputs
        x = hidden_states
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(self.padding2d(x))
        emb = self.time_emb_proj(time_emb)
        emb = emb[:, None, None, :]
        x = x + emb
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(self.padding2d(x))
        if self.conv_shortcut is not None:
            hidden_states = self.conv_shortcut(hidden_states)
        hidden_states = hidden_states + x
        return hidden_states


class Timesteps(layers.Layer):
    def __init__(self, num_channels: int = 320, trainable=False, name=None, **kwargs):
        super(Timesteps, self).__init__(name=name, trainable=trainable, **kwargs)
        self.num_channels = num_channels

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        half_dim = self.num_channels // 2
        exponent = -ops.log(10000.0) * ops.arange(0, half_dim, dtype="float32")
        exponent = exponent / (half_dim - 0.0)
        emb = ops.exp(exponent)
        emb = ops.cast(ops.expand_dims(inputs, axis=-1), "float32") * ops.expand_dims(emb, axis=0)
        sin_emb = ops.sin(emb)
        cos_emb = ops.cos(emb)
        emb = ops.concatenate([cos_emb, sin_emb], axis=-1)
        return emb

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_channels": self.num_channels,
        })
        return config


class GEGLU(layers.Layer):
    def __init__(self, out_features, trainable=True, name=None, **kwargs):
        super(GEGLU, self).__init__(name=name, trainable=trainable, **kwargs)
        self.proj = layers.Dense(out_features * 2, use_bias=True, name="proj")

    def call(self, inputs, *args, **kwargs):
        x_proj = self.proj(inputs)
        x1, x2 = ops.split(x_proj, 2, axis=-1)
        return x1 * ops.gelu(x2, approximate=True)


class BasicTransformerBlock(layers.Layer):
    def __init__(self, dim, num_attention_heads, attention_head_dim, trainable=True, name=None, **kwargs):
        super(BasicTransformerBlock, self).__init__(name=name, trainable=trainable, **kwargs)
        # 1. Self-Attn
        self.norm1 = layers.LayerNormalization(epsilon=1e-05, center=True, scale=True, name="norm1")
        self.attn1 = Attention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, name="attn1")
        # 2. Cross-Attn
        self.norm2 = layers.LayerNormalization(epsilon=1e-05, center=True, scale=True, name="norm2")
        self.attn2 = Attention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, name="attn2")
        # 3. Feed-forward
        self.norm3 = layers.LayerNormalization(epsilon=1e-05, center=True, scale=True, name="norm3")
        self.ff0 = GEGLU(dim * 4, name="ff.net.0")
        self.ff2 = layers.Dense(dim, name="ff.net.2")

    def call(self, inputs, *args, **kwargs):
        hidden_states, encoder_hidden_states = inputs
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff0(norm_hidden_states)
        ff_output = self.ff2(ff_output)
        hidden_states = ff_output + hidden_states
        return hidden_states


class AttentionBlock(layers.Layer):
    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1, norm_num_groups=32,
                 trainable=True, name=None, **kwargs):
        super(AttentionBlock, self).__init__(name=name, trainable=trainable, **kwargs)
        inner_dim = num_attention_heads * attention_head_dim
        self.norm = layers.GroupNormalization(norm_num_groups, epsilon=1e-06, center=True, scale=True,
                                              name="norm")
        self.proj_in = layers.Dense(inner_dim, name="proj_in")
        self.transformer_blocks = [
            BasicTransformerBlock(inner_dim, num_attention_heads, attention_head_dim,
                                  name="transformer_blocks.{}".format(idx)) for idx in range(num_layers)]
        self.proj_out = layers.Dense(in_channels, name="proj_out")

    def call(self, inputs, *args, **kwargs):
        hidden_states, text_emb = inputs
        batch = ops.shape(hidden_states)[0]
        height, width = hidden_states.shape[1:3]
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[-1]
        hidden_states = ops.reshape(hidden_states, (batch, height * width, inner_dim))
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                (hidden_states, text_emb))

        hidden_states = self.proj_out(hidden_states)
        hidden_states = ops.reshape(hidden_states, (batch, height, width, inner_dim))
        hidden_states = hidden_states + residual
        return hidden_states
