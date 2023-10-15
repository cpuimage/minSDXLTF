import os

import tensorflow as tf

from stable_diffusion_xl.ckpt_loader import load_weights_from_file, UNET_KEY_MAPPING, CKPT_MAPPING
from .layers import GroupNormalization, Linear, DownSampler, UpSampler, Timesteps, ResnetBlock, AttentionBlock


class DiffusionXLModel(tf.keras.Model):
    @staticmethod
    def push_block(hidden_states, res_stack):
        res_stack.append(hidden_states)
        return res_stack

    @staticmethod
    def pop_block(hidden_states, res_stack):
        res_hidden_states = res_stack.pop()
        hidden_states = tf.concat([hidden_states, res_hidden_states], axis=-1)
        return hidden_states, res_stack

    def __init__(self, img_height=1024, img_width=1024, name=None, ckpt_path=None, lora_dict=None):
        sample = tf.keras.layers.Input((img_height // 8, img_width // 8, 4))
        timestep = tf.keras.layers.Input(())
        text_emb = tf.keras.layers.Input((None, 2048))
        text_embeds = tf.keras.layers.Input((1280,))
        time_ids = tf.keras.layers.Input((6,))
        # 1. time
        t_emb = Timesteps(320, name="time_proj")(timestep)
        t_emb = tf.reshape(t_emb, (-1, 320))
        t_emb = Linear(1280, name="time_embedding.linear_1")(tf.cast(t_emb, sample.dtype))
        t_emb = tf.keras.layers.Activation("swish")(t_emb)
        t_emb = Linear(1280, name="time_embedding.linear_2")(t_emb)
        time_embeds = Timesteps(256, name="add_time_proj")(time_ids)
        time_embeds = tf.reshape(time_embeds, (-1, 1536))  # 6*256 = 1536
        add_embeds = tf.concat([text_embeds, time_embeds], axis=-1)
        add_embeds = tf.cast(add_embeds, sample.dtype)
        add_embeds = Linear(1280, name="add_embedding.linear_1")(add_embeds)
        add_embeds = tf.keras.layers.Activation("swish")(add_embeds)
        add_embeds = Linear(1280, name="add_embedding.linear_2")(add_embeds)
        time_emb = tf.keras.layers.Activation("swish")(t_emb + add_embeds)
        # 2. pre-process
        hidden_states = tf.keras.layers.Conv2D(320, kernel_size=3, strides=1, name="conv_in")(
            tf.keras.layers.ZeroPadding2D(1)(sample))
        res_stack = [hidden_states]
        # 3. blocks
        # DownBlock2D
        hidden_states = ResnetBlock(320, name="down_blocks.0.resnets.0")((hidden_states, time_emb))
        res_stack = self.push_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(320, name="down_blocks.0.resnets.1")((hidden_states, time_emb))
        res_stack = self.push_block(hidden_states, res_stack)
        hidden_states = DownSampler(320, name="down_blocks.0.downsamplers.0")(hidden_states)
        res_stack = self.push_block(hidden_states, res_stack)
        # CrossAttnDownBlock2D
        hidden_states = ResnetBlock(640, name="down_blocks.1.resnets.0")((hidden_states, time_emb))
        hidden_states = AttentionBlock(10, 64, 640, 2, name="down_blocks.1.attentions.0")((hidden_states, text_emb))
        res_stack = self.push_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(640, name="down_blocks.1.resnets.1")((hidden_states, time_emb))
        hidden_states = AttentionBlock(10, 64, 640, 2, name="down_blocks.1.attentions.1")((hidden_states, text_emb))
        res_stack = self.push_block(hidden_states, res_stack)
        hidden_states = DownSampler(640, name="down_blocks.1.downsamplers.0")(hidden_states)
        res_stack = self.push_block(hidden_states, res_stack)
        # CrossAttnDownBlock2D
        hidden_states = ResnetBlock(1280, name="down_blocks.2.resnets.0")((hidden_states, time_emb))
        hidden_states = AttentionBlock(20, 64, 1280, 10, name="down_blocks.2.attentions.0")((hidden_states, text_emb))
        res_stack = self.push_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(1280, name="down_blocks.2.resnets.1")((hidden_states, time_emb))
        hidden_states = AttentionBlock(20, 64, 1280, 10, name="down_blocks.2.attentions.1")((hidden_states, text_emb))
        res_stack = self.push_block(hidden_states, res_stack)
        # UNetMidBlock2DCrossAttn
        hidden_states = ResnetBlock(1280, name="mid_block.resnets.0")((hidden_states, time_emb))
        hidden_states = AttentionBlock(20, 64, 1280, 10, name="mid_block.attentions.0")((hidden_states, text_emb))
        hidden_states = ResnetBlock(1280, name="mid_block.resnets.1")((hidden_states, time_emb))
        # CrossAttnUpBlock2D
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(1280, name="up_blocks.0.resnets.0")((hidden_states, time_emb))
        hidden_states = AttentionBlock(20, 64, 1280, 10, name="up_blocks.0.attentions.0")((hidden_states, text_emb))
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(1280, name="up_blocks.0.resnets.1")((hidden_states, time_emb))
        hidden_states = AttentionBlock(20, 64, 1280, 10, name="up_blocks.0.attentions.1")((hidden_states, text_emb))
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(1280, name="up_blocks.0.resnets.2")((hidden_states, time_emb))
        hidden_states = AttentionBlock(20, 64, 1280, 10, name="up_blocks.0.attentions.2")((hidden_states, text_emb))
        hidden_states = UpSampler(1280, name="up_blocks.0.upsamplers.0")(hidden_states)
        # CrossAttnUpBlock2D
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(640, name="up_blocks.1.resnets.0")((hidden_states, time_emb))
        hidden_states = AttentionBlock(10, 64, 640, 2, name="up_blocks.1.attentions.0")((hidden_states, text_emb))
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(640, name="up_blocks.1.resnets.1")((hidden_states, time_emb))
        hidden_states = AttentionBlock(10, 64, 640, 2, name="up_blocks.1.attentions.1")((hidden_states, text_emb))
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(640, name="up_blocks.1.resnets.2")((hidden_states, time_emb))
        hidden_states = AttentionBlock(10, 64, 640, 2, name="up_blocks.1.attentions.2")((hidden_states, text_emb))
        hidden_states = UpSampler(640, name="up_blocks.1.upsamplers.0")(hidden_states)
        # UpBlock2D
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(320, name="up_blocks.2.resnets.0")((hidden_states, time_emb))
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(320, name="up_blocks.2.resnets.1")((hidden_states, time_emb))
        hidden_states, res_stack = self.pop_block(hidden_states, res_stack)
        hidden_states = ResnetBlock(320, name="up_blocks.2.resnets.2")((hidden_states, time_emb))
        hidden_states = GroupNormalization(32, epsilon=1e-05, center=True, scale=True,
                                           name="conv_norm_out")(
            hidden_states)
        hidden_states = tf.keras.layers.Activation("swish")(hidden_states)
        output = tf.keras.layers.Conv2D(4, kernel_size=3, strides=1, name="conv_out")(
            tf.keras.layers.ZeroPadding2D(1)(hidden_states))
        super().__init__([sample, timestep, text_emb, time_ids, text_embeds], output, name=name)
        origin = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors"
        ckpt_mapping = CKPT_MAPPING["diffusion_model"]
        if ckpt_path is not None:
            if os.path.exists(ckpt_path):
                load_weights_from_file(self, ckpt_path, ckpt_mapping=ckpt_mapping, key_mapping=UNET_KEY_MAPPING,
                                       lora_dict=lora_dict)
                return
            else:
                origin = ckpt_path
        model_weights_fpath = tf.keras.utils.get_file(origin=origin)
        if os.path.exists(model_weights_fpath):
            load_weights_from_file(self, model_weights_fpath, ckpt_mapping=ckpt_mapping, key_mapping=UNET_KEY_MAPPING,
                                   lora_dict=lora_dict)
