# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
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

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
import numpy as np

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import Attention, FeedForward
from ..attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, CogVideoXAttnProcessor2_0_wo_text, CogVideoXAttnProcessor2_0_resample, FusedCogVideoXAttnProcessor2_0
from ..embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNorm, CogVideoXLayerNormZero


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        wo_text: bool = False,
        id_pool_resample_learnable: bool = False, 
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        if wo_text:
            self.processor = CogVideoXAttnProcessor2_0_wo_text()
        elif id_pool_resample_learnable:
            self.processor = CogVideoXAttnProcessor2_0_resample()
        else:
            self.processor = CogVideoXAttnProcessor2_0()
        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=self.processor,
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        resample_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )
        if attention_kwargs and "prev_hidden_states" in attention_kwargs:
            prev_encoder_hidden_states, prev_hidden_states = attention_kwargs["prev_hidden_states"][:, :text_seq_length], attention_kwargs["prev_hidden_states"][:, text_seq_length:]
            norm_prev_hidden_states, norm_prev_encoder_hidden_states, prev_gate_msa, prev_enc_gate_msa = self.norm1(
                prev_hidden_states, prev_encoder_hidden_states, temb
            )
            attention_kwargs["prev_hidden_states"] = torch.cat([norm_prev_encoder_hidden_states, norm_prev_hidden_states], dim=1)

        # attention
        if isinstance(self.processor, CogVideoXAttnProcessor2_0_resample):
            attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                resample_mask=resample_mask,
                **attention_kwargs if attention_kwargs else {},
            )
        elif isinstance(self.processor, CogVideoXAttnProcessor2_0):
            attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                **attention_kwargs if attention_kwargs else {},
            )
        else:
            raise ValueError(f"Unsupported processor type: {type(self.processor)}")

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
        # hidden_states = hidden_states[:, :hidden_states.shape[1]//2, :]
        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states

    def forward_wo_text(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:

        # norm & modulate
        norm_hidden_states, gate_msa = self.norm1.forward_wo_text(
            hidden_states, temb
        )

        # attention
        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states

        # norm & modulate
        norm_hidden_states, gate_ff = self.norm2.forward_wo_text(
            hidden_states, temb
        )

        # feed-forward
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output

        return hidden_states


class CogVideoXTransformer3D3B112Model(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 48,
        attention_head_dim: int = 64,
        in_channels: int = 32,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 42,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = True,
        id_pool_resample_learnable: Optional[bool] = False,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=112,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    id_pool_resample_learnable=id_pool_resample_learnable,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    @classmethod
    def from_transformer(
        cls,
        transformer,
        load_weights_from_transformer=True,
    ):
        config = transformer.config
        branch = cls(**config)

        if load_weights_from_transformer:
            conv_in_condition_weight=torch.zeros_like(branch.patch_embed.proj.weight)
            # conv_in_condition_weight: [3072, 112, 2, 2] | transformer.patch_embed.proj.weight: [3072, 32, 2, 2]

            conv_in_condition_weight[:,:config.in_channels//2,...]=transformer.patch_embed.proj.weight[:,:config.in_channels//2,...]    # 16
            conv_in_condition_weight[:,config.in_channels//2:config.in_channels,...]=transformer.patch_embed.proj.weight[:,:config.in_channels//2,...]  # 32
            conv_in_condition_weight[:,config.in_channels:config.in_channels + config.in_channels//2,...]=transformer.patch_embed.proj.weight[:,:config.in_channels//2,...] # 48
            conv_in_condition_weight[:,config.in_channels + config.in_channels//2:config.in_channels*2,...]=transformer.patch_embed.proj.weight[:,:config.in_channels//2,...] # 64
            conv_in_condition_weight[:,config.in_channels*2:config.in_channels*2 + config.in_channels//2,...]=transformer.patch_embed.proj.weight[:,:config.in_channels//2,...] # 80
            conv_in_condition_weight[:,config.in_channels*2 + config.in_channels//2:config.in_channels*3,...]=transformer.patch_embed.proj.weight[:,:config.in_channels//2,...] # 96
            conv_in_condition_weight[:,config.in_channels*3:config.in_channels*3 + config.in_channels//2,...]=transformer.patch_embed.proj.weight[:,:config.in_channels//2,...] # 112

            branch.patch_embed.proj.weight.data=torch.nn.Parameter(conv_in_condition_weight)
            branch.patch_embed.proj.bias.data=transformer.patch_embed.proj.bias

            branch.embedding_dropout.load_state_dict(transformer.embedding_dropout.state_dict())
            branch.time_proj.load_state_dict(transformer.time_proj.state_dict())
            branch.time_embedding.load_state_dict(transformer.time_embedding.state_dict())
            
            branch.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)
        return branch


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        branch_block_samples: Optional[torch.Tensor] = None,
        fg_branch_block_samples: Optional[torch.Tensor] = None,
        branch_block_masks: Optional[torch.Tensor] = None,
        add_first: Optional[bool] = False,
        self_guidance_hidden_states: Optional[torch.Tensor] = None,
        self_guidance_masks: Optional[torch.Tensor] = None,
        return_hidden_states: Optional[bool] = False,
        return_resample_mask: Optional[bool] = False,
        id_pool_resample_learnable: Optional[bool] = False,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        if self_guidance_masks is not None:
            hidden_states, masks = self.patch_embed(encoder_hidden_states, hidden_states, masks=self_guidance_masks)
            # hidden_states: (B, Len_t + Len_v, C); masks: (B, Len_v, C)
            masks = masks.repeat(1, 1, int(hidden_states.shape[-1] / masks.shape[-1]))
        elif branch_block_masks is not None:
            hidden_states, masks = self.patch_embed(encoder_hidden_states, hidden_states, masks=branch_block_masks)         # 走这里, 只是做好准备
            masks = masks.repeat(1, 1, int(hidden_states.shape[-1] / masks.shape[-1]))
        else:
            hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # prepare the attention mask
        if id_pool_resample_learnable or return_resample_mask:

            if masks is None:
                raise ValueError("id_pool_resample needs masks")
            image_patch_len = hidden_states.shape[1] // num_frames
            total_length = encoder_hidden_states.shape[1] + hidden_states.shape[1]
            resample_mask = torch.zeros((batch_size, total_length), 
                                    device=hidden_states.device, 
                                    dtype=torch.bool)
            resample_mask[:, :text_seq_length] = False  
            resample_mask[:, text_seq_length:] = masks[:, :, 0].bool()
            attention_mask = None
        else:
            attention_mask = None
            resample_mask = None

        # 3. Transformer blocks
        hidden_states_list = []
        # fg_control = len(self.transformer_blocks) / len(fg_branch_block_samples)        # 42 / 7 = 6
        # fg_control = int(np.ceil(fg_control))       # 6
        for i, block in enumerate(self.transformer_blocks):

            # fg_states = fg_branch_block_samples[i // fg_control]

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_mask,
                    resample_mask,
                    attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                current_block_kwargs = attention_kwargs.copy() if attention_kwargs else {}
                if attention_kwargs and "prev_hidden_states" in attention_kwargs:
                    layer_states = attention_kwargs["prev_hidden_states"].get(i)
                    if layer_states is not None:
                        current_block_kwargs["prev_hidden_states"] = layer_states
                        current_block_kwargs["prev_clip_weight"] = attention_kwargs["prev_clip_weight"]
                    prev_resample_mask = attention_kwargs.get("prev_resample_mask")
                    if prev_resample_mask is not None:
                        current_block_kwargs["prev_resample_mask"] = prev_resample_mask
            
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask=attention_mask,
                    resample_mask=resample_mask,
                    attention_kwargs=current_block_kwargs,
                )
            if self_guidance_hidden_states is not None:
                hidden_states = torch.where(masks == False, self_guidance_hidden_states[i], hidden_states)

            if branch_block_samples is not None:
                if not add_first:
                    interval_control = len(self.transformer_blocks) / len(branch_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    if branch_block_masks is None:
                        hidden_states = hidden_states + branch_block_samples[i // interval_control]
                    else:
                        hidden_states = torch.where(masks == False, hidden_states + branch_block_samples[i // interval_control], hidden_states)     # 原文走这里, 前景就完全交给主网络，背景受branch影响
                else:
                    if i < len(branch_block_samples):
                        if branch_block_masks is None:
                            hidden_states = hidden_states + branch_block_samples[i]
                        else:
                            hidden_states = torch.where(masks == False, hidden_states + branch_block_samples[i], hidden_states)
            
            if return_hidden_states:
                hidden_states_list.append(torch.cat([encoder_hidden_states, hidden_states], dim=1))
        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size      # 2
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        # (b, f, -1, h, p, w, p) --> (b, f, -1, h*p, w*p)
        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            if return_hidden_states:
                if return_resample_mask:
                    return (output, hidden_states_list, resample_mask)
                else:
                    return (output, hidden_states_list)
            else:
                return (output,)
        return Transformer2DModelOutput(sample=output)
