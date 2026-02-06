"""
SD1.5 UNet - Config-Driven Pure PyTorch Implementation
=======================================================
Builds architecture dynamically from UNetConfig (diffusers config.json).
No diffusers dependency. State_dict keys match diffusers exactly.

Usage:
    from config.model_config import load_unet_config, UNetConfig

    # From HuggingFace repo
    config = load_unet_config("sd-legacy/stable-diffusion-v1-5")
    unet = SD15UNet(config)

    # Default SD1.5
    unet = SD15UNet()

    # Custom config
    config = UNetConfig(block_out_channels=[256, 512, 1024, 1024])
    unet = SD15UNet(config)

Author: AbstractPhil
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

from ..config.model_config import UNetConfig


# =============================================================================
# Core Components
# =============================================================================

class GEGLU(nn.Module):
    """Gated GELU activation. Projects to 2x dim, splits, gates."""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """FFN with GEGLU activation. Keys: net.0.proj, net.2"""
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head attention with separate Q/K/V projections.
    Keys: to_q, to_k, to_v, to_out.0
    """
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = query_dim // heads
        self.scale = self.head_dim ** -0.5

        kv_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, query_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, query_dim, bias=False)
        self.to_out = nn.ModuleList([
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        kv_input = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        q = q.view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, L, -1)

        attn = self.to_out[0](attn)
        attn = self.to_out[1](attn)
        return attn


class BasicTransformerBlock(nn.Module):
    """
    Self-Attn -> Cross-Attn -> FFN with pre-norm residual connections.
    Keys: norm1, attn1, norm2, attn2, norm3, ff
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cross_attention_dim: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(query_dim=dim, heads=num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_heads,
            dropout=dropout,
        )

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), encoder_hidden_states) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class Transformer2DModel(nn.Module):
    """
    Spatial transformer with configurable number of transformer blocks.
    Keys: norm, proj_in, transformer_blocks.{i}.*, proj_out
    """
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        cross_attention_dim: int = 768,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection

        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)

        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, in_channels)
        else:
            self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=in_channels,
                num_heads=num_heads,
                cross_attention_dim=cross_attention_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, in_channels)
        else:
            self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        if self.use_linear_projection:
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(B, H * W, C)
            hidden_states = self.proj_in(hidden_states)
        else:
            hidden_states = self.proj_in(hidden_states)
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(B, H * W, C)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)

        if self.use_linear_projection:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            hidden_states = hidden_states.reshape(B, H, W, C).permute(0, 3, 1, 2)
            hidden_states = self.proj_out(hidden_states)

        return hidden_states + residual


# =============================================================================
# ResNet Block
# =============================================================================

class ResnetBlock2D(nn.Module):
    """
    ResNet block with time embedding injection.
    Keys: norm1, conv1, time_emb_proj, norm2, conv2, [conv_shortcut]
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int = 1280,
        norm_groups: int = 32,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(norm_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(norm_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        temb_out = F.silu(temb)
        temb_out = self.time_emb_proj(temb_out)
        hidden_states = hidden_states + temb_out[:, :, None, None]

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual


# =============================================================================
# Downsample / Upsample
# =============================================================================

class Downsample2D(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, 3, stride=2, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_states)


# =============================================================================
# Down Blocks
# =============================================================================

class DownBlock2D(nn.Module):
    """Plain down block (no attention)."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        temb_channels: int = 1280,
        norm_groups: int = 32,
        norm_eps: float = 1e-5,
        add_downsample: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            res_in = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(res_in, out_channels, temb_channels, norm_groups, norm_eps)
            )

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels)])

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for ds in self.downsamplers:
                hidden_states = ds(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class CrossAttnDownBlock2D(nn.Module):
    """Down block with cross-attention."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        temb_channels: int = 1280,
        num_heads: int = 8,
        cross_attention_dim: int = 768,
        transformer_layers: int = 1,
        norm_groups: int = 32,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        use_linear_projection: bool = False,
        add_downsample: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i in range(num_layers):
            res_in = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(res_in, out_channels, temb_channels, norm_groups, norm_eps)
            )
            self.attentions.append(
                Transformer2DModel(
                    out_channels, num_heads, cross_attention_dim,
                    num_layers=transformer_layers,
                    dropout=dropout,
                    use_linear_projection=use_linear_projection,
                )
            )

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for ds in self.downsamplers:
                hidden_states = ds(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


# =============================================================================
# Mid Block
# =============================================================================

class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int = 1280,
        temb_channels: int = 1280,
        num_heads: int = 8,
        cross_attention_dim: int = 768,
        transformer_layers: int = 1,
        norm_groups: int = 32,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels, in_channels, temb_channels, norm_groups, norm_eps),
            ResnetBlock2D(in_channels, in_channels, temb_channels, norm_groups, norm_eps),
        ])
        self.attentions = nn.ModuleList([
            Transformer2DModel(
                in_channels, num_heads, cross_attention_dim,
                num_layers=transformer_layers,
                dropout=dropout,
                use_linear_projection=use_linear_projection,
            ),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states


# =============================================================================
# Up Blocks
# =============================================================================

class UpBlock2D(nn.Module):
    """Plain up block (no attention)."""
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        num_layers: int = 3,
        temb_channels: int = 1280,
        norm_groups: int = 32,
        norm_eps: float = 1e-5,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()

        for i in range(num_layers):
            res_skip = in_channels if (i == num_layers - 1) else out_channels
            res_in = prev_output_channel if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(res_in + res_skip, out_channels, temb_channels, norm_groups, norm_eps)
            )

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        for resnet in self.resnets:
            res_hidden = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for us in self.upsamplers:
                hidden_states = us(hidden_states)

        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    """Up block with cross-attention."""
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        num_layers: int = 3,
        temb_channels: int = 1280,
        num_heads: int = 8,
        cross_attention_dim: int = 768,
        transformer_layers: int = 1,
        norm_groups: int = 32,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        use_linear_projection: bool = False,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i in range(num_layers):
            res_skip = in_channels if (i == num_layers - 1) else out_channels
            res_in = prev_output_channel if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(res_in + res_skip, out_channels, temb_channels, norm_groups, norm_eps)
            )
            self.attentions.append(
                Transformer2DModel(
                    out_channels, num_heads, cross_attention_dim,
                    num_layers=transformer_layers,
                    dropout=dropout,
                    use_linear_projection=use_linear_projection,
                )
            )

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)

        if self.upsamplers is not None:
            for us in self.upsamplers:
                hidden_states = us(hidden_states)

        return hidden_states


# =============================================================================
# Timestep Embedding
# =============================================================================

class TimestepEmbedding(nn.Module):
    def __init__(self, channel: int = 320, time_embed_dim: int = 1280):
        super().__init__()
        self.linear_1 = nn.Linear(channel, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = F.silu(sample)
        sample = self.linear_2(sample)
        return sample


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int = 320,
    max_period: int = 10000,
    freq_shift: int = 0,
) -> torch.Tensor:
    """Sinusoidal positional embedding for timesteps."""
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half_dim, dtype=torch.float32, device=timesteps.device
    ) / (half_dim - freq_shift)
    emb = timesteps[:, None].float() * exponent[None, :].exp()
    emb = torch.cat([emb.cos(), emb.sin()], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# =============================================================================
# Block Registry
# =============================================================================

DOWN_BLOCK_TYPES = {
    "DownBlock2D": DownBlock2D,
    "CrossAttnDownBlock2D": CrossAttnDownBlock2D,
}

UP_BLOCK_TYPES = {
    "UpBlock2D": UpBlock2D,
    "CrossAttnUpBlock2D": CrossAttnUpBlock2D,
}


# =============================================================================
# Main UNet
# =============================================================================

class SD15UNet(nn.Module):
    """
    Config-driven SD1.5 UNet.

    Input:
        sample:                 (B, 4, H, W) noisy latents
        timestep:               (B,) or scalar
        encoder_hidden_states:  (B, seq, cross_attn_dim) text embeddings

    Output:
        (B, 4, H, W) predicted noise / velocity
    """

    def __init__(self, config: Optional[UNetConfig] = None):
        super().__init__()
        self.config = config or UNetConfig()
        cfg = self.config

        boc = cfg.block_out_channels
        temb_ch = cfg.temb_channels

        # --- Input ---
        self.conv_in = nn.Conv2d(cfg.in_channels, boc[0], 3, padding=1)

        # --- Time embedding ---
        self.time_embedding = TimestepEmbedding(boc[0], temb_ch)

        # --- Down blocks ---
        self.down_blocks = nn.ModuleList()
        output_channel = boc[0]
        for i, down_type in enumerate(cfg.down_block_types):
            input_channel = output_channel
            output_channel = boc[i]
            is_last = (i == len(boc) - 1)
            num_heads = cfg.get_attention_heads(i)
            transformer_layers = cfg.get_transformer_layers(i)

            block_cls = DOWN_BLOCK_TYPES[down_type]

            if down_type == "CrossAttnDownBlock2D":
                self.down_blocks.append(block_cls(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=cfg.layers_per_block,
                    temb_channels=temb_ch,
                    num_heads=num_heads,
                    cross_attention_dim=cfg.cross_attention_dim,
                    transformer_layers=transformer_layers,
                    norm_groups=cfg.norm_num_groups,
                    norm_eps=cfg.norm_eps,
                    dropout=cfg.dropout,
                    use_linear_projection=cfg.use_linear_projection,
                    add_downsample=not is_last,
                ))
            else:
                self.down_blocks.append(block_cls(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=cfg.layers_per_block,
                    temb_channels=temb_ch,
                    norm_groups=cfg.norm_num_groups,
                    norm_eps=cfg.norm_eps,
                    add_downsample=not is_last,
                ))

        # --- Mid block ---
        mid_num_heads = cfg.get_attention_heads(len(boc) - 1)
        mid_transformer_layers = cfg.get_transformer_layers(len(boc) - 1)
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=boc[-1],
            temb_channels=temb_ch,
            num_heads=mid_num_heads,
            cross_attention_dim=cfg.cross_attention_dim,
            transformer_layers=mid_transformer_layers,
            norm_groups=cfg.norm_num_groups,
            norm_eps=cfg.norm_eps,
            dropout=cfg.dropout,
            use_linear_projection=cfg.use_linear_projection,
        )

        # --- Up blocks ---
        self.up_blocks = nn.ModuleList()
        reversed_boc = list(reversed(boc))

        for i, up_type in enumerate(cfg.up_block_types):
            is_last = (i == len(reversed_boc) - 1)
            prev_output_channel = reversed_boc[max(i - 1, 0)]
            output_channel = reversed_boc[i]

            # Determine the 'in_channels' for skip connection sizing
            if i < len(reversed_boc) - 1:
                in_channel = reversed_boc[i + 1]
            else:
                in_channel = reversed_boc[-1]

            # Map to original block index for attention config
            orig_idx = len(boc) - 1 - i
            num_heads = cfg.get_attention_heads(max(orig_idx, 0))
            transformer_layers = cfg.get_transformer_layers(max(orig_idx, 0))

            block_cls = UP_BLOCK_TYPES[up_type]

            if up_type == "CrossAttnUpBlock2D":
                self.up_blocks.append(block_cls(
                    in_channels=in_channel,
                    prev_output_channel=prev_output_channel,
                    out_channels=output_channel,
                    num_layers=cfg.layers_per_block + 1,
                    temb_channels=temb_ch,
                    num_heads=num_heads,
                    cross_attention_dim=cfg.cross_attention_dim,
                    transformer_layers=transformer_layers,
                    norm_groups=cfg.norm_num_groups,
                    norm_eps=cfg.norm_eps,
                    dropout=cfg.dropout,
                    use_linear_projection=cfg.use_linear_projection,
                    add_upsample=not is_last,
                ))
            else:
                self.up_blocks.append(block_cls(
                    in_channels=in_channel,
                    prev_output_channel=prev_output_channel,
                    out_channels=output_channel,
                    num_layers=cfg.layers_per_block + 1,
                    temb_channels=temb_ch,
                    norm_groups=cfg.norm_num_groups,
                    norm_eps=cfg.norm_eps,
                    add_upsample=not is_last,
                ))

        # --- Output ---
        self.conv_norm_out = nn.GroupNorm(cfg.norm_num_groups, boc[0], eps=cfg.norm_eps)
        self.conv_out = nn.Conv2d(boc[0], cfg.out_channels, 3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config

        # --- Timestep embedding ---
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(sample.shape[0])

        t_emb = get_timestep_embedding(
            timestep,
            embedding_dim=cfg.block_out_channels[0],
            freq_shift=cfg.freq_shift,
        )
        t_emb = t_emb.to(dtype=sample.dtype)
        temb = self.time_embedding(t_emb)

        # --- Input conv ---
        sample = self.conv_in(sample)

        # --- Down path ---
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            if isinstance(down_block, CrossAttnDownBlock2D):
                sample, res_samples = down_block(sample, temb, encoder_hidden_states)
            else:
                sample, res_samples = down_block(sample, temb)
            down_block_res_samples += res_samples

        # --- Mid ---
        sample = self.mid_block(sample, temb, encoder_hidden_states)

        # --- Up path ---
        for up_block in self.up_blocks:
            n_resnets = len(up_block.resnets)
            res_samples = down_block_res_samples[-n_resnets:]
            down_block_res_samples = down_block_res_samples[:-n_resnets]

            if isinstance(up_block, CrossAttnUpBlock2D):
                sample = up_block(sample, temb, res_samples, encoder_hidden_states)
            else:
                sample = up_block(sample, temb, res_samples)

        # --- Output ---
        sample = self.conv_norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_out(sample)
        return sample


# =============================================================================
# Weight Loading
# =============================================================================

def load_sd15_unet(
    repo_id: str = "sd-legacy/stable-diffusion-v1-5",
    subfolder: str = "unet",
    filename: str = "diffusion_pytorch_model.safetensors",
    config: Optional[UNetConfig] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SD15UNet:
    """
    Load UNet weights from HuggingFace.
    If config is None, loads config.json from the same repo.
    """
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from ..config.model_config import load_unet_config

    if config is None:
        config = load_unet_config(repo_id)

    unet = SD15UNet(config)

    path = hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=filename)
    state_dict = load_file(path, device=device)

    missing, unexpected = unet.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Missing keys ({len(missing)}):")
        for k in missing:
            print(f"  {k}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}):")
        for k in unexpected:
            print(f"  {k}")
    if not missing and not unexpected:
        print("All UNet weights loaded successfully.")

    return unet.to(dtype=dtype, device=device)


def load_unet_from_safetensors(
    unet: SD15UNet,
    path: str,
    device: str = "cpu",
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load UNet weights from a local safetensors file."""
    from safetensors.torch import load_file
    state_dict = load_file(path, device=device)
    return unet.load_state_dict(state_dict, strict=strict)


# =============================================================================
# Verification
# =============================================================================

def verify_architecture(config: Optional[UNetConfig] = None):
    """Print architecture summary and verify structure."""
    config = config or UNetConfig()
    unet = SD15UNet(config)
    total = sum(p.numel() for p in unet.parameters())

    cross_attn = 0
    self_attn = 0
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            params = sum(p.numel() for p in module.parameters())
            if '.attn2.' in name or name.endswith('.attn2'):
                cross_attn += params
            elif '.attn1.' in name or name.endswith('.attn1'):
                self_attn += params

    print(f"SD15UNet Config-Driven Architecture")
    print(f"===================================")
    print(f"  Total params:      {total:,}")
    print(f"  Cross-attn params: {cross_attn:,} ({100*cross_attn/total:.1f}%)")
    print(f"  Self-attn params:  {self_attn:,} ({100*self_attn/total:.1f}%)")
    print(f"  Other params:      {total-cross_attn-self_attn:,}")
    print(f"\n  Block channels:    {config.block_out_channels}")
    print(f"  Down types:        {config.down_block_types}")
    print(f"  Up types:          {config.up_block_types}")
    print(f"  Cross-attn dim:    {config.cross_attention_dim}")
    print(f"  Layers/block:      {config.layers_per_block}")

    if total == 859_520_964:
        print(f"\n  ✓ Param count matches SD1.5 reference (859,520,964)")

    return unet


if __name__ == "__main__":
    verify_architecture()