"""
SD1.5 VAE - Config-Driven Pure PyTorch Implementation
=====================================================
Builds architecture dynamically from VAEConfig (diffusers config.json).
No diffusers dependency. State_dict keys match exactly.

Usage:
    from config.model_config import load_vae_config, VAEConfig

    # From HuggingFace repo
    config = load_vae_config("sd-legacy/stable-diffusion-v1-5")
    vae = SD15VAE(config)

    # Default SD1.5
    vae = SD15VAE()

Author: AbstractPhil
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from ..config.model_config import VAEConfig


# =============================================================================
# VAE ResNet Block
# =============================================================================

class VAEResnetBlock(nn.Module):
    """ResNet block for VAE. No time embedding."""
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        norm_groups: int = 32,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(norm_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual


# =============================================================================
# VAE Attention
# =============================================================================

class VAEAttention(nn.Module):
    """Single-head spatial self-attention with GroupNorm."""
    def __init__(self, channels: int, norm_groups: int = 32, norm_eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(norm_groups, channels, eps=norm_eps, affine=True)

        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, C, H, W = hidden_states.shape
        residual = hidden_states

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(B, H * W, C)

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        scale = C ** -0.5
        attn = torch.bmm(q, k.transpose(-1, -2)) * scale
        attn = F.softmax(attn, dim=-1)
        hidden_states = torch.bmm(attn, v)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = hidden_states.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return hidden_states + residual


# =============================================================================
# Downsample / Upsample
# =============================================================================

class VAEDownsample(nn.Module):
    """Asymmetric padding + stride-2 conv."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(hidden_states)


class VAEUpsample(nn.Module):
    """Nearest upsample + conv."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_states)


# =============================================================================
# Encoder / Decoder Blocks
# =============================================================================

class EncoderDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        norm_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            self.resnets.append(VAEResnetBlock(ch_in, out_channels, norm_groups, norm_eps))

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([VAEDownsample(out_channels)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for ds in self.downsamplers:
                hidden_states = ds(hidden_states)
        return hidden_states


class DecoderUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        norm_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            self.resnets.append(VAEResnetBlock(ch_in, out_channels, norm_groups, norm_eps))

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([VAEUpsample(out_channels)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.upsamplers is not None:
            for us in self.upsamplers:
                hidden_states = us(hidden_states)
        return hidden_states


# =============================================================================
# Mid Block
# =============================================================================

class VAEMidBlock(nn.Module):
    def __init__(
        self,
        channels: int = 512,
        norm_groups: int = 32,
        norm_eps: float = 1e-6,
        add_attention: bool = True,
    ):
        super().__init__()
        self.add_attention = add_attention
        self.resnets = nn.ModuleList([
            VAEResnetBlock(channels, channels, norm_groups, norm_eps),
            VAEResnetBlock(channels, channels, norm_groups, norm_eps),
        ])
        self.attentions = nn.ModuleList()
        if add_attention:
            self.attentions.append(VAEAttention(channels, norm_groups, norm_eps))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)
        if self.add_attention:
            hidden_states = self.attentions[0](hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states


# =============================================================================
# Encoder
# =============================================================================

class Encoder(nn.Module):
    """VAE Encoder: image -> latent params."""
    def __init__(self, config: VAEConfig):
        super().__init__()
        boc = config.block_out_channels

        self.conv_in = nn.Conv2d(config.in_channels, boc[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        in_ch = boc[0]
        for i, out_ch in enumerate(boc):
            add_ds = (i < len(boc) - 1)
            self.down_blocks.append(
                EncoderDownBlock(
                    in_ch, out_ch, config.layers_per_block,
                    config.norm_num_groups, config.norm_eps, add_ds,
                )
            )
            in_ch = out_ch

        self.mid_block = VAEMidBlock(
            boc[-1], config.norm_num_groups, config.norm_eps,
            add_attention=config.mid_block_add_attention,
        )

        self.conv_norm_out = nn.GroupNorm(
            config.norm_num_groups, boc[-1], eps=config.norm_eps, affine=True,
        )
        # Output: 2 * latent_channels (mean + logvar)
        self.conv_out = nn.Conv2d(boc[-1], config.latent_channels * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(x)
        for block in self.down_blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.mid_block(hidden_states)
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


# =============================================================================
# Decoder
# =============================================================================

class Decoder(nn.Module):
    """VAE Decoder: latent -> image."""
    def __init__(self, config: VAEConfig):
        super().__init__()
        boc = config.block_out_channels
        reversed_boc = list(reversed(boc))

        self.conv_in = nn.Conv2d(config.latent_channels, reversed_boc[0], 3, padding=1)

        self.mid_block = VAEMidBlock(
            reversed_boc[0], config.norm_num_groups, config.norm_eps,
            add_attention=config.mid_block_add_attention,
        )

        self.up_blocks = nn.ModuleList()
        in_ch = reversed_boc[0]
        for i, out_ch in enumerate(reversed_boc):
            add_us = (i < len(reversed_boc) - 1)
            self.up_blocks.append(
                DecoderUpBlock(
                    in_ch, out_ch,
                    num_layers=config.layers_per_block + 1,
                    norm_groups=config.norm_num_groups,
                    norm_eps=config.norm_eps,
                    add_upsample=add_us,
                )
            )
            in_ch = out_ch

        self.conv_norm_out = nn.GroupNorm(
            config.norm_num_groups, reversed_boc[-1], eps=config.norm_eps, affine=True,
        )
        self.conv_out = nn.Conv2d(reversed_boc[-1], config.out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(z)
        hidden_states = self.mid_block(hidden_states)
        for block in self.up_blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


# =============================================================================
# Full VAE
# =============================================================================

class SD15VAE(nn.Module):
    """
    Config-driven SD1.5 Variational Autoencoder.

    Encode: image (B, 3, H, W) -> latent (B, latent_ch, H/8, W/8)
    Decode: latent (B, latent_ch, H/8, W/8) -> image (B, 3, H, W)

    Keys: encoder.*, decoder.*, quant_conv.*, post_quant_conv.*
    """

    def __init__(self, config: Optional[VAEConfig] = None):
        super().__init__()
        self.config = config or VAEConfig()
        cfg = self.config

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.quant_conv = nn.Conv2d(cfg.latent_channels * 2, cfg.latent_channels * 2, 1)
        self.post_quant_conv = nn.Conv2d(cfg.latent_channels, cfg.latent_channels, 1)

    @property
    def scaling_factor(self) -> float:
        return self.config.scaling_factor

    def encode(
        self,
        x: torch.Tensor,
        sample: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Encode image to latent (NOT scaled by scaling_factor).
        Args:
            x: (B, 3, H, W) image tensor, expected [-1, 1] range
            sample: if True, reparameterize; if False, return mean
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = h.chunk(2, dim=1)

        if sample:
            logvar = torch.clamp(logvar, -30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            if generator is not None:
                z = mean + std * torch.randn(mean.shape, dtype=mean.dtype, device="cpu", generator=generator).to(mean.device)
            else:
                z = mean + std * torch.randn_like(mean)
            return z
        else:
            return mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image (expects unscaled latent)."""
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def encode_scaled(
        self,
        x: torch.Tensor,
        sample: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Encode and apply scaling factor."""
        return self.encode(x, sample, generator) * self.scaling_factor

    def decode_scaled(self, z: torch.Tensor) -> torch.Tensor:
        """Remove scaling factor and decode."""
        return self.decode(z / self.scaling_factor)


# =============================================================================
# Weight Loading
# =============================================================================

def load_sd15_vae(
    repo_id: str = "sd-legacy/stable-diffusion-v1-5",
    subfolder: str = "vae",
    filename: str = "diffusion_pytorch_model.safetensors",
    config: Optional[VAEConfig] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SD15VAE:
    """
    Load VAE weights from HuggingFace.
    If config is None, loads config.json from the same repo.
    """
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from ..config.model_config import load_vae_config

    if config is None:
        config = load_vae_config(repo_id)

    vae = SD15VAE(config)

    path = hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=filename)
    state_dict = load_file(path, device=device)

    missing, unexpected = vae.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Missing keys ({len(missing)}):")
        for k in missing:
            print(f"  {k}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}):")
        for k in unexpected:
            print(f"  {k}")
    if not missing and not unexpected:
        print("All VAE weights loaded successfully.")

    return vae.to(dtype=dtype, device=device)


def load_vae_from_safetensors(
    vae: SD15VAE,
    path: str,
    device: str = "cpu",
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load VAE weights from a local safetensors file."""
    from safetensors.torch import load_file
    state_dict = load_file(path, device=device)
    return vae.load_state_dict(state_dict, strict=strict)


# =============================================================================
# Verification
# =============================================================================

def verify_architecture(config: Optional[VAEConfig] = None):
    config = config or VAEConfig()
    vae = SD15VAE(config)
    total = sum(p.numel() for p in vae.parameters())
    enc_params = sum(p.numel() for p in vae.encoder.parameters())
    dec_params = sum(p.numel() for p in vae.decoder.parameters())

    print(f"SD15VAE Config-Driven Architecture")
    print(f"==================================")
    print(f"  Total params:      {total:,}")
    print(f"  Encoder:           {enc_params:,} ({100*enc_params/total:.1f}%)")
    print(f"  Decoder:           {dec_params:,} ({100*dec_params/total:.1f}%)")
    print(f"  Block channels:    {config.block_out_channels}")
    print(f"  Latent channels:   {config.latent_channels}")
    print(f"  Layers/block:      {config.layers_per_block}")
    print(f"  Scale factor:      {config.scaling_factor}")
    print(f"  Mid attention:     {config.mid_block_add_attention}")

    return vae


if __name__ == "__main__":
    verify_architecture()