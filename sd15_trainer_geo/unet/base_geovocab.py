"""
SD15UNetGeoVocab — KSimplex Prior + GeoVocab Conditioning
==========================================================
Extends SD15UNetSimplex:
    - Adds GeoVocabConditioner that processes patch-maker features
    - Gate vectors condition simplex deformation per-layer
    - Cross-attention enriches CLIP tokens before simplex prior

Training: freeze UNet backbone, train geo_prior + geo_conditioner.
Everything else identical to SD15UNetSimplex.

Author: AbstractPhil
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any

from .base_simplex import (
    SD15UNetSimplex,
    SimplexConfig,
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    get_timestep_embedding,
)
from ..config.model_config import UNetConfig
from ..conditioner.geovocab_conditioner import GeoVocabConfig, GeoVocabConditioner


class SD15UNetGeoVocab(SD15UNetSimplex):
    """
    SD1.5 UNet with KSimplex prior + GeoVocab conditioning.

    Two extra paths on top of base simplex:
        1. Gate vectors (64, 17) → per-layer deformation scales
        2. Patch features (64, 256) → cross-attention with CLIP tokens

    When gate_vectors/patch_features are None in forward(), falls back
    to standard SD15UNetSimplex behavior.
    """

    def __init__(
        self,
        unet_config: Optional[UNetConfig] = None,
        simplex_config: Optional[SimplexConfig] = None,
        geovocab_config: Optional[GeoVocabConfig] = None,
    ):
        super().__init__(unet_config, simplex_config)

        self.geovocab_config = geovocab_config or GeoVocabConfig(
            deform_num_layers=self.simplex_config.num_layers,
        )
        self.geo_conditioner = GeoVocabConditioner(self.geovocab_config)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        t_continuous: Optional[torch.Tensor] = None,
        gate_vectors: Optional[torch.Tensor] = None,
        patch_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extended forward with optional geovocab conditioning.

        Additional args:
            gate_vectors:   (B, 64, 17) from patch maker
            patch_features: (B, 64, 256) from patch maker
        """
        cfg = self.config

        # --- Timestep embedding ---
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(sample.shape[0])

        if t_continuous is not None:
            timestep_normalized = t_continuous.to(dtype=torch.float32)
        else:
            timestep_normalized = timestep.to(dtype=torch.float32) / 1000.0

        t_emb = get_timestep_embedding(
            timestep,
            embedding_dim=cfg.block_out_channels[0],
            freq_shift=cfg.freq_shift,
        )
        t_emb = t_emb.to(dtype=sample.dtype)
        temb = self.time_embedding(t_emb)

        # --- GeoVocab conditioning ---
        geo_deform_factors = None
        if gate_vectors is not None and patch_features is not None:
            geo_out = self.geo_conditioner(
                gate_vectors=gate_vectors.to(dtype=torch.float32, device=sample.device),
                patch_features=patch_features.to(dtype=torch.float32, device=sample.device),
                clip_tokens=encoder_hidden_states,
            )
            geo_deform_factors = geo_out["deform_factors"]

            # Cross-attention enrichment of CLIP tokens
            if geo_out["enriched_clip"] is not None:
                encoder_hidden_states = geo_out["enriched_clip"]

        # --- Geometric prior on CLIP conditioning ---
        encoder_hidden_states, prior_info = self.geo_prior(
            encoder_hidden_states,
            timestep_normalized=timestep_normalized,
        )
        self._last_prior_info = prior_info
        self._last_geo_deform = geo_deform_factors

        # --- Standard UNet forward ---
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            if isinstance(down_block, CrossAttnDownBlock2D):
                sample, res_samples = down_block(sample, temb, encoder_hidden_states)
            else:
                sample, res_samples = down_block(sample, temb)
            down_block_res_samples += res_samples

        sample = self.mid_block(sample, temb, encoder_hidden_states)

        for up_block in self.up_blocks:
            n_resnets = len(up_block.resnets)
            res_samples = down_block_res_samples[-n_resnets:]
            down_block_res_samples = down_block_res_samples[:-n_resnets]

            if isinstance(up_block, CrossAttnUpBlock2D):
                sample = up_block(sample, temb, res_samples, encoder_hidden_states)
            else:
                sample = up_block(sample, temb, res_samples)

        sample = self.conv_norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_out(sample)
        return sample

    def freeze_for_geovocab_training(self):
        """Freeze UNet backbone. Train geo_prior + geo_conditioner only."""
        for name, param in self.named_parameters():
            if name.startswith(("geo_prior.", "geo_loss.", "geo_conditioner.")):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_summary(self):
        prior_p = sum(
            p.numel() for n, p in self.named_parameters()
            if n.startswith("geo_prior.") and p.requires_grad
        )
        cond_p = sum(
            p.numel() for n, p in self.named_parameters()
            if n.startswith("geo_conditioner.") and p.requires_grad
        )
        total_p = sum(p.numel() for p in self.parameters())
        train_p = prior_p + cond_p
        print(f"Trainable: {train_p:,} ({100*train_p/total_p:.2f}%)")
        print(f"  geo_prior:       {prior_p:,}")
        print(f"  geo_conditioner: {cond_p:,}")
        print(f"  frozen:          {total_p - train_p:,}")