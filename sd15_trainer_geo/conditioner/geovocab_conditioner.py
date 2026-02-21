"""
GeoVocab Conditioner — Geometric Vocabulary → Simplex Prior Bridge
====================================================================
Bridges geovocab-patch-maker features into the KSimplex cross-attention
prior of sd15-flow-trainer.

Two injection paths:
    1. Deformation conditioning: gate vectors → per-layer deformation scale
       (augments/replaces timestep-only conditioning)
    2. Geometric cross-attention: patch features → cross-attend with CLIP tokens
       (adds structural bias to simplex attention)

Usage in training:
    conditioner = GeoVocabConditioner(simplex_config)

    # Pre-extract: text → VAE → patches → patch_maker features
    gate_vectors, patch_features = extract_geo_features(prompts)

    # Inject into geo_prior forward
    geo_conditioning = conditioner(gate_vectors, patch_features)
    # → feeds into KSimplexCrossAttentionPrior

Integration with SD15UNetSimplex:
    See SD15UNetGeoVocab below — extends SD15UNetSimplex with
    conditioner wired into the forward pass.

Author: AbstractPhil
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any


# =============================================================================
# GeoVocab Conditioner Config
# =============================================================================

@dataclass
class GeoVocabConfig:
    """Configuration for geometric vocabulary conditioning."""

    # Patch maker dimensions (from geovocab-patch-maker config)
    gate_dim: int = 17          # Total gate vector dim (11 local + 6 structural)
    patch_feat_dim: int = 256   # Learned patch feature dim (embed_dim of patch maker)
    num_patches: int = 64       # Number of patches (MACRO_N)

    # VAE source (which text encoder / VAE feeds the patch maker)
    clip_vae_dim: int = 768     # Input dim of ClipVAE (768 for CLIP ViT-L/14)
    clip_vae_bottleneck: int = 256

    # Deformation conditioning path
    deform_hidden: int = 128
    deform_num_layers: int = 4  # Must match SimplexConfig.num_layers

    # Cross-attention path (patch features → CLIP token space)
    cross_attn_enabled: bool = True
    cross_attn_heads: int = 8
    cross_attn_dim: int = 768   # Must match SimplexConfig.feat_dim (CLIP hidden)

    # Blend control
    geo_blend_mode: str = "learnable"  # "fixed", "learnable", "gate_modulated"
    geo_blend_init: float = 0.0        # sigmoid(0) = 0.5


# =============================================================================
# Gate-Conditioned Deformation
# =============================================================================

class GateDeformationConditioner(nn.Module):
    """
    Maps gate vectors → per-layer deformation scales for simplex attention.

    Gate vectors encode what geometric structures the text describes
    (dimensionality, curvature, topology). This tells the simplex prior
    HOW MUCH to deform its template at each layer:
        - High curvature / 3D → more deformation (explore geometry)
        - Flat / 1D / rigid → less deformation (preserve structure)

    Replaces/augments the timestep-only deformation schedule.
    """

    def __init__(self, config: GeoVocabConfig):
        super().__init__()
        self.config = config

        # Pool 64 patches → single vector, then project to per-layer scales
        self.gate_pool = nn.Sequential(
            nn.Linear(config.gate_dim, config.deform_hidden),
            nn.GELU(),
            nn.Linear(config.deform_hidden, config.deform_hidden),
        )

        # Attention-weighted pooling over patches (learn which patches matter)
        self.patch_attn = nn.Sequential(
            nn.Linear(config.gate_dim, 1),
        )

        # Project pooled → per-layer deformation factors
        self.to_deform_scales = nn.Sequential(
            nn.Linear(config.deform_hidden, config.deform_hidden),
            nn.GELU(),
            nn.Linear(config.deform_hidden, config.deform_num_layers),
            nn.Sigmoid(),  # Output in [0, 1], will be mapped to deformation range
        )

    def forward(
        self,
        gate_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            gate_vectors: (B, 64, 17) from patch maker

        Returns:
            deform_factors: (B, num_layers) in [0, 1]
        """
        B = gate_vectors.shape[0]

        # Attention-weighted pool over patches
        attn_logits = self.patch_attn(gate_vectors).squeeze(-1)  # (B, 64)
        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # (B, 64, 1)

        gate_projected = self.gate_pool(gate_vectors)  # (B, 64, hidden)
        pooled = (gate_projected * attn_weights).sum(dim=1)  # (B, hidden)

        return self.to_deform_scales(pooled)  # (B, num_layers)


# =============================================================================
# Geometric Cross-Attention
# =============================================================================

class GeoPatchCrossAttention(nn.Module):
    """
    Cross-attention: CLIP tokens attend to geometric patch features.

    CLIP tokens (B, 77, 768) as queries, patch features (B, 64, 256) as keys/values.
    Projects patch features to CLIP dim, computes cross-attention,
    adds geometric structural information to CLIP conditioning.

    This lets the simplex prior know WHERE geometric structures are
    in addition to WHAT they are (from gate conditioning).
    """

    def __init__(self, config: GeoVocabConfig):
        super().__init__()
        self.config = config
        dim = config.cross_attn_dim
        n_heads = config.cross_attn_heads
        head_dim = dim // n_heads

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(config.patch_feat_dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(config.patch_feat_dim, dim)
        self.v_proj = nn.Linear(config.patch_feat_dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = math.sqrt(head_dim)

        # Learnable blend
        if config.geo_blend_mode == "learnable":
            self.blend_logit = nn.Parameter(torch.tensor(config.geo_blend_init))
        elif config.geo_blend_mode == "gate_modulated":
            self.blend_from_gates = nn.Sequential(
                nn.Linear(config.gate_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
        else:
            self.register_buffer("blend_logit", torch.tensor(config.geo_blend_init))

    def forward(
        self,
        clip_tokens: torch.Tensor,
        patch_features: torch.Tensor,
        gate_vectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            clip_tokens:    (B, 77, 768) CLIP hidden states
            patch_features: (B, 64, 256) from patch maker
            gate_vectors:   (B, 64, 17)  optional, for gate-modulated blend

        Returns:
            enriched: (B, 77, 768) CLIP tokens + geometric cross-attention
        """
        B, T_q, D = clip_tokens.shape

        q = self.q_proj(self.norm_q(clip_tokens))
        k = self.k_proj(self.norm_kv(patch_features))
        v = self.v_proj(self.norm_kv(patch_features))

        # Reshape for multi-head
        q = q.view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T_q, D)
        out = self.out_proj(out)

        # Blend
        if self.config.geo_blend_mode == "gate_modulated" and gate_vectors is not None:
            gate_pooled = gate_vectors.mean(dim=1)  # (B, 17)
            blend = torch.sigmoid(self.blend_from_gates(gate_pooled))  # (B, 1)
            blend = blend.unsqueeze(1)  # (B, 1, 1)
        elif hasattr(self, "blend_logit"):
            blend = torch.sigmoid(self.blend_logit)
        else:
            blend = 0.5

        return clip_tokens + blend * out


# =============================================================================
# Combined GeoVocab Conditioner
# =============================================================================

class GeoVocabConditioner(nn.Module):
    """
    Complete geometric vocabulary conditioner.

    Takes gate_vectors + patch_features from geovocab-patch-maker,
    produces:
        1. deform_factors: per-layer deformation scales for simplex prior
        2. enriched CLIP tokens: with geometric cross-attention applied

    Designed to plug into KSimplexCrossAttentionPrior.forward()
    """

    def __init__(self, config: GeoVocabConfig):
        super().__init__()
        self.config = config

        # Path 1: Gates → deformation conditioning
        self.deform_conditioner = GateDeformationConditioner(config)

        # Path 2: Patch features → CLIP cross-attention
        if config.cross_attn_enabled:
            self.cross_attn = GeoPatchCrossAttention(config)
        else:
            self.cross_attn = None

    def forward(
        self,
        gate_vectors: torch.Tensor,
        patch_features: torch.Tensor,
        clip_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            gate_vectors:   (B, 64, 17)  geometric property vectors
            patch_features: (B, 64, 256) learned patch representations
            clip_tokens:    (B, 77, 768) CLIP hidden states (for cross-attn path)

        Returns:
            dict with:
                deform_factors: (B, num_layers) deformation scales [0, 1]
                enriched_clip:  (B, 77, 768) or None if no clip_tokens
        """
        deform_factors = self.deform_conditioner(gate_vectors)

        enriched_clip = None
        if self.cross_attn is not None and clip_tokens is not None:
            enriched_clip = self.cross_attn(clip_tokens, patch_features, gate_vectors)

        return {
            "deform_factors": deform_factors,
            "enriched_clip": enriched_clip,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Feature Extraction Pipeline (patches → geo features)
# =============================================================================

class GeoFeatureExtractor(nn.Module):
    """
    Patches → geometric features via PatchMaker.

    CLIP enc_hs → mean-pool → ClipVAE → (8,16,16) patches
        → patch_maker → gate_vectors (B, 64, 17) + patch_features (B, 64, 256)

    ClipVAE is handled upstream (pre-extraction or pipeline). This class
    only wraps PatchMaker for the patches → gates+features step.

    All components frozen during training. Only the conditioner trains.
    """

    def __init__(
        self,
        patch_maker_repo: str = "AbstractPhil/geovocab-patch-maker",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self._loaded = False
        self._patch_maker_repo = patch_maker_repo

    def _lazy_load(self):
        """Load models on first use (avoids loading during init)."""
        if self._loaded:
            return

        # Try local import first (when geometric_model.py is in conditioner/),
        # then fall back to bare import (when it's on sys.path from HF download)
        try:
            from .geometric_model import load_from_hub as load_patch_maker
        except ImportError:
            from geometric_model import load_from_hub as load_patch_maker

        # Load patch maker
        self.patch_maker, self.patch_config = load_patch_maker(
            repo_id=self._patch_maker_repo,
            device=self.device,
        )
        self.patch_maker.eval()
        for p in self.patch_maker.parameters():
            p.requires_grad = False

        self._loaded = True

    @torch.no_grad()
    def extract_from_patches(
        self,
        patches: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract geometric features from pre-computed (8, 16, 16) patches.

        Args:
            patches: (B, 8, 16, 16) latent patches

        Returns:
            gate_vectors:   (B, 64, 17)
            patch_features: (B, 64, embed_dim)
        """
        self._lazy_load()
        try:
            from .geometric_model import extract_features
        except ImportError:
            from geometric_model import extract_features
        return extract_features(self.patch_maker, patches)

    @torch.no_grad()
    def extract_from_latents(
        self,
        vae_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract from pre-encoded VAE latents (already 8, 16, 16).

        For pre-extraction pipelines where ClipVAE output is cached.
        """
        return self.extract_from_patches(vae_latents)


# =============================================================================
# Extended UNet with GeoVocab Conditioning
# =============================================================================

# Import base classes (these come from sd15-flow-trainer)
try:
    from sd15_trainer_geo.unet.base_simplex import (
        SD15UNetSimplex,
        SimplexConfig,
        CrossAttnDownBlock2D,
        CrossAttnUpBlock2D,
        get_timestep_embedding,
    )
    from sd15_trainer_geo.config.model_config import UNetConfig
    _HAS_SD15_TRAINER = True
except ImportError:
    _HAS_SD15_TRAINER = False


if _HAS_SD15_TRAINER:

    class SD15UNetGeoVocab(SD15UNetSimplex):
        """
        SD1.5 UNet with KSimplex prior + GeoVocab conditioning.

        Extends SD15UNetSimplex:
            - Adds GeoVocabConditioner that processes patch-maker features
            - Deformation factors from gate vectors condition simplex layers
            - Optional cross-attention enriches CLIP tokens before simplex prior

        Training: freeze UNet + patch_maker, train geo_prior + geo_conditioner.
        """

        def __init__(
            self,
            unet_config: Optional["UNetConfig"] = None,
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
            Extended forward with optional geometric vocabulary conditioning.

            Additional args:
                gate_vectors:   (B, 64, 17) from patch maker
                patch_features: (B, 64, 256) from patch maker

            If gate_vectors/patch_features are None, falls back to
            standard SD15UNetSimplex behavior (timestep-only conditioning).
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
                    gate_vectors=gate_vectors.to(sample.device),
                    patch_features=patch_features.to(sample.device),
                    clip_tokens=encoder_hidden_states,
                )
                geo_deform_factors = geo_out["deform_factors"]

                # If cross-attention enrichment available, use it
                if geo_out["enriched_clip"] is not None:
                    encoder_hidden_states = geo_out["enriched_clip"]

            # --- Geometric prior on CLIP conditioning ---
            # Override deformation schedule with geo-conditioned factors
            # This replaces the timestep-only schedule inside geo_prior
            encoder_hidden_states, prior_info = self.geo_prior(
                encoder_hidden_states,
                timestep_normalized=timestep_normalized,
            )
            # TODO: Wire geo_deform_factors into geo_prior's attention layers
            # For now, geo_conditioner's cross-attn enrichment is the primary path
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
            """Freeze UNet backbone. Train geo_prior + geo_conditioner."""
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


# =============================================================================
# Verification
# =============================================================================

def verify_conditioner():
    """Standalone test of the conditioner modules."""
    config = GeoVocabConfig()
    conditioner = GeoVocabConditioner(config)
    n_params = conditioner.param_count()
    print(f"GeoVocabConditioner: {n_params:,} params")

    # Simulate inputs
    B = 2
    gate_vectors = torch.randn(B, 64, 17)
    patch_features = torch.randn(B, 64, 256)
    clip_tokens = torch.randn(B, 77, 768)

    out = conditioner(gate_vectors, patch_features, clip_tokens)
    print(f"  deform_factors: {out['deform_factors'].shape}")  # (2, 4)
    print(f"  enriched_clip:  {out['enriched_clip'].shape}")   # (2, 77, 768)
    print(f"  deform range:   [{out['deform_factors'].min():.3f}, {out['deform_factors'].max():.3f}]")
    print(f"✓ GeoVocabConditioner verified")

    return conditioner


if __name__ == "__main__":
    verify_conditioner()