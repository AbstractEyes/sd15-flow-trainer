"""
SD1.5 UNet with KSimplex Geometric Cross-Attention Prior
==========================================================
Inserts a Cayley-Menger validated geometric attention layer that
modulates CLIP conditioning (B, 77, 768) before it enters the UNet.

Research basis (validated results):
    - KSimplex as attention: 89.13% FMNIST, 84.59% CIFAR-10, 69.08% CIFAR-100
    - KSimplex LLM: 54M params, PPL 113, 100% geometric validity
    - Attention sharpens with depth (entropy decreases through layers)
    - 25-77 tokens is the proven sweet spot for geometric attention
    - Deformation stability zone: 0.15-0.35, edim/k_max >= 8x

Architecture:
    CLIP (B, 77, 768)
        → KSimplexCrossAttentionPrior (geometric modulation)
        → modulated (B, 77, 768)
        → frozen SD1.5 UNet cross-attention blocks (all 16)

All original SD1.5 weights load cleanly. New geometric params are additive.

Usage:
    from sd15_trainer_geo.config import load_unet_config
    from sd15_trainer_geo.unet.base_simplex import SD15UNetSimplex, SimplexConfig

    unet_cfg = load_unet_config("sd-legacy/stable-diffusion-v1-5")
    geo_cfg = SimplexConfig(k=4, edim=32, num_layers=4)
    unet = SD15UNetSimplex(unet_cfg, geo_cfg)

    # Load pretrained SD1.5 weights (geo layers initialized fresh)
    unet.load_pretrained("sd-legacy/stable-diffusion-v1-5")

Author: AbstractPhil
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

from ..config.model_config import UNetConfig
from .base import (
    SD15UNet,
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    get_timestep_embedding,
)


# =============================================================================
# Simplex Configuration
# =============================================================================

@dataclass
class SimplexConfig:
    """Configuration for KSimplex geometric attention prior."""

    # Simplex geometry
    k: int = 4                          # Simplex dimension (4 = pentachoron, 5 vertices)
    edim: int = 32                      # Embedding dim for vertex coordinates
    feat_dim: int = 768                 # Feature dim (must match CLIP hidden_size)
    num_layers: int = 4                 # Stacked attention depth

    # Deformation
    base_deformation: float = 0.25      # Base deformation scale (within 0.15-0.35 zone)
    learnable_deformation: bool = True  # Per-layer learnable deformation
    timestep_conditioned: bool = True   # Phase 3: deformation varies with timestep

    # Attention
    num_heads: int = 8                  # Attention heads for value projection
    dropout: float = 0.0

    # Regularization
    cm_loss_weight: float = 0.01       # Cayley-Menger validity loss weight
    vol_consistency_weight: float = 0.005  # Volume consistency across k-levels

    # Residual blending
    residual_blend: str = "learnable"   # "fixed", "learnable", or "timestep"
    initial_blend: float = 0.0         # sigmoid(0) = 0.5 blend; start conservative

    @property
    def num_vertices(self) -> int:
        return self.k + 1

    @property
    def num_edges(self) -> int:
        return (self.num_vertices * (self.num_vertices - 1)) // 2


# =============================================================================
# Cayley-Menger Geometry
# =============================================================================

def cayley_menger_determinant(
    squared_distances: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Compute Cayley-Menger determinant for k-simplex validation.

    Args:
        squared_distances: (batch, num_edges) pairwise squared distances
        k: simplex dimension

    Returns:
        (batch,) determinant values
    """
    n = k + 1
    B = squared_distances.shape[0]
    orig_dtype = squared_distances.dtype

    # linalg.det requires float32+ (not supported for half)
    sq = squared_distances.float()

    cm = torch.zeros(B, n + 1, n + 1, device=sq.device, dtype=torch.float32)
    cm[:, 0, 1:] = 1.0
    cm[:, 1:, 0] = 1.0

    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            cm[:, i + 1, j + 1] = sq[:, pair_idx]
            cm[:, j + 1, i + 1] = sq[:, pair_idx]
            pair_idx += 1

    return torch.linalg.det(cm).to(dtype=orig_dtype)


def compute_simplex_volume_sq(
    squared_distances: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Compute squared volume of k-simplex from Cayley-Menger determinant.

    vol²(k-simplex) = (-1)^(k+1) / (2^k * (k!)²) * CM_det
    """
    det = cayley_menger_determinant(squared_distances, k)
    sign = (-1) ** (k + 1)
    factorial_k = math.factorial(k)
    denom = (2 ** k) * (factorial_k ** 2)
    return sign * det / denom


# =============================================================================
# Simplex Template Factory
# =============================================================================

class SimplexFactory:
    """
    Generates canonical regular simplex templates.
    Ported from geovocab2's battle-tested SimplexFactory.

    Algorithm:
        Vertex i has coordinate i = sqrt((k+1)/k), all others = -1/k.
        This guarantees all pairwise distances = sqrt(2(k+1)/k).
        Then center at origin and normalize to unit edge length.

    A regular k-simplex has all edge lengths equal.
    Templates are used as deformation anchors.
    """

    @staticmethod
    def regular(k: int, edim: int, scale: float = 1.0) -> torch.Tensor:
        """
        Generate regular k-simplex with all edges equal.

        Uses geovocab2 construction:
            - Fill (k+1, k+1) with -1/k
            - Set diagonal to sqrt((k+1)/k)
            - Embed into edim, center, normalize to unit edge

        Returns: (k+1, edim) vertex coordinates
        """
        n = k + 1
        assert edim >= k, f"edim ({edim}) must be >= k ({k})"

        if k == 0:
            return torch.zeros(1, edim)

        # Minimal dimension: need k+1 coords for k-simplex
        min_dim = k + 1

        # Fill all coordinates with -1/k
        vertices_minimal = torch.full((n, min_dim), -1.0 / k)

        # Diagonal: sqrt((k+1)/k)
        coef = math.sqrt((k + 1.0) / k)
        vertices_minimal[range(n), range(min_dim)] = coef

        # Embed into higher dimensional space if needed
        if edim > min_dim:
            vertices = torch.zeros(n, edim)
            vertices[:, :min_dim] = vertices_minimal
        else:
            vertices = vertices_minimal[:, :edim]

        # Center at origin
        vertices = vertices - vertices.mean(dim=0, keepdim=True)

        # Normalize to unit edge length, then apply scale
        edge_length = (vertices[0] - vertices[1]).norm()
        if edge_length > 1e-10:
            vertices = vertices / edge_length * scale

        return vertices

    @staticmethod
    def pairwise_squared_distances(vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute all pairwise squared distances (vectorized).

        Args:
            vertices: (n, edim)

        Returns:
            (num_edges,) flattened upper triangle
        """
        # (n, 1, d) - (1, n, d) → (n, n, d) → sum → (n, n)
        diff = vertices.unsqueeze(1) - vertices.unsqueeze(0)
        dist_sq_matrix = (diff ** 2).sum(dim=-1)

        # Extract upper triangle
        n = vertices.shape[0]
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
        return dist_sq_matrix[i_idx, j_idx]


# =============================================================================
# KSimplex Attention Layer
# =============================================================================

class KSimplexAttentionLayer(nn.Module):
    """
    Single layer of KSimplex geometric attention.

    Computes attention weights from simplex edge distances
    validated by Cayley-Menger determinants.

    For n_tokens tokens:
        1. Project tokens to simplex coordinate space (edim)
        2. Add deformed template coordinates
        3. Compute pairwise squared distances
        4. Convert distances to attention weights
        5. Apply attention to value projection

    The geometric constraint ensures attention patterns
    correspond to valid simplex configurations.
    """

    def __init__(
        self,
        n_tokens: int,
        feat_dim: int = 768,
        edim: int = 32,
        k: int = 4,
        num_heads: int = 8,
        base_deformation: float = 0.25,
        learnable_deformation: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.feat_dim = feat_dim
        self.edim = edim
        self.k = k
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads

        # Project features → simplex coordinates
        self.to_coords = nn.Linear(feat_dim, edim)

        # Template: canonical regular simplex (frozen anchor)
        template = SimplexFactory.regular(k, edim)
        self.register_buffer("template", template)  # (k+1, edim)

        # Learnable deformation offsets per token position
        # Each token gets assigned to nearest vertex via soft routing
        self.token_to_vertex = nn.Linear(feat_dim, k + 1)  # soft vertex assignment
        self.deformation_offsets = nn.Parameter(
            torch.randn(k + 1, edim) * 0.01
        )

        # Deformation scale
        if learnable_deformation:
            self.deformation_scale = nn.Parameter(
                torch.tensor(base_deformation)
            )
        else:
            self.register_buffer(
                "deformation_scale",
                torch.tensor(base_deformation),
            )

        # Value projection (standard attention path)
        self.to_v = nn.Linear(feat_dim, feat_dim)
        self.to_out = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)

        # Layer norm for stability
        self.norm = nn.LayerNorm(feat_dim)

    def _compute_geometric_attention(
        self,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute attention weights from simplex geometry.

        Args:
            coords: (B, T, edim) token coordinates in simplex space

        Returns:
            attn_weights: (B, num_heads, T, T)
            geometry_info: dict with distances, volumes for loss
        """
        B, T, E = coords.shape

        # Pairwise squared distances between all tokens
        # (B, T, 1, E) - (B, 1, T, E) → (B, T, T)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)  # (B, T, T)

        # Convert distances to attention: closer = higher weight
        # Use negative distance as logit (like dot-product attention)
        scale = self.edim ** -0.5
        attn_logits = -dist_sq * scale

        # Expand to multi-head: each head gets same geometric bias
        # but different value projections
        attn_logits = attn_logits.unsqueeze(1).expand(B, self.num_heads, T, T)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Geometry info for CM validation loss
        geometry_info = {
            "dist_sq": dist_sq,
            "coords": coords,
        }

        return attn_weights, geometry_info

    def _sample_simplex_distances(
        self,
        dist_sq: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample k+1 tokens and extract their pairwise distances
        for Cayley-Menger validation.

        Uses the first k+1 tokens (deterministic for stability).

        Returns:
            sampled_dist_sq: (B, num_edges)
            k: simplex dimension
        """
        n = min(self.k + 1, dist_sq.shape[1])
        k = n - 1

        # Extract upper triangle for first n tokens
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append(dist_sq[:, i, j])

        return torch.stack(edges, dim=-1), k

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: (B, T, feat_dim)

        Returns:
            output: (B, T, feat_dim)
            geometry_info: dict with CM validation data
        """
        B, T, D = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        # --- Simplex coordinate computation ---

        # Project to coordinate space
        coords = self.to_coords(hidden_states)  # (B, T, edim)

        # Soft vertex assignment: which template vertex each token is near
        vertex_logits = self.token_to_vertex(hidden_states)  # (B, T, k+1)
        vertex_weights = F.softmax(vertex_logits, dim=-1)    # (B, T, k+1)

        # Deformed template: base template + learned offsets
        deform_scale = torch.clamp(self.deformation_scale, 0.05, 0.5)
        deformed = self.template + self.deformation_offsets * deform_scale  # (k+1, edim)

        # Blend template coordinates into token coords via soft assignment
        template_contribution = torch.matmul(
            vertex_weights, deformed
        )  # (B, T, edim)

        # Final coordinates: learned projection + geometric anchor
        coords = coords + template_contribution

        # --- Geometric attention ---
        attn_weights, geometry_info = self._compute_geometric_attention(coords)

        # --- Value path (standard) ---
        v = self.to_v(hidden_states)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply geometric attention to values
        out = torch.matmul(attn_weights, v)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.to_out(out)

        # --- CM validation data ---
        sampled_dists, effective_k = self._sample_simplex_distances(
            geometry_info["dist_sq"]
        )
        geometry_info["sampled_dist_sq"] = sampled_dists
        geometry_info["effective_k"] = effective_k
        geometry_info["deformation_scale"] = deform_scale
        geometry_info["vertex_weights"] = vertex_weights

        return residual + out, geometry_info


# =============================================================================
# Stacked KSimplex Attention
# =============================================================================

class StackedKSimplexAttention(nn.Module):
    """
    Multiple KSimplex attention layers with progressive sharpening.

    Research finding: attention entropy decreases through layers,
    creating a natural coarse-to-fine geometric focus.
    """

    def __init__(
        self,
        n_tokens: int,
        feat_dim: int = 768,
        edim: int = 32,
        k: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
        base_deformation: float = 0.25,
        learnable_deformation: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            KSimplexAttentionLayer(
                n_tokens=n_tokens,
                feat_dim=feat_dim,
                edim=edim,
                k=k,
                num_heads=num_heads,
                base_deformation=base_deformation,
                learnable_deformation=learnable_deformation,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Args:
            hidden_states: (B, T, feat_dim)

        Returns:
            output: (B, T, feat_dim)
            all_geometry: list of per-layer geometry dicts
        """
        all_geometry = []
        for layer in self.layers:
            hidden_states, geom = layer(hidden_states)
            all_geometry.append(geom)

        return hidden_states, all_geometry


# =============================================================================
# Cross-Attention Prior (Phase 1)
# =============================================================================

class KSimplexCrossAttentionPrior(nn.Module):
    """
    Geometric attention prior that modulates CLIP conditioning
    before it enters the UNet cross-attention blocks.

    Operates on (B, 77, 768) CLIP hidden states.
    77 tokens is in the proven sweet spot for geometric attention.

    Optional timestep conditioning (Phase 3):
        - Early timesteps (high t): higher deformation → softer geometry
        - Late timesteps (low t): lower deformation → sharper geometry
        Maps directly to Sol's temperature scheduling insight.
    """

    def __init__(self, config: SimplexConfig):
        super().__init__()
        self.config = config

        # Core geometric attention stack
        self.attention = StackedKSimplexAttention(
            n_tokens=77,  # CLIP max tokens
            feat_dim=config.feat_dim,
            edim=config.edim,
            k=config.k,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            base_deformation=config.base_deformation,
            learnable_deformation=config.learnable_deformation,
            dropout=config.dropout,
        )

        # Residual blending: how much geometric modulation to apply
        if config.residual_blend == "learnable":
            self.blend_logit = nn.Parameter(
                torch.tensor(config.initial_blend)
            )
        elif config.residual_blend == "timestep":
            # Timestep-conditioned blend via small MLP
            self.blend_mlp = nn.Sequential(
                nn.Linear(1, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )
        else:
            self.register_buffer(
                "blend_logit",
                torch.tensor(config.initial_blend),
            )

        # Timestep conditioning for deformation (Phase 3)
        if config.timestep_conditioned:
            self.deform_schedule = nn.Sequential(
                nn.Linear(1, 64),
                nn.SiLU(),
                nn.Linear(64, config.num_layers),
                nn.Sigmoid(),  # Output in [0, 1], scaled to deformation range
            )

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        timestep_normalized: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            encoder_hidden_states: (B, 77, 768) CLIP conditioning
            timestep_normalized:   (B,) timestep in [0, 1] range
                                   0 = fully denoised, 1 = pure noise

        Returns:
            modulated: (B, 77, 768) geometrically modulated conditioning
            prior_info: dict with geometry data and blend values
        """
        B = encoder_hidden_states.shape[0]

        # Phase 3: timestep-conditioned deformation
        if (
            self.config.timestep_conditioned
            and timestep_normalized is not None
        ):
            t_input = timestep_normalized.to(dtype=encoder_hidden_states.dtype).unsqueeze(-1)  # (B, 1)
            deform_factors = self.deform_schedule(t_input)  # (B, num_layers)

            # Scale deformation per layer: high t → more deform, low t → less
            for i, layer in enumerate(self.attention.layers):
                factor = deform_factors[:, i].mean()  # batch average for shared param
                layer.deformation_scale.data.copy_(
                    layer.deformation_scale.data * (0.5 + factor)
                )

        # Geometric attention modulation
        modulated, all_geometry = self.attention(encoder_hidden_states)

        # Residual blend
        if self.config.residual_blend == "timestep" and timestep_normalized is not None:
            t_input = timestep_normalized.to(dtype=encoder_hidden_states.dtype).unsqueeze(-1)
            blend = torch.sigmoid(self.blend_mlp(t_input))  # (B, 1)
            blend = blend.unsqueeze(1)  # (B, 1, 1)
        elif hasattr(self, "blend_logit"):
            blend = torch.sigmoid(self.blend_logit)
        else:
            blend = 0.5

        # Blend: original + geometric modulation
        output = (1.0 - blend) * encoder_hidden_states + blend * modulated

        prior_info = {
            "all_geometry": all_geometry,
            "blend": blend if isinstance(blend, torch.Tensor) else torch.tensor(blend),
        }

        return output, prior_info


# =============================================================================
# Geometric Loss Functions
# =============================================================================

class GeometricLoss(nn.Module):
    """
    Combined geometric regularization loss.

    Components:
        1. CM Validity: Cayley-Menger determinant should have correct sign
        2. Volume Consistency: Encourage meaningful vol² across k-levels
    """

    def __init__(self, config: SimplexConfig):
        super().__init__()
        self.config = config

    def cm_validity_loss(
        self,
        all_geometry: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Penalize invalid simplex configurations.
        Valid k-simplex: (-1)^(k+1) * CM_det > 0
        """
        loss = torch.tensor(0.0, device=all_geometry[0]["coords"].device)
        count = 0

        for geom in all_geometry:
            dist_sq = geom["sampled_dist_sq"]
            k = geom["effective_k"]

            det = cayley_menger_determinant(dist_sq, k)
            expected_sign = (-1) ** (k + 1)

            # Signed volume should be positive
            signed = expected_sign * det
            # Hinge loss: penalize negative (invalid) configurations
            validity_loss = F.relu(-signed + 1e-6).mean()
            loss = loss + validity_loss
            count += 1

        return loss / max(count, 1)

    def volume_consistency_loss(
        self,
        all_geometry: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Encourage differentiation in volume across layers.
        Don't want all layers collapsing to same geometry.
        """
        if len(all_geometry) < 2:
            return torch.tensor(0.0, device=all_geometry[0]["coords"].device)

        volumes = []
        for geom in all_geometry:
            dist_sq = geom["sampled_dist_sq"]
            k = geom["effective_k"]
            vol_sq = compute_simplex_volume_sq(dist_sq, k)
            volumes.append(vol_sq.mean())

        vol_stack = torch.stack(volumes)
        # Negative std: encourage spread, not collapse
        return -torch.std(torch.log(vol_stack.abs() + 1e-10))

    def forward(
        self,
        all_geometry: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total geometric loss.

        Returns:
            total_loss: weighted sum
            components: dict of individual losses
        """
        cm_loss = self.cm_validity_loss(all_geometry)
        vol_loss = self.volume_consistency_loss(all_geometry)

        total = (
            self.config.cm_loss_weight * cm_loss
            + self.config.vol_consistency_weight * vol_loss
        )

        return total, {
            "cm_validity": cm_loss,
            "volume_consistency": vol_loss,
            "geometric_total": total,
        }


# =============================================================================
# SD15 UNet with KSimplex Prior
# =============================================================================

class SD15UNetSimplex(SD15UNet):
    """
    SD1.5 UNet with KSimplex geometric cross-attention prior.

    Inherits full SD1.5 UNet architecture. Adds a geometric attention
    module that modulates encoder_hidden_states before they reach
    the cross-attention blocks.

    All original SD1.5 weights load cleanly into the parent.
    New geometric parameters are additive and initialized fresh.
    """

    def __init__(
        self,
        unet_config: Optional[UNetConfig] = None,
        simplex_config: Optional[SimplexConfig] = None,
    ):
        super().__init__(unet_config)

        self.simplex_config = simplex_config or SimplexConfig()
        self.geo_prior = KSimplexCrossAttentionPrior(self.simplex_config)
        self.geo_loss = GeometricLoss(self.simplex_config)

        # Track geometry for loss computation
        self._last_prior_info: Optional[Dict[str, Any]] = None

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same interface as SD15UNet, but encoder_hidden_states
        are geometrically modulated before entering cross-attention.
        """
        cfg = self.config

        # --- Timestep embedding ---
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(sample.shape[0])

        # Normalize timestep to [0, 1] for geometric conditioning
        # SD1.5 uses timesteps 0-999
        timestep_normalized = timestep.to(dtype=sample.dtype) / 1000.0

        t_emb = get_timestep_embedding(
            timestep,
            embedding_dim=cfg.block_out_channels[0],
            freq_shift=cfg.freq_shift,
        )
        t_emb = t_emb.to(dtype=sample.dtype)
        temb = self.time_embedding(t_emb)

        # --- Geometric prior on CLIP conditioning ---
        encoder_hidden_states, prior_info = self.geo_prior(
            encoder_hidden_states,
            timestep_normalized=timestep_normalized,
        )
        self._last_prior_info = prior_info

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

    def compute_geometric_loss(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute geometric regularization loss from last forward pass.
        Call after forward() during training.

        Returns:
            total_loss: scalar
            components: dict of loss components
        """
        if self._last_prior_info is None:
            zero = torch.tensor(0.0)
            return zero, {"cm_validity": zero, "volume_consistency": zero, "geometric_total": zero}

        return self.geo_loss(self._last_prior_info["all_geometry"])

    def get_geometry_stats(self) -> Dict[str, float]:
        """
        Get geometry diagnostics from last forward pass.
        Useful for logging during training.
        """
        if self._last_prior_info is None:
            return {}

        stats = {}
        blend = self._last_prior_info["blend"]
        if isinstance(blend, torch.Tensor):
            stats["blend"] = blend.mean().item()

        for i, geom in enumerate(self._last_prior_info["all_geometry"]):
            dist_sq = geom["sampled_dist_sq"]
            k = geom["effective_k"]
            vol_sq = compute_simplex_volume_sq(dist_sq, k)

            stats[f"layer_{i}/vol_sq"] = vol_sq.mean().item()
            stats[f"layer_{i}/deform_scale"] = geom["deformation_scale"].item()

            # Attention entropy from distances
            dist_matrix = geom["dist_sq"]
            attn = F.softmax(-dist_matrix / (self.simplex_config.edim ** 0.5), dim=-1)
            entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean()
            stats[f"layer_{i}/entropy"] = entropy.item()

        return stats

    # =========================================================================
    # Weight Loading
    # =========================================================================

    def load_pretrained(
        self,
        repo_id: str = "sd-legacy/stable-diffusion-v1-5",
        subfolder: str = "unet",
        filename: str = "diffusion_pytorch_model.safetensors",
        device: str = "cpu",
        strict: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        Load pretrained SD1.5 UNet weights.
        Geometric layers (geo_prior, geo_loss) are left freshly initialized.

        Returns:
            missing_keys: keys in model but not in checkpoint (geo layers)
            unexpected_keys: keys in checkpoint but not in model
        """
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id, subfolder=subfolder, filename=filename
        )
        state_dict = load_file(path, device=device)

        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        # Separate geometric vs real missing keys
        geo_missing = [k for k in missing if k.startswith(("geo_prior.", "geo_loss."))]
        real_missing = [k for k in missing if not k.startswith(("geo_prior.", "geo_loss."))]

        if real_missing:
            print(f"WARNING: Missing UNet keys ({len(real_missing)}):")
            for k in real_missing:
                print(f"  {k}")
        if unexpected:
            print(f"WARNING: Unexpected keys ({len(unexpected)}):")
            for k in unexpected:
                print(f"  {k}")

        n_geo = sum(
            p.numel() for n, p in self.named_parameters()
            if n.startswith(("geo_prior.", "geo_loss."))
        )
        n_unet = sum(
            p.numel() for n, p in self.named_parameters()
            if not n.startswith(("geo_prior.", "geo_loss."))
        )

        print(f"SD1.5 UNet loaded: {n_unet:,} params")
        print(f"KSimplex prior:    {n_geo:,} params (freshly initialized)")
        print(f"Total:             {n_unet + n_geo:,} params")

        if not real_missing and not unexpected:
            print("✓ All UNet weights loaded successfully")

        return missing, unexpected

    def freeze_unet(self):
        """Freeze all original UNet parameters. Only geometric layers train."""
        for name, param in self.named_parameters():
            if not name.startswith(("geo_prior.", "geo_loss.")):
                param.requires_grad = False

    def unfreeze_unet(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get only trainable (geometric) parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Verification
# =============================================================================

def verify_simplex_unet(
    unet_config: Optional[UNetConfig] = None,
    simplex_config: Optional[SimplexConfig] = None,
):
    """Print architecture summary and verify structure."""

    unet_config = unet_config or UNetConfig()
    simplex_config = simplex_config or SimplexConfig()

    model = SD15UNetSimplex(unet_config, simplex_config)

    total = sum(p.numel() for p in model.parameters())
    geo_params = sum(
        p.numel() for n, p in model.named_parameters()
        if n.startswith(("geo_prior.", "geo_loss."))
    )
    unet_params = total - geo_params

    print(f"SD15UNetSimplex Architecture")
    print(f"============================")
    print(f"  UNet params:     {unet_params:,}")
    print(f"  Geo params:      {geo_params:,} ({100*geo_params/total:.2f}%)")
    print(f"  Total params:    {total:,}")
    print(f"")
    print(f"  Simplex config:")
    print(f"    k:             {simplex_config.k} ({simplex_config.num_vertices} vertices, {simplex_config.num_edges} edges)")
    print(f"    edim:          {simplex_config.edim}")
    print(f"    feat_dim:      {simplex_config.feat_dim}")
    print(f"    num_layers:    {simplex_config.num_layers}")
    print(f"    deformation:   {simplex_config.base_deformation}")
    print(f"    timestep_cond: {simplex_config.timestep_conditioned}")
    print(f"    blend:         {simplex_config.residual_blend}")
    print(f"    CM loss wt:    {simplex_config.cm_loss_weight}")

    # Test forward pass
    print(f"\n  Forward pass test...")
    model.eval()
    with torch.no_grad():
        sample = torch.randn(1, 4, 64, 64)
        timestep = torch.tensor([500])
        enc_hs = torch.randn(1, 77, 768)

        out = model(sample, timestep, enc_hs)
        print(f"    Input:  {sample.shape}")
        print(f"    Output: {out.shape}")

        # Geometry stats
        stats = model.get_geometry_stats()
        for k, v in stats.items():
            print(f"    {k}: {v:.4f}")

        # Geometric loss
        geo_loss, geo_components = model.compute_geometric_loss()
        print(f"\n    Geometric loss: {geo_loss.item():.6f}")
        for k, v in geo_components.items():
            print(f"    {k}: {v.item():.6f}")

    # Freeze test
    model.freeze_unet()
    trainable = model.trainable_param_count()
    print(f"\n  After freeze_unet():")
    print(f"    Trainable:     {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"    Frozen:        {total - trainable:,}")

    print(f"\n  ✓ SD15UNetSimplex verified")
    return model


if __name__ == "__main__":
    verify_simplex_unet()