"""
Superposition Patch Classifier — Standalone Inference Module
=============================================================
Two-tier gated geometric transformer that extracts structural
properties from (8, 16, 16) latent patches.

No dependencies beyond PyTorch. All grid/gate constants inlined.

Input:  (B, 8, 16, 16)  — adapted latent patches
Output: gate_vectors (B, 64, 17), patch_features (B, 64, 256), logits

Usage:
    from geometric_model import load_from_hub, extract_features

    model, config = load_from_hub()  # reads config.json + model.pt from Hub
    out = model(patches)

    # Gate vectors: explicit geometric properties per patch
    local_gates = torch.cat([
        F.softmax(out["local_dim_logits"], dim=-1),      # 4d: dimensionality
        F.softmax(out["local_curv_logits"], dim=-1),      # 3d: curvature class
        torch.sigmoid(out["local_bound_logits"]),          # 1d: boundary flag
        torch.sigmoid(out["local_axis_logits"]),           # 3d: active axes
    ], dim=-1)  # (B, 64, 11)

    structural_gates = torch.cat([
        F.softmax(out["struct_topo_logits"], dim=-1),      # 2d: topology
        torch.sigmoid(out["struct_neighbor_logits"]),       # 1d: neighbor density
        F.softmax(out["struct_role_logits"], dim=-1),       # 3d: surface role
    ], dim=-1)  # (B, 64, 6)

    gate_vectors = torch.cat([local_gates, structural_gates], dim=-1)  # (B, 64, 17)
    patch_features = out["patch_features"]                              # (B, 64, embed_dim)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Grid Constants (inlined from generator — no dependency needed)
# ══════════════════════════════════════════════════════════════════════════════

GZ, GY, GX = 8, 16, 16
PATCH_Z, PATCH_Y, PATCH_X = 2, 4, 4
PATCH_VOL = PATCH_Z * PATCH_Y * PATCH_X          # 32
MACRO_Z = GZ // PATCH_Z                           # 4
MACRO_Y = GY // PATCH_Y                           # 4
MACRO_X = GX // PATCH_X                           # 4
MACRO_N = MACRO_Z * MACRO_Y * MACRO_X             # 64

# Local gates: intrinsic per-patch (no cross-patch info)
NUM_LOCAL_DIMS = 4       # 0D point, 1D line, 2D surface, 3D volume
NUM_LOCAL_CURVS = 3      # rigid, curved, combined
NUM_LOCAL_BOUNDARY = 1   # partial fill flag
NUM_LOCAL_AXES = 3       # which axes have extent > 1
LOCAL_GATE_DIM = NUM_LOCAL_DIMS + NUM_LOCAL_CURVS + NUM_LOCAL_BOUNDARY + NUM_LOCAL_AXES  # 11

# Structural gates: relational (require neighborhood context)
NUM_STRUCT_TOPO = 2      # open / closed
NUM_STRUCT_NEIGHBOR = 1  # normalized neighbor count
NUM_STRUCT_ROLE = 3      # isolated / boundary / interior
STRUCTURAL_GATE_DIM = NUM_STRUCT_TOPO + NUM_STRUCT_NEIGHBOR + NUM_STRUCT_ROLE  # 6

TOTAL_GATE_DIM = LOCAL_GATE_DIM + STRUCTURAL_GATE_DIM  # 17

# Shape classes (27 geometric primitives)
CLASS_NAMES = [
    "point", "line", "corner", "cross", "arc", "helix", "circle",
    "triangle", "quad", "plane", "disc",
    "tetrahedron", "cube", "pyramid", "prism", "octahedron", "pentachoron", "wedge",
    "sphere", "hemisphere", "torus", "bowl", "saddle", "capsule", "cylinder", "cone", "channel"
]
NUM_CLASSES = len(CLASS_NAMES)

# Legacy gate names
GATES = ["rigid", "curved", "combined", "open", "closed"]
NUM_GATES = len(GATES)


# ══════════════════════════════════════════════════════════════════════════════
# Patch Embedding
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding3D(nn.Module):
    def __init__(self, patch_dim=64):
        super().__init__()
        self.proj = nn.Linear(PATCH_VOL, patch_dim)
        pz = torch.arange(MACRO_Z).float() / MACRO_Z
        py = torch.arange(MACRO_Y).float() / MACRO_Y
        px = torch.arange(MACRO_X).float() / MACRO_X
        pos = torch.stack(torch.meshgrid(pz, py, px, indexing='ij'), dim=-1).reshape(MACRO_N, 3)
        self.register_buffer('pos_embed', pos)
        self.pos_proj = nn.Linear(3, patch_dim)

    def forward(self, x):
        B = x.shape[0]
        patches = x.view(B, MACRO_Z, PATCH_Z, MACRO_Y, PATCH_Y, MACRO_X, PATCH_X)
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(B, MACRO_N, PATCH_VOL)
        return self.proj(patches) + self.pos_proj(self.pos_embed)


# ══════════════════════════════════════════════════════════════════════════════
# Transformer Blocks
# ══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        return x + self.ff(self.ln2(x))


class GatedGeometricAttention(nn.Module):
    """
    Multi-head attention with two-tier gate modulation.
    Q, K see both local and structural gates.
    V modulated by combined gate vector.
    Per-head compatibility bias from gate interactions.
    """

    def __init__(self, embed_dim, gate_dim, n_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim + gate_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim + gate_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.gate_q = nn.Linear(gate_dim, n_heads)
        self.gate_k = nn.Linear(gate_dim, n_heads)
        self.v_gate = nn.Sequential(nn.Linear(gate_dim, embed_dim), nn.Sigmoid())

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, h, gate_features):
        B, N, _ = h.shape
        hg = torch.cat([h, gate_features], dim=-1)
        Q = self.q_proj(hg).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hg).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        V = self.v_proj(h)
        V = (V * self.v_gate(gate_features)).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        content_scores = (Q @ K.transpose(-2, -1)) / self.scale
        gq = self.gate_q(gate_features)
        gk = self.gate_k(gate_features)
        compat = torch.einsum('bih,bjh->bhij', gq, gk)

        attn = F.softmax(content_scores + compat, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N, self.embed_dim)
        return self.out_proj(out)


class GeometricTransformerBlock(nn.Module):
    def __init__(self, embed_dim, gate_dim, n_heads, dropout=0.1, ff_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = GatedGeometricAttention(embed_dim, gate_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_mult, embed_dim), nn.Dropout(dropout)
        )

    def forward(self, h, gate_features):
        h = h + self.attn(self.ln1(h), gate_features)
        h = h + self.ff(self.ln2(h))
        return h


# ══════════════════════════════════════════════════════════════════════════════
# Main Model
# ══════════════════════════════════════════════════════════════════════════════

class SuperpositionPatchClassifier(nn.Module):
    """
    Two-tier gated geometric transformer.

    Stage 0: Local gates from raw patch embeddings (what IS in this patch)
    Stage 1: Bootstrap attention with local gate context
    Stage 1.5: Structural gates from post-attention features (what ROLE this patch plays)
    Stage 2: Geometric gated attention with both gate tiers
    Stage 3: Classification heads

    For feature extraction (no classification), use outputs:
        - gate vectors:    cat(local_gates, structural_gates) → (B, 64, 17)
        - patch_features:  out["patch_features"]              → (B, 64, embed_dim)
        - global_features: out["global_features"]             → (B, embed_dim)
    """

    def __init__(self, embed_dim=128, patch_dim=64, n_bootstrap=2, n_geometric=2,
                 n_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding3D(patch_dim)

        # Stage 0: Local encoder + gate heads
        local_hidden = patch_dim * 2
        self.local_encoder = nn.Sequential(
            nn.Linear(patch_dim, local_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(local_hidden, local_hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.local_dim_head = nn.Linear(local_hidden, NUM_LOCAL_DIMS)
        self.local_curv_head = nn.Linear(local_hidden, NUM_LOCAL_CURVS)
        self.local_bound_head = nn.Linear(local_hidden, NUM_LOCAL_BOUNDARY)
        self.local_axis_head = nn.Linear(local_hidden, NUM_LOCAL_AXES)

        # Projection into transformer dim
        self.proj = nn.Linear(patch_dim + LOCAL_GATE_DIM, embed_dim)

        # Stage 1: Bootstrap blocks
        self.bootstrap_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout)
            for _ in range(n_bootstrap)
        ])

        # Stage 1.5: Structural gate heads
        self.struct_topo_head = nn.Linear(embed_dim, NUM_STRUCT_TOPO)
        self.struct_neighbor_head = nn.Linear(embed_dim, NUM_STRUCT_NEIGHBOR)
        self.struct_role_head = nn.Linear(embed_dim, NUM_STRUCT_ROLE)

        # Stage 2: Geometric gated blocks
        self.geometric_blocks = nn.ModuleList([
            GeometricTransformerBlock(embed_dim, TOTAL_GATE_DIM, n_heads, dropout)
            for _ in range(n_geometric)
        ])

        # Stage 3: Classification heads
        gated_dim = embed_dim + TOTAL_GATE_DIM

        self.patch_shape_head = nn.Sequential(
            nn.Linear(gated_dim, embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, NUM_CLASSES)
        )

        self.global_pool = nn.Sequential(
            nn.Linear(gated_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.global_gate_head = nn.Linear(embed_dim, NUM_GATES)
        self.global_shape_head = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x):
        # Patch embedding
        e = self.patch_embed(x)

        # Stage 0: Local gates
        e_local = self.local_encoder(e)
        local_dim_logits = self.local_dim_head(e_local)
        local_curv_logits = self.local_curv_head(e_local)
        local_bound_logits = self.local_bound_head(e_local)
        local_axis_logits = self.local_axis_head(e_local)

        local_gates = torch.cat([
            F.softmax(local_dim_logits, dim=-1),
            F.softmax(local_curv_logits, dim=-1),
            torch.sigmoid(local_bound_logits),
            torch.sigmoid(local_axis_logits),
        ], dim=-1)

        # Stage 1: Bootstrap
        h = self.proj(torch.cat([e, local_gates], dim=-1))
        for blk in self.bootstrap_blocks:
            h = blk(h)

        # Stage 1.5: Structural gates
        struct_topo_logits = self.struct_topo_head(h)
        struct_neighbor_logits = self.struct_neighbor_head(h)
        struct_role_logits = self.struct_role_head(h)

        structural_gates = torch.cat([
            F.softmax(struct_topo_logits, dim=-1),
            torch.sigmoid(struct_neighbor_logits),
            F.softmax(struct_role_logits, dim=-1),
        ], dim=-1)

        all_gates = torch.cat([local_gates, structural_gates], dim=-1)

        # Stage 2: Geometric routing
        for blk in self.geometric_blocks:
            h = blk(h, all_gates)

        # Stage 3: Classification
        h_gated = torch.cat([h, all_gates], dim=-1)
        shape_logits = self.patch_shape_head(h_gated)
        g = self.global_pool(h_gated.mean(dim=1))

        return {
            "local_dim_logits": local_dim_logits,
            "local_curv_logits": local_curv_logits,
            "local_bound_logits": local_bound_logits,
            "local_axis_logits": local_axis_logits,
            "struct_topo_logits": struct_topo_logits,
            "struct_neighbor_logits": struct_neighbor_logits,
            "struct_role_logits": struct_role_logits,
            "patch_shape_logits": shape_logits,
            "patch_features": h,
            "global_features": g,
            "global_gates": self.global_gate_head(g),
            "global_shapes": self.global_shape_head(g),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Hub Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_config(repo_id="AbstractPhil/geovocab-patch-maker", config_file="config.json"):
    """Load model config from HuggingFace Hub."""
    import json
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=config_file)
    with open(path, "r") as f:
        return json.load(f)


def from_config(config, device="cpu"):
    """Instantiate model from config dict (no weights)."""
    return SuperpositionPatchClassifier(
        embed_dim=config["embed_dim"],
        patch_dim=config["patch_dim"],
        n_bootstrap=config["n_bootstrap"],
        n_geometric=config["n_geometric"],
        n_heads=config["n_heads"],
        dropout=config.get("dropout", 0.0),
    ).to(device)


def load_from_hub(
    repo_id="AbstractPhil/geovocab-patch-maker",
    weights_file="model.pt",
    config_file="config.json",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Load pretrained model from HuggingFace Hub.

    Reads config.json for architecture, model.pt for weights.
    Falls back to config embedded in checkpoint if config.json missing.
    """
    from huggingface_hub import hf_hub_download

    # Load config
    try:
        config = load_config(repo_id, config_file)
        print(f"✓ Config loaded from {config_file}")
    except Exception:
        config = None

    # Load weights
    weights_path = hf_hub_download(repo_id=repo_id, filename=weights_file)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # Config priority: config.json > checkpoint config
    if config is None:
        config = ckpt["config"]
        print(f"  Config from checkpoint (no {config_file} found)")

    model = from_config(config, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Loaded {repo_id} (epoch {epoch}, {n_params:,} params)")
    return model, config


@torch.no_grad()
def extract_features(model, patches, batch_size=256):
    """
    Convenience: patches → (gate_vectors, patch_features)

    Args:
        model:    SuperpositionPatchClassifier (eval mode)
        patches:  (N, 8, 16, 16) tensor
        batch_size: inference batch size

    Returns:
        gate_vectors:   (N, 64, 17) — explicit geometric properties
        patch_features: (N, 64, embed_dim) — learned representations
    """
    device = next(model.parameters()).device
    all_gates, all_patch = [], []

    for s in range(0, patches.shape[0], batch_size):
        batch = patches[s:s + batch_size].to(device)
        out = model(batch)

        local = torch.cat([
            F.softmax(out["local_dim_logits"], dim=-1),
            F.softmax(out["local_curv_logits"], dim=-1),
            torch.sigmoid(out["local_bound_logits"]),
            torch.sigmoid(out["local_axis_logits"]),
        ], dim=-1)

        struct = torch.cat([
            F.softmax(out["struct_topo_logits"], dim=-1),
            torch.sigmoid(out["struct_neighbor_logits"]),
            F.softmax(out["struct_role_logits"], dim=-1),
        ], dim=-1)

        all_gates.append(torch.cat([local, struct], dim=-1).cpu())
        all_patch.append(out["patch_features"].cpu())

    return torch.cat(all_gates), torch.cat(all_patch)


# ══════════════════════════════════════════════════════════════════════════════
# Quick test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    # Test 1: Direct instantiation
    model = SuperpositionPatchClassifier()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SuperpositionPatchClassifier: {n_params:,} parameters")

    x = torch.randn(2, 8, 16, 16)
    out = model(x)
    print(f"  Input:           {x.shape}")
    print(f"  patch_features:  {out['patch_features'].shape}")
    print(f"  local_dim:       {out['local_dim_logits'].shape}")
    print(f"  struct_topo:     {out['struct_topo_logits'].shape}")
    print(f"  patch_shapes:    {out['patch_shape_logits'].shape}")
    print(f"  global_features: {out['global_features'].shape}")

    # Test 2: From config
    import os
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            config = json.load(f)
        model2 = from_config(config)
        print(f"\n  from_config: {sum(p.numel() for p in model2.parameters()):,} params")
        print(f"  config: {config['model_type']} embed={config['embed_dim']} patches={config['num_patches']}")