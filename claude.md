# CLAUDE.md — Working on sd15-flow-trainer

## What This Project Is

A rectified flow matching trainer that trains a geometric attention prior (~4.8M params) on a frozen SD1.5 UNet. The prior uses KSimplex (pentachoron) geometry validated by Cayley-Menger determinants to modulate CLIP conditioning before it enters UNet cross-attention blocks. The GeoVocab extension adds conditioning from a pretrained geometric patch analyzer.

**This is NOT a standard LoRA/fine-tuning trainer.** The UNet weights never change. Only the geometric prior and conditioner train.

## Core Concepts

### Flow Matching (not DDPM)
This uses rectified flow, not the original SD1.5 DDPM noise schedule:
- `x_t = (1 - t) * x_0 + t * noise` (linear interpolation, NOT sqrt-alpha weighting)
- `v_target = noise - x_0` (velocity field, NOT noise prediction)
- `t` is continuous in [0, 1], NOT discrete timesteps
- The UNet predicts velocity, not noise
- `timestep` passed to UNet is `t * 1000` (float, not quantized int)
- Min-SNR weighting is adapted for velocity prediction

### KSimplex Geometry
- k=4 simplex (pentachoron): 5 vertices, 10 edges in 32-dim space
- Cayley-Menger determinant validates geometric configurations: `(-1)^(k+1) * CM_det > 0`
- Deformation stability zone: 0.15–0.35 (clamped, learnable per-layer)
- Attention from pairwise distances: closer tokens = higher attention weight
- Template is a regular simplex (frozen anchor), tokens soft-route to vertices
- **Always compute CM determinant in float32** — half precision causes numerical issues

### Two-Tier Geometric Gates (from patch-maker)
The patch-maker outputs 17-dimensional gate vectors per patch:
- **Local (dims 0–10)**: dimensionality (softmax 4), curvature (softmax 3), boundary (sigmoid 1), axis_active (sigmoid 3)
- **Structural (dims 11–16)**: topology (softmax 2), neighbor_density (sigmoid 1), surface_role (softmax 3)

Local gates are intrinsic per-patch. Structural gates require cross-patch attention. This distinction matters for conditioning — local gates tell you what IS there, structural gates tell you what ROLE it plays.

### Rosetta Stone Finding
Text embeddings projected through a 256d VAE bottleneck into (8,16,16) patch space produce 2.5–3.5× higher geometric discriminability than actual images through the same analyzer. Three architecturally different encoders (T5, BERT, Beatrix) converge to ±5%. The geometric structure is in the language itself.

## Architecture Map

```
[FROZEN]                           [TRAINABLE]
                                   
CLIP (123M)                        geo_prior (~4.8M)
  ↓                                  ├─ StackedKSimplexAttention (4 layers)
  (B, 77, 768)                      │    ├─ to_coords: Linear(768, 32)
  ↓                                  │    ├─ token_to_vertex: Linear(768, 5)
geo_conditioner (if geovocab)        │    ├─ deformation_offsets: (5, 32)
  ├─ cross_attn enriches CLIP        │    ├─ to_v, to_out: Linear(768, 768)
  ├─ deform_factors → prior          │    └─ norm: LayerNorm(768)
  ↓                                  ├─ blend_logit or blend_mlp
geo_prior modulates CLIP             └─ deform_schedule (if timestep_conditioned)
  ↓                                
  (B, 77, 768) modulated          geo_conditioner (~0.5M, if geovocab)
  ↓                                  ├─ GateDeformationConditioner
SD1.5 UNet (860M, frozen)             │    ├─ gate_pool: Linear chain
  ↓                                  │    ├─ patch_attn: Linear(17, 1)
  (B, 4, 64, 64) noise pred         │    └─ to_deform_scales: → (B, 4)
                                     └─ GeoPatchCrossAttention
                                          ├─ q_proj from CLIP dim
                                          ├─ k_proj, v_proj from patch dim
                                          └─ blend_logit
```

## File Responsibilities

| File | What It Does | Modify When |
|---|---|---|
| `config/model_config.py` | Dataclasses for UNet/VAE/CLIP configs, `load_*_config()` from HF repos | Adding new config fields |
| `unet/base.py` | Pure PyTorch SD1.5 UNet port. ~860M params. | Never (matches HF weights exactly) |
| `unet/base_simplex.py` | `SD15UNetSimplex` — adds geo_prior before cross-attn. `SimplexConfig`. | Changing simplex architecture |
| `unet/base_geovocab.py` | `SD15UNetGeoVocab` — extends simplex with geovocab conditioning | Changing how geo features integrate |
| `conditioner/geovocab_conditioner.py` | `GeoVocabConditioner`, `GeoFeatureExtractor`, `GeoVocabConfig` | Changing conditioning paths |
| `text_encoder/base.py` | Pure PyTorch CLIP text encoder port | Never |
| `vae/base_vae.py` | Pure PyTorch SD1.5 VAE port | Never |
| `pipeline.py` | `Pipeline` container — loads all components, provides `encode_prompt()`, `generate()`, `train_geo()` | Adding new pipeline features |
| `trainer.py` | `Trainer` + `TrainConfig` — rectified flow training loop | Changing training procedure |
| `generate.py` | Inference / sampling with CFG | Changing sampling |
| `analyze.py` | Pre-training model analysis | Analysis tooling |
| `analyze_post.py` | Post-training analysis | Analysis tooling |

## Key Conventions

### Weight Loading
- SD1.5 weights load via `load_pretrained()` with `strict=False`
- Geometric layers (`geo_prior.*`, `geo_loss.*`, `geo_conditioner.*`) are expected missing
- Any OTHER missing/unexpected keys are real bugs — warn loudly

### Precision
- UNet/CLIP/VAE run in fp16
- `geo_prior` runs in **fp32** (CM determinant needs it) — auto-cast in forward
- `geo_conditioner` can run fp16 or fp32 (no determinant computation)
- Trainer uses `torch.amp.autocast` with manual fp32 sections

### Freezing
- `freeze_unet()` / `freeze_for_geovocab_training()` — only geo params get gradients
- Patch-maker and TextVAE are always frozen (loaded in eval mode, requires_grad=False)
- CLIP and VAE are always frozen

### Pre-extraction
Training data is pre-encoded to avoid running CLIP/VAE every step:
```python
{
    "latent": (4, 64, 64),               # VAE-encoded, scaled by 0.18215
    "encoder_hidden_states": (77, 768),    # CLIP last hidden state
    "gate_vectors": (64, 17),              # Optional, for geovocab
    "patch_features": (64, 128),           # Optional, for geovocab
}
```

### Timestep Handling
- `t_continuous` (float [0,1]) is the authoritative timestep for geo_prior
- `timestep` (float, t*1000) is passed to UNet's sinusoidal embedding
- During training, `t_continuous` comes from the sampler
- During inference, derive from step index: `t = step / num_steps`
- **Never quantize to int** — this is flow matching, not DDPM

## Common Tasks

### Adding a new conditioning path
1. Add config fields to `GeoVocabConfig`
2. Add module to `GeoVocabConditioner.forward()` output dict
3. Wire into `SD15UNetGeoVocab.forward()` before the `geo_prior` call
4. Add to pre-extraction pipeline if needed

### Changing simplex geometry
1. Modify `SimplexConfig` (k, edim, num_layers)
2. Template generation uses `geovocab2.SimplexFactory` — needs `pip install git+https://github.com/AbstractEyes/lattice_vocabulary.git`
3. If k changes, `num_vertices` and `num_edges` auto-derive
4. CM validation adapts automatically to k

### Adding new gate dimensions to patch-maker
1. Update `geovocab-patch-maker/config.json` with new dims
2. Update `geovocab-patch-maker/geometric_model.py` — add head + update constants
3. Update `GeoVocabConfig.gate_dim` here
4. Retrain patch-maker, then retrain conditioner

### Debugging geometry
- `model.get_geometry_stats()` after forward pass — check `vol_sq`, `deform_scale`, `entropy` per layer
- Entropy should decrease through layers (coarse → fine)
- `deform_scale` should stay in [0.05, 0.5] (clamped)
- `blend` near 0.5 at init, should move during training
- If CM validity loss is high, deformation is too aggressive

## External Dependencies

### HuggingFace Repos
| Repo | What | Used By |
|---|---|---|
| `sd-legacy/stable-diffusion-v1-5` | UNet + CLIP + VAE weights | `pipeline.py` |
| `AbstractPhil/geovocab-patch-maker` | Geometric analyzer (config.json + model.pt) | `geovocab_conditioner.py` |
| `AbstractPhil/geovae-proto` | TextVAE weights (optional, for text→patch extraction) | `GeoFeatureExtractor` |

### Python Packages
```
torch >= 2.0
safetensors
huggingface_hub
transformers (for tokenizer only)
```

Optional for simplex template generation:
```
pip install git+https://github.com/AbstractEyes/lattice_vocabulary.git
```

## What NOT To Do

- **Don't modify `unet/base.py`** — it must match HF safetensors exactly
- **Don't quantize timesteps** to integers during training
- **Don't run CM determinant in fp16** — numerical explosion
- **Don't freeze geo_prior during training** — that's the whole point
- **Don't use diffusers schedulers** — this is rectified flow, not DDPM/DDIM
- **Don't assume CLIP output dim** — always read from config (768 for SD1.5, but parametric)
- **Don't cache geo_prior output across steps** — it's timestep-conditioned, changes every step