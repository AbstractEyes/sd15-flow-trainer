# SD1.5 Flow Trainer — Geometric Prior Edition

Rectified flow matching trainer for Stable Diffusion 1.5 with a KSimplex geometric cross-attention prior. Trains a lightweight geometric attention module (~4.8M params) that modulates CLIP conditioning before it enters the frozen UNet — steering generation through Cayley-Menger validated simplex geometry rather than modifying diffusion weights directly.

## What This Does

Standard SD1.5 LoRA training modifies UNet weights to shift generation behavior. This trainer takes a different approach: it inserts a **geometric attention prior** between CLIP and the UNet that reshapes how text conditioning flows into cross-attention blocks.

```
CLIP (B, 77, 768)
  → KSimplex Cross-Attention Prior (trainable, ~4.8M params)
  → Geometrically modulated conditioning (B, 77, 768)
  → Frozen SD1.5 UNet cross-attention (all 16 blocks)
  → Denoised output
```

The prior uses pentachoron (k=4 simplex) geometry with Cayley-Menger determinant validation to ensure attention patterns correspond to valid geometric configurations. Deformation is timestep-conditioned: early steps (high noise) get softer geometry, late steps (fine detail) get sharper.

### GeoVocab Extension

The `geovocab` extension adds a second conditioning path from [geovocab-patch-maker](https://huggingface.co/AbstractPhil/geovocab-patch-maker) — a pretrained geometric analyzer that extracts structural properties from latent patches. Text descriptions are projected into geometric patch space via lightweight VAEs, then the patch-maker reads gate vectors (dimensionality, curvature, topology) and learned features that condition the simplex prior.

```
Text Prompt → TextVAE → (8,16,16) patches → PatchMaker → gates + features
                                                              ↓
CLIP (B, 77, 768) ←── GeoPatchCrossAttention ←── patch features
         ↓
KSimplexCrossAttentionPrior ←── GateDeformationConditioner ←── gate vectors
         ↓
Frozen SD1.5 UNet
```

This creates a text → geometry → diffusion pipeline where the geometric structure discovered in the [Rosetta Stone experiments](https://huggingface.co/AbstractPhil/geovae-proto) directly steers image generation.

## Architecture

### UNet Variants

| Module | File | Description |
|---|---|---|
| `SD15UNet` | `unet/base.py` | Fully ported SD1.5 UNet (pure PyTorch, no diffusers dependency) |
| `SD15UNetSimplex` | `unet/base_simplex.py` | + KSimplex geometric cross-attention prior |
| `SD15UNetGeoVocab` | `unet/base_geovocab.py` | + GeoVocab conditioning from patch-maker features |

### KSimplex Prior

- **k=4** pentachoron (5 vertices, 10 edges) in 32-dimensional embedding space
- **4 stacked layers** with progressive attention sharpening (entropy decreases through depth)
- **Cayley-Menger validation** loss ensures valid simplex configurations
- **Timestep-conditioned deformation** in the proven stability zone (0.15–0.35)
- **Learnable residual blend** between original and modulated CLIP conditioning

### GeoVocab Conditioner

- **GateDeformationConditioner**: gate vectors (17d) → attention-pooled → per-layer deformation scales
- **GeoPatchCrossAttention**: CLIP tokens (77×768) attend to patch features (64×128), adding structural spatial information
- **GeoFeatureExtractor**: lazy-loads TextVAE + PatchMaker, runs text → geometric features (all frozen)

### Flow Matching

Rectified flow formulation matching the Lune trainer:

```
x_t = (1 - t) * x_0 + t * noise     # linear interpolation
v_target = noise - x_0                # velocity field
loss = MinSNR-weighted MSE(v_pred, v_target)
```

- Logit-normal timestep sampling (biases toward mid-range where learning signal is strongest)
- Configurable schedule shift (>1 biases toward high noise)
- CFG dropout (10% by default) for classifier-free guidance training
- Geometric regularization loss with warmup

## Installation

```bash
git clone https://github.com/AbstractEyes/sd15-flow-trainer.git
cd sd15-flow-trainer
pip install -e .
```

For GeoVocab conditioning, the patch-maker loads automatically from HuggingFace on first use.

## Quick Start

### Training (simplex prior only)

```python
from sd15_trainer_geo.pipeline import load_pipeline
from sd15_trainer_geo.trainer import TrainConfig, Trainer

pipe = load_pipeline(device="cuda", dtype=torch.float16)
pipe.summary()

config = TrainConfig(
    output_dir="runs/geo-v1",
    num_steps=5000,
    batch_size=4,
    base_lr=1e-4,
)

trainer = Trainer(pipe, config)
trainer.fit(dataset)
```

### Inference

```python
pipe = load_pipeline(device="cuda")
pipe.unet.load_geo_checkpoint("runs/geo-v1/checkpoint-5000.pt")

output = pipe.generate(
    "a cat sitting on a windowsill",
    num_steps=25,
    cfg_scale=7.5,
)
```

### Data Preparation

Pre-encode your dataset into latents + CLIP embeddings:

```python
# Single file format
{
    "latents": (N, 4, 64, 64),              # VAE-encoded, scaled
    "encoder_hidden_states": (N, 77, 768),   # CLIP embeddings
}

# For GeoVocab training, add:
{
    "gate_vectors": (N, 64, 17),             # from patch-maker
    "patch_features": (N, 64, 128),          # from patch-maker
}
```

## Project Structure

```
sd15-flow-trainer/
├── sd15_trainer_geo/
│   ├── config/
│   │   ├── __init__.py
│   │   └── model_config.py        # UNetConfig, VAEConfig, CLIPConfig
│   ├── unet/
│   │   ├── __init__.py
│   │   ├── base.py                # SD15UNet (pure PyTorch port)
│   │   ├── base_simplex.py        # SD15UNetSimplex + SimplexConfig
│   │   └── base_geovocab.py       # SD15UNetGeoVocab + GeoVocabConfig
│   ├── text_encoder/
│   │   ├── __init__.py
│   │   └── base.py                # CLIPTextModel
│   ├── vae/
│   │   ├── __init__.py
│   │   └── base_vae.py            # SD15VAE
│   ├── conditioner/
│   │   ├── __init__.py
│   │   └── geovocab_conditioner.py
│   ├── pipeline.py                # Pipeline loader
│   ├── trainer.py                 # Rectified flow trainer
│   ├── generate.py                # Inference / sampling
│   ├── analyze.py                 # Model analysis tools
│   └── analyze_post.py            # Post-training analysis
├── examples/
│   ├── trainer_1.py               # Basic training example
│   ├── trainer_2.py               # Advanced training example
│   ├── trainer_smoke_test.py      # Quick verification
│   ├── evaluate_unet.py           # UNet evaluation
│   └── inference_shift_test.py    # Schedule shift experiments
├── cli.py
├── api.py
├── pyproject.toml
└── requirements.txt
```

## Related Repositories

| Repo | Description |
|---|---|
| [AbstractPhil/geovocab-patch-maker](https://huggingface.co/AbstractPhil/geovocab-patch-maker) | Pretrained geometric analyzer — extracts gate vectors + patch features |
| [AbstractPhil/geovae-proto](https://huggingface.co/AbstractPhil/geovae-proto) | Rosetta Stone experiments — text→geometry VAEs proving encoder-agnostic geometric structure |
| [AbstractPhil/bert-beatrix-2048](https://huggingface.co/AbstractPhil/bert-beatrix-2048) | Categorical nomic_bert with 26 structural tokens |
| [AbstractPhil/synthetic-characters](https://huggingface.co/datasets/AbstractPhil/synthetic-characters) | 49k FLUX-generated character dataset |
| [AbstractPhil/grid-geometric-multishape](https://huggingface.co/AbstractPhil/grid-geometric-multishape) | Original geometric classifier training |

## Research Basis

- **KSimplex attention**: 89.13% FMNIST, 84.59% CIFAR-10, 69.08% CIFAR-100
- **Deformation stability zone**: 0.15–0.35, edim/k_max ≥ 8×
- **Attention sharpening**: entropy decreases through layers (coarse-to-fine)
- **Rosetta Stone**: text-derived patches produce 2.5–3.5× higher geometric discriminability than image-derived patches through the same analyzer
- **Encoder-agnostic**: T5, BERT, and Beatrix converge to ±5% of the same discriminability

## License

MIT