"""
SD15 KSimplex Pipeline Loader
==============================
One-call setup for loading SD1.5 with geometric attention prior.

Usage:
    from pipeline import load_pipeline

    pipe = load_pipeline()
    pipe.summary()

Author: AbstractPhil
License: MIT
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from .config.model_config import (
    UNetConfig, VAEConfig, CLIPConfig,
    load_unet_config, load_vae_config, load_clip_config,
)
from .unet.base_simplex import SD15UNetSimplex, SimplexConfig
from .text_encoder.base import CLIPTextModel, load_clip_text_encoder, get_tokenizer, tokenize
from .vae.base_vae import SD15VAE, load_sd15_vae


# =============================================================================
# Weight Paths
# =============================================================================

@dataclass
class WeightPaths:
    """
    Configurable subfolder/filename for each model component.
    Defaults match sd-legacy/stable-diffusion-v1-5 layout.
    """
    unet_subfolder: str = "unet"
    unet_filename: str = "diffusion_pytorch_model.safetensors"

    clip_subfolder: str = "text_encoder"
    clip_filename: str = "model.safetensors"

    vae_subfolder: str = "vae"
    vae_filename: str = "diffusion_pytorch_model.safetensors"

    tokenizer_subfolder: str = "tokenizer"


# =============================================================================
# Pipeline Container
# =============================================================================

@dataclass
class Pipeline:
    """Everything needed to run the geometric SD1.5 pipeline."""

    unet: SD15UNetSimplex
    vae: SD15VAE
    clip: CLIPTextModel
    tokenizer: Any

    simplex_config: SimplexConfig
    unet_config: UNetConfig
    vae_config: VAEConfig
    clip_config: CLIPConfig

    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    @property
    def geo_params(self) -> List[nn.Parameter]:
        """All geometric prior parameters."""
        return list(self.unet.geo_prior.parameters())

    @property
    def geo_param_count(self) -> int:
        return sum(p.numel() for p in self.unet.geo_prior.parameters())

    @property
    def unet_param_count(self) -> int:
        return sum(p.numel() for p in self.unet.parameters()) - self.geo_param_count

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize + encode a single prompt."""
        input_ids = tokenize(self.tokenizer, prompt, device=self.device)
        with torch.no_grad():
            return self.clip(input_ids)

    def encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Tokenize + encode a batch of prompts."""
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        with torch.no_grad():
            return self.clip(tokens)

    def encode_image(self, image: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Encode image to scaled latent. Input: (B,3,H,W) in [-1,1]."""
        with torch.no_grad():
            return self.vae.encode_scaled(image.to(self.device, self.dtype), sample=sample)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode scaled latent to image."""
        with torch.no_grad():
            return self.vae.decode_scaled(latent)

    def summary(self):
        """Print pipeline summary."""
        unet_p = self.unet_param_count
        geo_p = self.geo_param_count
        clip_p = sum(p.numel() for p in self.clip.parameters()) if self.clip else 0
        vae_p = sum(p.numel() for p in self.vae.parameters()) if self.vae else 0

        print("SD15 KSimplex Pipeline")
        print("=" * 50)
        print(f"  UNet:       {unet_p:>14,}")
        print(f"  Geo prior:  {geo_p:>14,}")
        print(f"  CLIP:       {clip_p:>14,}")
        print(f"  VAE:        {vae_p:>14,}")
        print(f"  Total:      {unet_p + geo_p + clip_p + vae_p:>14,}")
        print(f"\n  Simplex: k={self.simplex_config.k}, edim={self.simplex_config.edim}, "
              f"layers={self.simplex_config.num_layers}")
        print(f"  Device: {self.device}, Dtype: {self.dtype}")

    def generate(
        self,
        prompts,
        negative_prompt: str = "",
        num_steps: int = 25,
        cfg_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed=None,
        shift: float = 1.0,
    ):
        """
        Generate images from text prompts. See generate.py for full docs.

        Returns:
            GenerateOutput with .images (B,3,H,W) in [0,1] and .seed
        """
        from .generate import generate as _generate
        return _generate(
            self, prompts,
            negative_prompt=negative_prompt,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            seed=seed,
            shift=shift,
        )


# =============================================================================
# Constants
# =============================================================================

GEO_WEIGHTS_FILENAME = "geo_prior.safetensors"
GEO_CONFIG_FILENAME = "simplex_config.json"


# =============================================================================
# Helpers
# =============================================================================

def _simplex_config_to_dict(cfg: SimplexConfig) -> Dict[str, Any]:
    """Serialize SimplexConfig to JSON-safe dict."""
    return {
        "k": cfg.k,
        "edim": cfg.edim,
        "feat_dim": cfg.feat_dim,
        "num_layers": cfg.num_layers,
        "base_deformation": cfg.base_deformation,
        "learnable_deformation": cfg.learnable_deformation,
        "timestep_conditioned": cfg.timestep_conditioned,
        "num_heads": cfg.num_heads,
        "dropout": cfg.dropout,
        "cm_loss_weight": cfg.cm_loss_weight,
        "vol_consistency_weight": cfg.vol_consistency_weight,
        "residual_blend": cfg.residual_blend,
        "initial_blend": cfg.initial_blend,
    }


def _dict_to_simplex_config(d: Dict[str, Any]) -> SimplexConfig:
    """Reconstruct SimplexConfig from dict."""
    return SimplexConfig(**{k: v for k, v in d.items() if hasattr(SimplexConfig, k)})


# =============================================================================
# Loaders
# =============================================================================

def load_pipeline(
    repo_id: str = "sd-legacy/stable-diffusion-v1-5",
    simplex_config: Optional[SimplexConfig] = None,
    weight_paths: Optional[WeightPaths] = None,
    geo_repo_id: Optional[str] = None,
    geo_revision: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    load_vae: bool = True,
    load_clip: bool = True,
) -> Pipeline:
    """
    Load full SD1.5 + KSimplex pipeline in one call.

    Args:
        repo_id:         HuggingFace repo for SD1.5 weights
        simplex_config:  Geometric prior config (defaults to SimplexConfig())
        weight_paths:    Subfolder/filename for each component (defaults to SD1.5 layout)
        geo_repo_id:     Optional HF repo with trained geo prior weights
        geo_revision:    Git revision for geo_repo_id (branch/tag/hash)
        device:          Target device
        dtype:           Model dtype (fp16 recommended)
        load_vae:        Whether to load VAE (skip if pre-encoding latents)
        load_clip:       Whether to load CLIP (skip if pre-encoding embeddings)

    Returns:
        Pipeline with all models loaded and ready
    """
    wp = weight_paths or WeightPaths()

    # If loading from hub, pull simplex config from repo unless overridden
    if geo_repo_id and simplex_config is None:
        import json
        from huggingface_hub import hf_hub_download
        try:
            config_path = hf_hub_download(
                repo_id=geo_repo_id,
                filename=GEO_CONFIG_FILENAME,
                revision=geo_revision,
            )
            with open(config_path) as f:
                config_dict = json.load(f)
            simplex_config = _dict_to_simplex_config(config_dict)
            print(f"Using SimplexConfig from {geo_repo_id}")
        except Exception:
            pass  # Fall back to default

    simplex_config = simplex_config or SimplexConfig()

    # Load configs from HF
    print(f"Loading configs from {repo_id}...")
    unet_cfg = load_unet_config(repo_id)
    vae_cfg = load_vae_config(repo_id)
    clip_cfg = load_clip_config(repo_id)

    # Build UNet with geometric prior and load SD1.5 weights
    print("Building SD15UNetSimplex...")
    unet = SD15UNetSimplex(unet_config=unet_cfg, simplex_config=simplex_config)

    print(f"Loading UNet weights from {repo_id}/{wp.unet_subfolder}/{wp.unet_filename}...")
    unet.load_pretrained(
        repo_id=repo_id,
        subfolder=wp.unet_subfolder,
        filename=wp.unet_filename,
        device="cpu",
    )
    unet = unet.to(device=device, dtype=dtype)

    # Load CLIP
    clip = None
    tokenizer = None
    if load_clip:
        print(f"Loading CLIP from {repo_id}/{wp.clip_subfolder}/{wp.clip_filename}...")
        clip = load_clip_text_encoder(
            repo_id,
            subfolder=wp.clip_subfolder,
            filename=wp.clip_filename,
            config=clip_cfg,
            device=device,
            dtype=dtype,
        )
        clip.eval()
        tokenizer = get_tokenizer(repo_id, subfolder=wp.tokenizer_subfolder)

    # Load VAE
    vae = None
    if load_vae:
        print(f"Loading VAE from {repo_id}/{wp.vae_subfolder}/{wp.vae_filename}...")
        vae = load_sd15_vae(
            repo_id,
            subfolder=wp.vae_subfolder,
            filename=wp.vae_filename,
            config=vae_cfg,
            device=device,
            dtype=dtype,
        )
        vae.eval()

    pipe = Pipeline(
        unet=unet,
        vae=vae,
        clip=clip,
        tokenizer=tokenizer,
        simplex_config=simplex_config,
        unet_config=unet_cfg,
        vae_config=vae_cfg,
        clip_config=clip_cfg,
        device=device,
        dtype=dtype,
    )

    print("✓ Pipeline loaded")

    # Optionally load trained geo weights from hub
    if geo_repo_id:
        load_geo_from_hub(pipe, geo_repo_id, revision=geo_revision)

    return pipe


def load_pipeline_minimal(
    repo_id: str = "sd-legacy/stable-diffusion-v1-5",
    simplex_config: Optional[SimplexConfig] = None,
    weight_paths: Optional[WeightPaths] = None,
    geo_repo_id: Optional[str] = None,
    geo_revision: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Pipeline:
    """
    Minimal load — UNet + geo only, no VAE/CLIP.
    Use when latents and embeddings are pre-computed.
    """
    return load_pipeline(
        repo_id=repo_id,
        simplex_config=simplex_config,
        weight_paths=weight_paths,
        geo_repo_id=geo_repo_id,
        geo_revision=geo_revision,
        device=device,
        dtype=dtype,
        load_vae=False,
        load_clip=False,
    )


# =============================================================================
# Checkpoint Save/Load (Local)
# =============================================================================

def save_geo_checkpoint(pipe: Pipeline, path: str, extra: Optional[Dict] = None):
    """Save only the geometric prior weights (tiny file, .pt format)."""
    state = {
        "geo_prior": pipe.unet.geo_prior.state_dict(),
        "simplex_config": _simplex_config_to_dict(pipe.simplex_config),
    }
    if extra:
        state["extra"] = extra
    torch.save(state, path)
    size_kb = sum(v.numel() * v.element_size() for v in pipe.unet.geo_prior.state_dict().values()) / 1024
    print(f"Saved geo checkpoint: {path} ({size_kb:.0f} KB)")


def load_geo_checkpoint(pipe: Pipeline, path: str) -> Dict:
    """Load geometric prior weights from local .pt file."""
    state = torch.load(path, map_location=pipe.device, weights_only=False)
    pipe.unet.geo_prior.load_state_dict(state["geo_prior"])
    print(f"Loaded geo checkpoint: {path}")
    return state.get("extra", {})


# =============================================================================
# HuggingFace Hub Upload/Download
# =============================================================================

def _generate_model_card(
    pipe: Pipeline,
    repo_id: str,
    base_repo: str = "sd-legacy/stable-diffusion-v1-5",
    extra: Optional[Dict] = None,
) -> str:
    """Generate README.md model card for the geo prior."""
    cfg = pipe.simplex_config
    geo_p = pipe.geo_param_count
    unet_p = pipe.unet_param_count

    extra_section = ""
    if extra:
        lines = [f"- **{k}**: {v}" for k, v in extra.items()]
        extra_section = "\n## Training Info\n\n" + "\n".join(lines) + "\n"

    return f"""---
license: mit
library_name: sd15-flow-trainer
tags:
  - geometric-deep-learning
  - stable-diffusion
  - ksimplex
  - pentachoron
  - flow-matching
  - cross-attention-prior
base_model: {base_repo}
pipeline_tag: text-to-image
---

# KSimplex Geometric Attention Prior

Geometric cross-attention prior for SD1.5 using pentachoron (4-simplex) structures.

## Architecture

| Component | Params |
|-----------|--------|
| SD1.5 UNet (frozen) | {unet_p:,} |
| **Geo prior (trained)** | **{geo_p:,}** |

The geometric prior modulates CLIP encoder hidden states through
{cfg.num_layers}-layer stacked k-simplex attention before they reach
the {16} cross-attention blocks in the UNet.

## Simplex Configuration

| Parameter | Value |
|-----------|-------|
| k (simplex dim) | {cfg.k} |
| Embedding dim | {cfg.edim} |
| Feature dim | {cfg.feat_dim} |
| Stacked layers | {cfg.num_layers} |
| Attention heads | {cfg.num_heads} |
| Base deformation | {cfg.base_deformation} |
| Residual blend | {cfg.residual_blend} |
| Timestep conditioned | {cfg.timestep_conditioned} |

## Usage

```python
from sd15_trainer_geo.pipeline import load_pipeline, load_geo_from_hub

# Load base SD1.5 + fresh geo prior
pipe = load_pipeline()

# Load trained geo weights from this repo
load_geo_from_hub(pipe, "{repo_id}")

# Or one-shot: load base + geo in one call
pipe = load_pipeline(geo_repo_id="{repo_id}")
```
{extra_section}
## License

MIT — [AbstractPhil](https://huggingface.co/AbstractPhil)
"""


def push_geo_to_hub(
    pipe: Pipeline,
    repo_id: str,
    base_repo: str = "sd-legacy/stable-diffusion-v1-5",
    commit_message: str = "Upload geo prior checkpoint",
    private: bool = False,
    extra: Optional[Dict] = None,
    token: Optional[str] = None,
) -> str:
    """
    Upload geometric prior weights to HuggingFace Hub.

    Uploads:
      - geo_prior.safetensors   (weights)
      - simplex_config.json     (config)
      - README.md               (model card)

    Args:
        pipe:             Pipeline with trained geo prior
        repo_id:          HF repo id (e.g. "AbstractPhil/sd15-geo-prior-v1")
        base_repo:        Base SD1.5 repo used for training
        commit_message:   Git commit message
        private:          Whether repo should be private
        extra:            Extra metadata for model card (e.g. step, loss, dataset)
        token:            HF token (uses cached login if None)

    Returns:
        URL of the uploaded commit
    """
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from safetensors.torch import save_file
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Create repo if needed
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save geo prior weights as safetensors
        geo_sd = pipe.unet.geo_prior.state_dict()
        save_file(geo_sd, tmpdir / GEO_WEIGHTS_FILENAME)

        # Save simplex config as JSON
        config_dict = _simplex_config_to_dict(pipe.simplex_config)
        config_dict["_base_repo"] = base_repo
        with open(tmpdir / GEO_CONFIG_FILENAME, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Generate model card
        card = _generate_model_card(pipe, repo_id, base_repo, extra)
        (tmpdir / "README.md").write_text(card)

        # Upload all in single commit
        url = api.upload_folder(
            repo_id=repo_id,
            folder_path=str(tmpdir),
            commit_message=commit_message,
        )

    size_kb = sum(v.numel() * v.element_size() for v in geo_sd.values()) / 1024
    print(f"✓ Pushed geo prior to https://huggingface.co/{repo_id} ({size_kb:.0f} KB)")
    return url


def load_geo_from_hub(
    pipe: Pipeline,
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict:
    """
    Download and load geometric prior weights from HuggingFace Hub.

    Args:
        pipe:       Pipeline to load weights into
        repo_id:    HF repo with geo prior (e.g. "AbstractPhil/sd15-geo-prior-v1")
        revision:   Git revision (branch, tag, commit hash). Defaults to main.
        token:      HF token (uses cached login if None)

    Returns:
        simplex_config dict from the repo
    """
    import json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # Download weights
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=GEO_WEIGHTS_FILENAME,
        revision=revision,
        token=token,
    )
    geo_sd = load_file(weights_path, device=str(pipe.device))
    pipe.unet.geo_prior.load_state_dict(geo_sd)

    # Download config
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=GEO_CONFIG_FILENAME,
        revision=revision,
        token=token,
    )
    with open(config_path) as f:
        config_dict = json.load(f)

    print(f"✓ Loaded geo prior from https://huggingface.co/{repo_id}")
    return config_dict


# =============================================================================
# Verification
# =============================================================================

def verify_pipeline(pipe: Pipeline):
    """Run a forward + backward pass to verify everything works."""
    print("\nVerifying pipeline...")
    pipe.unet.train()

    with torch.amp.autocast("cuda", dtype=pipe.dtype):
        noise = torch.randn(1, 4, 64, 64, device=pipe.device, dtype=pipe.dtype)
        timestep = torch.tensor([500], device=pipe.device)
        enc_hs = torch.randn(1, 77, 768, device=pipe.device, dtype=pipe.dtype)

        pred = pipe.unet(noise, timestep, enc_hs)
        geo_total, geo_parts = pipe.unet.compute_geometric_loss()

        target = torch.randn_like(pred)
        task_loss = nn.functional.mse_loss(pred, target)
        loss = task_loss + 0.01 * geo_total

    loss.backward()

    geo_grads = sum(
        1 for p in pipe.unet.geo_prior.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_geo = sum(1 for _ in pipe.unet.geo_prior.parameters())
    stats = pipe.unet.get_geometry_stats()

    print(f"  Forward:  {noise.shape} -> {pred.shape}")
    print(f"  Task loss: {task_loss.item():.6f}")
    print(f"  Geo loss:  {geo_total.item():.6f}")
    for k, v in geo_parts.items():
        print(f"    {k}: {v.item():.6f}")
    print(f"  Gradients: {geo_grads}/{total_geo} geo params")
    print(f"  Blend: {stats.get('blend', 'N/A')}")

    has_nan = any(
        torch.isnan(p.grad).any()
        for p in pipe.unet.geo_prior.parameters()
        if p.grad is not None
    )
    print(f"  NaN grads: {'⚠ YES' if has_nan else '✓ none'}")

    pipe.unet.zero_grad()
    print("✓ Pipeline verified")