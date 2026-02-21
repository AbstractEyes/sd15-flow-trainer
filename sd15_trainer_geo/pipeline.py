"""
SD15 KSimplex Pipeline Loader
==============================
One-call setup for loading SD1.5 with geometric attention prior.
Optionally includes GeoVocab conditioning from patch-maker features.

Usage:
    from pipeline import load_pipeline

    # Standard simplex-only
    pipe = load_pipeline()

    # With GeoVocab conditioning
    from sd15_trainer_geo.conditioner.geovocab_conditioner import GeoVocabConfig
    pipe = load_pipeline(geovocab_config=GeoVocabConfig())

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

    unet: nn.Module  # SD15UNetSimplex or SD15UNetGeoVocab
    vae: SD15VAE
    clip: CLIPTextModel
    tokenizer: Any

    simplex_config: SimplexConfig
    unet_config: UNetConfig
    vae_config: VAEConfig
    clip_config: CLIPConfig

    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # Optional geovocab conditioning
    geovocab_config: Any = None  # GeoVocabConfig or None
    geo_extractor: Any = None    # GeoFeatureExtractor or None

    @property
    def has_geovocab(self) -> bool:
        return self.geovocab_config is not None

    @property
    def geo_params(self) -> List[nn.Parameter]:
        """All geometric prior parameters."""
        return list(self.unet.geo_prior.parameters())

    @property
    def geo_conditioner_params(self) -> List[nn.Parameter]:
        """GeoVocab conditioner parameters (empty if not using geovocab)."""
        if hasattr(self.unet, "geo_conditioner"):
            return list(self.unet.geo_conditioner.parameters())
        return []

    @property
    def all_trainable_geo_params(self) -> List[nn.Parameter]:
        """All geometric parameters: prior + conditioner."""
        params = self.geo_params
        params.extend(self.geo_conditioner_params)
        return params

    @property
    def geo_param_count(self) -> int:
        return sum(p.numel() for p in self.geo_params)

    @property
    def geo_conditioner_param_count(self) -> int:
        return sum(p.numel() for p in self.geo_conditioner_params)

    @property
    def unet_param_count(self) -> int:
        return sum(p.numel() for p in self.unet.parameters()) - self.geo_param_count - self.geo_conditioner_param_count

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

    @torch.no_grad()
    def extract_geo_features(
        self,
        patches: torch.Tensor,
    ):
        """
        Extract geometric features from (8,16,16) patches via patch-maker.
        Only available when geo_extractor is loaded.

        Returns:
            gate_vectors:   (B, 64, 17)
            patch_features: (B, 64, 128)
        """
        assert self.geo_extractor is not None, (
            "GeoFeatureExtractor not loaded. Pass geovocab_config to load_pipeline() "
            "or load_geo_extractor=True."
        )
        return self.geo_extractor.extract_from_patches(patches)

    def summary(self):
        """Print pipeline summary."""
        unet_p = self.unet_param_count
        geo_p = self.geo_param_count
        cond_p = self.geo_conditioner_param_count
        clip_p = sum(p.numel() for p in self.clip.parameters()) if self.clip else 0
        vae_p = sum(p.numel() for p in self.vae.parameters()) if self.vae else 0

        print("SD15 KSimplex Pipeline")
        print("=" * 50)
        print(f"  UNet:       {unet_p:>14,}")
        print(f"  Geo prior:  {geo_p:>14,}")
        if cond_p > 0:
            print(f"  Geo cond:   {cond_p:>14,}")
        print(f"  CLIP:       {clip_p:>14,}")
        print(f"  VAE:        {vae_p:>14,}")
        print(f"  Total:      {unet_p + geo_p + cond_p + clip_p + vae_p:>14,}")
        print(f"\n  Simplex: k={self.simplex_config.k}, edim={self.simplex_config.edim}, "
              f"layers={self.simplex_config.num_layers}")
        if self.has_geovocab:
            print(f"  GeoVocab: gate_dim={self.geovocab_config.gate_dim}, "
                  f"patch_feat_dim={self.geovocab_config.patch_feat_dim}, "
                  f"cross_attn={'on' if self.geovocab_config.cross_attn_enabled else 'off'}")
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

    def train_geo(
        self,
        dataset,
        config=None,
        callbacks=None,
        resume_from: Optional[str] = None,
    ):
        """
        Train the geometric prior with rectified flow matching.

        Args:
            dataset:      LatentDataset or any Dataset returning
                          {"latent": ..., "encoder_hidden_states": ...}
                          Optionally with "gate_vectors" and "patch_features"
                          for geovocab conditioning.
            config:       TrainConfig (uses defaults if None)
            callbacks:    Optional list of fn(trainer, step, logs)
            resume_from:  Path to checkpoint to resume from

        Returns:
            Trainer instance (for inspection / continued training)
        """
        from .trainer import TrainConfig, Trainer

        if config is None:
            config = TrainConfig()

        trainer = Trainer(self, config)

        if resume_from is not None:
            trainer.load_checkpoint(resume_from)

        trainer.fit(dataset, callbacks=callbacks)
        return trainer


# =============================================================================
# Constants
# =============================================================================

GEO_WEIGHTS_FILENAME = "geo_prior.safetensors"
GEO_CONDITIONER_WEIGHTS_FILENAME = "geo_conditioner.safetensors"
GEO_CONFIG_FILENAME = "simplex_config.json"
GEOVOCAB_CONFIG_FILENAME = "geovocab_config.json"


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


def _geovocab_config_to_dict(cfg) -> Dict[str, Any]:
    """Serialize GeoVocabConfig to JSON-safe dict."""
    from dataclasses import asdict
    return asdict(cfg)


def _dict_to_geovocab_config(d: Dict[str, Any]):
    """Reconstruct GeoVocabConfig from dict."""
    from .conditioner.geovocab_conditioner import GeoVocabConfig
    return GeoVocabConfig(**{k: v for k, v in d.items() if hasattr(GeoVocabConfig, k)})


# =============================================================================
# Loaders
# =============================================================================

def load_pipeline(
    repo_id: str = "sd-legacy/stable-diffusion-v1-5",
    simplex_config: Optional[SimplexConfig] = None,
    geovocab_config=None,
    weight_paths: Optional[WeightPaths] = None,
    geo_repo_id: Optional[str] = None,
    geo_revision: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    load_vae: bool = True,
    load_clip: bool = True,
    load_geo_extractor: bool = False,
    patch_maker_repo: str = "AbstractPhil/geovocab-patch-maker",
) -> Pipeline:
    """
    Load full SD1.5 + KSimplex pipeline in one call.

    Args:
        repo_id:           HuggingFace repo for SD1.5 weights
        simplex_config:    Geometric prior config (defaults to SimplexConfig())
        geovocab_config:   GeoVocab conditioning config. If provided, builds
                           SD15UNetGeoVocab instead of SD15UNetSimplex.
        weight_paths:      Subfolder/filename for each component
        geo_repo_id:       Optional HF repo with trained geo prior weights
        geo_revision:      Git revision for geo_repo_id
        device:            Target device
        dtype:             Model dtype (fp16 recommended)
        load_vae:          Whether to load VAE
        load_clip:         Whether to load CLIP
        load_geo_extractor: Whether to load GeoFeatureExtractor (for inference-time
                           geo extraction; not needed if pre-extracting)
        patch_maker_repo:  HF repo for patch-maker model

    Returns:
        Pipeline with all models loaded and ready
    """
    wp = weight_paths or WeightPaths()

    # If loading from hub, pull configs from repo unless overridden
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
            pass

    if geo_repo_id and geovocab_config is None:
        import json
        from huggingface_hub import hf_hub_download
        try:
            config_path = hf_hub_download(
                repo_id=geo_repo_id,
                filename=GEOVOCAB_CONFIG_FILENAME,
                revision=geo_revision,
            )
            with open(config_path) as f:
                config_dict = json.load(f)
            geovocab_config = _dict_to_geovocab_config(config_dict)
            print(f"Using GeoVocabConfig from {geo_repo_id}")
        except Exception:
            pass  # No geovocab config in repo — simplex-only

    simplex_config = simplex_config or SimplexConfig()

    # Load configs from HF
    print(f"Loading configs from {repo_id}...")
    unet_cfg = load_unet_config(repo_id)
    vae_cfg = load_vae_config(repo_id)
    clip_cfg = load_clip_config(repo_id)

    # Build UNet — choose variant based on geovocab_config
    if geovocab_config is not None:
        from .unet.base_geovocab import SD15UNetGeoVocab
        print("Building SD15UNetGeoVocab...")
        unet = SD15UNetGeoVocab(
            unet_config=unet_cfg,
            simplex_config=simplex_config,
            geovocab_config=geovocab_config,
        )
    else:
        print("Building SD15UNetSimplex...")
        unet = SD15UNetSimplex(
            unet_config=unet_cfg,
            simplex_config=simplex_config,
        )

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

    # Load GeoFeatureExtractor if requested
    geo_extractor = None
    if load_geo_extractor and geovocab_config is not None:
        from .conditioner.geovocab_conditioner import GeoFeatureExtractor
        print("Loading GeoFeatureExtractor...")
        geo_extractor = GeoFeatureExtractor(
            patch_maker_repo=patch_maker_repo,
            device=device,
        )

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
        geovocab_config=geovocab_config,
        geo_extractor=geo_extractor,
    )

    print("✓ Pipeline loaded")

    # Optionally load trained geo weights from hub
    if geo_repo_id:
        load_geo_from_hub(pipe, geo_repo_id, revision=geo_revision)

    return pipe


def load_pipeline_minimal(
    repo_id: str = "sd-legacy/stable-diffusion-v1-5",
    simplex_config: Optional[SimplexConfig] = None,
    geovocab_config=None,
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
        geovocab_config=geovocab_config,
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
    """Save geometric prior weights (and conditioner if present)."""
    state = {
        "geo_prior": pipe.unet.geo_prior.state_dict(),
        "simplex_config": _simplex_config_to_dict(pipe.simplex_config),
    }
    # Save conditioner if present
    if hasattr(pipe.unet, "geo_conditioner"):
        state["geo_conditioner"] = pipe.unet.geo_conditioner.state_dict()
        if pipe.geovocab_config is not None:
            state["geovocab_config"] = _geovocab_config_to_dict(pipe.geovocab_config)

    if extra:
        state["extra"] = extra
    torch.save(state, path)

    size_kb = sum(v.numel() * v.element_size() for v in pipe.unet.geo_prior.state_dict().values()) / 1024
    if hasattr(pipe.unet, "geo_conditioner"):
        size_kb += sum(v.numel() * v.element_size() for v in pipe.unet.geo_conditioner.state_dict().values()) / 1024
    print(f"Saved geo checkpoint: {path} ({size_kb:.0f} KB)")


def load_geo_checkpoint(pipe: Pipeline, path: str) -> Dict:
    """Load geometric prior weights (and conditioner if present) from local .pt file."""
    state = torch.load(path, map_location=pipe.device, weights_only=False)
    pipe.unet.geo_prior.load_state_dict(state["geo_prior"])

    if "geo_conditioner" in state and hasattr(pipe.unet, "geo_conditioner"):
        pipe.unet.geo_conditioner.load_state_dict(state["geo_conditioner"])
        print(f"Loaded geo checkpoint (prior + conditioner): {path}")
    else:
        print(f"Loaded geo checkpoint (prior only): {path}")

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
    cond_p = pipe.geo_conditioner_param_count
    unet_p = pipe.unet_param_count

    cond_row = ""
    if cond_p > 0:
        cond_row = f"| **Geo conditioner (trained)** | **{cond_p:,}** |\n"

    geovocab_section = ""
    if pipe.has_geovocab:
        gv = pipe.geovocab_config
        geovocab_section = f"""
## GeoVocab Conditioning

| Parameter | Value |
|-----------|-------|
| Gate dim | {gv.gate_dim} |
| Patch feat dim | {gv.patch_feat_dim} |
| Num patches | {gv.num_patches} |
| Cross-attention | {'enabled' if gv.cross_attn_enabled else 'disabled'} |
| Cross-attn heads | {gv.cross_attn_heads} |
| Blend mode | {gv.geo_blend_mode} |
"""

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
{cond_row}

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
{geovocab_section}

## Usage

```python
from sd15_trainer_geo.pipeline import load_pipeline

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
    Upload geometric prior weights (and conditioner) to HuggingFace Hub.

    Uploads:
      - geo_prior.safetensors       (prior weights)
      - geo_conditioner.safetensors (conditioner weights, if present)
      - simplex_config.json         (simplex config)
      - geovocab_config.json        (geovocab config, if present)
      - README.md                   (model card)
    """
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from safetensors.torch import save_file
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save geo prior weights
        geo_sd = pipe.unet.geo_prior.state_dict()
        save_file(geo_sd, tmpdir / GEO_WEIGHTS_FILENAME)

        # Save conditioner weights if present
        if hasattr(pipe.unet, "geo_conditioner"):
            cond_sd = pipe.unet.geo_conditioner.state_dict()
            save_file(cond_sd, tmpdir / GEO_CONDITIONER_WEIGHTS_FILENAME)

        # Save simplex config
        config_dict = _simplex_config_to_dict(pipe.simplex_config)
        config_dict["_base_repo"] = base_repo
        with open(tmpdir / GEO_CONFIG_FILENAME, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save geovocab config if present
        if pipe.geovocab_config is not None:
            with open(tmpdir / GEOVOCAB_CONFIG_FILENAME, "w") as f:
                json.dump(_geovocab_config_to_dict(pipe.geovocab_config), f, indent=2)

        # Model card
        card = _generate_model_card(pipe, repo_id, base_repo, extra)
        (tmpdir / "README.md").write_text(card)

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
    Download and load geometric prior weights (and conditioner) from Hub.
    """
    import json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # Download and load prior weights
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=GEO_WEIGHTS_FILENAME,
        revision=revision,
        token=token,
    )
    geo_sd = load_file(weights_path, device=str(pipe.device))
    pipe.unet.geo_prior.load_state_dict(geo_sd)

    # Download and load conditioner weights if present
    if hasattr(pipe.unet, "geo_conditioner"):
        try:
            cond_path = hf_hub_download(
                repo_id=repo_id,
                filename=GEO_CONDITIONER_WEIGHTS_FILENAME,
                revision=revision,
                token=token,
            )
            cond_sd = load_file(cond_path, device=str(pipe.device))
            pipe.unet.geo_conditioner.load_state_dict(cond_sd)
            print(f"✓ Loaded geo prior + conditioner from https://huggingface.co/{repo_id}")
        except Exception:
            print(f"✓ Loaded geo prior from https://huggingface.co/{repo_id} (no conditioner weights)")
    else:
        print(f"✓ Loaded geo prior from https://huggingface.co/{repo_id}")

    # Download config
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=GEO_CONFIG_FILENAME,
        revision=revision,
        token=token,
    )
    with open(config_path) as f:
        config_dict = json.load(f)

    return config_dict


# =============================================================================
# Verification
# =============================================================================

def verify_pipeline(pipe: Pipeline):
    """Run a forward + backward pass to verify everything works."""
    print("\nVerifying pipeline...")
    pipe.unet.train()

    # Build test inputs
    noise = torch.randn(1, 4, 64, 64, device=pipe.device, dtype=pipe.dtype)
    timestep = torch.tensor([500], device=pipe.device)
    enc_hs = torch.randn(1, 77, 768, device=pipe.device, dtype=pipe.dtype)

    # Build forward kwargs
    fwd_kwargs = dict(
        sample=noise,
        timestep=timestep,
        encoder_hidden_states=enc_hs,
    )

    # Add geo features if geovocab
    if pipe.has_geovocab:
        fwd_kwargs["gate_vectors"] = torch.randn(1, 64, 17, device=pipe.device)
        fwd_kwargs["patch_features"] = torch.randn(1, 64, 128, device=pipe.device)

    with torch.amp.autocast("cuda", dtype=pipe.dtype):
        pred = pipe.unet(**fwd_kwargs)
        geo_total, geo_parts = pipe.unet.compute_geometric_loss()

        target = torch.randn_like(pred)
        task_loss = nn.functional.mse_loss(pred, target)
        loss = task_loss + 0.01 * geo_total

    loss.backward()

    # Count grads
    geo_grads = sum(
        1 for p in pipe.unet.geo_prior.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_geo = sum(1 for _ in pipe.unet.geo_prior.parameters())

    cond_grads = 0
    total_cond = 0
    if hasattr(pipe.unet, "geo_conditioner"):
        cond_grads = sum(
            1 for p in pipe.unet.geo_conditioner.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        total_cond = sum(1 for _ in pipe.unet.geo_conditioner.parameters())

    stats = pipe.unet.get_geometry_stats()

    print(f"  Forward:  {noise.shape} -> {pred.shape}")
    print(f"  Task loss: {task_loss.item():.6f}")
    print(f"  Geo loss:  {geo_total.item():.6f}")
    for k, v in geo_parts.items():
        print(f"    {k}: {v.item():.6f}")
    print(f"  Gradients: {geo_grads}/{total_geo} geo prior params")
    if total_cond > 0:
        print(f"  Gradients: {cond_grads}/{total_cond} geo conditioner params")
    print(f"  Blend: {stats.get('blend', 'N/A')}")

    has_nan = any(
        torch.isnan(p.grad).any()
        for p in pipe.unet.geo_prior.parameters()
        if p.grad is not None
    )
    print(f"  NaN grads: {'⚠ YES' if has_nan else '✓ none'}")

    pipe.unet.zero_grad()
    print("✓ Pipeline verified")