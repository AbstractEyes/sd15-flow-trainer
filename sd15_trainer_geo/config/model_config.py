"""
Model Configuration - Parses Diffusers config.json
====================================================
Loads standard diffusers/transformers config.json files and provides
typed dataclasses for dynamic model construction.

Supports loading from:
  - HuggingFace repo (auto-downloads config.json)
  - Local directory
  - Dict / JSON string
  - Manual construction

Author: AbstractPhil
License: MIT
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Union, Dict, Any


# =============================================================================
# UNet Config
# =============================================================================

@dataclass
class UNetConfig:
    """
    Mirrors diffusers UNet2DConditionModel config.json.
    All fields have SD1.5 defaults.
    """
    # Architecture
    sample_size: int = 64
    in_channels: int = 4
    out_channels: int = 4
    center_input_sample: bool = False

    # Block structure
    down_block_types: List[str] = field(default_factory=lambda: [
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ])
    up_block_types: List[str] = field(default_factory=lambda: [
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ])
    block_out_channels: List[int] = field(default_factory=lambda: [320, 640, 1280, 1280])
    layers_per_block: int = 2

    # Attention
    cross_attention_dim: int = 768
    attention_head_dim: Union[int, List[int]] = 8
    num_attention_heads: Optional[Union[int, List[int]]] = None

    # Normalization
    norm_num_groups: int = 32
    norm_eps: float = 1e-5

    # Activation
    act_fn: str = "silu"

    # Time embedding
    freq_shift: int = 0

    # Misc
    mid_block_type: str = "UNetMidBlock2DCrossAttn"
    only_cross_attention: bool = False
    dual_cross_attention: bool = False
    use_linear_projection: bool = False
    class_embed_type: Optional[str] = None
    num_class_embeds: Optional[int] = None
    addition_embed_type: Optional[str] = None
    addition_time_embed_dim: Optional[int] = None
    projection_class_embeddings_input_dim: Optional[int] = None
    resnet_time_scale_shift: str = "default"
    transformer_layers_per_block: Union[int, List[int]] = 1
    mid_block_only_cross_attention: Optional[bool] = None

    # Dropout
    dropout: float = 0.0

    @property
    def num_down_blocks(self) -> int:
        return len(self.down_block_types)

    @property
    def num_up_blocks(self) -> int:
        return len(self.up_block_types)

    @property
    def temb_channels(self) -> int:
        return self.block_out_channels[0] * 4

    def get_attention_heads(self, block_idx: int) -> int:
        """Get number of attention heads for a given block index."""
        if self.num_attention_heads is not None:
            if isinstance(self.num_attention_heads, list):
                return self.num_attention_heads[block_idx]
            return self.num_attention_heads
        if isinstance(self.attention_head_dim, list):
            head_dim = self.attention_head_dim[block_idx]
            return self.block_out_channels[block_idx] // head_dim
        return self.attention_head_dim  # In SD1.5 this is actually num_heads=8

    def get_transformer_layers(self, block_idx: int) -> int:
        """Get number of transformer layers for a given block."""
        if isinstance(self.transformer_layers_per_block, list):
            return self.transformer_layers_per_block[block_idx]
        return self.transformer_layers_per_block

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# VAE Config
# =============================================================================

@dataclass
class VAEConfig:
    """
    Mirrors diffusers AutoencoderKL config.json.
    All fields have SD1.5 defaults.
    """
    # Architecture
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4

    # Block structure
    down_block_types: List[str] = field(default_factory=lambda: [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ])
    up_block_types: List[str] = field(default_factory=lambda: [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ])
    block_out_channels: List[int] = field(default_factory=lambda: [128, 256, 512, 512])
    layers_per_block: int = 2

    # Normalization
    norm_num_groups: int = 32
    norm_eps: float = 1e-6

    # Activation
    act_fn: str = "silu"

    # Scaling
    scaling_factor: float = 0.18215
    shift_factor: Optional[float] = None

    # Architecture options
    mid_block_add_attention: bool = True
    sample_size: int = 512

    @property
    def num_down_blocks(self) -> int:
        return len(self.down_block_types)

    @property
    def num_up_blocks(self) -> int:
        return len(self.up_block_types)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# CLIP Text Encoder Config
# =============================================================================

@dataclass
class CLIPConfig:
    """
    Mirrors transformers CLIPTextModel config.json.
    All fields have SD1.5 CLIP defaults (clip-vit-large-patch14).
    """
    vocab_size: int = 49408
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    projection_dim: int = 768

    # Transformers-specific fields we might see in config.json
    model_type: str = "clip_text_model"
    bos_token_id: int = 0
    eos_token_id: int = 2
    pad_token_id: int = 1

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Pipeline Config (combines all three)
# =============================================================================

@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    unet: UNetConfig = field(default_factory=UNetConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)

    # Pipeline-level settings
    prediction_type: str = "v_prediction"  # "epsilon", "v_prediction", "sample"
    scheduler_type: str = "flow_matching"  # "flow_matching", "ddpm", "ddim"

    # Flow matching settings
    shift: float = 2.0
    num_inference_steps: int = 20

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unet": self.unet.to_dict(),
            "vae": self.vae.to_dict(),
            "clip": self.clip.to_dict(),
            "prediction_type": self.prediction_type,
            "scheduler_type": self.scheduler_type,
            "shift": self.shift,
            "num_inference_steps": self.num_inference_steps,
        }


# =============================================================================
# Config Loading Utilities
# =============================================================================

def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def _filter_known_fields(data: Dict[str, Any], dataclass_type) -> Dict[str, Any]:
    """Filter dict to only include fields defined in the dataclass."""
    known = {f.name for f in dataclass_type.__dataclass_fields__.values()}
    return {k: v for k, v in data.items() if k in known}


def load_unet_config(source: Union[str, Dict[str, Any]]) -> UNetConfig:
    """
    Load UNet config from various sources.

    Args:
        source: One of:
            - HF repo ID: "sd-legacy/stable-diffusion-v1-5" (downloads unet/config.json)
            - Local path to directory containing config.json
            - Local path to config.json file
            - Dict with config values

    Returns:
        UNetConfig
    """
    if isinstance(source, dict):
        return UNetConfig(**_filter_known_fields(source, UNetConfig))

    if os.path.isfile(source):
        data = _load_json(source)
        return UNetConfig(**_filter_known_fields(data, UNetConfig))

    if os.path.isdir(source):
        config_path = os.path.join(source, "config.json")
        if os.path.exists(config_path):
            data = _load_json(config_path)
            return UNetConfig(**_filter_known_fields(data, UNetConfig))
        # Try unet subfolder
        config_path = os.path.join(source, "unet", "config.json")
        data = _load_json(config_path)
        return UNetConfig(**_filter_known_fields(data, UNetConfig))

    # Assume HuggingFace repo ID
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=source, subfolder="unet", filename="config.json")
    data = _load_json(path)
    return UNetConfig(**_filter_known_fields(data, UNetConfig))


def load_vae_config(source: Union[str, Dict[str, Any]]) -> VAEConfig:
    """
    Load VAE config. Same source options as load_unet_config.
    """
    if isinstance(source, dict):
        return VAEConfig(**_filter_known_fields(source, VAEConfig))

    if os.path.isfile(source):
        data = _load_json(source)
        return VAEConfig(**_filter_known_fields(data, VAEConfig))

    if os.path.isdir(source):
        config_path = os.path.join(source, "config.json")
        if os.path.exists(config_path):
            data = _load_json(config_path)
            return VAEConfig(**_filter_known_fields(data, VAEConfig))
        config_path = os.path.join(source, "vae", "config.json")
        data = _load_json(config_path)
        return VAEConfig(**_filter_known_fields(data, VAEConfig))

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=source, subfolder="vae", filename="config.json")
    data = _load_json(path)
    return VAEConfig(**_filter_known_fields(data, VAEConfig))


def load_clip_config(source: Union[str, Dict[str, Any]]) -> CLIPConfig:
    """
    Load CLIP text encoder config. Same source options as load_unet_config.
    """
    if isinstance(source, dict):
        return CLIPConfig(**_filter_known_fields(source, CLIPConfig))

    if os.path.isfile(source):
        data = _load_json(source)
        # Transformers nests CLIP config under different keys
        if "text_config" in data:
            data = data["text_config"]
        return CLIPConfig(**_filter_known_fields(data, CLIPConfig))

    if os.path.isdir(source):
        config_path = os.path.join(source, "config.json")
        if os.path.exists(config_path):
            data = _load_json(config_path)
            if "text_config" in data:
                data = data["text_config"]
            return CLIPConfig(**_filter_known_fields(data, CLIPConfig))
        config_path = os.path.join(source, "text_encoder", "config.json")
        data = _load_json(config_path)
        if "text_config" in data:
            data = data["text_config"]
        return CLIPConfig(**_filter_known_fields(data, CLIPConfig))

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=source, subfolder="text_encoder", filename="config.json")
    data = _load_json(path)
    if "text_config" in data:
        data = data["text_config"]
    return CLIPConfig(**_filter_known_fields(data, CLIPConfig))


def load_pipeline_config(
    source: str,
    prediction_type: str = "v_prediction",
    scheduler_type: str = "flow_matching",
    shift: float = 2.0,
) -> PipelineConfig:
    """
    Load full pipeline config from a HF repo or local directory.

    Usage:
        config = load_pipeline_config("sd-legacy/stable-diffusion-v1-5")
        config = load_pipeline_config("AbstractPhil/sd15-flow-lune")
        config = load_pipeline_config("/path/to/local/model")
    """
    return PipelineConfig(
        unet=load_unet_config(source),
        vae=load_vae_config(source),
        clip=load_clip_config(source),
        prediction_type=prediction_type,
        scheduler_type=scheduler_type,
        shift=shift,
    )


def save_config(config: Union[UNetConfig, VAEConfig, CLIPConfig, PipelineConfig], path: str):
    """Save config to JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


# =============================================================================
# SD1.5 Presets
# =============================================================================

def sd15_default() -> PipelineConfig:
    """Standard SD1.5 config (epsilon prediction, DDPM)."""
    return PipelineConfig(
        prediction_type="epsilon",
        scheduler_type="ddpm",
    )


def sd15_flow_lune() -> PipelineConfig:
    """SD15 Flow-Lune config (velocity prediction, flow matching, shift=2)."""
    return PipelineConfig(
        prediction_type="v_prediction",
        scheduler_type="flow_matching",
        shift=2.0,
    )


# =============================================================================
# Pretty Print
# =============================================================================

def print_config(config: PipelineConfig):
    """Print a readable config summary."""
    print("Pipeline Configuration")
    print("=" * 60)
    print(f"  Prediction type:  {config.prediction_type}")
    print(f"  Scheduler:        {config.scheduler_type}")
    print(f"  Shift:            {config.shift}")

    u = config.unet
    print(f"\n  UNet:")
    print(f"    Channels:       {u.in_channels} -> {u.out_channels}")
    print(f"    Block channels: {u.block_out_channels}")
    print(f"    Down blocks:    {u.down_block_types}")
    print(f"    Up blocks:      {u.up_block_types}")
    print(f"    Layers/block:   {u.layers_per_block}")
    print(f"    Cross-attn dim: {u.cross_attention_dim}")
    print(f"    Attn heads:     {u.attention_head_dim}")
    print(f"    Norm groups:    {u.norm_num_groups}")

    v = config.vae
    print(f"\n  VAE:")
    print(f"    Channels:       {v.in_channels} -> {v.out_channels}")
    print(f"    Latent ch:      {v.latent_channels}")
    print(f"    Block channels: {v.block_out_channels}")
    print(f"    Layers/block:   {v.layers_per_block}")
    print(f"    Scale factor:   {v.scaling_factor}")

    c = config.clip
    print(f"\n  CLIP:")
    print(f"    Hidden dim:     {c.hidden_size}")
    print(f"    Layers:         {c.num_hidden_layers}")
    print(f"    Heads:          {c.num_attention_heads}")
    print(f"    Intermediate:   {c.intermediate_size}")
    print(f"    Vocab:          {c.vocab_size}")
    print(f"    Max tokens:     {c.max_position_embeddings}")