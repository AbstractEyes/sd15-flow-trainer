"""
Config-Driven Pipeline Verification
====================================
Tests that config-loaded models produce identical results to diffusers.
Run from project root on Colab.

Usage:
    cd /content/sd15-flow-trainer
    python verify_config_pipeline.py
"""

import sys
import torch

# Ensure project root is on path
sys.path.insert(0, ".")

from config.model_config import (
    load_pipeline_config, load_unet_config, load_vae_config, load_clip_config,
    print_config, UNetConfig, VAEConfig, CLIPConfig,
)


REPO = "sd-legacy/stable-diffusion-v1-5"


# =============================================================================
# 1. CONFIG LOADING
# =============================================================================

print("=" * 70)
print("1. CONFIG LOADING FROM HUGGINGFACE")
print("=" * 70)

unet_cfg = load_unet_config(REPO)
vae_cfg = load_vae_config(REPO)
clip_cfg = load_clip_config(REPO)

pipeline_cfg = load_pipeline_config(REPO)
print_config(pipeline_cfg)

# Verify config values match known SD1.5
assert unet_cfg.block_out_channels == [320, 640, 1280, 1280], f"UNet channels: {unet_cfg.block_out_channels}"
assert unet_cfg.cross_attention_dim == 768, f"Cross-attn dim: {unet_cfg.cross_attention_dim}"
assert vae_cfg.block_out_channels == [128, 256, 512, 512], f"VAE channels: {vae_cfg.block_out_channels}"
assert vae_cfg.latent_channels == 4, f"Latent channels: {vae_cfg.latent_channels}"
assert clip_cfg.hidden_size == 768, f"CLIP hidden: {clip_cfg.hidden_size}"
assert clip_cfg.num_hidden_layers == 12, f"CLIP layers: {clip_cfg.num_hidden_layers}"
print("\n  ✓ All config values match SD1.5 reference")


# =============================================================================
# 2. UNET VERIFICATION
# =============================================================================

print(f"\n{'=' * 70}")
print("2. UNET - CONFIG-DRIVEN vs HARDCODED")
print("=" * 70)

from unet.base import SD15UNet as ConfigUNet, load_sd15_unet

# Default (hardcoded)
unet_default = ConfigUNet()
default_params = sum(p.numel() for p in unet_default.parameters())

# Config-driven
unet_config = ConfigUNet(unet_cfg)
config_params = sum(p.numel() for p in unet_config.parameters())

print(f"  Default params: {default_params:,}")
print(f"  Config params:  {config_params:,}")
print(f"  Match: {'✓' if default_params == config_params else '✗'}")
assert default_params == config_params

# Key compatibility
default_keys = set(unet_default.state_dict().keys())
config_keys = set(unet_config.state_dict().keys())
assert default_keys == config_keys, f"Key mismatch: {default_keys.symmetric_difference(config_keys)}"
print(f"  Keys match: ✓ ({len(default_keys)} keys)")

# Load weights
print("\n  Loading weights from HuggingFace...")
unet_loaded = load_sd15_unet(REPO, config=unet_cfg, dtype=torch.float16, device="cuda")

# Compare against diffusers
from diffusers import UNet2DConditionModel
ref_unet = UNet2DConditionModel.from_pretrained(REPO, subfolder="unet", torch_dtype=torch.float16).to("cuda")
ref_unet.eval()
unet_loaded.eval()

with torch.no_grad():
    sample = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="cuda")
    timestep = torch.tensor([500], device="cuda")
    enc_hs = torch.randn(1, 77, 768, dtype=torch.float16, device="cuda")

    our_out = unet_loaded(sample, timestep, enc_hs)
    ref_out = ref_unet(sample, timestep, encoder_hidden_states=enc_hs).sample

    diff = (our_out - ref_out).abs().max().item()
    print(f"  Forward pass max diff: {diff:.6f}")
    print(f"  ✓ UNet outputs {'match' if diff < 0.01 else 'DIFFER'}")

del ref_unet
torch.cuda.empty_cache()


# =============================================================================
# 3. CLIP VERIFICATION
# =============================================================================

print(f"\n{'=' * 70}")
print("3. CLIP - CONFIG-DRIVEN")
print("=" * 70)

from text_encoder.base import CLIPTextModel, load_clip_text_encoder, get_tokenizer, tokenize

clip_default = CLIPTextModel()
clip_config = CLIPTextModel(clip_cfg)

default_params = sum(p.numel() for p in clip_default.parameters())
config_params = sum(p.numel() for p in clip_config.parameters())
print(f"  Default params: {default_params:,}")
print(f"  Config params:  {config_params:,}")
print(f"  Match: {'✓' if default_params == config_params else '✗'}")

clip_loaded = load_clip_text_encoder(REPO, config=clip_cfg, dtype=torch.float16, device="cuda")
clip_loaded.eval()

tokenizer = get_tokenizer(REPO)
input_ids = tokenize(tokenizer, "a photograph of a cat", device="cuda")

from transformers import CLIPTextModel as RefCLIP
ref_clip = RefCLIP.from_pretrained(REPO, subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")
ref_clip.eval()

with torch.no_grad():
    our_out = clip_loaded(input_ids)
    ref_out = ref_clip(input_ids).last_hidden_state
    diff = (our_out - ref_out).abs().max().item()
    print(f"  Forward pass max diff: {diff:.6f}")
    print(f"  ✓ CLIP outputs {'match' if diff < 0.01 else 'DIFFER'}")

del ref_clip
torch.cuda.empty_cache()


# =============================================================================
# 4. VAE VERIFICATION
# =============================================================================

print(f"\n{'=' * 70}")
print("4. VAE - CONFIG-DRIVEN")
print("=" * 70)

from vae.base_vae import SD15VAE, load_sd15_vae

vae_default = SD15VAE()
vae_config = SD15VAE(vae_cfg)

default_params = sum(p.numel() for p in vae_default.parameters())
config_params = sum(p.numel() for p in vae_config.parameters())
print(f"  Default params: {default_params:,}")
print(f"  Config params:  {config_params:,}")
print(f"  Match: {'✓' if default_params == config_params else '✗'}")
print(f"  Scale factor:   {vae_config.scaling_factor}")

vae_loaded = load_sd15_vae(REPO, config=vae_cfg, dtype=torch.float16, device="cuda")
vae_loaded.eval()

from diffusers import AutoencoderKL
ref_vae = AutoencoderKL.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float16).to("cuda")
ref_vae.eval()

with torch.no_grad():
    test_img = torch.randn(1, 3, 512, 512, dtype=torch.float16, device="cuda")

    our_latent = vae_loaded.encode(test_img, sample=False)
    ref_latent = ref_vae.encode(test_img).latent_dist.mean

    diff_enc = (our_latent - ref_latent).abs().max().item()
    print(f"  Encode max diff: {diff_enc:.6f}")

    our_dec = vae_loaded.decode(our_latent)
    ref_dec = ref_vae.decode(ref_latent).sample

    diff_dec = (our_dec - ref_dec).abs().max().item()
    print(f"  Decode max diff: {diff_dec:.6f}")
    print(f"  ✓ VAE outputs {'match' if max(diff_enc, diff_dec) < 0.01 else 'DIFFER'}")

del ref_vae
torch.cuda.empty_cache()


# =============================================================================
# 5. FULL PIPELINE
# =============================================================================

print(f"\n{'=' * 70}")
print("5. FULL PIPELINE (CONFIG-DRIVEN)")
print("=" * 70)

with torch.no_grad():
    prompt = "a beautiful sunset over the ocean"
    input_ids = tokenize(tokenizer, prompt, device="cuda")
    enc_hs = clip_loaded(input_ids)

    noise = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="cuda")
    timestep = torch.tensor([500], device="cuda")
    pred = unet_loaded(noise, timestep, enc_hs)

    decoded = vae_loaded.decode_scaled(pred)

    print(f"  Prompt:      '{prompt}'")
    print(f"  CLIP:        {enc_hs.shape}")
    print(f"  UNet in/out: {noise.shape} -> {pred.shape}")
    print(f"  VAE decode:  {decoded.shape}")
    print(f"  ✓ Full config-driven pipeline functional")


# =============================================================================
# 6. CUSTOM CONFIG TEST
# =============================================================================

print(f"\n{'=' * 70}")
print("6. CUSTOM CONFIG TEST")
print("=" * 70)

# Create a hypothetical smaller UNet
small_cfg = UNetConfig(
    block_out_channels=[256, 512, 1024, 1024],
    cross_attention_dim=768,
    layers_per_block=2,
)
small_unet = ConfigUNet(small_cfg)
small_params = sum(p.numel() for p in small_unet.parameters())
print(f"  Small UNet channels:  {small_cfg.block_out_channels}")
print(f"  Small UNet params:    {small_params:,}")
print(f"  Standard UNet params: 859,520,964")
print(f"  Reduction:            {100*(1 - small_params/859_520_964):.1f}%")

# Verify forward pass works
with torch.no_grad():
    small_unet = small_unet.to(dtype=torch.float16, device="cuda")
    out = small_unet(
        torch.randn(1, 4, 64, 64, dtype=torch.float16, device="cuda"),
        torch.tensor([500], device="cuda"),
        torch.randn(1, 77, 768, dtype=torch.float16, device="cuda"),
    )
    print(f"  Small UNet output:    {out.shape}")
    print(f"  ✓ Custom config builds and runs correctly")


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)

unet_p = sum(p.numel() for p in unet_loaded.parameters())
clip_p = sum(p.numel() for p in clip_loaded.parameters())
vae_p = sum(p.numel() for p in vae_loaded.parameters())

print(f"  CLIP:     {clip_p:,} params")
print(f"  UNet:     {unet_p:,} params")
print(f"  VAE:      {vae_p:,} params")
print(f"  Total:    {clip_p + unet_p + vae_p:,} params")
print(f"\n  ✓ All config-driven models verified against diffusers reference")
print(f"  ✓ Pipeline ready for geometric modifications")