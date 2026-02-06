"""
SD15 Flow Matching Generation
==============================
Euler ODE sampler for rectified flow models (Lune, etc.).

Usage:
    from sd15_trainer_geo.pipeline import load_pipeline
    from sd15_trainer_geo.generate import generate, make_sigmas

    pipe = load_pipeline(device="cuda", dtype=torch.float16)
    images = generate(pipe, "a cat on a windowsill")
    images = generate(pipe, ["prompt1", "prompt2"], num_steps=30, cfg_scale=7.5)

Author: AbstractPhil
License: MIT
"""

from __future__ import annotations

import torch
from typing import Optional, Union, List, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .pipeline import Pipeline


# =============================================================================
# Sigma / Timestep Schedules
# =============================================================================

def make_sigmas(
    num_steps: int,
    shift: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Create sigma schedule for flow matching.

    For rectified flow: sigma goes from 1.0 (pure noise) to 0.0 (clean).
    shift > 1 biases toward higher noise levels (useful for high-res).

    Returns:
        sigmas: (num_steps + 1,) tensor from 1.0 to 0.0
    """
    # Linear spacing in [0, 1], then apply shift
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    if shift != 1.0:
        t = t * shift / (1.0 + (shift - 1.0) * t)
    return t


def sigmas_to_timesteps(sigmas: torch.Tensor, num_train_timesteps: int = 1000) -> torch.Tensor:
    """Convert sigma values to integer timesteps [0, 999]."""
    return (sigmas * num_train_timesteps).long().clamp(0, num_train_timesteps - 1)


# =============================================================================
# Generation Output
# =============================================================================

@dataclass
class GenerateOutput:
    """Output from generate()."""
    images: torch.Tensor       # (B, 3, H, W) decoded images in [0, 1]
    latents: torch.Tensor      # (B, 4, H/8, W/8) final latents
    prompts: List[str]         # Input prompts
    seed: int                  # Seed used


# =============================================================================
# Core Sampler
# =============================================================================

@torch.no_grad()
def euler_sample(
    unet: torch.nn.Module,
    latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    sigmas: torch.Tensor,
    encoder_hidden_states_uncond: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """
    Euler ODE integration for flow matching.

    Flow matching formulation:
        x_t = (1 - sigma) * x_0 + sigma * noise
        v_predicted = model(x_t, t)
        x_{t-1} = x_t + (sigma_{i+1} - sigma_i) * v_predicted

    Args:
        unet:                           UNet model (SD15UNetSimplex)
        latents:                        (B, 4, H/8, W/8) initial noise
        encoder_hidden_states:          (B, 77, 768) conditional CLIP embeddings
        sigmas:                         (num_steps+1,) noise schedule from 1.0 to 0.0
        encoder_hidden_states_uncond:   (B, 77, 768) unconditional embeddings for CFG
        cfg_scale:                      Classifier-free guidance scale (1.0 = no guidance)

    Returns:
        Denoised latents (B, 4, H/8, W/8)
    """
    do_cfg = cfg_scale > 1.0 and encoder_hidden_states_uncond is not None
    B = latents.shape[0]

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma  # Negative (going from noise to clean)

        # Integer timestep for UNet
        timestep = (sigma * 1000.0).long().clamp(0, 999)
        timestep = timestep.expand(B).to(latents.device)

        if do_cfg:
            # Batched CFG: concat cond + uncond
            latent_input = torch.cat([latents, latents], dim=0)
            t_input = torch.cat([timestep, timestep], dim=0)
            enc_input = torch.cat([encoder_hidden_states, encoder_hidden_states_uncond], dim=0)

            v_pred = unet(latent_input, t_input, enc_input)
            v_cond, v_uncond = v_pred.chunk(2, dim=0)
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v_pred = unet(latents, timestep, encoder_hidden_states)

        # Euler step
        latents = latents + dt * v_pred

    return latents


# =============================================================================
# High-Level Generate
# =============================================================================

def generate(
    pipe: "Pipeline",
    prompts: Union[str, List[str]],
    negative_prompt: str = "",
    num_steps: int = 25,
    cfg_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
    shift: float = 1.0,
    return_latents: bool = False,
) -> GenerateOutput:
    """
    Generate images from text prompts using flow matching Euler sampling.

    Args:
        pipe:             Loaded Pipeline (needs unet, clip, vae, tokenizer)
        prompts:          Single prompt string or list of prompts
        negative_prompt:  Negative prompt for CFG (shared across batch)
        num_steps:        Number of sampling steps
        cfg_scale:        Classifier-free guidance scale (1.0 = no guidance)
        width:            Output image width (must be divisible by 8)
        height:           Output image height (must be divisible by 8)
        seed:             Random seed (None = random)
        shift:            Sigma schedule shift (1.0 = linear)
        return_latents:   If True, also return raw latents

    Returns:
        GenerateOutput with images (B,3,H,W) in [0,1] range
    """
    assert pipe.clip is not None, "Pipeline needs CLIP loaded for generation"
    assert pipe.vae is not None, "Pipeline needs VAE loaded for generation"

    # Normalize to list
    if isinstance(prompts, str):
        prompts = [prompts]
    B = len(prompts)

    # Seed
    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Encode prompts
    pipe.unet.eval()
    enc_hs = pipe.encode_prompts(prompts)

    # Encode negative prompt for CFG
    enc_uncond = None
    if cfg_scale > 1.0:
        neg_prompts = [negative_prompt] * B
        enc_uncond = pipe.encode_prompts(neg_prompts)

    # Initial noise
    latent_h = height // 8
    latent_w = width // 8
    latents = torch.randn(
        B, 4, latent_h, latent_w,
        generator=generator,
        dtype=pipe.dtype,
        device="cpu",
    ).to(pipe.device)

    # Sigma schedule
    sigmas = make_sigmas(num_steps, shift=shift, device=pipe.device)

    # Sample
    latents = euler_sample(
        unet=pipe.unet,
        latents=latents,
        encoder_hidden_states=enc_hs,
        sigmas=sigmas,
        encoder_hidden_states_uncond=enc_uncond,
        cfg_scale=cfg_scale,
    )

    # Decode
    images = pipe.decode_latent(latents)
    images = images.clamp(-1, 1) * 0.5 + 0.5  # [-1,1] -> [0,1]

    return GenerateOutput(
        images=images,
        latents=latents if return_latents else latents.detach(),
        prompts=prompts,
        seed=seed,
    )


# =============================================================================
# Convenience: save images
# =============================================================================

def save_images(
    output: GenerateOutput,
    path_template: str = "gen_{i:03d}.png",
):
    """
    Save generated images to disk.

    Args:
        output:         GenerateOutput from generate()
        path_template:  Path with {i} placeholder for image index
    """
    from PIL import Image
    import numpy as np

    for i in range(output.images.shape[0]):
        img = output.images[i].cpu().float().clamp(0, 1)
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        path = path_template.format(i=i)
        Image.fromarray(img).save(path)
        print(f"  Saved: {path}")


def show_images(output: GenerateOutput, cols: int = 4):
    """Display generated images in a matplotlib grid."""
    import matplotlib.pyplot as plt

    B = output.images.shape[0]
    rows = (B + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.axis("off")
        if i < B:
            img = output.images[i].cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(img)
            if i < len(output.prompts):
                ax.set_title(output.prompts[i][:40], fontsize=8)

    fig.suptitle(f"seed={output.seed}", fontsize=10)
    plt.tight_layout()
    plt.show()