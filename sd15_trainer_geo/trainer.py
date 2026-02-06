"""
SD15 Rectified Flow Trainer
============================
Trains the geometric prior (geo_prior) on a frozen Lune/SD1.5 UNet
using rectified flow matching.

Flow matching formulation:
    x_t = (1 - t) * x_0 + t * noise          (interpolation)
    v_target = noise - x_0                     (velocity field)
    loss = MSE(v_predicted, v_target)          (regression target)

Only geo_prior parameters receive gradients. The UNet backbone,
CLIP, and VAE are frozen.

Usage:
    from sd15_trainer_geo.pipeline import load_pipeline
    from sd15_trainer_geo.trainer import TrainConfig, Trainer

    pipe = load_pipeline(device="cuda", dtype=torch.float16)
    config = TrainConfig(output_dir="runs/geo-v1", num_steps=5000)
    trainer = Trainer(pipe, config)
    trainer.fit(dataset)

Author: AbstractPhil
License: MIT
"""

from __future__ import annotations

import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import Pipeline


# =============================================================================
# Training Config
# =============================================================================

@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""

    # --- Core ---
    num_steps: int = 10000
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # --- Flow matching ---
    t_min: float = 0.0              # Minimum timestep (0 = clean)
    t_max: float = 1.0              # Maximum timestep (1 = noise)
    t_sample: str = "logit_normal"  # "uniform" or "logit_normal"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0

    # --- Geometric loss ---
    geo_loss_weight: float = 0.01   # Weight for geometric regularization
    geo_loss_warmup: int = 500      # Steps before full geo loss weight

    # --- LR schedule ---
    lr_scheduler: str = "cosine"    # "cosine", "constant", "linear"
    warmup_steps: int = 200
    min_lr: float = 1e-6

    # --- Mixed precision ---
    use_amp: bool = True
    grad_clip: float = 1.0

    # --- Gradient accumulation ---
    grad_accum_steps: int = 1

    # --- Logging ---
    log_every: int = 50
    sample_every: int = 500
    save_every: int = 1000
    sample_prompts: List[str] = field(default_factory=lambda: [
        "a cat sitting on a windowsill",
        "a landscape painting of mountains at sunset",
        "a bowl of ramen, studio photography",
        "an astronaut riding a horse on mars",
    ])
    sample_steps: int = 25
    sample_cfg: float = 7.5

    # --- Output ---
    output_dir: str = "runs/geo-train"
    hub_repo_id: Optional[str] = None   # Push to HF Hub if set
    hub_push_every: int = 2500

    # --- Data ---
    num_workers: int = 2
    pin_memory: bool = True

    # --- Seed ---
    seed: int = 42


# =============================================================================
# Timestep Sampling
# =============================================================================

def sample_timesteps(
    batch_size: int,
    config: TrainConfig,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Sample timesteps t in [t_min, t_max] for flow matching.

    logit_normal: Sample from N(mean, std), then sigmoid.
    Biases toward middle timesteps where learning signal is strongest.

    Returns:
        t: (B,) float tensor in [t_min, t_max]
    """
    if config.t_sample == "uniform":
        t = torch.rand(batch_size, device=device)
    elif config.t_sample == "logit_normal":
        normal = torch.randn(batch_size, device=device)
        normal = normal * config.logit_normal_std + config.logit_normal_mean
        t = torch.sigmoid(normal)
    else:
        raise ValueError(f"Unknown t_sample: {config.t_sample}")

    # Scale to [t_min, t_max]
    t = config.t_min + t * (config.t_max - config.t_min)
    return t


# =============================================================================
# LR Scheduling
# =============================================================================

def get_lr(step: int, config: TrainConfig) -> float:
    """Compute learning rate at a given step."""
    # Warmup phase
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    # Post-warmup
    progress = (step - config.warmup_steps) / max(config.num_steps - config.warmup_steps, 1)
    progress = min(progress, 1.0)

    if config.lr_scheduler == "constant":
        return config.learning_rate
    elif config.lr_scheduler == "linear":
        return config.learning_rate + (config.min_lr - config.learning_rate) * progress
    elif config.lr_scheduler == "cosine":
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr + (config.learning_rate - config.min_lr) * cosine_decay
    else:
        raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# =============================================================================
# Pre-encoded Dataset Interface
# =============================================================================

class LatentDataset(Dataset):
    """
    Dataset of pre-encoded latents + CLIP embeddings.

    Expected format: directory of .pt files, each containing:
        {
            "latent": (4, 64, 64) tensor,        # VAE-encoded, scaled
            "encoder_hidden_states": (77, 768),   # CLIP embedding
        }

    Or a single .pt file with:
        {
            "latents": (N, 4, 64, 64),
            "encoder_hidden_states": (N, 77, 768),
        }
    """

    def __init__(self, path: str, device: str = "cpu"):
        self.device = device

        if os.path.isfile(path) and path.endswith(".pt"):
            # Single file with all data
            data = torch.load(path, map_location=device, weights_only=True)
            self.latents = data["latents"]
            self.encoder_hidden_states = data["encoder_hidden_states"]
        elif os.path.isdir(path):
            # Directory of individual .pt files
            files = sorted(f for f in os.listdir(path) if f.endswith(".pt"))
            latents, enc_hs = [], []
            for f in files:
                d = torch.load(os.path.join(path, f), map_location=device, weights_only=True)
                latents.append(d["latent"] if "latent" in d else d["latents"])
                enc_hs.append(
                    d["encoder_hidden_states"]
                    if "encoder_hidden_states" in d
                    else d["text_embeddings"]
                )
            self.latents = torch.stack(latents)
            self.encoder_hidden_states = torch.stack(enc_hs)
        else:
            raise ValueError(f"Invalid dataset path: {path}")

        assert self.latents.shape[0] == self.encoder_hidden_states.shape[0]
        print(f"Dataset: {len(self)} samples, latents={self.latents.shape}, "
              f"enc_hs={self.encoder_hidden_states.shape}")

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        return {
            "latent": self.latents[idx],
            "encoder_hidden_states": self.encoder_hidden_states[idx],
        }


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    Rectified flow trainer for the geometric prior.

    Freezes everything except geo_prior, trains with flow matching loss
    + geometric regularization.
    """

    def __init__(self, pipe: "Pipeline", config: TrainConfig):
        self.pipe = pipe
        self.config = config
        self.device = pipe.device
        self.dtype = pipe.dtype
        self.step = 0
        self.log_history: List[Dict[str, float]] = []

        # Freeze everything, unfreeze geo_prior
        self._freeze_backbone()

        # Optimizer — only geo_prior params
        self.optimizer = torch.optim.AdamW(
            pipe.unet.geo_prior.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )

        # GradScaler for mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

        # Output dir
        os.makedirs(config.output_dir, exist_ok=True)

        # Save config
        with open(os.path.join(config.output_dir, "train_config.json"), "w") as f:
            json.dump(asdict(config), f, indent=2)

    def _freeze_backbone(self):
        """Freeze everything except geo_prior."""
        # Freeze entire UNet
        for p in self.pipe.unet.parameters():
            p.requires_grad_(False)
        # Unfreeze geo_prior
        for p in self.pipe.unet.geo_prior.parameters():
            p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.pipe.unet.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.pipe.unet.parameters() if not p.requires_grad)
        print(f"Trainable: {trainable:,}  Frozen: {frozen:,}  "
              f"({trainable / (trainable + frozen) * 100:.2f}% trainable)")

    # -----------------------------------------------------------------
    # Flow matching core
    # -----------------------------------------------------------------

    def flow_matching_step(
        self,
        latent: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Single flow matching training step.

        x_t = (1 - t) * x_0 + t * noise
        v_target = noise - x_0
        loss = MSE(v_pred, v_target)
        """
        B = latent.shape[0]

        # Sample timesteps
        t = sample_timesteps(B, self.config, device=self.device)  # (B,)

        # Sample noise
        noise = torch.randn_like(latent)

        # Interpolate: x_t = (1-t)*x_0 + t*noise
        t_expand = t.view(B, 1, 1, 1)
        x_t = (1.0 - t_expand) * latent + t_expand * noise

        # Velocity target: v = noise - x_0
        v_target = noise - latent

        # Integer timesteps for UNet: t in [0,1] -> [0, 999]
        timesteps = (t * 1000.0).long().clamp(0, 999)

        # Forward pass
        v_pred = self.pipe.unet(x_t, timesteps, encoder_hidden_states)

        # Task loss: MSE on velocity
        task_loss = F.mse_loss(v_pred.float(), v_target.float())

        # Geometric regularization
        geo_total, geo_parts = self.pipe.unet.compute_geometric_loss()

        # Geo loss warmup
        geo_weight = self.config.geo_loss_weight
        if self.step < self.config.geo_loss_warmup:
            geo_weight *= self.step / self.config.geo_loss_warmup

        total_loss = task_loss + geo_weight * geo_total

        return {
            "loss": total_loss,
            "task_loss": task_loss,
            "geo_loss": geo_total,
            "geo_weight": torch.tensor(geo_weight),
            **{f"geo/{k}": v for k, v in geo_parts.items()},
        }

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------

    def fit(
        self,
        dataset: Dataset,
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Main training loop.

        Args:
            dataset:    LatentDataset or any Dataset returning
                        {"latent": ..., "encoder_hidden_states": ...}
            callbacks:  Optional list of fn(trainer, step, logs) called each log step
        """
        config = self.config
        callbacks = callbacks or []

        # Seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        # Dataloader (infinite)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True,
        )
        data_iter = _infinite_loader(loader)

        self.pipe.unet.train()
        start_time = time.time()
        accum_logs: Dict[str, float] = {}

        print(f"\nStarting training: {config.num_steps} steps, "
              f"bs={config.batch_size}, lr={config.learning_rate}")
        print(f"  Flow: {config.t_sample}, geo_weight={config.geo_loss_weight}, "
              f"geo_warmup={config.geo_loss_warmup}")
        print(f"  Scheduler: {config.lr_scheduler}, warmup={config.warmup_steps}")
        print(f"  Output: {config.output_dir}")
        print()

        for self.step in range(self.step, config.num_steps):
            # --- LR schedule ---
            lr = get_lr(self.step, config)
            set_lr(self.optimizer, lr)

            # --- Gradient accumulation ---
            self.optimizer.zero_grad()
            step_logs: Dict[str, float] = {}

            for micro_step in range(config.grad_accum_steps):
                batch = next(data_iter)
                latent = batch["latent"].to(self.device, self.dtype)
                enc_hs = batch["encoder_hidden_states"].to(self.device, self.dtype)

                with torch.amp.autocast("cuda", dtype=self.dtype, enabled=config.use_amp):
                    losses = self.flow_matching_step(latent, enc_hs)

                scaled_loss = losses["loss"] / config.grad_accum_steps
                self.scaler.scale(scaled_loss).backward()

                # Accumulate logs
                for k, v in losses.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    step_logs[k] = step_logs.get(k, 0.0) + val / config.grad_accum_steps

            # --- Optimizer step ---
            if config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.pipe.unet.geo_prior.parameters(),
                    config.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            step_logs["lr"] = lr

            # Accumulate for averaging
            for k, v in step_logs.items():
                accum_logs[k] = accum_logs.get(k, 0.0) + v

            # --- Logging ---
            if (self.step + 1) % config.log_every == 0:
                avg_logs = {k: v / config.log_every for k, v in accum_logs.items()}
                avg_logs["step"] = self.step + 1
                elapsed = time.time() - start_time
                avg_logs["steps_per_sec"] = (self.step + 1) / elapsed

                self._print_log(avg_logs)
                self.log_history.append(avg_logs)

                for cb in callbacks:
                    cb(self, self.step + 1, avg_logs)

                accum_logs = {}

            # --- Sampling ---
            if (self.step + 1) % config.sample_every == 0:
                self._sample(self.step + 1)

            # --- Checkpointing ---
            if (self.step + 1) % config.save_every == 0:
                self._save_checkpoint(self.step + 1)

            # --- Hub push ---
            if config.hub_repo_id and (self.step + 1) % config.hub_push_every == 0:
                self._push_hub(self.step + 1)

        # Final save
        self._save_checkpoint(self.step + 1, tag="final")
        if config.hub_repo_id:
            self._push_hub(self.step + 1, tag="final")

        elapsed = time.time() - start_time
        print(f"\n✅ Training complete: {config.num_steps} steps in {elapsed:.0f}s "
              f"({config.num_steps / elapsed:.1f} steps/s)")

    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------

    def _print_log(self, logs: Dict[str, float]):
        parts = [
            f"step {int(logs['step']):>6d}",
            f"loss={logs.get('loss', 0):.4f}",
            f"task={logs.get('task_loss', 0):.4f}",
            f"geo={logs.get('geo_loss', 0):.6f}",
            f"lr={logs.get('lr', 0):.2e}",
            f"spd={logs.get('steps_per_sec', 0):.1f}it/s",
        ]
        print(" | ".join(parts))

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _sample(self, step: int):
        """Generate sample images during training."""
        from .generate import generate, save_images

        self.pipe.unet.eval()

        sample_dir = os.path.join(self.config.output_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)

        out = generate(
            self.pipe,
            self.config.sample_prompts,
            num_steps=self.config.sample_steps,
            cfg_scale=self.config.sample_cfg,
            seed=self.config.seed,
        )
        save_images(
            out,
            path_template=os.path.join(sample_dir, f"step{step:06d}_{{i:02d}}.png"),
        )
        print(f"  📸 Samples saved: {sample_dir}/step{step:06d}_*.png")

        self.pipe.unet.train()

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------

    def _save_checkpoint(self, step: int, tag: Optional[str] = None):
        from .pipeline import save_geo_checkpoint

        name = f"geo_prior_{tag or f'step{step:06d}'}.pt"
        path = os.path.join(self.config.output_dir, name)
        save_geo_checkpoint(
            self.pipe, path,
            extra={
                "step": step,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "config": asdict(self.config),
            },
        )

    def load_checkpoint(self, path: str):
        """Resume training from a checkpoint."""
        from .pipeline import load_geo_checkpoint

        extra = load_geo_checkpoint(self.pipe, path)
        if "optimizer" in extra:
            self.optimizer.load_state_dict(extra["optimizer"])
        if "scaler" in extra:
            self.scaler.load_state_dict(extra["scaler"])
        if "step" in extra:
            self.step = extra["step"]
            print(f"Resuming from step {self.step}")
        return extra

    # -----------------------------------------------------------------
    # Hub push
    # -----------------------------------------------------------------

    def _push_hub(self, step: int, tag: Optional[str] = None):
        from .pipeline import push_geo_to_hub

        try:
            push_geo_to_hub(
                self.pipe,
                repo_id=self.config.hub_repo_id,
                extra={
                    "step": step,
                    "tag": tag or f"step{step}",
                    "config": asdict(self.config),
                },
            )
            print(f"  ☁️  Pushed to {self.config.hub_repo_id}")
        except Exception as e:
            print(f"  ⚠️  Hub push failed: {e}")

    # -----------------------------------------------------------------
    # Convenience: get training stats
    # -----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        geo_stats = self.pipe.unet.get_geometry_stats()
        return {
            "step": self.step,
            "lr": get_lr(self.step, self.config),
            "log_history": self.log_history[-10:],
            "geometry": geo_stats,
        }


# =============================================================================
# Helpers
# =============================================================================

def _infinite_loader(loader: DataLoader):
    """Infinite iterator over a DataLoader."""
    while True:
        yield from loader


# =============================================================================
# Pre-encode a dataset (convenience)
# =============================================================================

@torch.no_grad()
def pre_encode_dataset(
    pipe: "Pipeline",
    image_dir: str,
    prompt_file: str,
    output_path: str,
    image_size: int = 512,
    batch_size: int = 8,
):
    """
    Pre-encode images + prompts to latents + CLIP embeddings.
    Saves a single .pt file for use with LatentDataset.

    Args:
        pipe:           Loaded Pipeline with CLIP + VAE
        image_dir:      Directory of images (.png, .jpg)
        prompt_file:    Text file with one prompt per line (matched by filename sort order)
                        OR a .json file: {"filename.png": "prompt", ...}
        output_path:    Where to save the .pt file
        image_size:     Resize images to this size (square crop)
        batch_size:     Batch size for encoding
    """
    from PIL import Image
    from torchvision import transforms
    import numpy as np

    assert pipe.clip is not None, "Need CLIP for encoding"
    assert pipe.vae is not None, "Need VAE for encoding"

    # Load prompts
    if prompt_file.endswith(".json"):
        with open(prompt_file) as f:
            prompt_map = json.load(f)
    else:
        with open(prompt_file) as f:
            prompt_list = [line.strip() for line in f if line.strip()]
        prompt_map = None

    # Collect image paths
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in exts
    )

    # Match prompts
    if prompt_map:
        paired = [(f, prompt_map[f]) for f in image_files if f in prompt_map]
    else:
        paired = list(zip(image_files, prompt_list[:len(image_files)]))

    print(f"Encoding {len(paired)} image-prompt pairs...")

    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # [0,1] -> [-1,1]
    ])

    all_latents = []
    all_enc_hs = []

    for i in range(0, len(paired), batch_size):
        batch_pairs = paired[i:i + batch_size]
        images = []
        prompts = []

        for fname, prompt in batch_pairs:
            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            images.append(transform(img))
            prompts.append(prompt)

        # Stack and encode
        img_batch = torch.stack(images).to(pipe.device, pipe.dtype)
        latents = pipe.encode_image(img_batch, sample=False)
        enc_hs = pipe.encode_prompts(prompts)

        all_latents.append(latents.cpu())
        all_enc_hs.append(enc_hs.cpu())

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Encoded {min(i + batch_size, len(paired))}/{len(paired)}")

    result = {
        "latents": torch.cat(all_latents, dim=0),
        "encoder_hidden_states": torch.cat(all_enc_hs, dim=0),
    }
    torch.save(result, output_path)
    print(f"✓ Saved {result['latents'].shape[0]} samples to {output_path}")
    print(f"  Latents: {result['latents'].shape}")
    print(f"  Enc_hs:  {result['encoder_hidden_states'].shape}")