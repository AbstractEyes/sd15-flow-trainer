# =============================================================================
# SD15 Geo Prior Training — ImageNet-Synthetic (Schnell)
# Target: L4 (24GB VRAM)
# =============================================================================
# Cell 1: Install Colab
# =============================================================================
# try:
#   !pip uninstall -qy sd15-flow-trainer[dev]
# except:
#   pass
#
# !pip install "sd15-flow-trainer[dev] @ git+https://github.com/AbstractEyes/sd15-flow-trainer.git" -q

# =============================================================================
# Cell 2: Pre-encode VAE + CLIP latents (cached to disk)
# =============================================================================
import torch
import os

CACHE_DIR = "/content/latent_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "imagenet_synthetic_flux_10k.pt")
os.makedirs(CACHE_DIR, exist_ok=True)

if os.path.exists(CACHE_FILE):
    print(f"✓ Cache exists: {CACHE_FILE}")
else:
    from sd15_trainer_geo.pipeline import load_pipeline
    from sd15_trainer_geo.trainer import pre_encode_hf_dataset

    # Load pipeline with VAE + CLIP for encoding
    pipe = load_pipeline(device="cuda", dtype=torch.float16)

    pre_encode_hf_dataset(
        pipe,
        dataset_name="AbstractPhil/imagenet-synthetic",
        subset="flux_schnell_512",
        split="train",
        image_column="image",
        prompt_column="prompt",
        output_path=CACHE_FILE,
        image_size=512,
        batch_size=16,        # L4 handles 16 for encoding
    )

    # Free VAE + CLIP memory before training
    del pipe
    torch.cuda.empty_cache()
    print("✓ Encoding complete, VRAM cleared")

# =============================================================================
# Cell 3: Load pipeline + Lune for training
# =============================================================================
from sd15_trainer_geo.pipeline import load_pipeline
from sd15_trainer_geo.trainer import TrainConfig, Trainer, LatentDataset
from sd15_trainer_geo.generate import generate, show_images, save_images

pipe = load_pipeline(device="cuda", dtype=torch.float16)
pipe.unet.load_pretrained(
    repo_id="AbstractPhil/tinyflux-experts",
    subfolder="",
    filename="sd15-flow-lune-unet.safetensors",
)

# Verify Lune generates coherently before training
print("\n--- Pre-training baseline ---")
pre_out = generate(
    pipe,
    ["a tabby cat on a windowsill",
     "mountains at sunset, landscape painting",
     "a bowl of ramen, studio photography",
     "an astronaut riding a horse on mars"],
    num_steps=25, cfg_scale=7.5, shift=2.5, seed=42,
)
save_images(pre_out, "/content/baseline_samples")
show_images(pre_out)

# =============================================================================
# Cell 4: Configure and train
# =============================================================================
dataset = LatentDataset(CACHE_FILE)

# 10k images / bs=6 = 1667 steps per epoch
# L4: bs=6 fits comfortably with frozen UNet fp16 + geo_prior fp32
config = TrainConfig(
    # Core
    num_steps=1667,           # ~1 epoch
    batch_size=6,             # L4-safe with frozen backbone
    base_lr=1e-4,             # geo_prior only — higher than full UNet LR
    weight_decay=0.01,

    # Flow matching — match Lune
    shift=2.5,
    t_sample="logit_normal",
    logit_normal_mean=0.0,
    logit_normal_std=1.0,
    t_min=0.001,
    t_max=1.0,

    # CFG dropout — critical for inference quality
    cfg_dropout=0.1,

    # Min-SNR — match Lune
    min_snr_gamma=5.0,

    # Geometric loss
    geo_loss_weight=0.01,
    geo_loss_warmup=200,

    # LR schedule
    lr_scheduler="cosine",
    warmup_steps=100,
    min_lr=1e-6,

    # Mixed precision
    use_amp=True,
    grad_clip=1.0,

    # Logging + sampling
    log_every=50,
    sample_every=500,
    save_every=500,
    sample_prompts=[
        "a tabby cat sitting on a windowsill",
        "mountains at sunset, landscape painting",
        "a bowl of ramen, studio photography",
        "an astronaut riding a horse on mars",
    ],
    sample_steps=25,
    sample_cfg=7.5,

    # Output
    output_dir="/content/geo_train_imagenet",
    hub_repo_id=None,         # Set to push checkpoints

    # Data
    num_workers=2,
    pin_memory=True,
    seed=42,
)

trainer = Trainer(pipe, config)
trainer.fit(dataset)

# =============================================================================
# Cell 5: Compare before/after
# =============================================================================
print("\n--- Post-training samples ---")
post_out = generate(
    pipe,
    ["a tabby cat on a windowsill",
     "mountains at sunset, landscape painting",
     "a bowl of ramen, studio photography",
     "an astronaut riding a horse on mars"],
    num_steps=25, cfg_scale=7.5, shift=2.5, seed=42,
)
save_images(post_out, "/content/post_train_samples")
show_images(post_out)

# Also try prompts NOT in training set
print("\n--- Novel prompts (not in training set) ---")
novel_out = generate(
    pipe,
    ["a cyberpunk cityscape at night with neon lights",
     "a golden retriever playing in autumn leaves",
     "a steampunk clocktower, detailed illustration",
     "an underwater coral reef, macro photography"],
    num_steps=25, cfg_scale=7.5, shift=2.5, seed=123,
)
save_images(novel_out, "/content/novel_samples")
show_images(novel_out)

# Print training summary
print(f"\nTraining: {len(trainer.log_history)} logged steps")
if trainer.log_history:
    first = trainer.log_history[0]
    last = trainer.log_history[-1]
    print(f"  Loss: {first['loss']:.4f} → {last['loss']:.4f}")
    print(f"  Task: {first['task_loss']:.4f} → {last['task_loss']:.4f}")
    print(f"  Geo:  {first['geo_loss']:.6f} → {last['geo_loss']:.6f}")
    print(f"  t_mean: {last.get('t_mean', 0):.3f} ± {last.get('t_std', 0):.3f}")


# Push trained geo_prior to HuggingFace Hub
# Disabled in this trainer by default, trainer 2 shows an upload example by default
# from sd15_trainer_geo.pipeline import push_geo_to_hub
#
# push_geo_to_hub(
#     pipe,
#     repo_id="AbstractPhil/sd15-rectified-geometric-matching",
#     base_repo="sd-legacy/stable-diffusion-v1-5",
#     commit_message="geo_prior v1: 1 epoch imagenet-synthetic-schnell-10k, shift=2.5",
#     extra={
#         "dataset": "AbstractPhil/imagenet-synthetic (flux_schnell_512)",
#         "samples": 10000,
#         "epochs": 1,
#         "shift": 2.5,
#         "base_lr": 1e-4,
#         "min_snr_gamma": 5.0,
#         "cfg_dropout": 0.1,
#         "batch_size": 6,
#         "loss_final": trainer.log_history[-1]["loss"] if trainer.log_history else "n/a",
#     },
# )