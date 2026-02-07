# SD15 Rectified Flow Trainer — Smoke Test (single cell)
# !pip install -q "sd15-flow-trainer[train] @ git+https://github.com/AbstractEyes/sd15-flow-trainer.git"

import torch
from torch.utils.data import Dataset
from sd15_trainer_geo.pipeline import load_pipeline
from sd15_trainer_geo.trainer import TrainConfig, Trainer

# --- Load pipeline + swap Lune ---
pipe = load_pipeline(device="cuda", dtype=torch.float16)
pipe.unet.load_pretrained(
    repo_id="AbstractPhil/tinyflux-experts",
    subfolder="",
    filename="sd15-flow-lune-unet.safetensors",
)

# --- Synthetic dataset for smoke test ---
class SyntheticDataset(Dataset):
    def __init__(self, n=256):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return {
            "latent": torch.randn(4, 64, 64),
            "encoder_hidden_states": torch.randn(77, 768),
        }

# --- Train with shift=2 rectified flow ---
config = TrainConfig(
    num_steps=100,
    batch_size=2,
    min_lr=1e-4,
    shift=2.0,                # Rectified flow shift
    t_sample="logit_normal",  # Biases toward mid-range t
    geo_loss_weight=0.01,
    geo_loss_warmup=50,
    log_every=20,
    sample_every=50,
    save_every=100,
    warmup_steps=10,
    output_dir="/content/smoke_test_shift2",
    num_workers=0,
)
trainer = Trainer(pipe, config)
trainer.fit(SyntheticDataset(256))

# --- Check ---
first = trainer.log_history[0]["loss"]
last = trainer.log_history[-1]["loss"]
print(f"\nLoss: {first:.4f} -> {last:.4f} ({'↓' if last < first else '↑'})")
print(f"t distribution: mean={trainer.log_history[-1].get('t_mean',0):.3f} "
      f"std={trainer.log_history[-1].get('t_std',0):.3f}")

print("\n✅ Shift-2 trainer smoke test passed")