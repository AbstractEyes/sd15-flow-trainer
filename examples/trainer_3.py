# =============================================================================
# Cell 1: Install + setup
# =============================================================================
# try:
#   !pip uninstall -qy sd15-flow-trainer[dev]
# except:
#   pass
#
# !pip install "sd15-flow-trainer[dev] @ git+https://github.com/AbstractEyes/sd15-flow-trainer.git" -q

# =============================================================================
# Cell 2: Encode dataset
# =============================================================================
import torch
import gc, os

from sd15_trainer_geo.pipeline import load_pipeline
from sd15_trainer_geo.trainer import pre_encode_hf_dataset

pipe = load_pipeline(device="cuda", dtype=torch.float16)

CACHE_PATH = "/content/latent_cache/object_relations_schnell_512_2.pt"

if not os.path.exists(CACHE_PATH):
    pre_encode_hf_dataset(
        pipe,
        dataset_name="AbstractPhil/synthetic-object-relations",
        subset="schnell_512_2",
        split="train",
        image_column="image",
        prompt_column="prompt",
        output_path=CACHE_PATH,
        image_size=512,
        batch_size=16,
        max_samples=50_000,
    )
else:
    print(f"✓ Cache exists: {CACHE_PATH}")

del pipe.vae, pipe.clip
gc.collect()
torch.cuda.empty_cache()
print(f"VRAM after encoding cleanup: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# =============================================================================
# Cell 3: Load pipeline + Lune UNet + baseline samples
# =============================================================================
from sd15_trainer_geo.pipeline import load_pipeline
from sd15_trainer_geo.generate import generate, save_images, show_images

pipe = load_pipeline(device="cuda", dtype=torch.float16)

pipe.unet.load_pretrained(
    "AbstractPhil/tinyflux-experts",
    subfolder="",
    filename="sd15-flow-lune-unet.safetensors",
)

spatial_prompts = [
    "a red cup on top of a blue book",
    "a cat sitting beside a vase of flowers",
    "a small ball inside a glass bowl on a table",
    "a pair of shoes next to an umbrella by the door",
]

novel_prompts = [
    "a guitar leaning against a piano in a dim room",
    "three candles arranged in a triangle on a wooden tray",
    "a telescope pointed at the moon through an open window",
    "a child's drawing pinned to a refrigerator with magnets",
]

print("=" * 60)
print("BASELINE (before geo_prior training)")
print("=" * 60)

baseline_spatial = generate(pipe, spatial_prompts, shift=2.5, seed=42, num_steps=30)
save_images(baseline_spatial, "/content/samples_baseline_spatial")

baseline_novel = generate(pipe, novel_prompts, shift=2.5, seed=42, num_steps=30)
save_images(baseline_novel, "/content/samples_baseline_novel")

show_images(baseline_spatial)
show_images(baseline_novel)

# =============================================================================
# Cell 4: Train with geometry profiler
# =============================================================================
from sd15_trainer_geo.trainer import Trainer, TrainConfig, LatentDataset
from sd15_trainer_geo.analyze import GeometryProfiler

config = TrainConfig(
    num_steps=8333,
    batch_size=6,
    base_lr=5e-5,
    min_lr=1e-6,
    lr_scheduler="cosine",
    warmup_steps=200,
    shift=2.5,
    cfg_dropout=0.1,
    min_snr_gamma=5.0,
    geo_loss_weight=0.01,
    geo_loss_warmup=400,
    log_every=100,
    sample_every=2000,
    save_every=2000,
    sample_prompts=spatial_prompts[:2] + novel_prompts[:2],
    seed=42,
    output_dir="/content/geo_prior_object_relations",
)

dataset = LatentDataset(CACHE_PATH)
trainer = Trainer(pipe, config)

profiler = GeometryProfiler(pipe, every=100)
trainer.fit(dataset, callbacks=[profiler])

# Save profiler + log data
import json
profiler.save("/content/geo_prior_object_relations/profiler.json")
with open("/content/geo_prior_object_relations/log_history.json", "w") as f:
    json.dump(trainer.log_history, f, indent=2)

# =============================================================================
# Cell 5: Run training analysis + generate charts
# =============================================================================
from sd15_trainer_geo.analyze import analyze

summary = analyze(trainer, profiler, save_dir="/content/analysis")

# =============================================================================
# Cell 6: Compare before/after samples
# =============================================================================
print("=" * 60)
print("AFTER TRAINING — Spatial Prompts (in-distribution)")
print("=" * 60)
trained_spatial = generate(pipe, spatial_prompts, shift=2.5, seed=42, num_steps=30)
save_images(trained_spatial, "/content/samples_trained_spatial")
show_images(trained_spatial)

print("=" * 60)
print("AFTER TRAINING — Novel Prompts (out-of-distribution)")
print("=" * 60)
trained_novel = generate(pipe, novel_prompts, shift=2.5, seed=42, num_steps=30)
save_images(trained_novel, "/content/samples_trained_novel")
show_images(trained_novel)

hard_spatial = [
    "a book on top of a cup",
    "a lamp beneath a table",
    "a knife to the left of a fork on a plate",
    "a hat resting on a basketball",
    "a key inside a shoe next to the door",
    "a red apple behind a green bottle",
]
print("=" * 60)
print("HARD SPATIAL (never seen, complex relations)")
print("=" * 60)
hard_out = generate(pipe, hard_spatial, shift=2.5, seed=42, num_steps=30)
save_images(hard_out, "/content/samples_hard_spatial")
show_images(hard_out)

# =============================================================================
# Cell 7: Push weights to hub
# =============================================================================
from sd15_trainer_geo.pipeline import push_geo_to_hub

push_geo_to_hub(
    pipe,
    repo_id="AbstractPhil/sd15-geoflow-object-association",
    base_repo="sd-legacy/stable-diffusion-v1-5",
    commit_message="geo_prior v1: 1 epoch 50k object-relations schnell_512_2",
    extra={
        "dataset": "AbstractPhil/synthetic-object-relations (schnell_512_2)",
        "samples": 50000,
        "epochs": 1,
        "steps": 8333,
        "shift": 2.5,
        "base_lr": 5e-5,
        "min_snr_gamma": 5.0,
        "cfg_dropout": 0.1,
        "batch_size": 6,
        "geo_loss_weight": 0.01,
        "loss_final": trainer.log_history[-1]["loss"] if trainer.log_history else "n/a",
    },
)

# =============================================================================
# Cell 8: Upload everything to hub
# =============================================================================
from huggingface_hub import HfApi
import glob

api = HfApi()
repo_id = "AbstractPhil/sd15-geoflow-object-association"

# Analysis charts + data
for f in glob.glob("/content/analysis/*.png") + glob.glob("/content/analysis/*.json"):
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f"analysis/{os.path.basename(f)}",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"✓ {os.path.basename(f)}")

# Profiler + log history
for f in ["/content/geo_prior_object_relations/profiler.json",
          "/content/geo_prior_object_relations/log_history.json"]:
    if os.path.exists(f):
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f"analysis/{os.path.basename(f)}",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"✓ {os.path.basename(f)}")

# Sample images
sample_dirs = {
    "samples/baseline_spatial": "/content/samples_baseline_spatial",
    "samples/baseline_novel": "/content/samples_baseline_novel",
    "samples/trained_spatial": "/content/samples_trained_spatial",
    "samples/trained_novel": "/content/samples_trained_novel",
    "samples/hard_spatial": "/content/samples_hard_spatial",
}

trainer_samples = glob.glob("/content/geo_prior_object_relations/samples/step*_*.png")
for img_path in trainer_samples:
    fname = os.path.basename(img_path)
    api.upload_file(
        path_or_fileobj=img_path,
        path_in_repo=f"samples/training/{fname}",
        repo_id=repo_id,
        repo_type="model",
    )
print(f"✓ {len(trainer_samples)} training samples")

for repo_path, local_dir in sample_dirs.items():
    if not os.path.exists(local_dir):
        print(f"⏭ Skipping {repo_path}")
        continue
    images = sorted(glob.glob(f"{local_dir}/*.png"))
    for img_path in images:
        api.upload_file(
            path_or_fileobj=img_path,
            path_in_repo=f"{repo_path}/{os.path.basename(img_path)}",
            repo_id=repo_id,
            repo_type="model",
        )
    print(f"✓ {len(images)} images → {repo_path}/")

print(f"\nhttps://huggingface.co/{repo_id}")