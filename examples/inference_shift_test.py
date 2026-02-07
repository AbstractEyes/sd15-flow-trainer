# Verify sampler works independently — no training, just generate
import torch
from sd15_trainer_geo.pipeline import load_pipeline
from sd15_trainer_geo.generate import generate, show_images

pipe = load_pipeline(device="cuda", dtype=torch.float16)
pipe.unet.load_pretrained(
    repo_id="AbstractPhil/tinyflux-experts",
    subfolder="",
    filename="sd15-flow-lune-unet.safetensors",
)
pipe.unet.eval()

# Generate with shift=1 (baseline)
out1 = generate(pipe, ["a cat on a windowsill", "mountains at sunset"], shift=1.0, seed=42)
show_images(out1)

# Generate with shift=2 (should also work fine)
out2 = generate(pipe, ["a cat on a windowsill", "mountains at sunset"], shift=2.0, seed=42)
show_images(out2)

# Generate with shift=3 (Flux-style)
out3 = generate(pipe, ["a cat on a windowsill", "mountains at sunset"], shift=3.0, seed=42)
show_images(out3)

print("If all three produce coherent images, the sampler is correct.")
print("The training issue is synthetic data corrupting geo_prior conditioning.")