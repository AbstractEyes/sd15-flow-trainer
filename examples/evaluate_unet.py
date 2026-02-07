# SD15 Flow Lune — Generate Test (single cell)
# !pip install -q "sd15-flow-trainer[train] @ git+https://github.com/AbstractEyes/sd15-flow-trainer.git"

import torch
from sd15_trainer_geo.pipeline import load_pipeline
from sd15_trainer_geo.generate import generate, show_images, save_images

# --- Load pipeline + swap in Lune UNet ---
pipe = load_pipeline(device="cuda", dtype=torch.float16)
pipe.unet.load_pretrained(
    repo_id="AbstractPhil/tinyflux-experts",
    subfolder="",
    filename="sd15-flow-lune-unet.safetensors",
)
pipe.unet.eval()

# --- Generate single image ---
out = generate(pipe, "a cat sitting on a windowsill, oil painting", seed=42)
show_images(out)

# --- Generate batch ---
out = generate(
    pipe,
    ["a red sports car on a mountain road",
     "a cyberpunk cityscape at night",
     "a bowl of ramen, studio photography",
     "an astronaut riding a horse on mars"],
    num_steps=30,
    cfg_scale=7.5,
    seed=123,
)
show_images(out, cols=4)

# --- Or via pipe.generate() convenience ---
out = pipe.generate("a peaceful japanese garden, watercolor", seed=777)
show_images(out)

print(f"\n✅ Generation complete (seed={out.seed})")