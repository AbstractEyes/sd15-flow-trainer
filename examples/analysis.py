# =============================================================================
# Standalone: Load trained model + run post-training analysis
# =============================================================================
# Prerequisites:
#   !pip install -q torch torchvision safetensors transformers
#   !cd /content && git clone https://github.com/AbstractPhil/sd15-trainer-geo.git
#   !cd /content/sd15-trainer-geo && pip install -e .
#   Upload analyze_post.py to /content/

import torch
from sd15_trainer_geo.pipeline import load_pipeline, load_geo_from_hub

# 1. Load base SD1.5 pipeline
pipe = load_pipeline(device="cuda", dtype=torch.float16)

# 2. Load Lune UNet weights (flow-matching fine-tuned SD1.5)
pipe.unet.load_pretrained(
    "AbstractPhil/tinyflux-experts",
    subfolder="",
    filename="sd15-flow-lune-unet.safetensors",
)

# 3. Load trained geo_prior weights from hub
load_geo_from_hub(pipe, "AbstractPhil/sd15-geoflow-object-association")

print("✓ Pipeline loaded with trained geo_prior")

# Run full analysis
from sd15_trainer_geo.analyze_post import PostTrainingAnalyzer

analyzer = PostTrainingAnalyzer(pipe)
results = analyzer.run_all(save_dir="/content/post_analysis")