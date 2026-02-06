"""
CLIP Text Encoder - Config-Driven Pure PyTorch Implementation
==============================================================
Builds architecture dynamically from CLIPConfig (transformers config.json).
No diffusers/transformers dependency at runtime. State_dict keys match exactly.

Usage:
    from config.model_config import load_clip_config, CLIPConfig

    # From HuggingFace repo
    config = load_clip_config("sd-legacy/stable-diffusion-v1-5")
    clip = CLIPTextModel(config)

    # Default SD1.5
    clip = CLIPTextModel()

Author: AbstractPhil
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from ..config.model_config import CLIPConfig


# =============================================================================
# Activation
# =============================================================================

def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """OpenAI CLIP uses this instead of standard GELU."""
    return x * torch.sigmoid(1.702 * x)


ACTIVATIONS = {
    "quick_gelu": quick_gelu,
    "gelu": F.gelu,
    "silu": F.silu,
}


# =============================================================================
# CLIP Components
# =============================================================================

class CLIPTextEmbeddings(nn.Module):
    """Token + positional embeddings."""
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[-1]
        position_ids = self.position_ids[:, :seq_len]
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class CLIPAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        hidden_size = config.hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = attn_weights + causal_attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(attn_output)


class CLIPMLP(nn.Module):
    """Feed-forward with configurable activation."""
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation_fn = ACTIVATIONS.get(config.hidden_act, quick_gelu)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    """Pre-norm transformer layer."""
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """Stack of encoder layers."""
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_attention_mask)
        return hidden_states


# =============================================================================
# Inner Transformer
# =============================================================================

class CLIPTextTransformer(nn.Module):
    """Inner transformer wrapped by CLIPTextModel."""
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _build_causal_attention_mask(
        self, seq_len: int, dtype: torch.dtype, device: torch.device,
    ) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[-1]
        hidden_states = self.embeddings(input_ids)
        causal_mask = self._build_causal_attention_mask(
            seq_len, hidden_states.dtype, hidden_states.device,
        )
        hidden_states = self.encoder(hidden_states, causal_mask)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


# =============================================================================
# Main CLIP Text Model
# =============================================================================

class CLIPTextModel(nn.Module):
    """
    Config-driven CLIP text encoder.

    Input:
        input_ids: (B, max_tokens) token IDs

    Output:
        last_hidden_state: (B, max_tokens, hidden_size)
    """

    def __init__(self, config: Optional[CLIPConfig] = None):
        super().__init__()
        self.config = config or CLIPConfig()
        self.text_model = CLIPTextTransformer(self.config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_model(input_ids)


# =============================================================================
# Weight Loading
# =============================================================================

def load_clip_text_encoder(
    repo_id: str = "sd-legacy/stable-diffusion-v1-5",
    subfolder: str = "text_encoder",
    filename: str = "model.safetensors",
    config: Optional[CLIPConfig] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CLIPTextModel:
    """
    Load CLIP text encoder weights from HuggingFace.
    If config is None, loads config.json from the same repo.
    """
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from config.model_config import load_clip_config

    if config is None:
        config = load_clip_config(repo_id)

    model = CLIPTextModel(config)

    path = hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=filename)
    state_dict = load_file(path, device=device)

    # Filter out position_ids buffer
    filtered = {k: v for k, v in state_dict.items() if "position_ids" not in k}
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    real_missing = [k for k in (missing or []) if "position_ids" not in k]
    if real_missing:
        print(f"Missing keys ({len(real_missing)}):")
        for k in real_missing:
            print(f"  {k}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}):")
        for k in unexpected:
            print(f"  {k}")
    if not real_missing and not unexpected:
        print("All CLIP weights loaded successfully.")

    return model.to(dtype=dtype, device=device)


def load_clip_from_safetensors(
    model: CLIPTextModel,
    path: str,
    device: str = "cpu",
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load CLIP weights from a local safetensors file."""
    from safetensors.torch import load_file
    state_dict = load_file(path, device=device)
    filtered = {k: v for k, v in state_dict.items() if "position_ids" not in k}
    return model.load_state_dict(filtered, strict=strict)


# =============================================================================
# Tokenizer Helper (still uses transformers for BPE)
# =============================================================================

def get_tokenizer(
    repo_id: str = "sd-legacy/stable-diffusion-v1-5",
    subfolder: str = "tokenizer",
):
    from transformers import CLIPTokenizer
    return CLIPTokenizer.from_pretrained(repo_id, subfolder=subfolder)


def tokenize(
    tokenizer,
    prompt: str,
    max_length: int = 77,
    device: str = "cpu",
) -> torch.Tensor:
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return tokens.input_ids.to(device)


# =============================================================================
# Verification
# =============================================================================

def verify_architecture(config: Optional[CLIPConfig] = None):
    config = config or CLIPConfig()
    model = CLIPTextModel(config)
    total = sum(p.numel() for p in model.parameters())

    print(f"CLIPTextModel Config-Driven Architecture")
    print(f"========================================")
    print(f"  Total params:      {total:,}")
    print(f"  Hidden dim:        {config.hidden_size}")
    print(f"  Layers:            {config.num_hidden_layers}")
    print(f"  Heads:             {config.num_attention_heads}")
    print(f"  Intermediate:      {config.intermediate_size}")
    print(f"  Vocab:             {config.vocab_size}")
    print(f"  Max tokens:        {config.max_position_embeddings}")
    print(f"  Activation:        {config.hidden_act}")
    print(f"  Output shape:      (B, {config.max_position_embeddings}, {config.hidden_size})")

    return model


if __name__ == "__main__":
    verify_architecture()