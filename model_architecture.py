#!/usr/bin/env python3
"""
High on Therapy - Native Architecture & Parameter Tiers
Defines modern, fully bootstrapped Decoder-Only Transformer architectures
(RMSNorm, RoPE, SwiGLU, GQA) initialized with 100% random weights.
Zero external weights. Zero external models.
"""

import math
from transformers import LlamaConfig, LlamaForCausalLM

TIER_SPECS = {
    "nano": {
        "description": "HighOnTherapy-Nano (~135M params) - Rapid iteration & Free Colab T4 training (4-8 hours)",
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
    },
    "base": {
        "description": "HighOnTherapy-Base (~360M params) - Balanced clinical reasoning (12-24 hours on A100/L4)",
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
    },
    "prime": {
        "description": "HighOnTherapy-Prime (~1.1B params) - Master clinical depth (Multi-day Colab Pro training)",
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 22,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 4096,
    },
    "ultra": {
        "description": "HighOnTherapy-Ultra (~3.2B params) - Advanced therapeutic specialization (Multi-GPU/Cluster training)",
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 28,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
    }
}

def create_highontherapy_config(tier="nano", vocab_size=16384):
    """
    Creates a pure HuggingFace-compatible CausalLM architecture configuration
    with modern LLM architectural enhancements:
    - RMSNorm for stable training gradients
    - Rotary Position Embedding (RoPE) for long-context therapeutic recall
    - Grouped Query Attention (GQA) for memory-efficient inference
    - SwiGLU activation for superior expressiveness
    """
    tier = tier.lower()
    if tier not in TIER_SPECS:
        raise ValueError(f"Unknown tier: {tier}. Available tiers: {list(TIER_SPECS.keys())}")
        
    spec = TIER_SPECS[tier]
    
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=spec["hidden_size"],
        intermediate_size=spec["intermediate_size"],
        num_hidden_layers=spec["num_hidden_layers"],
        num_attention_heads=spec["num_attention_heads"],
        num_key_value_heads=spec["num_key_value_heads"],
        max_position_embeddings=spec["max_position_embeddings"],
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=5, # <|im_end|>
        use_cache=True,
    )
    return config

def initialize_from_scratch(tier="nano", vocab_size=16384):
    """
    Initializes a completely randomized model from scratch.
    NO pretrained weights are downloaded or loaded.
    Every parameter is 100% newly created and initialized.
    """
    config = create_highontherapy_config(tier=tier, vocab_size=vocab_size)
    print(f"Creating {TIER_SPECS[tier]['description']}...")
    model = LlamaForCausalLM(config)
    
    # Verify parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f} Million)")
    print(f"Trainable Parameters: {trainable_params:,} (100% trainable from scratch)")
    return model, config

if __name__ == "__main__":
    print("=== High on Therapy Architectural Tiers ===")
    for tier, spec in TIER_SPECS.items():
        cfg = create_highontherapy_config(tier, vocab_size=16384)
        # Approximate parameter calculation
        v = cfg.vocab_size
        h = cfg.hidden_size
        i = cfg.intermediate_size
        l = cfg.num_hidden_layers
        heads = cfg.num_attention_heads
        kv_heads = cfg.num_key_value_heads
        
        # Embeddings + Attention + MLP + Norms per layer
        attn_params = (h * h) + 2 * (h * (kv_heads * (h // heads))) + (h * h)
        mlp_params = 3 * (h * i)
        layer_params = attn_params + mlp_params + (2 * h)
        total_approx = (v * h) + (l * layer_params) + (2 * h)
        print(f"[{tier.upper()}] ~{total_approx / 1e6:.1f}M params | {l} layers | hidden {h} | {spec['description']}")
