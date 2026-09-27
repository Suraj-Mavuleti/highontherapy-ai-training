#!/usr/bin/env python3
"""
High on Therapy - Custom Tokenizer Trainer
Trains a 100% proprietary Byte-Level BPE tokenizer specifically for counseling
and clinical psychology dialogue, with native ChatML special tokens.
"""

import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast

def train_highontherapy_tokenizer(
    corpus_file="ai_engine/data/highontherapy_corpus.txt",
    output_dir="ai_engine/tokenizer",
    vocab_size=16384
):
    print(f"Initializing custom BPE tokenizer training from {corpus_file}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize empty BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 2. Define clinical & conversational special tokens
    special_tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "<pad>",
        "<|im_start|>",
        "<|im_end|>",
        "<|thought|>",
        "<|somatic_note|>"
    ]

    # 3. Train on counseling corpus
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet()
    )

    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    print("Training BPE merges on clinical corpus...")
    tokenizer.train([corpus_file], trainer)
    
    # Save raw tokenizer.json
    raw_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(raw_path)
    print(f"Saved base tokenizer to {raw_path}")

    # 4. Wrap with Hugging Face PreTrainedTokenizerFast
    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=raw_path,
        bos_token="<s>",
        eos_token="<|im_end|>",
        unk_token="<unk>",
        pad_token="<pad>",
        chat_template=chat_template
    )

    fast_tokenizer.save_pretrained(output_dir)
    print(f"Native High on Therapy Tokenizer successfully compiled into {output_dir}")
    print(f"Vocabulary size: {len(fast_tokenizer)}")
    return fast_tokenizer

if __name__ == "__main__":
    import sys
    corpus = sys.argv[1] if len(sys.argv) > 1 else "ai_engine/data/highontherapy_corpus.txt"
    out = sys.argv[2] if len(sys.argv) > 2 else "ai_engine/tokenizer"
    train_highontherapy_tokenizer(corpus, out)
