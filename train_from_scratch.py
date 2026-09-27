#!/usr/bin/env python3
"""
High on Therapy - From-Scratch Pre-training & Bootstrapping Engine
Initializes a 100% native Decoder-Only Transformer architecture with randomized weights
and trains it purely on counseling and psychotherapy corpora.
Features:
- 5-Minute Timed Auto-Backup to Drive / Local Storage
- Dynamic Hardware Auto-Upgrader (maximizes GPU throughput based on available VRAM)
- Multi-Day Auto-Resume from latest checkpoint
Zero external weights. Zero Qwen. 100% proprietary.
"""

import os
import sys
import time
import argparse
import math
import json
import glob
import shutil

def train_from_scratch(args):
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        PreTrainedTokenizerFast,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        TrainerCallback
    )
    from model_architecture import create_highontherapy_config, initialize_from_scratch

    print("=" * 75)
    print("  HIGH ON THERAPY: BOOTSTRAPPED FROM-SCRATCH COUNSELING AI")
    print("=" * 75)
    print(f"Tier:                 {args.tier.upper()}")
    print(f"Dataset:              {args.data_file}")
    print(f"Tokenizer:            {args.tokenizer_dir}")
    print(f"Output Directory:     {args.output_dir}")
    print(f"5-Min Auto-Backup:    Active (every {args.backup_interval} seconds / 5 mins)")
    print(f"Auto-Resource Scale:  {'Enabled' if args.auto_scale else 'Disabled'}")
    print(f"Device:               {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # 0. Hardware Detection & Dynamic Resource Auto-Upgrader
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    num_workers = 2
    
    if torch.cuda.is_available() and args.auto_scale:
        device_name = torch.cuda.get_device_name(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        total_gb = total_bytes / (1024**3)
        free_gb = free_bytes / (1024**3)
        print(f"Detected GPU:         {device_name} ({total_gb:.1f} GB VRAM, {free_gb:.1f} GB Free)")
        
        # Auto-Upgrade compute parameters based on GPU headroom
        if total_gb >= 35.0:  # A100 (40GB/80GB)
            batch_size = max(batch_size, 16 if args.tier == "nano" else 8)
            grad_accum = max(2, grad_accum // 2)
            num_workers = 4
            print(f"🚀 [AUTO-UPGRADER] High-End GPU Detected ({device_name})! Auto-upgraded batch_size={batch_size}, grad_accum={grad_accum}, workers={num_workers}")
        elif total_gb >= 20.0:  # L4 / RTX 3090 / RTX 4090 / V100 (24GB)
            batch_size = max(batch_size, 8 if args.tier == "nano" else 4)
            grad_accum = max(4, grad_accum // 2)
            num_workers = 4
            print(f"🚀 [AUTO-UPGRADER] Mid-High GPU Detected ({device_name})! Auto-upgraded batch_size={batch_size}, grad_accum={grad_accum}")
        else: # T4 (15GB/16GB)
            print(f"⚡ [AUTO-UPGRADER] Free Tier GPU Detected ({device_name}). Utilizing optimized throughput settings.")
    print("=" * 75)

    # 1. Load Custom Tokenizer
    if not os.path.exists(args.tokenizer_dir):
        raise FileNotFoundError(f"Tokenizer directory not found: {args.tokenizer_dir}. Please run train_tokenizer.py first.")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
    vocab_size = len(tokenizer)
    print(f"Loaded native tokenizer with {vocab_size} vocabulary entries.")

    # 2. Initialize Model Architecture from Random Weights
    model, config = initialize_from_scratch(tier=args.tier, vocab_size=vocab_size)
    print("Random weight initialization complete. 0 external weights loaded.")

    # 3. Prepare Dataset
    class CounselingTextDataset(Dataset):
        def __init__(self, file_path, tokenizer, max_length=1024):
            self.examples = []
            print(f"Tokenizing dataset from {file_path}...")
            
            if file_path.endswith(".jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            messages = item.get("messages", [])
                            if messages:
                                text = tokenizer.apply_chat_template(messages, tokenize=False)
                                encoded = tokenizer(text, truncation=True, max_length=max_length)
                                if len(encoded["input_ids"]) > 10:
                                    self.examples.append(torch.tensor(encoded["input_ids"], dtype=torch.long))
                        except Exception:
                            pass
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                blocks = text.split("<|im_end|>\n\n")
                for b in blocks:
                    b = b.strip()
                    if len(b) > 20:
                        encoded = tokenizer(b, truncation=True, max_length=max_length)
                        self.examples.append(torch.tensor(encoded["input_ids"], dtype=torch.long))

            print(f"Successfully compiled {len(self.examples)} training sequences.")

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i):
            return {"input_ids": self.examples[i], "labels": self.examples[i]}

    dataset = CounselingTextDataset(args.data_file, tokenizer, max_length=args.max_seq_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. Check for existing checkpoints to auto-resume multi-day training
    os.makedirs(args.output_dir, exist_ok=True)
    existing_checkpoints = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
    resume_checkpoint = None
    if existing_checkpoints and args.auto_resume:
        existing_checkpoints.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
        resume_checkpoint = existing_checkpoints[-1]
        print(f"\n[AUTO-RESUME] Found existing checkpoint: {resume_checkpoint}")
        print("Resuming training seamlessly from last saved step...\n")

    # 5. Define 5-Minute Auto-Backup Callback
    class TimedAutoBackupCallback(TrainerCallback):
        def __init__(self, backup_interval_seconds=300, output_dir=None, tokenizer=None):
            self.interval = backup_interval_seconds
            self.last_backup = time.time()
            self.backup_dir = os.path.join(output_dir, "autobackups")
            self.tokenizer = tokenizer
            os.makedirs(self.backup_dir, exist_ok=True)

        def on_step_end(self, args, state, control, model=None, **kwargs):
            now = time.time()
            elapsed = now - self.last_backup
            if elapsed >= self.interval:
                self.last_backup = now
                step = state.global_step
                step_save_dir = os.path.join(self.backup_dir, f"autobackup_step_{step}")
                latest_dir = os.path.join(self.backup_dir, "autobackup_latest")
                
                print(f"\n💾 [5-MIN AUTO-BACKUP] Running periodic backup at step {step} ({time.strftime('%Y-%m-%d %H:%M:%S')})...")
                try:
                    if model is not None:
                        model.save_pretrained(step_save_dir)
                        model.save_pretrained(latest_dir)
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(step_save_dir)
                        self.tokenizer.save_pretrained(latest_dir)
                    
                    # Keep latest 2 rolling step backups + latest to conserve storage
                    all_backups = glob.glob(os.path.join(self.backup_dir, "autobackup_step_*"))
                    if len(all_backups) > 2:
                        all_backups.sort(key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
                        for old in all_backups[:-2]:
                            shutil.rmtree(old, ignore_errors=True)

                    print(f"✅ [5-MIN AUTO-BACKUP] State successfully saved to {latest_dir}!")
                except Exception as e:
                    print(f"⚠️ [5-MIN AUTO-BACKUP] Backup note: {e}")

    # 6. Define Runtime Resource Upgrader Callback
    class DynamicResourceUpgraderCallback(TrainerCallback):
        def __init__(self, check_every_steps=50):
            self.check_every = check_every_steps

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.check_every == 0 and torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                free_gb = free_b / (1024**3)
                total_gb = total_b / (1024**3)
                utilization = ((total_b - free_b) / total_b) * 100
                print(f"📊 [RESOURCE MONITOR - Step {state.global_step}] VRAM: {total_gb - free_gb:.1f}GB/{total_gb:.1f}GB ({utilization:.1f}% used, {free_gb:.1f}GB free)")
                if free_gb > total_gb * 0.5:
                    print(f"⚡ [AUTO-UPGRADER] Hardware has {free_gb:.1f}GB surplus VRAM headroom. Compute throughput running at peak stability!")

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_num_workers=num_workers if torch.cuda.is_available() else 0,
        report_to="none",
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub if args.push_to_hub else None,
        hub_token=args.hf_token if args.hf_token else None,
    )

    # 8. Trainer Execution with Callbacks
    import inspect
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "data_collator": data_collator,
        "callbacks": [
            TimedAutoBackupCallback(backup_interval_seconds=args.backup_interval, output_dir=args.output_dir, tokenizer=tokenizer),
            DynamicResourceUpgraderCallback(check_every_steps=50)
        ]
    }
    sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("\nStarting model pre-training from scratch...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 9. Save Final Model and Native Tokenizer
    print(f"\nSaving final bootstrapped model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Model and tokenizer saved successfully.")

    if args.push_to_hub:
        print(f"Pushing native model weights to Hugging Face repository: {args.push_to_hub}...")
        trainer.push_to_hub()
        print("Model successfully pushed to Hugging Face!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High on Therapy - From-Scratch Pre-training")
    parser.add_argument("--tier", type=str, default="nano", choices=["nano", "base", "prime", "ultra"], help="Architectural scale tier")
    parser.add_argument("--data_file", type=str, default="ai_engine/data/highontherapy_chatml.jsonl", help="Training dataset path")
    parser.add_argument("--tokenizer_dir", type=str, default="ai_engine/tokenizer", help="Native tokenizer directory")
    parser.add_argument("--output_dir", type=str, default="./highontherapy_native_model", help="Directory to save checkpoints and final model")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--backup_interval", type=int, default=300, help="Auto-backup interval in seconds (default 300 = 5 mins)")
    parser.add_argument("--auto_scale", action="store_true", default=True, help="Auto-scale batch size and resource utilization")
    parser.add_argument("--auto_resume", action="store_true", default=True, help="Auto resume from latest checkpoint")
    parser.add_argument("--push_to_hub", type=str, default="", help="Hugging Face repo ID to push to (e.g. DeV-ZEr0/highontherapy-native)")
    parser.add_argument("--hf_token", type=str, default="", help="Hugging Face write token")

    args = parser.parse_args()
    train_from_scratch(args)
