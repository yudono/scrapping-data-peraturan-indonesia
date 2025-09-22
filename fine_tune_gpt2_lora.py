#!/usr/bin/env python3
"""
Fine-tune GPT-2 on Indonesian legal text using LoRA (parameter-efficient and lightweight).
By default, it trains only on dataset_txt/1945/* to keep it light for testing.

Usage examples:
  # Install dependencies first
  # pip install -r requirements.txt

  # Quick test on 1945 dataset with tiny settings
  # python fine_tune_gpt2_lora.py \
  #   --dataset_dir dataset_txt/1945 \
  #   --output_dir outputs/gpt2-lora-1945 \
  #   --block_size 256 --batch_size 1 --grad_accum 8 \
  #   --epochs 1 --max_train_steps 200 --save_steps 200

  # Resume training later (if needed)
  # python fine_tune_gpt2_lora.py --dataset_dir dataset_txt/1945 --output_dir outputs/gpt2-lora-1945 --resume_from_checkpoint yes

This script saves LoRA adapter weights (PEFT) to output_dir.
"""
import os
import math
import argparse
import random
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# Try to import GPT-2 Conv1D to detect linear-like layers used in GPT-2 blocks
try:
    from transformers.models.gpt2.modeling_gpt2 import Conv1D as GPT2Conv1D
except Exception:
    GPT2Conv1D = None


class TextChunkDataset(Dataset):
    """Simple dataset that reads all .txt files under dataset_dir,
    concatenates them with separators, tokenizes once, and returns fixed-size chunks."""

    def __init__(self, tokenizer, dataset_dir, block_size=256, limit_chars=None):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Read all .txt files under dataset_dir
        files = sorted(glob(os.path.join(dataset_dir, "*.txt")))
        texts = []
        sep = "\n\n<|sep|>\n\n"
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                    if txt:
                        texts.append(txt.strip())
            except Exception as e:
                print(f"[WARN] Gagal membaca: {fp} -> {e}")
        if not texts:
            raise RuntimeError(f"Tidak ada file .txt di {dataset_dir}")

        corpus = sep.join(texts)
        if limit_chars is not None:
            corpus = corpus[:limit_chars]

        # Tokenize once and create chunks
        toks = tokenizer(
            corpus,
            return_tensors=None,
            padding=False,
            truncation=False,
        )["input_ids"]

        # Create contiguous chunks of size block_size
        self.examples = []
        for i in range(0, len(toks) - block_size + 1, block_size):
            self.examples.append(torch.tensor(toks[i : i + block_size], dtype=torch.long))

        if len(self.examples) == 0:
            raise RuntimeError(
                f"Dataset terlalu kecil. Coba kurangi block_size (sekarang {block_size}) atau pastikan teks cukup."
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        return {"input_ids": x, "labels": x.clone()}


def build_datasets(tokenizer, dataset_dir, block_size, val_ratio=0.1, seed=42, limit_chars=None):
    full_ds = TextChunkDataset(tokenizer, dataset_dir, block_size=block_size, limit_chars=limit_chars)
    val_len = max(1, int(len(full_ds) * val_ratio))
    train_len = len(full_ds) - val_len
    set_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])
    return train_ds, val_ds


def create_lora_model(model_name, lora_r=8, lora_alpha=16, lora_dropout=0.05):
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # GPT-2 has no pad token by default; we will adjust later outside this function

    # Detect target modules dynamically (handle GPT-2 Conv1D and Linear)
    candidate_substrings = [
        "c_attn", "c_proj", "c_fc", "q_attn",
        "q_lin", "k_lin", "v_lin", "out_lin",
        "q_proj", "k_proj", "v_proj", "o_proj",
        "fc_in", "fc_out",
    ]
    detected_names = []
    for name, m in model.named_modules():
        is_linear = isinstance(m, nn.Linear)
        is_gpt2_conv1d = GPT2Conv1D is not None and isinstance(m, GPT2Conv1D)
        if is_linear or is_gpt2_conv1d:
            detected_names.append(name)

    # Exclude output head from LoRA target to avoid issues
    detected_names = [n for n in detected_names if "lm_head" not in n]

    present_substrings = sorted({s for s in candidate_substrings if any(s in n for n in detected_names)})

    # Robust fallback for GPT-2: if nothing detected, use common GPT-2 linear-like layers
    default_gpt2_targets = ["c_attn", "c_proj", "c_fc"]
    if present_substrings:
        target_modules = present_substrings
    elif detected_names:
        # fall back to full names list
        target_modules = detected_names
    else:
        target_modules = default_gpt2_targets

    print(f"[LoRA] target_modules candidates: {target_modules[:8]}{' ...' if len(target_modules) > 8 else ''}")

    # Prepare LoRA config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Optional: prepare model for k-bit training if you later load quantized weights
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        # Safe to ignore if not using quantization
        pass

    model = get_peft_model(model, peft_config)
    # Enable gradient checkpointing for memory saving
    try:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            # required by HF to work nicely with checkpointing in GPT2
            setattr(model.config, "use_cache", False)
        # Important for PyTorch checkpointing: ensure inputs create graph on embeddings
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    except Exception as e:
        print(f"[WARN] Failed enabling gradient checkpointing tweaks: {e}")

    # Log trainable params to ensure LoRA hooked correctly
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Trainable params: {trainable} / {total} ({trainable/total*100:.4f}%)")
    if trainable == 0:
        raise RuntimeError("LoRA tidak menemukan layer target yang sesuai. Coba ganti --model_name (mis. gpt2) atau ubah daftar target_modules.")
    return model


def train(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build datasets
    train_ds, val_ds = build_datasets(
        tokenizer,
        args.dataset_dir,
        block_size=args.block_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit_chars=args.limit_chars,
    )

    # Create LoRA-wrapped model
    model = create_lora_model(
        args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Ensure model uses correct pad id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Data collator for causal LM (no MLM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    save_total_limit = max(1, args.save_total_limit)
    fp16 = args.fp16 and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=save_total_limit,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        fp16=fp16,
        bf16=args.bf16,
        max_steps=args.max_train_steps if args.max_train_steps > 0 else None,
        report_to=[],  # disable wandb by default
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # Train
    ckpt = args.resume_from_checkpoint if args.resume_from_checkpoint not in (None, "", "no", "false", "False") else None
    trainer.train(resume_from_checkpoint=ckpt)

    # Save adapter (LoRA) weights and tokenizer
    trainer.save_model(args.output_dir)  # saves adapter because PEFT wraps model
    tokenizer.save_pretrained(args.output_dir)

    # Quick sample generation after training
    if args.sample_after_train:
        prompt = args.sample_prompt or "Undang-undang"
        print("\n=== Contoh Generasi Setelah Finetune ===")
        print("Prompt:", prompt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=80,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        out = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(out)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 with LoRA on Indonesian legal text")

    # Data & paths
    parser.add_argument("--dataset_dir", type=str, default="dataset_txt/1945", help="Path folder berisi file .txt")
    parser.add_argument("--output_dir", type=str, default="outputs/gpt2-lora-1945", help="Folder output model/adaptor")

    # Model
    parser.add_argument("--model_name", type=str, default="gpt2", help="Nama model HF (mis. gpt2, gpt2-medium)")

    # Tokenization / chunking
    parser.add_argument("--block_size", type=int, default=256, help="Panjang konteks token per contoh")
    parser.add_argument("--limit_chars", type=int, default=None, help="Batasi jumlah karakter korpus untuk debug (opsional)")

    # Split & seed
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Rasio validasi dari dataset")
    parser.add_argument("--seed", type=int, default=42, help="Seed")

    # LoRA params
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training hyperparams
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")

    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--max_train_steps", type=int, default=0, help="0 berarti pakai epochs, >0 untuk batasi total steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default="no")

    # Precision
    parser.add_argument("--fp16", action="store_true", help="Aktifkan fp16 bila GPU tersedia")
    parser.add_argument("--bf16", action="store_true", help="Aktifkan bf16 bila GPU tersedia")

    # Sampling preview
    parser.add_argument("--sample_after_train", action="store_true", help="Generate contoh teks setelah training")
    parser.add_argument("--sample_prompt", type=str, default="Undang-undang")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Config:", args)
    train(args)