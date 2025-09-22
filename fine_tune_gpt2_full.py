#!/usr/bin/env python3
"""
Full fine-tune GPT-2 on Indonesian legal text (bukan LoRA, tapi full model).
Hasil: model baru lengkap yang bisa digunakan langsung tanpa base model.

Usage examples:
  # Install dependencies first
  # pip install -r requirements.txt

  # Full fine-tuning dengan dataset_law.txt
  # python fine_tune_gpt2_full.py \
  #   --dataset_path dataset_law.txt \
  #   --output_dir outputs/gpt2-full-law \
  #   --block_size 256 --batch_size 1 --grad_accum 8 \
  #   --epochs 2 --save_steps 500

  # Resume training later (if needed)
  # python fine_tune_gpt2_full.py --dataset_path dataset_law.txt --output_dir outputs/gpt2-full-law --resume_from_checkpoint yes

Script ini menyimpan model lengkap (bukan adapter) ke output_dir.
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


class TextChunkDataset(Dataset):
    """
    Dataset untuk membaca teks dari file .txt atau folder berisi file .txt,
    lalu memotongnya menjadi chunk dengan panjang block_size.
    """
    def __init__(self, tokenizer, dataset_path, block_size=256, limit_chars=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Baca semua teks
        all_text = ""
        if os.path.isfile(dataset_path):
            # Single file
            print(f"Reading single file: {dataset_path}")
            with open(dataset_path, 'r', encoding='utf-8') as f:
                all_text = f.read()
        elif os.path.isdir(dataset_path):
            # Directory dengan file .txt
            txt_files = glob(os.path.join(dataset_path, "*.txt"))
            if not txt_files:
                raise ValueError(f"No .txt files found in {dataset_path}")
            
            print(f"Reading {len(txt_files)} .txt files from {dataset_path}")
            for txt_file in sorted(txt_files):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    all_text += f.read() + "\n\n"
        else:
            raise ValueError(f"dataset_path must be a file or directory: {dataset_path}")
        
        # Batasi karakter jika diminta (untuk debug)
        if limit_chars and limit_chars > 0:
            all_text = all_text[:limit_chars]
            print(f"Limited to {limit_chars} characters")
        
        print(f"Total characters: {len(all_text):,}")
        
        # Tokenize semua teks
        print("Tokenizing...")
        tokens = self.tokenizer(all_text, return_tensors="pt", truncation=False)
        self.input_ids = tokens["input_ids"].squeeze(0)  # shape: [total_tokens]
        
        print(f"Total tokens: {len(self.input_ids):,}")
        
        # Hitung jumlah chunk
        self.num_chunks = max(1, len(self.input_ids) // self.block_size)
        print(f"Number of chunks (block_size={self.block_size}): {self.num_chunks}")
    
    def __len__(self):
        return self.num_chunks
    
    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        
        # Ambil chunk
        chunk = self.input_ids[start:end]
        
        # Jika chunk kurang dari block_size, pad dengan eos_token_id
        if len(chunk) < self.block_size:
            pad_length = self.block_size - len(chunk)
            pad_token_id = self.tokenizer.eos_token_id
            chunk = torch.cat([chunk, torch.full((pad_length,), pad_token_id)])
        
        return {"input_ids": chunk, "labels": chunk.clone()}


def build_datasets(tokenizer, dataset_path, block_size, val_ratio=0.1, seed=42, limit_chars=None):
    full_ds = TextChunkDataset(tokenizer, dataset_path, block_size=block_size, limit_chars=limit_chars)
    val_len = max(1, int(len(full_ds) * val_ratio))
    train_len = len(full_ds) - val_len
    set_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])
    return train_ds, val_ds


def train(args):
    # Set seed
    set_seed(args.seed)
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    # GPT-2 tidak punya pad token, gunakan eos sebagai pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Build datasets
    train_ds, val_ds = build_datasets(
        tokenizer,
        args.dataset_path,
        block_size=args.block_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit_chars=args.limit_chars,
    )
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 adalah causal LM, bukan masked LM
    )
    
    # Training arguments
    save_total_limit = args.save_total_limit if args.save_total_limit > 0 else None
    fp16 = args.fp16 and torch.cuda.is_available()
    
    # Handle max_steps properly
    max_steps = args.max_train_steps if args.max_train_steps > 0 else -1
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="steps",
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
        max_steps=max_steps,
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

    # Save full model and tokenizer
    print(f"Saving full model to: {args.output_dir}")
    trainer.save_model(args.output_dir)  # saves full model
    tokenizer.save_pretrained(args.output_dir)

    # Quick sample generation after training
    if args.sample_after_train:
        prompt = args.sample_prompt or "Undang-undang"
        print("\n=== Contoh Generasi Setelah Finetune ===")
        print("Prompt:", prompt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated:", generated_text)


def parse_args():
    parser = argparse.ArgumentParser(description="Full fine-tune GPT-2 dengan Indonesian legal text")
    
    # Data & paths
    parser.add_argument("--dataset_path", type=str, default="dataset_law.txt", help="Path ke file .txt atau folder berisi file .txt")
    parser.add_argument("--output_dir", type=str, default="outputs/gpt2-full-law", help="Folder output model lengkap")
    
    # Model
    parser.add_argument("--model_name", type=str, default="gpt2", help="Nama model HF (mis. gpt2, gpt2-medium)")
    parser.add_argument("--block_size", type=int, default=256, help="Panjang konteks token per contoh")
    parser.add_argument("--limit_chars", type=int, default=None, help="Batasi jumlah karakter korpus untuk debug (opsional)")
    
    # Dataset split
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Rasio validasi dari dataset")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    
    # Training hyperparams
    parser.add_argument("--epochs", type=float, default=3.0, help="Jumlah epoch")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="LR scheduler")
    
    # Logging & saving
    parser.add_argument("--eval_steps", type=int, default=100, help="Eval setiap N steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save setiap N steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log setiap N steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maksimal checkpoint tersimpan")
    parser.add_argument("--max_train_steps", type=int, default=0, help="0 berarti pakai epochs, >0 untuk batasi total steps")
    
    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default="no", help="Resume dari checkpoint")
    
    # Mixed precision
    parser.add_argument("--fp16", action="store_true", help="Aktifkan fp16 bila GPU tersedia")
    parser.add_argument("--bf16", action="store_true", help="Aktifkan bf16 bila GPU tersedia")
    
    # Sampling
    parser.add_argument("--sample_after_train", action="store_true", help="Generate contoh teks setelah training")
    parser.add_argument("--sample_prompt", type=str, default="Undang-undang", help="Prompt untuk sampling")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print("Config:", args)
    train(args)