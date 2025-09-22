#!/usr/bin/env python3
"""
Script untuk menggunakan model LoRA yang sudah di-fine tune dengan dataset_law.txt.

Cara pakai:
  python run_law_model.py --prompt "Undang-undang tentang ..." \
      --adapter_dir outputs/gpt2-lora-law \
      --base_model gpt2 \
      --max_new_tokens 200 --top_p 0.95 --temperature 0.8

Mode interaktif:
  python run_law_model.py --interactive --adapter_dir outputs/gpt2-lora-law

Catatan:
- Adapter disimpan di outputs/gpt2-lora-law (hasil fine-tuning dengan dataset_law.txt)
- Base model default gpt2. Pastikan sesuai dengan model yang digunakan saat fine-tuning.
"""
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel


def load_model_and_tokenizer(adapter_dir: str, base_model: str, device: str = None):
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory tidak ditemukan: {adapter_dir}")

    print(f"Memuat tokenizer dari: {adapter_dir} (fallback: {base_model})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token is None:
        # GPT-2 family tidak punya pad token; gunakan EOS sebagai pad
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Memuat base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(base_model)

    print(f"Memuat PEFT adapter dari: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)

    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, device


def generate(model, tokenizer, device, prompt: str, max_new_tokens: int = 200,
             do_sample: bool = True, top_k: int = 50, top_p: float = 0.95,
             temperature: float = 0.8, seed: int = 42):
    set_seed(seed)
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode hanya token baru (tanpa prompt)
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text


def parse_args():
    parser = argparse.ArgumentParser(description="Generate teks hukum dengan model LoRA")
    parser.add_argument("--prompt", type=str, default="Undang-undang", help="Prompt untuk generate teks")
    parser.add_argument("--adapter_dir", type=str, default="outputs/gpt2-lora-law", help="Path ke adapter LoRA")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model (gpt2, distilgpt2, dll)")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maksimal token baru")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature untuk sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--interactive", action="store_true", help="Mode interaktif")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=== Loading Model ===")
    model, tokenizer, device = load_model_and_tokenizer(args.adapter_dir, args.base_model)
    print(f"Model loaded on device: {device}")
    
    if args.interactive:
        print("\n=== Mode Interaktif ===")
        print("Ketik 'quit' atau 'exit' untuk keluar")
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if not prompt:
                    continue
                    
                print("\nGenerating...")
                generated = generate(
                    model, tokenizer, device, prompt,
                    max_new_tokens=args.max_new_tokens,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    seed=args.seed
                )
                print(f"\n=== Hasil ===")
                print(f"Prompt: {prompt}")
                print(f"Generated: {generated}")
                
            except KeyboardInterrupt:
                print("\nBye!")
                break
    else:
        print(f"\n=== Single Generation ===")
        print(f"Prompt: {args.prompt}")
        print("Generating...")
        
        generated = generate(
            model, tokenizer, device, args.prompt,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            seed=args.seed
        )
        
        print(f"\n=== Hasil ===")
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {generated}")


if __name__ == "__main__":
    main()