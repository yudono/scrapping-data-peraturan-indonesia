#!/usr/bin/env python3
"""
Script untuk menggunakan model LoRA hukum_1945 (adapter) untuk generate teks dari prompt Anda.

Cara pakai:
  python run_hukum_1945.py --prompt "Undang-undang tentang ..." \
      --adapter_dir outputs/hukum_1945 \
      --base_model distilgpt2 \
      --max_new_tokens 200 --top_p 0.95 --temperature 0.8

Mode interaktif:
  python run_hukum_1945.py --interactive --adapter_dir outputs/hukum_1945

Catatan:
- Adapter disimpan di outputs/hukum_1945 (hasil fine-tuning sebelumnya)
- Base model default distilgpt2 (ringan). Anda bisa ganti ke gpt2 jika mau.
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

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def parse_args():
    p = argparse.ArgumentParser(description="Run text generation with hukum_1945 LoRA adapter")
    p.add_argument("--adapter_dir", type=str, default="outputs/hukum_1945",
                   help="Folder adapter LoRA (hasil fine-tuning)")
    p.add_argument("--base_model", type=str, default="distilgpt2",
                   help="Nama base model HF, mis. distilgpt2 / gpt2")
    p.add_argument("--prompt", type=str, default=None,
                   help="Prompt teks untuk generate (abaikan jika --interactive)")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"],
                   help="Pilih device secara manual (default otomatis)")
    p.add_argument("--interactive", action="store_true",
                   help="Aktifkan mode interaktif (masukkan prompt berulang-ulang)")
    return p.parse_args()


def main():
    args = parse_args()

    model, tokenizer, device = load_model_and_tokenizer(
        adapter_dir=args.adapter_dir,
        base_model=args.base_model,
        device=args.device,
    )

    if args.interactive:
        print("Mode interaktif. Ketik prompt lalu Enter. Ketik /quit untuk keluar.")
        while True:
            try:
                prompt = input("Prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nKeluar.")
                break
            if not prompt:
                continue
            if prompt.lower() in {"/quit", ":q", "exit", "keluar"}:
                print("Keluar.")
                break
            text = generate(
                model, tokenizer, device, prompt,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                seed=args.seed,
            )
            print("\n=== Hasil ===\n")
            print(text)
            print("\n============\n")
    else:
        if not args.prompt:
            raise SystemExit("Harap isi --prompt atau gunakan --interactive")
        text = generate(
            model, tokenizer, device, args.prompt,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            seed=args.seed,
        )
        print(text)


if __name__ == "__main__":
    main()