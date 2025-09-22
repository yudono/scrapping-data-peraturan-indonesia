#!/usr/bin/env python3
"""
Script untuk menjalankan model GPT-2 yang sudah di-full fine-tune dengan dataset hukum Indonesia.
Model ini adalah model lengkap (bukan adapter), jadi tidak perlu base model.

Usage examples:
  # Generate single text
  python run_full_law_model.py \
    --model_dir outputs/gpt2-full-law \
    --prompt "Undang-undang tentang" \
    --max_tokens 200

  # Interactive mode
  python run_full_law_model.py \
    --model_dir outputs/gpt2-full-law \
    --interactive

  # Batch generation dengan beberapa prompt
  python run_full_law_model.py \
    --model_dir outputs/gpt2-full-law \
    --prompts "Pasal 1" "Ayat (1)" "Ketentuan umum" \
    --max_tokens 150
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_dir):
    """
    Load full fine-tuned model dan tokenizer dari directory.
    """
    print(f"Loading model from: {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    # Set pad token jika belum ada
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt, max_tokens=200, temperature=0.8, top_p=0.95, do_sample=True):
    """
    Generate text dari prompt menggunakan model yang sudah di-fine-tune.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode hasil
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Hapus prompt dari hasil (ambil bagian yang di-generate saja)
    new_text = generated_text[len(prompt):].strip()
    
    return generated_text, new_text


def interactive_mode(model, tokenizer, device, max_tokens=200, temperature=0.8, top_p=0.95):
    """
    Mode interaktif untuk chat dengan model.
    """
    print("\n=== Mode Interaktif ===")
    print("Ketik prompt Anda (atau 'quit' untuk keluar):")
    print("Contoh prompt: 'Undang-undang tentang', 'Pasal 1', 'Ayat (1)', dll.\n")
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Keluar dari mode interaktif.")
                break
            
            if not prompt:
                continue
            
            print("\nGenerating...")
            full_text, new_text = generate_text(
                model, tokenizer, device, prompt, 
                max_tokens=max_tokens, temperature=temperature, top_p=top_p
            )
            
            print(f"\n--- Hasil ---")
            print(f"Full text: {full_text}")
            print(f"\n--- Generated part ---")
            print(f"{new_text}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nKeluar dari mode interaktif.")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_generate(model, tokenizer, device, prompts, max_tokens=200, temperature=0.8, top_p=0.95):
    """
    Generate text untuk beberapa prompt sekaligus.
    """
    print(f"\n=== Batch Generation ({len(prompts)} prompts) ===")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")
        print("Generating...")
        
        try:
            full_text, new_text = generate_text(
                model, tokenizer, device, prompt,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p
            )
            
            print(f"Full text: {full_text}")
            print(f"Generated: {new_text}")
            
        except Exception as e:
            print(f"Error generating for prompt '{prompt}': {e}")
        
        print("-" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Run full fine-tuned GPT-2 model untuk teks hukum Indonesia")
    
    # Model path
    parser.add_argument("--model_dir", type=str, required=True, 
                       help="Path ke directory model yang sudah di-full fine-tune")
    
    # Generation modes
    parser.add_argument("--prompt", type=str, default=None, 
                       help="Single prompt untuk generate")
    parser.add_argument("--prompts", type=str, nargs="+", default=None, 
                       help="Multiple prompts untuk batch generation")
    parser.add_argument("--interactive", action="store_true", 
                       help="Mode interaktif")
    
    # Generation parameters
    parser.add_argument("--max_tokens", type=int, default=200, 
                       help="Maksimal token baru yang di-generate")
    parser.add_argument("--temperature", type=float, default=0.8, 
                       help="Temperature untuk sampling (0.1-2.0)")
    parser.add_argument("--top_p", type=float, default=0.95, 
                       help="Top-p untuk nucleus sampling (0.1-1.0)")
    parser.add_argument("--no_sample", action="store_true", 
                       help="Gunakan greedy decoding (deterministic)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(args.model_dir)
    
    do_sample = not args.no_sample
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, device, 
                        max_tokens=args.max_tokens, 
                        temperature=args.temperature, 
                        top_p=args.top_p)
    
    elif args.prompts:
        # Batch generation
        batch_generate(model, tokenizer, device, args.prompts,
                      max_tokens=args.max_tokens,
                      temperature=args.temperature,
                      top_p=args.top_p)
    
    elif args.prompt:
        # Single generation
        print(f"\n=== Single Generation ===")
        print(f"Prompt: {args.prompt}")
        print("Generating...")
        
        full_text, new_text = generate_text(
            model, tokenizer, device, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=do_sample
        )
        
        print(f"\nFull text: {full_text}")
        print(f"\nGenerated part: {new_text}")
    
    else:
        # Default: interactive mode
        print("No specific mode selected. Starting interactive mode...")
        interactive_mode(model, tokenizer, device,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p)


if __name__ == "__main__":
    main()