#!/usr/bin/env python3
"""
Merge existing LoRA checkpoint with base model.
Usage: python helpers/merge_lora_checkpoint.py --base_model <path> --lora_adapters <path> --output_dir <path>
"""
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from peft import PeftModel


def merge_lora_checkpoint(base_model_path: str, lora_path: str, output_dir: str, force: bool = False):
    """Merge LoRA adapters with base model."""

    output_path = Path(output_dir)

    if output_path.exists() and not force:
        print(f"⚠️  Output directory already exists: {output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    print(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    print(f"Loading LoRA adapters from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_path), safe_serialization=True)

    print("✓ Merge complete!")
    print(f"  Merged model saved to: {output_path}")
    print(f"\nYou can now evaluate with:")
    print(f"  python scripts/06_eval_sft.py --model_path {output_path} ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base model")
    parser.add_argument('--base_model', type=str, required=True,
                       help='Path to base/pretrained model')
    parser.add_argument('--lora_adapters', type=str, required=True,
                       help='Path to LoRA adapters checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for merged model')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite output directory without confirmation')

    args = parser.parse_args()

    merge_lora_checkpoint(
        base_model_path=args.base_model,
        lora_path=args.lora_adapters,
        output_dir=args.output_dir,
        force=args.force
    )
