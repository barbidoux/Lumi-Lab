#!/usr/bin/env python3
"""
Alignment script via Direct Preference Optimization (DPO).
Uses trl DPOTrainer to align the model according to human preferences.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer

from utils.model_utils import load_pretrained_model


def format_dpo_example(example: Dict) -> Dict:
    """Format a DPO example with prompt, chosen and rejected."""
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }


def create_dpo_dataset(dataset_path: str) -> Dataset:
    """Create a formatted dataset for DPO."""
    
    # Load dataset
    if dataset_path.endswith(('.json', '.jsonl')):
        dataset = load_dataset('json', data_files=dataset_path)['train']
    else:
        dataset = load_dataset(dataset_path)['train']
    
    # Check required columns
    required_columns = ["prompt", "chosen", "rejected"]
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"Missing column: {col}. Dataset must contain: {required_columns}")
    
    # Format examples
    def format_examples(examples):
        formatted_examples = {"prompt": [], "chosen": [], "rejected": []}
        
        for i in range(len(examples['prompt'])):
            example = {k: v[i] for k, v in examples.items()}
            formatted = format_dpo_example(example)
            
            formatted_examples["prompt"].append(formatted["prompt"])
            formatted_examples["chosen"].append(formatted["chosen"])
            formatted_examples["rejected"].append(formatted["rejected"])
        
        return formatted_examples
    
    # Apply formatting
    formatted_dataset = dataset.map(
        format_examples,
        batched=True,
        desc="Formatting DPO examples"
    )
    
    return formatted_dataset


class EarlyStoppingCallback:
    """Callback for early stopping based on validation loss."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def __call__(self, logs: Dict):
        if "eval_loss" in logs:
            current_loss = logs["eval_loss"]
            
            if current_loss < self.best_loss - self.min_delta:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                self.should_stop = True
                print(f"Early stopping triggered after {self.patience} evaluations without improvement")


def main():
    parser = argparse.ArgumentParser(description="Direct Preference Optimization (DPO)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to SFT model")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to DPO dataset (with chosen/rejected pairs)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/dpo",
                       help="Output directory")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to tokenizer")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="Beta parameter for DPO")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                       help="Learning rate for DPO")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                       help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                       help="Maximum number of steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=512,
                       help="Maximum prompt length")
    
    args = parser.parse_args()
    
    print("Loading SFT model...")
    
    # Load SFT model (with LoRA adapters if applicable)
    # DPOTrainer handles PEFT models directly without fusion
    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")) or \
       os.path.exists(os.path.join(args.model_path, "adapter_model.safetensors")):
        # Model with LoRA adapters - DPOTrainer handles them natively
        print("LoRA adapters detected - direct loading for DPO")
        
        # Load base model
        if os.path.exists(os.path.join(args.model_path, "base_model")):
            base_model_path = os.path.join(args.model_path, "base_model")
        else:
            # Search for base model in parent directories
            parent_path = Path(args.model_path).parent
            base_model_path = str(parent_path)
        
        base_model = load_pretrained_model(base_model_path)
        model = PeftModel.from_pretrained(base_model, args.model_path)
        
        print("LoRA model loaded - DPOTrainer will use adapters directly")
        
    else:
        # Standard model (already merged or without LoRA)
        print("Loading standard model")
        model = load_pretrained_model(args.model_path)
    
    # Load tokenizer
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load DPO dataset
    print("Loading DPO dataset...")
    train_dataset = create_dpo_dataset(args.dataset_path)
    
    print(f"DPO dataset: {len(train_dataset)} examples")
    print("DPO example:")
    print(f"Prompt: {train_dataset[0]['prompt'][:100]}...")
    print(f"Chosen: {train_dataset[0]['chosen'][:100]}...")
    print(f"Rejected: {train_dataset[0]['rejected'][:100]}...")
    
    # Split train/validation if dataset is large enough
    if len(train_dataset) > 100:
        train_size = int(0.9 * len(train_dataset))
        indices = list(range(len(train_dataset)))
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]
        
        eval_dataset = train_dataset.select(eval_indices)
        train_dataset = train_dataset.select(train_indices)
    else:
        eval_dataset = None
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        eval_steps=50 if eval_dataset else -1,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name="dpo_training",
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
    )
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(patience=3, min_delta=0.001)
    
    # Initialize DPO trainer
    # DPOTrainer automatically handles PEFT models and creates reference model
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPOTrainer automatically creates reference copy
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    
    print(f"Starting DPO training with beta={args.beta}...")
    if hasattr(model, 'peft_config'):
        print("DPO mode with LoRA adapters - VRAM optimization enabled")
    print(f"Reference model created automatically by DPOTrainer")
    
    # Training with metrics monitoring
    trainer.train()
    
    # Save final model
    print("Saving DPO model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print(f"DPO model saved to {training_args.output_dir}")
    
    # Quick comparison test
    print("\nComparative generation test:")
    test_prompt = "How can I improve my productivity at work?"
    
    # Tokenization
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=256)
    
    # Generation
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"DPO response: {response[len(test_prompt):]}")
    
    # Display final metrics
    if hasattr(trainer.state, 'log_history'):
        final_logs = trainer.state.log_history[-1]
        print(f"\nFinal metrics:")
        for key, value in final_logs.items():
            if key not in ['epoch', 'step']:
                print(f"  {key}: {value:.4f}")
    
    print("DPO training completed!")


if __name__ == "__main__":
    main()