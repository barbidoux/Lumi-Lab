#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script with LoRA.
Uses trl and PEFT for efficient fine-tuning of pre-trained model.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from utils.model_utils import load_pretrained_model


def format_prompt_response(example: Dict, prompt_key: str = "prompt", response_key: str = "response", 
                          template: str = "instruct") -> str:
    """Format an example as prompt-response for SFT training.
    
    Args:
        example: Dictionary with prompt and response keys
        prompt_key: Key for the prompt
        response_key: Key for the response
        template: Template to use ("instruct", "chat", "chatml")
    """
    prompt = example[prompt_key]
    response = example[response_key]
    
    if template == "chatml":
        # Recommended ChatML format: <|im_start|>user...\n<|im_end|>\n<|im_start|>assistant...\n<|im_end|>
        formatted_text = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n{response}\n<|im_end|>"
    elif template == "chat":
        # Simple conversational format
        formatted_text = f"Human: {prompt}\n\nAssistant: {response}<|endoftext|>"
    elif template == "instruct":
        # Instruction format (default)
        formatted_text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}<|endoftext|>"
    else:
        # Custom template or basic format
        formatted_text = f"{prompt}\n{response}<|endoftext|>"
    
    return formatted_text


def create_sft_dataset(dataset_path: str, tokenizer, max_seq_length: int = 2048, 
                       prompt_template: str = "instruct") -> Dataset:
    """Create a formatted dataset for SFT."""
    
    # Dataset loading
    if dataset_path.endswith(('.json', '.jsonl')):
        dataset = load_dataset('json', data_files=dataset_path)['train']
    else:
        dataset = load_dataset(dataset_path)['train']
    
    # Format examples
    def format_examples(examples):
        formatted_texts = []
        for i in range(len(examples['prompt'])):
            example = {k: v[i] for k, v in examples.items()}
            formatted_text = format_prompt_response(example, template=prompt_template)
            formatted_texts.append(formatted_text)
        return {"text": formatted_texts}
    
    # Apply formatting
    formatted_dataset = dataset.map(
        format_examples,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting examples"
    )
    
    return formatted_dataset


def setup_lora_model(model, lora_config: Dict):
    """Configure model with LoRA."""
    
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning avec LoRA")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to pre-trained model")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to SFT dataset")
    parser.add_argument("--config_path", type=str, default="./config/sft.json",
                       help="Path to SFT configuration")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft",
                       help="Output directory")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to tokenizer (optional)")
    parser.add_argument("--prompt_template", type=str, default="instruct",
                       choices=["instruct", "chat", "chatml", "custom"],
                       help="Prompt formatting template (default: instruct)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    print("Loading model and tokenizer...")
    
    # Load pre-trained model
    model = load_pretrained_model(args.model_path)
    
    # Load tokenizer
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        # Use default tokenizer if not specified
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    
    # Tokenizer configuration
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA configuration
    print("LoRA configuration...")
    model = setup_lora_model(model, config["lora_config"])
    
    # Chargement du dataset
    print(f"Loading and formatting dataset (template: {args.prompt_template})...")
    train_dataset = create_sft_dataset(args.dataset_path, tokenizer, 
                                     prompt_template=args.prompt_template)
    
    print(f"SFT dataset: {len(train_dataset)} examples")
    print(f"Template used: {args.prompt_template}")
    print("Formatted example:")
    print(train_dataset[0]["text"][:500] + "...")
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"] if config["max_steps"] > 0 else -1,
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config.get("eval_steps", 500),
        max_grad_norm=config["max_grad_norm"],
        dataloader_num_workers=config["dataloader_num_workers"],
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", False),
        gradient_checkpointing=config["gradient_checkpointing"],
        remove_unused_columns=False,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name=config.get("run_name", "sft_training"),
        save_total_limit=3,
        load_best_model_at_end=False,
    )
    
    # Data collator to mask prompts - template dependent on format
    if args.prompt_template == "chatml":
        response_template = "<|im_start|>assistant\n"
    elif args.prompt_template == "chat":
        response_template = "Assistant: "
    elif args.prompt_template == "instruct":
        response_template = "### Response:\n"
    else:
        response_template = "### Response:\n"  # Fallback
    
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        max_seq_length=2048,
        packing=False,  # Disabled for better sequence control
    )
    
    print("Starting SFT training...")
    
    # Training
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Save LoRA adapters separately
    lora_output_dir = Path(training_args.output_dir) / "lora_adapters"
    model.save_pretrained(lora_output_dir)
    
    print(f"SFT model saved to {training_args.output_dir}")
    print(f"LoRA adapters saved to {lora_output_dir}")
    
    # Quick generation test
    print("\nGeneration test:")
    if args.prompt_template == "chatml":
        test_prompt = "<|im_start|>user\nExplain what artificial intelligence is.\n<|im_end|>\n<|im_start|>assistant\n"
    elif args.prompt_template == "chat":
        test_prompt = "Human: Explain what artificial intelligence is.\n\nAssistant: "
    else:
        test_prompt = "### Instruction:\nExplain what artificial intelligence is.\n\n### Response:\n"
    
    # Tokenization
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Generation
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Generated response: {response[len(test_prompt):]}")
    
    print("SFT training completed!")


if __name__ == "__main__":
    main()