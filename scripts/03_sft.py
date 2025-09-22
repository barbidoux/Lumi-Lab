#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) script with LoRA.
Uses trl and PEFT for efficient fine-tuning of pre-trained model.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    # Fallback for older TRL versions
    from transformers import DataCollatorForLanguageModeling
    DataCollatorForCompletionOnlyLM = None
import sentencepiece as spm
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import load_pretrained_model


def validate_sp32k_tokenizer(tokenizer_path: str, model_path: str = None) -> AutoTokenizer:
    """Load and validate SP32k tokenizer using model metadata."""

    if not tokenizer_path:
        print("ERROR: --tokenizer_path is required. No fallback tokenizer allowed.")
        print("Please provide the path to your SP32k tokenizer from pretraining.")
        sys.exit(1)

    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer path does not exist: {tokenizer_path}")
        sys.exit(1)

    # Load tokenizer metadata from model if available
    tokenizer_metadata = None
    if model_path:
        metadata_path = os.path.join(model_path, "tokenizer_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    tokenizer_metadata = data.get("tokenizer_metadata", {})
                print(f"✓ Loaded tokenizer metadata from {metadata_path}")
                print(f"  Vocab size: {tokenizer_metadata.get('tokenizer_vocab_size', 'unknown')}")
                print(f"  Special tokens: {tokenizer_metadata.get('special_tokens', {})}")
            except Exception as e:
                print(f"WARNING: Could not load tokenizer metadata: {e}")

    try:
        # Load SentencePiece model to validate
        if tokenizer_path.endswith('.model'):
            sp_model = spm.SentencePieceProcessor()
            sp_model.load(tokenizer_path)

            vocab_size = sp_model.get_piece_size()
            print(f"Loaded SentencePiece model with vocab size: {vocab_size}")

            # Validate against metadata if available
            if tokenizer_metadata and 'tokenizer_vocab_size' in tokenizer_metadata:
                expected_vocab_size = tokenizer_metadata['tokenizer_vocab_size']
                if vocab_size != expected_vocab_size:
                    print(f"ERROR: Vocab size mismatch!")
                    print(f"Expected: {expected_vocab_size}, Got: {vocab_size}")
                    print("This tokenizer doesn't match the one used for pretraining.")
                    sys.exit(1)

            # Create tokenizer using SentencePiece directly with known config
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer(vocab_file=tokenizer_path, legacy=True)

            # Set special tokens based on metadata or defaults
            if tokenizer_metadata and 'special_tokens' in tokenizer_metadata:
                special_tokens = tokenizer_metadata['special_tokens']
                tokenizer.pad_token_id = special_tokens.get('pad', 0)
                tokenizer.unk_token_id = special_tokens.get('unk', 1)
                tokenizer.bos_token_id = special_tokens.get('bos', 2)
                tokenizer.eos_token_id = special_tokens.get('eos', 3)
            else:
                # Default SP32k mapping
                tokenizer.pad_token_id = 0
                tokenizer.unk_token_id = 1
                tokenizer.bos_token_id = 2
                tokenizer.eos_token_id = 3

            print("✓ Created HF tokenizer from SentencePiece model")
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    except Exception as e:
        print(f"ERROR: Failed to load tokenizer from {tokenizer_path}")
        print(f"Error: {e}")
        sys.exit(1)

    # Validate special tokens mapping
    expected_special_tokens = {
        "pad": 0,
        "unk": 1,
        "bos": 2,
        "eos": 3
    }

    actual_tokens = {
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id
    }

    print(f"Special tokens mapping: {actual_tokens}")

    # Check if mapping matches expected SP32k format
    for token_name, expected_id in expected_special_tokens.items():
        actual_id = actual_tokens[token_name]
        if actual_id != expected_id:
            print(f"ERROR: Special token '{token_name}' mismatch!")
            print(f"Expected: {expected_id}, Got: {actual_id}")
            print("This tokenizer is not compatible with SP32k pretraining.")
            sys.exit(1)

    print("✓ Tokenizer validation passed - SP32k compatible")
    return tokenizer


def add_chatml_tokens(tokenizer, model):
    """Add ChatML special tokens if missing and resize model embeddings."""

    new_tokens = []

    # Check and add ChatML tokens
    chatml_tokens = ["<|im_start|>", "<|im_end|>"]
    for token in chatml_tokens:
        if token not in tokenizer.get_vocab():
            new_tokens.append(token)

    if new_tokens:
        print(f"Adding new tokens: {new_tokens}")
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        print(f"Added {num_added} new tokens")

        # Resize model embeddings
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model embeddings to {len(tokenizer)} tokens")
    else:
        print("✓ All required ChatML tokens already present")

    return tokenizer, model


def format_prompt_response(example: Dict, prompt_key: str = "prompt", response_key: str = "response",
                          template: str = "instruct", tokenizer=None) -> str:
    """Format an example as prompt-response for SFT training.

    Args:
        example: Dictionary with prompt and response keys
        prompt_key: Key for the prompt
        response_key: Key for the response
        template: Template to use ("instruct", "chat", "chatml")
        tokenizer: Tokenizer for EOS token
    """
    prompt = example[prompt_key].strip()
    response = example[response_key].strip()

    # Get EOS token
    eos_token = tokenizer.eos_token if tokenizer else "<|endoftext|>"

    if template == "chatml":
        # ChatML format with EOS before <|im_end|>
        formatted_text = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n{response}{eos_token}\n<|im_end|>"
    elif template == "chat":
        # Simple conversational format
        formatted_text = f"Human: {prompt}\n\nAssistant: {response}{eos_token}"
    elif template == "instruct":
        # Instruction format (default)
        formatted_text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}{eos_token}"
    else:
        # Custom template or basic format
        formatted_text = f"{prompt}\n{response}{eos_token}"

    return formatted_text


def create_sft_dataset(dataset_paths: List[str], tokenizer,
                       prompt_template: str = "instruct", val_split_ratio: float = 0.02,
                       dataset_weights: Optional[List[float]] = None,
                       max_seq_length: Optional[int] = None) -> tuple:
    """Create formatted train/val datasets for SFT with optional weighting."""

    # Set default max_seq_length if not provided
    if max_seq_length is None:
        max_seq_length = 2048

    datasets = []
    total_samples = 0

    for dataset_path in dataset_paths:
        print(f"Loading dataset: {dataset_path}")
        # Dataset loading
        if dataset_path.endswith(('.json', '.jsonl')):
            dataset = load_dataset('json', data_files=dataset_path)['train']
        else:
            dataset = load_dataset(dataset_path)['train']
        datasets.append(dataset)
        total_samples += len(dataset)
        print(f"  → {len(dataset)} samples")

    # Apply dataset weighting if specified
    if dataset_weights is not None:
        if len(dataset_weights) != len(datasets):
            raise ValueError(f"Number of weights ({len(dataset_weights)}) must match number of datasets ({len(datasets)})")

        print(f"\nApplying dataset weights: {dataset_weights}")
        weighted_datasets = []

        for i, (dataset, weight) in enumerate(zip(datasets, dataset_weights)):
            if weight <= 0:
                print(f"  Skipping dataset {i} (weight={weight})")
                continue

            # Calculate target number of samples
            target_samples = int(weight * total_samples)
            current_samples = len(dataset)

            if target_samples > current_samples:
                # Upsample by repeating examples
                repeat_factor = target_samples // current_samples
                remainder = target_samples % current_samples

                repeated_dataset = dataset
                for _ in range(repeat_factor - 1):
                    repeated_dataset = concatenate_datasets([repeated_dataset, dataset])

                if remainder > 0:
                    remainder_dataset = dataset.select(range(remainder))
                    repeated_dataset = concatenate_datasets([repeated_dataset, remainder_dataset])

                weighted_datasets.append(repeated_dataset)
                print(f"  Dataset {i}: {current_samples} → {len(repeated_dataset)} samples (upsampled)")

            elif target_samples < current_samples:
                # Downsample by selecting random subset
                indices = torch.randperm(current_samples)[:target_samples].tolist()
                downsampled_dataset = dataset.select(indices)
                weighted_datasets.append(downsampled_dataset)
                print(f"  Dataset {i}: {current_samples} → {len(downsampled_dataset)} samples (downsampled)")

            else:
                # Keep as is
                weighted_datasets.append(dataset)
                print(f"  Dataset {i}: {current_samples} samples (unchanged)")

        datasets = weighted_datasets

    # Concatenate all datasets
    if len(datasets) > 1:
        combined_dataset = concatenate_datasets(datasets)
        print(f"Combined dataset: {len(combined_dataset)} total samples")
    else:
        combined_dataset = datasets[0]

    # Clean and filter dataset
    def clean_examples(examples):
        cleaned_prompts = []
        cleaned_responses = []

        for i in range(len(examples['prompt'])):
            prompt = examples['prompt'][i].strip() if examples['prompt'][i] else ""
            response = examples['response'][i].strip() if examples['response'][i] else ""

            # Skip empty examples
            if not prompt or not response:
                continue

            # Remove control characters
            prompt = ''.join(char for char in prompt if ord(char) >= 32 or char in '\n\t')
            response = ''.join(char for char in response if ord(char) >= 32 or char in '\n\t')

            cleaned_prompts.append(prompt)
            cleaned_responses.append(response)

        return {"prompt": cleaned_prompts, "response": cleaned_responses}

    # Clean the dataset
    cleaned_dataset = combined_dataset.map(
        clean_examples,
        batched=True,
        desc="Cleaning examples"
    )

    # Format examples
    def format_examples(examples):
        formatted_texts = []
        for i in range(len(examples['prompt'])):
            example = {k: v[i] for k, v in examples.items()}
            formatted_text = format_prompt_response(example, template=prompt_template, tokenizer=tokenizer)

            # Truncate if too long
            tokenized = tokenizer(formatted_text, truncation=True, max_length=max_seq_length)
            if len(tokenized['input_ids']) == max_seq_length:
                formatted_text = tokenizer.decode(tokenized['input_ids'], skip_special_tokens=False)

            formatted_texts.append(formatted_text)
        return {"text": formatted_texts}

    # Apply formatting
    formatted_dataset = cleaned_dataset.map(
        format_examples,
        batched=True,
        remove_columns=cleaned_dataset.column_names,
        desc="Formatting examples"
    )

    # Split train/val
    if val_split_ratio > 0:
        split_dataset = formatted_dataset.train_test_split(test_size=val_split_ratio, seed=42)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
    else:
        train_dataset = formatted_dataset
        val_dataset = None

    return train_dataset, val_dataset


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
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with LoRA")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to pre-trained model")
    parser.add_argument("--dataset_paths", type=str, nargs="+", required=True,
                       help="Paths to SFT datasets (can be multiple)")
    parser.add_argument("--dataset_weights", type=float, nargs="*", default=None,
                       help="Relative weights for datasets (must match number of datasets)")
    parser.add_argument("--config_path", type=str, default="./config/sft.json",
                       help="Path to SFT configuration")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                       help="Path to SP32k tokenizer (required)")
    parser.add_argument("--prompt_template", type=str, default="instruct",
                       choices=["instruct", "chat", "chatml", "custom"],
                       help="Prompt formatting template (default: instruct)")

    # LoRA options
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA (default: True, use --no_lora to disable)")
    parser.add_argument("--no_lora", action="store_true",
                       help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--lora_r", type=int, default=None,
                       help="LoRA rank (override config)")
    parser.add_argument("--lora_alpha", type=int, default=None,
                       help="LoRA alpha (override config)")
    parser.add_argument("--lora_dropout", type=float, default=None,
                       help="LoRA dropout (override config)")
    parser.add_argument("--merge_adapters", action="store_true",
                       help="Merge and save LoRA adapters after training")

    # Training options
    parser.add_argument("--packing", action="store_true", default=True,
                       help="Enable packing (default: True)")
    parser.add_argument("--no_packing", action="store_true",
                       help="Disable packing")
    parser.add_argument("--resume_from_checkpoint", type=str, nargs="?", const="auto", default=None,
                       help="Resume training from checkpoint (use 'auto' to find latest, or provide path)")
    parser.add_argument("--val_split_ratio", type=float, default=0.02,
                       help="Validation split ratio (default: 0.02)")
    parser.add_argument("--system_prompt", type=str, default=None,
                       help="System prompt for ChatML format")
    parser.add_argument("--do_gen_test", action="store_true",
                       help="Perform generation test after training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    # Training hyperparameters
    parser.add_argument("--max_seq_length", type=int, default=None,
                       help="Maximum sequence length")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None,
                       help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default=None,
                       help="Learning rate scheduler type")
    parser.add_argument("--warmup_ratio", type=float, default=None,
                       help="Warmup ratio")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum training steps")
    parser.add_argument("--eval_steps", type=int, default=None,
                       help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=None,
                       help="Save checkpoint steps")
    parser.add_argument("--logging_steps", type=int, default=None,
                       help="Logging steps")

    args = parser.parse_args()

    # Validate arguments
    if args.no_lora:
        use_lora = False
    else:
        use_lora = not args.no_lora  # Default to True unless --no_lora

    if args.no_packing:
        use_packing = False
    else:
        use_packing = args.packing  # Default to True

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load configuration
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    print("=== SFT Training Configuration ===")
    print(f"Model: {args.model_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Datasets: {args.dataset_paths}")
    print(f"Template: {args.prompt_template}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA: {use_lora}")
    print(f"Packing: {use_packing}")
    print(f"Seed: {args.seed}")

    print("\nLoading and validating tokenizer...")
    # Load and validate SP32k tokenizer
    tokenizer = validate_sp32k_tokenizer(args.tokenizer_path, args.model_path)

    print("\nLoading model...")
    # Load pre-trained model
    model = load_pretrained_model(args.model_path)

    # Add ChatML tokens if using ChatML template
    if args.prompt_template == "chatml":
        print("\nAdding ChatML special tokens...")
        tokenizer, model = add_chatml_tokens(tokenizer, model)
    
    # LoRA configuration
    if use_lora:
        print("\nConfiguring LoRA...")
        # Override config with CLI arguments if provided
        lora_config = config["lora_config"].copy()
        if args.lora_r is not None:
            lora_config["r"] = args.lora_r
        if args.lora_alpha is not None:
            lora_config["lora_alpha"] = args.lora_alpha
        if args.lora_dropout is not None:
            lora_config["lora_dropout"] = args.lora_dropout

        model = setup_lora_model(model, lora_config)
    else:
        print("\nUsing full fine-tuning (no LoRA)")

    # Validate dataset weights if provided
    if args.dataset_weights is not None:
        if len(args.dataset_weights) != len(args.dataset_paths):
            print(f"ERROR: Number of weights ({len(args.dataset_weights)}) must match number of datasets ({len(args.dataset_paths)})")
            sys.exit(1)
        if any(w < 0 for w in args.dataset_weights):
            print("ERROR: Dataset weights must be non-negative")
            sys.exit(1)
        if sum(args.dataset_weights) == 0:
            print("ERROR: At least one dataset weight must be positive")
            sys.exit(1)

    # Load and format datasets
    print(f"\nLoading and formatting datasets (template: {args.prompt_template})...")
    train_dataset, val_dataset = create_sft_dataset(
        args.dataset_paths,
        tokenizer,
        prompt_template=args.prompt_template,
        val_split_ratio=args.val_split_ratio,
        dataset_weights=args.dataset_weights,
        max_seq_length=args.max_seq_length
    )

    print(f"Train dataset: {len(train_dataset)} examples")
    if val_dataset:
        print(f"Validation dataset: {len(val_dataset)} examples")
    print(f"Template used: {args.prompt_template}")
    print("Formatted example:")
    print(train_dataset[0]["text"][:500] + "...")
    
    # Training configuration
    print(f"\nConfiguring training arguments...")

    # Determine precision
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        use_bf16 = True
        use_fp16 = False
        print("Using bf16 precision (Ampere+ GPU detected)")
    else:
        use_bf16 = False
        use_fp16 = config.get("fp16", True)
        print(f"Using fp16 precision: {use_fp16}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,  # Use CLI argument, not config
        per_device_train_batch_size=args.per_device_train_batch_size if args.per_device_train_batch_size is not None else config["batch_size"],
        per_device_eval_batch_size=args.per_device_train_batch_size if args.per_device_train_batch_size is not None else config["batch_size"],
        gradient_accumulation_steps=args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"] if args.max_steps is None else 10,  # Set high epoch limit when using max_steps
        max_steps=args.max_steps if args.max_steps is not None else (config["max_steps"] if config.get("max_steps", -1) > 0 else -1),
        learning_rate=args.learning_rate if args.learning_rate is not None else config["learning_rate"],
        lr_scheduler_type=args.lr_scheduler_type if args.lr_scheduler_type is not None else "cosine",
        warmup_ratio=args.warmup_ratio if args.warmup_ratio is not None else config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        optim="adamw_torch",
        logging_steps=args.logging_steps if args.logging_steps is not None else config["logging_steps"],
        save_steps=args.save_steps if args.save_steps is not None else config["save_steps"],
        eval_steps=args.eval_steps if args.eval_steps is not None else config.get("eval_steps", 500),
        eval_strategy="steps" if val_dataset else "no",
        max_grad_norm=config["max_grad_norm"],
        dataloader_num_workers=config["dataloader_num_workers"],
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing if args.gradient_checkpointing else config.get("gradient_checkpointing", True),
        remove_unused_columns=False,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        run_name=config.get("run_name", "sft_training"),
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    
    # TRL 0.23.0 handles data collation internally - no need for custom collator
    collator = None
    print("Using SFTTrainer's internal data collation (TRL 0.23.0+)")

    # Prepare callbacks
    callbacks = []
    if val_dataset:
        print("Adding early stopping callback...")
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.001
        )
        callbacks.append(early_stopping)

    # Initialize SFT trainer
    print(f"\nInitializing SFT trainer (packing={use_packing})...")
    # Initialize SFT trainer (TRL 0.23.0+ uses processing_class instead of tokenizer)
    # Handle max_seq_length if provided
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "processing_class": tokenizer,  # TRL 0.23.0+ parameter name
        "callbacks": callbacks,
    }

    # Note: max_seq_length is handled differently in TRL 0.23.0
    # It should be controlled via dataset preprocessing or tokenizer settings
    if args.max_seq_length is not None:
        print(f"Note: max_seq_length ({args.max_seq_length}) will be handled via tokenizer truncation")

    trainer = SFTTrainer(**trainer_kwargs)
    print("✓ SFTTrainer initialized with full features")
    
    print("Starting SFT training...")

    # Training
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            # Auto-detect latest checkpoint
            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")] if os.path.exists(args.output_dir) else []
            if checkpoints:
                latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                resume_checkpoint = os.path.join(args.output_dir, latest_checkpoint)
                print(f"Auto-resuming from latest checkpoint: {resume_checkpoint}")
            else:
                print("No checkpoints found for auto-resume")
        elif os.path.exists(args.resume_from_checkpoint):
            resume_checkpoint = args.resume_from_checkpoint
            print(f"Resuming from checkpoint: {resume_checkpoint}")
        else:
            print(f"WARNING: Checkpoint not found: {args.resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    # Save generation config
    generation_config = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with open(os.path.join(training_args.output_dir, "generation_config.json"), "w") as f:
        json.dump(generation_config, f, indent=2)

    # Save LoRA adapters separately if using LoRA
    if use_lora:
        lora_output_dir = Path(training_args.output_dir) / "lora_adapters"
        model.save_pretrained(lora_output_dir)
        print(f"LoRA adapters saved to {lora_output_dir}")

        # Merge adapters if requested
        if args.merge_adapters:
            print("\nMerging LoRA adapters...")
            merged_model = model.merge_and_unload()
            merged_output_dir = Path(training_args.output_dir) / "merged"
            merged_model.save_pretrained(merged_output_dir)
            tokenizer.save_pretrained(merged_output_dir)
            print(f"Merged model saved to {merged_output_dir}")

    print(f"\nSFT model saved to {training_args.output_dir}")
    
    # Optional generation test
    if args.do_gen_test:
        print("\n=== Generation Test ===")
        if args.prompt_template == "chatml":
            # Include system prompt for better quality
            system_prompt = args.system_prompt if args.system_prompt else "You are a helpful, accurate, and concise AI assistant."
            test_prompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n<|im_start|>user\nExplain what artificial intelligence is.\n<|im_end|>\n<|im_start|>assistant\n"
        elif args.prompt_template == "chat":
            test_prompt = "Human: Explain what artificial intelligence is.\n\nAssistant: "
        else:
            test_prompt = "### Instruction:\nExplain what artificial intelligence is.\n\n### Response:\n"

        # Tokenization
        inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generation with conservative settings
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=80,
                temperature=0.2,  # More conservative
                top_p=0.95,       # More conservative
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Reduce repetition
                early_stopping=True       # Stop at EOS
            )

        # Proper decoding
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part
        if args.prompt_template == "chatml":
            # Remove the entire prompt including system
            prompt_for_extraction = full_response.split("<|im_start|>assistant\n")[-1]
            generated_text = prompt_for_extraction.split("<|im_end|>")[0] if "<|im_end|>" in prompt_for_extraction else prompt_for_extraction
        else:
            generated_text = full_response[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]

        print(f"Prompt template: {args.prompt_template}")
        print(f"Generated text:\n{generated_text.strip()}")
        print("=" * 50)

    print("\n✓ SFT training completed successfully!")


if __name__ == "__main__":
    main()