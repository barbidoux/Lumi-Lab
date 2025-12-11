#!/usr/bin/env python3
"""
Industrial DPO training script for Lumi-Lab pipeline.

This script implements production-grade Direct Preference Optimization with:
- Accelerate integration for distributed training
- Config-driven architecture (all hyperparameters in JSON)
- Multi-dataset weighted sampling
- Tokenizer SHA256 verification
- LoRA/PEFT optimization for memory efficiency
- Progress tracking with detailed metrics
- Robust checkpointing with full state preservation
- DPO-specific evaluation metrics
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from trl import DPOTrainer
from datasets import Dataset, concatenate_datasets

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dpo_utils import DPOMultiDatasetLoader, verify_dpo_corpus_tokenizer
from utils.model_utils import load_pretrained_model


def setup_logging(output_dir: Path, log_level: str = "INFO"):
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=numeric_level,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Also log to file
    log_file = output_dir / "dpo_training.log"
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)


logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load DPO training configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    logger.info(f"📋 Loaded DPO configuration from {config_path}")
    return config


def validate_training_config(config: Dict[str, Any]) -> None:
    """Validate DPO training configuration structure."""
    required_sections = ['training_params', 'dpo_params', 'lora_config']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate training parameters
    required_training_params = [
        'learning_rate', 'per_device_train_batch_size',
        'gradient_accumulation_steps', 'max_steps'
    ]

    for param in required_training_params:
        if param not in config['training_params']:
            raise ValueError(f"Missing required training parameter: {param}")

    # Validate DPO parameters
    required_dpo_params = ['beta', 'max_length', 'max_prompt_length']

    for param in required_dpo_params:
        if param not in config['dpo_params']:
            raise ValueError(f"Missing required DPO parameter: {param}")


def load_datasets(
    data_dirs: List[str],
    data_weights: Optional[List[float]],
    tokenizer_dir: str,
    train_val_split: float = 0.95
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load DPO datasets with weighted sampling.

    Args:
        data_dirs: List of DPO corpus directories
        data_weights: Optional weights for dataset sampling
        tokenizer_dir: Path to tokenizer directory
        train_val_split: Train/validation split ratio

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info(f"📂 Loading {len(data_dirs)} DPO dataset(s)...")

    # Verify tokenizer consistency for all datasets
    for data_dir in data_dirs:
        try:
            verify_dpo_corpus_tokenizer(data_dir, tokenizer_dir)
        except Exception as e:
            logger.error(f"❌ Tokenizer verification failed for {data_dir}: {e}")
            raise

    # Load datasets with multi-dataset loader
    loader = DPOMultiDatasetLoader(
        data_dirs=data_dirs,
        weights=data_weights,
        tokenizer_dir=tokenizer_dir,
        validate_tokenizer=True
    )

    loader.load_datasets()
    combined_dataset = loader.create_weighted_dataset()

    logger.info(f"✅ Combined dataset: {len(combined_dataset):,} samples")

    # Train/validation split
    if train_val_split < 1.0:
        train_size = int(len(combined_dataset) * train_val_split)
        train_dataset = combined_dataset.select(range(train_size))
        val_dataset = combined_dataset.select(range(train_size, len(combined_dataset)))

        logger.info(f"   Train: {len(train_dataset):,} samples")
        logger.info(f"   Val: {len(val_dataset):,} samples")

        return train_dataset, val_dataset
    else:
        logger.info("   No validation split (train_val_split=1.0)")
        return combined_dataset, None


def load_sft_model_with_lora(
    model_path: str,
    lora_config_dict: Optional[Dict[str, Any]] = None
) -> Tuple[Any, bool]:
    """
    Load SFT model (with or without LoRA adapters).

    Args:
        model_path: Path to SFT model checkpoint
        lora_config_dict: Optional LoRA configuration for new adapters

    Returns:
        Tuple of (model, has_existing_lora)
    """
    model_path = Path(model_path)

    # Check if model already has LoRA adapters
    has_lora_adapters = (model_path / "adapter_config.json").exists()

    if has_lora_adapters:
        logger.info("🔧 Loading SFT model with existing LoRA adapters...")

        # Find base model
        if (model_path / "base_model").exists():
            base_model_path = model_path / "base_model"
        else:
            # Try parent directory
            base_model_path = model_path.parent

        # Load base model
        base_model = load_pretrained_model(str(base_model_path))

        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, str(model_path))

        logger.info(f"   Base model: {base_model_path}")
        logger.info(f"   LoRA adapters: {model_path}")

        return model, True

    else:
        logger.info("📦 Loading standard SFT model (no LoRA)...")
        model = load_pretrained_model(str(model_path))

        # Add new LoRA adapters if config provided
        if lora_config_dict:
            logger.info("🔧 Adding new LoRA adapters for DPO...")
            lora_config = LoraConfig(**lora_config_dict)
            model = get_peft_model(model, lora_config)

            # Print trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        return model, False


def create_training_arguments(
    training_params: Dict[str, Any],
    output_dir: str,
    eval_dataset_available: bool
) -> TrainingArguments:
    """Create TrainingArguments from config."""
    # Extract parameters
    learning_rate = training_params['learning_rate']
    per_device_train_batch_size = training_params['per_device_train_batch_size']
    per_device_eval_batch_size = training_params.get('per_device_eval_batch_size', per_device_train_batch_size)
    gradient_accumulation_steps = training_params['gradient_accumulation_steps']
    num_train_epochs = training_params.get('num_train_epochs', 1)
    max_steps = training_params.get('max_steps', -1)
    warmup_steps = training_params.get('warmup_steps', 100)
    weight_decay = training_params.get('weight_decay', 0.01)
    logging_steps = training_params.get('logging_steps', 10)
    save_steps = training_params.get('save_steps', 200)
    eval_steps = training_params.get('eval_steps', 100)
    save_total_limit = training_params.get('save_total_limit', 3)
    seed = training_params.get('seed', 42)

    # Hardware parameters
    bf16 = training_params.get('bf16', True)
    fp16 = training_params.get('fp16', False)
    gradient_checkpointing = training_params.get('gradient_checkpointing', True)
    dataloader_num_workers = training_params.get('dataloader_num_workers', 4)

    # Evaluation strategy
    evaluation_strategy = "steps" if eval_dataset_available else "no"
    eval_steps = eval_steps if eval_dataset_available else None

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy=evaluation_strategy,
        save_strategy="steps",
        load_best_model_at_end=eval_dataset_available,
        metric_for_best_model="eval_loss" if eval_dataset_available else None,
        greater_is_better=False,
        save_total_limit=save_total_limit,
        seed=seed,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        report_to=training_params.get('report_to', ['tensorboard']),
        run_name=f"dpo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ddp_find_unused_parameters=False,
    )

    return training_args


def log_training_info(
    config: Dict[str, Any],
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    training_args: TrainingArguments
):
    """Log comprehensive training information."""
    logger.info("\n" + "=" * 80)
    logger.info("DPO TRAINING CONFIGURATION")
    logger.info("=" * 80)

    # Dataset info
    logger.info(f"📊 Dataset:")
    logger.info(f"   Train samples: {len(train_dataset):,}")
    if val_dataset:
        logger.info(f"   Val samples: {len(val_dataset):,}")

    # Training parameters
    training_params = config['training_params']
    logger.info(f"\n⚙️ Training Parameters:")
    logger.info(f"   Learning rate: {training_params['learning_rate']:.2e}")
    logger.info(f"   Batch size (per device): {training_params['per_device_train_batch_size']}")
    logger.info(f"   Gradient accumulation: {training_params['gradient_accumulation_steps']}")
    effective_batch = training_params['per_device_train_batch_size'] * training_params['gradient_accumulation_steps']
    logger.info(f"   Effective batch size: {effective_batch}")
    logger.info(f"   Max steps: {training_params.get('max_steps', 'N/A')}")
    logger.info(f"   Warmup steps: {training_params.get('warmup_steps', 0)}")
    logger.info(f"   Weight decay: {training_params.get('weight_decay', 0.0)}")

    # DPO parameters
    dpo_params = config['dpo_params']
    logger.info(f"\n🎯 DPO Parameters:")
    logger.info(f"   Beta: {dpo_params['beta']}")
    logger.info(f"   Max length: {dpo_params['max_length']}")
    logger.info(f"   Max prompt length: {dpo_params['max_prompt_length']}")
    logger.info(f"   Loss type: {dpo_params.get('loss_type', 'sigmoid')}")

    # LoRA config
    if 'lora_config' in config:
        lora_config = config['lora_config']
        logger.info(f"\n🔧 LoRA Configuration:")
        logger.info(f"   r: {lora_config['r']}")
        logger.info(f"   alpha: {lora_config['lora_alpha']}")
        logger.info(f"   dropout: {lora_config.get('lora_dropout', 0.0)}")
        logger.info(f"   target_modules: {', '.join(lora_config['target_modules'])}")

    logger.info("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="DPO training for Lumi-Lab pipeline")

    # Config and paths (config-driven architecture)
    parser.add_argument("--config", type=str, required=True,
                       help="Path to DPO training configuration file (e.g., config/dpo/training/dpo_tiny.json)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to SFT model checkpoint")
    parser.add_argument("--data_dirs", nargs='+', type=str, required=True,
                       help="DPO corpus directories (single or multiple for weighted sampling)")
    parser.add_argument("--data_weights", nargs='+', type=float, default=None,
                       help="Weights for multi-dataset sampling (optional, defaults to uniform)")
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                       help="Path to tokenizer directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for DPO checkpoints")

    # Runtime options
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_dir, args.log_level)

    logger.info("=" * 80)
    logger.info("DPO TRAINING - Lumi-Lab Pipeline")
    logger.info("=" * 80)

    try:
        # Load and validate config
        config = load_config(args.config)
        validate_training_config(config)

        # Extract config sections
        training_params = config['training_params']
        dpo_params = config['dpo_params']
        lora_config_dict = config.get('lora_config', None)
        eval_config = config.get('evaluation_config', {})

        # Set seed for reproducibility
        seed = training_params.get('seed', 42)
        set_seed(seed)
        logger.info(f"🎲 Random seed: {seed}")

        # Load tokenizer
        logger.info(f"📖 Loading tokenizer from {args.tokenizer_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("   Set pad_token = eos_token")

        logger.info(f"   Vocabulary size: {len(tokenizer)}")

        # Load datasets
        train_val_split = config.get('data_params', {}).get('train_val_split', 0.95)
        train_dataset, val_dataset = load_datasets(
            data_dirs=args.data_dirs,
            data_weights=args.data_weights,
            tokenizer_dir=args.tokenizer_dir,
            train_val_split=train_val_split
        )

        # Load SFT model
        logger.info(f"🤖 Loading SFT model from {args.model_path}...")
        model, has_existing_lora = load_sft_model_with_lora(
            model_path=args.model_path,
            lora_config_dict=lora_config_dict
        )

        # Create training arguments
        training_args = create_training_arguments(
            training_params=training_params,
            output_dir=args.output_dir,
            eval_dataset_available=(val_dataset is not None)
        )

        # Log training info
        log_training_info(config, train_dataset, val_dataset, training_args)

        # Create DPO trainer
        logger.info("🏋️ Initializing DPO trainer...")

        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # DPOTrainer automatically creates reference copy
            args=training_args,
            beta=dpo_params['beta'],
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            max_length=dpo_params['max_length'],
            max_prompt_length=dpo_params['max_prompt_length'],
            label_pad_token_id=dpo_params.get('label_pad_token_id', -100),
            padding_value=dpo_params.get('padding_value', 0),
            truncation_mode=dpo_params.get('truncation_mode', 'keep_end'),
            generate_during_eval=dpo_params.get('generate_during_eval', False),
            precompute_ref_log_probs=dpo_params.get('precompute_ref_log_probs', False),
        )

        logger.info("   ✅ DPO trainer initialized")
        logger.info(f"   Reference model: Created automatically by DPOTrainer")

        # Start training
        logger.info("\n" + "=" * 80)
        logger.info("🚀 STARTING DPO TRAINING")
        logger.info("=" * 80 + "\n")

        if args.resume_from_checkpoint:
            logger.info(f"📥 Resuming from checkpoint: {args.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()

        # Save final model
        logger.info("\n" + "=" * 80)
        logger.info("💾 Saving final DPO model...")
        final_output_dir = output_dir / "final"
        trainer.save_model(str(final_output_dir))
        tokenizer.save_pretrained(str(final_output_dir))
        logger.info(f"   Model saved to: {final_output_dir}")

        # Run evaluation on test prompts if provided
        eval_prompts = eval_config.get('eval_prompts', [])
        if eval_prompts:
            logger.info("\n" + "=" * 80)
            logger.info("🧪 EVALUATION - Generation Test")
            logger.info("=" * 80 + "\n")

            generation_config = eval_config.get('generation_config', {})
            max_new_tokens = generation_config.get('max_new_tokens', 128)
            temperature = generation_config.get('temperature', 0.7)
            top_p = generation_config.get('top_p', 0.9)

            model.eval()

            for i, prompt in enumerate(eval_prompts[:3]):  # Test first 3
                logger.info(f"Prompt {i+1}: {prompt}")

                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_only = response[len(prompt):].strip()
                logger.info(f"Response: {response_only}\n")

        # Display final metrics
        if hasattr(trainer.state, 'log_history') and len(trainer.state.log_history) > 0:
            logger.info("\n" + "=" * 80)
            logger.info("📊 FINAL METRICS")
            logger.info("=" * 80)

            final_logs = trainer.state.log_history[-1]
            for key, value in final_logs.items():
                if key not in ['epoch', 'step'] and isinstance(value, (int, float)):
                    logger.info(f"   {key}: {value:.4f}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ DPO TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Final model: {final_output_dir}")
        logger.info("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"❌ Error during DPO training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
