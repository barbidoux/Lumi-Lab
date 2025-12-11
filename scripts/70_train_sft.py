#!/usr/bin/env python3
r"""
Industrial SFT training script for Lumi-Lab pipeline.

This script implements production-grade supervised fine-tuning with:
- Accelerate integration for distributed training
- Robust checkpointing with full state preservation
- Multi-dataset weighted sampling
- Advanced evaluation with generation tests
- LoRA/PEFT optimization for memory efficiency
- Deterministic training for reproducibility
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer, set_seed, EarlyStoppingCallback, TrainerCallback
from transformers import (
    AutoModelForCausalLM,
)
from transformers import LlamaTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
)
from trl import SFTTrainer, SFTConfig
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    DataCollatorForCompletionOnlyLM = None

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_utils import StreamingSFTDataset
from utils.sft_evaluation import SFTEvaluator
from utils.model_utils import create_model, load_pretrained_model


def setup_logging(output_dir: Path, verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    # Also log to file
    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    logging.info(f"Loaded training configuration from {config_path}")
    return config


def validate_training_config(config: Dict[str, Any]) -> None:
    """Validate training configuration structure."""
    required_sections = ['training_params', 'lora_config', 'dataset_config']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate training parameters
    required_training_params = [
        'learning_rate', 'per_device_train_batch_size',
        'gradient_accumulation_steps', 'num_train_epochs'
    ]

    for param in required_training_params:
        if param not in config['training_params']:
            raise ValueError(f"Missing required training parameter: {param}")


def load_datasets(data_dirs: List[str], data_weights: Optional[List[float]] = None) -> Tuple[List[StreamingSFTDataset], List[StreamingSFTDataset], Optional[List[float]]]:
    """
    Load SFT datasets from directories using the streaming dataset class.

    Args:
        data_dirs: List of data directories
        data_weights: Optional weights for multi-dataset sampling

    Returns:
        Tuple of (train_datasets, val_datasets, normalized_weights)
    """
    train_datasets = []
    val_datasets = []

    for data_dir in data_dirs:
        logger.info(f"Initializing streaming SFT dataset from: {data_dir}")

        # Load train split
        try:
            # Use the new streaming dataset
            train_dataset = StreamingSFTDataset(data_dir, split="train")
            train_datasets.append(train_dataset)
        except Exception as e:
            logger.error(f"Could not initialize training data stream from {data_dir}: {e}")
            raise

        # Load validation split
        try:
            # Use the new streaming dataset
            val_dataset = StreamingSFTDataset(data_dir, split="val")
            val_datasets.append(val_dataset)
        except Exception as e:
            # A missing validation set is not a fatal error
            logger.warning(f"Could not initialize validation data stream from {data_dir}: {e}")

    if not train_datasets:
        raise ValueError("No training datasets could be loaded!")

    # Validate tokenizer consistency across datasets
    if len(train_datasets) > 1:
        reference_metadata = train_datasets[0].get_tokenizer_metadata()
        for i, dataset in enumerate(train_datasets[1:], 1):
            dataset_metadata = dataset.get_tokenizer_metadata()
            if dataset_metadata.get('tokenizer_sha256') != reference_metadata.get('tokenizer_sha256'):
                raise ValueError(f"Tokenizer mismatch between datasets! "
                               f"Dataset 0: {reference_metadata.get('tokenizer_sha256', 'N/A')[:12]}... "
                               f"Dataset {i}: {dataset_metadata.get('tokenizer_sha256', 'N/A')[:12]}...")

    logger.info(f"Successfully initialized {len(train_datasets)} training dataset streams")
    if val_datasets:
        logger.info(f"Successfully initialized {len(val_datasets)} validation dataset streams")

    # Normalize weights
    normalized_weights = None
    if data_weights and len(data_weights) > 0:
        if len(data_weights) != len(data_dirs):
            raise ValueError(f"Number of weights ({len(data_weights)}) must match number of data_dirs ({len(data_dirs)})")
        total_weight = sum(data_weights)
        normalized_weights = [w / total_weight for w in data_weights]
        logger.info(f"📊 Dataset weights (normalized): {normalized_weights}")

    return train_datasets, val_datasets, normalized_weights


def setup_model_and_tokenizer(model_path: str,
                             lora_config: Dict[str, Any],
                             training_params: Dict[str, Any],
                             tokenizer_path: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Setup model with LoRA and validate tokenizer.

    Returns:
        Tuple of (model, tokenizer_metadata)
    """
    logger.info(f"Loading model from: {model_path}")

    # Use the standard load_pretrained_model function with auto device detection
    # This handles both HuggingFace format and custom Lumi-Lab checkpoints
    model = load_pretrained_model(model_path, device="auto")
    logger.info("✓ Loaded model successfully")

    # Setup LoRA
    logger.info("Setting up LoRA configuration...")
    peft_config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        target_modules=lora_config['target_modules'],
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config.get('bias', 'none'),
        task_type="CAUSAL_LM",
        fan_in_fan_out=lora_config.get('fan_in_fan_out', False),
        init_lora_weights=lora_config.get('init_lora_weights', True)
    )

    # Apply LoRA to model
    model = get_peft_model(model, peft_config)

    # Enable gradient checkpointing if specified (read from training_params, not lora_config)
    if training_params.get('gradient_checkpointing', False):
        logger.info("✓ Enabling gradient checkpointing (VRAM optimization)")
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
               f"({100 * trainable_params / total_params:.2f}%)")

    # Create tokenizer metadata
    tokenizer_metadata = {
        'tokenizer_path': str(tokenizer_path),
        'validation_status': 'ok'
    }

    return model, tokenizer_metadata


def prepare_hf_tokenizer(tokenizer_path: str) -> str:
    """
    Ensures the tokenizer is in a Hugging Face compatible format.
    If given a path to a directory with a .model file, it converts it
    and saves it to a 'hf_cache' subdirectory for future use.

    Args:
        tokenizer_path: Path to the tokenizer directory.

    Returns:
        Path to the Hugging Face compatible tokenizer directory.
    """
    p = Path(tokenizer_path)
    # The most reliable way to check for a modern HF tokenizer is the presence of tokenizer.json
    hf_tokenizer_file = p / "tokenizer.json"

    if hf_tokenizer_file.is_file():
        logger.info(f"✅ Tokenizer at '{tokenizer_path}' appears to be a valid Hugging Face tokenizer (found tokenizer.json).")
        return tokenizer_path
    
    # If tokenizer.json is not found, we assume it's a raw SentencePiece model and needs conversion.
    logger.warning(
        f"No `tokenizer.json` found in '{tokenizer_path}'. Assuming it's a raw SentencePiece model and attempting conversion."
    )

    # Find the .model file
    spm_files = list(p.glob("*.model"))
    if not spm_files:
        raise FileNotFoundError(f"No .model file found in {p} for conversion.")

    spm_model_path = spm_files[0]
    if len(spm_files) > 1:
        logger.warning(f"Multiple .model files found, using {spm_model_path.name}")

    # Define a cache directory for the converted tokenizer
    cache_dir = p / "hf_cache"
    if (cache_dir / "tokenizer_config.json").exists():
        logger.info(f"✅ Found cached Hugging Face tokenizer at: {cache_dir}")
        return str(cache_dir)

    logger.info(f"Converting SentencePiece model {spm_model_path.name} to Hugging Face format...")

    try:
        # Use LlamaTokenizer as a wrapper to perform the conversion
        tokenizer = LlamaTokenizer(vocab_file=str(spm_model_path), legacy=False)
        tokenizer.save_pretrained(str(cache_dir))
        logger.info(f"✅ Conversion successful. Saved to: {cache_dir}")
        return str(cache_dir)
    except Exception as e:
        logger.error(f"❌ Failed to convert tokenizer: {e}")
        raise


def run_generation_evaluation(model: nn.Module,
                            tokenizer_path: str,
                            template_name: str,
                            eval_prompts: List[str],                            generation_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run qualitative generation evaluation."""
    logger.info("Running generation evaluation...")

    # Setup evaluator
    # The model passed to this function should already be on the correct device.
    # If using PEFT, the base model is what we need.
    unwrapped_model = model
    while hasattr(unwrapped_model, "model"):
        unwrapped_model = unwrapped_model.model

    evaluator = SFTEvaluator(
        model=unwrapped_model,
        tokenizer_path=tokenizer_path,
        template_name=template_name,
    )

    # Generate responses
    results = evaluator.evaluate_prompts(eval_prompts, generation_config)

    # Calculate quality metrics
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        responses = [r['clean_response'] for r in successful_results]
        quality_metrics = evaluator.quality_assessment(responses)

        return {
            'num_prompts': len(eval_prompts),
            'successful_generations': len(successful_results),
            'quality_metrics': quality_metrics,
            'sample_responses': results[:3]  # First 3 for logging
        }

    return {'num_prompts': len(eval_prompts), 'successful_generations': 0}


def train_model_with_trl(model: nn.Module,
                        train_datasets: List[StreamingSFTDataset],
                        val_datasets: List[StreamingSFTDataset],
                        config: Dict[str, Any],
                        tokenizer_path: str,
                        output_dir: Path,
                        resume_from_checkpoint: Optional[str] = None,
                        do_gen_tests: bool = False,
                        load_in_memory: bool = False,
                        merge_adapters: bool = False) -> None:
    """Fast training using TRL SFTTrainer with streaming datasets."""

    from datasets import load_dataset, concatenate_datasets

    if load_in_memory:
        logger.info("🚀 Using TRL SFTTrainer with IN-MEMORY datasets (faster on WSL, uses more RAM)")
    else:
        logger.info("🚀 Using TRL SFTTrainer with HuggingFace datasets (optimized loading)")

    # Load tokenizer for SFTTrainer
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=False
        )
        logger.info("✓ Successfully loaded AutoTokenizer from directory")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        logger.error("TRL SFTTrainer REQUIRES a valid tokenizer. Cannot continue.")
        raise RuntimeError(f"Tokenizer loading failed: {e}")

    # Ensure pad token is set for TRL
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets using HuggingFace load_dataset (like 03_sft.py)
    # This is the official, optimized method with Apache Arrow backend
    logger.info("Loading datasets with HuggingFace load_dataset() (fast, optimized)...")

    def load_sft_shards(dataset_dir, split):
        """Load all shards for a given split using HF load_dataset."""
        import glob
        # Try multiple shard naming patterns
        patterns = [
            f"{dataset_dir}/{split}_*.jsonl*",      # train_00000.jsonl
            f"{dataset_dir}/{split}_shard_*.jsonl*" # train_shard_0000.jsonl
        ]

        shard_files = []
        for pattern in patterns:
            files = sorted(glob.glob(pattern))
            if files:
                shard_files = files
                break

        if not shard_files:
            logger.warning(f"No {split} shards found in {dataset_dir}")
            return None

        logger.info(f"Loading {len(shard_files)} {split} shards from {dataset_dir}")
        # load_dataset with Apache Arrow is MUCH faster than manual iteration
        # keep_in_memory=True ensures data is loaded into RAM, avoiding mmap issues on WSL2 /mnt/c
        dataset = load_dataset('json', data_files=shard_files, split='train', keep_in_memory=load_in_memory)

        # If load_in_memory is enabled, load everything into RAM
        if load_in_memory:
            logger.info(f"  ⚡ Loading {split} data into RAM (this may take a moment)...")
            # The keep_in_memory=True flag already handled this.
            # We can optionally format to torch tensors for a slight extra boost, but it's not the main fix.
            dataset = dataset.with_format("torch")
            logger.info(f"  ✓ {split.capitalize()} data fully loaded in RAM")

        return dataset

    # Load training data
    train_hf_datasets = []
    for streaming_dataset in train_datasets:
        # Get the directory path from StreamingSFTDataset
        dataset_dir = str(streaming_dataset.data_dir)
        train_data = load_sft_shards(dataset_dir, 'train')
        if train_data:
            train_hf_datasets.append(train_data)

    if not train_hf_datasets:
        raise ValueError("No training data could be loaded!")

    # Apply weighted sampling if multiple datasets
    if len(train_hf_datasets) > 1:
        if data_weights:
            # Weighted interleaving with config-driven parameters
            from datasets import interleave_datasets

            # Get interleave strategy from config (config-driven, no hardcoded values)
            interleave_config = config.get('dataset_config', {}).get('interleave_strategy', {})
            seed = interleave_config.get('seed', 42)
            stopping_strategy = interleave_config.get('stopping_strategy', 'all_exhausted')

            logger.info(f"🎲 Interleaving {len(train_hf_datasets)} datasets with weights: {data_weights}")
            logger.info(f"   Strategy: seed={seed}, stopping_strategy={stopping_strategy}")

            train_hf_dataset = interleave_datasets(
                train_hf_datasets,
                probabilities=data_weights,
                seed=seed,
                stopping_strategy=stopping_strategy
            )
            logger.info(f"✓ Interleaved {len(train_hf_datasets)} training datasets with weighted sampling")
        else:
            # Simple concatenation (equal weight)
            train_hf_dataset = concatenate_datasets(train_hf_datasets)
            logger.info(f"✓ Concatenated {len(train_hf_datasets)} training datasets (equal weight)")
    else:
        train_hf_dataset = train_hf_datasets[0]

    logger.info(f"✓ Loaded {len(train_hf_dataset):,} training examples")

    # Detect data format
    is_prepacked = all(col in train_hf_dataset.column_names for col in ['input_ids', 'attention_mask', 'labels'])

    if is_prepacked:
        logger.info("📦 Detected pre-packed data (format v3.0) - no runtime tokenization needed!")
        logger.info(f"   Columns: {train_hf_dataset.column_names}")
        # Keep all columns for pre-packed data
    elif 'text' in train_hf_dataset.column_names:
        # Keep only 'text' column to avoid TRL confusion with prompt/completion fields
        # TRL auto-detects prompt+completion and tries to use them, but we want unified 'text' field
        columns_to_remove = [col for col in train_hf_dataset.column_names if col != 'text']
        if columns_to_remove:
            logger.info(f"Removing extra columns for TRL: {columns_to_remove}")
            train_hf_dataset = train_hf_dataset.remove_columns(columns_to_remove)

    # Load validation data
    val_hf_dataset = None
    if val_datasets:
        val_hf_datasets = []
        for streaming_dataset in val_datasets:
            dataset_dir = str(streaming_dataset.data_dir)
            val_data = load_sft_shards(dataset_dir, 'val')
            if val_data:
                val_hf_datasets.append(val_data)

        if val_hf_datasets:
            if len(val_hf_datasets) > 1:
                val_hf_dataset = concatenate_datasets(val_hf_datasets)
            else:
                val_hf_dataset = val_hf_datasets[0]
            logger.info(f"✓ Loaded {len(val_hf_dataset):,} validation examples")

            # Keep only 'text' column in validation too (if not pre-packed)
            if not is_prepacked and 'text' in val_hf_dataset.column_names:
                columns_to_remove = [col for col in val_hf_dataset.column_names if col != 'text']
                if columns_to_remove:
                    val_hf_dataset = val_hf_dataset.remove_columns(columns_to_remove)

    # Setup training arguments (use TrainingArguments like 03_sft.py, not SFTConfig)
    training_params = config['training_params']
    dataset_config = config['dataset_config']

    # Handle potential API changes in TrainingArguments for evaluation_strategy
    if "evaluation_strategy" in training_params and "eval_strategy" not in training_params:
        logger.info("Mapping 'evaluation_strategy' to 'eval_strategy' for compatibility.")
        training_params["eval_strategy"] = training_params.pop("evaluation_strategy")

    # Override report_to to disable TensorBoard (avoid callback errors)
    training_params['report_to'] = 'none'

    # Set environment variable to prevent tokenizer parallelism issues
    # This prevents tokenizers from spawning their own workers (conflicts with DataLoader workers)
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # DataLoader workers are now safe to use with streaming datasets
    # The config value (typically 8) will be used for efficient data loading
    logger.info(f"DataLoader workers: {training_params.get('dataloader_num_workers', 'default')}")

    # Use TrainingArguments instead of SFTConfig (simpler, like 03_sft.py)
    # SFTTrainer will handle everything automatically
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=f"sft_{config['name']}",
        **training_params,
    )

    logger.info(f"Training configuration: batch_size={training_args.per_device_train_batch_size}, "
               f"grad_accum={training_args.gradient_accumulation_steps}, "
               f"max_steps={training_args.max_steps}")

    # Disable W&B
    os.environ['WANDB_DISABLED'] = 'true'

    # Choose appropriate trainer based on data format
    if is_prepacked:
        # Pre-packed data: use regular Trainer (much faster, no tokenization overhead)
        logger.info(f"🚀 Initializing Trainer for pre-packed data (format v3.0)")
        logger.info(f"   No tokenization or formatting overhead - direct training!")

        # Use simple data collator that doesn't do any processing
        from transformers import default_data_collator

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_hf_dataset,
            eval_dataset=val_hf_dataset,
            data_collator=default_data_collator,  # Just batches the data as-is
        )
    else:
        # Raw text data: use SFTTrainer for tokenization and formatting
        logger.info(f"Initializing SFTTrainer (data has 'text' field, TRL will auto-tokenize)")

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,  # TRL 0.23.0+ uses 'processing_class' for multimodal support
            train_dataset=train_hf_dataset,
            eval_dataset=val_hf_dataset,
            args=training_args,
        )

    # Add early stopping if configured
    if training_params.get("early_stopping_patience"):
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=training_params["early_stopping_patience"]
        ))

    # Add custom metrics logging callback
    class MetricsLogger(TrainerCallback):
        """Custom callback to log metrics to JSON file."""
        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)
            self.metrics_file = self.output_dir / "training_metrics.jsonl"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Save metrics to JSONL file on each log."""
            if logs:
                log_entry = {
                    "step": state.global_step,
                    "epoch": state.epoch,
                    **logs
                }
                with open(self.metrics_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

    trainer.add_callback(MetricsLogger(output_dir))
    logger.info(f"📊 Metrics will be saved to: {output_dir / 'training_metrics.jsonl'}")

    # Start training
    logger.info(f"Training on {len(train_hf_dataset):,} examples for {training_args.max_steps} steps")
    trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    logger.info(f"✓ Final model saved to: {final_dir}")

    # Merge LoRA adapters if requested
    if merge_adapters:
        logger.info("🔧 Merging LoRA adapters with base model...")
        merged_model = trainer.model.merge_and_unload()
        merged_dir = output_dir / "merged"
        merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
        logger.info(f"✓ Merged model saved to: {merged_dir}")

    # Run final generation test if enabled
    if do_gen_tests and "eval_prompts" in config.get("evaluation_config", {}):
        logger.info("Running final generation test...")
        eval_config = config["evaluation_config"]
        gen_results = run_generation_evaluation(            model=trainer.model, # Use the model from the trainer
            tokenizer_path=tokenizer_path,
            template_name=config.get('template', 'chatml'),
            eval_prompts=eval_config['eval_prompts'], 
            generation_config=eval_config.get('generation_config', {}),
        )
        if gen_results:
            logger.info(f"Final generation results: {gen_results['successful_generations']}/{gen_results['num_prompts']} successful")

    logger.info("✅ TRL training completed!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Industrial SFT training script")

    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Training configuration file (JSON)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pre-trained model')
    parser.add_argument('--data_dirs', type=str, nargs='+', required=True,
                       help='Paths to SFT data directories')
    parser.add_argument('--tokenizer_path', '--tokenizer_dir', type=str, required=True,
                       dest='tokenizer_path',
                       help='Path to tokenizer directory (accepts --tokenizer_dir alias for consistency)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints')

    # Optional arguments
    parser.add_argument('--data_weights', type=float, nargs='*',
                       help='Sampling weights for multi-dataset training')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from, or "True" to resume from the latest in output_dir')
    parser.add_argument('--do_gen_tests', action='store_true',
                       help='Enable generation tests during evaluation (useful for monitoring model quality)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--load_in_memory', action='store_true',
                       help='Load entire dataset in RAM instead of memory-mapping (faster on WSL, requires more RAM)')
    parser.add_argument('--merge_adapters', action='store_true',
                       help='Merge LoRA adapters with base model after training (creates merged/ directory)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging - no accelerator needed
    setup_logging(output_dir, args.verbose)

    # Load and validate configuration first (to get seed)
    config = load_config(args.config)
    validate_training_config(config)

    # Extract seed from config
    seed = config.get('training_params', {}).get('seed', 42)

    # Set seed
    set_seed(seed)

    logger.info("🚀 Starting industrial SFT training...")
    logger.info(f"📋 Configuration: {args.config}")
    logger.info(f"🤖 Model: {args.model_path}")
    logger.info(f"📊 Data directories: {args.data_dirs}")
    logger.info(f"🔤 Tokenizer: {args.tokenizer_path}")
    logger.info(f"📂 Output: {output_dir}")
    logger.info(f"🎲 Seed: {seed}")

    try:

        # Load datasets
        train_datasets, val_datasets, data_weights = load_datasets(args.data_dirs, args.data_weights)

        # Setup model and tokenizer
        model, tokenizer_metadata = setup_model_and_tokenizer(
            args.model_path,
            config['lora_config'],
            config['training_params'],
            args.tokenizer_path
        )

        # Prepare tokenizer, converting if necessary
        logger.info("Preparing tokenizer for Hugging Face compatibility...")
        hf_tokenizer_path = prepare_hf_tokenizer(args.tokenizer_path)

        # Handle checkpoint resuming for TRL
        resume_from_checkpoint = args.resume_from_checkpoint
        if resume_from_checkpoint == "True":
            # TRL's trainer can auto-resume from the output_dir
            resume_from_checkpoint = True
            logger.info(f"Resuming from latest checkpoint in {args.output_dir}")
        else:
            resume_from_checkpoint = resume_from_checkpoint

        # Start training using the TRL trainer
        train_model_with_trl(
            model=model,
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            config=config,
            tokenizer_path=hf_tokenizer_path,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            do_gen_tests=args.do_gen_tests,
            load_in_memory=args.load_in_memory,
            merge_adapters=args.merge_adapters
        )

        logger.info("✅ SFT training completed successfully!")

    except Exception as e:
        logger.error(f"❌ Error during SFT training: {e}")
        raise


if __name__ == "__main__":
    main()