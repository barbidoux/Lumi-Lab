#!/usr/bin/env python3
"""
Training script from scratch for a mini-LLM.
Uses accelerate for training management and checkpoints.
"""

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_utils import TokenizedDataset, WeightedMultiDatasetSampler, PackedDataset, ShardedTokenizedDataset
from utils.model_utils import create_model, save_checkpoint, load_checkpoint


def find_latest_checkpoint(output_dir: str, model_name: str) -> Optional[str]:
    """
    Find the latest checkpoint in the output directory.

    This function looks for checkpoints in two possible locations:
    1. {output_dir}/{model_name}/step_* (recommended structure)
    2. {output_dir}/step_* (fallback for different structures)

    Args:
        output_dir: The base output directory
        model_name: The model name

    Returns:
        Path to the latest checkpoint directory, or None if no checkpoints found
    """
    def scan_for_checkpoints(search_dir: Path) -> list:
        """Scan a directory for checkpoint directories."""
        step_dirs = []

        if not search_dir.exists():
            return step_dirs

        for item in search_dir.iterdir():
            if item.is_dir() and item.name.startswith("step_"):
                try:
                    step_num = int(item.name.split("_")[1])
                    step_dirs.append((step_num, item))
                except (ValueError, IndexError):
                    continue

        # Also check for 'final' directory
        final_dir = search_dir / "final"
        if final_dir.exists() and final_dir.is_dir():
            # Get the highest step number + 1 for final dir to make it the latest
            max_step = max([step for step, _ in step_dirs], default=-1)
            step_dirs.append((max_step + 1, final_dir))

        return step_dirs

    # Method 1: Look in {output_dir}/{model_name}/ (standard structure)
    model_dir = Path(output_dir) / model_name
    step_dirs = scan_for_checkpoints(model_dir)

    # Method 2: If no checkpoints found, look directly in {output_dir}/ (fallback)
    if not step_dirs:
        output_path = Path(output_dir)
        step_dirs = scan_for_checkpoints(output_path)

    if not step_dirs:
        return None

    # Sort by step number (latest first) to check from newest to oldest
    step_dirs.sort(key=lambda x: x[0], reverse=True)

    # Check each checkpoint from latest to oldest until we find a valid one
    for step_num, checkpoint_dir in step_dirs:
        # Check for alternative model file formats
        model_files = [
            "pytorch_model.bin",
            "model.safetensors"
        ]

        has_model_file = any(
            (checkpoint_dir / model_file).exists()
            for model_file in model_files
        )

        # Check required files for Accelerate checkpoints
        # These files are CRITICAL for proper training resumption
        required_files = [
            "config.json",            # Model configuration
            "meta.json",              # Training metadata (global_step, etc.)
            "rng_state.pth"          # RNG states for reproducibility
        ]

        # Check for Accelerate optimizer/scheduler files (different naming patterns)
        accelerate_state_files = [
            "optimizer.bin",          # Accelerate optimizer state
            "scheduler.bin"           # Accelerate scheduler state
        ]

        missing_required = []
        missing_accelerate = []

        # Check required files
        for required_file in required_files:
            if not (checkpoint_dir / required_file).exists():
                missing_required.append(required_file)

        # Check Accelerate state files
        for state_file in accelerate_state_files:
            if not (checkpoint_dir / state_file).exists():
                missing_accelerate.append(state_file)

        # A checkpoint is valid if it has model + required files + accelerate states
        if not missing_required and not missing_accelerate and has_model_file:
            return str(checkpoint_dir)
        else:
            # This checkpoint is invalid, log and continue to next
            print(f"⚠️  Warning: Checkpoint {checkpoint_dir} is incomplete, skipping:")
            if missing_required:
                print(f"    Missing REQUIRED files: {missing_required}")
            if missing_accelerate:
                print(f"    Missing ACCELERATE state files: {missing_accelerate}")
            if not has_model_file:
                print(f"    No model file found (checked: {model_files})")
            print(f"    Complete resume requires: model + meta + optimizer + scheduler + RNG state")

    # No valid checkpoint found
    return None


def validate_tokenizer_consistency(data_dirs: List[str], tokenizer_dir: str, accelerator: Accelerator) -> Dict:
    """Validate that all datasets are compatible with the given tokenizer using SHA256 hashes."""
    if not data_dirs:
        return {}

    accelerator.print("🔍 Validating tokenizer consistency across datasets...")

    # Load tokenizer config and compute its hash for comparison
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"

    if not tokenizer_config_path.exists():
        raise FileNotFoundError(f"❌ Tokenizer config not found: {tokenizer_config_path}")

    # Load and hash tokenizer config
    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
        tokenizer_config = json.load(f)

    tokenizer_config_str = json.dumps(tokenizer_config, sort_keys=True, ensure_ascii=False)
    tokenizer_hash = hashlib.sha256(tokenizer_config_str.encode('utf-8')).hexdigest()

    accelerator.print(f"📊 Reference tokenizer hash: {tokenizer_hash[:16]}...")

    # Check if datasets use new PackedDataset format vs legacy format
    packed_datasets = []
    legacy_datasets = []

    for data_dir in data_dirs:
        final_manifest_path = Path(data_dir) / 'final_manifest.json'
        old_manifest_path = Path(data_dir) / 'manifest.json'

        if final_manifest_path.exists():
            packed_datasets.append(data_dir)
        elif old_manifest_path.exists():
            legacy_datasets.append(data_dir)
        else:
            raise FileNotFoundError(
                f"❌ No manifest file found in {data_dir}\n"
                f"Please ensure this dataset has been processed correctly."
            )

    if legacy_datasets:
        accelerator.print(f"⚠️  Warning: {len(legacy_datasets)} datasets use legacy format")
        accelerator.print("   Legacy datasets will use old validation method")
        # Fall back to legacy validation for old format datasets
        return validate_tokenizer_consistency_legacy(legacy_datasets, accelerator)

    # All datasets use new PackedDataset format - use robust SHA256 validation
    accelerator.print(f"✅ All {len(packed_datasets)} datasets use modern PackedDataset format")

    reference_metadata = None

    for data_dir in packed_datasets:
        try:
            # Load final_manifest.json directly for hash comparison
            final_manifest_path = Path(data_dir) / 'final_manifest.json'
            with open(final_manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            # Get the tokenizer hash from manifest
            dataset_tokenizer_hash = manifest.get('tokenizer_config_hash')

            if not dataset_tokenizer_hash:
                raise ValueError(
                    f"❌ No tokenizer_config_hash found in {final_manifest_path}\n"
                    f"This dataset was processed with an older version.\n"
                    f"Please re-process this dataset."
                )

            # Compare hashes
            if dataset_tokenizer_hash != tokenizer_hash:
                raise ValueError(
                    f"❌ Tokenizer mismatch for dataset {data_dir}\n"
                    f"   Dataset hash:   {dataset_tokenizer_hash[:16]}...\n"
                    f"   Tokenizer hash: {tokenizer_hash[:16]}...\n"
                    f"Please re-process this dataset with the current tokenizer."
                )

            # Extract metadata for logging
            if reference_metadata is None:
                reference_metadata = {
                    'vocab_size': manifest.get('statistics', {}).get('vocab_size', 'N/A'),
                    'tokenizer_config_hash': dataset_tokenizer_hash
                }
                accelerator.print(f"   Vocab size: {reference_metadata['vocab_size']:,}" if isinstance(reference_metadata['vocab_size'], int) else f"   Vocab size: {reference_metadata['vocab_size']}")

            accelerator.print(f"✅ {data_dir}: tokenizer compatibility verified (hash match)")

        except Exception as e:
            raise RuntimeError(f"❌ Failed to validate dataset {data_dir}: {e}")

    accelerator.print(f"✅ All {len(packed_datasets)} datasets are compatible with tokenizer")
    return reference_metadata


def validate_tokenizer_consistency_legacy(data_dirs: List[str], accelerator: Accelerator) -> Dict:
    """Legacy validation for old format datasets (backward compatibility)."""
    reference_metadata = None
    reference_dir = None

    for data_dir in data_dirs:
        manifest_path = Path(data_dir) / 'manifest.json'

        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load manifest {manifest_path}: {e}")

        if 'tokenizer_metadata' not in manifest:
            raise ValueError(
                f"❌ No tokenizer metadata in manifest: {manifest_path}\n"
                f"This dataset was processed with an older version.\n"
                f"Please re-process this dataset."
            )

        tokenizer_metadata = manifest['tokenizer_metadata']

        if reference_metadata is None:
            reference_metadata = tokenizer_metadata
            reference_dir = data_dir
            accelerator.print(f"📊 Reference tokenizer (from {data_dir}):")
            accelerator.print(f"   SHA256: {reference_metadata.get('tokenizer_sha256', 'N/A')[:16]}...")
            accelerator.print(f"   Vocab size: {reference_metadata.get('tokenizer_vocab_size', 'N/A'):,}")
        else:
            # Basic comparison for legacy format
            if (tokenizer_metadata.get('tokenizer_sha256') != reference_metadata.get('tokenizer_sha256')):
                raise ValueError(
                    f"❌ Tokenizer mismatch between {reference_dir} and {data_dir}\n"
                    f"Please ensure all datasets use the same tokenizer."
                )

            accelerator.print(f"✅ {data_dir}: tokenizer matches reference")

    return reference_metadata


def set_deterministic_training(seed: int = 42):
    """Configure deterministic training with fixed seed."""
    print(f"🎲 Configuring deterministic training with seed {seed}")
    
    # Seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CUDNN configuration for reproducibility (slower but deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Generator for DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    
    print("✅ Deterministic training configured")
    print(f"   Random state: {hash(str(random.getstate()))}")
    print(f"   NumPy state: {hash(str(np.random.get_state()))}")
    print(f"   PyTorch state: {hash(str(torch.get_rng_state()))}")
    
    return g


def validate_training_resumption(accelerator, global_step: int, scheduler, optimizer):
    """Validate that training resumption is working correctly."""
    accelerator.print(f"\n🔍 Training resumption validation:")
    accelerator.print(f"   Global step: {global_step}")
    accelerator.print(f"   Learning rate: {scheduler.get_last_lr()[0]:.2e}")

    # Handle AcceleratedScheduler wrapper - access the underlying scheduler
    try:
        if hasattr(scheduler, 'scheduler'):
            # AcceleratedScheduler has a .scheduler attribute
            last_epoch = scheduler.scheduler.last_epoch
        elif hasattr(scheduler, 'last_epoch'):
            # Direct access
            last_epoch = scheduler.last_epoch
        else:
            # Fallback - use global_step as approximation
            last_epoch = global_step
        accelerator.print(f"   Scheduler last_epoch: {last_epoch}")
    except Exception as e:
        accelerator.print(f"   ⚠️  Could not access scheduler last_epoch: {e}")
        last_epoch = global_step

    accelerator.print(f"   Optimizer state groups: {len(optimizer.state_dict()['state'])}")

    # Check if optimizer has momentum/variance terms (indicating it's not fresh)
    if len(optimizer.state_dict()['state']) > 0:
        accelerator.print(f"   ✅ Optimizer state loaded (has momentum/variance terms)")
    else:
        accelerator.print(f"   ⚠️  Optimizer state is empty (fresh start)")

    # Check scheduler consistency
    expected_lr_step = global_step
    if abs(last_epoch - expected_lr_step) > 1:
        accelerator.print(f"   ⚠️  Scheduler step mismatch: last_epoch={last_epoch}, expected ~{expected_lr_step}")
    else:
        accelerator.print(f"   ✅ Scheduler step is consistent")

    accelerator.print("🔍 Validation complete\n")


class PretrainConfig:
    """Configuration for pre-training."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Model parameters
        self.model_name = model_config['model_name']
        self.n_layer = model_config['n_layer']
        self.d_model = model_config['d_model']
        self.n_head = model_config['n_head']
        self.d_ff = model_config['d_ff']
        self.vocab_size = model_config['vocab_size']
        self.sequence_length = model_config['sequence_length']
        self.dropout = model_config.get('dropout', 0.1)
        
        # Training hyperparameters (defaults)
        self.learning_rate = 3e-4
        self.batch_size = 8
        self.gradient_accumulation_steps = 8
        self.num_train_epochs = 1
        self.max_steps = -1
        self.warmup_steps = 2000
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.max_grad_norm = 1.0
        self.logging_steps = 100
        self.save_steps = 5000
        self.eval_steps = 1000


def calculate_perplexity(model: nn.Module, dataloader: DataLoader, device: torch.device, accelerator = None) -> float:
    """Calculate perplexity on a validation dataset."""
    model.eval()
    
    # For FlashAttention models, temporarily switch to eager attention and float32
    original_dtype = None
    original_attn_implementation = None
    
    if hasattr(model, 'dtype') and model.dtype == torch.float16:
        original_dtype = model.dtype
        
        # Switch to eager attention to avoid FlashAttention dtype issues
        if hasattr(model, 'config') and hasattr(model.config, '_attn_implementation'):
            original_attn_implementation = model.config._attn_implementation
            model.config._attn_implementation = "eager"
            if accelerator:
                accelerator.print("Debug: Switching to eager attention for evaluation")
        
        model = model.to(torch.float32)
        if accelerator:
            accelerator.print("Debug: Converting model to float32 for stable evaluation")
            
            # Check for NaN/Inf in model weights
            nan_params = 0
            inf_params = 0
            total_params = 0
            for name, param in model.named_parameters():
                total_params += param.numel()
                if torch.isnan(param).any():
                    nan_params += torch.isnan(param).sum().item()
                if torch.isinf(param).any():
                    inf_params += torch.isinf(param).sum().item()
            
            if nan_params > 0 or inf_params > 0:
                accelerator.print(f"Debug: Found {nan_params} NaN and {inf_params} Inf values in model weights out of {total_params} total parameters!")
            else:
                accelerator.print("Debug: Model weights are clean (no NaN/Inf)")
    
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            batch_count += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Convert loss to float32 to avoid numerical issues
            loss_value = float(loss.item())
            
            if accelerator and batch_count == 1:  # Debug only first batch
                accelerator.print(f"Debug: loss_value = {loss_value}")
            
            # Check if loss is valid
            if math.isnan(loss_value) or math.isinf(loss_value):
                if accelerator:
                    accelerator.print(f"Debug: Skipping batch due to invalid loss: {loss_value}")
                continue
            
            # Count only non-masked tokens
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss_value * valid_tokens
            total_tokens += valid_tokens
    
    if accelerator:
        accelerator.print(f"Debug: total_tokens = {total_tokens}, batch_count = {batch_count}")
    
    if total_tokens == 0:
        if accelerator:
            accelerator.print("Debug: Returning inf due to total_tokens == 0")
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    
    # Debug: display average loss
    if accelerator:
        accelerator.print(f"Debug: avg_loss = {avg_loss:.4f}, total_tokens = {total_tokens}")
    else:
        print(f"Debug: avg_loss = {avg_loss:.4f}, total_tokens = {total_tokens}")
    
    # Clamp avg_loss to avoid overflow in exp() 
    # exp(20) ≈ 485M, exp(30) ≈ 10^13, more than sufficient
    avg_loss = min(avg_loss, 20.0)
    
    perplexity = math.exp(avg_loss)
    
    # Restore original configuration and dtype
    if original_attn_implementation is not None:
        model.config._attn_implementation = original_attn_implementation
        if accelerator:
            accelerator.print("Debug: Restoring FlashAttention implementation")
    
    if original_dtype is not None:
        model = model.to(original_dtype)
        if accelerator:
            accelerator.print("Debug: Restoring model to float16")
    
    model.train()
    return perplexity


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
    config: PretrainConfig,
    epoch: int,
    global_step: int,
    use_flash_attention: bool = False,
    mixed_precision: str = "no"
) -> tuple[int, float]:
    """Train the model for one epoch (single dataset mode)."""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
        with accelerator.accumulate(model):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # Let Accelerate handle mixed precision automatically
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                # Use accelerator's gradient clipping (handles mixed precision correctly)
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if accelerator.sync_gradients:
            global_step += 1
            total_loss += loss.item()
            
            # Logging
            if global_step % config.logging_steps == 0:
                avg_loss = total_loss / config.logging_steps
                current_lr = scheduler.get_last_lr()[0]
                accelerator.print(f"Step {global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")
                total_loss = 0.0
            
            # Save checkpoints
            if global_step % config.save_steps == 0:
                if accelerator.is_main_process:
                    checkpoint_dir = f"{config.output_dir}/{config.model_name}/step_{global_step}"
                    save_checkpoint(model, optimizer, scheduler, global_step, checkpoint_dir, accelerator)
                    accelerator.print(f"💾 Checkpoint saved at step {global_step} -> {checkpoint_dir}")
        
        progress_bar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
        
        # Early stopping if max_steps reached
        if config.max_steps > 0 and global_step >= config.max_steps:
            break
    
    return global_step, total_loss


def train_multi_dataset(
    model: nn.Module,
    multi_dataset_sampler: WeightedMultiDatasetSampler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
    config: PretrainConfig,
    global_step: int,
    max_steps: int,
    log_dataset_mix_steps: int = 500,
    val_dataloader: DataLoader = None
) -> tuple[int, float]:
    """Train the model using multi-dataset sampler (step-based training)."""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(total=max_steps, desc="Multi-dataset training", initial=global_step)
    
    while global_step < max_steps:
        # Get batch from multi-dataset sampler
        batch = multi_dataset_sampler.get_batch()
        
        # Move batch to device
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        
        with accelerator.accumulate(model):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if accelerator.sync_gradients:
            global_step += 1
            total_loss += loss.item()
            progress_bar.update(1)
            
            # Logging
            if global_step % config.logging_steps == 0:
                avg_loss = total_loss / config.logging_steps
                current_lr = scheduler.get_last_lr()[0]
                accelerator.print(f"Step {global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")
                total_loss = 0.0
            
            # Log dataset mixing every N steps
            if global_step % log_dataset_mix_steps == 0:
                mix_observed = multi_dataset_sampler.get_observed_mix()
                mix_str = ", ".join([f"{name}={ratio:.2f}" for name, ratio in mix_observed.items()])
                accelerator.print(f"Step {global_step}: mix_observed: {mix_str}")
                multi_dataset_sampler.reset_mix_tracking()  # Reset for next interval
            
            # Evaluation
            if global_step % config.eval_steps == 0 and val_dataloader is not None:
                if accelerator.is_main_process:
                    accelerator.print("Evaluation...")
                    perplexity = calculate_perplexity(model, val_dataloader, accelerator.device, accelerator)
                    accelerator.print(f"Step {global_step}: Validation perplexity: {perplexity:.2f}")

            # Save checkpoints
            if global_step % config.save_steps == 0:
                if accelerator.is_main_process:
                    checkpoint_dir = f"{config.output_dir}/{config.model_name}/step_{global_step}"
                    # Save multi-dataset sampler state as well
                    sampler_state = multi_dataset_sampler.state_dict()
                    # Correct the sampler step to match the actual training step
                    sampler_state['current_step'] = global_step
                    save_checkpoint(
                        model, optimizer, scheduler, global_step, checkpoint_dir,
                        accelerator, extra_state={"multi_dataset_sampler": sampler_state}
                    )
                    accelerator.print(f"💾 Multi-dataset checkpoint saved at step {global_step} -> {checkpoint_dir}")

            # Stop if max steps reached
            if global_step >= max_steps:
                break
    
    progress_bar.close()
    return global_step, total_loss


def main():
    parser = argparse.ArgumentParser(description="Pre-training of a mini-LLM")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to the model configuration file")
    parser.add_argument("--data_path", type=str,
                       help="Path to tokenized data file (legacy format)")
    parser.add_argument("--data_dir", type=str,
                       help="Path to directory with packed data (.bin + .idx files)")
    parser.add_argument("--tokenizer_dir", type=str,
                       help="Path to directory containing tokenizer for validation")

    # Multi-dataset training arguments
    parser.add_argument("--data_dirs", nargs='+', type=str,
                       help="Multiple data directories for weighted multi-dataset training")
    parser.add_argument("--data_weights", nargs='+', type=float,
                       help="Weights for multi-dataset sampling (must match --data_dirs length)")
    parser.add_argument("--num_workers", type=int, default=1,
                       help="Number of worker processes for data loading (multi-dataset)")
    parser.add_argument("--log_dataset_mix_steps", type=int, default=500,
                       help="Log observed dataset mix every N steps")
    
    # Compatibility: allow either data_path or data_dir
    parser.add_argument("--output_dir", type=str, default="./checkpoints/pretrain",
                       help="Output directory for checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to the checkpoint to resume from, or 'auto' to automatically find the latest checkpoint")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                       help="Maximum number of steps (-1 for unlimited)")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                       help="Number of warmup steps")
    parser.add_argument("--save_steps", type=int, default=5000,
                       help="Save interval")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Logging interval")
    parser.add_argument("--use_flash_attn", action="store_true", default=True,
                       help="Use FlashAttention-2 (default: True)")
    parser.add_argument("--no_flash_attn", action="store_true",
                       help="Disable FlashAttention-2 (force fallback SDPA)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed for reproducibility")
    parser.add_argument("--no_deterministic", action="store_true",
                       help="Disable deterministic training (faster)")
    
    args = parser.parse_args()
    
    # FlashAttention-2 management - use Accelerate's mixed_precision
    use_flash_attention = args.use_flash_attn and not args.no_flash_attn
    mixed_precision = "fp16" if use_flash_attention else "no"
    
    # Initialize accelerate  
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb" if os.getenv("WANDB_API_KEY") else None,
        kwargs_handlers=[ddp_kwargs]
    )
    
    # Configuration
    config = PretrainConfig(args.config)
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.num_train_epochs = args.num_train_epochs
    config.max_steps = args.max_steps
    config.warmup_steps = args.warmup_steps
    config.save_steps = args.save_steps
    config.logging_steps = args.logging_steps
    config.output_dir = args.output_dir
    
    # Deterministic configuration
    dataloader_generator = None
    if not args.no_deterministic:
        dataloader_generator = set_deterministic_training(args.seed)
    
    # FlashAttention-2 already configured above
    
    # Model creation
    accelerator.print("Creating model...")
    model = create_model(config, use_flash_attention=use_flash_attention)
    accelerator.print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Data loading - with multi-dataset support
    accelerator.print("Loading data...")
    use_multi_dataset = False
    multi_dataset_sampler = None
    tokenizer_metadata = {}
    
    if args.data_dirs:
        # Multi-dataset mode
        accelerator.print(f"🌐 Using multi-dataset mode with {len(args.data_dirs)} datasets")
        
        # Validate arguments
        if args.data_weights and len(args.data_weights) != len(args.data_dirs):
            raise ValueError(f"Number of weights ({len(args.data_weights)}) must match number of data_dirs ({len(args.data_dirs)})")
        
        # CRITICAL: Validate tokenizer consistency across all datasets
        if not args.tokenizer_dir:
            raise ValueError("--tokenizer_dir is required for multi-dataset training")
        tokenizer_metadata = validate_tokenizer_consistency(args.data_dirs, args.tokenizer_dir, accelerator)
        
        # Initialize multi-dataset sampler
        multi_dataset_sampler = WeightedMultiDatasetSampler(
            data_dirs=args.data_dirs,
            weights=args.data_weights,
            seed=args.seed,
            batch_size=args.batch_size,
            split="train"
        )
        use_multi_dataset = True
        train_dataset = None  # Not used in multi-dataset mode
        
    elif args.data_dir:
        accelerator.print(f"📁 Using single packed data directory: {args.data_dir}")

        # Check if it's new PackedDataset format or legacy format
        final_manifest_path = Path(args.data_dir) / 'final_manifest.json'
        if final_manifest_path.exists():
            # New PackedDataset format
            accelerator.print("✅ Detected PackedDataset format (.bin + .idx files)")

            if args.tokenizer_dir:
                tokenizer_metadata = validate_tokenizer_consistency([args.data_dir], args.tokenizer_dir, accelerator)
            else:
                accelerator.print("⚠️  Warning: No tokenizer validation (--tokenizer_dir not provided)")
                tokenizer_metadata = {}

            train_dataset = PackedDataset(args.data_dir, split="train")
        else:
            # Legacy format
            accelerator.print("ℹ️  Using legacy TokenizedDataset format")
            if args.tokenizer_dir:
                tokenizer_metadata = validate_tokenizer_consistency([args.data_dir], args.tokenizer_dir, accelerator)
            else:
                tokenizer_metadata = {}
            train_dataset = TokenizedDataset(args.data_dir, config.sequence_length)
    elif args.data_path:
        accelerator.print(f"📄 Using legacy data file: {args.data_path}")
        train_dataset = TokenizedDataset(args.data_path, config.sequence_length)
    else:
        raise ValueError("Either --data_dirs, --data_dir, or --data_path must be specified")
    
    # Setup data loaders based on mode
    if use_multi_dataset:
        # Multi-dataset mode: no traditional DataLoader needed for training
        train_dataloader = None

        # Try to create validation dataloader from the first high-quality dataset
        val_dataloader = None
        try:
            # Use the first dataset for validation (assuming it's high quality)
            first_data_dir = args.data_dirs[0]
            final_manifest_path = Path(first_data_dir) / 'final_manifest.json'

            if final_manifest_path.exists():
                # Try PackedDataset validation split
                try:
                    val_dataset = PackedDataset(first_data_dir, split="val")
                    val_dataloader = DataLoader(
                        val_dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True,
                        generator=dataloader_generator
                    )
                    accelerator.print(f"📊 Multi-dataset validation: using {len(val_dataset)} samples from {Path(first_data_dir).name}")
                except (FileNotFoundError, Exception) as e:
                    accelerator.print(f"ℹ️  No validation data found in {first_data_dir}: {e}")
            else:
                # Legacy format - try to create validation from ShardedTokenizedDataset
                try:
                    val_dataset = ShardedTokenizedDataset(first_data_dir, split="val")
                    val_dataloader = DataLoader(
                        val_dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True,
                        generator=dataloader_generator
                    )
                    accelerator.print(f"📊 Multi-dataset validation: using {len(val_dataset)} samples from {Path(first_data_dir).name} (legacy)")
                except Exception as e:
                    accelerator.print(f"ℹ️  No validation data found in legacy format: {e}")

        except Exception as e:
            accelerator.print(f"⚠️  Could not create validation dataloader for multi-dataset mode: {e}")
            accelerator.print("   Training will continue without validation")

        total_samples = multi_dataset_sampler.get_total_samples()
        accelerator.print(f"📊 Multi-dataset total samples: {total_samples:,}")

    else:
        # Single dataset mode (backward compatibility)
        # Check if we have separate validation data or need to split
        val_dataset = None

        if isinstance(train_dataset, PackedDataset):
            # Try to load validation split for PackedDataset
            try:
                val_dataset = PackedDataset(args.data_dir, split="val")
                accelerator.print(f"📊 Using separate validation data: {len(val_dataset)} samples")
            except (FileNotFoundError, Exception):
                accelerator.print("ℹ️  No separate validation data found, will split training data")

        if val_dataset is None:
            # Split the training dataset (90/10)
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            accelerator.print(f"📊 Split dataset: {train_size} train, {val_size} val")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            generator=dataloader_generator
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            generator=dataloader_generator
        )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    # Calculate total number of steps
    if config.max_steps > 0:
        max_train_steps = config.max_steps
    else:
        max_train_steps = len(train_dataloader) * config.num_train_epochs
        max_train_steps = max_train_steps // config.gradient_accumulation_steps
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=max_train_steps
    )
    
    # Prepare with accelerate
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    
    # Get reference to the scaler if using mixed precision
    scaler = getattr(accelerator, "scaler", None)
    if scaler is not None:
        accelerator.print(f"🔧 Mixed precision scaler detected: {type(scaler).__name__}")
    else:
        accelerator.print("🔧 No mixed precision scaler (using full precision or automatic mixed precision)")
    
    # Resume from checkpoint if specified
    global_step = 0
    start_epoch = 0

    # Handle 'auto' checkpoint detection
    resume_checkpoint_path = args.resume_from_checkpoint
    if args.resume_from_checkpoint == "auto":
        accelerator.print("🔍 Auto-detecting latest checkpoint...")
        resume_checkpoint_path = find_latest_checkpoint(args.output_dir, config.model_name)
        if resume_checkpoint_path:
            accelerator.print(f"✅ Found latest checkpoint: {resume_checkpoint_path}")
        else:
            accelerator.print("ℹ️  No existing checkpoints found, starting fresh training")
            resume_checkpoint_path = None

    if resume_checkpoint_path:
        accelerator.print(f"🔄 Resuming training from checkpoint: {resume_checkpoint_path}")
        
        try:
            # Load complete checkpoint state (includes model, optimizer, scheduler, scaler, RNG states)
            global_step, extra_state = load_checkpoint(
                model, optimizer, scheduler, resume_checkpoint_path, accelerator
            )
            
            # Restore multi-dataset sampler state if resuming multi-dataset training
            if use_multi_dataset and "multi_dataset_sampler" in extra_state:
                multi_dataset_sampler.load_state_dict(extra_state["multi_dataset_sampler"])
                accelerator.print("📊 Multi-dataset sampler state restored")
            
            if not use_multi_dataset:
                start_epoch = global_step // len(train_dataloader)
            
            accelerator.print(f"✅ Successfully resumed from step {global_step}")
            
            # Validate that resumption is working correctly
            validate_training_resumption(accelerator, global_step, scheduler, optimizer)
            
        except Exception as e:
            accelerator.print(f"❌ Failed to load checkpoint: {e}")
            accelerator.print("Continuing with fresh training...")
            global_step = 0
            start_epoch = 0
    
    # Log tokenizer information before training starts
    if tokenizer_metadata:
        accelerator.print("\n🔒 Tokenizer Information:")
        accelerator.print(f"   SHA256: {tokenizer_metadata.get('tokenizer_config_hash', tokenizer_metadata.get('tokenizer_sha256', 'N/A'))}")
        accelerator.print(f"   Vocab Size: {tokenizer_metadata.get('vocab_size', tokenizer_metadata.get('tokenizer_vocab_size', 'N/A'))}")

        if use_multi_dataset and args.data_dirs:
            accelerator.print(f"   Datasets using this tokenizer:")
            for i, data_dir in enumerate(args.data_dirs):
                weight = args.data_weights[i] if args.data_weights else "auto"
                accelerator.print(f"     - {data_dir} (weight: {weight})")
    
    # Training loop
    accelerator.print("\nStarting training...")
    
    if use_multi_dataset:
        # Multi-dataset training (step-based)
        accelerator.print(f"🌐 Starting multi-dataset training for {config.max_steps} steps")
        
        global_step, _ = train_multi_dataset(
            model, multi_dataset_sampler, optimizer, scheduler,
            accelerator, config, global_step, config.max_steps, args.log_dataset_mix_steps,
            val_dataloader
        )
        
    else:
        # Traditional epoch-based training (backward compatibility)
        for epoch in range(start_epoch, config.num_train_epochs):
            accelerator.print(f"Epoch {epoch + 1}/{config.num_train_epochs}")
            
            global_step, _ = train_epoch(
                model, train_dataloader, optimizer, scheduler, 
                accelerator, config, epoch + 1, global_step, use_flash_attention, mixed_precision
            )
            
            # Evaluation
            if accelerator.is_main_process and (epoch + 1) % 1 == 0:  # Evaluation every epoch
                accelerator.print("Evaluation...")
                perplexity = calculate_perplexity(model, val_dataloader, accelerator.device, accelerator)
                accelerator.print(f"Validation perplexity: {perplexity:.2f}")
            
            # Stop if max_steps reached
            if config.max_steps > 0 and global_step >= config.max_steps:
                break
    
    # Final save with tokenizer metadata
    if accelerator.is_main_process:
        final_dir = f"{args.output_dir}/{config.model_name}/final"
        
        # Prepare extra metadata including tokenizer information
        extra_state = {}
        if tokenizer_metadata:
            extra_state["tokenizer_metadata"] = tokenizer_metadata
        if use_multi_dataset and args.data_dirs:
            extra_state["training_datasets"] = args.data_dirs
            if args.data_weights:
                extra_state["dataset_weights"] = args.data_weights
        
        save_checkpoint(model, optimizer, scheduler, global_step, final_dir, accelerator, extra_state=extra_state)
        accelerator.print(f"Final model saved in {final_dir}")
        
        # Save tokenizer metadata to a separate JSON file for easy access
        if tokenizer_metadata:
            metadata_path = Path(final_dir) / "tokenizer_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "tokenizer_metadata": tokenizer_metadata,
                    "training_datasets": args.data_dirs if args.data_dirs else [args.data_dir] if args.data_dir else [],
                    "dataset_weights": args.data_weights if args.data_weights else [],
                    "training_date": str(Path().cwd()),  # Will be replaced by actual date in practice
                }, f, indent=2)
            accelerator.print(f"📄 Tokenizer metadata saved: {metadata_path}")
    
    accelerator.print("Training completed!")


if __name__ == "__main__":
    main()