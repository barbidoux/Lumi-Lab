#!/usr/bin/env python3
"""
DPO utilities for Lumi-Lab pipeline.

This module provides utilities for Direct Preference Optimization (DPO):
- Multi-dataset loading and weighted sampling
- DPO example validation
- Tokenizer consistency verification
- Reward metrics calculation
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset


logger = logging.getLogger(__name__)


class DPOMultiDatasetLoader:
    """Load and manage multiple DPO datasets with weighted sampling."""

    def __init__(
        self,
        data_dirs: List[str],
        weights: Optional[List[float]] = None,
        tokenizer_dir: Optional[str] = None,
        validate_tokenizer: bool = True
    ):
        """
        Initialize DPO multi-dataset loader.

        Args:
            data_dirs: List of directories containing DPO corpus shards
            weights: Optional weights for dataset sampling (must sum to 1.0)
            tokenizer_dir: Path to tokenizer directory for validation
            validate_tokenizer: Whether to validate tokenizer consistency
        """
        self.data_dirs = [Path(d) for d in data_dirs]
        self.weights = weights
        self.tokenizer_dir = tokenizer_dir
        self.validate_tokenizer = validate_tokenizer

        # Validate inputs
        if self.weights is not None:
            if len(self.weights) != len(self.data_dirs):
                raise ValueError(f"Number of weights ({len(self.weights)}) must match number of data_dirs ({len(self.data_dirs)})")

            # Normalize weights to sum to 1.0
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]
        else:
            # Uniform weights
            self.weights = [1.0 / len(self.data_dirs)] * len(self.data_dirs)

        self.datasets = []
        self.manifests = []

    def load_datasets(self) -> List[Dataset]:
        """
        Load all DPO datasets from corpus directories.

        Returns:
            List of loaded datasets
        """
        logger.info(f"📂 Loading {len(self.data_dirs)} DPO dataset(s)...")

        for i, data_dir in enumerate(self.data_dirs):
            if not data_dir.exists():
                raise FileNotFoundError(f"DPO corpus directory not found: {data_dir}")

            # Load manifest
            manifest_path = data_dir / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")

            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            self.manifests.append(manifest)

            # Validate tokenizer if requested
            if self.validate_tokenizer and self.tokenizer_dir:
                self._validate_tokenizer_consistency(manifest, data_dir)

            # Load all shards
            shard_files = sorted(data_dir.glob("shard_*.jsonl"))
            if not shard_files:
                raise FileNotFoundError(f"No shard files found in {data_dir}")

            logger.info(f"   Dataset {i+1}/{len(self.data_dirs)}: {manifest['dataset_name']}")
            logger.info(f"      Samples: {manifest['total_samples']:,}")
            logger.info(f"      Shards: {len(shard_files)}")
            logger.info(f"      Weight: {self.weights[i]:.2%}")

            # Load dataset from JSONL shards
            dataset = load_dataset(
                'json',
                data_files=[str(f) for f in shard_files],
                split='train'
            )

            self.datasets.append(dataset)

        logger.info(f"✅ Loaded {len(self.datasets)} DPO dataset(s)")
        return self.datasets

    def create_weighted_dataset(self) -> Dataset:
        """
        Create a weighted combined dataset from all loaded datasets.

        Returns:
            Combined dataset with weighted sampling
        """
        if not self.datasets:
            raise ValueError("No datasets loaded. Call load_datasets() first.")

        logger.info("🔀 Creating weighted combined dataset...")

        # Calculate number of samples to take from each dataset
        total_samples = sum(len(ds) for ds in self.datasets)
        weighted_samples = []

        for i, (dataset, weight) in enumerate(zip(self.datasets, self.weights)):
            # Number of samples proportional to weight
            num_samples = int(len(dataset) * weight * len(self.datasets))
            num_samples = min(num_samples, len(dataset))  # Don't exceed dataset size

            # Sample with replacement if needed
            if num_samples > len(dataset):
                # Oversample
                indices = np.random.choice(len(dataset), size=num_samples, replace=True)
            else:
                # Subsample or use all
                indices = np.random.choice(len(dataset), size=num_samples, replace=False)

            sampled_dataset = dataset.select(indices)
            weighted_samples.append(sampled_dataset)

            logger.info(f"   Dataset {i+1}: {num_samples:,} samples (weight={weight:.2%})")

        # Concatenate all weighted datasets
        combined_dataset = concatenate_datasets(weighted_samples)

        # Shuffle
        combined_dataset = combined_dataset.shuffle(seed=42)

        logger.info(f"✅ Combined dataset created: {len(combined_dataset):,} samples")
        return combined_dataset

    def _validate_tokenizer_consistency(self, manifest: Dict, data_dir: Path) -> None:
        """Validate tokenizer consistency between corpus and current tokenizer."""
        from utils.tokenizer_utils import verify_tokenizer_consistency

        if 'tokenizer_config_hash' not in manifest:
            logger.warning(f"⚠️ No tokenizer hash in manifest for {data_dir.name}")
            return

        # Load tokenizer config
        tokenizer_config_path = Path(self.tokenizer_dir) / "tokenizer_config.json"
        if not tokenizer_config_path.exists():
            raise FileNotFoundError(f"Tokenizer config not found: {tokenizer_config_path}")

        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)

        expected_hash = manifest['tokenizer_config_hash']
        actual_hash = tokenizer_config.get('sha256_hash', '')

        if expected_hash != actual_hash:
            raise ValueError(
                f"❌ Tokenizer mismatch for dataset {data_dir.name}!\n"
                f"   Expected hash: {expected_hash}\n"
                f"   Actual hash:   {actual_hash}\n"
                f"   This means the DPO corpus was prepared with a different tokenizer.\n"
                f"   Re-run 65_prepare_dpo_corpus.py with the correct tokenizer."
            )

        logger.debug(f"✅ Tokenizer validation passed for {data_dir.name}")


def validate_dpo_example(example: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate a single DPO example.

    Args:
        example: DPO example with 'prompt', 'chosen', 'rejected' fields

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_keys = ['prompt', 'chosen', 'rejected']

    # Check required keys
    for key in required_keys:
        if key not in example:
            return False, f"Missing required key: {key}"

    # Check types
    for key in required_keys:
        if not isinstance(example[key], str):
            return False, f"Field '{key}' must be a string, got {type(example[key])}"

    # Check non-empty
    for key in required_keys:
        if len(example[key].strip()) == 0:
            return False, f"Field '{key}' cannot be empty"

    # Check chosen != rejected
    if example['chosen'] == example['rejected']:
        return False, "Chosen and rejected responses must be different"

    return True, None


def validate_dpo_dataset(dataset: Dataset, sample_size: int = 1000) -> Dict[str, Any]:
    """
    Validate a DPO dataset by sampling examples.

    Args:
        dataset: DPO dataset to validate
        sample_size: Number of examples to validate

    Returns:
        Validation statistics
    """
    logger.info(f"🔍 Validating DPO dataset ({sample_size} samples)...")

    sample_size = min(sample_size, len(dataset))
    indices = np.random.choice(len(dataset), size=sample_size, replace=False)

    valid_count = 0
    error_counts = {}

    for idx in indices:
        example = dataset[int(idx)]
        is_valid, error_msg = validate_dpo_example(example)

        if is_valid:
            valid_count += 1
        else:
            error_counts[error_msg] = error_counts.get(error_msg, 0) + 1

    validation_rate = valid_count / sample_size

    stats = {
        'total_sampled': sample_size,
        'valid_count': valid_count,
        'invalid_count': sample_size - valid_count,
        'validation_rate': validation_rate,
        'error_counts': error_counts
    }

    logger.info(f"   Validation rate: {validation_rate:.2%} ({valid_count}/{sample_size})")

    if error_counts:
        logger.warning("⚠️ Validation errors found:")
        for error_msg, count in error_counts.items():
            logger.warning(f"   - {error_msg}: {count} occurrences")

    return stats


def calculate_reward_margin(
    chosen_rewards: np.ndarray,
    rejected_rewards: np.ndarray
) -> float:
    """
    Calculate average reward margin (chosen - rejected).

    Args:
        chosen_rewards: Array of chosen response rewards
        rejected_rewards: Array of rejected response rewards

    Returns:
        Average reward margin
    """
    if len(chosen_rewards) != len(rejected_rewards):
        raise ValueError("Chosen and rejected reward arrays must have same length")

    margins = chosen_rewards - rejected_rewards
    return float(np.mean(margins))


def calculate_win_rate(
    chosen_rewards: np.ndarray,
    rejected_rewards: np.ndarray
) -> float:
    """
    Calculate win rate (fraction where chosen > rejected).

    Args:
        chosen_rewards: Array of chosen response rewards
        rejected_rewards: Array of rejected response rewards

    Returns:
        Win rate (0.0 to 1.0)
    """
    if len(chosen_rewards) != len(rejected_rewards):
        raise ValueError("Chosen and rejected reward arrays must have same length")

    wins = chosen_rewards > rejected_rewards
    return float(np.mean(wins))


def calculate_reward_accuracy(
    chosen_rewards: np.ndarray,
    rejected_rewards: np.ndarray,
    margin_threshold: float = 0.0
) -> float:
    """
    Calculate reward accuracy with margin threshold.

    Args:
        chosen_rewards: Array of chosen response rewards
        rejected_rewards: Array of rejected response rewards
        margin_threshold: Minimum margin to count as correct

    Returns:
        Accuracy (0.0 to 1.0)
    """
    if len(chosen_rewards) != len(rejected_rewards):
        raise ValueError("Chosen and rejected reward arrays must have same length")

    margins = chosen_rewards - rejected_rewards
    correct = margins > margin_threshold
    return float(np.mean(correct))


def compute_dpo_statistics(
    chosen_rewards: np.ndarray,
    rejected_rewards: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive DPO statistics.

    Args:
        chosen_rewards: Array of chosen response rewards
        rejected_rewards: Array of rejected response rewards

    Returns:
        Dictionary of statistics
    """
    margins = chosen_rewards - rejected_rewards

    stats = {
        'reward_margin_mean': float(np.mean(margins)),
        'reward_margin_std': float(np.std(margins)),
        'reward_margin_median': float(np.median(margins)),
        'reward_margin_min': float(np.min(margins)),
        'reward_margin_max': float(np.max(margins)),
        'win_rate': calculate_win_rate(chosen_rewards, rejected_rewards),
        'chosen_reward_mean': float(np.mean(chosen_rewards)),
        'chosen_reward_std': float(np.std(chosen_rewards)),
        'rejected_reward_mean': float(np.mean(rejected_rewards)),
        'rejected_reward_std': float(np.std(rejected_rewards)),
        'accuracy_0.0': calculate_reward_accuracy(chosen_rewards, rejected_rewards, 0.0),
        'accuracy_0.1': calculate_reward_accuracy(chosen_rewards, rejected_rewards, 0.1),
        'accuracy_0.5': calculate_reward_accuracy(chosen_rewards, rejected_rewards, 0.5),
    }

    return stats


def verify_dpo_corpus_tokenizer(corpus_dir: str, tokenizer_dir: str) -> bool:
    """
    Verify tokenizer consistency for DPO corpus.

    Args:
        corpus_dir: Path to DPO corpus directory
        tokenizer_dir: Path to tokenizer directory

    Returns:
        True if consistent, raises ValueError otherwise
    """
    corpus_path = Path(corpus_dir)
    tokenizer_path = Path(tokenizer_dir)

    # Load manifest
    manifest_path = corpus_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # Load tokenizer config
    tokenizer_config_path = tokenizer_path / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        raise FileNotFoundError(f"Tokenizer config not found: {tokenizer_config_path}")

    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
        tokenizer_config = json.load(f)

    # Compare hashes
    expected_hash = manifest.get('tokenizer_config_hash', '')
    actual_hash = tokenizer_config.get('sha256_hash', '')

    if expected_hash != actual_hash:
        raise ValueError(
            f"❌ Tokenizer mismatch!\n"
            f"   Corpus: {corpus_path.name}\n"
            f"   Expected hash: {expected_hash}\n"
            f"   Actual hash:   {actual_hash}\n"
            f"   Re-run corpus preparation with correct tokenizer."
        )

    logger.info(f"✅ Tokenizer verification passed for {corpus_path.name}")
    return True
