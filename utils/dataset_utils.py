"""
Utilities for dataset management and data loading.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import random
import math


class ShardedTokenizedDataset(Dataset):
    """Dataset for sharded tokenized data with manifest support."""
    
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir: Path to directory containing sharded data
            split: Dataset split ('train' or 'val')
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Check for manifest.json
        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            print(f"📋 Loading dataset from manifest: {manifest_path}")
            self.shard_files = self._load_from_manifest(manifest_path, split)
        else:
            print(f"🔍 No manifest found, using glob fallback in {self.data_dir}")
            self.shard_files = self._load_from_glob(self.data_dir, split)
        
        if not self.shard_files:
            raise FileNotFoundError(f"No {split} shards found in {data_dir}")
        
        print(f"Found {len(self.shard_files)} {split} shards")
        
        # Load all data into memory
        self.samples = []
        total_tokens = 0
        
        for shard_file in self.shard_files:
            shard_path = self.data_dir / shard_file
            with open(shard_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
                    labels = torch.tensor(sample['labels'], dtype=torch.long)
                    self.samples.append({
                        'input_ids': input_ids,
                        'labels': labels,
                        'attention_mask': torch.ones_like(input_ids)
                    })
                    total_tokens += len(sample['input_ids'])
        
        print(f"📊 Dataset loaded: {len(self.samples):,} samples, {total_tokens:,} tokens")
    
    def _load_from_manifest(self, manifest_path: Path, split: str) -> List[str]:
        """Load shard files from manifest."""
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        shard_key = f"{split}_shards"
        if shard_key not in manifest:
            raise KeyError(f"Manifest does not contain '{shard_key}' key")
        
        return manifest[shard_key]
    
    def _load_from_glob(self, data_dir: Path, split: str) -> List[str]:
        """Fallback: load shard files using glob."""
        pattern = str(data_dir / f"{split}_*.jsonl")
        shard_paths = glob.glob(pattern)
        
        # Convert to relative filenames
        return [Path(p).name for p in sorted(shard_paths)]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class TokenizedDataset(Dataset):
    """Legacy dataset for tokenized data saved in JSON format."""
    
    def __init__(self, data_path: str, sequence_length: int = 1024, stride: int = None):
        """
        Args:
            data_path: Path to JSON file containing tokenized data or directory with shards
            sequence_length: Sequence length for training
            stride: Stride between sequences (default = sequence_length)
        """
        data_path = Path(data_path)
        
        # Check if path is a directory (new sharded format)
        if data_path.is_dir():
            print("📁 Detected directory path, delegating to ShardedTokenizedDataset")
            sharded_dataset = ShardedTokenizedDataset(str(data_path), split="train")
            # Copy attributes for compatibility
            self.samples = sharded_dataset.samples
            self.num_sequences = len(self.samples)
            self.sequence_length = sequence_length
            return
        
        # Legacy single file format
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        
        # Data loading
        print(f"Loading legacy format data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.tokenized_texts = json.load(f)
        
        # Concatenation of all tokens
        self.all_tokens = []
        for tokens in self.tokenized_texts:
            self.all_tokens.extend(tokens)
        
        self.all_tokens = torch.tensor(self.all_tokens, dtype=torch.long)
        
        # Calculate number of sequences
        self.num_sequences = max(0, (len(self.all_tokens) - sequence_length) // self.stride + 1)
        
        print(f"Dataset loaded: {len(self.tokenized_texts)} texts, "
              f"{len(self.all_tokens):,} tokens, "
              f"{self.num_sequences:,} sequences of length {sequence_length}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # New sharded format
        if hasattr(self, 'samples'):
            return self.samples[idx]
        
        # Legacy format
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        
        # Extract sequence
        sequence = self.all_tokens[start_idx:end_idx]
        
        # For training, input_ids and labels are shifted by one token
        input_ids = sequence[:-1]
        labels = sequence[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones_like(input_ids)
        }


class ConversationalDataset(Dataset):
    """Dataset for conversations (SFT, DPO)."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048, format_template: str = "chat"):
        """
        Args:
            data_path: Path to the dataset
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            format_template: Format template ("chat", "instruct", etc.)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_template = format_template
        
        # Data loading
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"File format not supported: {data_path}")
        
        print(f"Conversational dataset loaded: {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def format_example(self, example: Dict) -> str:
        """Format an example according to the chosen template."""
        if self.format_template == "chat":
            return f"Human: {example['prompt']}\n\nAssistant: {example['response']}<|endoftext|>"
        elif self.format_template == "instruct":
            return f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}<|endoftext|>"
        else:
            # Custom format
            return f"{example['prompt']}\n{example['response']}<|endoftext|>"
    
    def __getitem__(self, idx):
        example = self.data[idx]
        formatted_text = self.format_example(example)
        
        # Tokenization
        encoded = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze().clone()
        }


def create_dataloader(dataset: Dataset, batch_size: int = 8, shuffle: bool = True, 
                     num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    """Create a DataLoader with optimized parameters."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Important for stable training
    )


def split_dataset(dataset: Dataset, train_ratio: float = 0.9, seed: int = 42):
    """Split a dataset into train/validation."""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    # Deterministic split
    torch.manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, val_size])


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collation function for batches."""
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key in ['input_ids', 'labels', 'attention_mask']:
            # Padding for sequences of different lengths
            sequences = [item[key] for item in batch]
            padded = torch.nn.utils.rnn.pad_sequence(
                sequences, 
                batch_first=True, 
                padding_value=0 if key != 'labels' else -100
            )
            collated[key] = padded
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    
    return collated


def load_and_prepare_dataset(
    data_path: str, 
    dataset_type: str = "pretrain",
    sequence_length: int = 1024,
    batch_size: int = 8,
    tokenizer = None,
    train_ratio: float = 0.9
) -> Dict[str, DataLoader]:
    """
    High-level function to load and prepare a dataset.
    
    Args:
        data_path: Path to the data
        dataset_type: Dataset type ("pretrain", "sft", "dpo")
        sequence_length: Sequence length
        batch_size: Batch size
        tokenizer: Tokenizer (required for sft/dpo)
        train_ratio: Train/validation ratio
    
    Returns:
        Dict containing train and validation DataLoaders
    """
    if dataset_type == "pretrain":
        dataset = TokenizedDataset(data_path, sequence_length)
        train_dataset, val_dataset = split_dataset(dataset, train_ratio)
        
        train_loader = create_dataloader(train_dataset, batch_size, shuffle=True)
        val_loader = create_dataloader(val_dataset, batch_size, shuffle=False)
        
    elif dataset_type in ["sft", "dpo"]:
        if tokenizer is None:
            raise ValueError("Tokenizer required for SFT/DPO datasets")
        
        dataset = ConversationalDataset(data_path, tokenizer, sequence_length)
        train_dataset, val_dataset = split_dataset(dataset, train_ratio)
        
        train_loader = create_dataloader(train_dataset, batch_size, shuffle=True)
        val_loader = create_dataloader(val_dataset, batch_size, shuffle=False)
    
    else:
        raise ValueError(f"Dataset type not supported: {dataset_type}")
    
    return {
        "train": train_loader,
        "validation": val_loader,
        "train_size": len(train_dataset) if 'train_dataset' in locals() else 0,
        "val_size": len(val_dataset) if 'val_dataset' in locals() else 0
    }


def get_dataset_stats(data_path: str) -> Dict:
    """Calculate statistics on a tokenized dataset."""
    with open(data_path, 'r', encoding='utf-8') as f:
        tokenized_texts = json.load(f)
    
    # Calculate statistics
    lengths = [len(tokens) for tokens in tokenized_texts]
    total_tokens = sum(lengths)
    
    stats = {
        "num_texts": len(tokenized_texts),
        "total_tokens": total_tokens,
        "avg_tokens_per_text": np.mean(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "median_tokens": np.median(lengths),
        "std_tokens": np.std(lengths)
    }
    
    return stats


def create_vocabulary_stats(tokenizer, data_path: str) -> Dict:
    """Analyze vocabulary usage in a dataset."""
    with open(data_path, 'r', encoding='utf-8') as f:
        tokenized_texts = json.load(f)
    
    # Token counting
    token_counts = {}
    total_tokens = 0
    
    for tokens in tokenized_texts:
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
            total_tokens += 1
    
    # Vocabulary statistics
    unique_tokens = len(token_counts)
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
    coverage = unique_tokens / vocab_size
    
    # Most frequent tokens
    most_common = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    stats = {
        "vocab_size": vocab_size,
        "unique_tokens_used": unique_tokens,
        "vocabulary_coverage": coverage,
        "total_token_occurrences": total_tokens,
        "most_common_tokens": [
            {"token_id": token_id, "count": count, "token": tokenizer.decode([token_id])}
            for token_id, count in most_common
        ]
    }
    
    return stats


class WeightedMultiDatasetSampler:
    """
    Multi-dataset sampler with weighted sampling for training.
    Supports deterministic sampling, checkpoint state management, and async prefetch.
    """
    
    def __init__(
        self,
        data_dirs: List[str],
        weights: Optional[List[float]] = None,
        seed: int = 42,
        shuffle_shards: bool = True,
        batch_size: int = 8,
        split: str = "train"
    ):
        """
        Args:
            data_dirs: List of directories containing sharded datasets
            weights: Sampling weights for each dataset (normalized automatically)
            seed: Random seed for deterministic sampling
            shuffle_shards: Whether to shuffle shard order on each epoch
            batch_size: Batch size for sampling
            split: Dataset split ('train' or 'val')
        """
        self.data_dirs = [Path(d) for d in data_dirs]
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.batch_size = batch_size
        self.split = split
        
        # Normalize weights
        if weights is None:
            weights = [1.0] * len(data_dirs)
        elif len(weights) != len(data_dirs):
            raise ValueError(f"Length of weights ({len(weights)}) must match data_dirs ({len(data_dirs)})")
        
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        
        # Initialize RNG for deterministic sampling
        self.rng = random.Random(seed)
        
        # Load datasets
        self.datasets = []
        self.dataset_names = []
        
        for i, data_dir in enumerate(self.data_dirs):
            try:
                dataset = ShardedTokenizedDataset(str(data_dir), split=split)
                self.datasets.append(dataset)
                self.dataset_names.append(data_dir.name)
                print(f"📊 Dataset {i}: {data_dir.name} ({len(dataset)} samples, weight: {self.weights[i]:.3f})")
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset from {data_dir}: {e}")
        
        # Validate tokenizer consistency
        self._validate_tokenizer_consistency()
        
        # Initialize sampling state
        self.current_step = 0
        self.dataset_indices = list(range(len(self.datasets)))
        self.sample_counts = [0] * len(self.datasets)  # Track actual sampling
        
        print(f"✅ Multi-dataset sampler initialized:")
        print(f"   - {len(self.datasets)} datasets")
        print(f"   - Weights: {[f'{name}={w:.3f}' for name, w in zip(self.dataset_names, self.weights)]}")
        print(f"   - Total samples: {sum(len(d) for d in self.datasets):,}")
    
    def _validate_tokenizer_consistency(self):
        """Ensure all datasets use compatible tokenizers."""
        if len(self.datasets) <= 1:
            return
        
        # Check if all datasets have similar token ranges
        token_ranges = []
        for dataset in self.datasets:
            if len(dataset.samples) > 0:
                sample_tokens = dataset.samples[0]['input_ids']
                if isinstance(sample_tokens, torch.Tensor):
                    sample_tokens = sample_tokens.tolist()
                token_ranges.append((min(sample_tokens), max(sample_tokens)))
        
        if len(set(token_ranges)) > 1:
            print(f"⚠️  Warning: Datasets may use different tokenizers. Token ranges: {token_ranges}")
    
    def sample_dataset_index(self) -> int:
        """Sample a dataset index according to weights."""
        return self.rng.choices(self.dataset_indices, weights=self.weights)[0]
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        Get a batch from the weighted multi-dataset sampler.
        
        Returns:
            Batch dictionary with input_ids, attention_mask, labels
        """
        batch_input_ids = []
        batch_labels = []
        
        for _ in range(self.batch_size):
            # Choose dataset according to weights
            dataset_idx = self.sample_dataset_index()
            self.sample_counts[dataset_idx] += 1
            
            # Sample from chosen dataset
            dataset = self.datasets[dataset_idx]
            sample_idx = self.rng.randint(0, len(dataset) - 1)
            sample = dataset[sample_idx]
            
            batch_input_ids.append(sample['input_ids'])
            batch_labels.append(sample['labels'])
        
        # Stack into tensors
        batch = {
            'input_ids': torch.stack(batch_input_ids),
            'labels': torch.stack(batch_labels)
        }
        
        # Create attention mask
        batch['attention_mask'] = (batch['input_ids'] != 0).long()
        
        self.current_step += 1
        return batch
    
    def get_observed_mix(self) -> Dict[str, float]:
        """Get the actually observed dataset mixing ratios."""
        total_samples = sum(self.sample_counts)
        if total_samples == 0:
            return {name: 0.0 for name in self.dataset_names}
        
        return {
            name: count / total_samples 
            for name, count in zip(self.dataset_names, self.sample_counts)
        }
    
    def reset_mix_tracking(self):
        """Reset the mix tracking counters."""
        self.sample_counts = [0] * len(self.datasets)
    
    def state_dict(self) -> Dict:
        """Get checkpoint state for resuming training."""
        return {
            'current_step': self.current_step,
            'rng_state': self.rng.getstate(),
            'sample_counts': self.sample_counts.copy(),
            'seed': self.seed,
            'weights': self.weights.copy(),
            'dataset_names': self.dataset_names.copy()
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load checkpoint state for resuming training."""
        self.current_step = state_dict['current_step']

        # Handle RNG state - JSON serialization converts tuples to lists recursively
        def convert_to_tuple(obj):
            """Recursively convert lists back to tuples for RNG state compatibility."""
            if isinstance(obj, list):
                return tuple(convert_to_tuple(item) for item in obj)
            else:
                return obj

        rng_state = state_dict['rng_state']
        if isinstance(rng_state, list):
            rng_state = convert_to_tuple(rng_state)
        self.rng.setstate(rng_state)

        self.sample_counts = state_dict['sample_counts'].copy()
        
        # Validate consistency
        if state_dict.get('weights') != self.weights:
            print("⚠️  Warning: Loaded weights differ from current configuration")
        if state_dict.get('dataset_names') != self.dataset_names:
            print("⚠️  Warning: Loaded dataset names differ from current configuration")
        
        print(f"📁 Resumed multi-dataset sampler from step {self.current_step}")
    
    def get_total_samples(self) -> int:
        """Get total number of samples across all datasets."""
        return sum(len(dataset) for dataset in self.datasets)
    
    def __len__(self) -> int:
        """Return virtual length for compatibility (infinite sampling)."""
        return sum(len(dataset) for dataset in self.datasets)