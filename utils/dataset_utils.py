"""
Utilities for dataset management and data loading.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TokenizedDataset(Dataset):
    """Dataset for tokenized data saved in JSON format."""
    
    def __init__(self, data_path: str, sequence_length: int = 1024, stride: int = None):
        """
        Args:
            data_path: Path to JSON file containing tokenized data
            sequence_length: Sequence length for training
            stride: Stride between sequences (default = sequence_length)
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        
        # Data loading
        print(f"Loading data from {data_path}...")
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