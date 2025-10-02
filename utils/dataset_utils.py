"""
Utilities for dataset management and data loading.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import glob
import random
import math
import logging
import re
import hashlib
from collections import defaultdict
import gzip


class StreamingSFTDataset(IterableDataset):
    """
    Iterable/streaming dataset for SFT training.
    Loads conversations from sharded JSONL files one by one, avoiding high memory usage.
    """

    def __init__(self, data_dir: str, split: str = "train", use_pretokenized: bool = True):
        """
        Args:
            data_dir: Path to directory containing SFT shards and manifest.json
            split: Dataset split ('train' or 'val')
            use_pretokenized: If True and data is pre-tokenized, yield input_ids instead of text
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_pretokenized = use_pretokenized

        # Load manifest to get shard file list
        self.manifest_path = self.data_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        if split not in manifest['splits']:
            available_splits = list(manifest['splits'].keys())
            raise ValueError(f"Split '{split}' not found. Available: {available_splits}")

        self.split_info = manifest['splits'][split]
        self.shard_files = [shard['filename'] for shard in self.split_info['shards']]

        # Store tokenizer and dataset metadata
        self.tokenizer_metadata = manifest.get('tokenizer_metadata', {})
        self.total_conversations = self.split_info.get('conversations', 0)

        # Detect dataset format version
        format_version = manifest.get('format_version', '1.0')
        self.format_version = format_version

        # Version 3.0 = pre-packed sequences (input_ids, attention_mask, labels)
        # Version 2.0 = raw text only, no tokenization
        # Version 1.0 = pre-tokenized with tokens field
        self.is_prepacked = (format_version == '3.0')
        self.is_pretokenized = (format_version == '1.0' and 'tokenizer_metadata' in manifest and use_pretokenized)

        logging.info(f"StreamingSFTDataset initialized for '{split}' split:")
        logging.info(f"  - Format version: {format_version}")
        logging.info(f"  - {self.split_info.get('num_shards', 0)} shards")

        if self.is_prepacked:
            total_items = self.split_info.get('sequences', 0)
            logging.info(f"  - {total_items:,} pre-packed sequences")
            logging.info(f"  - Using pre-packed data (input_ids, attention_mask, labels)")
            packing_stats = manifest.get('packing_metadata', {})
            if packing_stats:
                logging.info(f"  - Packing efficiency: {packing_stats.get('packing_efficiency', 0):.2f}%")
        else:
            total_items = self.total_conversations
            logging.info(f"  - {total_items:,} total conversations")
            if self.is_pretokenized:
                logging.info(f"  - Using pre-tokenized data (input_ids)")
            else:
                logging.info(f"  - Using raw text (TRL will tokenize)")

    def __len__(self):
        # Length is required by TRL for progress bars and step calculation
        if self.is_prepacked:
            return self.split_info.get('sequences', 0)
        else:
            return self.total_conversations

    def __iter__(self):
        """Yields conversations or pre-packed sequences from shards one by one."""
        worker_info = torch.utils.data.get_worker_info()
        shard_files_to_process = self.shard_files

        if worker_info is not None:
            # Multi-worker data loading: each worker gets a subset of shards
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            shard_files_to_process = [shard for i, shard in enumerate(self.shard_files) if i % num_workers == worker_id]

        # Shuffle shards for each iteration (epoch)
        random.shuffle(shard_files_to_process)

        for shard_filename in shard_files_to_process:
            shard_path = self.data_dir / shard_filename

            # Support compressed or uncompressed shards
            open_fn = gzip.open if shard_path.name.endswith('.gz') else open

            try:
                with open_fn(shard_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                item = json.loads(line)

                                # Format v3.0: Pre-packed sequences with input_ids, attention_mask, labels
                                if self.is_prepacked and 'input_ids' in item:
                                    yield {
                                        "input_ids": item['input_ids'],
                                        "attention_mask": item['attention_mask'],
                                        "labels": item['labels']
                                    }
                                # Format v1.0: Pre-tokenized with tokens field
                                elif self.is_pretokenized and 'tokens' in item:
                                    yield {"input_ids": item['tokens']}
                                # Format v2.0: Raw text (TRL will tokenize)
                                elif 'text' in item:
                                    yield {"text": item['text']}
                            except json.JSONDecodeError:
                                logging.warning(f"Skipping malformed JSON line in {shard_filename}")
                                continue
            except FileNotFoundError:
                logging.warning(f"Shard file not found: {shard_path}, skipping.")
                continue
    
    def get_tokenizer_metadata(self) -> Dict[str, Any]:
        """Get tokenizer metadata from manifest."""
        return self.tokenizer_metadata

    @property
    def column_names(self) -> List[str]:
        """Return column names for TRL compatibility."""
        if self.is_prepacked:
            return ["input_ids", "attention_mask", "labels"]
        elif self.is_pretokenized:
            return ["input_ids"]
        else:
            return ["text"]


class SFTDataset(Dataset):
    """
    Dataset class for SFT training data.
    Loads pre-processed and sharded SFT conversations.
    """

    def __init__(self, data_dir: str, split: str = "train", use_cache: bool = True):
        """
        Args:
            data_dir: Path to directory containing SFT shards
            split: Dataset split ('train' or 'val')
            use_cache: Use cached conversations if available
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_cache = use_cache

        # Load manifest
        self.manifest_path = self.data_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)

        # Verify split exists
        if split not in self.manifest['splits']:
            available_splits = list(self.manifest['splits'].keys())
            raise ValueError(f"Split '{split}' not found. Available: {available_splits}")

        self.split_info = self.manifest['splits'][split]

        # Load conversations (with optional caching)
        self.conversations = self._load_conversations_cached() if use_cache else self._load_conversations()

        logging.info(f"SFTDataset loaded: {len(self.conversations):,} conversations "
                    f"from {self.split_info['num_shards']} shards")

    def _load_conversations(self) -> List[Dict[str, Any]]:
        """Load all conversations from shard files with optimized loading."""
        conversations = []
        total_shards = len(self.split_info['shards'])

        for i, shard_info in enumerate(self.split_info['shards']):
            shard_path = self.data_dir / shard_info['filename']

            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")

            # Read entire file at once (faster than line by line)
            with open(shard_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # Split lines and parse JSON in batch
            if content:
                lines = content.split('\n')
                shard_conversations = [json.loads(line) for line in lines if line.strip()]
                conversations.extend(shard_conversations)

            # Progress indicator for large datasets
            if i % 10 == 0 or i == total_shards - 1:
                logging.info(f"Loaded {i+1}/{total_shards} shards ({len(conversations):,} conversations so far)")

        return conversations

    def _load_conversations_cached(self) -> List[Dict[str, Any]]:
        """Load conversations with caching for faster subsequent loads."""
        cache_file = self.data_dir / f".cache_{self.split}_conversations.json"

        # Check if cache exists and is newer than manifest
        if cache_file.exists():
            try:
                cache_mtime = cache_file.stat().st_mtime
                manifest_mtime = self.manifest_path.stat().st_mtime

                if cache_mtime > manifest_mtime:
                    logging.info(f"Loading conversations from cache: {cache_file}")
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    logging.info("Cache outdated, reloading conversations...")
            except Exception as e:
                logging.warning(f"Cache read failed: {e}, reloading conversations...")

        # Load conversations and save to cache
        conversations = self._load_conversations()

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, separators=(',', ':'))
            logging.info(f"Conversations cached to: {cache_file}")
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")

        return conversations

    def __len__(self) -> int:
        """Return number of conversations."""
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get conversation by index."""
        if idx >= len(self.conversations):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.conversations)}")

        return self.conversations[idx]

    def get_tokenizer_metadata(self) -> Dict[str, Any]:
        """Get tokenizer metadata from manifest."""
        return self.manifest['tokenizer_metadata']

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'total_conversations': len(self.conversations),
            'total_tokens': sum(conv['token_count'] for conv in self.conversations),
            'avg_tokens_per_conversation': sum(conv['token_count'] for conv in self.conversations) / len(self.conversations),
            'template': self.manifest['config']['template']
        }


class WeightedSFTSampler:
    """
    Weighted sampler for multiple SFT datasets.
    Similar to WeightedMultiDatasetSampler but for SFT conversations.
    """

    def __init__(self, datasets: List[SFTDataset], weights: List[float],
                 total_samples: int, seed: int = 42):
        """
        Args:
            datasets: List of SFTDataset instances
            weights: Sampling weights for each dataset
            total_samples: Total number of samples to generate
            seed: Random seed for reproducibility
        """
        self.datasets = datasets
        self.weights = np.array(weights)
        self.total_samples = total_samples
        self.seed = seed

        # Normalize weights
        self.weights = self.weights / self.weights.sum()

        # Validate inputs
        if len(datasets) != len(weights):
            raise ValueError("Number of datasets must match number of weights")

        if any(len(ds) == 0 for ds in datasets):
            raise ValueError("All datasets must contain at least one conversation")

        # Setup random state
        self.rng = np.random.RandomState(seed)

        logging.info(f"WeightedSFTSampler initialized with {len(datasets)} datasets")
        for i, (ds, weight) in enumerate(zip(datasets, self.weights)):
            logging.info(f"  Dataset {i}: {len(ds):,} conversations, weight: {weight:.3f}")

    def __iter__(self):
        """Generate weighted samples."""
        for _ in range(self.total_samples):
            # Choose dataset based on weights
            dataset_idx = self.rng.choice(len(self.datasets), p=self.weights)
            dataset = self.datasets[dataset_idx]

            # Choose random conversation from selected dataset
            conv_idx = self.rng.randint(0, len(dataset))
            conversation = dataset[conv_idx]

            # Add metadata
            conversation['_dataset_idx'] = dataset_idx
            conversation['_original_idx'] = conv_idx

            yield conversation

    def __len__(self):
        """Return total number of samples."""
        return self.total_samples


class SFTCollator:
    """
    Collator for SFT training data.
    Handles tokenization and batching of conversations.
    """

    def __init__(self, tokenizer_path: str, max_length: int = 1024,
                 padding: bool = True, truncation: bool = True):
        """
        Args:
            tokenizer_path: Path to SentencePiece tokenizer
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
        """
        import sentencepiece as spm

        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        # Get special tokens
        self.pad_token_id = self.tokenizer.pad_id() if hasattr(self.tokenizer, 'pad_id') else 0
        self.eos_token_id = self.tokenizer.eos_id()

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of conversations.

        Args:
            batch: List of conversation dictionaries

        Returns:
            Batch dictionary with tensors
        """
        texts = []
        labels = []

        for conversation in batch:
            # Use pre-tokenized tokens if available
            if 'tokens' in conversation:
                tokens = conversation['tokens']
            else:
                # Tokenize on the fly (fallback)
                tokens = self.tokenizer.encode(conversation['text'])

            # Truncate if necessary
            if self.truncation and len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]

            # Add EOS token if not present
            if tokens[-1] != self.eos_token_id:
                tokens.append(self.eos_token_id)

            texts.append(tokens)
            labels.append(tokens.copy())  # For causal LM, labels = input_ids

        # Pad sequences
        if self.padding:
            max_len = min(max(len(t) for t in texts), self.max_length)

            padded_texts = []
            padded_labels = []

            for text, label in zip(texts, labels):
                # Pad sequences
                padding_length = max_len - len(text)
                if padding_length > 0:
                    text = text + [self.pad_token_id] * padding_length
                    label = label + [-100] * padding_length  # -100 is ignored by loss

                padded_texts.append(text)
                padded_labels.append(label)

            texts = padded_texts
            labels = padded_labels

        return {
            'input_ids': torch.tensor(texts, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(
                [[1 if token != self.pad_token_id else 0 for token in text] for text in texts],
                dtype=torch.long
            )
        }


class PackedDataset(Dataset):
    """
    Modern dataset class for reading packed binary data (.bin + .idx files).
    Uses memory mapping for scalability with large datasets.
    """

    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir: Path to directory containing packed data (train.bin/val.bin + .idx)
            split: Dataset split ('train' or 'val')
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # File paths
        self.data_path = self.data_dir / f"{split}.bin"
        self.index_path = self.data_dir / f"{split}.idx"
        self.manifest_path = self.data_dir / "final_manifest.json"

        # Validate files exist
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        # Load index metadata
        with open(self.index_path, 'r') as f:
            self.index_info = json.load(f)

        # Load manifest for additional info
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)

        # Extract shape and dtype
        self.shape = tuple(self.index_info['shape'])
        self.dtype = getattr(np, self.index_info['dtype'])
        self.sequence_length = self.index_info['sequence_length']
        self.num_sequences = self.index_info['num_sequences']

        # Create memory-mapped array (doesn't load data into RAM)
        self.data = np.memmap(
            str(self.data_path),
            dtype=self.dtype,
            mode='r',
            shape=self.shape
        )

        logging.info(f"📊 PackedDataset loaded: {split} split")
        logging.info(f"   - Shape: {self.shape}")
        logging.info(f"   - Dtype: {self.dtype}")
        logging.info(f"   - Sequence length: {self.sequence_length}")
        logging.info(f"   - Memory mapped: {self.data_path}")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Get a sequence by index.
        Returns dict with input_ids, labels, and attention_mask.
        """
        if idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_sequences}")

        # Get the full sequence from memory map
        sequence = torch.tensor(self.data[idx], dtype=torch.long)

        # For training, input_ids and labels are shifted by one token
        input_ids = sequence[:-1]
        labels = sequence[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones_like(input_ids)
        }

    def get_manifest(self) -> Dict:
        """Return the dataset manifest for metadata access."""
        return self.manifest.copy()

    def validate_tokenizer_compatibility(self, tokenizer_dir: str) -> bool:
        """
        Validate that this dataset is compatible with the given tokenizer.
        Uses SHA256 hash comparison for robustness.
        """
        tokenizer_dir = Path(tokenizer_dir)
        tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"

        if not tokenizer_config_path.exists():
            logging.warning(f"Tokenizer config not found: {tokenizer_config_path}")
            return False

        # Load tokenizer config
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)

        # Calculate SHA256 hash of tokenizer config
        tokenizer_config_str = json.dumps(tokenizer_config, sort_keys=True, ensure_ascii=False)
        tokenizer_hash = hashlib.sha256(tokenizer_config_str.encode('utf-8')).hexdigest()

        # Compare with hash stored in manifest
        expected_hash = self.manifest.get('tokenizer_config_hash')

        if tokenizer_hash == expected_hash:
            logging.info("✅ Tokenizer compatibility validated via SHA256 hash")
            return True
        else:
            logging.error(f"❌ Tokenizer incompatibility detected!")
            logging.error(f"   Expected hash: {expected_hash}")
            logging.error(f"   Actual hash:   {tokenizer_hash}")
            return False


class SmartTokenEstimator:
    """
    Smart token estimator that simulates tokenizer behavior without actual tokenizer.
    Uses linguistic heuristics to provide realistic token counts.
    """

    def __init__(self, vocab_size: int = 32768, base_chars_per_token: float = 4.0):
        self.vocab_size = vocab_size
        self.base_chars_per_token = base_chars_per_token

        # Common patterns that affect tokenization
        self.word_pattern = re.compile(r'\b\w+\b')
        self.punct_pattern = re.compile(r'[.!?;:,]')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.whitespace_pattern = re.compile(r'\s+')

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using linguistic heuristics.
        Simulates how a real tokenizer would behave.
        """
        if not text.strip():
            return 0

        # Base estimation
        base_tokens = len(text) / self.base_chars_per_token

        # Adjustments based on text characteristics

        # 1. Word boundaries (tokenizers often respect word boundaries)
        words = self.word_pattern.findall(text)
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 5

        # Longer words = more subwords = more tokens
        word_factor = 1.0 + (max(0, avg_word_length - 5) * 0.02)

        # 2. Punctuation usually gets separate tokens
        punct_count = len(self.punct_pattern.findall(text))
        punct_adjustment = punct_count * 0.5  # Each punct adds ~0.5 tokens

        # 3. Numbers often get tokenized as separate units
        numbers = self.number_pattern.findall(text)
        number_adjustment = len(numbers) * 0.3

        # 4. Excessive whitespace
        whitespaces = self.whitespace_pattern.findall(text)
        whitespace_penalty = max(0, len(whitespaces) - len(text.split())) * 0.1

        # 5. Text quality affects tokenization efficiency
        char_variety = len(set(text.lower()))
        variety_factor = min(1.2, 0.8 + (char_variety / 100))

        # Final calculation
        estimated_tokens = (base_tokens * word_factor * variety_factor +
                          punct_adjustment + number_adjustment - whitespace_penalty)

        return max(1, int(estimated_tokens))  # At least 1 token

    def encode(self, text: str) -> List[int]:
        """Simulate tokenizer.encode() - returns fake token IDs."""
        token_count = self.estimate_tokens(text)
        # Return fake token IDs (not used, just for compatibility)
        return list(range(token_count))


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
                # Try PackedDataset first (new format), fallback to legacy
                final_manifest_path = data_dir / 'final_manifest.json'
                if final_manifest_path.exists():
                    # New PackedDataset format
                    dataset = PackedDataset(str(data_dir), split=split)
                    print(f"📊 Dataset {i}: {data_dir.name} (PackedDataset, {len(dataset)} samples, weight: {self.weights[i]:.3f})")
                else:
                    # Legacy ShardedTokenizedDataset format
                    dataset = ShardedTokenizedDataset(str(data_dir), split=split)
                    print(f"📊 Dataset {i}: {data_dir.name} (Legacy, {len(dataset)} samples, weight: {self.weights[i]:.3f})")

                self.datasets.append(dataset)
                self.dataset_names.append(data_dir.name)

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
        batch_attention_masks = []

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
            batch_attention_masks.append(sample.get('attention_mask', torch.ones_like(sample['input_ids'])))

        # Stack into tensors
        batch = {
            'input_ids': torch.stack(batch_input_ids),
            'labels': torch.stack(batch_labels),
            'attention_mask': torch.stack(batch_attention_masks)
        }
        
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


def estimate_avg_tokens_per_sample(
    stream_factory: Callable,
    sample_size: int = 100,
    tokenizer: Optional[Any] = None,
    text_keys: Union[str, List[str]] = "text",
    chars_per_token: float = 4.0,
    tokenizer_path: Optional[str] = None
) -> Dict:
    """
    Estimate average tokens per sample from a streaming dataset.

    Args:
        stream_factory: Function that returns a fresh dataset iterator when called
        sample_size: Number of samples to analyze
        tokenizer: Optional tokenizer for precise token counting
        text_keys: Key(s) containing text data (str or list of str)
        chars_per_token: Character-to-token ratio for heuristic estimation
        tokenizer_path: Path to tokenizer for loading if needed

    Returns:
        Dictionary with statistics:
        {
            "avg_tokens": float,
            "std_tokens": float,
            "p50_tokens": float,  # Median
            "p90_tokens": float,  # 90th percentile
            "n_samples_analyzed": int,
            "method": "tokenizer" | "chars_heuristic"
        }
    """
    logging.info(f"Analyzing {sample_size} samples to estimate token counts...")

    # Ensure text_keys is a list
    if isinstance(text_keys, str):
        text_keys = [text_keys]

    # Load tokenizer if path provided and no tokenizer given
    if tokenizer is None and tokenizer_path:
        try:
            import sentencepiece as smp
            from pathlib import Path
            logging.info(f"Loading tokenizer from {tokenizer_path} for accurate token counting...")
            tokenizer = smp.SentencePieceProcessor()
            model_path = tokenizer_path if tokenizer_path.endswith('.model') else f"{tokenizer_path}/spm.model"
            if Path(model_path).exists():
                tokenizer.load(model_path)
                logging.info(f"✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size()}")
            else:
                logging.warning(f"Tokenizer not found at {model_path}. Will use SMART simulation.")
                tokenizer = None
        except Exception as e:
            logging.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}. Using smart simulation.")
            tokenizer = None

    # Create smart token estimator if no real tokenizer
    smart_estimator = None
    if tokenizer is None:
        # Get vocab size from config if available
        vocab_size = 32768  # Default
        if tokenizer_path and 'training_params' in str(tokenizer_path):
            vocab_size = 32768  # Will be overridden by actual config parsing

        smart_estimator = SmartTokenEstimator(
            vocab_size=vocab_size,
            base_chars_per_token=chars_per_token
        )

    token_lengths = []
    method = "real_tokenizer" if tokenizer is not None else "smart_simulation"

    if method == "real_tokenizer":
        logging.info("🎯 Using REAL TOKENIZER for accurate token counting")
    else:
        logging.info(f"🧠 Using SMART TOKEN SIMULATION (linguistic heuristics, {chars_per_token} base chars/token)")

    try:
        # Get fresh stream
        stream = stream_factory()
        samples_processed = 0

        for sample in stream:
            if samples_processed >= sample_size:
                break

            try:
                # Extract and concatenate text from specified keys
                text_parts = []
                for key in text_keys:
                    if key in sample and sample[key] is not None:
                        text_parts.append(str(sample[key]))

                if not text_parts:
                    logging.warning(f"No text found in sample with keys {text_keys}")
                    continue

                concatenated_text = " ".join(text_parts)

                # Calculate token count
                if tokenizer is not None:
                    # Use actual tokenizer
                    try:
                        if hasattr(tokenizer, 'encode'):
                            # SentencePiece tokenizer
                            tokens = tokenizer.encode(concatenated_text)
                            token_count = len(tokens)
                        elif hasattr(tokenizer, 'encode_as_ids'):
                            # SentencePiece tokenizer alternative
                            tokens = tokenizer.encode_as_ids(concatenated_text)
                            token_count = len(tokens)
                        elif hasattr(tokenizer, '__call__'):
                            # HuggingFace tokenizer
                            tokens = tokenizer(concatenated_text)['input_ids']
                            token_count = len(tokens)
                        else:
                            raise AttributeError("Tokenizer must have 'encode' method or be callable")
                    except Exception as e:
                        logging.warning(f"Tokenizer failed on sample {samples_processed}: {e}")
                        continue
                elif smart_estimator is not None:
                    # Use smart token estimation
                    token_count = smart_estimator.estimate_tokens(concatenated_text)
                else:
                    # Fallback to simple character-based heuristic
                    token_count = len(concatenated_text) / chars_per_token

                # Filter outliers: ignore samples with >10x current median
                if len(token_lengths) > 10:  # Only filter after we have some data
                    current_median = np.median(token_lengths)
                    if token_count > 10 * current_median:
                        logging.debug(f"Filtering outlier sample with {token_count} tokens (median: {current_median})")
                        continue

                token_lengths.append(token_count)
                samples_processed += 1

                if samples_processed % 50 == 0:
                    logging.debug(f"Processed {samples_processed}/{sample_size} samples...")

            except Exception as e:
                logging.warning(f"Error processing sample {samples_processed}: {e}")
                continue

    except Exception as e:
        logging.error(f"Error during stream processing: {e}")
        raise

    if not token_lengths:
        raise ValueError("No valid samples found for token estimation")

    # Calculate statistics
    token_array = np.array(token_lengths)
    stats = {
        "avg_tokens": float(np.mean(token_array)),
        "std_tokens": float(np.std(token_array)),
        "p50_tokens": float(np.median(token_array)),
        "p90_tokens": float(np.percentile(token_array, 90)),
        "n_samples_analyzed": len(token_lengths),
        "method": method
    }

    logging.info(
        f"Token estimation complete: avg={stats['avg_tokens']:.1f}, "
        f"std={stats['std_tokens']:.1f}, p50={stats['p50_tokens']:.1f}, "
        f"p90={stats['p90_tokens']:.1f} (method: {method})"
    )

    return stats


def plan_samples_for_token_budget(
    token_budget: int,
    estimation_stats: Dict,
    margin_ratio: float = 0.02
) -> int:
    """
    Calculate number of samples needed to reach a token budget.

    Args:
        token_budget: Target number of tokens
        estimation_stats: Statistics from estimate_avg_tokens_per_sample
        margin_ratio: Safety margin (default: 2%)

    Returns:
        Number of samples to download (integer)
    """
    # Use realistic estimation: avg + 0.5*std for reasonable safety
    conservative_tokens_per_sample = (
        estimation_stats['avg_tokens'] + 0.5 * estimation_stats['std_tokens']
    )

    # Calculate base number of samples needed
    base_samples_needed = token_budget / conservative_tokens_per_sample

    # Add safety margin
    samples_with_margin = base_samples_needed * (1 + margin_ratio)

    # Round up to integer
    final_samples = int(math.ceil(samples_with_margin))

    logging.info(
        f"Token budget planning: {token_budget:,} tokens target, "
        f"conservative estimate: {conservative_tokens_per_sample:.1f} tokens/sample, "
        f"base need: {base_samples_needed:.0f} samples, "
        f"with {margin_ratio:.1%} margin: {final_samples:,} samples"
    )

    return final_samples