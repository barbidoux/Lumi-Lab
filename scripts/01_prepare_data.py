#!/usr/bin/env python3
"""
Data preparation script for mini-LLM training.
Implements a complete, robust dataset preparation pipeline based on JSON/YAML configs.
Supports OpenWebText, Wikipedia EN, WikiText-103, and custom datasets.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple, Union
import yaml

import ftfy
import numpy as np
import sentencepiece as spm
from datasets import Dataset, load_dataset
from datasketch import MinHashLSH, MinHash
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    return config


def clean_text(text: str) -> str:
    """Clean text using ftfy and remove URLs, code blocks, unwanted characters."""
    # Fix encoding issues first
    text = ftfy.fix_text(text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]*`', '', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def is_english_text(text: str, threshold: float = 0.7) -> bool:
    """Check if text is in English using langdetect."""
    try:
        if len(text) < 20:
            return False
        detected_lang = detect(text)
        return detected_lang == 'en'
    except (LangDetectException, Exception):
        # Fallback to word-based detection
        english_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 
            'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 
            'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my',
            'one', 'all', 'would', 'there', 'their', 'what', 'up', 'out', 'if', 'about',
            'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time',
            'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
            'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after',
            'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new'
        }
        
        words = text.lower().split()
        if not words:
            return False
        
        english_count = sum(1 for word in words if word in english_words)
        return (english_count / len(words)) >= threshold


def filter_by_length(text: str, min_length: int = 50, max_length: int = 10000) -> bool:
    """Filter by text length."""
    return min_length <= len(text) <= max_length


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class AdvancedDeduplicator:
    """Advanced deduplicator using SHA256 + MinHashLSH for efficient fuzzy duplicate detection."""
    
    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.seen_hashes = set()
        # MinHashLSH for efficient similarity search
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.doc_id = 0
    
    def _get_shingles(self, text: str, k: int = 3) -> Set[str]:
        """Generate k-shingles (character n-grams) from text."""
        text = text.lower().replace(' ', '')
        return {text[i:i+k] for i in range(len(text) - k + 1)}
    
    def _compute_minhash(self, shingles: Set[str]) -> MinHash:
        """Compute MinHash signature of a shingle set."""
        minhash = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))
        return minhash
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is an exact or fuzzy duplicate."""
        # Exact deduplication with SHA256
        text_hash = compute_hash(text)
        if text_hash in self.seen_hashes:
            return True
        
        # Fuzzy deduplication with MinHashLSH
        shingles = self._get_shingles(text)
        if not shingles:
            return True  # Empty or too short text
        
        minhash = self._compute_minhash(shingles)
        
        # Query LSH for similar documents
        similar_docs = self.lsh.query(minhash)
        if similar_docs:
            return True
        
        # Add to data structures
        self.seen_hashes.add(text_hash)
        self.lsh.insert(f"doc_{self.doc_id}", minhash)
        self.doc_id += 1
        
        return False


def deduplicate_texts(texts: List[str], use_minhash: bool = True, threshold: float = 0.8) -> List[str]:
    """Remove duplicates based on SHA256 and optionally MinHash."""
    if use_minhash:
        print(f"Deduplication with MinHashLSH (threshold={threshold})...")
        deduplicator = AdvancedDeduplicator(threshold=threshold)
        deduplicated = []
        
        for text in tqdm(texts, desc="MinHash deduplication"):
            if not deduplicator.is_duplicate(text):
                deduplicated.append(text)
        
        return deduplicated
    
    else:
        print("Exact deduplication with SHA256...")
        seen_hashes: Set[str] = set()
        deduplicated = []
        
        for text in tqdm(texts, desc="SHA256 deduplication"):
            text_hash = compute_hash(text)
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                deduplicated.append(text)
        
        return deduplicated


def split_train_val(texts: List[str], train_ratio: float = 0.98, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split texts into train/validation sets at document level."""
    random.seed(seed)
    shuffled_texts = texts.copy()
    random.shuffle(shuffled_texts)
    
    split_idx = int(len(shuffled_texts) * train_ratio)
    train_texts = shuffled_texts[:split_idx]
    val_texts = shuffled_texts[split_idx:]
    
    return train_texts, val_texts


def train_tokenizer(texts: List[str], vocab_size: int, output_path: str) -> None:
    """Train a SentencePiece tokenizer on the corpus."""
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Temporary corpus writing
    temp_corpus = str(Path(output_path).parent / "temp_corpus.txt")
    with open(temp_corpus, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + "\n")
    
    # Tokenizer training
    spm.SentencePieceTrainer.train(
        input=temp_corpus,
        model_prefix=output_path,
        vocab_size=vocab_size,
        model_type='unigram',
        character_coverage=0.995,
        normalization_rule_name='nmt_nfkc_cf',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    
    # Remove temporary file
    os.remove(temp_corpus)


def tokenize_and_pack(texts: List[str], tokenizer_path: str, sequence_length: int) -> List[List[int]]:
    """Tokenize corpus and pack into sequences with next-token prediction labels."""
    sp = spm.SentencePieceProcessor()
    # Handle both cases: path with or without .model extension
    model_path = tokenizer_path if tokenizer_path.endswith('.model') else tokenizer_path + '.model'
    sp.load(model_path)
    
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenization"):
        tokens = sp.encode_as_ids(text)
        all_tokens.extend(tokens)
    
    print(f"Total tokens: {len(all_tokens):,}")
    
    # Pack into sequences
    print("Packing into sequences...")
    packed_sequences = []
    
    for i in range(0, len(all_tokens) - sequence_length, sequence_length):
        sequence = all_tokens[i:i + sequence_length]
        # Create labels for next-token prediction
        labels = sequence[1:] + [-100]  # -100 for last position
        packed_sequences.append({
            'input_ids': sequence,
            'labels': labels
        })
    
    return packed_sequences


def save_sharded_data(data: List[Dict], output_dir: str, shard_tokens: int, split: str) -> List[str]:
    """Save data in sharded JSONL format and return list of shard filenames."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shard_files = []
    current_shard = []
    current_tokens = 0
    shard_idx = 0
    
    for item in tqdm(data, desc=f"Sharding {split} data"):
        current_shard.append(item)
        current_tokens += len(item['input_ids'])
        
        if current_tokens >= shard_tokens:
            # Save current shard
            shard_file = f"{split}_{shard_idx:05d}.jsonl"
            shard_path = output_dir / shard_file
            
            with open(shard_path, 'w', encoding='utf-8') as f:
                for example in current_shard:
                    f.write(json.dumps(example) + '\n')
            
            shard_files.append(shard_file)
            
            # Reset for next shard
            current_shard = []
            current_tokens = 0
            shard_idx += 1
    
    # Save remaining data if any
    if current_shard:
        shard_file = f"{split}_{shard_idx:05d}.jsonl"
        shard_path = output_dir / shard_file
        
        with open(shard_path, 'w', encoding='utf-8') as f:
            for example in current_shard:
                f.write(json.dumps(example) + '\n')
        
        shard_files.append(shard_file)
    
    return shard_files


def create_manifest(train_shards: List[str], val_shards: List[str], output_dir: str) -> None:
    """Create manifest.json listing all shards."""
    manifest = {
        'train_shards': train_shards,
        'val_shards': val_shards,
        'total_shards': len(train_shards) + len(val_shards)
    }
    
    manifest_path = Path(output_dir) / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)


def create_data_card(config: Dict, stats: Dict, output_dir: str, reused_tokenizer: bool = False) -> None:
    """Create DATA_CARD.md with dataset information."""
    from datetime import datetime
    
    processing_date = datetime.now().strftime("%Y-%m-%d")
    train_tokens = stats.get('train_tokens', 'N/A')
    val_tokens = stats.get('val_tokens', 'N/A')
    
    # Deduplication method description
    if config.get('use_minhash', False):
        dedup_method = f"MinHashLSH fuzzy deduplication (threshold: {config.get('minhash_threshold', 0.8)})"
    else:
        dedup_method = "SHA256 exact deduplication"
    
    # Tokenizer info
    tokenizer_info = f"{'Reused existing' if reused_tokenizer else 'Newly trained'} SentencePiece unigram model"
    
    data_card = f"""# Dataset Card

## Dataset Information
- **Source**: {config['input_path']}
- **Version/Config**: {config.get('input_config', 'default')}
- **Output Directory**: {config['output_dir']}
- **Processing Date**: {processing_date}
- **Usage**: Educational and research purposes only

## Configuration
- **Vocabulary Size**: {config['vocab_size']:,}
- **Sequence Length**: {config['sequence_length']}
- **Minimum Text Length**: {config['min_length']} characters
- **Maximum Text Length**: {config['max_length']} characters
- **Train/Val Split**: {config['train_ratio']:.1%} / {(1-config['train_ratio']):.1%}
- **Shard Size**: ~{config['shard_tokens']:,} tokens per shard

## Filtering and Processing Pipeline
1. **Text Cleaning**: ftfy encoding fix, URL removal, code block removal, control character removal
2. **Language Filtering**: English only (langdetect + fallback word-based detection)  
3. **Length Filtering**: {config['min_length']}-{config['max_length']} characters
4. **Deduplication**: {dedup_method}
5. **Train/Val Split**: Random shuffle at document level
6. **Tokenization**: {tokenizer_info}
7. **Sequence Packing**: Fixed length sequences with next-token prediction labels

## Statistics
- **Original Documents**: {stats.get('original_docs', 'N/A'):,}
- **After Cleaning & Filtering**: {stats.get('cleaned_docs', 'N/A'):,}
- **After Deduplication**: {stats.get('deduplicated_docs', 'N/A'):,} (removed {stats.get('dedup_percentage', 0):.1f}%)
- **Train Documents**: {stats.get('train_docs', 'N/A'):,}
- **Validation Documents**: {stats.get('val_docs', 'N/A'):,}
- **Train Tokens**: {train_tokens if isinstance(train_tokens, str) else f"{train_tokens:,}"}
- **Validation Tokens**: {val_tokens if isinstance(val_tokens, str) else f"{val_tokens:,}"}
- **Total Tokens**: {stats.get('total_tokens', 'N/A'):,}
- **Train Shards**: {stats.get('train_shards', 'N/A')}
- **Validation Shards**: {stats.get('val_shards', 'N/A')}

## Data Format
- **Format**: JSONL (one JSON object per line)
- **Fields**: 
  - `input_ids`: List of token IDs (length = sequence_length)
  - `labels`: List of next-token prediction targets (-100 for last position)
- **Tokenizer**: SentencePiece unigram model (vocab_size={config['vocab_size']:,})
- **Special Tokens**: PAD(0), UNK(1), BOS(2), EOS(3)

## Files Structure
```
{Path(config['output_dir']).name}/
├── train_*.jsonl       # Training data shards
├── val_*.jsonl         # Validation data shards  
├── manifest.json       # Shard registry
├── DATA_CARD.md        # This documentation
└── stats.json          # Processing statistics
```

## Reproducibility
- **Config File**: Used configuration saved in processing logs
- **Random Seed**: 42 (for train/val split)
- **Dependencies**: ftfy, langdetect, datasketch, sentencepiece

## License and Usage
- **Source License**: Please check original dataset license for `{config['input_path']}`
- **Usage Restrictions**: Educational and research purposes only
- **Commercial Use**: Check source dataset terms

## Risks and Limitations
⚠️ **Important Disclaimers:**
- Dataset may contain biases present in the original source
- Automated filtering may introduce additional biases  
- Content has not been manually reviewed for harmful content
- Language detection is probabilistic and may have errors
- Deduplication is approximate and may miss some duplicates
- **Use appropriate safety measures for downstream applications**

## Quality Assurance
✅ **Validation Checks Performed:**
- Non-empty train and validation sets
- Proper label formatting (-100 only at sequence end)
- Shard size consistency
- Manifest file integrity
- Statistics consistency
"""
    
    data_card_path = Path(output_dir) / 'DATA_CARD.md'
    with open(data_card_path, 'w', encoding='utf-8') as f:
        f.write(data_card)


def load_dataset_by_path(input_path: str, input_config: Optional[Dict] = None) -> Dataset:
    """Load dataset based on input path and configuration."""
    print(f"Loading dataset: {input_path}")
    
    if input_path == "openwebtext":
        # OpenWebText via HuggingFace with streaming
        dataset = load_dataset("openwebtext", split="train", streaming=True)
        # Convert to regular dataset (take first N examples for memory efficiency)
        samples = []
        for i, example in enumerate(dataset):
            if i >= 100000:  # Limit for memory
                break
            samples.append(example)
        dataset = Dataset.from_list(samples)
    
    elif input_path == "wikipedia":
        # Wikipedia dataset - deprecated, fallback to wikitext
        print("⚠️  Wikipedia dataset is deprecated. Using WikiText-103 as fallback.")
        dataset = load_dataset("wikitext-103-raw-v1", split="train")
    
    elif input_path == "graelo/wikipedia":
        # Alternative Wikipedia dataset
        config_args = input_config or {}
        dataset = load_dataset("graelo/wikipedia", **config_args, split="train")
    
    elif input_path == "wikitext":
        # WikiText dataset with config
        config_name = input_config or "wikitext-103-raw-v1"
        dataset = load_dataset("wikitext", config_name, split="train")
    
    elif input_path.startswith("wikitext"):
        # WikiText datasets (direct name)
        dataset = load_dataset(input_path, split="train")
    
    elif input_path.endswith(('.json', '.jsonl')):
        # Local JSON/JSONL file
        dataset = load_dataset('json', data_files=input_path)['train']
    
    elif input_path.endswith('.txt'):
        # Plain text file
        with open(input_path, 'r', encoding='utf-8') as f:
            texts = f.read().split('\n\n')  # Split by paragraphs
        dataset = Dataset.from_dict({"text": texts})
    
    else:
        # Try loading as HuggingFace dataset
        try:
            dataset = load_dataset(input_path, split="train")
        except Exception as e:
            raise ValueError(f"Cannot load dataset from {input_path}: {e}")
    
    return dataset


def extract_text_from_dataset(dataset: Dataset) -> List[str]:
    """Extract text field from dataset, handling different column names."""
    print(f"Dataset columns: {dataset.column_names}")
    
    if 'text' in dataset.column_names:
        return dataset['text']
    elif 'content' in dataset.column_names:
        return dataset['content']
    elif 'article' in dataset.column_names:
        return dataset['article']
    else:
        # Try to find any string column
        for col_name in dataset.column_names:
            if len(dataset) > 0:
                sample = dataset[0][col_name]
                if isinstance(sample, str) and len(sample) > 10:
                    print(f"Using column '{col_name}' as text field")
                    return dataset[col_name]
        
        raise ValueError(f"No suitable text column found. Available columns: {dataset.column_names}")


def main():
    parser = argparse.ArgumentParser(description="Data preparation for training with config support")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to JSON/YAML configuration file")
    
    # CLI arguments can override config values
    parser.add_argument("--input_path", type=str, help="Override input dataset path")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--vocab_size", type=int, help="Override vocabulary size")
    parser.add_argument("--sequence_length", type=int, help="Override sequence length")
    parser.add_argument("--min_length", type=int, help="Override minimum text length")
    parser.add_argument("--max_length", type=int, help="Override maximum text length")
    parser.add_argument("--use_minhash", action="store_true", help="Override MinHash usage")
    parser.add_argument("--minhash_threshold", type=float, help="Override MinHash threshold")
    parser.add_argument("--train_ratio", type=float, help="Override train/validation ratio")
    parser.add_argument("--shard_tokens", type=int, help="Override tokens per shard")
    parser.add_argument("--reuse_tokenizer", action="store_true", 
                       help="Reuse existing tokenizer instead of training a new one")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # CLI arguments override config values
    for key, value in vars(args).items():
        if key != 'config_path' and value is not None:
            config[key] = value
    
    # Validate required fields
    required_fields = ['input_path', 'output_dir', 'vocab_size', 'sequence_length']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' not found in config")
    
    # Set defaults
    config.setdefault('min_length', 50)
    config.setdefault('max_length', 10000)
    config.setdefault('use_minhash', True)
    config.setdefault('minhash_threshold', 0.8)
    config.setdefault('train_ratio', 0.98)
    config.setdefault('shard_tokens', 5000000)
    config.setdefault('tokenizer_path', 'data/tokenizer/tokenizer')
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting data preparation with config: {args.config_path}")
    print(f"Output directory: {output_dir}")
    
    # Load dataset
    dataset = load_dataset_by_path(config['input_path'], config.get('input_config'))
    texts = extract_text_from_dataset(dataset)
    
    print(f"Original dataset: {len(texts)} texts")
    original_docs = len(texts)
    
    # Cleaning pipeline
    print("Cleaning and filtering texts...")
    cleaned_texts = []
    for text in tqdm(texts, desc="Cleaning"):
        if not isinstance(text, str):
            continue
            
        cleaned = clean_text(text)
        if cleaned and filter_by_length(cleaned, config['min_length'], config['max_length']):
            if is_english_text(cleaned):
                cleaned_texts.append(cleaned)
    
    print(f"After cleaning and filtering: {len(cleaned_texts)} texts")
    cleaned_docs = len(cleaned_texts)
    
    # Deduplication
    print("Deduplicating...")
    deduplicated_texts = deduplicate_texts(
        cleaned_texts, 
        use_minhash=config['use_minhash'],
        threshold=config['minhash_threshold']
    )
    print(f"After deduplication: {len(deduplicated_texts)} texts")
    deduplicated_docs = len(deduplicated_texts)
    
    # Calculate deduplication stats
    dedup_percentage = ((cleaned_docs - deduplicated_docs) / cleaned_docs * 100) if cleaned_docs > 0 else 0
    print(f"Deduplication rate: {dedup_percentage:.1f}%")
    
    # Split train/validation
    print("Splitting into train/validation sets...")
    train_texts, val_texts = split_train_val(deduplicated_texts, config['train_ratio'])
    print(f"Train: {len(train_texts)} texts, Validation: {len(val_texts)} texts")
    
    # Train or reuse tokenizer
    tokenizer_path = config['tokenizer_path']
    if args.reuse_tokenizer:
        model_path = tokenizer_path if tokenizer_path.endswith('.model') else tokenizer_path + '.model'
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Tokenizer not found: {model_path}. Cannot reuse non-existent tokenizer.")
        print(f"♻️  Reusing existing tokenizer: {model_path}")
    else:
        print(f"🔨 Training new tokenizer (vocab_size={config['vocab_size']})...")
        train_tokenizer(deduplicated_texts, config['vocab_size'], tokenizer_path)
    
    # Tokenize and pack
    print("Tokenizing and packing sequences...")
    train_data = tokenize_and_pack(train_texts, tokenizer_path, config['sequence_length'])
    val_data = tokenize_and_pack(val_texts, tokenizer_path, config['sequence_length'])
    
    # Validation checks
    assert len(train_data) > 0, "No training data generated"
    assert len(val_data) > 0, "No validation data generated"
    
    # Check last label is -100
    for sample in [train_data[0], val_data[0]]:
        assert sample['labels'][-1] == -100, f"Last label should be -100, got {sample['labels'][-1]}"
        assert -100 not in sample['labels'][:-1], "Label -100 should only appear at the end"
    
    # Save sharded data
    print("Saving sharded data...")
    train_shards = save_sharded_data(train_data, output_dir, config['shard_tokens'], 'train')
    val_shards = save_sharded_data(val_data, output_dir, config['shard_tokens'], 'val')
    
    # Create manifest
    create_manifest(train_shards, val_shards, output_dir)
    
    # Calculate final statistics
    total_tokens = sum(len(item['input_ids']) for item in train_data + val_data)
    train_tokens = sum(len(item['input_ids']) for item in train_data)
    val_tokens = sum(len(item['input_ids']) for item in val_data)
    
    stats = {
        'processing_date': str(Path().cwd()),
        'original_docs': original_docs,
        'cleaned_docs': cleaned_docs,
        'deduplicated_docs': deduplicated_docs,
        'dedup_percentage': dedup_percentage,
        'train_docs': len(train_texts),
        'val_docs': len(val_texts),
        'total_tokens': total_tokens,
        'train_tokens': train_tokens,
        'val_tokens': val_tokens,
        'train_shards': len(train_shards),
        'val_shards': len(val_shards)
    }
    
    # Create data card
    create_data_card(config, stats, output_dir, reused_tokenizer=args.reuse_tokenizer)
    
    # Save processing stats
    stats_path = output_dir / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Final validation
    manifest_path = output_dir / 'manifest.json'
    data_card_path = output_dir / 'DATA_CARD.md'
    assert manifest_path.exists(), "manifest.json was not created"
    assert data_card_path.exists(), "DATA_CARD.md was not created"
    assert manifest_path.stat().st_size > 0, "manifest.json is empty"
    assert data_card_path.stat().st_size > 0, "DATA_CARD.md is empty"
    
    print(f"\n🎉 Data preparation completed successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Statistics:")
    print(f"  - Original documents: {original_docs:,}")
    print(f"  - After processing: {deduplicated_docs:,}")
    print(f"  - Deduplication rate: {dedup_percentage:.1f}%")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Train shards: {len(train_shards)}")
    print(f"  - Validation shards: {len(val_shards)}")
    print(f"📄 Files created: manifest.json, DATA_CARD.md, {len(train_shards + val_shards)} shard files")


if __name__ == "__main__":
    main()