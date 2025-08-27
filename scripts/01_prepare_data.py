#!/usr/bin/env python3
"""
Data preparation script for mini-LLM training.
Implements the cleaning, deduplication, filtering and tokenization pipeline.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import sentencepiece as spm
from datasets import Dataset, load_dataset
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Clean text by removing URLs, code and unwanted characters."""
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
    """Check if text is probably in English."""
    # List of frequent English words
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


class MinHashDeduplicator:
    """Deduplicator using MinHash for fuzzy duplicate detection."""
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.8, shingle_size: int = 3):
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.seen_signatures = []
        self.seen_hashes = set()
    
    def _get_shingles(self, text: str) -> Set[str]:
        """Generate shingles (character n-grams) from text."""
        text = text.lower().replace(' ', '')
        return {text[i:i+self.shingle_size] for i in range(len(text) - self.shingle_size + 1)}
    
    def _compute_minhash(self, shingles: Set[str]) -> List[int]:
        """Compute MinHash signature of a shingle set."""
        if not shingles:
            return [0] * self.num_perm
        
        # Use simple hash functions based on prime numbers
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        signature = []
        
        for i in range(self.num_perm):
            prime_a = primes[i % len(primes)]
            prime_b = primes[(i + 1) % len(primes)]
            
            min_hash = float('inf')
            for shingle in shingles:
                hash_val = hash((shingle, prime_a, prime_b)) % (2**32)
                min_hash = min(min_hash, hash_val)
            
            signature.append(min_hash if min_hash != float('inf') else 0)
        
        return signature
    
    def _jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Compute estimated Jaccard similarity between two signatures."""
        if len(sig1) != len(sig2):
            return 0.0
        return sum(1 for a, b in zip(sig1, sig2) if a == b) / len(sig1)
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is an exact or fuzzy duplicate."""
        # Exact deduplication with SHA256
        text_hash = compute_hash(text)
        if text_hash in self.seen_hashes:
            return True
        
        # Fuzzy deduplication with MinHash
        shingles = self._get_shingles(text)
        if not shingles:
            return True  # Empty or too short text
        
        signature = self._compute_minhash(shingles)
        
        # Compare with existing signatures
        for existing_sig in self.seen_signatures:
            similarity = self._jaccard_similarity(signature, existing_sig)
            if similarity >= self.threshold:
                return True
        
        # Add to data structures
        self.seen_hashes.add(text_hash)
        self.seen_signatures.append(signature)
        
        return False


def deduplicate_texts(texts: List[str], use_minhash: bool = True) -> List[str]:
    """Remove duplicates based on SHA256 and optionally MinHash."""
    if use_minhash:
        print("Deduplication with MinHash (fuzzy duplicate detection)...")
        deduplicator = MinHashDeduplicator(num_perm=128, threshold=0.8)
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


def train_tokenizer(texts: List[str], vocab_size: int, output_path: str) -> None:
    """Train a SentencePiece tokenizer on the corpus."""
    # Temporary corpus writing
    temp_corpus = output_path + "_temp_corpus.txt"
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
        normalization_rule_name='nmt_nfkc_cf'
    )
    
    # Remove temporary file
    os.remove(temp_corpus)


def tokenize_corpus(texts: List[str], tokenizer_path: str) -> List[List[int]]:
    """Tokenize corpus with SentencePiece tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path + ".model")
    
    tokenized = []
    for text in tqdm(texts, desc="Tokenization"):
        tokens = sp.encode_as_ids(text)
        tokenized.append(tokens)
    
    return tokenized


def save_tokenized_data(tokenized_data: List[List[int]], output_path: str) -> None:
    """Save tokenized data in JSON format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tokenized_data, f)


def main():
    parser = argparse.ArgumentParser(description="Data preparation for training")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to input dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for prepared data")
    parser.add_argument("--vocab_size", type=int, default=32768,
                       help="Vocabulary size for tokenizer")
    parser.add_argument("--min_length", type=int, default=50,
                       help="Minimum text length")
    parser.add_argument("--max_length", type=int, default=10000,
                       help="Maximum text length")
    parser.add_argument("--use_minhash", action="store_true",
                       help="Use MinHash for fuzzy deduplication")
    parser.add_argument("--minhash_threshold", type=float, default=0.8,
                       help="Similarity threshold for MinHash (default: 0.8)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset...")
    if args.input_path.endswith(('.json', '.jsonl')):
        dataset = load_dataset('json', data_files=args.input_path)['train']
    else:
        dataset = load_dataset(args.input_path)['train']
    
    # Text extraction (adapted to dataset structure)
    if 'text' in dataset.column_names:
        texts = dataset['text']
    elif 'content' in dataset.column_names:
        texts = dataset['content']
    else:
        raise ValueError("No 'text' or 'content' column found in dataset")
    
    print(f"Original dataset: {len(texts)} texts")
    
    # Cleaning pipeline
    print("Cleaning texts...")
    cleaned_texts = []
    for text in tqdm(texts, desc="Cleaning"):
        cleaned = clean_text(str(text))
        if cleaned and filter_by_length(cleaned, args.min_length, args.max_length):
            if is_english_text(cleaned):
                cleaned_texts.append(cleaned)
    
    print(f"After cleaning: {len(cleaned_texts)} texts")
    
    # Deduplication
    print("Deduplication...")
    deduplicated_texts = deduplicate_texts(cleaned_texts, use_minhash=args.use_minhash)
    print(f"After deduplication: {len(deduplicated_texts)} texts")
    
    # Deduplication statistics
    dedup_ratio = (len(cleaned_texts) - len(deduplicated_texts)) / len(cleaned_texts) * 100
    print(f"Deduplication rate: {dedup_ratio:.1f}%")
    
    # Tokenizer training
    tokenizer_path = str(output_dir / "tokenizer")
    print(f"Training tokenizer (vocab_size={args.vocab_size})...")
    train_tokenizer(deduplicated_texts, args.vocab_size, tokenizer_path)
    
    # Tokenization
    print("Tokenizing corpus...")
    tokenized_data = tokenize_corpus(deduplicated_texts, tokenizer_path)
    
    # Saving
    output_path = output_dir / "tokenized_data.json"
    print(f"Saving to {output_path}...")
    save_tokenized_data(tokenized_data, str(output_path))
    
    # Final statistics
    total_tokens = sum(len(tokens) for tokens in tokenized_data)
    avg_tokens = total_tokens / len(tokenized_data) if tokenized_data else 0
    
    stats = {
        "num_texts": len(tokenized_data),
        "total_tokens": total_tokens,
        "avg_tokens_per_text": avg_tokens,
        "vocab_size": args.vocab_size
    }
    
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Preparation completed!")
    print(f"- Number of texts: {stats['num_texts']}")
    print(f"- Total tokens: {stats['total_tokens']:,}")
    print(f"- Average tokens/text: {stats['avg_tokens_per_text']:.1f}")


if __name__ == "__main__":
    main()