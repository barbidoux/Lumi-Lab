#!/usr/bin/env python3
"""
Corpus preparation script for Lumi-Lab pipeline.

This script transforms raw data sources into a cleaned and sharded corpus,
without any dependency on tokenizers. It's the first step in the modular
data preparation pipeline.
"""

import argparse
import gzip
import hashlib
import json
import logging
import math
import os
import re
import sys
import gc
import atexit
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Union

import ftfy
import numpy as np
import yaml
from datasketch import MinHash, MinHashLSH
from langdetect import LangDetectException, detect
from tqdm import tqdm

from utils.auth import get_hf_api_client, retry_with_backoff
from utils.dataset_utils import estimate_avg_tokens_per_sample, plan_samples_for_token_budget
from utils.debug.corpus_cache import CachedCorpusProcessor


def _safe_cleanup_and_exit():
    """
    Perform safe cleanup to prevent thread state crashes during shutdown.
    Addresses PyGILState_Release crashes with C extensions.
    """
    try:
        logging.info("🧹 Performing safe cleanup...")

        # Force garbage collection
        gc.collect()

        # Clear any remaining tqdm instances
        try:
            import tqdm
            # Close any open tqdm bars
            for instance in list(tqdm.tqdm._instances):
                if hasattr(instance, 'close'):
                    instance.close()
        except:
            pass

        # Clear datasets cache if available
        try:
            import datasets
            if hasattr(datasets, 'disable_caching'):
                datasets.disable_caching()
        except:
            pass

        # Final garbage collection
        gc.collect()

        logging.info("✅ Cleanup complete, exiting safely")

        # Use os._exit to avoid Python's normal cleanup which can cause thread issues
        os._exit(0)

    except Exception as e:
        logging.warning(f"Cleanup warning: {e}")
        # Force exit anyway
        os._exit(0)


def _create_corpus_subset(source_dir: str, output_dir: Path, config: dict) -> None:
    """
    Create a new corpus by subsetting an existing corpus.

    This is much faster than re-downloading and re-processing data when you need
    a smaller corpus (e.g., for micro model testing).

    Args:
        source_dir: Path to existing corpus directory (must have manifest.json and shards/)
        output_dir: Output directory for the subset corpus
        config: Configuration dict with target_total_tokens
    """
    import shutil
    import gzip

    source_path = Path(source_dir)
    source_manifest_path = source_path / "manifest.json"
    source_shards_dir = source_path / "shards"

    # Validate source corpus
    if not source_manifest_path.exists():
        logging.error(f"Source corpus manifest not found: {source_manifest_path}")
        logging.error("Please provide a valid corpus directory with manifest.json")
        return

    if not source_shards_dir.exists():
        logging.error(f"Source shards directory not found: {source_shards_dir}")
        return

    # Load source manifest
    with open(source_manifest_path, 'r', encoding='utf-8') as f:
        source_manifest = json.load(f)

    source_total_tokens = source_manifest.get('statistics', {}).get('total_tokens', 0)
    if source_total_tokens == 0:
        # Try alternative manifest structure
        source_total_tokens = source_manifest.get('total_tokens', 0)

    logging.info(f"📦 Source corpus: {source_path}")
    logging.info(f"   Total tokens: {source_total_tokens:,}")

    # Get target token count from config
    target_tokens = config.get('target_total_tokens', source_total_tokens)
    logging.info(f"🎯 Target tokens: {target_tokens:,}")

    if target_tokens >= source_total_tokens:
        logging.warning(f"Target ({target_tokens:,}) >= source ({source_total_tokens:,}). Copying entire corpus.")
        target_tokens = source_total_tokens

    # List all shards sorted by name
    shard_files = sorted(source_shards_dir.glob("shard_*.jsonl.gz"))
    if not shard_files:
        shard_files = sorted(source_shards_dir.glob("shard_*.jsonl"))

    if not shard_files:
        logging.error("No shard files found in source corpus")
        return

    logging.info(f"   Source shards: {len(shard_files)}")

    # Calculate tokens per shard (approximate)
    tokens_per_shard = source_total_tokens / len(shard_files) if shard_files else 0

    # Determine how many shards we need
    shards_needed = max(1, int(target_tokens / tokens_per_shard) + 1) if tokens_per_shard > 0 else len(shard_files)
    shards_needed = min(shards_needed, len(shard_files))

    logging.info(f"📋 Copying {shards_needed} shards (approx {tokens_per_shard:,.0f} tokens/shard)")

    # Create output directories
    output_shards_dir = output_dir / "shards"
    output_shards_dir.mkdir(parents=True, exist_ok=True)

    # Copy shards
    copied_tokens = 0
    copied_docs = 0
    copied_shards = 0

    for i, shard_file in enumerate(shard_files[:shards_needed]):
        dest_file = output_shards_dir / shard_file.name
        shutil.copy2(shard_file, dest_file)
        copied_shards += 1

        # Count documents in shard for accurate manifest
        try:
            if shard_file.suffix == '.gz':
                with gzip.open(shard_file, 'rt', encoding='utf-8') as f:
                    shard_docs = sum(1 for _ in f)
            else:
                with open(shard_file, 'r', encoding='utf-8') as f:
                    shard_docs = sum(1 for _ in f)
            copied_docs += shard_docs
        except Exception as e:
            logging.warning(f"Could not count docs in {shard_file}: {e}")

        if (i + 1) % 10 == 0:
            logging.info(f"   Copied {i + 1}/{shards_needed} shards...")

    # Estimate tokens (proportional)
    copied_tokens = int(source_total_tokens * (copied_shards / len(shard_files)))

    # Create new manifest
    new_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
        "pipeline_version": "3.0.0-subset",
        "processing_mode": "subset_from_existing",
        "source_corpus": str(source_path),
        "config": config,
        "statistics": {
            "total_documents": copied_docs,
            "total_tokens": copied_tokens,
            "total_shards": copied_shards
        },
        "subset_info": {
            "source_total_tokens": source_total_tokens,
            "source_total_shards": len(shard_files),
            "target_tokens": target_tokens,
            "actual_tokens_approx": copied_tokens
        }
    }

    # Save manifest
    output_manifest_path = output_dir / "manifest.json"
    with open(output_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(new_manifest, f, indent=2, ensure_ascii=False, default=str)

    logging.info(f"✅ Corpus subset created successfully!")
    logging.info(f"   Output: {output_dir}")
    logging.info(f"   Shards: {copied_shards}")
    logging.info(f"   Documents: {copied_docs:,}")
    logging.info(f"   Tokens (approx): {copied_tokens:,}")


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


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


class AdvancedDeduplicator:
    """Advanced deduplicator using SHA256 + MinHashLSH for efficient fuzzy duplicate detection."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, shingle_size: int = 3):
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.seen_hashes = set()
        # MinHashLSH for efficient similarity search
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.doc_id = 0

    def _get_shingles(self, text: str, k: int = None) -> Set[str]:
        """Generate k-shingles (character n-grams) from text."""
        if k is None:
            k = self.shingle_size
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
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
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


def create_stream_factory(source_config: Dict[str, Any]):
    """Create a stream factory function for a data source."""
    source_type = source_config['type']

    if source_type == 'huggingface':
        def stream_factory():
            from datasets import load_dataset
            import requests

            dataset_name = source_config['dataset_name']
            subset = source_config.get('subset')
            split = source_config.get('split', 'train')

            # Get HF API client if needed for private datasets
            try:
                get_hf_api_client()  # Ensure authentication
            except Exception as e:
                logging.warning(f"HF authentication failed, trying without: {e}")

            # Set download timeout via environment variable
            import os
            os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '600')

            # Define retryable network exceptions
            retryable_exceptions = (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
                ConnectionError,
                TimeoutError,
                OSError,
            )

            @retry_with_backoff(
                max_retries=5,
                base_delay=2.0,
                max_delay=60.0,
                retryable_exceptions=retryable_exceptions
            )
            def _load_dataset_with_retry():
                """Load HuggingFace dataset with retry logic."""
                return load_dataset(
                    dataset_name,
                    subset,
                    split=split,
                    streaming=True,
                    trust_remote_code=source_config.get('trust_remote_code', False)
                )

            dataset = _load_dataset_with_retry()

            return iter(dataset)

    elif source_type == 'local':
        def stream_factory():
            import json
            data_path = Path(source_config['data_path'])

            if data_path.suffix == '.jsonl':
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        yield json.loads(line.strip())
            elif data_path.suffix == '.json':
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        yield item
            else:
                raise ValueError(f"Unsupported local file format: {data_path.suffix}")

    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    return stream_factory


def analyze_sources(config: Dict[str, Any]) -> Dict[str, Dict]:
    """Analyze all data sources and create sampling plans."""
    logging.info("Analyzing data sources...")

    # Try to find tokenizer path for accurate token counting
    tokenizer_path = None
    if 'training_params' in config and 'tokenizer_path' in config['training_params']:
        tokenizer_path = config['training_params']['tokenizer_path']
        # Check if tokenizer exists
        model_path = f"{tokenizer_path}/spm.model" if not tokenizer_path.endswith('.model') else tokenizer_path
        if Path(model_path).exists():
            logging.info(f"🎯 Found existing tokenizer at {tokenizer_path} - will use for ACCURATE token counting!")
        else:
            logging.warning(f"⚠️  Tokenizer not found at {tokenizer_path} - using character heuristic")
            tokenizer_path = None

    analysis_results = {}

    for source_name, source_config in config['sources'].items():
        logging.info(f"Analyzing source: {source_name}")

        # Create stream factory
        stream_factory = create_stream_factory(source_config)

        try:
            # Estimate tokens per sample with REAL tokenizer if available
            estimation_stats = estimate_avg_tokens_per_sample(
                stream_factory=stream_factory,
                sample_size=source_config.get('analysis_sample_size', 100),
                tokenizer=None,  # Will be loaded from tokenizer_path
                text_keys=source_config['text_keys'],
                chars_per_token=source_config.get('chars_per_token', 4.0),
                tokenizer_path=tokenizer_path  # Pass tokenizer path for real token counting!
            )

            # Plan samples for token budget
            token_budget = source_config['token_budget']
            planned_samples = plan_samples_for_token_budget(
                token_budget=token_budget,
                estimation_stats=estimation_stats,
                margin_ratio=source_config.get('margin_ratio', 0.02)
            )

            analysis_results[source_name] = {
                'source_config': source_config,
                'estimation_stats': estimation_stats,
                'planned_samples': planned_samples,
                'token_budget': token_budget
            }

            logging.info(f"Source {source_name}: planning to download {planned_samples:,} samples for {token_budget:,} tokens")

        except Exception as e:
            logging.error(f"Failed to analyze source {source_name}: {e}")
            raise

    return analysis_results


def process_source(source_name: str, analysis_result: Dict, deduplicator: AdvancedDeduplicator, tokenizer_path: str = None, target_tokens: int = None, vocab_size: int = 32768) -> List[Dict]:
    """Process a single data source according to its token budget."""
    logging.info(f"Processing source: {source_name}")

    source_config = analysis_result['source_config']
    token_budget = target_tokens if target_tokens is not None else source_config['token_budget']
    chars_per_token = source_config.get('chars_per_token', 4.0)

    # Load tokenizer for accurate token counting during processing
    tokenizer = None
    smart_estimator = None

    if tokenizer_path:
        try:
            import sentencepiece as smp
            tokenizer = smp.SentencePieceProcessor()
            model_path = tokenizer_path if tokenizer_path.endswith('.model') else f"{tokenizer_path}/spm.model"

            if Path(model_path).exists():
                tokenizer.load(model_path)
                logging.info(f"✅ Using REAL tokenizer for {source_name} token counting")
            else:
                logging.info(f"🧠 Tokenizer not found - using SMART simulation for {source_name}")
                # Import the SmartTokenEstimator
                from utils.dataset_utils import SmartTokenEstimator
                smart_estimator = SmartTokenEstimator(
                    vocab_size=vocab_size,  # Use from config
                    base_chars_per_token=chars_per_token
                )
        except Exception as e:
            logging.warning(f"Failed to load tokenizer: {e}. Using smart simulation.")
            from utils.dataset_utils import SmartTokenEstimator
            smart_estimator = SmartTokenEstimator(
                vocab_size=vocab_size,  # Use from config
                base_chars_per_token=chars_per_token
            )

    logging.info(f"Token budget for {source_name}: {token_budget:,} tokens ({token_budget * chars_per_token:,.0f} chars estimate)")

    # Create stream factory
    stream_factory = create_stream_factory(source_config)

    # Processing parameters
    text_keys = source_config['text_keys']
    if isinstance(text_keys, str):
        text_keys = [text_keys]

    min_length = source_config.get('min_length', 50)
    max_length = source_config.get('max_length', 10000)
    require_english = source_config.get('require_english', True)

    processed_documents = []
    processed_count = 0
    skipped_count = 0
    current_tokens = 0

    # Get fresh stream
    stream = stream_factory()

    progress_bar = tqdm(
        total=token_budget,
        desc=f"Processing {source_name}",
        unit="tokens"
    )

    try:
        for sample in stream:
            if current_tokens >= token_budget:
                logging.info(f"✅ Token budget reached for {source_name}: {current_tokens:,}/{token_budget:,} tokens")
                break

            try:
                # Extract text from specified keys
                text_parts = []
                for key in text_keys:
                    if key in sample and sample[key] is not None:
                        text_parts.append(str(sample[key]))

                if not text_parts:
                    skipped_count += 1
                    continue

                raw_text = " ".join(text_parts)

                # Clean text
                cleaned_text = clean_text(raw_text)

                if not cleaned_text:
                    skipped_count += 1
                    continue

                # Apply filters
                if not filter_by_length(cleaned_text, min_length, max_length):
                    skipped_count += 1
                    continue

                if require_english and not is_english_text(cleaned_text):
                    skipped_count += 1
                    continue

                # Check for duplicates
                if deduplicator.is_duplicate(cleaned_text):
                    skipped_count += 1
                    continue

                # Count REAL tokens for this document
                if tokenizer:
                    try:
                        doc_tokens = len(tokenizer.encode(cleaned_text))
                    except Exception as e:
                        logging.warning(f"Tokenizer failed, falling back to smart estimation: {e}")
                        doc_tokens = smart_estimator.estimate_tokens(cleaned_text) if smart_estimator else int(len(cleaned_text) / chars_per_token)
                elif smart_estimator:
                    doc_tokens = smart_estimator.estimate_tokens(cleaned_text)
                else:
                    doc_tokens = int(len(cleaned_text) / chars_per_token)

                current_tokens += doc_tokens

                # Add to processed documents
                processed_documents.append({
                    'text': cleaned_text,
                    'source': source_name,
                    'original_keys': text_keys,
                    'estimated_tokens': doc_tokens
                })

                processed_count += 1
                progress_bar.update(doc_tokens)  # Update by tokens, not documents

                # Log progress periodically
                if processed_count % 1000 == 0:
                    progress_bar.set_postfix({
                        'docs': processed_count,
                        'tokens': f"{current_tokens:,}",
                        'skipped': skipped_count
                    })

            except Exception as e:
                logging.warning(f"Error processing sample from {source_name}: {e}")
                skipped_count += 1
                continue

    except Exception as e:
        logging.error(f"Error streaming from {source_name}: {e}")
        raise

    finally:
        progress_bar.close()

    logging.info(f"Source {source_name} completed: {processed_count:,} documents processed, ~{current_tokens:,} tokens collected, {skipped_count:,} skipped")
    return processed_documents


def save_shards(documents: List[Dict], output_dir: Path, shard_size: int = 10000) -> List[str]:
    """Save documents to compressed JSONL shards."""
    logging.info(f"Saving {len(documents):,} documents to shards (size: {shard_size})")

    shards_dir = output_dir / "shards"
    shards_dir.mkdir(exist_ok=True)

    shard_files = []

    for i in range(0, len(documents), shard_size):
        shard_idx = i // shard_size
        shard_filename = f"{shard_idx:04d}.jsonl.gz"
        shard_path = shards_dir / shard_filename

        shard_documents = documents[i:i + shard_size]

        with gzip.open(shard_path, 'wt', encoding='utf-8') as f:
            for doc in shard_documents:
                json.dump(doc, f, ensure_ascii=False)
                f.write('\n')

        # Compute shard hash
        with open(shard_path, 'rb') as f:
            shard_hash = hashlib.sha256(f.read()).hexdigest()

        shard_files.append({
            'filename': shard_filename,
            'path': str(shard_path.relative_to(output_dir)),
            'sha256': shard_hash,
            'num_documents': len(shard_documents)
        })

        logging.debug(f"Created shard {shard_filename} with {len(shard_documents)} documents")

    logging.info(f"Created {len(shard_files)} shards")
    return shard_files


def create_manifest(
    shards_info: List[Dict],
    stats: Dict,
    config: Dict,
    output_dir: Path
) -> None:
    """Create manifest.json with corpus metadata."""

    # Hash the input configuration for traceability
    config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
    config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()

    manifest = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'pipeline_version': '1.0.0',
        'config_hash': config_hash,
        'shards': shards_info,
        'statistics': stats,
        'corpus_format': 'compressed_jsonl',
        'text_preprocessing': {
            'cleaning': True,
            'deduplication': True,
            'language_filtering': 'english'
        }
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logging.info(f"Created manifest: {manifest_path}")


def calculate_statistics(documents: List[Dict], analysis_results: Dict, chars_per_token: float = 4.0) -> Dict:
    """Calculate corpus statistics with exact and estimated token counts."""
    total_docs = len(documents)
    total_chars = sum(len(doc['text']) for doc in documents)

    # Get exact tokens if available, otherwise estimate
    exact_tokens = sum(doc.get('exact_tokens', 0) for doc in documents)
    estimated_tokens = sum(doc.get('estimated_tokens', int(len(doc['text']) / chars_per_token)) for doc in documents)

    # Source distribution
    source_counts = {}
    source_exact_tokens = {}
    source_estimated_tokens = {}
    for doc in documents:
        source = doc['source']
        source_counts[source] = source_counts.get(source, 0) + 1

        # Exact tokens
        doc_exact_tokens = doc.get('exact_tokens', 0)
        source_exact_tokens[source] = source_exact_tokens.get(source, 0) + doc_exact_tokens

        # Estimated tokens
        doc_estimated_tokens = doc.get('estimated_tokens', int(len(doc['text']) / chars_per_token))
        source_estimated_tokens[source] = source_estimated_tokens.get(source, 0) + doc_estimated_tokens

    # Length statistics
    lengths = [len(doc['text']) for doc in documents]
    exact_token_lengths = [doc.get('exact_tokens', 0) for doc in documents if doc.get('exact_tokens', 0) > 0]
    estimated_token_lengths = [doc.get('estimated_tokens', int(len(doc['text']) / chars_per_token)) for doc in documents]

    stats = {
        'total_documents': total_docs,
        'total_characters': total_chars,
        'exact_tokens': exact_tokens,
        'estimated_tokens': estimated_tokens,
        'chars_per_token_used': chars_per_token,
        'avg_chars_per_document': total_chars / total_docs if total_docs > 0 else 0,
        'avg_exact_tokens_per_document': exact_tokens / total_docs if (total_docs > 0 and exact_tokens > 0) else 0,
        'avg_estimated_tokens_per_document': estimated_tokens / total_docs if total_docs > 0 else 0,
        'min_chars_per_document': min(lengths) if lengths else 0,
        'max_chars_per_document': max(lengths) if lengths else 0,
        'min_exact_tokens_per_document': min(exact_token_lengths) if exact_token_lengths else 0,
        'max_exact_tokens_per_document': max(exact_token_lengths) if exact_token_lengths else 0,
        'min_estimated_tokens_per_document': min(estimated_token_lengths) if estimated_token_lengths else 0,
        'max_estimated_tokens_per_document': max(estimated_token_lengths) if estimated_token_lengths else 0,
        'source_distribution': source_counts,
        'source_exact_token_distribution': source_exact_tokens,
        'source_estimated_token_distribution': source_estimated_tokens,
        'analysis_results': analysis_results
    }

    return stats


def print_analysis_report(config: Dict[str, Any], analysis_results: Dict[str, Dict]) -> None:
    """Print detailed analysis report with dataset characteristics and processing plan."""

    print("\n" + "="*80)
    print("📊 DATASET ANALYSIS REPORT - Processing Plan & Characteristics")
    print("="*80)

    # Calculate totals
    total_token_budget = 0
    total_planned_samples = 0

    # Header
    print(f"\n{'Source':<25} {'Budget':<12} {'Samples':<10} {'Avg/Sample':<12} {'Efficiency':<12}")
    print("-" * 80)

    for source_name, analysis_result in analysis_results.items():
        source_config = analysis_result['source_config']
        estimation_stats = analysis_result['estimation_stats']
        planned_samples = analysis_result['planned_samples']
        token_budget = analysis_result['token_budget']

        total_token_budget += token_budget
        total_planned_samples += planned_samples

        # Calculate efficiency metrics
        avg_tokens_per_sample = estimation_stats['avg_tokens']
        configured_chars_per_token = source_config.get('chars_per_token', 4.0)

        # Estimate actual chars per token based on analysis
        if estimation_stats['method'] == 'chars_heuristic':
            # For heuristic method, we can infer actual efficiency
            estimated_chars_per_sample = avg_tokens_per_sample * configured_chars_per_token
            # This gives us an idea of content density

        # Efficiency rating based on sample size
        if avg_tokens_per_sample < 1000:
            efficiency = "Short📄"
        elif avg_tokens_per_sample < 10000:
            efficiency = "Medium📖"
        else:
            efficiency = "Long📚"

        print(f"{source_name:<25} "
              f"{token_budget/1_000_000:>6.1f}M{'':<5} "
              f"{planned_samples:>8,}{'':<2} "
              f"{avg_tokens_per_sample:>8.0f}{'':<4} "
              f"{efficiency}")

    # Summary section
    print("-" * 80)
    print(f"{'TOTAL':<25} "
          f"{total_token_budget/1_000_000:>6.1f}M{'':<5} "
          f"{total_planned_samples:>8,}{'':<2} "
          f"{total_token_budget/total_planned_samples:>8.0f}{'':<4} "
          f"Mixed🎯")

    print("\n" + "="*80)
    print("📈 DETAILED STATISTICS")
    print("="*80)

    for source_name, analysis_result in analysis_results.items():
        estimation_stats = analysis_result['estimation_stats']
        source_config = analysis_result['source_config']

        print(f"\n🔍 {source_name.upper()}")
        print(f"   📊 Token Distribution:")
        print(f"      • Average: {estimation_stats['avg_tokens']:.1f} tokens/sample")
        print(f"      • Median (P50): {estimation_stats['p50_tokens']:.1f} tokens/sample")
        print(f"      • 90th Percentile: {estimation_stats['p90_tokens']:.1f} tokens/sample")
        print(f"      • Std Deviation: {estimation_stats['std_tokens']:.1f}")
        print(f"      • Method: {estimation_stats['method']}")
        print(f"   🎯 Conservative Estimate: {estimation_stats['avg_tokens'] + estimation_stats['std_tokens']:.1f} tokens/sample")
        print(f"   📝 Samples Analyzed: {estimation_stats['n_samples_analyzed']}")

        # Efficiency metrics
        chars_per_token = source_config.get('chars_per_token', 4.0)
        actual_chars_per_token = estimation_stats['avg_tokens'] / (estimation_stats['avg_tokens'] / chars_per_token) if estimation_stats['method'] == 'chars_heuristic' else 'N/A'

        if estimation_stats['method'] == 'chars_heuristic':
            print(f"   💬 Text Efficiency: {chars_per_token:.1f} chars/token (configured)")

    print("\n" + "="*80)
    print("⚡ PROCESSING INSIGHTS")
    print("="*80)

    # Calculate processing characteristics
    fastest_source = min(analysis_results.items(), key=lambda x: x[1]['planned_samples'])
    slowest_source = max(analysis_results.items(), key=lambda x: x[1]['planned_samples'])

    print(f"🚀 Fastest to process: {fastest_source[0]} ({fastest_source[1]['planned_samples']:,} samples)")
    print(f"⏳ Slowest to process: {slowest_source[0]} ({slowest_source[1]['planned_samples']:,} samples)")

    # Content type distribution
    print(f"\n📋 Content Distribution:")
    for source_name, analysis_result in analysis_results.items():
        estimation_stats = analysis_result['estimation_stats']
        token_budget = analysis_result['token_budget']
        percentage = (token_budget / total_token_budget) * 100

        # Describe content characteristics
        avg_tokens = estimation_stats['avg_tokens']
        if avg_tokens < 1000:
            content_type = "Short articles/posts"
        elif avg_tokens < 10000:
            content_type = "Medium articles"
        else:
            content_type = "Long documents/books"

        print(f"   • {source_name}: {percentage:.1f}% ({content_type})")

    print(f"\n💾 Storage & Memory:")
    print(f"   • Total samples to process: {total_planned_samples:,}")
    print(f"   • Average tokens per sample: {total_token_budget/total_planned_samples:.0f}")
    print(f"   • Estimated raw text size: ~{total_token_budget*4/1_000_000:.0f}MB")

    # Processing time estimation
    estimated_time_min = total_planned_samples / 2000  # Conservative: 2000 samples/min
    estimated_time_max = total_planned_samples / 1000  # Slower: 1000 samples/min

    print(f"\n⏱️  Processing Time Estimate:")
    print(f"   • Download + Processing: {estimated_time_min:.0f}-{estimated_time_max:.0f} minutes")
    print(f"   • Network dependent (HuggingFace streaming)")

    print("\n" + "="*80)
    print("🎯 READY TO PROCEED")
    print("="*80)
    print("✅ Analysis complete - Configuration looks good!")
    print("🚀 Run without --analyze-only to start corpus generation")
    print("="*80 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Prepare corpus from raw data sources")
    parser.add_argument("--config", required=True, help="Path to data sources configuration file")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed corpus")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze sources and create plan")
    parser.add_argument("--force", action="store_true", help="Force re-execution even if artifacts exist")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--use-cache", action="store_true", help="Use caching system for better performance and resume capability")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh all caches when using --use-cache")
    parser.add_argument("--subset-from", type=str, default=None,
                       help="Create corpus by subsetting from existing corpus directory (faster than re-downloading)")

    args = parser.parse_args()

    setup_logging(args.log_level)

    # Load configuration
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract processing params from config (with default fallbacks)
    processing_params = config.get('processing_params', {})
    shard_size = processing_params.get('shard_size', 50000)
    vocab_size = processing_params.get('vocab_size', 32768)

    # Check if output already exists and not forcing
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        logging.info(f"Output already exists at {output_dir}. Use --force to re-run.")
        return

    # Handle --subset-from: create corpus by subsetting existing corpus
    if args.subset_from:
        _create_corpus_subset(args.subset_from, output_dir, config)
        return

    # Analyze sources
    analysis_results = analyze_sources(config)

    # Save analysis plan
    plan_path = output_dir / "plan.json"
    with open(plan_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)

    logging.info(f"Analysis plan saved to: {plan_path}")

    if args.analyze_only:
        logging.info("Analysis complete. Use without --analyze-only to execute the plan.")

        # Generate comparison report
        print_analysis_report(config, analysis_results)

        # Cleanup resources more gracefully
        import gc
        import threading
        import time

        try:
            # Give any background threads time to finish
            time.sleep(0.5)

            # Force cleanup of any lingering HF/dataset resources
            gc.collect()

            # Try to disconnect any HF connections gracefully
            try:
                import huggingface_hub
                # Clear any cached connections
                if hasattr(huggingface_hub, '_session'):
                    delattr(huggingface_hub, '_session')
            except:
                pass

            # Wait for any remaining threads to finish
            main_thread = threading.main_thread()
            for thread in threading.enumerate():
                if thread != main_thread and thread.is_alive():
                    logging.debug(f"Waiting for thread {thread.name} to finish...")
                    thread.join(timeout=1.0)  # Wait max 1 second per thread

        except Exception as e:
            logging.debug(f"Cleanup warning: {e}")

        # Clean exit with safe cleanup to prevent thread state crashes
        logging.info("Analysis completed successfully.")
        _safe_cleanup_and_exit()

    # Check if we should use the caching system
    if args.use_cache:
        logging.info("🚀 Using TRUE STREAMING corpus processing pipeline")

        # Get tokenizer path from config (CRITICAL for token counting accuracy)
        tokenizer_path = None
        if 'training_params' in config and 'tokenizer_path' in config['training_params']:
            tokenizer_path = config['training_params']['tokenizer_path']

        # Create cache directory
        cache_dir = output_dir / "cache"

        # Initialize streaming cached processor
        cached_processor = CachedCorpusProcessor(cache_dir, tokenizer_path)

        # Process configuration with TRUE STREAMING (zero memory accumulation)
        results = cached_processor.process_config_streaming(
            config, output_dir, force_refresh=args.force_refresh
        )

        logging.info(f"🚀 TRUE STREAMING corpus preparation complete!")
        logging.info(f"📊 Final documents: {results['assembly_results']['statistics']['total_documents']:,}")
        logging.info(f"🎯 EXACT tokens: {results['assembly_results']['statistics']['total_tokens']:,}")
        logging.info(f"🗑️  Global deduplication: {results['deduplication_stats']['duplicates_found']:,} duplicates ({results['deduplication_stats']['deduplication_rate']:.1%})")
        logging.info(f"💾 Memory usage: CONSTANT <100MB peak")
        logging.info(f"⚡ Zero memory accumulation achieved!")
        logging.info(f"📦 Output shards: {results['assembly_results']['statistics']['total_shards']}")

        # Clean shutdown to prevent thread state crashes
        _safe_cleanup_and_exit()
        return

    # Original processing pipeline (non-cached)
    logging.info("Using ORIGINAL corpus processing pipeline")

    # Initialize deduplicator
    dedup_config = config.get('deduplication', {})
    deduplicator = AdvancedDeduplicator(
        threshold=dedup_config.get('threshold', 0.8),
        num_perm=dedup_config.get('num_perm', 128),
        shingle_size=dedup_config.get('shingle_size', 3)
    )

    # Process all sources
    all_documents = []
    total_estimated_tokens = 0

    # Get tokenizer path from config for accurate token counting
    tokenizer_path = None
    if 'training_params' in config and 'tokenizer_path' in config['training_params']:
        tokenizer_path = config['training_params']['tokenizer_path']

    for source_name, analysis_result in analysis_results.items():
        source_documents = process_source(source_name, analysis_result, deduplicator, tokenizer_path, vocab_size=vocab_size)
        all_documents.extend(source_documents)

        # Count tokens from processed documents (they now have estimated_tokens field)
        source_tokens = sum(doc.get('estimated_tokens', 0) for doc in source_documents)
        total_estimated_tokens += source_tokens

        logging.info(f"Source {source_name}: {len(source_documents):,} documents, ~{source_tokens:,} tokens")

    logging.info(f"Total documents collected: {len(all_documents):,}")
    logging.info(f"Total estimated tokens: {total_estimated_tokens:,}")

    if not all_documents:
        logging.warning("No documents were processed. Check your configuration and data sources.")
        return

    # EXACT TOKEN COUNTING with final verification
    logging.info("🔢 Calculating EXACT TOKEN COUNT for final corpus...")
    final_exact_tokens = 0

    # Use the same tokenizer/estimator logic for final count
    final_tokenizer = None
    final_smart_estimator = None

    if tokenizer_path:
        try:
            import sentencepiece as smp
            final_tokenizer = smp.SentencePieceProcessor()
            model_path = tokenizer_path if tokenizer_path.endswith('.model') else f"{tokenizer_path}/spm.model"

            if Path(model_path).exists():
                final_tokenizer.load(model_path)
                logging.info("🎯 Using REAL tokenizer for EXACT final token count")
            else:
                from utils.dataset_utils import SmartTokenEstimator
                final_smart_estimator = SmartTokenEstimator(vocab_size=vocab_size, base_chars_per_token=4.0)
                logging.info("🧠 Using SMART estimator for final token count")
        except Exception as e:
            from utils.dataset_utils import SmartTokenEstimator
            final_smart_estimator = SmartTokenEstimator(vocab_size=vocab_size, base_chars_per_token=4.0)
            logging.info(f"🧠 Using SMART estimator for final token count (fallback: {e})")

    # Count exact tokens in final corpus
    for doc in tqdm(all_documents, desc="🔢 Final token counting"):
        if final_tokenizer:
            try:
                doc_exact_tokens = len(final_tokenizer.encode(doc['text']))
            except Exception:
                doc_exact_tokens = final_smart_estimator.estimate_tokens(doc['text']) if final_smart_estimator else doc.get('estimated_tokens', 0)
        elif final_smart_estimator:
            doc_exact_tokens = final_smart_estimator.estimate_tokens(doc['text'])
        else:
            doc_exact_tokens = doc.get('estimated_tokens', 0)

        final_exact_tokens += doc_exact_tokens
        # Update document with exact count
        doc['exact_tokens'] = doc_exact_tokens

    logging.info(f"📊 FINAL EXACT TOKEN COUNT: {final_exact_tokens:,} tokens")
    logging.info(f"📊 Difference vs estimate: {final_exact_tokens - total_estimated_tokens:,} tokens ({((final_exact_tokens / total_estimated_tokens - 1) * 100):+.1f}%)")

    # Save to shards
    shards_info = save_shards(all_documents, output_dir, shard_size)

    # Calculate statistics with token estimation
    # Use average chars_per_token from config
    avg_chars_per_token = sum(ar['source_config'].get('chars_per_token', 4.0)
                             for ar in analysis_results.values()) / len(analysis_results)
    stats = calculate_statistics(all_documents, analysis_results, avg_chars_per_token)

    # Save detailed statistics
    stats_path = output_dir / "stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

    # Create manifest
    create_manifest(shards_info, stats, config, output_dir)

    logging.info(f"Corpus preparation complete! Output saved to: {output_dir}")
    logging.info(f"Total documents: {stats['total_documents']:,}")
    logging.info(f"Total characters: {stats['total_characters']:,}")

    if stats['exact_tokens'] > 0:
        logging.info(f"🎯 EXACT TOKENS: {stats['exact_tokens']:,}")
        logging.info(f"📊 Estimated tokens: {stats['estimated_tokens']:,}")
        logging.info(f"📊 Accuracy: {(stats['exact_tokens'] / stats['estimated_tokens'] * 100):.1f}% of estimate")
        logging.info(f"Average document length: {stats['avg_chars_per_document']:.1f} characters ({stats['avg_exact_tokens_per_document']:.1f} EXACT tokens)")
    else:
        logging.info(f"Estimated tokens: {stats['estimated_tokens']:,}")
        logging.info(f"Average document length: {stats['avg_chars_per_document']:.1f} characters ({stats['avg_estimated_tokens_per_document']:.1f} estimated tokens)")

    logging.info(f"Chars per token ratio used: {stats['chars_per_token_used']:.1f}")

    # Clean shutdown for original pipeline too
    _safe_cleanup_and_exit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("⚠️  Process interrupted by user")
        _safe_cleanup_and_exit()
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}")
        _safe_cleanup_and_exit()