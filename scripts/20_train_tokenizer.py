#!/usr/bin/env python3
"""
Enhanced Tokenizer Training Script for Lumi-Lab Pipeline v2.1

This script trains a SentencePiece tokenizer on a prepared corpus with:
- Robust imports with intelligent fallbacks
- Enhanced CLI metrics with colors and progress tracking
- Adaptive token estimation based on corpus characteristics
- Memory-efficient streaming processing
- Comprehensive quality validation and testing

Features:
- 🔄 Automatic retry logic for I/O operations
- 📊 Real-time metrics with colored CLI output
- 🎯 Self-calibrating token estimation
- ✅ Comprehensive validation and testing
"""

import argparse
import gzip
import hashlib
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterator, List, Tuple

import sentencepiece as spm
from tqdm import tqdm

# Import our enhanced modules
from utils.debug.robust_imports import (
    smart_sentence_segmentation,
    get_validation_modules,
    is_nltk_available,
    retry_on_failure
)
from utils.debug.tokenizer_metrics import (
    TokenizerTrainingMetrics,
    setup_enhanced_logging
)
from utils.debug.adaptive_estimation import AdaptiveTokenEstimator


# Legacy functions removed - now handled by our robust modules


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load and validate corpus manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # Validate required fields - support both old and new manifest formats
    required_fields = ['shards', 'statistics']
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"Invalid manifest: missing field '{field}'")

    return manifest


def verify_shard_integrity(manifest: Dict[str, Any], corpus_dir: Path) -> bool:
    """Verify integrity of all shards using SHA256 hashes."""
    logging.info("Verifying shard integrity...")

    all_valid = True
    shards = manifest['shards']

    for shard_info in shards:
        # Handle both old format (direct path) and new format (shards/ subdir)
        shard_relative_path = shard_info['path']
        shard_path = corpus_dir / shard_relative_path

        # If not found, try in shards subdirectory
        if not shard_path.exists():
            shard_path = corpus_dir / "shards" / shard_relative_path

        if not shard_path.exists():
            logging.error(f"Shard file missing: {shard_path}")
            all_valid = False
            continue

        # Calculate actual hash
        with open(shard_path, 'rb') as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()

        expected_hash = shard_info['sha256']
        if actual_hash != expected_hash:
            logging.error(f"Hash mismatch for {shard_path}: expected {expected_hash}, got {actual_hash}")
            all_valid = False
        else:
            logging.debug(f"Shard {shard_path.name} integrity verified")

    if all_valid:
        logging.info("All shards verified successfully")
    else:
        logging.error("Shard integrity verification failed")

    return all_valid


@retry_on_failure(max_retries=2, delay=1.0)
def analyze_corpus_statistics(
    manifest: Dict[str, Any],
    corpus_dir: Path,
    metrics: TokenizerTrainingMetrics,
    estimator: AdaptiveTokenEstimator,
    min_sentence_length: int = 5,
    sentence_filter_max_length: int = 2048
) -> Dict[str, Any]:
    """Analyze corpus statistics with enhanced metrics and adaptive estimation."""
    shards = manifest['shards']
    total_docs = sum(shard['num_documents'] for shard in shards)

    metrics.start_phase("Corpus Analysis")

    shard_stats = []
    total_estimated_tokens = 0
    total_sentences = 0
    total_characters = 0

    for i, shard_info in enumerate(shards):
        shard_relative_path = shard_info['path']
        shard_path = corpus_dir / shard_relative_path

        if not shard_path.exists():
            shard_path = corpus_dir / "shards" / shard_relative_path

        # Debug logging to ensure all shards are processed
        logging.debug(f"📋 Processing shard {i+1}/{len(shards)}: {shard_path.name}")

        if not shard_path.exists():
            logging.warning(f"⚠️  Shard not found: {shard_path}")
            continue

        # Update progress
        metrics.log_progress(i, len(shards), "shards", {
            'current_shard': shard_path.name,
            'sentences': total_sentences,
            'chars': total_characters
        })

        shard_sentences = 0
        shard_characters = 0
        shard_estimated_tokens = 0

        try:
            with gzip.open(shard_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        doc = json.loads(line.strip())
                        text = doc['text']

                        # Segment into sentences using robust method
                        sentences = smart_sentence_segmentation(text)

                        # Apply same preprocessing as sentence streaming for consistency
                        valid_sentences = []
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if min_sentence_length <= len(sentence) <= sentence_filter_max_length:
                                valid_sentences.append(sentence)

                        # Count based on processed sentences (consistent with streaming)
                        processed_text = '\n'.join(valid_sentences)
                        shard_sentences += len(valid_sentences)
                        shard_characters += len(processed_text)
                        shard_estimated_tokens += estimator.estimate_tokens_in_text(processed_text)

                    except json.JSONDecodeError as e:
                        metrics.update_metrics(errors_encountered=1)
                        continue

        except Exception as e:
            raise Exception(f"Error analyzing shard {shard_path}: {e}")

        # Debug logging to confirm shard completion
        logging.debug(f"✅ Completed shard {i+1}/{len(shards)}: {shard_sentences:,} sentences, {shard_estimated_tokens:,} tokens")

        shard_stat = {
            'path': shard_info['path'],
            'documents': shard_info['num_documents'],
            'manifest_tokens': shard_info['num_tokens'],
            'estimated_tokens': shard_estimated_tokens,
            'sentences': shard_sentences,
            'characters': shard_characters,
            'file_size_mb': shard_info['file_size_bytes'] / (1024 * 1024),
            'sentences_per_doc': round(shard_sentences / shard_info['num_documents'], 1)
        }

        shard_stats.append(shard_stat)
        total_estimated_tokens += shard_estimated_tokens
        total_sentences += shard_sentences
        total_characters += shard_characters

        # Update metrics
        metrics.update_metrics(
            files_processed=1,
            documents_processed=shard_info['num_documents'],
            sentences_generated=shard_sentences,
            characters_processed=shard_characters,
            estimated_tokens=shard_estimated_tokens
        )

    print()  # New line after progress

    # Final verification that all shards were processed
    logging.info(f"📋 Corpus Analysis Summary: {len(shard_stats)}/{len(shards)} shards processed successfully")
    if len(shard_stats) != len(shards):
        logging.warning(f"⚠️  Only {len(shard_stats)} out of {len(shards)} shards were processed!")

    stats = {
        'total_documents': total_docs,
        'total_sentences': total_sentences,
        'total_characters': total_characters,
        'manifest_total_tokens': sum(s['num_tokens'] for s in shards),
        'estimated_total_tokens': total_estimated_tokens,
        'avg_sentences_per_doc': round(total_sentences / total_docs, 1),
        'avg_chars_per_sentence': round(total_characters / total_sentences, 1) if total_sentences > 0 else 0,
        'shard_statistics': shard_stats
    }

    metrics.complete_phase("Corpus Analysis")
    metrics.log_metrics_update()

    return stats


@retry_on_failure(max_retries=2, delay=1.0)
def stream_sentences_from_shards(
    manifest: Dict[str, Any],
    corpus_dir: Path,
    metrics: TokenizerTrainingMetrics,
    estimator: AdaptiveTokenEstimator,
    max_sentences: int = 10000000,
    min_sentence_length: int = 5,
    sentence_filter_max_length: int = 2048
) -> Iterator[Tuple[str, Dict[str, int]]]:
    """Stream sentences from compressed shards with enhanced progress tracking."""
    shards = manifest['shards']
    total_docs = sum(shard['num_documents'] for shard in shards)

    metrics.start_phase("Sentence Streaming")

    stats = {
        'documents_processed': 0,
        'sentences_yielded': 0,
        'sentences_filtered': 0,
        'characters_processed': 0,
        'estimated_tokens': 0
    }

    for shard_idx, shard_info in enumerate(shards):
        if stats['sentences_yielded'] >= max_sentences:
            break

        # Handle both old format (direct path) and new format (shards/ subdir)
        shard_relative_path = shard_info['path']
        shard_path = corpus_dir / shard_relative_path

        if not shard_path.exists():
            shard_path = corpus_dir / "shards" / shard_relative_path

        # Debug logging to track shard processing
        logging.debug(f"📋 Streaming from shard {shard_idx+1}/{len(shards)}: {shard_path.name}")

        if not shard_path.exists():
            logging.warning(f"⚠️  Shard not found: {shard_path}")
            continue

        try:
            with gzip.open(shard_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if stats['sentences_yielded'] >= max_sentences:
                        break

                    try:
                        doc = json.loads(line.strip())
                        text = doc['text']

                        # Segment document into sentences using robust method
                        sentences = smart_sentence_segmentation(text)

                        # Apply same preprocessing as corpus analysis for consistency
                        valid_sentences = []
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if min_sentence_length <= len(sentence) <= sentence_filter_max_length:
                                valid_sentences.append(sentence)

                        # Update statistics based on filtered sentences only (consistent with analysis)
                        processed_text = '\n'.join(valid_sentences)
                        stats['documents_processed'] += 1
                        stats['characters_processed'] += len(processed_text)
                        # Only estimate tokens for processed (filtered) text, not the raw document
                        stats['estimated_tokens'] += estimator.estimate_tokens_in_text(processed_text)
                        sentences = valid_sentences  # Use filtered sentences

                        # Yield sentences
                        for sentence in sentences:
                            if stats['sentences_yielded'] >= max_sentences:
                                break

                            yield sentence, stats
                            stats['sentences_yielded'] += 1

                        # Update metrics periodically
                        if stats['documents_processed'] % 1000 == 0:
                            metrics.update_metrics(
                                documents_processed=1000,
                                sentences_generated=len(sentences),
                                characters_processed=len(processed_text),
                                estimated_tokens=estimator.estimate_tokens_in_text(processed_text)
                            )

                            # Log progress
                            metrics.log_progress(
                                stats['sentences_yielded'],
                                max_sentences,
                                "sentences",
                                {
                                    'docs': stats['documents_processed'],
                                    'tokens': stats['estimated_tokens']
                                }
                            )

                    except json.JSONDecodeError:
                        metrics.update_metrics(errors_encountered=1)
                        continue

        except Exception as e:
            raise Exception(f"Error reading shard {shard_path}: {e}")

        # Debug logging to confirm shard completion in streaming
        logging.debug(f"✅ Completed streaming from shard {shard_idx+1}/{len(shards)}: {shard_path.name}")

    print()  # New line after progress
    metrics.complete_phase("Sentence Streaming")


def train_sentencepiece_tokenizer(
    sentence_iterator: Iterator[Tuple[str, Dict[str, int]]],
    output_dir: Path,
    estimator: AdaptiveTokenEstimator,
    corpus_stats: Dict[str, Any],
    vocab_size: int = 32000,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    normalization_rule_name: str = "nfkc",
    max_training_sentences: int = 10000000,
    max_sentence_length: int = 32768,
    shuffle_input_sentence: bool = True,
    seed_sentencepiece_size: int = 1000000,
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = 2,
    eos_id: int = 3,
    min_sentence_length: int = 5,
    control_symbols: list = None,
    num_threads: int = 8,
    train_extremely_large_corpus: bool = True,
    split_digits: bool = True,
    allow_whitespace_only_pieces: bool = False,
    byte_fallback: bool = False
) -> Dict[str, Any]:
    """Train a SentencePiece tokenizer with proper sentence handling."""
    logging.info(f"🚀 Training SentencePiece tokenizer:")
    logging.info(f"  📄 vocab_size={vocab_size:,}, model_type={model_type}")
    logging.info(f"  📊 character_coverage={character_coverage}")
    logging.info(f"  🔍 max_sentence_length={max_sentence_length:,}")

    # Create temporary file for training data
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as temp_file:
        temp_path = Path(temp_file.name)

        logging.info("📋 Writing training sentences to temporary file...")
        sentences_written = 0
        total_chars = 0
        sentences_skipped = 0
        estimated_tokens = 0

        try:
            for sentence, stats in sentence_iterator:
                if sentences_written >= max_training_sentences:
                    break

                # Additional validation for sentence quality
                sentence = sentence.strip()

                # Skip if too short, too long, or low quality
                if not sentence:
                    continue

                if len(sentence) < min_sentence_length:
                    sentences_skipped += 1
                    continue

                if len(sentence) > max_sentence_length:
                    # Truncate very long sentences instead of skipping
                    sentence = sentence[:max_sentence_length].strip()
                    if len(sentence) < min_sentence_length:
                        sentences_skipped += 1
                        continue

                # Write sentence
                temp_file.write(sentence + '\n')
                sentences_written += 1
                total_chars += len(sentence)
                estimated_tokens += estimator.estimate_tokens_in_text(sentence)

                # Progress is already tracked by the progress bar in stream_sentences_from_shards
                # Removed redundant logging that interferes with progress bar display

        except Exception as e:
            logging.error(f"Error writing training data: {e}")
            temp_path.unlink(missing_ok=True)
            raise

    # Training statistics
    avg_chars_per_sentence = round(total_chars / max(1, sentences_written), 1)
    avg_tokens_per_sentence = round(estimated_tokens / max(1, sentences_written), 1)

    logging.info(f"🎯 Training data prepared:")
    logging.info(f"  📄 {sentences_written:,} sentences written")
    logging.info(f"  ⛔ {sentences_skipped:,} sentences skipped")
    logging.info(f"  📊 {estimated_tokens:,} estimated tokens")
    logging.info(f"  📈 {avg_chars_per_sentence} chars/sentence, {avg_tokens_per_sentence} tokens/sentence")

    if sentences_written == 0:
        raise ValueError("No valid sentences found for training")

    # Prepare SentencePiece training arguments with optimized settings
    model_prefix = str(output_dir / "spm")

    spm_args = [
        f'--input={temp_path}',
        f'--model_prefix={model_prefix}',
        f'--vocab_size={vocab_size}',
        f'--model_type={model_type}',
        f'--character_coverage={character_coverage}',
        f'--normalization_rule_name={normalization_rule_name}',
        '--add_dummy_prefix=true',
        '--remove_extra_whitespaces=true',
        f'--max_sentence_length={max_sentence_length}',  # Use dynamic max length
        f'--shuffle_input_sentence={str(shuffle_input_sentence).lower()}',
        f'--input_sentence_size={min(sentences_written, seed_sentencepiece_size)}',
        f'--train_extremely_large_corpus={str(train_extremely_large_corpus).lower()}',
        f'--num_threads={num_threads}',
        # Explicit special token IDs from config
        f'--pad_id={pad_id}',
        f'--unk_id={unk_id}',
        f'--bos_id={bos_id}',
        f'--eos_id={eos_id}',
        # Additional optimization flags
        f'--split_digits={str(split_digits).lower()}',
        f'--allow_whitespace_only_pieces={str(allow_whitespace_only_pieces).lower()}',
        f'--byte_fallback={str(byte_fallback).lower()}'
    ]

    # Add special control tokens from config
    if control_symbols is None:
        control_symbols = ['<pad>', '<mask>']
    if control_symbols:
        control_symbols_str = ','.join(control_symbols)
        spm_args.append(f'--control_symbols={control_symbols_str}')

    training_stats = {
        'sentences_used': sentences_written,
        'sentences_skipped': sentences_skipped,
        'estimated_tokens': estimated_tokens,
        'total_characters': total_chars,
        'avg_chars_per_sentence': avg_chars_per_sentence,
        'avg_tokens_per_sentence': avg_tokens_per_sentence,
        'documents_processed': 0  # Will be set by caller with actual count
    }

    try:
        # Count total arguments for SentencePiece training
        logging.info(f"🎆 Starting SentencePiece training with {len(spm_args)} configuration parameters...")
        logging.debug(f"🔧 Command arguments: {spm_args}")

        # Train the tokenizer
        spm.SentencePieceTrainer.train(' '.join(spm_args))
        logging.info("✅ SentencePiece training completed successfully")

        # Load trained tokenizer to get actual token statistics
        model_path = output_dir / "spm.model"
        sp = spm.SentencePieceProcessor(model_file=str(model_path))

        # RIGOROUS CALIBRATION: Use statistical sampling across entire training file
        logging.info("📊 Performing rigorous tokenizer calibration...")

        # Count total lines first for proper sampling
        with temp_path.open('r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        # Statistical sampling: sample every N-th line for representativeness
        sample_size = min(5000, total_lines)  # Use up to 5000 samples
        sampling_interval = max(1, total_lines // sample_size)

        sample_lines = []
        total_sample_chars = 0

        with temp_path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i % sampling_interval == 0 and len(sample_lines) < sample_size:
                    line_content = line.strip()
                    if line_content:  # Skip empty lines
                        sample_lines.append(line_content)
                        total_sample_chars += len(line_content)

        # Tokenize samples to get accurate ratio
        total_sample_tokens = 0
        for line in sample_lines:
            tokens = sp.encode(line, out_type=int)
            total_sample_tokens += len(tokens)

        # Calculate accurate chars/token ratio from representative sampling
        if total_sample_tokens > 0:
            actual_chars_per_token = total_sample_chars / total_sample_tokens

            # Use CORPUS total characters, not training file characters
            corpus_total_chars = corpus_stats['total_characters']
            actual_estimated_tokens = corpus_total_chars / actual_chars_per_token

            # Update with scientifically rigorous estimate
            training_stats['actual_estimated_tokens'] = int(actual_estimated_tokens)
            training_stats['actual_chars_per_token'] = actual_chars_per_token
            training_stats['calibration_sample_size'] = len(sample_lines)
            training_stats['calibration_sampling_method'] = 'statistical_interval'
            training_stats['corpus_chars_used'] = corpus_total_chars

            logging.info(f"📊 Rigorous token estimation calibration:")
            logging.info(f"   🔬 Sampling method: Statistical interval sampling")
            logging.info(f"   📋 Representative sample: {len(sample_lines):,} sentences (every {sampling_interval})")
            logging.info(f"   🔤 Sample chars: {total_sample_chars:,}")
            logging.info(f"   🎯 Sample tokens: {total_sample_tokens:,}")
            logging.info(f"   📏 Calibrated ratio: {actual_chars_per_token:.4f} chars/token")
            logging.info(f"   🎯 Corpus total chars: {corpus_total_chars:,}")
            logging.info(f"   🎯 Final estimated tokens: {int(actual_estimated_tokens):,}")

    except Exception as e:
        logging.error(f"SentencePiece training failed: {e}")
        raise

    finally:
        # Clean up temporary file
        temp_path.unlink(missing_ok=True)

    # Verify output files exist
    model_path = output_dir / "spm.model"
    vocab_path = output_dir / "spm.vocab"

    if not model_path.exists():
        raise FileNotFoundError(f"Expected tokenizer model not created: {model_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Expected tokenizer vocab not created: {vocab_path}")

    logging.info(f"💾 Tokenizer saved to: {model_path}")
    logging.info(f"📁 Vocabulary saved to: {vocab_path}")

    return training_stats


def create_tokenizer_config(
    output_dir: Path,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
    normalization_rule_name: str,
    manifest_hash: str
) -> None:
    """Create tokenizer configuration metadata."""
    config = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'pipeline_version': '1.0.0',
        'tokenizer_type': 'sentencepiece',
        'vocab_size': vocab_size,
        'model_type': model_type,
        'character_coverage': character_coverage,
        'normalization_rule_name': normalization_rule_name,
        'corpus_manifest_hash': manifest_hash,
        'model_file': 'spm.model',
        'vocab_file': 'spm.vocab'
    }

    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logging.info(f"Tokenizer config saved to: {config_path}")


def test_tokenizer(tokenizer_dir: Path) -> None:
    """Test the trained tokenizer with sample text."""
    model_path = tokenizer_dir / "spm.model"

    try:
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))

        # Test texts
        test_texts = [
            "Hello, world! This is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning and artificial intelligence are transforming our world."
        ]

        logging.info("Testing tokenizer...")

        for i, text in enumerate(test_texts, 1):
            # Encode
            tokens = sp.encode(text)
            # Decode
            decoded = sp.decode(tokens)

            logging.info(f"Test {i}:")
            logging.info(f"  Original: {text}")
            logging.info(f"  Tokens ({len(tokens)}): {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            logging.info(f"  Decoded: {decoded}")

        vocab_size = sp.vocab_size()
        logging.info(f"Tokenizer vocabulary size: {vocab_size:,}")

    except Exception as e:
        logging.error(f"Tokenizer test failed: {e}")
        raise


def main():
    """Enhanced main execution with robust imports, adaptive estimation, and rich CLI metrics."""
    parser = argparse.ArgumentParser(
        description="Enhanced Tokenizer Training v2.1 - Train SentencePiece tokenizer with adaptive estimation and rich CLI metrics"
    )
    parser.add_argument("--config", required=True, help="Path to tokenizer config JSON (e.g., config/pretrain/tokenizer/spm32k.json)")
    parser.add_argument("--output-dir", required=True, help="Output directory for tokenizer")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze corpus statistics without training")
    parser.add_argument("--force", action="store_true", help="Force re-training even if tokenizer exists")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Load tokenizer config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Tokenizer config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        tokenizer_config = json.load(f)

    # Extract parameters from config
    vocab_size = tokenizer_config['tokenizer_params']['vocab_size']
    model_type = tokenizer_config['tokenizer_params']['model_type']
    character_coverage = tokenizer_config['tokenizer_params']['character_coverage']
    normalization_rule = tokenizer_config['tokenizer_params']['normalization_rule']

    max_training_sentences = tokenizer_config['training_params']['max_training_sentences']
    max_sentence_length = tokenizer_config['training_params']['max_sentence_length']
    calibration_samples = tokenizer_config['training_params']['calibration_samples']
    shuffle_input_sentence = tokenizer_config['training_params'].get('shuffle_input_sentence', True)
    seed_sentencepiece_size = tokenizer_config['training_params'].get('seed_sentencepiece_size', 1000000)

    # NEW: Load sentence filtering parameters
    min_sentence_length = tokenizer_config['training_params'].get('min_sentence_length', 5)
    sentence_filter_max_length = tokenizer_config['training_params'].get('sentence_filter_max_length', 2048)
    initial_chars_per_token = tokenizer_config['training_params'].get('initial_chars_per_token', 4.0)

    # NEW: Load SentencePiece advanced flags
    num_threads = tokenizer_config['training_params'].get('num_threads', 8)
    train_extremely_large_corpus = tokenizer_config['training_params'].get('train_extremely_large_corpus', True)
    split_digits = tokenizer_config['training_params'].get('split_digits', True)
    allow_whitespace_only_pieces = tokenizer_config['training_params'].get('allow_whitespace_only_pieces', False)
    byte_fallback = tokenizer_config['training_params'].get('byte_fallback', False)

    # Get special tokens from config
    special_tokens_config = tokenizer_config.get('special_tokens', {})
    pad_id = special_tokens_config.get('pad_id', 0)
    unk_id = special_tokens_config.get('unk_id', 1)
    bos_id = special_tokens_config.get('bos_id', 2)
    eos_id = special_tokens_config.get('eos_id', 3)
    control_symbols = special_tokens_config.get('control_symbols', ['<pad>', '<mask>'])

    # NEW: Load validation parameters
    validation_params = tokenizer_config.get('validation_params', {})
    dataset_sample_rate = validation_params.get('dataset_sample_rate', 0.1)
    tokenizer_validation_samples = validation_params.get('tokenizer_validation_samples', 500)

    # Get corpus manifest path from config
    corpus_manifest_path = tokenizer_config['corpus_manifest']
    manifest_path = Path(corpus_manifest_path)

    # Setup enhanced logging
    setup_enhanced_logging(args.log_level)

    # Initialize metrics tracker
    metrics = TokenizerTrainingMetrics("Enhanced Tokenizer Training")

    # Validate inputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if tokenizer already exists
    model_path = output_dir / "spm.model"
    config_path = output_dir / "tokenizer_config.json"

    if model_path.exists() and config_path.exists() and not args.force and not args.analyze_only:
        print(f"✅ Tokenizer already exists at {output_dir}. Use --force to re-train.")
        return

    # Load and validate manifest
    manifest = load_manifest(manifest_path)
    corpus_dir = manifest_path.parent

    # Display configuration header
    config_info = {
        'corpus_path': str(manifest_path),
        'total_shards': len(manifest['shards']),
        'total_docs': sum(shard['num_documents'] for shard in manifest['shards']),
        'vocab_size': vocab_size,
        'model_type': model_type,
        'character_coverage': character_coverage,
        'normalization_rule': normalization_rule
    }
    metrics.log_header(config_info)

    # Initialize adaptive estimator
    estimator = AdaptiveTokenEstimator(initial_estimate=initial_chars_per_token)

    # Calibrate estimation
    old_estimate = estimator.current_estimate
    estimator.calibrate_from_corpus(manifest, corpus_dir, calibration_samples)

    if abs(estimator.current_estimate - old_estimate) > 0.1:
        metrics.log_estimation_calibration(old_estimate, estimator.current_estimate, calibration_samples)

    # Create manifest hash for traceability
    config_str = json.dumps(manifest.get('config', manifest), sort_keys=True, ensure_ascii=False)
    manifest_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()

    # Verify corpus integrity with retry logic
    metrics.start_phase("Integrity Verification")
    if not verify_shard_integrity(manifest, corpus_dir):
        raise ValueError("Corpus integrity verification failed")
    metrics.complete_phase("Integrity Verification")

    # Analyze corpus statistics
    corpus_stats = analyze_corpus_statistics(
        manifest, corpus_dir, metrics, estimator,
        min_sentence_length, sentence_filter_max_length
    )

    # Save corpus analysis
    stats_path = output_dir / "corpus_analysis.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(corpus_stats, f, indent=2, ensure_ascii=False)

    if args.analyze_only:
        print("✅ Corpus analysis complete! Remove --analyze-only to train tokenizer.")
        return

    # Stream sentences for training
    sentence_iterator = stream_sentences_from_shards(
        manifest, corpus_dir, metrics, estimator, max_training_sentences,
        min_sentence_length, sentence_filter_max_length
    )

    # Train tokenizer with enhanced pipeline
    training_stats = train_sentencepiece_tokenizer(
        sentence_iterator=sentence_iterator,
        output_dir=output_dir,
        estimator=estimator,
        corpus_stats=corpus_stats,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        normalization_rule_name=normalization_rule,
        max_training_sentences=max_training_sentences,
        max_sentence_length=max_sentence_length,
        shuffle_input_sentence=shuffle_input_sentence,
        seed_sentencepiece_size=seed_sentencepiece_size,
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
        min_sentence_length=min_sentence_length,
        control_symbols=control_symbols,
        num_threads=num_threads,
        train_extremely_large_corpus=train_extremely_large_corpus,
        split_digits=split_digits,
        allow_whitespace_only_pieces=allow_whitespace_only_pieces,
        byte_fallback=byte_fallback
    )

    # Fix documents_processed count from manifest
    total_documents = sum(shard['num_documents'] for shard in manifest['shards'])
    training_stats['documents_processed'] = total_documents

    # Create enhanced configuration with training stats
    enhanced_config = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'pipeline_version': '2.0.0-enhanced',
        'tokenizer_type': 'sentencepiece',
        'vocab_size': vocab_size,
        'model_type': model_type,
        'character_coverage': character_coverage,
        'normalization_rule_name': normalization_rule,
        'max_sentence_length': max_sentence_length,
        'corpus_manifest_hash': manifest_hash,
        'model_file': 'spm.model',
        'vocab_file': 'spm.vocab',
        'corpus_statistics': corpus_stats,
        'training_statistics': training_stats
    }

    enhanced_config_path = output_dir / "tokenizer_config.json"
    with open(enhanced_config_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_config, f, indent=2, ensure_ascii=False)

    logging.info(f"📋 Enhanced tokenizer config saved to: {enhanced_config_path}")

    # Test tokenizer with basic functionality
    metrics.start_phase("Tokenizer Testing")
    test_tokenizer(output_dir)
    metrics.complete_phase("Tokenizer Testing")

    # Comprehensive validation if modules are available
    validation_modules = get_validation_modules()
    validation_results = None  # Initialize to ensure it's always defined

    if validation_modules:
        metrics.start_phase("Quality Validation")

        try:
            # Precise token counting validation
            token_counter = validation_modules['PreciseTokenCounter'](output_dir)
            dataset_validation = token_counter.validate_dataset_quality(
                manifest, corpus_dir, sample_rate=dataset_sample_rate
            )

            # Comprehensive tokenizer validation
            validator = validation_modules['TokenizerValidator'](output_dir)

            # Extract sample texts for validation from ALL shards (representative sampling)
            sample_texts = []
            target_samples = tokenizer_validation_samples
            samples_per_shard = max(1, target_samples // len(manifest['shards']))

            logging.info(f"🎯 Sampling {samples_per_shard} documents from each of {len(manifest['shards'])} shards")

            for shard_idx, shard_info in enumerate(manifest['shards']):
                if len(sample_texts) >= target_samples:
                    break

                shard_path = corpus_dir / shard_info['path']
                if not shard_path.exists():
                    shard_path = corpus_dir / "shards" / shard_info['path']

                try:
                    with gzip.open(shard_path, 'rt', encoding='utf-8') as f:
                        shard_samples = 0
                        for i, line in enumerate(f):
                            if shard_samples >= samples_per_shard:
                                break
                            # Sample every Nth document for representativeness
                            sampling_interval = max(1, shard_info['num_documents'] // (samples_per_shard * 2))
                            if i % sampling_interval == 0:
                                try:
                                    doc = json.loads(line.strip())
                                    sample_texts.append(doc['text'])
                                    shard_samples += 1
                                except json.JSONDecodeError:
                                    continue

                    logging.debug(f"   📋 Shard {shard_idx}: {shard_samples} samples extracted")
                except Exception as e:
                    logging.warning(f"   ⚠️ Shard {shard_idx} sampling failed: {e}")
                    continue

            logging.info(f"✅ Total validation samples collected: {len(sample_texts)} from {len(manifest['shards'])} shards")

            validation_results = validator.comprehensive_validation(sample_texts)

            # Generate detailed report
            report_path = output_dir / "tokenizer_validation_report.json"
            validator.generate_report(validation_results, report_path)

            # SCIENTIFIC RIGOR: Cross-validation between calibration and precise counting
            calibration_estimate = training_stats.get('actual_estimated_tokens', 0)
            precise_count_extrapolated = int(dataset_validation['token_counting_results']['total_exact_tokens'] *
                                           (corpus_stats['total_documents'] / dataset_validation['token_counting_results']['total_documents']))

            estimation_accuracy = (calibration_estimate / precise_count_extrapolated) * 100 if precise_count_extrapolated > 0 else 0
            estimation_difference = abs(calibration_estimate - precise_count_extrapolated)

            cross_validation_results = {
                'calibration_method_tokens': calibration_estimate,
                'precise_counting_extrapolated_tokens': precise_count_extrapolated,
                'cross_validation_accuracy_percent': estimation_accuracy,
                'absolute_difference': estimation_difference,
                'relative_error_percent': abs(100 - estimation_accuracy),
                'method_agreement': 'excellent' if abs(estimation_accuracy - 100) < 5 else 'good' if abs(estimation_accuracy - 100) < 10 else 'poor'
            }

            logging.info(f"🔬 Scientific cross-validation:")
            logging.info(f"   📊 Calibration estimate: {calibration_estimate:,} tokens")
            logging.info(f"   🎯 Precise count (extrapolated): {precise_count_extrapolated:,} tokens")
            logging.info(f"   📈 Cross-validation accuracy: {estimation_accuracy:.2f}%")
            logging.info(f"   📊 Method agreement: {cross_validation_results['method_agreement']}")

            if abs(estimation_accuracy - 100) > 10:
                logging.warning(f"⚠️  Large discrepancy detected between estimation methods!")

                # Analyze corpus heterogeneity to explain discrepancy
                shard_stats = corpus_stats.get('shard_statistics', [])
                normal_shards = []
                anomalous_shards = []

                for shard in shard_stats:
                    docs_per_shard = shard.get('documents', 1)
                    sentences_per_doc = shard.get('sentences_per_doc', 0)

                    if sentences_per_doc > 1000:  # Threshold for anomalous shards
                        anomalous_shards.append({
                            'path': shard.get('path', 'unknown'),
                            'docs': docs_per_shard,
                            'sentences_per_doc': sentences_per_doc,
                            'estimated_tokens': shard.get('estimated_tokens', 0)
                        })
                    else:
                        normal_shards.append(shard)

                if anomalous_shards:
                    logging.warning(f"   📊 Corpus heterogeneity detected:")
                    logging.warning(f"      • Normal shards: {len(normal_shards)} (avg ~400 sentences/doc)")
                    logging.warning(f"      • Anomalous shards: {len(anomalous_shards)} (books/long texts)")

                    for anomaly in anomalous_shards:
                        logging.warning(f"      • {anomaly['path']}: {anomaly['docs']} docs, {anomaly['sentences_per_doc']:.0f} sent/doc")

                logging.warning(f"   🔍 Discrepancy analysis:")
                logging.warning(f"      • Calibration method samples uniformly across sentences")
                logging.warning(f"      • Long documents have disproportionate impact on total count")
                logging.warning(f"      • Statistical sampling under-represents document length variance")
                logging.warning(f"   📊 Recommended token count: {precise_count_extrapolated:,} (precise method)")

                # Add detailed analysis to cross_validation_results
                cross_validation_results['corpus_heterogeneity_analysis'] = {
                    'normal_shards_count': len(normal_shards),
                    'anomalous_shards_count': len(anomalous_shards),
                    'anomalous_shards_details': anomalous_shards,
                    'heterogeneity_impact_explanation': (
                        "Calibration method uses uniform sentence sampling which under-represents "
                        "the contribution of very long documents (books, articles). Precise counting "
                        "method accounts for actual document structure and is more reliable."
                    )
                }

            # Update enhanced config with validation results
            enhanced_config['validation_results'] = {
                'dataset_validation': dataset_validation,
                'cross_validation_analysis': cross_validation_results,
                'tokenizer_validation_summary': {
                    'overall_quality_score': validation_results.get('overall_quality_score', 0),
                    'unk_rate_percent': validation_results.get('coverage_analysis', {}).get('unk_rate_percent', 0),
                    'coverage_quality': validation_results.get('coverage_analysis', {}).get('coverage_quality', 'unknown')
                }
            }

        except Exception as e:
            print(f"⚠️  Validation encountered issues: {e}")
            logging.error(f"Validation exception details: {e}", exc_info=True)
            # Ensure validation_results stays None on exception

        metrics.complete_phase("Quality Validation")

    # Save final enhanced config
    enhanced_config_path = output_dir / "tokenizer_config.json"
    try:
        with open(enhanced_config_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_config, f, indent=2, ensure_ascii=False)
        logging.info(f"📋 Enhanced config saved successfully")
    except Exception as e:
        logging.error(f"Failed to save enhanced config: {e}")

    # Generate comprehensive final report
    try:
        logging.info("📊 Generating final report...")
        metrics.log_final_report(training_stats, validation_results)
    except Exception as e:
        logging.error(f"Failed to generate final report: {e}", exc_info=True)
        print(f"❌ Report generation failed: {e}")


if __name__ == "__main__":
    main()