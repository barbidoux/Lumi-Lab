#!/usr/bin/env python3
"""
Sequence packing script for Lumi-Lab pipeline.
v2.1: Enhanced with scientific logging and real-time token-based progress.

This script tokenizes the corpus and packs it into fixed-length sequences
in an optimized binary format for training. It's the third and final step
in the modular data preparation pipeline.
"""

import argparse
import gzip
import hashlib
import json
import logging
import time
import struct
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterator, List

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load and validate corpus or tokenizer manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    return manifest


def load_tokenizer(tokenizer_dir: Path) -> spm.SentencePieceProcessor:
    """Load and validate SentencePiece tokenizer."""
    model_path = tokenizer_dir / "spm.model"
    config_path = tokenizer_dir / "tokenizer_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Tokenizer config not found: {config_path}")

    # Load config for validation
    with open(config_path, 'r', encoding='utf-8') as f:
        tokenizer_config = json.load(f)

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    # Validate vocab size matches
    actual_vocab_size = sp.vocab_size()
    expected_vocab_size = tokenizer_config['vocab_size']

    if actual_vocab_size != expected_vocab_size:
        raise ValueError(
            f"Vocabulary size mismatch: tokenizer has {actual_vocab_size}, "
            f"config expects {expected_vocab_size}"
        )

    logging.info(f"Loaded tokenizer: vocab_size={actual_vocab_size}, model_type={tokenizer_config['model_type']}")

    return sp, tokenizer_config


def validate_compatibility(corpus_manifest: Dict, tokenizer_config: Dict) -> bool:
    """Validate that corpus and tokenizer are compatible."""
    corpus_hash = corpus_manifest.get('config_hash')
    tokenizer_corpus_hash = tokenizer_config.get('corpus_manifest_hash')

    if corpus_hash and tokenizer_corpus_hash:
        # Check if tokenizer was trained on same corpus version
        corpus_manifest_str = json.dumps(corpus_manifest, sort_keys=True, ensure_ascii=False)
        actual_corpus_hash = hashlib.sha256(corpus_manifest_str.encode('utf-8')).hexdigest()

        if actual_corpus_hash != tokenizer_corpus_hash:
            logging.warning(
                f"Tokenizer may not be compatible with this corpus version. "
                f"Tokenizer trained on corpus hash: {tokenizer_corpus_hash}, "
                f"Current corpus hash: {actual_corpus_hash}"
            )
            return False

    return True


def stream_text_from_corpus(corpus_manifest: Dict[str, Any], corpus_dir: Path) -> Iterator[str]:
    """Stream text content from corpus shards."""
    shards = corpus_manifest['shards']
    total_docs = sum(shard['num_documents'] for shard in shards)

    logging.info(f"Streaming text from {len(shards)} shards ({total_docs:,} documents)")

    for shard_info in shards:
        shard_path = corpus_dir / shard_info['path']

        # Handle both old format (direct path) and new format (shards/ subdir)
        if not shard_path.exists():
            shard_path = corpus_dir / "shards" / shard_info['path']


        try:
            with gzip.open(shard_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line.strip())
                        yield doc['text']
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logging.error(f"Error reading shard {shard_path}: {e}")
            raise


def tokenize_and_pack_sequences(
    text_iterator: Iterator[str],
    corpus_manifest: Dict[str, Any],
    tokenizer: spm.SentencePieceProcessor,
    sequence_length: int,
    output_dir: Path,
    train_val_split: float = 0.95,
    shuffle: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """Tokenize text and pack into fixed-length sequences using a memory-efficient streaming approach."""
    logging.info(f"Tokenizing and packing sequences (length={sequence_length}) with TRUE STREAMING")

    total_documents = 0
    total_tokens = 0
    temp_token_file = output_dir / "temp_tokens.bin"
    start_time = time.time()
    last_log_time = start_time

    # Get total expected documents for progress tracking
    total_expected_docs = corpus_manifest.get("statistics", {}).get("total_documents", 0)

    try:
        # Step 1: Stream all tokens to a temporary binary file
        with open(temp_token_file, 'wb') as f:
            with tqdm(total=total_expected_docs, unit="doc", desc="🔬 Tokenizing", unit_scale=True) as pbar:
                try:
                    for text in text_iterator:
                        try:
                            tokens = tokenizer.encode(text)
                            if tokens:
                                token_count = len(tokens)
                                f.write(np.array(tokens, dtype=np.uint16).tobytes())
                                total_tokens += token_count
                                total_documents += 1
                                pbar.update(1)

                                # Log progress periodically
                                current_time = time.time()
                                if current_time - last_log_time > 10:
                                    elapsed_time = current_time - start_time
                                    avg_speed = total_tokens / elapsed_time
                                    doc_progress = total_documents / total_expected_docs if total_expected_docs > 0 else 0
                                    logging.info(f"   Progress: {doc_progress:.1%}, Docs: {total_documents:,}, Tokens: {total_tokens:,}, Speed: {avg_speed:,.0f} tokens/s")
                                    last_log_time = current_time

                        except Exception as e:
                            logging.warning(f"Error tokenizing document: {e}")
                            continue
                finally:
                    pbar.close() # Ensure pbar closes even on error

        end_time = time.time()
        duration = end_time - start_time
        avg_speed = total_tokens / duration if duration > 0 else 0
        temp_file_size_mb = temp_token_file.stat().st_size / (1024 * 1024)

        logging.info("✅ Tokenization phase complete.")
        logging.info(f"   - ⏱️  Duration: {duration:.2f} seconds")
        logging.info(f"   - ⚡ Average Speed: {avg_speed:,.0f} tokens/s")
        logging.info(f"   - 📄 Documents Processed: {total_documents:,}")
        logging.info(f"   - 🎯 Total Tokens Generated: {total_tokens:,}")
        logging.info(f"   - 📊 Tokens per Document (avg): {total_tokens / total_documents:.1f}" if total_documents > 0 else "N/A")
        logging.info(f"   - 💾 Temp File Size: {temp_file_size_mb:.2f} MB")

        if total_tokens < sequence_length:
            raise ValueError(f"Not enough tokens ({total_tokens}) to create sequences of length {sequence_length}")

        # Step 2: Use memmap to handle the large token buffer without loading it into RAM
        tokens_array = np.memmap(str(temp_token_file), dtype=np.uint16, mode='r', shape=(total_tokens,))

        # Step 3: Reshape into sequences, trimming excess tokens
        num_sequences = total_tokens // sequence_length
        total_tokens_used = num_sequences * sequence_length
        sequences = tokens_array[:total_tokens_used].reshape(num_sequences, sequence_length)

        logging.info(f"Created {num_sequences:,} sequences of length {sequence_length:,}")
        logging.info(f"Using {total_tokens_used:,} tokens ({total_tokens - total_tokens_used:,} discarded)")

        # Step 4: Shuffle sequences if requested (for better training distribution)
        # Note: We generate shuffled indices but apply them during write to avoid RAM copy
        if shuffle:
            logging.info(f"Generating shuffle indices for {num_sequences:,} sequences with seed={seed}")
            np.random.seed(seed)
            indices = np.arange(num_sequences)
            np.random.shuffle(indices)
        else:
            logging.info("Skipping shuffle (deterministic ordering)")
            indices = np.arange(num_sequences)

        # Step 5: Split into train/validation
        train_size = int(num_sequences * train_val_split)
        val_size = num_sequences - train_size
        logging.info(f"Split: {train_size:,} train sequences, {val_size:,} validation sequences")

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Step 6: Write to final binary files using chunked streaming (memory efficient)
        train_path = output_dir / "train.bin"
        val_path = output_dir / "val.bin"

        chunk_size = 10000  # Write 10k sequences at a time to limit RAM usage

        logging.info(f"Writing train.bin ({train_size:,} sequences) in chunks of {chunk_size:,}...")
        with open(train_path, 'wb') as f:
            for i in range(0, len(train_indices), chunk_size):
                chunk_indices = train_indices[i:i+chunk_size]
                chunk_data = sequences[chunk_indices]
                f.write(chunk_data.tobytes())

        logging.info(f"Writing val.bin ({val_size:,} sequences) in chunks of {chunk_size:,}...")
        with open(val_path, 'wb') as f:
            for i in range(0, len(val_indices), chunk_size):
                chunk_indices = val_indices[i:i+chunk_size]
                chunk_data = sequences[chunk_indices]
                f.write(chunk_data.tobytes())

        # Step 7: Create index files
        _create_index_files(output_dir, train_size, val_size, sequence_length)

    finally:
        # Clean up the temporary file
        if temp_token_file.exists():
            temp_token_file.unlink()

    stats = {
        'total_documents': total_documents,
        'corpus_total_tokens': total_tokens,
        'total_tokens_used': total_tokens_used,
        'sequence_length': sequence_length,
        'num_sequences': num_sequences,
        'train_sequences': train_size,
        'val_sequences': val_size,
        'train_val_split': train_val_split,
        'vocab_size': tokenizer.vocab_size()
    }
    return stats


def _create_index_files(output_dir: Path, train_size: int, val_size: int, sequence_length: int) -> None:
    """Create index files for train and validation data."""

    # Training index
    if train_size > 0:
        train_idx_path = output_dir / "train.idx"
        train_info = {
            'shape': [train_size, sequence_length],
            'dtype': 'uint16',
            'sequence_length': sequence_length,
            'num_sequences': train_size
        }
        with open(train_idx_path, 'w') as f:
            json.dump(train_info, f)

    # Validation index
    if val_size > 0:
        val_idx_path = output_dir / "val.idx"
        val_info = {
            'shape': [val_size, sequence_length],
            'dtype': 'uint16',
            'sequence_length': sequence_length,
            'num_sequences': val_size
        }
        with open(val_idx_path, 'w') as f:
            json.dump(val_info, f)


def create_final_manifest(
    output_dir: Path,
    stats: Dict[str, Any],
    corpus_manifest_hash: str,
    tokenizer_config_hash: str,
    sequence_length: int
) -> None:
    """Create final manifest for training data."""
    manifest = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'pipeline_version': '2.1.0',
        'data_format': 'packed_sequences',
        'sequence_length': sequence_length,
        'corpus_manifest_hash': corpus_manifest_hash,
        'tokenizer_config_hash': tokenizer_config_hash,
        'statistics': stats,
        'files': {
            'train_data': 'train.bin',
            'train_index': 'train.idx',
            'val_data': 'val.bin' if stats['val_sequences'] > 0 else None,
            'val_index': 'val.idx' if stats['val_sequences'] > 0 else None
        },
        'data_loading_info': {
            'dtype': 'uint16',
            'memory_mapped': True,
            'format': 'numpy_memmap'
        }
    }

    manifest_path = output_dir / "final_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logging.info(f"Final manifest saved to: {manifest_path}")


def test_data_loading(output_dir: Path) -> None:
    """Test loading the packed data."""
    logging.info("Testing data loading...")

    train_path = output_dir / "train.bin"
    train_idx_path = output_dir / "train.idx"

    if not train_path.exists() or not train_idx_path.exists():
        logging.warning("Training data not found, skipping test")
        return

    try:
        # Load index
        with open(train_idx_path, 'r') as f:
            train_info = json.load(f)

        # Load data using memmap
        train_data = np.memmap(
            str(train_path),
            dtype=np.uint16,
            mode='r',
            shape=tuple(train_info['shape'])
        )

        logging.info(f"Successfully loaded training data: shape={train_data.shape}")

        # Show sample
        if len(train_data) > 0:
            sample_sequence = train_data[0]
            logging.info(f"Sample sequence (first 10 tokens): {sample_sequence[:10].tolist()}")

        # Test validation data if exists
        val_path = output_dir / "val.bin"
        val_idx_path = output_dir / "val.idx"

        if val_path.exists() and val_idx_path.exists():
            with open(val_idx_path, 'r') as f:
                val_info = json.load(f)

            val_data = np.memmap(
                str(val_path),
                dtype=np.uint16,
                mode='r',
                shape=tuple(val_info['shape'])
            )

            logging.info(f"Successfully loaded validation data: shape={val_data.shape}")

    except Exception as e:
        logging.error(f"Data loading test failed: {e}")
        raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Pack corpus into training sequences")
    parser.add_argument("--config", required=True, help="Path to packing config JSON (e.g., config/pretrain/packing/default.json)")
    parser.add_argument("--corpus-dir", required=True, help="Directory containing corpus (with manifest.json)")
    parser.add_argument("--tokenizer-dir", required=True, help="Directory containing trained tokenizer")
    parser.add_argument("--output-dir", required=True, help="Output directory for packed sequences")
    parser.add_argument("--force", action="store_true", help="Force re-packing even if data exists")
    parser.add_argument("--skip-tokenizer-check", action="store_true",
                        help="Skip tokenizer/corpus compatibility check (use when tokenizer is trained on separate corpus)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Load packing config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Packing config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        packing_config = json.load(f)

    # Extract parameters from config
    packing_params = packing_config['packing_params']
    sequence_length = packing_params['sequence_length']
    train_val_split = packing_params['train_val_split']
    shuffle = packing_params.get('shuffle', True)
    seed = packing_params.get('seed', 42)

    # Note: packing_strategy, shard_size, num_workers, buffer_size are reserved for future use
    # Current implementation uses greedy packing with memory-mapped streaming

    setup_logging(args.log_level)

    # Validate inputs
    corpus_dir = Path(args.corpus_dir)
    manifest_path = corpus_dir / "manifest.json"
    tokenizer_dir = Path(args.tokenizer_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    final_manifest_path = output_dir / "final_manifest.json"
    if final_manifest_path.exists() and not args.force:
        logging.info(f"Packed data already exists at {output_dir}. Use --force to re-pack.")
        return

    # Load corpus manifest
    corpus_manifest = load_manifest(manifest_path)

    # Load tokenizer
    tokenizer, tokenizer_config = load_tokenizer(tokenizer_dir)

    # Validate compatibility (blocking unless skipped)
    if not validate_compatibility(corpus_manifest, tokenizer_config):
        if args.skip_tokenizer_check:
            logging.warning(
                "Tokenizer/corpus compatibility check SKIPPED via --skip-tokenizer-check. "
                "Proceeding with universal tokenizer."
            )
        else:
            raise ValueError(
                "Corpus and tokenizer are incompatible. This could lead to poor tokenization quality. "
                "Please retrain the tokenizer on the current corpus or use the correct corpus. "
                "If using a universal tokenizer trained on a separate corpus, add --skip-tokenizer-check"
            )

    # Create hashes for traceability
    corpus_manifest_str = json.dumps(corpus_manifest, sort_keys=True, ensure_ascii=False)
    corpus_manifest_hash = hashlib.sha256(corpus_manifest_str.encode('utf-8')).hexdigest()

    tokenizer_config_str = json.dumps(tokenizer_config, sort_keys=True, ensure_ascii=False)
    tokenizer_config_hash = hashlib.sha256(tokenizer_config_str.encode('utf-8')).hexdigest()

    # Stream and tokenize corpus
    text_iterator = stream_text_from_corpus(corpus_manifest, corpus_dir)

    # Pack sequences
    stats = tokenize_and_pack_sequences(
        text_iterator=text_iterator,
        corpus_manifest=corpus_manifest,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        output_dir=output_dir,
        train_val_split=train_val_split,
        shuffle=shuffle,
        seed=seed
    )

    # Create final manifest
    create_final_manifest(
        output_dir=output_dir,
        stats=stats,
        corpus_manifest_hash=corpus_manifest_hash,
        tokenizer_config_hash=tokenizer_config_hash,
        sequence_length=sequence_length
    )

    # Test data loading
    test_data_loading(output_dir)

    logging.info(f"Sequence packing complete! Output saved to: {output_dir}")
    logging.info(f"Training sequences: {stats['train_sequences']:,}")
    logging.info(f"Validation sequences: {stats['val_sequences']:,}")
    logging.info(f"Total tokens used: {stats['total_tokens_used']:,}")
    logging.info(f"Sequence length: {stats['sequence_length']:,}")


if __name__ == "__main__":
    main()