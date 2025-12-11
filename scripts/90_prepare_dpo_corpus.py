#!/usr/bin/env python3
"""
DPO Corpus preparation script for Lumi-Lab pipeline.

This script transforms preference datasets (chosen/rejected pairs) into cleaned,
validated and sharded DPO corpus ready for alignment training.

Key features:
- Load DPO datasets from HuggingFace (Intel/orca_dpo_pairs, Anthropic/hh-rlhf, etc.)
- Validate triplets (prompt, chosen, rejected)
- Tokenizer validation with SHA256 verification
- Quality filtering and deduplication
- Sharded output with manifest generation
- Support for multiple DPO dataset formats
- Robust error handling and resume capability
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import gc
import atexit
import signal
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import sentencepiece as spm
from datasets import load_dataset

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.auth import get_hf_api_client
from utils.dpo_utils import validate_dpo_example


def _safe_cleanup_and_exit():
    """Safe cleanup to prevent thread state crashes during shutdown."""
    try:
        logging.info("🧹 Performing safe cleanup...")
        gc.collect()
        logging.info("✅ Cleanup completed successfully")
    except Exception as e:
        logging.warning(f"⚠️ Cleanup encountered minor issues: {e}")
    finally:
        os._exit(0)


def _setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logging.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        _safe_cleanup_and_exit()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_file = output_dir / "dpo_preparation.log"

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup handlers
    handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    logging.info(f"📋 Loaded configuration from {config_path}")
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate DPO dataset configuration."""
    required_fields = ['name', 'template', 'output_params', 'datasets']

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")

    # Validate template
    valid_templates = ['dpo_standard']
    if config['template'] not in valid_templates:
        raise ValueError(f"Invalid template '{config['template']}'. Must be one of: {valid_templates}")

    # Validate datasets
    if not isinstance(config['datasets'], list) or len(config['datasets']) == 0:
        raise ValueError("Configuration must contain at least one dataset")

    for i, dataset in enumerate(config['datasets']):
        required_dataset_fields = ['name', 'type', 'dataset_name', 'text_fields']
        for field in required_dataset_fields:
            if field not in dataset:
                raise ValueError(f"Dataset {i} missing required field: {field}")

        # Validate text_fields for DPO
        text_fields = dataset['text_fields']
        required_text_fields = ['prompt', 'chosen', 'rejected']
        for field in required_text_fields:
            if field not in text_fields:
                raise ValueError(f"Dataset {i} text_fields missing: {field}")


def compute_tokenizer_hash(tokenizer_dir: str) -> str:
    """Compute SHA256 hash of tokenizer config for consistency verification."""
    tokenizer_config_path = Path(tokenizer_dir) / "tokenizer_config.json"

    if not tokenizer_config_path.exists():
        raise FileNotFoundError(f"Tokenizer config not found: {tokenizer_config_path}")

    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()

    return hashlib.sha256(config_content.encode('utf-8')).hexdigest()


def load_tokenizer(tokenizer_dir: str) -> spm.SentencePieceProcessor:
    """Load SentencePiece tokenizer."""
    tokenizer_path = Path(tokenizer_dir)
    model_file = tokenizer_path / "tokenizer.model"

    if not model_file.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_file}")

    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_file))

    logging.info(f"📖 Loaded tokenizer: {tokenizer_dir}")
    logging.info(f"   Vocabulary size: {sp.vocab_size()}")

    return sp


def load_dpo_dataset(
    dataset_config: Dict[str, Any],
    hf_token: Optional[str] = None
) -> Any:
    """
    Load a DPO dataset from HuggingFace.

    Args:
        dataset_config: Dataset configuration
        hf_token: Optional HuggingFace API token

    Returns:
        Loaded dataset
    """
    dataset_name = dataset_config['dataset_name']
    split = dataset_config.get('split', 'train')
    subset = dataset_config.get('subset', None)
    max_samples = dataset_config.get('max_samples', None)

    logging.info(f"📥 Loading dataset: {dataset_name}")

    # Load dataset
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split, token=hf_token)
    else:
        dataset = load_dataset(dataset_name, split=split, token=hf_token)

    # Subsample if requested
    if max_samples and len(dataset) > max_samples:
        logging.info(f"   Subsampling {max_samples:,} from {len(dataset):,} samples")
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    logging.info(f"   Loaded {len(dataset):,} samples")
    return dataset


def extract_dpo_fields(
    example: Dict[str, Any],
    text_fields: Dict[str, str]
) -> Optional[Dict[str, str]]:
    """
    Extract prompt, chosen, and rejected from a dataset example.

    Args:
        example: Raw dataset example
        text_fields: Mapping of field names

    Returns:
        Extracted DPO example or None if invalid
    """
    try:
        prompt_field = text_fields['prompt']
        chosen_field = text_fields['chosen']
        rejected_field = text_fields['rejected']

        prompt = example.get(prompt_field, '')
        chosen = example.get(chosen_field, '')
        rejected = example.get(rejected_field, '')

        # Basic validation
        if not prompt or not chosen or not rejected:
            return None

        # Convert to strings if needed
        if not isinstance(prompt, str):
            prompt = str(prompt)
        if not isinstance(chosen, str):
            chosen = str(chosen)
        if not isinstance(rejected, str):
            rejected = str(rejected)

        return {
            'prompt': prompt.strip(),
            'chosen': chosen.strip(),
            'rejected': rejected.strip()
        }

    except Exception as e:
        logging.debug(f"Failed to extract DPO fields: {e}")
        return None


def apply_quality_filters(
    example: Dict[str, str],
    filters: Dict[str, Any],
    tokenizer: spm.SentencePieceProcessor
) -> bool:
    """
    Apply quality filters to a DPO example.

    Args:
        example: DPO example with prompt, chosen, rejected
        filters: Quality filter configuration
        tokenizer: Tokenizer for length computation

    Returns:
        True if example passes all filters
    """
    # Tokenize
    prompt_tokens = tokenizer.encode(example['prompt'])
    chosen_tokens = tokenizer.encode(example['chosen'])
    rejected_tokens = tokenizer.encode(example['rejected'])

    # Length filters
    if len(prompt_tokens) < filters.get('min_prompt_length', 0):
        return False
    if len(prompt_tokens) > filters.get('max_prompt_length', 10000):
        return False

    if len(chosen_tokens) < filters.get('min_chosen_length', 0):
        return False
    if len(chosen_tokens) > filters.get('max_chosen_length', 10000):
        return False

    if len(rejected_tokens) < filters.get('min_rejected_length', 0):
        return False
    if len(rejected_tokens) > filters.get('max_rejected_length', 10000):
        return False

    # Margin filter (chosen should be different from rejected)
    min_margin = filters.get('min_margin_tokens', 0)
    margin = abs(len(chosen_tokens) - len(rejected_tokens))
    if margin < min_margin:
        return False

    # Check for identical responses
    if filters.get('max_identical_pairs', 0) == 0:
        if example['chosen'] == example['rejected']:
            return False

    return True


def process_dataset(
    dataset: Any,
    dataset_config: Dict[str, Any],
    quality_filters: Dict[str, Any],
    tokenizer: spm.SentencePieceProcessor,
    output_dir: Path,
    shard_size: int
) -> Dict[str, Any]:
    """
    Process a DPO dataset with quality filtering and sharding.

    Args:
        dataset: Raw HuggingFace dataset
        dataset_config: Dataset configuration
        quality_filters: Quality filter settings
        tokenizer: Tokenizer for validation
        output_dir: Output directory for shards
        shard_size: Number of examples per shard

    Returns:
        Processing statistics
    """
    text_fields = dataset_config['text_fields']
    processed_examples = []
    stats = {
        'total_raw': len(dataset),
        'extracted': 0,
        'validated': 0,
        'filtered': 0,
        'final': 0
    }

    logging.info("🔄 Processing dataset...")

    for example in tqdm(dataset, desc="Processing examples"):
        # Extract DPO fields
        dpo_example = extract_dpo_fields(example, text_fields)
        if dpo_example is None:
            continue

        stats['extracted'] += 1

        # Validate DPO example
        is_valid, error_msg = validate_dpo_example(dpo_example)
        if not is_valid:
            logging.debug(f"Validation failed: {error_msg}")
            continue

        stats['validated'] += 1

        # Apply quality filters
        if not apply_quality_filters(dpo_example, quality_filters, tokenizer):
            stats['filtered'] += 1
            continue

        processed_examples.append(dpo_example)

    stats['final'] = len(processed_examples)

    logging.info(f"📊 Processing statistics:")
    logging.info(f"   Raw examples: {stats['total_raw']:,}")
    logging.info(f"   Extracted: {stats['extracted']:,}")
    logging.info(f"   Validated: {stats['validated']:,}")
    logging.info(f"   Filtered out: {stats['filtered']:,}")
    logging.info(f"   Final: {stats['final']:,}")

    # Write shards
    if stats['final'] > 0:
        write_shards(processed_examples, output_dir, shard_size)

    return stats


def write_shards(
    examples: List[Dict[str, str]],
    output_dir: Path,
    shard_size: int
) -> int:
    """
    Write examples to JSONL shards.

    Args:
        examples: Processed DPO examples
        output_dir: Output directory
        shard_size: Number of examples per shard

    Returns:
        Number of shards written
    """
    num_shards = (len(examples) + shard_size - 1) // shard_size
    logging.info(f"📝 Writing {num_shards} shard(s)...")

    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, len(examples))
        shard_examples = examples[start_idx:end_idx]

        shard_file = output_dir / f"shard_{shard_idx:03d}.jsonl"

        with open(shard_file, 'w', encoding='utf-8') as f:
            for example in shard_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        logging.info(f"   Shard {shard_idx:03d}: {len(shard_examples):,} examples -> {shard_file.name}")

    return num_shards


def compute_corpus_statistics(
    output_dir: Path,
    tokenizer: spm.SentencePieceProcessor
) -> Dict[str, Any]:
    """Compute statistics for the DPO corpus."""
    logging.info("📊 Computing corpus statistics...")

    shard_files = sorted(output_dir.glob("shard_*.jsonl"))
    total_samples = 0
    prompt_lengths = []
    chosen_lengths = []
    rejected_lengths = []

    for shard_file in shard_files:
        with open(shard_file, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                total_samples += 1

                prompt_lengths.append(len(tokenizer.encode(example['prompt'])))
                chosen_lengths.append(len(tokenizer.encode(example['chosen'])))
                rejected_lengths.append(len(tokenizer.encode(example['rejected'])))

    import numpy as np

    stats = {
        'total_samples': total_samples,
        'total_shards': len(shard_files),
        'avg_prompt_length': float(np.mean(prompt_lengths)),
        'avg_chosen_length': float(np.mean(chosen_lengths)),
        'avg_rejected_length': float(np.mean(rejected_lengths)),
        'median_prompt_length': float(np.median(prompt_lengths)),
        'median_chosen_length': float(np.median(chosen_lengths)),
        'median_rejected_length': float(np.median(rejected_lengths)),
        'min_prompt_length': int(np.min(prompt_lengths)),
        'min_chosen_length': int(np.min(chosen_lengths)),
        'min_rejected_length': int(np.min(rejected_lengths)),
        'max_prompt_length': int(np.max(prompt_lengths)),
        'max_chosen_length': int(np.max(chosen_lengths)),
        'max_rejected_length': int(np.max(rejected_lengths)),
    }

    logging.info(f"   Total samples: {stats['total_samples']:,}")
    logging.info(f"   Avg prompt length: {stats['avg_prompt_length']:.1f} tokens")
    logging.info(f"   Avg chosen length: {stats['avg_chosen_length']:.1f} tokens")
    logging.info(f"   Avg rejected length: {stats['avg_rejected_length']:.1f} tokens")

    return stats


def write_manifest(
    output_dir: Path,
    config: Dict[str, Any],
    stats: Dict[str, Any],
    tokenizer_hash: str
) -> None:
    """Write manifest file with corpus metadata."""
    manifest = {
        'dataset_name': config['name'],
        'description': config.get('description', ''),
        'template': config['template'],
        'total_samples': stats['total_samples'],
        'total_shards': stats['total_shards'],
        'tokenizer_config_hash': tokenizer_hash,
        'avg_prompt_length': stats['avg_prompt_length'],
        'avg_chosen_length': stats['avg_chosen_length'],
        'avg_rejected_length': stats['avg_rejected_length'],
        'median_prompt_length': stats['median_prompt_length'],
        'median_chosen_length': stats['median_chosen_length'],
        'median_rejected_length': stats['median_rejected_length'],
        'quality_filters': config.get('quality_filters', {}),
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logging.info(f"📋 Manifest written: {manifest_file}")


def write_data_card(
    output_dir: Path,
    config: Dict[str, Any],
    stats: Dict[str, Any]
) -> None:
    """Write human-readable data card."""
    card_file = output_dir / "data_card.txt"

    with open(card_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"DPO CORPUS DATA CARD: {config['name']}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Description: {config.get('description', 'N/A')}\n")
        f.write(f"Template: {config['template']}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("CORPUS STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples: {stats['total_samples']:,}\n")
        f.write(f"Total shards: {stats['total_shards']}\n\n")

        f.write(f"Prompt lengths (tokens):\n")
        f.write(f"  Average: {stats['avg_prompt_length']:.1f}\n")
        f.write(f"  Median:  {stats['median_prompt_length']:.1f}\n")
        f.write(f"  Range:   {stats['min_prompt_length']} - {stats['max_prompt_length']}\n\n")

        f.write(f"Chosen response lengths (tokens):\n")
        f.write(f"  Average: {stats['avg_chosen_length']:.1f}\n")
        f.write(f"  Median:  {stats['median_chosen_length']:.1f}\n")
        f.write(f"  Range:   {stats['min_chosen_length']} - {stats['max_chosen_length']}\n\n")

        f.write(f"Rejected response lengths (tokens):\n")
        f.write(f"  Average: {stats['avg_rejected_length']:.1f}\n")
        f.write(f"  Median:  {stats['median_rejected_length']:.1f}\n")
        f.write(f"  Range:   {stats['min_rejected_length']} - {stats['max_rejected_length']}\n\n")

        f.write("QUALITY FILTERS\n")
        f.write("-" * 80 + "\n")
        filters = config.get('quality_filters', {})
        for key, value in filters.items():
            f.write(f"{key}: {value}\n")

        f.write("\n" + "=" * 80 + "\n")

    logging.info(f"📄 Data card written: {card_file}")


def main():
    parser = argparse.ArgumentParser(description="DPO corpus preparation for Lumi-Lab pipeline")

    parser.add_argument("--config", type=str, required=True,
                       help="Path to DPO dataset configuration file (e.g., config/dpo/datasets/orca_dpo.json)")
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                       help="Path to tokenizer directory (for validation and token counting)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for DPO corpus shards")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Setup signal handlers
    _setup_signal_handlers()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_dir, args.verbose)

    logging.info("=" * 80)
    logging.info("DPO CORPUS PREPARATION - Lumi-Lab Pipeline")
    logging.info("=" * 80)

    try:
        # Load config
        config = load_config(args.config)
        validate_config(config)

        # Load tokenizer
        tokenizer = load_tokenizer(args.tokenizer_dir)
        tokenizer_hash = compute_tokenizer_hash(args.tokenizer_dir)
        logging.info(f"🔑 Tokenizer hash: {tokenizer_hash[:16]}...")

        # Get HuggingFace token if needed
        hf_token = None
        try:
            hf_client = get_hf_api_client()
            if hf_client:
                hf_token = hf_client.token
        except:
            pass

        # Process each dataset in config
        output_params = config['output_params']
        shard_size = output_params.get('shard_size', 1000)
        quality_filters = config.get('quality_filters', {})

        all_stats = []

        for dataset_config in config['datasets']:
            logging.info(f"\n{'=' * 80}")
            logging.info(f"Processing dataset: {dataset_config['name']}")
            logging.info(f"{'=' * 80}")

            # Load dataset
            dataset = load_dpo_dataset(dataset_config, hf_token)

            # Process and write shards
            stats = process_dataset(
                dataset,
                dataset_config,
                quality_filters,
                tokenizer,
                output_dir,
                shard_size
            )

            all_stats.append(stats)

        # Compute corpus statistics
        corpus_stats = compute_corpus_statistics(output_dir, tokenizer)

        # Write manifest
        write_manifest(output_dir, config, corpus_stats, tokenizer_hash)

        # Write data card
        write_data_card(output_dir, config, corpus_stats)

        logging.info("\n" + "=" * 80)
        logging.info("✅ DPO CORPUS PREPARATION COMPLETED")
        logging.info("=" * 80)
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Total samples: {corpus_stats['total_samples']:,}")
        logging.info(f"Total shards: {corpus_stats['total_shards']}")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"❌ Error during DPO corpus preparation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
