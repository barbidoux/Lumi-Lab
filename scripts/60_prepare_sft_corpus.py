#!/usr/bin/env python3
"""
SFT Corpus preparation script for Lumi-Lab pipeline.

This script transforms conversational datasets into cleaned, validated and sharded
SFT corpus ready for supervised fine-tuning. It mirrors the robustness and scalability
of the main corpus preparation pipeline (01_prepare_corpus.py).

Key features:
- Template-based conversation formatting (ChatML, Instruct, Chat)
- Tokenizer validation with SHA256 verification
- Quality filtering and validation
- Sharded output with manifest generation
- Support for multiple conversation formats
- Robust error handling and resume capability
"""

import argparse
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Union, Optional
import yaml
from tqdm import tqdm
import sentencepiece as spm

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.auth import get_hf_api_client
from utils.sft_templates import ConversationTemplateProcessor
from utils.dataset_utils import estimate_avg_tokens_per_sample, plan_samples_for_token_budget


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


def setup_logging(output_dir: Path, verbose: bool = False, log_level_str: Optional[str] = None) -> None:
    """Setup logging configuration."""
    # Priority: --log-level > --verbose > default (INFO)
    if log_level_str:
        log_level = getattr(logging, log_level_str.upper())
    else:
        log_level = logging.DEBUG if verbose else logging.INFO

    log_file = output_dir / "sft_preparation.log"

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
    """Load configuration from JSON or YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    logging.info(f"📋 Loaded configuration from {config_path}")
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate SFT dataset configuration."""
    required_fields = ['name', 'template', 'output_params', 'datasets']

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")

    # Validate template
    valid_templates = ['chatml', 'instruct', 'chat', 'alpaca']
    if config['template'] not in valid_templates:
        raise ValueError(f"Invalid template '{config['template']}'. Must be one of: {valid_templates}")

    # Validate sampling_strategy (if present)
    if 'sampling_strategy' in config:
        sampling_strategy = config['sampling_strategy']
        mode = sampling_strategy.get('mode', 'max_samples')

        valid_modes = ['target_tokens', 'target_conversations', 'max_samples']
        if mode not in valid_modes:
            raise ValueError(f"Invalid sampling mode '{mode}'. Must be one of: {valid_modes}")

        # Validate mode-specific fields
        if mode == 'target_tokens':
            if 'target_tokens' not in sampling_strategy or sampling_strategy['target_tokens'] is None:
                raise ValueError("sampling_strategy.mode='target_tokens' requires 'target_tokens' field")
        elif mode == 'target_conversations':
            if 'target_conversations' not in sampling_strategy or sampling_strategy['target_conversations'] is None:
                raise ValueError("sampling_strategy.mode='target_conversations' requires 'target_conversations' field")

    # Validate datasets
    if not isinstance(config['datasets'], list) or len(config['datasets']) == 0:
        raise ValueError("Configuration must contain at least one dataset")

    for i, dataset in enumerate(config['datasets']):
        required_dataset_fields = ['name', 'type', 'dataset_name', 'text_fields']
        for field in required_dataset_fields:
            if field not in dataset:
                raise ValueError(f"Dataset {i} missing required field: {field}")


def normalize_tokenizer_path(tokenizer_path: str) -> str:
    """
    Normalize tokenizer path to point to the .model file.

    Args:
        tokenizer_path: Either directory containing tokenizer or direct path to .model file

    Returns:
        Path to the .model file
    """
    tokenizer_path = Path(tokenizer_path)

    if tokenizer_path.is_file() and tokenizer_path.suffix == '.model':
        # Direct path to .model file
        return str(tokenizer_path)
    elif tokenizer_path.is_dir():
        # Directory containing tokenizer - look for .model file
        model_files = list(tokenizer_path.glob("*.model"))
        if len(model_files) == 1:
            return str(model_files[0])
        elif len(model_files) > 1:
            # Multiple .model files, prefer spm.model or similar
            for model_file in model_files:
                if model_file.name in ['spm.model', 'tokenizer.model', 'spm32k.model']:
                    return str(model_file)
            # Fallback to first one found
            return str(model_files[0])
        else:
            raise FileNotFoundError(f"No .model files found in directory: {tokenizer_path}")
    else:
        # Try to auto-detect
        if tokenizer_path.exists():
            if tokenizer_path.is_file():
                return str(tokenizer_path)  # Assume it's the model file
            else:
                raise ValueError(f"Path exists but is neither file nor directory: {tokenizer_path}")
        else:
            raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")


def verify_tokenizer(tokenizer_path: str) -> Dict[str, Any]:
    """Verify tokenizer and return metadata."""
    # Normalize the tokenizer path
    model_path = normalize_tokenizer_path(tokenizer_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer model file not found: {model_path}")

    # Load tokenizer
    import sentencepiece as spm
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(model_path)

    # Calculate SHA256
    with open(model_path, 'rb') as f:
        tokenizer_hash = hashlib.sha256(f.read()).hexdigest()

    # Get metadata
    metadata = {
        'tokenizer_path': model_path,
        'tokenizer_sha256': tokenizer_hash,
        'vocab_size': sp_model.vocab_size(),
        'eos_token': sp_model.eos_id(),
        'bos_token': sp_model.bos_id(),
        'unk_token': sp_model.unk_id(),
        'pad_token': sp_model.pad_id() if hasattr(sp_model, 'pad_id') else sp_model.unk_id()
    }

    logging.info(f"🔐 Tokenizer verified: {tokenizer_hash[:12]}...")
    logging.info(f"📊 Vocab size: {metadata['vocab_size']:,}")

    return metadata


def load_dataset_source(dataset_config: Dict[str, Any], hf_client) -> List[Dict[str, Any]]:
    """Load dataset from various sources (HuggingFace, local files, etc.)."""
    dataset_type = dataset_config['type'].lower()
    dataset_name = dataset_config['dataset_name']

    logging.info(f"📂 Loading dataset: {dataset_name} (type: {dataset_type})")

    if dataset_type == 'huggingface':
        try:
            from datasets import load_dataset

            # Load dataset with optional parameters
            load_params = {
                'path': dataset_name,
                'split': dataset_config.get('split', 'train'),
                'trust_remote_code': dataset_config.get('trust_remote_code', False)
            }

            # Add subset if specified
            if 'subset' in dataset_config and dataset_config['subset']:
                load_params['name'] = dataset_config['subset']

            # Add streaming if specified
            if dataset_config.get('streaming', False):
                load_params['streaming'] = True

            dataset = load_dataset(**load_params)

            # Convert to list if not streaming
            if not dataset_config.get('streaming', False):
                data = list(dataset)
            else:
                # For streaming, take limited samples
                max_samples = dataset_config.get('max_samples', 10000)
                data = []
                for i, sample in enumerate(dataset):
                    if i >= max_samples:
                        break
                    data.append(sample)

            logging.info(f"✅ Loaded {len(data):,} samples from HuggingFace")
            return data

        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace dataset '{dataset_name}': {e}")

    elif dataset_type == 'jsonl':
        # Load JSONL file
        data = []
        with open(dataset_name, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logging.warning(f"⚠️ Skipping invalid JSON on line {line_num}: {e}")

        logging.info(f"✅ Loaded {len(data):,} samples from JSONL file")
        return data

    elif dataset_type == 'json':
        # Load JSON file
        with open(dataset_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"JSON file must contain a list of conversations")

        logging.info(f"✅ Loaded {len(data):,} samples from JSON file")
        return data

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def extract_conversation(sample: Dict[str, Any], text_fields: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Extract conversation from a dataset sample based on text field mappings."""
    try:
        conversation = {}

        # Extract required fields
        for field_type, field_path in text_fields.items():
            value = sample

            # Navigate nested fields (e.g., "messages.0.content")
            for key in field_path.split('.'):
                if key.isdigit():
                    value = value[int(key)]
                else:
                    value = value[key]

            conversation[field_type] = str(value).strip()

        # Validate conversation has required fields
        if 'prompt' not in conversation:
            return None

        # If no response field specified, try common alternatives
        if 'response' not in conversation:
            for alt_field in ['answer', 'completion', 'output', 'assistant']:
                if alt_field in conversation:
                    conversation['response'] = conversation[alt_field]
                    break

        # Ensure we have both prompt and response
        if not conversation.get('prompt') or not conversation.get('response'):
            return None

        return conversation

    except (KeyError, IndexError, TypeError, ValueError):
        return None


def apply_quality_filters(conversation: Dict[str, str], filters: Dict[str, Any]) -> bool:
    """Apply quality filters to conversation."""
    prompt = conversation['prompt']
    response = conversation['response']

    # Length filters
    if 'min_prompt_length' in filters:
        if len(prompt) < filters['min_prompt_length']:
            return False

    if 'max_prompt_length' in filters:
        if len(prompt) > filters['max_prompt_length']:
            return False

    if 'min_response_length' in filters:
        if len(response) < filters['min_response_length']:
            return False

    if 'max_response_length' in filters:
        if len(response) > filters['max_response_length']:
            return False

    # Content filters
    if filters.get('filter_urls', True):
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        if re.search(url_pattern, prompt) or re.search(url_pattern, response):
            return False

    if filters.get('filter_code_blocks', True):
        code_pattern = r'```|`{3,}'
        if re.search(code_pattern, prompt) or re.search(code_pattern, response):
            return False

    # Language filter (basic)
    if filters.get('require_ascii', False):
        try:
            prompt.encode('ascii')
            response.encode('ascii')
        except UnicodeEncodeError:
            return False

    return True


def create_stream_factory_sft(dataset_config: Dict[str, Any], hf_client):
    """Create a stream factory for SFT dataset analysis."""
    dataset_type = dataset_config['type']

    if dataset_type == 'huggingface':
        def stream_factory():
            from datasets import load_dataset

            dataset_name = dataset_config['dataset_name']
            subset = dataset_config.get('subset')
            split = dataset_config.get('split', 'train')

            dataset = load_dataset(
                dataset_name,
                subset,
                split=split,
                streaming=True,
                trust_remote_code=dataset_config.get('trust_remote_code', False)
            )

            return iter(dataset)

    elif dataset_type == 'local':
        def stream_factory():
            import json
            data_path = Path(dataset_config['data_path'])

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
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return stream_factory


def plan_sampling_strategy(config: Dict[str, Any], tokenizer_path: str) -> Dict[str, Any]:
    """
    Plan sampling for each dataset based on sampling_strategy.

    Supports 3 modes:
    - target_tokens: Plan samples to reach target token budget (like pretrain)
    - target_conversations: Plan samples to reach target conversation count
    - max_samples: Use max_samples from each dataset config (legacy mode)

    Returns:
        Dict mapping dataset name to planned_samples
    """
    sampling_strategy = config.get('sampling_strategy', {})
    mode = sampling_strategy.get('mode', 'max_samples')
    seed = sampling_strategy.get('seed', 42)

    logging.info(f"📊 Planning sampling strategy: mode={mode}")

    # Get HF client for dataset loading
    from utils.auth import get_hf_api_client
    hf_client = get_hf_api_client()

    planned_samples = {}

    if mode == 'target_tokens':
        # Mode 1: Token budget (like pretrain)
        token_budget = sampling_strategy['target_tokens']
        chars_per_token = sampling_strategy.get('chars_per_token', 4.0)
        analysis_sample_size = sampling_strategy.get('analysis_sample_size', 100)

        logging.info(f"🎯 Target token budget: {token_budget:,} tokens")

        # Step 1: Analyze each dataset to estimate avg tokens per conversation
        total_weight = sum(d.get('weight', 1.0) for d in config['datasets'])

        for dataset_config in config['datasets']:
            dataset_name = dataset_config['name']
            weight = dataset_config.get('weight', 1.0)

            logging.info(f"   Analyzing {dataset_name} (weight={weight:.2f})...")

            # Create stream factory
            stream_factory = create_stream_factory_sft(dataset_config, hf_client)

            # Estimate tokens per conversation
            text_keys = []
            for key in ['prompt', 'response', 'text']:
                if key in dataset_config.get('text_fields', {}):
                    text_keys.append(dataset_config['text_fields'][key])
            if not text_keys and 'text' in dataset_config.get('text_fields', {}):
                text_keys = [dataset_config['text_fields']['text']]

            estimation_stats = estimate_avg_tokens_per_sample(
                stream_factory=stream_factory,
                sample_size=analysis_sample_size,
                tokenizer=None,
                text_keys=text_keys if text_keys else ['text'],
                chars_per_token=chars_per_token,
                tokenizer_path=tokenizer_path
            )

            avg_tokens = estimation_stats['avg_tokens']
            dataset_config['_avg_tokens'] = avg_tokens

            # Step 2: Plan samples based on weight
            dataset_token_budget = int(token_budget * (weight / total_weight))
            planned = plan_samples_for_token_budget(
                token_budget=dataset_token_budget,
                estimation_stats=estimation_stats,
                margin_ratio=0.02
            )

            planned_samples[dataset_name] = planned

            logging.info(f"      → {avg_tokens:.0f} tokens/conv, budget={dataset_token_budget:,} → {planned:,} conversations")

            # Store planned info for later validation
            dataset_config['_planned_samples'] = planned
            dataset_config['_token_budget'] = dataset_token_budget

    elif mode == 'target_conversations':
        # Mode 2: Target conversation count
        target_conversations = sampling_strategy['target_conversations']
        total_weight = sum(d.get('weight', 1.0) for d in config['datasets'])

        logging.info(f"🎯 Target conversations: {target_conversations:,}")

        for dataset_config in config['datasets']:
            dataset_name = dataset_config['name']
            weight = dataset_config.get('weight', 1.0)

            planned = int(target_conversations * (weight / total_weight))
            planned_samples[dataset_name] = planned

            logging.info(f"   {dataset_name}: weight={weight:.2f} → {planned:,} conversations")

    elif mode == 'max_samples':
        # Mode 3: Legacy mode - use max_samples from config
        logging.info(f"📋 Using max_samples from dataset configs (legacy mode)")

        for dataset_config in config['datasets']:
            dataset_name = dataset_config['name']
            max_samples = dataset_config.get('max_samples', None)
            weight = dataset_config.get('weight', 1.0)

            planned_samples[dataset_name] = {
                'max_samples': max_samples,
                'weight': weight
            }

            if max_samples:
                logging.info(f"   {dataset_name}: max_samples={max_samples:,}, weight={weight:.2f}")
            else:
                logging.info(f"   {dataset_name}: no limit, weight={weight:.2f}")

    else:
        raise ValueError(f"Unknown sampling mode: {mode}")

    return planned_samples


def process_dataset(dataset_config: Dict[str, Any],
                   template_processor: ConversationTemplateProcessor,
                   quality_filters: Dict[str, Any],
                   hf_client) -> List[Dict[str, Any]]:
    """Process a single dataset and return formatted conversations."""

    # Load dataset
    raw_data = load_dataset_source(dataset_config, hf_client)

    processed_conversations = []
    stats = {
        'total_samples': len(raw_data),
        'extracted_conversations': 0,
        'quality_filtered': 0,
        'template_formatted': 0,
        'final_conversations': 0
    }

    logging.info(f"🔄 Processing {stats['total_samples']:,} samples from {dataset_config['name']}")

    for sample in tqdm(raw_data, desc=f"Processing {dataset_config['name']}"):
        # Extract conversation
        conversation = extract_conversation(sample, dataset_config['text_fields'])
        if not conversation:
            continue
        stats['extracted_conversations'] += 1

        # Apply quality filters
        if not apply_quality_filters(conversation, quality_filters):
            continue
        stats['quality_filtered'] += 1

        # Format with template
        try:
            formatted_text = template_processor.format_conversation(
                conversation['prompt'],
                conversation['response']
            )

            processed_conversation = {
                'text': formatted_text,
                'prompt': conversation['prompt'],
                'response': conversation['response'],
                'source_dataset': dataset_config['name'],
                'template': template_processor.template_name
            }

            processed_conversations.append(processed_conversation)
            stats['template_formatted'] += 1

        except Exception as e:
            logging.warning(f"⚠️ Failed to format conversation: {e}")
            continue

    stats['final_conversations'] = len(processed_conversations)

    # Log statistics
    logging.info(f"📊 Dataset {dataset_config['name']} processing stats:")
    for key, value in stats.items():
        logging.info(f"  {key}: {value:,}")

    return processed_conversations


def tokenize_and_validate(conversations: List[Dict[str, Any]],
                         tokenizer_path: str,
                         sequence_length: int) -> List[Dict[str, Any]]:
    """
    Validate conversations for TRL training (no tokenization - TRL will do it).

    NOTE: We do NOT tokenize here anymore. TRL SFTTrainer will tokenize on-the-fly
    during training, which is much faster and avoids memory issues.
    """

    # Normalize tokenizer path and load (only for length estimation)
    model_path = normalize_tokenizer_path(tokenizer_path)
    import sentencepiece as spm
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(model_path)

    validated_conversations = []
    stats = {
        'total_conversations': len(conversations),
        'length_filtered': 0,
        'final_conversations': 0
    }

    logging.info(f"✅ Validating {len(conversations):,} conversations (no tokenization - TRL will handle it)...")

    for conv in tqdm(conversations, desc="Validating"):
        try:
            # Estimate token count for length filtering only (not saved)
            estimated_tokens = sp_model.encode(conv['text'])

            # Check length constraints
            if len(estimated_tokens) > sequence_length:
                stats['length_filtered'] += 1
                continue

            # No tokenization stored - TRL will tokenize during training
            # Only keep text and metadata
            validated_conversations.append(conv)

        except Exception as e:
            logging.warning(f"⚠️ Validation error: {e}")
            continue

    stats['final_conversations'] = len(validated_conversations)

    # Log statistics
    logging.info(f"📊 Validation stats:")
    for key, value in stats.items():
        logging.info(f"  {key}: {value:,}")

    return validated_conversations


def pack_sequences(conversations: List[Dict[str, Any]],
                   tokenizer_path: str,
                   max_seq_length: int) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Pack multiple short conversations into fixed-length sequences for maximum GPU efficiency.

    This implements pre-packing to avoid TRL's dynamic packing overhead during training.
    Each packed sequence contains multiple conversations separated by EOS tokens.

    Args:
        conversations: List of validated conversations with 'text' field
        tokenizer_path: Path to SentencePiece tokenizer model
        max_seq_length: Maximum sequence length (e.g., 1024)

    Returns:
        Tuple of (packed_examples, packing_stats)
        - packed_examples: List of dicts with 'input_ids', 'attention_mask', 'labels'
        - packing_stats: Dictionary with packing efficiency metrics
    """

    # Load tokenizer
    model_path = normalize_tokenizer_path(tokenizer_path)
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(model_path)

    eos_token_id = sp_model.eos_id()
    pad_token_id = sp_model.pad_id() if hasattr(sp_model, 'pad_id') else sp_model.unk_id()

    logging.info(f"📦 Starting sequence packing (max_length={max_seq_length})...")
    logging.info(f"  EOS token ID: {eos_token_id}")
    logging.info(f"  PAD token ID: {pad_token_id}")

    # Tokenize all conversations first
    tokenized_conversations = []
    total_raw_tokens = 0

    for conv in tqdm(conversations, desc="Tokenizing"):
        token_ids = sp_model.encode(conv['text'])
        # Add EOS at the end of each conversation
        token_ids.append(eos_token_id)
        tokenized_conversations.append({
            'token_ids': token_ids,
            'metadata': conv
        })
        total_raw_tokens += len(token_ids)

    logging.info(f"✅ Tokenized {len(tokenized_conversations):,} conversations")
    logging.info(f"  Total raw tokens: {total_raw_tokens:,}")

    # Sort by length for better packing efficiency (shortest first)
    tokenized_conversations.sort(key=lambda x: len(x['token_ids']))

    # Pack sequences
    packed_examples = []
    current_sequence = []
    current_length = 0
    conversations_packed = 0
    conversations_per_sequence = []
    wasted_padding = 0

    for tokenized_conv in tqdm(tokenized_conversations, desc="Packing"):
        token_ids = tokenized_conv['token_ids']
        conv_length = len(token_ids)

        # Check if adding this conversation would exceed max_seq_length
        if current_length + conv_length <= max_seq_length:
            # Add to current sequence
            current_sequence.extend(token_ids)
            current_length += conv_length
            conversations_packed += 1
        else:
            # Save current sequence if not empty
            if current_sequence:
                # Pad to max_seq_length
                padding_needed = max_seq_length - current_length
                wasted_padding += padding_needed

                input_ids = current_sequence + [pad_token_id] * padding_needed
                attention_mask = [1] * current_length + [0] * padding_needed
                # Labels: -100 for padding tokens (ignored in loss)
                labels = current_sequence + [-100] * padding_needed

                packed_examples.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                })

                conversations_per_sequence.append(conversations_packed)

                # Reset for next sequence
                current_sequence = []
                current_length = 0
                conversations_packed = 0

            # Start new sequence with current conversation
            if conv_length <= max_seq_length:
                current_sequence = token_ids
                current_length = conv_length
                conversations_packed = 1
            else:
                # Conversation too long - truncate it
                logging.warning(f"⚠️ Conversation too long ({conv_length} tokens), truncating to {max_seq_length}")
                truncated_ids = token_ids[:max_seq_length]

                packed_examples.append({
                    'input_ids': truncated_ids,
                    'attention_mask': [1] * max_seq_length,
                    'labels': truncated_ids
                })
                conversations_per_sequence.append(1)

    # Don't forget the last sequence
    if current_sequence:
        padding_needed = max_seq_length - current_length
        wasted_padding += padding_needed

        input_ids = current_sequence + [pad_token_id] * padding_needed
        attention_mask = [1] * current_length + [0] * padding_needed
        labels = current_sequence + [-100] * padding_needed

        packed_examples.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        })
        conversations_per_sequence.append(conversations_packed)

    # Calculate packing statistics
    total_packed_tokens = len(packed_examples) * max_seq_length
    total_real_tokens = total_packed_tokens - wasted_padding
    packing_efficiency = (total_real_tokens / total_packed_tokens) * 100 if total_packed_tokens > 0 else 0
    avg_conversations_per_sequence = sum(conversations_per_sequence) / len(conversations_per_sequence) if conversations_per_sequence else 0

    packing_stats = {
        'total_conversations': len(tokenized_conversations),
        'total_packed_sequences': len(packed_examples),
        'total_raw_tokens': total_raw_tokens,
        'total_packed_tokens': total_packed_tokens,
        'total_real_tokens': total_real_tokens,
        'wasted_padding': wasted_padding,
        'packing_efficiency': round(packing_efficiency, 2),
        'avg_conversations_per_sequence': round(avg_conversations_per_sequence, 2),
        'compression_ratio': round(len(tokenized_conversations) / len(packed_examples), 2) if packed_examples else 0
    }

    logging.info(f"✅ Packing completed!")
    logging.info(f"📊 Packing statistics:")
    logging.info(f"  Total conversations: {packing_stats['total_conversations']:,}")
    logging.info(f"  Packed sequences: {packing_stats['total_packed_sequences']:,}")
    logging.info(f"  Packing efficiency: {packing_stats['packing_efficiency']:.2f}%")
    logging.info(f"  Avg conversations/sequence: {packing_stats['avg_conversations_per_sequence']:.2f}")
    logging.info(f"  Compression ratio: {packing_stats['compression_ratio']:.2f}x")
    logging.info(f"  Wasted padding tokens: {packing_stats['wasted_padding']:,}")

    return packed_examples, packing_stats


def create_shards(conversations: List[Dict[str, Any]],
                 output_dir: Path,
                 shard_size: int,
                 train_ratio: float,
                 is_packed: bool = False) -> Dict[str, Any]:
    """
    Create training and validation shards.

    Args:
        conversations: List of conversations (raw or packed)
        output_dir: Output directory for shards
        shard_size: Number of examples per shard
        train_ratio: Ratio of training data (e.g., 0.9)
        is_packed: Whether data is pre-packed (input_ids) or raw text

    Returns:
        Dictionary with split information
    """

    # Shuffle conversations
    import random
    random.shuffle(conversations)

    # Split into train/val
    split_idx = int(len(conversations) * train_ratio)
    train_conversations = conversations[:split_idx]
    val_conversations = conversations[split_idx:]

    data_type = "packed sequences" if is_packed else "conversations"
    logging.info(f"📂 Creating shards:")
    logging.info(f"  Train: {len(train_conversations):,} {data_type}")
    logging.info(f"  Validation: {len(val_conversations):,} {data_type}")

    # Create shards for each split
    splits_info = {}

    for split_name, split_conversations in [('train', train_conversations), ('val', val_conversations)]:
        if not split_conversations:
            continue

        # Calculate number of shards
        num_shards = max(1, math.ceil(len(split_conversations) / shard_size))
        shard_files = []
        total_tokens = 0

        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, len(split_conversations))
            shard_conversations = split_conversations[start_idx:end_idx]

            # Write shard
            shard_filename = f"{split_name}_{shard_idx:05d}.jsonl"
            shard_path = output_dir / shard_filename

            shard_tokens = 0
            with open(shard_path, 'w', encoding='utf-8') as f:
                for conv in shard_conversations:
                    f.write(json.dumps(conv, ensure_ascii=False) + '\n')
                    # Count tokens if packed data
                    if is_packed and 'input_ids' in conv:
                        shard_tokens += len(conv['input_ids'])

            total_tokens += shard_tokens

            shard_info = {
                'filename': shard_filename,
            }

            if is_packed:
                shard_info['sequences'] = len(shard_conversations)
                shard_info['tokens'] = shard_tokens
            else:
                shard_info['conversations'] = len(shard_conversations)

            shard_files.append(shard_info)

        split_info = {
            'num_shards': num_shards,
            'shards': shard_files
        }

        if is_packed:
            split_info['sequences'] = len(split_conversations)
            split_info['total_tokens'] = total_tokens
        else:
            split_info['conversations'] = len(split_conversations)

        splits_info[split_name] = split_info

    return splits_info


def create_manifest(config: Dict[str, Any],
                   tokenizer_metadata: Dict[str, Any],
                   splits_info: Dict[str, Any],
                   output_dir: Path,
                   packing_stats: Optional[Dict[str, int]] = None) -> None:
    """
    Create manifest file with all metadata.

    Args:
        config: Dataset configuration
        tokenizer_metadata: Tokenizer metadata (only for packed data)
        splits_info: Information about train/val splits
        output_dir: Output directory
        packing_stats: Packing statistics (if pre-packed data)
    """

    is_packed = packing_stats is not None

    manifest = {
        'created_at': datetime.now().isoformat(),
        'config': config,
        'splits': splits_info,
    }

    if is_packed:
        # Format v3.0: Pre-packed with input_ids, attention_mask, labels
        manifest['format_version'] = '3.0'
        manifest['pipeline_version'] = 'sft_industrial_v3_prepacked'
        manifest['tokenizer_metadata'] = tokenizer_metadata
        manifest['packing_metadata'] = packing_stats
        manifest['total_sequences'] = sum(split_info.get('sequences', 0) for split_info in splits_info.values())
        manifest['total_tokens'] = sum(split_info.get('total_tokens', 0) for split_info in splits_info.values())
    else:
        # Format v2.0: Raw text, TRL will tokenize
        manifest['format_version'] = '2.0'
        manifest['pipeline_version'] = 'sft_industrial_v2_raw_text'
        manifest['total_conversations'] = sum(split_info.get('conversations', 0) for split_info in splits_info.values())
        # No tokenizer_metadata - data is not tokenized

    # Write manifest
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logging.info(f"📋 Created manifest: {manifest_path}")
    if is_packed:
        logging.info(f"  Format: v3.0 (pre-packed sequences)")
        logging.info(f"  Packing efficiency: {packing_stats['packing_efficiency']:.2f}%")


def create_data_card(config: Dict[str, Any],
                    splits_info: Dict[str, Any],
                    output_dir: Path,
                    packing_stats: Optional[Dict[str, int]] = None) -> None:
    """
    Create data card documentation.

    Args:
        config: Dataset configuration
        splits_info: Information about train/val splits
        output_dir: Output directory
        packing_stats: Packing statistics (if pre-packed data)
    """

    is_packed = packing_stats is not None

    if is_packed:
        total_sequences = sum(split_info.get('sequences', 0) for split_info in splits_info.values())
        total_tokens = sum(split_info.get('total_tokens', 0) for split_info in splits_info.values())

        data_card = f"""# SFT Dataset Card: {config['name']}

## Overview
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Template**: {config['template']}
- **Format**: Pre-packed sequences (v3.0)
- **Total Sequences**: {total_sequences:,}
- **Total Tokens**: {total_tokens:,}

## Packing Statistics
- **Original Conversations**: {packing_stats['total_conversations']:,}
- **Packed Sequences**: {packing_stats['total_packed_sequences']:,}
- **Packing Efficiency**: {packing_stats['packing_efficiency']:.2f}%
- **Avg Conversations/Sequence**: {packing_stats['avg_conversations_per_sequence']:.2f}
- **Compression Ratio**: {packing_stats['compression_ratio']:.2f}x
- **Wasted Padding**: {packing_stats['wasted_padding']:,} tokens

## Data Splits
"""

        for split_name, split_info in splits_info.items():
            data_card += f"""
### {split_name.title()} Split
- **Sequences**: {split_info.get('sequences', 0):,}
- **Tokens**: {split_info.get('total_tokens', 0):,}
- **Shards**: {split_info['num_shards']}
"""
    else:
        total_conversations = sum(split_info.get('conversations', 0) for split_info in splits_info.values())

        data_card = f"""# SFT Dataset Card: {config['name']}

## Overview
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Template**: {config['template']}
- **Total Conversations**: {total_conversations:,}
- **Format**: Raw text (v2.0) - TRL will tokenize during training

## Data Splits
"""

        for split_name, split_info in splits_info.items():
            data_card += f"""
### {split_name.title()} Split
- **Conversations**: {split_info.get('conversations', 0):,}
- **Shards**: {split_info['num_shards']}
"""

    data_card += f"""
## Source Datasets
"""

    for dataset in config['datasets']:
        data_card += f"""
### {dataset['name']}
- **Type**: {dataset['type']}
- **Source**: {dataset['dataset_name']}
- **Fields**: {dataset['text_fields']}
"""

    data_card += f"""
## Quality Filters Applied
"""

    if 'quality_filters' in config:
        for filter_name, filter_value in config['quality_filters'].items():
            data_card += f"- **{filter_name}**: {filter_value}\n"

    data_card += f"""
## Template Format: {config['template']}

This dataset uses the {config['template']} conversation template for formatting.

## Usage

Load this dataset with the Lumi-Lab SFT pipeline:

```python
from utils.dataset_utils import SFTDataset

dataset = SFTDataset(
    data_dir="{output_dir.name}",
    split="train"
)
```
"""

    # Write data card
    data_card_path = output_dir / 'DATA_CARD.md'
    with open(data_card_path, 'w', encoding='utf-8') as f:
        f.write(data_card)

    logging.info(f"📄 Created data card: {data_card_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare SFT corpus for industrial training")

    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Configuration file (JSON/YAML)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed corpus')
    parser.add_argument('--tokenizer_path', '--tokenizer_dir', type=str, required=True,
                       dest='tokenizer_path',
                       help='Path to SentencePiece tokenizer (accepts --tokenizer_dir alias)')

    # Optional arguments
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing output directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dry_run', action='store_true',
                       help='Perform dry run without writing files')
    parser.add_argument('--log-level', type=str, default=None,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set logging level (overrides --verbose)')

    # Streaming mode arguments
    parser.add_argument('--use-streaming', action='store_true',
                       help='Use TRUE STREAMING mode for scalable processing (constant memory <100MB)')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh all caches when using --use-streaming')

    args = parser.parse_args()

    # Setup signal handlers
    _setup_signal_handlers()

    # Create output directory
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"Output directory exists: {output_dir}. Use --force to overwrite.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_dir, args.verbose, getattr(args, 'log_level', None))

    # Load and validate configuration first (to get enable_packing)
    config = load_config(args.config)
    validate_config(config)

    # Extract enable_packing from config (with default fallback)
    enable_packing = config.get('output_params', {}).get('enable_packing', False)

    logging.info("🚀 Starting SFT corpus preparation...")
    logging.info(f"📋 Configuration: {args.config}")
    logging.info(f"📂 Output directory: {output_dir}")
    logging.info(f"🔤 Tokenizer: {args.tokenizer_path}")
    if enable_packing:
        logging.info(f"📦 Packing: ENABLED (format v3.0)")
    else:
        logging.info(f"📦 Packing: DISABLED (format v2.0)")

    try:

        # ============================================================
        # STREAMING MODE (NEW - Scalable, constant memory)
        # ============================================================
        if args.use_streaming:
            logging.info("=" * 80)
            logging.info("🚀 Using TRUE STREAMING SFT processing mode")
            logging.info("=" * 80)
            logging.info("✅ Constant memory usage <100MB (regardless of dataset size)")
            logging.info("✅ Token budget enforcement with real-time stop")
            logging.info("✅ Resume capability via Parquet caching")
            logging.info("✅ Global cross-dataset deduplication")
            logging.info("=" * 80)

            # Create cache directory
            cache_dir = output_dir / "cache"

            # Initialize streaming processor
            from utils.sft_streaming import SFTStreamingProcessor

            streaming_processor = SFTStreamingProcessor(
                cache_dir=cache_dir,
                tokenizer_path=args.tokenizer_path,
                template_name=config['template']
            )

            # Process with streaming
            results = streaming_processor.process_config_streaming(
                config=config,
                output_dir=output_dir,
                force_refresh=args.force_refresh
            )

            # Check if packing was used (v3.0 format)
            is_packed = results['assembly_result'].get('is_packed', False)
            packing_stats = results['assembly_result'].get('packing_stats', None)

            # Create manifest
            create_manifest(
                config=config,
                tokenizer_metadata=verify_tokenizer(args.tokenizer_path),
                splits_info=results['assembly_result']['splits_info'],
                output_dir=output_dir,
                packing_stats=packing_stats  # Pass packing stats if packing was enabled
            )

            # Create data card
            create_data_card(
                config=config,
                splits_info=results['assembly_result']['splits_info'],
                output_dir=output_dir,
                packing_stats=packing_stats
            )

            # Log final summary
            logging.info("")
            logging.info("=" * 80)
            logging.info("🎉 TRUE STREAMING SFT corpus preparation complete!")
            logging.info("=" * 80)
            logging.info(f"📊 Total conversations: {results['total_conversations']:,}")
            logging.info(f"📊 Total tokens: {results['total_tokens']:,} ({results['total_tokens']/1e6:.1f}M)")
            logging.info(f"🗑️  Duplicates removed: {results['deduplication_stats']['duplicates_found']:,} ({results['deduplication_stats']['deduplication_rate']*100:.1f}%)")
            if is_packed and packing_stats:
                logging.info(f"📦 Format: Pre-packed v3.0 (input_ids, attention_mask, labels)")
                logging.info(f"📦 Packed sequences: {packing_stats['total_packed_sequences']:,}")
                logging.info(f"📦 Packing efficiency: {packing_stats['packing_efficiency']:.1f}%")
            else:
                logging.info(f"📝 Format: Raw text v2.0 (TRL will tokenize during training)")
            logging.info(f"💾 Memory usage: CONSTANT <100MB peak")
            logging.info(f"⚡ Scalability: UNLIMITED dataset size")
            logging.info(f"📂 Output: {output_dir}")
            logging.info("=" * 80)

            # Clean exit
            _safe_cleanup_and_exit()
            return

        # ============================================================
        # LEGACY MODE (Original - Memory accumulation)
        # ============================================================
        logging.info("📋 Using ORIGINAL SFT processing pipeline (legacy mode)")
        logging.info("⚠️  Memory usage scales with dataset size (~5GB for 1M conversations)")
        logging.info("💡 Use --use-streaming for unlimited scalability")
        logging.info("")

        # Verify tokenizer
        tokenizer_metadata = verify_tokenizer(args.tokenizer_path)

        # Initialize template processor
        template_processor = ConversationTemplateProcessor(config['template'])

        # Get HuggingFace client
        hf_client = get_hf_api_client()

        # Plan sampling strategy (supports 3 modes: target_tokens, target_conversations, max_samples)
        planned_samples_map = plan_sampling_strategy(config, args.tokenizer_path)

        # Get seed from config (config-driven, no hardcoded values)
        sampling_strategy = config.get('sampling_strategy', {})
        seed = sampling_strategy.get('seed', 42)
        mode = sampling_strategy.get('mode', 'max_samples')

        # Process all datasets and apply planned sampling
        all_conversations = []
        dataset_stats = []

        for dataset_config in config['datasets']:
            dataset_name = dataset_config['name']

            conversations = process_dataset(
                dataset_config,
                template_processor,
                config.get('quality_filters', {}),
                hf_client
            )

            num_conversations = len(conversations)

            # Apply planned sampling based on mode
            if mode == 'max_samples':
                # Legacy mode: use weight-based sampling
                plan = planned_samples_map[dataset_name]
                weight = plan['weight']
                max_samples = plan.get('max_samples')

                if max_samples and num_conversations > max_samples:
                    num_sampled = max_samples
                else:
                    num_sampled = int(num_conversations * weight)

            else:
                # Modern modes (target_tokens, target_conversations): use planned samples
                num_sampled = min(planned_samples_map[dataset_name], num_conversations)

            # Sample conversations deterministically
            if num_sampled < num_conversations:
                import random
                random.seed(seed)
                sampled_conversations = random.sample(conversations, num_sampled)
            else:
                sampled_conversations = conversations

            all_conversations.extend(sampled_conversations)

            # Track stats
            dataset_stats.append({
                'name': dataset_name,
                'total': num_conversations,
                'planned': num_sampled,
                'sampled': len(sampled_conversations)
            })

        # Log dataset sampling stats
        logging.info(f"📊 Dataset sampling statistics:")

        total_tokens_planned = 0
        total_tokens_actual = 0

        for stat in dataset_stats:
            logging.info(f"   {stat['name']}: {stat['total']:,} → {stat['sampled']:,} (planned: {stat['planned']:,})")

            # Calculate token shortfall for target_tokens mode
            if mode == 'target_tokens':
                # Find the dataset config with stored planning info
                dataset_cfg = next((d for d in config['datasets'] if d['name'] == stat['name']), None)
                if dataset_cfg and '_planned_samples' in dataset_cfg:
                    planned = dataset_cfg['_planned_samples']
                    token_budget = dataset_cfg['_token_budget']
                    avg_tokens = dataset_cfg.get('_avg_tokens', 250)

                    actual_samples = stat['sampled']
                    actual_tokens = actual_samples * avg_tokens

                    total_tokens_planned += token_budget
                    total_tokens_actual += actual_tokens

                    # Warn if dataset significantly undersized
                    if actual_samples < planned * 0.5:  # Less than 50% of plan
                        shortfall_pct = ((planned - actual_samples) / planned) * 100
                        token_shortfall = token_budget - actual_tokens
                        logging.warning(
                            f"⚠️  Dataset '{stat['name']}' is too small! "
                            f"Planned {planned:,} conversations but only {actual_samples:,} available "
                            f"({shortfall_pct:.1f}% shortfall, ~{token_shortfall/1e6:.1f}M tokens missing)"
                        )

        # Overall token budget warning for target_tokens mode
        if mode == 'target_tokens' and total_tokens_actual > 0:
            target_tokens = sampling_strategy['target_tokens']
            shortfall = target_tokens - total_tokens_actual
            achievement_pct = (total_tokens_actual / target_tokens) * 100

            logging.info(f"")
            logging.info(f"📊 Token Budget Summary:")
            logging.info(f"   Target: {target_tokens/1e6:.1f}M tokens")
            logging.info(f"   Actual: {total_tokens_actual/1e6:.1f}M tokens ({achievement_pct:.1f}% of target)")

            if achievement_pct < 80:
                logging.warning(
                    f"⚠️  Token budget significantly underachieved! "
                    f"Missing {shortfall/1e6:.1f}M tokens ({100-achievement_pct:.1f}% shortfall)"
                )
                logging.warning(f"💡 Consider:")
                logging.warning(f"   1. Reducing target_tokens to {int(total_tokens_actual)} (~{total_tokens_actual/1e6:.0f}M)")
                logging.warning(f"   2. Adding larger datasets (UltraChat 200k, ShareGPT 90k, WizardLM 70k)")
                logging.warning(f"   3. Adjusting weights to use available data more effectively")
            elif achievement_pct < 95:
                logging.info(f"ℹ️  Token budget slightly underachieved but acceptable")

        logging.info(f"✅ Total conversations collected: {len(all_conversations):,}")

        if not all_conversations:
            raise ValueError("No conversations were successfully processed!")

        # Tokenize and validate
        validated_conversations = tokenize_and_validate(
            all_conversations,
            args.tokenizer_path,
            config['output_params']['sequence_length']
        )

        # Optionally pack sequences
        packing_stats = None
        if enable_packing:
            packed_data, packing_stats = pack_sequences(
                validated_conversations,
                args.tokenizer_path,
                config['output_params']['sequence_length']
            )
            data_to_shard = packed_data
            is_packed = True
        else:
            data_to_shard = validated_conversations
            is_packed = False

        if not args.dry_run:
            # Create shards
            splits_info = create_shards(
                data_to_shard,
                output_dir,
                config['output_params']['shard_size'],
                config['output_params']['train_ratio'],
                is_packed=is_packed
            )

            # Create manifest
            create_manifest(config, tokenizer_metadata, splits_info, output_dir, packing_stats)

            # Create data card
            create_data_card(config, splits_info, output_dir, packing_stats)

            logging.info(f"🎉 SFT corpus preparation completed successfully!")
            logging.info(f"📊 Final stats:")
            if is_packed:
                logging.info(f"  Total sequences: {packing_stats['total_packed_sequences']:,}")
                logging.info(f"  Total tokens: {packing_stats['total_packed_tokens']:,}")
                logging.info(f"  Packing efficiency: {packing_stats['packing_efficiency']:.2f}%")
                logging.info(f"  Format: Pre-packed (v3.0)")
            else:
                logging.info(f"  Total conversations: {len(validated_conversations):,}")
                logging.info(f"  Format: Raw text (v2.0) - TRL will tokenize during training")
            logging.info(f"  Output directory: {output_dir}")
        else:
            logging.info("🔍 Dry run completed - no files written")

    except Exception as e:
        logging.error(f"❌ Error during SFT corpus preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()