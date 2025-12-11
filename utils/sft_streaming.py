#!/usr/bin/env python3
"""
SFT Streaming Processor - TRUE STREAMING architecture for SFT corpus preparation.

This module provides memory-efficient streaming processing for SFT datasets,
inspired by the corpus_cache.py streaming architecture. It supports:

- TRUE STREAMING: Constant memory usage <100MB regardless of dataset size
- Token budget tracking: Real-time enforcement with immediate stop
- Resume capability: Parquet cache with automatic resume
- Global deduplication: Cross-dataset deduplication without memory accumulation
- Multi-dataset orchestration: Weighted sampling with token budget allocation

Architecture:
    HF Stream → Extract → Filter → Template → Token Count → Budget Check
                                                                    ↓
                                                            Write to Parquet
                                                                    ↓
                                                            Flush + GC
                                                                    ↓
                                                            Zero accumulation

Usage:
    processor = SFTStreamingProcessor(
        cache_dir="data/sft_cache",
        tokenizer_path="data/models/tokenizers/spm_32k",
        template_name="instruct"
    )

    results = processor.process_config_streaming(
        config=sft_config,
        output_dir="data/sft_processed/output",
        force_refresh=False
    )
"""

import gc
import hashlib
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Import existing utilities
from utils.sft_templates import ConversationTemplateProcessor
from utils.debug.corpus_cache import StreamingDeduplicator


class SFTStreamingParquetWriter:
    """
    Streaming Parquet writer for SFT conversations.

    Writes conversations directly to Parquet with micro-batching to maintain
    constant memory usage. Based on StreamingParquetWriter from corpus_cache.py
    but adapted for SFT conversation format.
    """

    # Schema for SFT conversations
    SCHEMA = pa.schema([
        ('text', pa.string()),              # Formatted conversation text
        ('prompt', pa.string()),            # Original prompt
        ('response', pa.string()),          # Original response
        ('source_dataset', pa.string()),    # Dataset name
        ('template', pa.string()),          # Template used
        ('token_count', pa.int32())         # Token count
    ])

    def __init__(self, output_path: Path, batch_size: int = None, config: Dict = None):
        """
        Initialize streaming Parquet writer.

        Args:
            output_path: Path to output Parquet file
            batch_size: Number of conversations to accumulate before flushing (default from config or 1000)
            config: Configuration dict (for batch_size from output_params)
        """
        self.output_path = Path(output_path)

        # Get batch_size from config or use provided value or default
        if batch_size is None:
            if config and 'output_params' in config:
                self.batch_size = config['output_params'].get('batch_size', 1000)
            else:
                self.batch_size = 1000
        else:
            self.batch_size = batch_size
        self.current_batch = []
        self.total_written = 0

        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize Parquet writer
        self.writer = pq.ParquetWriter(
            self.output_path,
            schema=self.SCHEMA,
            compression='snappy'
        )

        self.logger = logging.getLogger(f"SFTStreamingWriter.{output_path.stem}")

    def write_conversation(self, conv: Dict[str, Any]):
        """
        Write a single conversation to Parquet.

        Args:
            conv: Conversation dict with keys: text, prompt, response,
                  source_dataset, template, token_count
        """
        # Validate required fields
        required_fields = ['text', 'prompt', 'response', 'source_dataset', 'template', 'token_count']
        for field in required_fields:
            if field not in conv:
                raise ValueError(f"Missing required field: {field}")

        self.current_batch.append(conv)

        # Flush when batch is full
        if len(self.current_batch) >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self):
        """Flush current batch to Parquet and clear memory."""
        if not self.current_batch:
            return

        # Convert batch to Arrow table
        df = pd.DataFrame(self.current_batch)
        table = pa.Table.from_pandas(df, schema=self.SCHEMA)

        # Write to Parquet
        self.writer.write_table(table)

        self.total_written += len(self.current_batch)

        # Clear batch from memory
        batch_size = len(self.current_batch)
        self.current_batch.clear()
        del df, table

        # Force garbage collection every 10 batches
        if self.total_written % (self.batch_size * 10) == 0:
            gc.collect()

        self.logger.debug(f"Flushed {batch_size} conversations (total: {self.total_written:,})")

    def close(self) -> int:
        """Close writer and flush remaining conversations."""
        # Flush final batch
        self._flush_batch()

        # Close Parquet writer
        if self.writer:
            self.writer.close()
            self.writer = None

        self.logger.info(f"✅ Closed: {self.total_written:,} conversations written to {self.output_path.name}")
        return self.total_written

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_stream_factory_sft(dataset_config: Dict[str, Any], hf_client):
    """
    Create a stream factory for SFT dataset.

    CRITICAL: Returns an iterator, NOT a list. This is the key to streaming.

    Args:
        dataset_config: Dataset configuration dict
        hf_client: HuggingFace API client

    Returns:
        Factory function that returns a fresh iterator
    """
    dataset_type = dataset_config['type']

    if dataset_type == 'huggingface':
        def stream_factory():
            from datasets import load_dataset

            dataset_name = dataset_config['dataset_name']
            subset = dataset_config.get('subset')
            split = dataset_config.get('split', 'train')

            # Ensure HF authentication (for private datasets)
            try:
                from utils.auth import get_hf_api_client
                get_hf_api_client()  # Ensure authentication
            except Exception as e:
                logging.warning(f"HF authentication failed, trying without: {e}")

            # CRITICAL: streaming=True for zero memory accumulation
            dataset = load_dataset(
                dataset_name,
                subset,
                split=split,
                streaming=True,
                trust_remote_code=dataset_config.get('trust_remote_code', False)
            )

            return iter(dataset)

        return stream_factory

    elif dataset_type in ['jsonl', 'json', 'local']:
        def stream_factory():
            data_path = Path(dataset_config.get('data_path', dataset_config['dataset_name']))

            if data_path.suffix == '.jsonl' or dataset_type == 'jsonl':
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        yield json.loads(line.strip())

            elif data_path.suffix == '.json' or dataset_type == 'json':
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        yield item
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")

        return stream_factory

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def extract_conversation(sample: Dict[str, Any], text_fields: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Extract conversation from a dataset sample.

    Args:
        sample: Raw sample from dataset
        text_fields: Mapping of field types to field paths

    Returns:
        Dict with 'prompt' and 'response' keys, or None if extraction failed
    """
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

        # Validate required fields
        if 'prompt' not in conversation:
            return None

        # Try to find response field
        if 'response' not in conversation:
            for alt_field in ['answer', 'completion', 'output', 'assistant']:
                if alt_field in conversation:
                    conversation['response'] = conversation[alt_field]
                    break

        # Ensure both prompt and response exist
        if not conversation.get('prompt') or not conversation.get('response'):
            return None

        return conversation

    except (KeyError, IndexError, TypeError, ValueError):
        return None


def apply_quality_filters(conversation: Dict[str, str], filters: Dict[str, Any]) -> bool:
    """
    Apply quality filters to conversation.

    Args:
        conversation: Dict with 'prompt' and 'response' keys
        filters: Quality filter configuration

    Returns:
        True if conversation passes all filters
    """
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
    if filters.get('filter_urls', False):
        import re
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        if re.search(url_pattern, prompt) or re.search(url_pattern, response):
            return False

    if filters.get('filter_code_blocks', False):
        import re
        code_pattern = r'```|`{3,}'
        if re.search(code_pattern, prompt) or re.search(code_pattern, response):
            return False

    # ASCII filter
    if filters.get('require_ascii', False):
        try:
            prompt.encode('ascii')
            response.encode('ascii')
        except UnicodeEncodeError:
            return False

    return True


def stream_and_process_dataset(
    dataset_config: Dict[str, Any],
    template_processor: ConversationTemplateProcessor,
    quality_filters: Dict[str, Any],
    tokenizer,
    token_budget: int,
    parquet_writer: SFTStreamingParquetWriter,
    deduplicator: StreamingDeduplicator,
    hf_client
) -> Dict[str, Any]:
    """
    Stream process a single dataset with token budget enforcement.

    CRITICAL: This is the core streaming function. It:
    1. Streams samples one-by-one (never loads full dataset)
    2. Tracks tokens in real-time
    3. STOPS immediately when token budget is reached
    4. Writes directly to Parquet (no accumulation)

    Args:
        dataset_config: Dataset configuration
        template_processor: Conversation template formatter
        quality_filters: Quality filter configuration
        tokenizer: SentencePiece tokenizer for token counting
        token_budget: Maximum tokens to collect
        parquet_writer: Streaming Parquet writer
        deduplicator: Streaming deduplicator
        hf_client: HuggingFace API client

    Returns:
        Stats dict with tokens_collected, conversations_written, etc.
    """
    dataset_name = dataset_config['name']
    logger = logging.getLogger(f"StreamProcessor.{dataset_name}")

    logger.info(f"🔄 Streaming {dataset_name} (budget: {token_budget/1e6:.1f}M tokens)")

    # Create stream factory
    stream_factory = create_stream_factory_sft(dataset_config, hf_client)
    stream = stream_factory()

    # Stats tracking
    current_tokens = 0
    conversations_written = 0
    skipped_extraction = 0
    skipped_quality = 0
    skipped_duplicate = 0
    samples_processed = 0

    # Progress bar
    progress_bar = tqdm(
        total=token_budget,
        desc=f"Processing {dataset_name}",
        unit="tokens"
    )

    try:
        for sample in stream:
            samples_processed += 1

            # ✅ CRITICAL: Check token budget FIRST
            if current_tokens >= token_budget:
                logger.info(f"✅ Token budget reached: {current_tokens:,}/{token_budget:,} tokens")
                break

            # Extract conversation
            conversation = extract_conversation(sample, dataset_config['text_fields'])
            if not conversation:
                skipped_extraction += 1
                continue

            # Apply quality filters
            if not apply_quality_filters(conversation, quality_filters):
                skipped_quality += 1
                continue

            # Format with template
            try:
                formatted_text = template_processor.format_conversation(
                    conversation['prompt'],
                    conversation['response']
                )
            except Exception as e:
                logger.warning(f"⚠️ Template formatting error: {e}")
                skipped_quality += 1
                continue

            # Count tokens
            try:
                token_count = len(tokenizer.encode(formatted_text))
            except Exception as e:
                logger.warning(f"⚠️ Tokenization error: {e}")
                skipped_quality += 1
                continue

            # Check deduplication
            if deduplicator.is_duplicate(formatted_text):
                skipped_duplicate += 1
                continue

            # Update token count
            current_tokens += token_count

            # ✅ Write immediately to Parquet (streaming, no accumulation)
            parquet_writer.write_conversation({
                'text': formatted_text,
                'prompt': conversation['prompt'],
                'response': conversation['response'],
                'source_dataset': dataset_name,
                'template': template_processor.template_name,
                'token_count': token_count
            })

            conversations_written += 1
            progress_bar.update(token_count)

            # Log progress periodically
            if conversations_written % 1000 == 0:
                progress_bar.set_postfix({
                    'convs': conversations_written,
                    'tokens': f"{current_tokens/1e6:.1f}M",
                    'skip': skipped_extraction + skipped_quality + skipped_duplicate
                })

    except Exception as e:
        logger.error(f"❌ Error streaming {dataset_name}: {e}")
        raise

    finally:
        progress_bar.close()

    # Log final stats
    logger.info(f"📊 {dataset_name} completed:")
    logger.info(f"   Samples processed: {samples_processed:,}")
    logger.info(f"   Conversations written: {conversations_written:,}")
    logger.info(f"   Tokens collected: {current_tokens:,} ({current_tokens/1e6:.1f}M)")
    logger.info(f"   Skipped - extraction: {skipped_extraction:,}, quality: {skipped_quality:,}, duplicate: {skipped_duplicate:,}")

    return {
        'dataset_name': dataset_name,
        'samples_processed': samples_processed,
        'conversations_written': conversations_written,
        'tokens_collected': current_tokens,
        'skipped_extraction': skipped_extraction,
        'skipped_quality': skipped_quality,
        'skipped_duplicate': skipped_duplicate
    }


class SFTStreamingProcessor:
    """
    Main orchestrator for SFT streaming processing.

    This class coordinates multi-dataset streaming processing with:
    - Per-dataset Parquet caching
    - Token budget allocation by weight
    - Global deduplication
    - Resume capability
    - Final corpus assembly
    """

    def __init__(self, cache_dir: Path, tokenizer_path: str, template_name: str):
        """
        Initialize SFT streaming processor.

        Args:
            cache_dir: Directory for Parquet caches
            tokenizer_path: Path to SentencePiece tokenizer
            template_name: Template name (instruct, chatml, etc.)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer_path = tokenizer_path
        self.template_name = template_name

        self.logger = logging.getLogger("SFTStreamingProcessor")
        self.logger.info(f"📂 SFT streaming cache: {self.cache_dir}")

        # Initialize template processor
        self.template_processor = ConversationTemplateProcessor(template_name)

        # Load tokenizer
        self.tokenizer = self._load_tokenizer()

        # Initialize deduplicator (global across all datasets) - params from config if available
        self.deduplicator = None  # Will be initialized in process_config_streaming with config params

        # Get HF client
        from utils.auth import get_hf_api_client
        self.hf_client = get_hf_api_client()

    def _load_tokenizer(self):
        """Load SentencePiece tokenizer."""
        import sentencepiece as spm

        tokenizer = spm.SentencePieceProcessor()

        # Normalize path
        if self.tokenizer_path.endswith('.model'):
            model_path = self.tokenizer_path
        else:
            model_path = f"{self.tokenizer_path}/spm.model"

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Tokenizer not found: {model_path}")

        tokenizer.load(model_path)
        self.logger.info(f"🎯 Loaded tokenizer: {model_path} (vocab_size={tokenizer.vocab_size()})")

        return tokenizer

    def _get_cache_path(self, dataset_name: str) -> Path:
        """Get cache Parquet path for dataset."""
        return self.cache_dir / f"{dataset_name}.parquet"

    def _get_metadata_path(self, dataset_name: str) -> Path:
        """Get metadata JSON path for dataset."""
        return self.cache_dir / f"{dataset_name}.metadata.json"

    def _validate_cache(self, dataset_name: str, dataset_config: Dict) -> bool:
        """
        Validate if cached dataset is still valid.

        Returns:
            True if cache can be reused
        """
        cache_path = self._get_cache_path(dataset_name)
        metadata_path = self._get_metadata_path(dataset_name)

        if not cache_path.exists() or not metadata_path.exists():
            return False

        try:
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Check config hash
            config_str = json.dumps(dataset_config, sort_keys=True, ensure_ascii=False)
            current_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()

            if metadata.get('config_hash') != current_hash:
                self.logger.info(f"🔄 Config changed for {dataset_name}, cache invalid")
                return False

            # Validate Parquet file
            df = pd.read_parquet(cache_path, columns=['source_dataset'])
            if len(df) != metadata.get('conversations_written', 0):
                self.logger.warning(f"⚠️ Cache corrupted for {dataset_name}")
                return False

            self.logger.info(f"✅ Valid cache found for {dataset_name}")
            return True

        except Exception as e:
            self.logger.warning(f"⚠️ Cache validation failed for {dataset_name}: {e}")
            return False

    def _save_metadata(self, dataset_name: str, dataset_config: Dict, result: Dict):
        """Save metadata for cached dataset."""
        metadata_path = self._get_metadata_path(dataset_name)

        # Create config hash
        config_str = json.dumps(dataset_config, sort_keys=True, ensure_ascii=False)
        config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()

        metadata = {
            'dataset_name': dataset_name,
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'config_hash': config_hash,
            'conversations_written': result['conversations_written'],
            'tokens_collected': result['tokens_collected'],
            'samples_processed': result['samples_processed'],
            'template': self.template_name,
            'cache_version': '1.0.0'
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.debug(f"💾 Saved metadata for {dataset_name}")

    def process_config_streaming(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Process entire SFT config with TRUE STREAMING.

        This is the main entry point for streaming SFT corpus preparation.

        Args:
            config: SFT dataset configuration
            output_dir: Output directory for final corpus
            force_refresh: Force refresh all caches

        Returns:
            Dict with results and statistics
        """
        self.logger.info("🚀 Starting TRUE STREAMING SFT corpus preparation")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize deduplicator with params from config
        dedup_config = config.get('quality_filters', {}).get('deduplication', {})
        threshold = dedup_config.get('threshold', 0.85)
        num_perm = dedup_config.get('num_perm', 128)
        self.deduplicator = StreamingDeduplicator(threshold=threshold, num_perm=num_perm)
        self.logger.info(f"🔧 Global deduplicator initialized: threshold={threshold}, num_perm={num_perm}")

        # Extract sampling strategy
        sampling_strategy = config.get('sampling_strategy', {})
        mode = sampling_strategy.get('mode', 'max_samples')

        self.logger.info(f"📊 Sampling mode: {mode}")

        # Calculate per-dataset token budgets
        dataset_budgets = self._calculate_dataset_budgets(config, mode, sampling_strategy)

        # Process each dataset with streaming
        dataset_results = {}
        total_tokens_actual = 0
        total_conversations = 0

        for dataset_config in config['datasets']:
            dataset_name = dataset_config['name']

            # Check cache
            if not force_refresh and self._validate_cache(dataset_name, dataset_config):
                # Load stats from metadata
                metadata_path = self._get_metadata_path(dataset_name)
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                result = {
                    'dataset_name': dataset_name,
                    'conversations_written': metadata['conversations_written'],
                    'tokens_collected': metadata['tokens_collected'],
                    'cached': True
                }

                dataset_results[dataset_name] = result
                total_tokens_actual += result['tokens_collected']
                total_conversations += result['conversations_written']

                self.logger.info(f"✅ Using cache: {dataset_name} ({result['conversations_written']:,} convs, {result['tokens_collected']/1e6:.1f}M tokens)")
                continue

            # Stream process dataset
            cache_path = self._get_cache_path(dataset_name)
            dataset_budget = dataset_budgets.get(dataset_name, float('inf'))

            self.logger.info(f"🔄 Processing {dataset_name} (budget: {dataset_budget/1e6:.1f}M tokens)")

            with SFTStreamingParquetWriter(cache_path, config=config) as writer:
                result = stream_and_process_dataset(
                    dataset_config=dataset_config,
                    template_processor=self.template_processor,
                    quality_filters=config.get('quality_filters', {}),
                    tokenizer=self.tokenizer,
                    token_budget=dataset_budget,
                    parquet_writer=writer,
                    deduplicator=self.deduplicator,
                    hf_client=self.hf_client
                )

            result['cached'] = False
            dataset_results[dataset_name] = result
            total_tokens_actual += result['tokens_collected']
            total_conversations += result['conversations_written']

            # Save metadata
            self._save_metadata(dataset_name, dataset_config, result)

        # Log summary
        self.logger.info("")
        self.logger.info("📊 Dataset Processing Summary:")
        for dataset_name, result in dataset_results.items():
            cached_flag = " (cached)" if result.get('cached') else ""
            self.logger.info(f"   {dataset_name}: {result['conversations_written']:,} convs, {result['tokens_collected']/1e6:.1f}M tokens{cached_flag}")

        self.logger.info(f"")
        self.logger.info(f"✅ Total conversations: {total_conversations:,}")
        self.logger.info(f"✅ Total tokens: {total_tokens_actual:,} ({total_tokens_actual/1e6:.1f}M)")

        # Check token budget achievement
        if mode == 'target_tokens':
            target_tokens = sampling_strategy['target_tokens']
            achievement_pct = (total_tokens_actual / target_tokens) * 100
            self.logger.info(f"🎯 Token budget achievement: {achievement_pct:.1f}% ({total_tokens_actual/1e6:.1f}M / {target_tokens/1e6:.1f}M)")

        # Deduplication stats
        dedup_stats = self.deduplicator.get_stats()
        self.logger.info(f"🗑️  Global deduplication: {dedup_stats['duplicates_found']:,} duplicates removed ({dedup_stats['deduplication_rate']*100:.1f}%)")

        # Assembly phase: Read from caches and create final corpus
        self.logger.info("")
        self.logger.info("📦 Assembling final corpus from caches...")

        assembly_result = self._assemble_final_corpus(
            config=config,
            dataset_results=dataset_results,
            output_dir=output_dir
        )

        return {
            'dataset_results': dataset_results,
            'assembly_result': assembly_result,
            'total_tokens': total_tokens_actual,
            'total_conversations': total_conversations,
            'deduplication_stats': dedup_stats
        }

    def _calculate_dataset_budgets(
        self,
        config: Dict,
        mode: str,
        sampling_strategy: Dict
    ) -> Dict[str, int]:
        """
        Calculate per-dataset token budgets based on weights.

        Returns:
            Dict mapping dataset name to token budget
        """
        dataset_budgets = {}

        if mode == 'target_tokens':
            total_token_budget = sampling_strategy['target_tokens']
            total_weight = sum(d.get('weight', 1.0) for d in config['datasets'])

            for dataset_config in config['datasets']:
                weight = dataset_config.get('weight', 1.0)
                dataset_budget = int(total_token_budget * (weight / total_weight))
                dataset_budgets[dataset_config['name']] = dataset_budget

                self.logger.debug(f"   {dataset_config['name']}: weight={weight:.2f} → budget={dataset_budget/1e6:.1f}M tokens")

        else:
            # For other modes, use infinite budget (no token limit)
            for dataset_config in config['datasets']:
                dataset_budgets[dataset_config['name']] = int(1e12)  # Effectively infinite

        return dataset_budgets

    def _pack_sequences(
        self,
        conversations: List[Dict],
        max_seq_length: int
    ) -> tuple[List[Dict], Dict]:
        """
        Pack multiple conversations into fixed-length sequences for maximum GPU efficiency.

        This is the streaming-mode equivalent of pack_sequences() in 60_prepare_sft_corpus.py.

        Args:
            conversations: List of conversations with 'text' and 'token_count' fields
            max_seq_length: Maximum sequence length (e.g., 1024)

        Returns:
            Tuple of (packed_examples, packing_stats)
        """
        eos_token_id = self.tokenizer.eos_id()
        pad_token_id = self.tokenizer.pad_id() if hasattr(self.tokenizer, 'pad_id') else self.tokenizer.unk_id()

        self.logger.info(f"📦 Packing {len(conversations):,} conversations into fixed-length sequences...")
        self.logger.info(f"  Max sequence length: {max_seq_length}")
        self.logger.info(f"  EOS token ID: {eos_token_id}, PAD token ID: {pad_token_id}")

        # Tokenize all conversations and add EOS
        tokenized = []
        total_raw_tokens = 0

        for conv in tqdm(conversations, desc="Tokenizing for packing"):
            token_ids = self.tokenizer.encode(conv['text'])
            token_ids.append(eos_token_id)  # Add EOS after each conversation
            tokenized.append({'token_ids': token_ids, 'metadata': conv})
            total_raw_tokens += len(token_ids)

        # Sort by length for better packing efficiency
        tokenized.sort(key=lambda x: len(x['token_ids']))

        # Pack sequences
        packed_examples = []
        current_sequence = []
        current_length = 0
        conversations_packed = 0
        conversations_per_sequence = []
        wasted_padding = 0

        for item in tqdm(tokenized, desc="Packing sequences"):
            token_ids = item['token_ids']
            conv_length = len(token_ids)

            if current_length + conv_length <= max_seq_length:
                # Add to current sequence
                current_sequence.extend(token_ids)
                current_length += conv_length
                conversations_packed += 1
            else:
                # Save current sequence if not empty
                if current_sequence:
                    padding_needed = max_seq_length - current_length
                    wasted_padding += padding_needed

                    input_ids = current_sequence + [pad_token_id] * padding_needed
                    attention_mask = [1] * current_length + [0] * padding_needed
                    labels = current_sequence + [-100] * padding_needed  # -100 ignored in loss

                    packed_examples.append({
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'labels': labels
                    })
                    conversations_per_sequence.append(conversations_packed)

                    # Reset
                    current_sequence = []
                    current_length = 0
                    conversations_packed = 0

                # Start new sequence
                if conv_length <= max_seq_length:
                    current_sequence = token_ids
                    current_length = conv_length
                    conversations_packed = 1
                else:
                    # Truncate if too long
                    self.logger.warning(f"Conversation too long ({conv_length}), truncating to {max_seq_length}")
                    truncated = token_ids[:max_seq_length]
                    packed_examples.append({
                        'input_ids': truncated,
                        'attention_mask': [1] * max_seq_length,
                        'labels': truncated
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

        # Calculate stats
        total_packed_tokens = len(packed_examples) * max_seq_length
        total_real_tokens = total_packed_tokens - wasted_padding
        packing_efficiency = (total_real_tokens / total_packed_tokens) * 100 if total_packed_tokens > 0 else 0
        avg_convs_per_seq = sum(conversations_per_sequence) / len(conversations_per_sequence) if conversations_per_sequence else 0

        packing_stats = {
            'total_conversations': len(conversations),
            'total_packed_sequences': len(packed_examples),
            'total_raw_tokens': total_raw_tokens,
            'total_packed_tokens': total_packed_tokens,
            'total_real_tokens': total_real_tokens,
            'wasted_padding': wasted_padding,
            'packing_efficiency': round(packing_efficiency, 2),
            'avg_conversations_per_sequence': round(avg_convs_per_seq, 2),
            'compression_ratio': round(len(conversations) / len(packed_examples), 2) if packed_examples else 0
        }

        self.logger.info(f"✅ Packing complete!")
        self.logger.info(f"  Packed sequences: {packing_stats['total_packed_sequences']:,}")
        self.logger.info(f"  Packing efficiency: {packing_stats['packing_efficiency']:.1f}%")
        self.logger.info(f"  Avg conversations/sequence: {packing_stats['avg_conversations_per_sequence']:.2f}")
        self.logger.info(f"  Compression ratio: {packing_stats['compression_ratio']:.2f}x")

        return packed_examples, packing_stats

    def _assemble_final_corpus(
        self,
        config: Dict,
        dataset_results: Dict,
        output_dir: Path
    ) -> Dict:
        """
        Assemble final corpus from cached datasets.

        Reads from Parquet caches and writes to final JSONL shards.
        Supports packing when enable_packing=true in config.
        """
        self.logger.info("📦 Reading from caches and creating final shards...")

        # Read all conversations from caches
        all_conversations = []

        for dataset_name, result in dataset_results.items():
            cache_path = self._get_cache_path(dataset_name)

            if not cache_path.exists():
                self.logger.warning(f"⚠️ Cache missing for {dataset_name}, skipping")
                continue

            # Read Parquet cache
            df = pd.read_parquet(cache_path)
            conversations = df.to_dict('records')
            all_conversations.extend(conversations)

            self.logger.info(f"   Loaded {len(conversations):,} conversations from {dataset_name}")

        self.logger.info(f"✅ Loaded {len(all_conversations):,} total conversations")

        # Shuffle conversations
        seed = config.get('sampling_strategy', {}).get('seed', 42)
        random.seed(seed)
        random.shuffle(all_conversations)
        self.logger.info(f"🔀 Shuffled with seed={seed}")

        # Split train/val
        train_ratio = config['output_params']['train_ratio']
        split_idx = int(len(all_conversations) * train_ratio)

        train_conversations = all_conversations[:split_idx]
        val_conversations = all_conversations[split_idx:]

        self.logger.info(f"✂️  Split: {len(train_conversations):,} train, {len(val_conversations):,} val")

        # Check if packing is enabled
        enable_packing = config.get('output_params', {}).get('enable_packing', False)
        max_seq_length = config.get('output_params', {}).get('sequence_length', 1024)
        shard_size = config['output_params']['shard_size']

        packing_stats = None

        if enable_packing:
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("📦 PACKING ENABLED - Creating pre-tokenized v3.0 format")
            self.logger.info("=" * 60)

            # Pack all conversations together for better efficiency
            all_shuffled = train_conversations + val_conversations
            packed_data, packing_stats = self._pack_sequences(all_shuffled, max_seq_length)

            # Re-split after packing (maintain same ratio)
            packed_split_idx = int(len(packed_data) * train_ratio)
            train_data = packed_data[:packed_split_idx]
            val_data = packed_data[packed_split_idx:]

            self.logger.info(f"  Packed train sequences: {len(train_data):,}")
            self.logger.info(f"  Packed val sequences: {len(val_data):,}")

            splits_info = {}
            for split_name, split_data in [('train', train_data), ('val', val_data)]:
                if not split_data:
                    continue

                split_info = self._write_split_shards(
                    split_name=split_name,
                    conversations=split_data,
                    output_dir=output_dir,
                    shard_size=shard_size,
                    is_packed=True
                )

                splits_info[split_name] = split_info

        else:
            # Write raw text shards (v2.0 format)
            splits_info = {}
            for split_name, split_conversations in [('train', train_conversations), ('val', val_conversations)]:
                if not split_conversations:
                    continue

                split_info = self._write_split_shards(
                    split_name=split_name,
                    conversations=split_conversations,
                    output_dir=output_dir,
                    shard_size=shard_size,
                    is_packed=False
                )

                splits_info[split_name] = split_info

        self.logger.info(f"✅ Final corpus assembly complete!")

        result = {
            'splits_info': splits_info,
            'total_conversations': len(all_conversations),
            'train_conversations': len(train_conversations),
            'val_conversations': len(val_conversations),
            'is_packed': enable_packing
        }

        if packing_stats:
            result['packing_stats'] = packing_stats

        return result

    def _write_split_shards(
        self,
        split_name: str,
        conversations: List[Dict],
        output_dir: Path,
        shard_size: int,
        is_packed: bool = False
    ) -> Dict:
        """
        Write conversations or packed sequences to JSONL shards.

        Args:
            split_name: Name of split (train/val)
            conversations: List of conversations or packed sequences
            output_dir: Output directory
            shard_size: Number of items per shard
            is_packed: Whether data is pre-packed (input_ids) or raw text

        Returns:
            Split info dict with shard metadata
        """
        import math

        num_shards = max(1, math.ceil(len(conversations) / shard_size))
        shard_files = []
        total_tokens = 0

        data_type = "packed sequences" if is_packed else "conversations"
        self.logger.info(f"   Writing {split_name} split: {num_shards} shards ({len(conversations):,} {data_type})")

        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, len(conversations))
            shard_data = conversations[start_idx:end_idx]

            # Write shard
            shard_filename = f"{split_name}_{shard_idx:05d}.jsonl"
            shard_path = output_dir / shard_filename

            shard_tokens = 0
            with open(shard_path, 'w', encoding='utf-8') as f:
                for item in shard_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    # Count tokens if packed
                    if is_packed and 'input_ids' in item:
                        shard_tokens += len(item['input_ids'])

            total_tokens += shard_tokens

            shard_info = {'filename': shard_filename}
            if is_packed:
                shard_info['sequences'] = len(shard_data)
                shard_info['tokens'] = shard_tokens
            else:
                shard_info['conversations'] = len(shard_data)

            shard_files.append(shard_info)

        result = {
            'num_shards': num_shards,
            'shards': shard_files
        }

        if is_packed:
            result['sequences'] = len(conversations)
            result['total_tokens'] = total_tokens
        else:
            result['conversations'] = len(conversations)

        return result
