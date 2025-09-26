#!/usr/bin/env python3
"""
Precise token counter with real tokenizer validation.

This module provides exact token counting using the actual trained tokenizer,
replacing estimation with precise measurement for quality control.
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Iterator
from collections import Counter

import sentencepiece as spm
from tqdm import tqdm


class PreciseTokenCounter:
    """Precise token counting using actual tokenizer."""

    def __init__(self, tokenizer_path: Path):
        """Initialize with trained tokenizer."""
        self.tokenizer_path = tokenizer_path
        self.model_path = tokenizer_path / "spm.model"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {self.model_path}")

        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))

        logging.info(f"🔢 PreciseTokenCounter initialized with {self.sp.vocab_size():,} vocab size")

    def count_tokens_in_shards(
        self,
        manifest: Dict[str, Any],
        corpus_dir: Path,
        sample_rate: float = 1.0
    ) -> Dict[str, Any]:
        """Count exact tokens in corpus shards with progress tracking."""
        shards = manifest['shards']
        logging.info(f"🔢 Counting exact tokens in {len(shards)} shards...")

        shard_results = []
        total_exact_tokens = 0
        total_documents = 0
        total_unk_tokens = 0
        token_length_distribution = Counter()

        # Progress bar for shards
        shard_pbar = tqdm(
            shards,
            desc="📊 Processing shards",
            unit="shard",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )

        for shard_info in shard_pbar:
            shard_relative_path = shard_info['path']
            shard_path = corpus_dir / shard_relative_path

            if not shard_path.exists():
                shard_path = corpus_dir / "shards" / shard_relative_path

            shard_pbar.set_postfix_str(f"Processing {shard_path.name}")

            # Count tokens in this shard
            shard_result = self._count_shard_tokens(
                shard_path, shard_info, sample_rate
            )

            shard_results.append(shard_result)
            total_exact_tokens += shard_result['exact_tokens']
            total_documents += shard_result['documents_processed']
            total_unk_tokens += shard_result['unk_tokens']

            # Update token length distribution
            for length, count in shard_result['token_length_distribution'].items():
                token_length_distribution[length] += count

        shard_pbar.close()

        # Calculate statistics
        avg_tokens_per_doc = total_exact_tokens / max(1, total_documents)
        unk_rate = (total_unk_tokens / max(1, total_exact_tokens)) * 100

        manifest_total = sum(s['num_tokens'] for s in shards)
        accuracy = (total_exact_tokens / max(1, manifest_total)) * 100

        results = {
            'counting_method': 'precise_tokenizer',
            'sample_rate': sample_rate,
            'total_documents': total_documents,
            'total_exact_tokens': total_exact_tokens,
            'total_unk_tokens': total_unk_tokens,
            'unk_rate_percent': unk_rate,
            'avg_tokens_per_document': avg_tokens_per_doc,
            'manifest_comparison': {
                'manifest_total_tokens': manifest_total,
                'exact_total_tokens': total_exact_tokens,
                'accuracy_percent': accuracy,
                'difference': total_exact_tokens - manifest_total
            },
            'token_length_distribution': dict(token_length_distribution),
            'shard_details': shard_results
        }

        logging.info(f"🎯 Exact counting complete:")
        logging.info(f"  📄 {total_documents:,} documents processed")
        logging.info(f"  🔢 {total_exact_tokens:,} exact tokens counted")
        logging.info(f"  📊 {avg_tokens_per_doc:.1f} avg tokens/doc")
        logging.info(f"  🎯 {accuracy:.2f}% accuracy vs manifest")

        return results

    def _count_shard_tokens(
        self,
        shard_path: Path,
        shard_info: Dict[str, Any],
        sample_rate: float
    ) -> Dict[str, Any]:
        """Count exact tokens in a single shard."""
        exact_tokens = 0
        documents_processed = 0
        unk_tokens = 0
        characters_processed = 0
        token_length_distribution = Counter()

        expected_docs = shard_info['num_documents']
        docs_to_process = int(expected_docs * sample_rate) if sample_rate < 1.0 else expected_docs

        try:
            with gzip.open(shard_path, 'rt', encoding='utf-8') as f:
                # Progress bar for documents in shard
                doc_pbar = tqdm(
                    total=docs_to_process,
                    desc=f"📄 {shard_path.name}",
                    unit="doc",
                    leave=False,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]"
                )

                for line in f:
                    if documents_processed >= docs_to_process:
                        break

                    try:
                        doc = json.loads(line.strip())
                        text = doc['text']

                        # Tokenize with actual tokenizer
                        tokens = self.sp.encode(text)

                        # Count statistics
                        exact_tokens += len(tokens)
                        characters_processed += len(text)
                        documents_processed += 1

                        # Count UNK tokens
                        unk_tokens += sum(1 for token_id in tokens if token_id == self.sp.unk_id())

                        # Token length distribution
                        for token_id in tokens:
                            token_piece = self.sp.id_to_piece(token_id)
                            token_length_distribution[len(token_piece)] += 1

                        doc_pbar.update(1)
                        doc_pbar.set_postfix({
                            'tokens': f"{exact_tokens:,}",
                            'chars': f"{characters_processed:,}"
                        })

                    except json.JSONDecodeError as e:
                        logging.warning(f"Invalid JSON in {shard_path}: {e}")
                        continue

                doc_pbar.close()

        except Exception as e:
            logging.error(f"Error processing shard {shard_path}: {e}")
            raise

        return {
            'shard_path': shard_info['path'],
            'expected_documents': expected_docs,
            'documents_processed': documents_processed,
            'exact_tokens': exact_tokens,
            'unk_tokens': unk_tokens,
            'characters_processed': characters_processed,
            'manifest_tokens': shard_info['num_tokens'],
            'token_accuracy_percent': (exact_tokens / max(1, shard_info['num_tokens'])) * 100,
            'avg_tokens_per_doc': exact_tokens / max(1, documents_processed),
            'avg_chars_per_token': characters_processed / max(1, exact_tokens),
            'unk_rate_percent': (unk_tokens / max(1, exact_tokens)) * 100,
            'token_length_distribution': dict(token_length_distribution)
        }

    def validate_dataset_quality(
        self,
        manifest: Dict[str, Any],
        corpus_dir: Path,
        sample_rate: float = 0.1
    ) -> Dict[str, Any]:
        """Validate dataset quality using precise token counting."""
        logging.info(f"🔍 Validating dataset quality (sample rate: {sample_rate:.1%})...")

        # Count exact tokens
        counting_results = self.count_tokens_in_shards(manifest, corpus_dir, sample_rate)

        # Extract sample texts for validation
        sample_texts = self._extract_sample_texts(manifest, corpus_dir, max_samples=1000)

        # Additional quality checks
        quality_checks = {
            'token_count_accuracy': self._check_token_count_accuracy(counting_results),
            'unk_rate_quality': self._check_unk_rate_quality(counting_results),
            'token_distribution_health': self._check_token_distribution(counting_results),
            'sample_text_validation': self._validate_sample_texts(sample_texts)
        }

        # Overall quality score
        quality_score = self._calculate_quality_score(quality_checks, counting_results)

        return {
            'validation_metadata': {
                'sample_rate': sample_rate,
                'validation_method': 'precise_tokenizer_validation',
                'samples_analyzed': len(sample_texts)
            },
            'token_counting_results': counting_results,
            'quality_checks': quality_checks,
            'overall_quality_score': quality_score,
            'recommendations': self._generate_recommendations(quality_checks, counting_results)
        }

    def _extract_sample_texts(
        self,
        manifest: Dict[str, Any],
        corpus_dir: Path,
        max_samples: int = 1000
    ) -> List[str]:
        """Extract sample texts for validation."""
        sample_texts = []
        samples_per_shard = max_samples // len(manifest['shards'])

        for shard_info in manifest['shards']:
            shard_path = corpus_dir / shard_info['path']
            if not shard_path.exists():
                shard_path = corpus_dir / "shards" / shard_info['path']

            try:
                with gzip.open(shard_path, 'rt', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if len(sample_texts) >= max_samples:
                            break
                        if i % (shard_info['num_documents'] // max(1, samples_per_shard)) == 0:
                            try:
                                doc = json.loads(line.strip())
                                sample_texts.append(doc['text'])
                            except json.JSONDecodeError:
                                continue

            except Exception as e:
                logging.warning(f"Error sampling from {shard_path}: {e}")
                continue

            if len(sample_texts) >= max_samples:
                break

        return sample_texts

    def _check_token_count_accuracy(self, counting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check accuracy of token counting vs manifest."""
        accuracy = counting_results['manifest_comparison']['accuracy_percent']

        return {
            'accuracy_percent': accuracy,
            'status': 'excellent' if accuracy > 95 else 'good' if accuracy > 90 else 'poor',
            'difference': counting_results['manifest_comparison']['difference'],
            'recommendation': 'Token counting is accurate' if accuracy > 95 else 'Consider recalibrating token estimation'
        }

    def _check_unk_rate_quality(self, counting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check UNK token rate quality."""
        unk_rate = counting_results['unk_rate_percent']

        return {
            'unk_rate_percent': unk_rate,
            'status': 'excellent' if unk_rate < 0.1 else 'good' if unk_rate < 0.5 else 'acceptable' if unk_rate < 1.0 else 'poor',
            'recommendation': 'UNK rate is optimal' if unk_rate < 0.5 else 'Consider increasing vocabulary size or improving training data'
        }

    def _check_token_distribution(self, counting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check token length distribution health."""
        distribution = counting_results['token_length_distribution']
        total_tokens = sum(distribution.values())

        # Calculate distribution metrics
        avg_length = sum(length * count for length, count in distribution.items()) / max(1, total_tokens)
        single_char_percent = (distribution.get(1, 0) / max(1, total_tokens)) * 100

        return {
            'avg_token_length': avg_length,
            'single_char_percent': single_char_percent,
            'distribution': distribution,
            'status': 'good' if 2.5 <= avg_length <= 4.5 else 'needs_attention',
            'recommendation': 'Token length distribution is healthy' if 2.5 <= avg_length <= 4.5 else 'Token length distribution may indicate training issues'
        }

    def _validate_sample_texts(self, sample_texts: List[str]) -> Dict[str, Any]:
        """Validate sample texts with tokenizer."""
        if not sample_texts:
            return {'error': 'No sample texts available'}

        compression_ratios = []
        roundtrip_successes = 0

        for text in sample_texts[:100]:  # Limit for performance
            if not text.strip():
                continue

            # Test compression
            tokens = self.sp.encode(text)
            compression_ratio = len(tokens) / max(1, len(text))
            compression_ratios.append(compression_ratio)

            # Test roundtrip fidelity
            decoded = self.sp.decode(tokens)
            if text.strip() == decoded.strip():
                roundtrip_successes += 1

        avg_compression = sum(compression_ratios) / max(1, len(compression_ratios)) if compression_ratios else 0
        roundtrip_accuracy = (roundtrip_successes / max(1, len(sample_texts))) * 100

        return {
            'samples_tested': len(sample_texts),
            'avg_compression_ratio': avg_compression,
            'roundtrip_accuracy_percent': roundtrip_accuracy,
            'status': 'good' if roundtrip_accuracy > 95 and 0.2 <= avg_compression <= 0.4 else 'needs_attention'
        }

    def _calculate_quality_score(
        self,
        quality_checks: Dict[str, Any],
        counting_results: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score."""
        score = 0.0
        weights = {
            'token_count_accuracy': 0.3,
            'unk_rate_quality': 0.3,
            'token_distribution_health': 0.2,
            'sample_text_validation': 0.2
        }

        # Token count accuracy
        accuracy = quality_checks['token_count_accuracy']['accuracy_percent']
        score += min(100, accuracy) * weights['token_count_accuracy']

        # UNK rate (inverse score)
        unk_rate = quality_checks['unk_rate_quality']['unk_rate_percent']
        unk_score = max(0, 100 - unk_rate * 10)
        score += unk_score * weights['unk_rate_quality']

        # Token distribution
        dist_score = 100 if quality_checks['token_distribution_health']['status'] == 'good' else 70
        score += dist_score * weights['token_distribution_health']

        # Sample validation
        sample_score = 100 if quality_checks['sample_text_validation'].get('status') == 'good' else 70
        score += sample_score * weights['sample_text_validation']

        return score / sum(weights.values())

    def _generate_recommendations(
        self,
        quality_checks: Dict[str, Any],
        counting_results: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Token accuracy recommendations
        if quality_checks['token_count_accuracy']['accuracy_percent'] < 90:
            recommendations.append("Consider recalibrating token estimation methods")

        # UNK rate recommendations
        if quality_checks['unk_rate_quality']['unk_rate_percent'] > 1.0:
            recommendations.append("High UNK rate detected - consider expanding vocabulary or improving training data diversity")

        # Distribution recommendations
        if quality_checks['token_distribution_health']['status'] != 'good':
            recommendations.append("Token length distribution suggests suboptimal tokenization - review training parameters")

        # Sample validation recommendations
        if quality_checks['sample_text_validation'].get('status') != 'good':
            recommendations.append("Sample validation issues detected - review tokenizer training quality")

        if not recommendations:
            recommendations.append("Dataset quality is excellent - no immediate improvements needed")

        return recommendations