#!/usr/bin/env python3
"""
Adaptive token estimation for tokenizer training.

This module provides intelligent, self-calibrating token estimation that adapts
to the actual content characteristics, replacing hardcoded chars_per_token ratios.
"""

import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import gzip

import sentencepiece as spm

logger = logging.getLogger(__name__)


class AdaptiveTokenEstimator:
    """Intelligent token estimation with self-calibration."""

    def __init__(self, initial_estimate: float = 4.0, min_samples: int = 1000):
        self.initial_estimate = initial_estimate
        self.current_estimate = initial_estimate
        self.min_samples = min_samples

        # Calibration history
        self.calibration_history = []
        self.sample_data = []

        # Content type detection
        self.content_characteristics = {
            'avg_word_length': 0.0,
            'punctuation_density': 0.0,
            'digit_density': 0.0,
            'whitespace_ratio': 0.0,
            'line_break_density': 0.0
        }

        logger.info(f"🎯 AdaptiveTokenEstimator initialized with {initial_estimate:.2f} chars/token")

    def calibrate_from_corpus(
        self,
        manifest: Dict[str, Any],
        corpus_dir: Path,
        sample_size: int = 1000,
        trained_tokenizer_path: Optional[Path] = None
    ) -> float:
        """
        Calibrate estimation by analyzing a sample of the corpus.

        If a trained tokenizer is available, use it for precise measurement.
        Otherwise, use intelligent heuristics based on content analysis.
        """
        logger.info(f"📊 Calibrating token estimation with {sample_size:,} document samples...")

        # Extract sample documents
        sample_docs = self._extract_sample_documents(manifest, corpus_dir, sample_size)

        if not sample_docs:
            logger.warning("⚠️  No samples extracted, keeping current estimate")
            return self.current_estimate

        # Use trained tokenizer if available
        if trained_tokenizer_path and (trained_tokenizer_path / "spm.model").exists():
            new_estimate = self._calibrate_with_tokenizer(sample_docs, trained_tokenizer_path)
        else:
            new_estimate = self._calibrate_with_heuristics(sample_docs)

        # Update estimation
        old_estimate = self.current_estimate
        self.current_estimate = new_estimate
        self.calibration_history.append({
            'timestamp': None,  # Will be set by caller
            'old_estimate': old_estimate,
            'new_estimate': new_estimate,
            'sample_size': len(sample_docs),
            'method': 'tokenizer' if trained_tokenizer_path else 'heuristic'
        })

        change_percent = ((new_estimate - old_estimate) / old_estimate) * 100
        logger.info(f"📈 Calibration complete: {old_estimate:.2f} → {new_estimate:.2f} chars/token ({change_percent:+.1f}%)")

        return new_estimate

    def _extract_sample_documents(
        self,
        manifest: Dict[str, Any],
        corpus_dir: Path,
        sample_size: int
    ) -> List[str]:
        """Extract representative sample documents from corpus."""
        sample_docs = []
        total_docs = sum(shard['num_documents'] for shard in manifest['shards'])

        # Calculate sampling rate
        sample_rate = min(1.0, sample_size / total_docs)
        docs_per_shard = max(1, int(sample_size / len(manifest['shards'])))

        logger.debug(f"📋 Extracting samples: {sample_rate:.3f} rate, ~{docs_per_shard} docs/shard")

        # More aggressive sampling to reach exact target
        remaining_needed = sample_size
        shards_remaining = len(manifest['shards'])

        for shard_idx, shard_info in enumerate(manifest['shards']):
            if len(sample_docs) >= sample_size:
                break

            # Adaptive allocation: give remaining shards their fair share
            target_from_this_shard = min(
                remaining_needed // shards_remaining if shards_remaining > 0 else remaining_needed,
                shard_info['num_documents']  # Can't extract more than available
            )

            # Handle both old and new manifest formats
            shard_relative_path = shard_info['path']
            shard_path = corpus_dir / shard_relative_path

            if not shard_path.exists():
                shard_path = corpus_dir / "shards" / shard_relative_path

            if not shard_path.exists():
                logger.warning(f"⚠️  Shard not found: {shard_path}")
                shards_remaining -= 1
                continue

            # Extract documents with systematic sampling
            try:
                with gzip.open(shard_path, 'rt', encoding='utf-8') as f:
                    shard_samples = 0
                    skip_factor = max(1, shard_info['num_documents'] // max(1, target_from_this_shard))

                    for line_num, line in enumerate(f):
                        if shard_samples >= target_from_this_shard or len(sample_docs) >= sample_size:
                            break

                        # Systematic sampling
                        if line_num % skip_factor == 0:
                            try:
                                doc = json.loads(line.strip())
                                text = doc['text']

                                # Basic quality filtering
                                if 50 <= len(text) <= 10000:  # Reasonable text length
                                    sample_docs.append(text)
                                    shard_samples += 1

                            except (json.JSONDecodeError, KeyError):
                                continue

            except Exception as e:
                logger.warning(f"⚠️  Error sampling from {shard_path}: {e}")

            finally:
                # Update counters for adaptive allocation
                remaining_needed = max(0, sample_size - len(sample_docs))
                shards_remaining -= 1

        logger.info(f"📋 Extracted {len(sample_docs):,} sample documents for calibration")
        if len(sample_docs) < sample_size:
            logger.info(f"⚠️  Only got {len(sample_docs)}/{sample_size} samples due to data availability")

        return sample_docs

    def _calibrate_with_tokenizer(self, sample_docs: List[str], tokenizer_path: Path) -> float:
        """Calibrate using actual trained tokenizer (most accurate method)."""
        logger.info("🎯 Using trained tokenizer for precise calibration...")

        try:
            # Load tokenizer
            sp = spm.SentencePieceProcessor()
            sp.load(str(tokenizer_path / "spm.model"))

            char_token_ratios = []

            for text in sample_docs:
                if not text.strip():
                    continue

                # Tokenize and measure
                tokens = sp.encode(text)
                char_count = len(text)
                token_count = len(tokens)

                if token_count > 0:
                    ratio = char_count / token_count
                    char_token_ratios.append(ratio)

            if char_token_ratios:
                # Use median for robustness against outliers
                calibrated_estimate = statistics.median(char_token_ratios)
                logger.info(f"✅ Tokenizer-based calibration: {calibrated_estimate:.3f} chars/token")
                return calibrated_estimate

        except Exception as e:
            logger.warning(f"⚠️  Tokenizer calibration failed: {e}, falling back to heuristics")

        # Fallback to heuristics
        return self._calibrate_with_heuristics(sample_docs)

    def _calibrate_with_heuristics(self, sample_docs: List[str]) -> float:
        """Calibrate using content analysis heuristics."""
        logger.info("📊 Using content analysis heuristics for calibration...")

        if not sample_docs:
            return self.current_estimate

        # Analyze content characteristics
        characteristics = self._analyze_content_characteristics(sample_docs)
        self.content_characteristics = characteristics

        # Estimate based on content type
        base_estimate = 4.0  # Standard baseline

        # Adjust based on content characteristics
        adjustments = []

        # Word length adjustment
        if characteristics['avg_word_length'] > 6:
            adjustments.append(('long_words', 0.3))  # Longer words = more chars per token
        elif characteristics['avg_word_length'] < 4:
            adjustments.append(('short_words', -0.2))  # Shorter words = fewer chars per token

        # Punctuation density adjustment
        if characteristics['punctuation_density'] > 0.1:
            adjustments.append(('high_punctuation', -0.1))  # More punctuation = more tokens
        elif characteristics['punctuation_density'] < 0.03:
            adjustments.append(('low_punctuation', 0.1))  # Less punctuation = fewer tokens

        # Digit density adjustment
        if characteristics['digit_density'] > 0.05:
            adjustments.append(('high_digits', 0.1))  # Numbers often tokenize as single tokens

        # Whitespace ratio adjustment
        if characteristics['whitespace_ratio'] > 0.25:
            adjustments.append(('high_whitespace', -0.1))  # More spaces = more word boundaries

        # Line break density adjustment
        if characteristics['line_break_density'] > 0.05:
            adjustments.append(('structured_text', -0.1))  # Structured text = more boundaries

        # Apply adjustments
        total_adjustment = sum(adj[1] for adj in adjustments)
        estimated_cpt = max(2.0, min(8.0, base_estimate + total_adjustment))  # Clamp to reasonable range

        # Log reasoning
        if adjustments:
            adj_reasons = [f"{reason}: {adj:+.1f}" for reason, adj in adjustments]
            logger.info(f"📈 Heuristic adjustments: {', '.join(adj_reasons)}")

        logger.info(f"✅ Heuristic-based calibration: {estimated_cpt:.3f} chars/token")
        return estimated_cpt

    def _analyze_content_characteristics(self, sample_docs: List[str]) -> Dict[str, float]:
        """Analyze content characteristics for heuristic estimation."""
        if not sample_docs:
            return self.content_characteristics

        total_chars = 0
        total_words = 0
        total_punctuation = 0
        total_digits = 0
        total_whitespace = 0
        total_line_breaks = 0

        for text in sample_docs:
            total_chars += len(text)

            # Count different character types
            for char in text:
                if char.isalpha():
                    pass  # Regular letters
                elif char.isdigit():
                    total_digits += 1
                elif char.isspace():
                    if char == '\n':
                        total_line_breaks += 1
                    total_whitespace += 1
                elif char in '.,!?;:':
                    total_punctuation += 1

            # Count words (rough approximation)
            words = text.split()
            total_words += len(words)

        if total_chars == 0:
            return self.content_characteristics

        # Calculate characteristics
        characteristics = {
            'avg_word_length': (total_chars - total_whitespace) / max(1, total_words),
            'punctuation_density': total_punctuation / total_chars,
            'digit_density': total_digits / total_chars,
            'whitespace_ratio': total_whitespace / total_chars,
            'line_break_density': total_line_breaks / total_chars
        }

        logger.debug(f"📊 Content analysis: {characteristics}")
        return characteristics

    def estimate_tokens_in_text(self, text: str) -> int:
        """Estimate number of tokens in text using current calibrated ratio."""
        if not text.strip():
            return 0

        return max(1, int(len(text) / self.current_estimate))

    def get_estimation_stats(self) -> Dict[str, Any]:
        """Get current estimation statistics."""
        return {
            'current_estimate': self.current_estimate,
            'initial_estimate': self.initial_estimate,
            'calibration_count': len(self.calibration_history),
            'content_characteristics': self.content_characteristics.copy(),
            'calibration_history': self.calibration_history.copy()
        }

    def predict_total_tokens(self, total_characters: int) -> int:
        """Predict total tokens for a corpus based on character count."""
        return max(1, int(total_characters / self.current_estimate))

    def get_adaptive_estimate_for_content_type(self, content_type: str = "mixed") -> float:
        """Get content-type specific estimates."""
        content_estimates = {
            "code": 3.2,        # Code has shorter tokens due to symbols
            "dialogue": 3.8,    # Conversational text
            "technical": 4.5,   # Technical writing has longer terms
            "literature": 4.2,  # Prose tends to have moderate token length
            "news": 3.9,        # News text is fairly compact
            "social": 3.5,      # Social media has abbreviations
            "mixed": self.current_estimate  # Use calibrated estimate
        }

        return content_estimates.get(content_type, self.current_estimate)


def create_adaptive_estimator(
    manifest: Dict[str, Any],
    corpus_dir: Path,
    initial_estimate: float = 4.0,
    calibration_samples: int = 1000
) -> AdaptiveTokenEstimator:
    """Create and calibrate an adaptive token estimator."""
    estimator = AdaptiveTokenEstimator(initial_estimate)

    # Perform initial calibration
    estimator.calibrate_from_corpus(manifest, corpus_dir, calibration_samples)

    return estimator