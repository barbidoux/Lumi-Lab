#!/usr/bin/env python3
"""
Advanced tokenizer validation and quality control module.

This module provides comprehensive quantitative analysis of tokenizer quality,
including coverage analysis, compression efficiency, and statistical validation.
"""

import json
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
import re
import math

import sentencepiece as spm


class TokenizerValidator:
    """Comprehensive tokenizer validation and quality assessment."""

    def __init__(self, tokenizer_path: Path):
        """Initialize validator with trained tokenizer."""
        self.tokenizer_path = tokenizer_path
        self.model_path = tokenizer_path / "spm.model"
        self.vocab_path = tokenizer_path / "spm.vocab"

        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))

        # Load vocabulary
        self.vocabulary = self._load_vocabulary()

        logging.info(f"🔍 TokenizerValidator initialized with vocab size: {len(self.vocabulary):,}")

    def _load_vocabulary(self) -> Dict[int, Dict[str, Any]]:
        """Load and parse vocabulary file."""
        vocabulary = {}

        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        token = parts[0]
                        score = float(parts[1]) if len(parts) > 1 else 0.0

                        vocabulary[line_num] = {
                            'token': token,
                            'score': score,
                            'id': line_num,
                            'length': len(token),
                            'is_special': token.startswith('<') and token.endswith('>'),
                            'is_subword': token.startswith('▁'),
                            'char_count': len(token.replace('▁', ''))
                        }
        except FileNotFoundError:
            logging.warning(f"Vocabulary file not found: {self.vocab_path}")
            # Fallback: generate vocabulary from model
            for i in range(self.sp.vocab_size()):
                token = self.sp.id_to_piece(i)
                vocabulary[i] = {
                    'token': token,
                    'score': 0.0,
                    'id': i,
                    'length': len(token),
                    'is_special': token.startswith('<') and token.endswith('>'),
                    'is_subword': token.startswith('▁'),
                    'char_count': len(token.replace('▁', ''))
                }

        return vocabulary

    def comprehensive_validation(self, test_texts: List[str]) -> Dict[str, Any]:
        """Perform comprehensive tokenizer validation."""
        logging.info("🧪 Starting comprehensive tokenizer validation...")

        # Perform each analysis once and store results
        vocabulary_analysis = self._analyze_vocabulary()
        coverage_analysis = self._analyze_coverage(test_texts)
        compression_analysis = self._analyze_compression(test_texts)
        statistical_analysis = self._analyze_statistics(test_texts)
        linguistic_analysis = self._analyze_linguistic_patterns(test_texts)

        results = {
            'vocabulary_analysis': vocabulary_analysis,
            'coverage_analysis': coverage_analysis,
            'compression_analysis': compression_analysis,
            'statistical_analysis': statistical_analysis,
            'linguistic_analysis': linguistic_analysis
        }

        # Calculate quality metrics using existing analysis results (no duplication)
        results['quality_metrics'] = self._calculate_quality_metrics_from_results(results, test_texts)

        # Calculate overall quality score
        results['overall_quality_score'] = self._calculate_overall_score(results)

        logging.info(f"✅ Validation complete. Overall quality score: {results['overall_quality_score']:.2f}/100")

        return results

    def _analyze_vocabulary(self) -> Dict[str, Any]:
        """Analyze vocabulary composition and statistics."""
        logging.info("📊 Analyzing vocabulary composition...")

        special_tokens = [v for v in self.vocabulary.values() if v['is_special']]
        subword_tokens = [v for v in self.vocabulary.values() if v['is_subword']]
        regular_tokens = [v for v in self.vocabulary.values() if not v['is_special']]

        token_lengths = [v['length'] for v in regular_tokens]
        char_counts = [v['char_count'] for v in regular_tokens]

        return {
            'total_vocab_size': len(self.vocabulary),
            'special_tokens': len(special_tokens),
            'subword_tokens': len(subword_tokens),
            'regular_tokens': len(regular_tokens),
            'avg_token_length': statistics.mean(token_lengths) if token_lengths else 0,
            'median_token_length': statistics.median(token_lengths) if token_lengths else 0,
            'avg_char_count': statistics.mean(char_counts) if char_counts else 0,
            'token_length_distribution': {
                'min': min(token_lengths) if token_lengths else 0,
                'max': max(token_lengths) if token_lengths else 0,
                'std': statistics.stdev(token_lengths) if len(token_lengths) > 1 else 0
            },
            'special_token_list': [v['token'] for v in special_tokens]
        }

    def _analyze_coverage(self, test_texts: List[str]) -> Dict[str, Any]:
        """Analyze tokenizer coverage and UNK rate."""
        logging.info("🎯 Analyzing tokenizer coverage...")

        total_tokens = 0
        unk_tokens = 0
        unique_chars = set()
        char_coverage = Counter()
        token_usage = Counter()

        for text in test_texts:
            # Character analysis
            for char in text:
                unique_chars.add(char)
                char_coverage[char] += 1

            # Token analysis
            tokens = self.sp.encode(text)
            total_tokens += len(tokens)

            for token_id in tokens:
                token_usage[token_id] += 1
                if token_id == self.sp.unk_id():
                    unk_tokens += 1

        # Calculate coverage metrics
        unk_rate = (unk_tokens / max(1, total_tokens)) * 100
        vocab_utilization = (len(token_usage) / len(self.vocabulary)) * 100

        # Most and least used tokens
        most_used = token_usage.most_common(10)
        least_used_vocab = set(range(len(self.vocabulary))) - set(token_usage.keys())

        return {
            'total_tokens_analyzed': total_tokens,
            'unk_tokens': unk_tokens,
            'unk_rate_percent': unk_rate,
            'unique_characters': len(unique_chars),
            'vocab_utilization_percent': vocab_utilization,
            'unused_vocab_tokens': len(least_used_vocab),
            'most_used_tokens': [
                {
                    'token_id': tid,
                    'token': self.sp.id_to_piece(tid),
                    'count': count,
                    'frequency_percent': (count / total_tokens) * 100
                }
                for tid, count in most_used
            ],
            'character_diversity': len(unique_chars),
            'coverage_quality': 'excellent' if unk_rate < 0.1 else 'good' if unk_rate < 1.0 else 'needs_improvement'
        }

    def _analyze_compression(self, test_texts: List[str]) -> Dict[str, Any]:
        """Analyze compression efficiency and token statistics."""
        logging.info("📈 Analyzing compression efficiency...")

        compression_ratios = []
        tokens_per_char = []
        chars_per_token = []

        for text in test_texts:
            if not text.strip():
                continue

            tokens = self.sp.encode(text)
            char_count = len(text)
            token_count = len(tokens)

            if char_count > 0 and token_count > 0:
                compression_ratio = token_count / char_count
                compression_ratios.append(compression_ratio)
                tokens_per_char.append(token_count / char_count)
                chars_per_token.append(char_count / token_count)

        if not compression_ratios:
            return {'error': 'No valid texts for compression analysis'}

        return {
            'avg_compression_ratio': statistics.mean(compression_ratios),
            'median_compression_ratio': statistics.median(compression_ratios),
            'avg_tokens_per_char': statistics.mean(tokens_per_char),
            'avg_chars_per_token': statistics.mean(chars_per_token),
            'compression_efficiency': {
                'excellent': sum(1 for r in compression_ratios if r < 0.2),
                'good': sum(1 for r in compression_ratios if 0.2 <= r < 0.3),
                'acceptable': sum(1 for r in compression_ratios if 0.3 <= r < 0.4),
                'poor': sum(1 for r in compression_ratios if r >= 0.4)
            },
            'compression_variability': statistics.stdev(compression_ratios) if len(compression_ratios) > 1 else 0
        }

    def _analyze_statistics(self, test_texts: List[str]) -> Dict[str, Any]:
        """Analyze statistical properties of tokenization."""
        logging.info("📊 Analyzing tokenization statistics...")

        all_tokens = []
        sequence_lengths = []

        for text in test_texts:
            tokens = self.sp.encode(text)
            all_tokens.extend(tokens)
            sequence_lengths.append(len(tokens))

        token_frequency = Counter(all_tokens)

        return {
            'total_sequences': len(test_texts),
            'total_tokens': len(all_tokens),
            'unique_tokens_used': len(token_frequency),
            'avg_sequence_length': statistics.mean(sequence_lengths) if sequence_lengths else 0,
            'median_sequence_length': statistics.median(sequence_lengths) if sequence_lengths else 0,
            'sequence_length_stats': {
                'min': min(sequence_lengths) if sequence_lengths else 0,
                'max': max(sequence_lengths) if sequence_lengths else 0,
                'std': statistics.stdev(sequence_lengths) if len(sequence_lengths) > 1 else 0
            },
            'token_frequency_distribution': {
                'most_frequent': token_frequency.most_common(10),
                'singleton_tokens': sum(1 for count in token_frequency.values() if count == 1),
                'frequency_entropy': self._calculate_entropy([count for count in token_frequency.values()])
            }
        }

    def _analyze_linguistic_patterns(self, test_texts: List[str]) -> Dict[str, Any]:
        """Analyze linguistic patterns in tokenization."""
        logging.info("🔤 Analyzing linguistic patterns...")

        word_boundary_preservation = 0
        total_words = 0
        subword_splits = 0

        for text in test_texts:
            # Analyze word boundary preservation
            words = re.findall(r'\b\w+\b', text.lower())
            total_words += len(words)

            for word in words:
                tokens = self.sp.encode(word)
                decoded = self.sp.decode(tokens)

                if word == decoded.strip():
                    word_boundary_preservation += 1

                if len(tokens) > 1:
                    subword_splits += 1

        word_preservation_rate = (word_boundary_preservation / max(1, total_words)) * 100
        subword_usage_rate = (subword_splits / max(1, total_words)) * 100

        return {
            'word_boundary_preservation_rate': word_preservation_rate,
            'subword_usage_rate': subword_usage_rate,
            'total_words_analyzed': total_words,
            'linguistic_quality': 'excellent' if word_preservation_rate > 80 else 'good' if word_preservation_rate > 60 else 'needs_improvement'
        }

    def _calculate_quality_metrics_from_results(self, results: Dict[str, Any], test_texts: List[str]) -> Dict[str, float]:
        """Calculate overall quality metrics from existing analysis results."""
        logging.info("🎯 Calculating quality metrics...")

        total_score = 0
        metrics = {}

        # Coverage score (30% weight)
        coverage_data = results['coverage_analysis']
        coverage_score = max(0, 100 - coverage_data['unk_rate_percent'] * 10)
        metrics['coverage_score'] = coverage_score
        total_score += coverage_score * 0.3

        # Compression score (25% weight)
        compression_data = results['compression_analysis']
        if 'avg_compression_ratio' in compression_data:
            # Optimal compression ratio is around 0.25 (4 chars per token)
            compression_score = max(0, 100 - abs(compression_data['avg_compression_ratio'] - 0.25) * 200)
            metrics['compression_score'] = compression_score
            total_score += compression_score * 0.25

        # Vocabulary efficiency score (20% weight)
        vocab_data = results['vocabulary_analysis']
        vocab_score = min(100, vocab_data['regular_tokens'] / 320)  # Normalize to 32k vocab
        metrics['vocabulary_efficiency_score'] = vocab_score
        total_score += vocab_score * 0.2

        # Linguistic quality score (25% weight)
        linguistic_data = results['linguistic_analysis']
        linguistic_score = linguistic_data['word_boundary_preservation_rate']
        metrics['linguistic_score'] = linguistic_score
        total_score += linguistic_score * 0.25

        metrics['weighted_total_score'] = total_score

        return metrics

    def _calculate_quality_metrics(self, test_texts: List[str]) -> Dict[str, float]:
        """Legacy method - retained for compatibility."""
        logging.warning("⚠️ Using deprecated _calculate_quality_metrics method. Use _calculate_quality_metrics_from_results instead.")

        # Sample a subset for detailed analysis
        sample_texts = test_texts[:1000] if len(test_texts) > 1000 else test_texts

        total_score = 0
        metrics = {}

        # Coverage score (30% weight)
        coverage_data = self._analyze_coverage(sample_texts)
        coverage_score = max(0, 100 - coverage_data['unk_rate_percent'] * 10)
        metrics['coverage_score'] = coverage_score
        total_score += coverage_score * 0.3

        # Compression score (25% weight)
        compression_data = self._analyze_compression(sample_texts)
        if 'avg_compression_ratio' in compression_data:
            # Optimal compression ratio is around 0.25 (4 chars per token)
            compression_score = max(0, 100 - abs(compression_data['avg_compression_ratio'] - 0.25) * 200)
            metrics['compression_score'] = compression_score
            total_score += compression_score * 0.25

        # Vocabulary efficiency score (20% weight)
        vocab_data = self._analyze_vocabulary()
        vocab_score = min(100, vocab_data['regular_tokens'] / 320)  # Normalize to 32k vocab
        metrics['vocabulary_efficiency_score'] = vocab_score
        total_score += vocab_score * 0.2

        # Linguistic quality score (25% weight)
        linguistic_data = self._analyze_linguistic_patterns(sample_texts)
        linguistic_score = linguistic_data['word_boundary_preservation_rate']
        metrics['linguistic_score'] = linguistic_score
        total_score += linguistic_score * 0.25

        metrics['weighted_total_score'] = total_score

        return metrics

    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall tokenizer quality score."""
        return results['quality_metrics']['weighted_total_score']

    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate Shannon entropy of a distribution."""
        if not values:
            return 0.0

        total = sum(values)
        probabilities = [v / total for v in values if v > 0]

        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        return entropy

    def generate_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Generate comprehensive validation report."""
        logging.info(f"📋 Generating validation report: {output_path}")

        report = {
            'tokenizer_validation_report': {
                'metadata': {
                    'tokenizer_path': str(self.tokenizer_path),
                    'vocab_size': len(self.vocabulary),
                    'validation_timestamp': None  # Will be set by caller
                },
                'executive_summary': {
                    'overall_quality_score': results['overall_quality_score'],
                    'quality_grade': self._get_quality_grade(results['overall_quality_score']),
                    'key_strengths': self._identify_strengths(results),
                    'areas_for_improvement': self._identify_improvements(results)
                },
                'detailed_analysis': results
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logging.info(f"✅ Validation report saved: {output_path}")

    def _get_quality_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90: return "A (Excellent)"
        elif score >= 80: return "B (Good)"
        elif score >= 70: return "C (Acceptable)"
        elif score >= 60: return "D (Needs Improvement)"
        else: return "F (Poor)"

    def _identify_strengths(self, results: Dict[str, Any]) -> List[str]:
        """Identify tokenizer strengths."""
        strengths = []

        coverage = results['coverage_analysis']
        if coverage['unk_rate_percent'] < 0.5:
            strengths.append("Excellent coverage with very low UNK rate")

        compression = results['compression_analysis']
        if 'avg_chars_per_token' in compression and 3.5 <= compression['avg_chars_per_token'] <= 4.5:
            strengths.append("Optimal compression efficiency")

        linguistic = results['linguistic_analysis']
        if linguistic['word_boundary_preservation_rate'] > 80:
            strengths.append("Good linguistic pattern preservation")

        return strengths

    def _identify_improvements(self, results: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement."""
        improvements = []

        coverage = results['coverage_analysis']
        if coverage['unk_rate_percent'] > 1.0:
            improvements.append("High UNK rate indicates coverage issues")

        if coverage['vocab_utilization_percent'] < 80:
            improvements.append("Low vocabulary utilization - consider smaller vocab")

        linguistic = results['linguistic_analysis']
        if linguistic['word_boundary_preservation_rate'] < 60:
            improvements.append("Poor word boundary preservation")

        return improvements