#!/usr/bin/env python3
"""
Enhanced CLI metrics and logging for tokenizer training.

This module provides colorful, structured logging with real-time metrics,
progress bars, and comprehensive final reports for tokenizer training.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from collections import defaultdict
import sys

# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Apply color to text if terminal supports it."""
        if sys.stdout.isatty():
            return f"{color}{text}{cls.ENDC}"
        return text


class TokenizerTrainingMetrics:
    """Comprehensive metrics tracking for tokenizer training."""

    def __init__(self, training_name: str = "Tokenizer Training"):
        self.training_name = training_name
        self.start_time = time.time()
        self.phase_start_time = time.time()
        self._last_progress_length = 0

        # Metrics storage
        self.metrics = {
            'documents_processed': 0,
            'sentences_generated': 0,
            'characters_processed': 0,
            'estimated_tokens': 0,
            'unk_tokens': 0,
            'files_processed': 0,
            'errors_encountered': 0
        }

        # Performance tracking
        self.performance = {
            'docs_per_second': 0.0,
            'chars_per_second': 0.0,
            'tokens_per_second': 0.0,
            'avg_sentence_length': 0.0,
            'avg_tokens_per_doc': 0.0
        }

        # Phase tracking
        self.current_phase = "Initialization"
        self.phases_completed = []

        # Estimation calibration
        self.chars_per_token_history = []
        self.current_chars_per_token = 4.0

    def log_header(self, config: Dict[str, Any]):
        """Log colorful header with configuration."""
        header = f"""
{Colors.colorize('🚀 ' + self.training_name + ' v2.1', Colors.BOLD + Colors.CYAN)}
{Colors.colorize('=' * 60, Colors.BLUE)}

📁 {Colors.colorize('Corpus:', Colors.BOLD)} {config.get('corpus_path', 'N/A')}
   📊 {config.get('total_shards', 0)} shards, {config.get('total_docs', 0):,} documents

🎯 {Colors.colorize('Configuration:', Colors.BOLD)}
   📝 Vocab size: {Colors.colorize(str(config.get('vocab_size', 32000)), Colors.GREEN)}
   🔧 Model type: {Colors.colorize(config.get('model_type', 'unigram'), Colors.GREEN)}
   🎭 Character coverage: {Colors.colorize(str(config.get('character_coverage', 0.9995)), Colors.GREEN)}
   🔤 Normalization: {Colors.colorize(config.get('normalization_rule', 'nfkc'), Colors.GREEN)}

{Colors.colorize('=' * 60, Colors.BLUE)}
        """
        print(header)

    def start_phase(self, phase_name: str):
        """Start a new training phase."""
        self.current_phase = phase_name
        self.phase_start_time = time.time()

        phase_msg = f"🔄 {Colors.colorize('Starting phase:', Colors.BOLD)} {Colors.colorize(phase_name, Colors.CYAN)}"
        print(phase_msg)

    def complete_phase(self, phase_name: str):
        """Complete current phase and log duration."""
        phase_duration = time.time() - self.phase_start_time
        self.phases_completed.append({
            'name': phase_name,
            'duration': phase_duration
        })

        duration_str = self._format_duration(phase_duration)
        completion_msg = f"✅ {Colors.colorize('Completed:', Colors.GREEN)} {phase_name} {Colors.colorize(f'({duration_str})', Colors.CYAN)}"
        print(completion_msg)

    def update_metrics(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key in self.metrics:
                if isinstance(value, (int, float)):
                    self.metrics[key] += value
                else:
                    self.metrics[key] = value

        # Update performance metrics
        self._calculate_performance()

    def update_estimation(self, chars_per_token: float):
        """Update token estimation ratio."""
        self.chars_per_token_history.append(chars_per_token)
        self.current_chars_per_token = chars_per_token

    def log_progress(self, current: int, total: int, item_name: str = "items", extra_info: Optional[Dict] = None):
        """Log progress with performance metrics."""
        if total <= 0:
            return

        percentage = (current / total) * 100
        elapsed = time.time() - self.start_time

        # Calculate ETA
        if current > 0:
            rate = current / elapsed
            remaining = (total - current) / rate if rate > 0 else 0
            eta_str = self._format_duration(remaining)
        else:
            eta_str = "calculating..."

        # Progress bar
        bar_length = 20
        filled_length = int(bar_length * current // total)
        bar = '▓' * filled_length + '░' * (bar_length - filled_length)

        # Base progress message
        progress_msg = f"📊 {bar} {percentage:5.1f}% | {current:,}/{total:,} {item_name}"

        # Add performance info
        if self.performance['docs_per_second'] > 0:
            docs_per_sec = self.performance['docs_per_second']
            progress_msg += f" | {Colors.colorize(f'{docs_per_sec:.1f}/s', Colors.GREEN)}"

        progress_msg += f" | ETA: {Colors.colorize(eta_str, Colors.CYAN)}"

        # Add extra info
        if extra_info:
            extras = []
            for key, value in extra_info.items():
                if isinstance(value, float):
                    extras.append(f"{key}: {value:.2f}")
                else:
                    extras.append(f"{key}: {value}")
            if extras:
                progress_msg += f" | {' | '.join(extras)}"

        # Clear line before showing progress to avoid log interference
        self._clear_progress_line()
        print(f"\r{progress_msg}", end="", flush=True)
        self._last_progress_length = len(progress_msg)

    def _clear_progress_line(self):
        """Clear the current progress line to prevent interference with logs."""
        if self._last_progress_length > 0:
            print(f"\r{' ' * self._last_progress_length}", end="", flush=True)

    def log_metrics_update(self):
        """Log current metrics in a structured way."""
        elapsed = time.time() - self.start_time

        metrics_info = []

        if self.metrics['documents_processed'] > 0:
            docs_count = self.metrics['documents_processed']
            metrics_info.append(f"📄 Docs: {Colors.colorize(f'{docs_count:,}', Colors.GREEN)}")

        if self.metrics['sentences_generated'] > 0:
            sentences_count = self.metrics['sentences_generated']
            metrics_info.append(f"📝 Sentences: {Colors.colorize(f'{sentences_count:,}', Colors.GREEN)}")

        if self.metrics['estimated_tokens'] > 0:
            tokens_count = self.metrics['estimated_tokens']
            metrics_info.append(f"🎯 Tokens: {Colors.colorize(f'{tokens_count:,}', Colors.GREEN)}")

        if self.metrics['unk_tokens'] > 0:
            unk_rate = (self.metrics['unk_tokens'] / max(1, self.metrics['estimated_tokens'])) * 100
            color = Colors.GREEN if unk_rate < 0.1 else Colors.WARNING if unk_rate < 1.0 else Colors.FAIL
            metrics_info.append(f"❓ UNK: {Colors.colorize(f'{unk_rate:.3f}%', color)}")

        if self.current_chars_per_token != 4.0:
            metrics_info.append(f"📏 CPT: {Colors.colorize(f'{self.current_chars_per_token:.2f}', Colors.CYAN)}")

        if metrics_info:
            print(f"   📈 {' | '.join(metrics_info)}")

    def log_estimation_calibration(self, old_cpt: float, new_cpt: float, sample_size: int):
        """Log estimation calibration update."""
        change_pct = ((new_cpt - old_cpt) / old_cpt) * 100
        change_color = Colors.GREEN if abs(change_pct) < 5 else Colors.WARNING

        calibration_msg = f"""
📊 {Colors.colorize('Auto-calibrating estimation...', Colors.BOLD)}
   📝 Sample size: {sample_size:,} documents
   📏 Old estimate: {old_cpt:.2f} chars/token
   📏 New estimate: {Colors.colorize(f'{new_cpt:.2f} chars/token', Colors.GREEN)}
   📊 Change: {Colors.colorize(f'{change_pct:+.1f}%', change_color)}
   ✅ Using adaptive estimation: {Colors.colorize(f'{new_cpt:.2f} chars/token', Colors.BOLD)}
"""
        print(calibration_msg)

    def _validate_mathematical_consistency(self, training_stats: Dict[str, Any], validation_results: Optional[Dict[str, Any]]):
        """Validate mathematical consistency across all metrics."""
        warnings = []

        # Check token estimation consistency
        pipeline_tokens = training_stats.get('estimated_tokens', 0)
        corrected_tokens = training_stats.get('actual_estimated_tokens', 0)

        if corrected_tokens > 0 and pipeline_tokens > 0:
            ratio = abs(corrected_tokens - pipeline_tokens) / pipeline_tokens
            if ratio > 0.5:  # More than 50% difference
                warnings.append(f"Large estimation discrepancy: {ratio*100:.1f}%")

        # Check documents vs sentences ratio
        docs = training_stats.get('documents_processed', 0)
        sentences = training_stats.get('sentences_used', 0)

        if docs > 0 and sentences > 0:
            sentences_per_doc = sentences / docs
            if sentences_per_doc < 1.0 or sentences_per_doc > 1000:
                warnings.append(f"Unusual sentences/doc ratio: {sentences_per_doc:.1f}")

        # Validate chars/token ratio
        total_chars = training_stats.get('total_characters', 0)
        actual_cpt = training_stats.get('actual_chars_per_token', 0)

        if validation_results:
            validation_cpt = validation_results.get('compression_analysis', {}).get('avg_chars_per_token', 0)
            if actual_cpt > 0 and validation_cpt > 0:
                cpt_ratio = abs(actual_cpt - validation_cpt) / validation_cpt
                if cpt_ratio > 0.3:  # More than 30% difference
                    warnings.append(f"CPT inconsistency: {actual_cpt:.2f} vs {validation_cpt:.2f}")

        # Display warnings if any
        if warnings:
            print(f"\n⚠️  {Colors.colorize('Mathematical Consistency Warnings:', Colors.WARNING)}")
            for warning in warnings:
                print(f"   🔧 {Colors.colorize(warning, Colors.WARNING)}")
        else:
            logging.debug("✅ Mathematical consistency validation passed")

    def log_final_report(self, training_stats: Dict[str, Any], validation_results: Optional[Dict[str, Any]] = None):
        """Log comprehensive final report."""
        total_duration = time.time() - self.start_time

        # Validation debug (remove in production)
        # if validation_results:
        #     logging.debug(f"🔧 validation_results keys: {list(validation_results.keys())}")
        # else:
        #     logging.debug("🔧 validation_results is None or empty")

        # Header
        print(f"\n{Colors.colorize('🎉 Training Complete!', Colors.BOLD + Colors.GREEN)}")
        print(f"{Colors.colorize('=' * 60, Colors.GREEN)}")

        # Training Summary
        print(f"\n📊 {Colors.colorize('Training Summary:', Colors.BOLD)}")
        sentences_used = training_stats.get('sentences_used', 0)
        estimated_tokens = training_stats.get('estimated_tokens', 0)
        sentences_skipped = training_stats.get('sentences_skipped', 0)
        documents_processed = training_stats.get('documents_processed', 0)

        # Show correct values: documents != sentences
        print(f"   📄 Documents processed: {Colors.colorize(f'{documents_processed:,}', Colors.GREEN)}")
        print(f"   📝 Sentences used: {Colors.colorize(f'{sentences_used:,}', Colors.GREEN)}")

        # Show both estimates for transparency
        print(f"   🎯 Estimated tokens (pipeline): {Colors.colorize(f'{estimated_tokens:,}', Colors.WARNING)}")
        actual_estimated = training_stats.get('actual_estimated_tokens', 0)
        if actual_estimated > 0:
            print(f"   🎯 Estimated tokens (corrected): {Colors.colorize(f'{actual_estimated:,}', Colors.GREEN)}")

        print(f"   ⛔ Sentences skipped: {Colors.colorize(f'{sentences_skipped:,}', Colors.WARNING)}")

        # Scientific Cross-Validation Section
        if validation_results:
            cross_validation = validation_results.get('cross_validation_analysis', {})
            if cross_validation:
                print(f"\n🔬 {Colors.colorize('Scientific Cross-Validation:', Colors.BOLD)}")

                calibration_estimate = cross_validation.get('calibration_method_tokens', 0)
                precise_estimate = cross_validation.get('precise_counting_extrapolated_tokens', 0)
                accuracy = cross_validation.get('cross_validation_accuracy_percent', 0)
                agreement = cross_validation.get('method_agreement', 'unknown')

                agreement_color = Colors.GREEN if agreement == 'excellent' else Colors.WARNING if agreement == 'good' else Colors.FAIL

                print(f"   📊 Calibration estimate: {Colors.colorize(f'{calibration_estimate:,} tokens', Colors.WARNING)}")
                print(f"   🎯 Precise count estimate: {Colors.colorize(f'{precise_estimate:,} tokens', Colors.BOLD + Colors.GREEN)}")
                print(f"   📈 Method agreement: {Colors.colorize(f'{agreement} ({accuracy:.1f}%)', agreement_color)}")

                # Display heterogeneity analysis if available
                heterogeneity = cross_validation.get('corpus_heterogeneity_analysis', {})
                if heterogeneity:
                    normal_count = heterogeneity.get('normal_shards_count', 0)
                    anomalous_count = heterogeneity.get('anomalous_shards_count', 0)
                    anomalous_details = heterogeneity.get('anomalous_shards_details', [])

                    if anomalous_count > 0:
                        print(f"   📊 Corpus structure: {Colors.colorize(f'{normal_count} normal', Colors.GREEN)} + {Colors.colorize(f'{anomalous_count} long-form shards', Colors.WARNING)}")
                        print(f"   🔍 {Colors.colorize('Discrepancy cause:', Colors.BOLD)} Long documents under-represented in calibration sampling")

                        if anomalous_details:
                            print(f"   📚 Long-form content detected:")
                            for detail in anomalous_details[:3]:  # Show first 3
                                path_short = detail['path'].replace('.jsonl.gz', '')
                                print(f"      • {Colors.colorize(path_short, Colors.CYAN)}: {detail['docs']} docs, {detail['sentences_per_doc']:.0f} sent/doc")
                            if len(anomalous_details) > 3:
                                print(f"      • ... and {len(anomalous_details) - 3} more")

                if agreement in ['poor', 'bad']:
                    print(f"   📊 {Colors.colorize('RECOMMENDED TOKEN COUNT:', Colors.BOLD + Colors.GREEN)} {Colors.colorize(f'{precise_estimate:,}', Colors.BOLD + Colors.GREEN)} (precise method)")

        # Mathematical consistency validation
        self._validate_mathematical_consistency(training_stats, validation_results)

        # Quality Metrics
        if validation_results:
            print(f"\n🧪 {Colors.colorize('Quality Metrics:', Colors.BOLD)}")

            # Extract data - validation_results has flat structure, not nested under detailed_analysis
            coverage_analysis = validation_results.get('coverage_analysis', {})
            compression_analysis = validation_results.get('compression_analysis', {})

            # UNK rate
            unk_rate = coverage_analysis.get('unk_rate_percent', 0)
            unk_color = Colors.GREEN if unk_rate < 0.1 else Colors.WARNING if unk_rate < 0.5 else Colors.FAIL
            print(f"   ❓ UNK rate: {Colors.colorize(f'{unk_rate:.3f}%', unk_color)}")

            # Compression
            compression = compression_analysis.get('avg_chars_per_token', 0)
            comp_color = Colors.GREEN if 3.5 <= compression <= 4.5 else Colors.WARNING
            print(f"   🗜️  Compression: {Colors.colorize(f'{compression:.2f} chars/token', comp_color)}")

            # Coverage quality
            coverage_quality = coverage_analysis.get('coverage_quality', 'unknown')
            cov_color = Colors.GREEN if coverage_quality == 'excellent' else Colors.WARNING if coverage_quality == 'good' else Colors.FAIL
            print(f"   📊 Coverage: {Colors.colorize(coverage_quality, cov_color)}")

        # Performance Metrics
        print(f"\n⏱️  {Colors.colorize('Performance:', Colors.BOLD)}")
        print(f"   🕐 Duration: {Colors.colorize(self._format_duration(total_duration), Colors.CYAN)}")

        if self.performance['docs_per_second'] > 0:
            processing_rate = self.performance['docs_per_second']
            print(f"   📈 Processing rate: {Colors.colorize(f'{processing_rate:.0f} docs/s', Colors.GREEN)}")

        # Phases breakdown
        if self.phases_completed:
            print(f"\n📋 {Colors.colorize('Phase Breakdown:', Colors.BOLD)}")
            for phase in self.phases_completed:
                duration_str = self._format_duration(phase['duration'])
                print(f"   ✅ {phase['name']}: {Colors.colorize(duration_str, Colors.CYAN)}")

        # Estimation accuracy
        if len(self.chars_per_token_history) > 1:
            print(f"\n📏 {Colors.colorize('Estimation Evolution:', Colors.BOLD)}")
            print(f"   📊 Initial: 4.0 chars/token")
            print(f"   📊 Final: {Colors.colorize(f'{self.current_chars_per_token:.2f} chars/token', Colors.GREEN)}")

            adaptation = abs(4.0 - self.current_chars_per_token) / 4.0 * 100
            adapt_color = Colors.GREEN if adaptation > 5 else Colors.CYAN
            print(f"   📈 Adaptation: {Colors.colorize(f'{adaptation:.1f}%', adapt_color)} improvement")

        print(f"\n{Colors.colorize('=' * 60, Colors.GREEN)}")

    def _calculate_performance(self):
        """Calculate performance metrics."""
        elapsed = time.time() - self.start_time

        if elapsed > 0:
            self.performance['docs_per_second'] = self.metrics['documents_processed'] / elapsed
            self.performance['chars_per_second'] = self.metrics['characters_processed'] / elapsed
            self.performance['tokens_per_second'] = self.metrics['estimated_tokens'] / elapsed

        if self.metrics['documents_processed'] > 0:
            self.performance['avg_tokens_per_doc'] = self.metrics['estimated_tokens'] / self.metrics['documents_processed']

        if self.metrics['sentences_generated'] > 0:
            self.performance['avg_sentence_length'] = self.metrics['characters_processed'] / self.metrics['sentences_generated']

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


# Convenience functions
def setup_enhanced_logging(level: str = "INFO"):
    """Setup enhanced logging with colors."""
    # Configure basic logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add color to log levels if terminal supports it
    if sys.stdout.isatty():
        logging.addLevelName(logging.DEBUG, Colors.colorize('DEBUG', Colors.CYAN))
        logging.addLevelName(logging.INFO, Colors.colorize('INFO', Colors.GREEN))
        logging.addLevelName(logging.WARNING, Colors.colorize('WARNING', Colors.WARNING))
        logging.addLevelName(logging.ERROR, Colors.colorize('ERROR', Colors.FAIL))
        logging.addLevelName(logging.CRITICAL, Colors.colorize('CRITICAL', Colors.FAIL + Colors.BOLD))