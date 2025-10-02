#!/usr/bin/env python3
"""
Robust imports with intelligent fallbacks for tokenizer training.

This module provides clean, safe imports with fallback mechanisms for optional dependencies
like NLTK, spaCy, etc., replacing fragile sys.path.append() approaches.
"""

import logging
import re
from typing import List, Optional, Callable, Any

# Configure logger
logger = logging.getLogger(__name__)


class ImportFallbackManager:
    """Manages import fallbacks with intelligent degradation."""

    def __init__(self):
        self.available_libraries = {}
        self.fallback_functions = {}

    def register_library(self, name: str, import_func: Callable, fallback_func: Optional[Callable] = None):
        """Register a library with its import function and optional fallback."""
        try:
            result = import_func()
            self.available_libraries[name] = result
            logger.debug(f"✅ {name} successfully imported")
            return result
        except ImportError as e:
            logger.warning(f"⚠️  {name} not available: {e}")
            if fallback_func:
                self.fallback_functions[name] = fallback_func
                logger.info(f"🔄 Fallback registered for {name}")
            return None

    def get_library(self, name: str):
        """Get library if available, otherwise return fallback."""
        if name in self.available_libraries:
            return self.available_libraries[name]
        elif name in self.fallback_functions:
            return self.fallback_functions[name]
        else:
            raise ImportError(f"Neither {name} nor its fallback is available")


# Global fallback manager instance
fallback_manager = ImportFallbackManager()


# NLTK Import with Fallback
def _import_nltk():
    """Import NLTK and ensure punkt tokenizer is available."""
    import nltk
    from nltk.tokenize import sent_tokenize

    # Ensure punkt_tab tokenizer is downloaded (newer NLTK versions)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        logger.info("📥 Downloading NLTK punkt_tab tokenizer...")
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            # Fallback to older punkt if punkt_tab fails
            logger.info("📥 Downloading NLTK punkt tokenizer as fallback...")
            nltk.download('punkt', quiet=True)

    return {'sent_tokenize': sent_tokenize, 'nltk': nltk}


def _nltk_fallback():
    """Regex-based fallback for NLTK sentence tokenization."""

    def sent_tokenize_fallback(text: str) -> List[str]:
        """Advanced regex-based sentence segmentation."""
        if not text.strip():
            return []

        # Split on sentence endings followed by whitespace and capital letter or quote
        # Handles common abbreviations and edge cases
        pattern = r'(?<=[.!?])\s+(?=[A-Z\"\'\(\[])'
        sentences = re.split(pattern, text)

        # Also split on paragraph breaks (double newlines)
        all_sentences = []
        for sentence in sentences:
            # Split on double newlines (paragraph breaks)
            paragraphs = sentence.split('\n\n')
            for paragraph in paragraphs:
                # Split on single newlines if they contain sentence endings
                lines = paragraph.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        all_sentences.append(line)

        return all_sentences

    return {'sent_tokenize': sent_tokenize_fallback, 'nltk': None}


# Register NLTK with fallback
nltk_module = fallback_manager.register_library('nltk', _import_nltk, _nltk_fallback)


# Validation modules import
def _import_validation_modules():
    """Import validation modules from utils."""
    try:
        # Use relative imports instead of sys.path.append
        from .tokenizer_validation import TokenizerValidator
        from .precise_token_counter import PreciseTokenCounter

        return {
            'TokenizerValidator': TokenizerValidator,
            'PreciseTokenCounter': PreciseTokenCounter
        }
    except ImportError as e:
        logger.warning(f"Validation modules not available: {e}")
        return None


validation_modules = fallback_manager.register_library('validation', _import_validation_modules)


# Retry decorator for robust I/O operations
def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry functions on failure with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"🔄 Attempt {attempt + 1}/{max_retries + 1} failed: {e}")
                        logger.info(f"⏳ Retrying in {current_delay:.1f}s...")
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"❌ All {max_retries + 1} attempts failed")

            raise last_exception
        return wrapper
    return decorator


# Public API functions
def get_sentence_tokenizer():
    """Get the best available sentence tokenizer."""
    nltk_funcs = fallback_manager.get_library('nltk')
    return nltk_funcs['sent_tokenize']


def get_validation_modules():
    """Get validation modules if available."""
    return fallback_manager.get_library('validation')


def is_nltk_available() -> bool:
    """Check if NLTK is available (not using fallback)."""
    return 'nltk' in fallback_manager.available_libraries


def get_available_libraries() -> List[str]:
    """Get list of successfully imported libraries."""
    return list(fallback_manager.available_libraries.keys())


def get_fallback_libraries() -> List[str]:
    """Get list of libraries using fallbacks."""
    return list(fallback_manager.fallback_functions.keys())


def smart_sentence_segmentation(text: str) -> List[str]:
    """Intelligent sentence segmentation using best available method."""
    sent_tokenize = get_sentence_tokenizer()

    try:
        sentences = sent_tokenize(text)

        # Filter and clean sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out very short or very long sentences
            if 5 <= len(sentence) <= 2048:
                clean_sentences.append(sentence)

        return clean_sentences

    except Exception as e:
        logger.error(f"❌ Sentence segmentation failed: {e}")
        # Ultimate fallback: split on periods
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 5]
        return sentences


# Log import status on module load
def _log_import_status():
    """Log the status of all imports."""
    available = get_available_libraries()
    fallbacks = get_fallback_libraries()

    if available:
        logger.info(f"✅ Available libraries: {', '.join(available)}")
    if fallbacks:
        logger.info(f"🔄 Using fallbacks for: {', '.join(fallbacks)}")

    # Special logging for key functionality
    if is_nltk_available():
        logger.info("🎯 Using NLTK for optimal sentence segmentation")
    else:
        logger.info("⚡ Using regex fallback for sentence segmentation")


# Initialize logging when module is imported
_log_import_status()