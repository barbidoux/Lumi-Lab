"""
Tokenizer utilities and wrapper classes.
"""

import os
import sys
import json
import torch
import sentencepiece as spm
from pathlib import Path
from transformers import AutoTokenizer, LlamaTokenizer


def validate_sp32k_tokenizer(tokenizer_path: str, model_path: str = None) -> AutoTokenizer:
    """
    Load and validate SP32k tokenizer using model metadata.

    Args:
        tokenizer_path: Path to tokenizer (.model file or HF directory)
        model_path: Optional path to model for metadata validation

    Returns:
        Validated AutoTokenizer instance
    """
    if not tokenizer_path:
        print("ERROR: --tokenizer_path is required. No fallback tokenizer allowed.")
        print("Please provide the path to your SP32k tokenizer from pretraining.")
        sys.exit(1)

    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer path does not exist: {tokenizer_path}")
        sys.exit(1)

    # Load tokenizer metadata from model if available
    tokenizer_metadata = None
    if model_path:
        metadata_path = os.path.join(model_path, "tokenizer_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    tokenizer_metadata = data.get("tokenizer_metadata", {})
                print(f"✓ Loaded tokenizer metadata from {metadata_path}")
                print(f"  Vocab size: {tokenizer_metadata.get('tokenizer_vocab_size', 'unknown')}")
                print(f"  Special tokens: {tokenizer_metadata.get('special_tokens', {})}")
            except Exception as e:
                print(f"WARNING: Could not load tokenizer metadata: {e}")

    try:
        # Load SentencePiece model to validate
        if tokenizer_path.endswith('.model'):
            sp_model = spm.SentencePieceProcessor()
            sp_model.load(tokenizer_path)

            vocab_size = sp_model.get_piece_size()
            print(f"Loaded SentencePiece model with vocab size: {vocab_size}")

            # Validate against metadata if available
            if tokenizer_metadata and 'tokenizer_vocab_size' in tokenizer_metadata:
                expected_vocab_size = tokenizer_metadata['tokenizer_vocab_size']
                if vocab_size != expected_vocab_size:
                    print(f"ERROR: Vocab size mismatch!")
                    print(f"Expected: {expected_vocab_size}, Got: {vocab_size}")
                    print("This tokenizer doesn't match the one used for pretraining.")
                    sys.exit(1)

            # Create tokenizer using SentencePiece directly with known config
            tokenizer = LlamaTokenizer(vocab_file=tokenizer_path, legacy=True)

            # Set special tokens based on metadata or defaults
            if tokenizer_metadata and 'special_tokens' in tokenizer_metadata:
                special_tokens = tokenizer_metadata['special_tokens']
                tokenizer.pad_token_id = special_tokens.get('pad', 0)
                tokenizer.unk_token_id = special_tokens.get('unk', 1)
                tokenizer.bos_token_id = special_tokens.get('bos', 2)
                tokenizer.eos_token_id = special_tokens.get('eos', 3)
            else:
                # Default SP32k mapping
                tokenizer.pad_token_id = 0
                tokenizer.unk_token_id = 1
                tokenizer.bos_token_id = 2
                tokenizer.eos_token_id = 3

            print("✓ Created HF tokenizer from SentencePiece model")
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    except Exception as e:
        print(f"ERROR: Failed to load tokenizer from {tokenizer_path}")
        print(f"Error: {e}")
        sys.exit(1)

    # Validate special tokens mapping
    expected_special_tokens = {
        "pad": 0,
        "unk": 1,
        "bos": 2,
        "eos": 3
    }

    actual_tokens = {
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id
    }

    print(f"Special tokens mapping: {actual_tokens}")

    # Check if mapping matches expected SP32k format
    for token_name, expected_id in expected_special_tokens.items():
        actual_id = actual_tokens[token_name]
        if actual_id != expected_id:
            print(f"ERROR: Special token '{token_name}' mismatch!")
            print(f"Expected: {expected_id}, Got: {actual_id}")
            print("This tokenizer is not compatible with SP32k pretraining.")
            sys.exit(1)

    print("✓ Tokenizer validation passed - SP32k compatible")
    return tokenizer


class SentencePieceTokenizerWrapper:
    """Simple wrapper to make SentencePiece tokenizer compatible with HuggingFace interface."""

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.eos_token = '</s>'
        self.pad_token = '</s>'
        self.eos_token_id = self.sp.piece_to_id('</s>')
        self.pad_token_id = self.eos_token_id
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str, add_special_tokens=True, **kwargs):
        """Encode text to token IDs."""
        return self.sp.encode_as_ids(text)

    def decode(self, ids, skip_special_tokens=False):
        """Decode token IDs to text using proper SentencePiece API."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # Filter out special tokens
        if skip_special_tokens:
            # Define special tokens to filter
            special_tokens = {
                0,  # PAD (usually)
                1,  # UNK
                2,  # BOS (if exists)
                self.pad_token_id,
                self.eos_token_id
            }

            # Remove special tokens and stop at EOS if present
            filtered_ids = []
            for token_id in ids:
                if token_id in special_tokens:
                    if token_id == self.eos_token_id:  # Stop at EOS
                        break
                    continue  # Skip other special tokens
                filtered_ids.append(token_id)
            ids = filtered_ids
        else:
            # Always filter UNK tokens (ID=1) as they produce garbage
            ids = [token_id for token_id in ids if token_id != 1]

        # Use SentencePiece's decode method directly
        try:
            decoded = self.sp.decode(ids)
            return decoded
        except Exception:
            # Fallback to decode_ids if decode fails
            return self.sp.decode_ids(ids)

    def __call__(self, text, return_tensors=None, max_length=None, truncation=False, **kwargs):
        """Tokenize text and return in requested format."""
        # Simple encoding
        input_ids = self.encode(text)

        # Truncate if needed
        if truncation and max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        # Return as tensors if requested
        if return_tensors == "pt":
            input_ids = torch.tensor([input_ids], dtype=torch.long)  # Add batch dimension
            attention_mask = torch.ones_like(input_ids)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

        return {'input_ids': input_ids}