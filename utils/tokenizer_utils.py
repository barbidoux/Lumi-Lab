"""
Tokenizer utilities and wrapper classes.
"""

import torch
import sentencepiece as smp
from pathlib import Path


class SentencePieceTokenizerWrapper:
    """Simple wrapper to make SentencePiece tokenizer compatible with HuggingFace interface."""

    def __init__(self, model_path: str):
        self.sp = smp.SentencePieceProcessor()
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