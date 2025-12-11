"""
Evaluation utilities for SFT training.

This module provides tools for evaluating SFT models during and after training,
including qualitative generation tests and quantitative metrics.
"""

import json
import logging
import torch
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sft_templates import ConversationTemplateProcessor


class SFTEvaluator:
    """
    Evaluator for SFT models with generation and quality assessment.
    """

    def __init__(self, model, tokenizer_path: str, template_name: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize SFT evaluator.

        Args:
            model: Trained model (with or without adapters)
            tokenizer_path: Path to SentencePiece tokenizer
            template_name: Name of conversation template
            device: Device for inference
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Load tokenizer
        import sentencepiece as spm
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)

        # Initialize template processor
        self.template_processor = ConversationTemplateProcessor(template_name)

        # Get special tokens
        self.pad_token_id = self.tokenizer.pad_id() if hasattr(self.tokenizer, 'pad_id') else 0
        self.eos_token_id = self.tokenizer.eos_id()

        logging.info(f"SFTEvaluator initialized with template: {template_name}")

    def generate_response(self, prompt: str,
                         max_new_tokens: int = 128,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         top_k: int = 50,
                         do_sample: bool = True,
                         system_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate response for a given prompt.

        Args:
            prompt: User prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            do_sample: Whether to use sampling
            system_message: Optional system message

        Returns:
            Dictionary with generation results
        """
        # Format prompt with template
        formatted_prompt = self.template_processor.format_conversation(
            prompt, "", system_message
        )

        # Remove the empty response part to get just the prompt
        if self.template_processor.template_name == 'chatml':
            # Remove the empty assistant part
            formatted_prompt = formatted_prompt.rsplit('<|im_start|>assistant\n\n<|im_end|>', 1)[0]
            formatted_prompt += '<|im_start|>assistant\n'
        elif self.template_processor.template_name == 'instruct':
            formatted_prompt = formatted_prompt.rsplit('### Response:\n', 1)[0]
            formatted_prompt += '### Response:\n'
        elif self.template_processor.template_name == 'chat':
            formatted_prompt = formatted_prompt.rsplit('Assistant: ', 1)[0]
            formatted_prompt += 'Assistant: '

        # Tokenize input
        input_tokens = self.tokenizer.encode(formatted_prompt)
        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)

        # Generate
        with torch.no_grad():
            generation_config = {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'do_sample': do_sample,
                'pad_token_id': self.pad_token_id,
                'eos_token_id': self.eos_token_id,
                'use_cache': True,
                'return_dict_in_generate': True,
                'output_scores': True
            }

            outputs = self.model.generate(input_ids, **generation_config)

        # Decode generated tokens
        generated_tokens = outputs.sequences[0][len(input_tokens):].tolist()
        generated_text = self.tokenizer.decode(generated_tokens)

        # Extract clean response
        clean_response = self.template_processor.extract_response_from_generated(
            formatted_prompt + generated_text
        )

        # Calculate generation statistics
        generation_stats = {
            'input_length': len(input_tokens),
            'output_length': len(generated_tokens),
            'total_length': len(input_tokens) + len(generated_tokens)
        }

        # Calculate perplexity if scores available
        if hasattr(outputs, 'scores') and outputs.scores:
            avg_log_prob = self._calculate_average_log_probability(outputs.scores)
            generation_stats['perplexity'] = np.exp(-avg_log_prob)

        return {
            'prompt': prompt,
            'formatted_prompt': formatted_prompt,
            'generated_text': generated_text,
            'clean_response': clean_response,
            'generation_config': generation_config,
            'stats': generation_stats
        }

    def _calculate_average_log_probability(self, scores: List[torch.Tensor]) -> float:
        """Calculate average log probability from generation scores."""
        log_probs = []

        for step_scores in scores:
            # Get log probabilities
            log_probs_step = torch.log_softmax(step_scores, dim=-1)
            # We would need the actual selected tokens to get exact log prob
            # For now, use the max probability as approximation
            max_log_prob = log_probs_step.max().item()
            log_probs.append(max_log_prob)

        return np.mean(log_probs)

    def evaluate_prompts(self, prompts: List[str],
                        generation_config: Optional[Dict[str, Any]] = None,
                        batch_size: int = 4) -> List[Dict[str, Any]]:
        """
        Evaluate model on a list of prompts with optional batching.

        Args:
            prompts: List of prompts to evaluate
            generation_config: Optional generation configuration
            batch_size: Number of prompts to generate in parallel (default: 4)

        Returns:
            List of evaluation results
        """
        if generation_config is None:
            generation_config = {
                'max_new_tokens': 128,
                'temperature': 0.7,
                'top_p': 0.9,
                'do_sample': True
            }

        results = []
        logging.info(f"Evaluating {len(prompts)} prompts (batch_size={batch_size})...")

        # Process prompts in batches for faster generation
        for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            batch_results = self._generate_batch(batch_prompts, generation_config)

            for i, result in enumerate(batch_results):
                result['prompt_index'] = batch_start + i
                results.append(result)

        return results

    def _generate_batch(self, prompts: List[str], generation_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of prompts in parallel.

        Args:
            prompts: List of prompts
            generation_config: Generation configuration

        Returns:
            List of generation results
        """
        results = []

        # Format all prompts
        formatted_prompts = []
        for prompt in prompts:
            formatted_prompt = self.template_processor.format_conversation(prompt, "", None)

            # Remove the empty response part
            if self.template_processor.template_name == 'chatml':
                formatted_prompt = formatted_prompt.rsplit('<|im_start|>assistant\n\n<|im_end|>', 1)[0]
                formatted_prompt += '<|im_start|>assistant\n'
            elif self.template_processor.template_name == 'instruct':
                formatted_prompt = formatted_prompt.rsplit('### Response:\n', 1)[0]
                formatted_prompt += '### Response:\n'
            elif self.template_processor.template_name == 'chat':
                formatted_prompt = formatted_prompt.rsplit('Assistant: ', 1)[0]
                formatted_prompt += 'Assistant: '

            formatted_prompts.append(formatted_prompt)

        # Tokenize all prompts with padding
        all_tokens = [self.tokenizer.encode(fp) for fp in formatted_prompts]
        max_len = max(len(t) for t in all_tokens)

        # Pad to same length (left padding for generation)
        padded_tokens = []
        attention_masks = []
        input_lengths = []

        for tokens in all_tokens:
            pad_len = max_len - len(tokens)
            padded = [self.pad_token_id] * pad_len + tokens
            mask = [0] * pad_len + [1] * len(tokens)
            padded_tokens.append(padded)
            attention_masks.append(mask)
            input_lengths.append(len(tokens))

        input_ids = torch.tensor(padded_tokens, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self.device)

        # Generate for entire batch
        try:
            with torch.no_grad():
                gen_config = {
                    'max_new_tokens': generation_config.get('max_new_tokens', 128),
                    'temperature': generation_config.get('temperature', 0.7),
                    'top_p': generation_config.get('top_p', 0.9),
                    'top_k': generation_config.get('top_k', 50),
                    'do_sample': generation_config.get('do_sample', True),
                    'pad_token_id': self.pad_token_id,
                    'eos_token_id': self.eos_token_id,
                    'use_cache': True,
                    'attention_mask': attention_mask
                }

                outputs = self.model.generate(input_ids, **gen_config)

            # Decode each output
            for i, (prompt, formatted_prompt, input_len) in enumerate(zip(prompts, formatted_prompts, input_lengths)):
                # Extract generated tokens (skip input)
                generated_tokens = outputs[i][max_len:].tolist()
                generated_text = self.tokenizer.decode(generated_tokens)

                # Extract clean response
                clean_response = self.template_processor.extract_response_from_generated(
                    formatted_prompt + generated_text
                )

                results.append({
                    'prompt': prompt,
                    'formatted_prompt': formatted_prompt,
                    'generated_text': generated_text,
                    'clean_response': clean_response,
                    'generation_config': generation_config,
                    'stats': {
                        'input_length': input_len,
                        'output_length': len(generated_tokens),
                        'total_length': input_len + len(generated_tokens)
                    }
                })

        except Exception as e:
            logging.warning(f"Batch generation failed, falling back to sequential: {e}")
            # Fallback to sequential generation
            for i, prompt in enumerate(prompts):
                try:
                    result = self.generate_response(prompt, **generation_config)
                    results.append(result)
                except Exception as inner_e:
                    results.append({
                        'prompt': prompt,
                        'error': str(inner_e)
                    })

        return results

    def calculate_perplexity(self, conversations: List[Dict[str, Any]],
                           batch_size: int = 8) -> float:
        """
        Calculate perplexity on a set of conversations.

        Args:
            conversations: List of conversation dictionaries
            batch_size: Batch size for evaluation

        Returns:
            Average perplexity
        """
        total_log_likelihood = 0.0
        total_tokens = 0

        # Process in batches
        for i in range(0, len(conversations), batch_size):
            batch_conversations = conversations[i:i + batch_size]
            batch_texts = []

            for conv in batch_conversations:
                if 'text' in conv:
                    batch_texts.append(conv['text'])
                else:
                    # Fallback: format from prompt/response
                    formatted_text = self.template_processor.format_conversation(
                        conv['prompt'], conv['response']
                    )
                    batch_texts.append(formatted_text)

            # Tokenize batch
            batch_tokens = []
            for text in batch_texts:
                tokens = self.tokenizer.encode(text)
                batch_tokens.append(tokens)

            # Calculate log likelihood for batch
            batch_ll, batch_token_count = self._calculate_batch_log_likelihood(batch_tokens)
            total_log_likelihood += batch_ll
            total_tokens += batch_token_count

        # Calculate perplexity
        avg_log_likelihood = total_log_likelihood / total_tokens
        perplexity = np.exp(-avg_log_likelihood)

        return perplexity

    def _calculate_batch_log_likelihood(self, batch_tokens: List[List[int]]) -> Tuple[float, int]:
        """Calculate log likelihood for a batch of token sequences."""
        # Find max length for padding
        max_len = max(len(tokens) for tokens in batch_tokens)

        # Pad sequences
        padded_input_ids = []
        attention_masks = []

        for tokens in batch_tokens:
            padding_length = max_len - len(tokens)
            padded_tokens = tokens + [self.pad_token_id] * padding_length
            attention_mask = [1] * len(tokens) + [0] * padding_length

            padded_input_ids.append(padded_tokens)
            attention_masks.append(attention_mask)

        # Convert to tensors
        input_ids = torch.tensor(padded_input_ids, device=self.device)
        attention_mask = torch.tensor(attention_masks, device=self.device)

        # Calculate log likelihood
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift for causal LM loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention = attention_mask[..., 1:].contiguous()

            # Calculate log probabilities
            log_probs = torch.log_softmax(shift_logits, dim=-1)

            # Gather log probabilities for actual tokens
            batch_size, seq_len, vocab_size = log_probs.shape
            gathered_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

            # Mask out padding tokens
            masked_log_probs = gathered_log_probs * shift_attention

            # Sum log probabilities
            total_log_prob = masked_log_probs.sum().item()
            total_tokens = shift_attention.sum().item()

        return total_log_prob, total_tokens

    def quality_assessment(self, responses: List[str]) -> Dict[str, float]:
        """
        Assess the quality of generated responses.

        Args:
            responses: List of generated responses

        Returns:
            Quality metrics dictionary
        """
        metrics = {
            'avg_length': np.mean([len(response.split()) for response in responses]),
            'response_diversity': self._calculate_diversity(responses),
            'repetition_score': self._calculate_repetition_score(responses),
            'coherence_score': self._calculate_coherence_score(responses)
        }

        return metrics

    def _calculate_diversity(self, responses: List[str]) -> float:
        """Calculate lexical diversity of responses."""
        all_words = []
        for response in responses:
            words = response.lower().split()
            all_words.extend(words)

        if not all_words:
            return 0.0

        unique_words = set(all_words)
        diversity = len(unique_words) / len(all_words)

        return diversity

    def _calculate_repetition_score(self, responses: List[str]) -> float:
        """Calculate repetition score (lower is better)."""
        repetition_scores = []

        for response in responses:
            words = response.split()
            if len(words) < 2:
                repetition_scores.append(0.0)
                continue

            # Count repeated consecutive words
            consecutive_repeats = 0
            for i in range(len(words) - 1):
                if words[i] == words[i + 1]:
                    consecutive_repeats += 1

            repetition_score = consecutive_repeats / (len(words) - 1)
            repetition_scores.append(repetition_score)

        return np.mean(repetition_scores)

    def _calculate_coherence_score(self, responses: List[str]) -> float:
        """Calculate basic coherence score based on sentence structure."""
        coherence_scores = []

        for response in responses:
            sentences = response.split('.')
            coherent_sentences = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 5 and ' ' in sentence:  # Basic coherence check
                    coherent_sentences += 1

            if len(sentences) > 0:
                coherence = coherent_sentences / len(sentences)
            else:
                coherence = 0.0

            coherence_scores.append(coherence)

        return np.mean(coherence_scores)

    def save_evaluation_results(self, results: List[Dict[str, Any]],
                               output_path: str,
                               include_quality_assessment: bool = True) -> None:
        """
        Save evaluation results to file.

        Args:
            results: Evaluation results
            output_path: Output file path
            include_quality_assessment: Whether to include quality assessment
        """
        output_data = {
            'evaluation_results': results,
            'summary': {
                'total_prompts': len(results),
                'successful_generations': len([r for r in results if 'error' not in r]),
                'failed_generations': len([r for r in results if 'error' in r])
            }
        }

        if include_quality_assessment:
            successful_results = [r for r in results if 'error' not in r]
            if successful_results:
                responses = [r['clean_response'] for r in successful_results]
                quality_metrics = self.quality_assessment(responses)
                output_data['quality_assessment'] = quality_metrics

        # Save to file
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logging.info(f"Evaluation results saved to: {output_path}")


def load_evaluation_prompts(prompts_file: str) -> List[str]:
    """
    Load evaluation prompts from file.

    Args:
        prompts_file: Path to prompts file (JSON)

    Returns:
        List of prompts
    """
    with open(prompts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'prompts' in data:
        return data['prompts']
    else:
        raise ValueError("Invalid prompts file format")


def create_evaluation_report(results_file: str, output_file: str) -> None:
    """
    Create a markdown evaluation report from results.

    Args:
        results_file: Path to evaluation results JSON
        output_file: Path to output markdown report
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data['evaluation_results']
    summary = data['summary']
    quality = data.get('quality_assessment', {})

    # Create markdown report
    report = f"""# SFT Evaluation Report

## Summary
- **Total Prompts**: {summary['total_prompts']}
- **Successful Generations**: {summary['successful_generations']}
- **Failed Generations**: {summary['failed_generations']}
- **Success Rate**: {summary['successful_generations'] / summary['total_prompts'] * 100:.1f}%

"""

    if quality:
        report += f"""## Quality Assessment
- **Average Response Length**: {quality['avg_length']:.1f} words
- **Response Diversity**: {quality['response_diversity']:.3f}
- **Repetition Score**: {quality['repetition_score']:.3f} (lower is better)
- **Coherence Score**: {quality['coherence_score']:.3f}

"""

    report += """## Sample Responses

"""

    # Add sample responses
    successful_results = [r for r in results if 'error' not in r]
    for i, result in enumerate(successful_results[:10]):  # Show first 10
        report += f"""### Sample {i + 1}

**Prompt**: {result['prompt']}

**Response**: {result['clean_response']}

**Stats**: {result['stats']['output_length']} tokens generated

---

"""

    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    logging.info(f"Evaluation report saved to: {output_file}")