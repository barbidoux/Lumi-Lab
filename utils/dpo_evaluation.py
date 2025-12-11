#!/usr/bin/env python3
"""
DPO evaluation utilities for Lumi-Lab pipeline.

This module provides comprehensive evaluation functions for DPO models:
- Reward margin and win rate calculation
- Perplexity comparison (chosen vs rejected)
- BoolQ evaluation
- Generation quality assessment
- Comparative analysis with SFT baseline
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


logger = logging.getLogger(__name__)


def calculate_sequence_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device
) -> float:
    """
    Calculate perplexity for a single sequence.

    Args:
        model: Language model
        input_ids: Token IDs
        attention_mask: Attention mask
        device: Torch device

    Returns:
        Perplexity value
    """
    model.eval()

    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        perplexity = torch.exp(loss).item()

    return perplexity


def evaluate_dpo_reward_metrics(
    model: nn.Module,
    tokenizer: Any,
    eval_dataset: Any,
    device: torch.device,
    max_samples: int = 1000,
    max_length: int = 1024
) -> Dict[str, float]:
    """
    Evaluate DPO reward metrics (margin, win rate).

    Args:
        model: DPO-trained model
        tokenizer: Tokenizer
        eval_dataset: DPO evaluation dataset with chosen/rejected pairs
        device: Torch device
        max_samples: Maximum number of samples to evaluate
        max_length: Maximum sequence length

    Returns:
        Dictionary with reward metrics
    """
    logger.info(f"📊 Evaluating DPO reward metrics on {min(max_samples, len(eval_dataset))} samples...")

    model.eval()

    chosen_losses = []
    rejected_losses = []

    num_samples = min(max_samples, len(eval_dataset))

    for i in tqdm(range(num_samples), desc="Computing rewards"):
        example = eval_dataset[i]

        # Prepare chosen sequence
        chosen_text = example['prompt'] + example['chosen']
        chosen_inputs = tokenizer(
            chosen_text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )

        # Prepare rejected sequence
        rejected_text = example['prompt'] + example['rejected']
        rejected_inputs = tokenizer(
            rejected_text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )

        # Compute losses (negative log-likelihood)
        with torch.no_grad():
            chosen_outputs = model(
                input_ids=chosen_inputs['input_ids'].to(device),
                attention_mask=chosen_inputs['attention_mask'].to(device),
                labels=chosen_inputs['input_ids'].to(device)
            )
            chosen_loss = chosen_outputs.loss.item()

            rejected_outputs = model(
                input_ids=rejected_inputs['input_ids'].to(device),
                attention_mask=rejected_inputs['attention_mask'].to(device),
                labels=rejected_inputs['input_ids'].to(device)
            )
            rejected_loss = rejected_outputs.loss.item()

        chosen_losses.append(chosen_loss)
        rejected_losses.append(rejected_loss)

    # Convert losses to rewards (negative loss)
    chosen_rewards = -np.array(chosen_losses)
    rejected_rewards = -np.array(rejected_losses)

    # Calculate metrics
    reward_margin = float(np.mean(chosen_rewards - rejected_rewards))
    win_rate = float(np.mean(chosen_rewards > rejected_rewards))
    accuracy_0_0 = float(np.mean((chosen_rewards - rejected_rewards) > 0.0))
    accuracy_0_1 = float(np.mean((chosen_rewards - rejected_rewards) > 0.1))

    metrics = {
        'reward_margin': reward_margin,
        'win_rate': win_rate,
        'accuracy_threshold_0.0': accuracy_0_0,
        'accuracy_threshold_0.1': accuracy_0_1,
        'chosen_reward_mean': float(np.mean(chosen_rewards)),
        'chosen_reward_std': float(np.std(chosen_rewards)),
        'rejected_reward_mean': float(np.mean(rejected_rewards)),
        'rejected_reward_std': float(np.std(rejected_rewards)),
    }

    logger.info(f"   Reward margin: {reward_margin:.4f}")
    logger.info(f"   Win rate: {win_rate:.2%}")

    return metrics


def evaluate_perplexity_comparison(
    model: nn.Module,
    tokenizer: Any,
    eval_dataset: Any,
    device: torch.device,
    max_samples: int = 500,
    max_length: int = 1024
) -> Dict[str, float]:
    """
    Evaluate perplexity on chosen vs rejected responses.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_dataset: DPO dataset with chosen/rejected
        device: Torch device
        max_samples: Maximum samples to evaluate
        max_length: Maximum sequence length

    Returns:
        Dictionary with perplexity metrics
    """
    logger.info(f"📈 Evaluating perplexity comparison on {min(max_samples, len(eval_dataset))} samples...")

    model.eval()

    chosen_perplexities = []
    rejected_perplexities = []

    num_samples = min(max_samples, len(eval_dataset))

    for i in tqdm(range(num_samples), desc="Computing perplexities"):
        example = eval_dataset[i]

        # Chosen perplexity
        chosen_text = example['prompt'] + example['chosen']
        chosen_inputs = tokenizer(
            chosen_text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length
        )

        chosen_ppl = calculate_sequence_perplexity(
            model,
            chosen_inputs['input_ids'],
            chosen_inputs['attention_mask'],
            device
        )
        chosen_perplexities.append(chosen_ppl)

        # Rejected perplexity
        rejected_text = example['prompt'] + example['rejected']
        rejected_inputs = tokenizer(
            rejected_text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length
        )

        rejected_ppl = calculate_sequence_perplexity(
            model,
            rejected_inputs['input_ids'],
            rejected_inputs['attention_mask'],
            device
        )
        rejected_perplexities.append(rejected_ppl)

    metrics = {
        'chosen_perplexity_mean': float(np.mean(chosen_perplexities)),
        'chosen_perplexity_median': float(np.median(chosen_perplexities)),
        'rejected_perplexity_mean': float(np.mean(rejected_perplexities)),
        'rejected_perplexity_median': float(np.median(rejected_perplexities)),
        'perplexity_improvement': float(np.mean(rejected_perplexities) - np.mean(chosen_perplexities)),
    }

    logger.info(f"   Chosen perplexity: {metrics['chosen_perplexity_mean']:.2f}")
    logger.info(f"   Rejected perplexity: {metrics['rejected_perplexity_mean']:.2f}")
    logger.info(f"   Improvement: {metrics['perplexity_improvement']:.2f}")

    return metrics


def evaluate_boolq(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    num_samples: int = 500
) -> Dict[str, float]:
    """
    Evaluate model on BoolQ task.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        device: Torch device
        num_samples: Number of samples to evaluate

    Returns:
        Dictionary with BoolQ metrics
    """
    logger.info(f"🎯 Evaluating BoolQ (binary question answering) on {num_samples} samples...")

    # Load BoolQ dataset
    try:
        boolq = load_dataset("google/boolq", split="validation")
    except Exception as e:
        logger.warning(f"Failed to load BoolQ dataset: {e}")
        return {'boolq_accuracy': 0.0, 'boolq_samples': 0}

    model.eval()

    correct = 0
    total = 0

    # Subsample if needed
    if len(boolq) > num_samples:
        indices = np.random.choice(len(boolq), size=num_samples, replace=False)
        boolq = boolq.select(indices)

    for example in tqdm(boolq, desc="BoolQ evaluation"):
        passage = example['passage']
        question = example['question']
        label = example['answer']  # True or False

        # Format prompt
        prompt = f"Passage: {passage}\n\nQuestion: {question}\n\nAnswer (true or false):"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response[len(prompt):].strip().lower()

        # Check if prediction matches label
        if label:
            prediction_correct = 'true' in response_text or 'yes' in response_text
        else:
            prediction_correct = 'false' in response_text or 'no' in response_text

        if prediction_correct:
            correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"   BoolQ accuracy: {accuracy:.2%} ({correct}/{total})")

    return {
        'boolq_accuracy': accuracy,
        'boolq_correct': correct,
        'boolq_total': total
    }


def generate_comparison_samples(
    model: nn.Module,
    tokenizer: Any,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> List[Dict[str, str]]:
    """
    Generate responses for comparison prompts.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        prompts: List of prompts
        device: Torch device
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        List of prompt/response pairs
    """
    logger.info(f"💬 Generating comparison samples for {len(prompts)} prompts...")

    model.eval()
    samples = []

    for prompt in tqdm(prompts, desc="Generating responses"):
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = response[len(prompt):].strip()

        samples.append({
            'prompt': prompt,
            'response': response_only
        })

    return samples


def run_smoke_tests(
    model: nn.Module,
    tokenizer: Any,
    device: torch.device
) -> Dict[str, Any]:
    """
    Run basic smoke tests to verify model sanity.

    Args:
        model: Model to test
        tokenizer: Tokenizer
        device: Torch device

    Returns:
        Dictionary with smoke test results
    """
    logger.info("🔬 Running smoke tests...")

    model.eval()

    test_prompts = [
        "Hello, how are you?",
        "What is 2+2?",
        "Explain photosynthesis in one sentence.",
    ]

    results = []

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = response[len(prompt):].strip()

        # Check if response is not empty and not repetitive
        is_valid = (
            len(response_only) > 5 and
            len(set(response_only.split())) > 3
        )

        results.append({
            'prompt': prompt,
            'response': response_only,
            'valid': is_valid
        })

    passed = sum(1 for r in results if r['valid'])
    total = len(results)

    logger.info(f"   Smoke tests passed: {passed}/{total}")

    return {
        'smoke_tests_passed': passed,
        'smoke_tests_total': total,
        'smoke_tests_pass_rate': passed / total if total > 0 else 0.0,
        'smoke_test_results': results
    }


def evaluate_dpo_model(
    model: nn.Module,
    tokenizer: Any,
    eval_dataset: Any,
    device: torch.device,
    eval_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Comprehensive DPO model evaluation.

    Args:
        model: DPO-trained model
        tokenizer: Tokenizer
        eval_dataset: DPO evaluation dataset
        device: Torch device
        eval_config: Evaluation configuration

    Returns:
        Dictionary with all evaluation metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("DPO MODEL EVALUATION")
    logger.info("=" * 80 + "\n")

    all_metrics = {}

    # 1. Reward metrics (margin, win rate)
    if eval_config.get('metrics', {}).get('reward_margin', True):
        reward_metrics = evaluate_dpo_reward_metrics(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            device=device,
            max_samples=eval_config.get('eval_dataset', {}).get('max_samples', 1000)
        )
        all_metrics.update(reward_metrics)

    # 2. Perplexity comparison
    if eval_config.get('metrics', {}).get('perplexity', {}).get('chosen', True):
        perplexity_metrics = evaluate_perplexity_comparison(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            device=device,
            max_samples=500
        )
        all_metrics.update(perplexity_metrics)

    # 3. BoolQ evaluation
    if eval_config.get('metrics', {}).get('boolq', True):
        boolq_metrics = evaluate_boolq(
            model=model,
            tokenizer=tokenizer,
            device=device,
            num_samples=500
        )
        all_metrics.update(boolq_metrics)

    # 4. Smoke tests
    smoke_test_results = run_smoke_tests(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    all_metrics.update(smoke_test_results)

    # 5. Generation quality samples
    if eval_config.get('comparison_prompts'):
        generation_config = eval_config.get('generation_config', {})
        comparison_samples = generate_comparison_samples(
            model=model,
            tokenizer=tokenizer,
            prompts=eval_config['comparison_prompts'],
            device=device,
            max_new_tokens=generation_config.get('max_new_tokens', 128),
            temperature=generation_config.get('temperature', 0.7),
            top_p=generation_config.get('top_p', 0.9)
        )
        all_metrics['generation_samples'] = comparison_samples

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 80 + "\n")

    return all_metrics
