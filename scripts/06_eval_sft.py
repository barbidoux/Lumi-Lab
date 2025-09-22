#!/usr/bin/env python3
"""
Comprehensive evaluation script for SFT models.
Provides perplexity, BoolQ accuracy, smoke tests, and inference capabilities.
"""

import argparse
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import math

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# TRL imports for version 0.23.0+
try:
    from trl import SFTTrainer
    from trl.trainer.utils import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from trl import SFTTrainer
        from transformers import DataCollatorForLanguageModeling as DataCollatorForCompletionOnlyLM
    except ImportError:
        SFTTrainer = None
        DataCollatorForCompletionOnlyLM = None

# Import from SFT script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SFT functions with proper path
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("sft_module", os.path.join(os.path.dirname(__file__), "03_sft.py"))
    sft_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sft_module)
    create_sft_dataset = sft_module.create_sft_dataset
    validate_sp32k_tokenizer = sft_module.validate_sp32k_tokenizer
    format_prompt_response = sft_module.format_prompt_response
except ImportError as e:
    print(f"Error importing SFT functions: {e}")
    print("Please ensure 03_sft.py is in the same directory")
    sys.exit(1)


def load_sft_model(model_path: str, tokenizer_path: str, use_lora: bool = True):
    """Load SFT model and tokenizer with enhanced loading logic."""

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = validate_sp32k_tokenizer(tokenizer_path, model_path)

    print(f"Loading model from: {model_path}")

    # Determine the best model to load
    merged_path = os.path.join(model_path, "merged")
    lora_path = os.path.join(model_path, "lora_adapters")

    if use_lora and os.path.exists(lora_path):
        # Load base model + LoRA adapters
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, lora_path)
        print("✓ Loaded model with LoRA adapters")

    elif os.path.exists(merged_path):
        # Load merged model (preferred for evaluation)
        model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        print("✓ Loaded merged model")

    else:
        # Load regular model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        print("✓ Loaded base model")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    return model, tokenizer


def calculate_sft_adapted_perplexity(model, tokenizer, prompt_template: str = "chatml",
                                     max_samples: Optional[int] = None) -> Dict:
    """Calculate perplexity using the same method as 05_evaluate but with SFT format."""

    print(f"\n📈 Calculating SFT-adapted perplexity (using real text data)...")

    # Load some real text data like WikiText-2 but format it conversationally
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        print("  Using validation split")

    texts = [text.strip() for text in dataset["text"] if text.strip() and len(text.strip()) > 50]

    if max_samples and len(texts) > max_samples:
        texts = texts[:max_samples]

    print(f"  Processing {len(texts)} text samples in RAW format (like 05_evaluate)...")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    processed_samples = 0

    with torch.no_grad():
        for i, text in enumerate(texts[:50]):  # Limit to 50 for faster eval
            if not text.strip() or len(text.strip()) < 50:
                continue

            # Use RAW TEXT exactly like 05_evaluate - NO FORMATTING!
            formatted_text = text.strip()

            # Tokenize exactly like 05_evaluate
            inputs = tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                add_special_tokens=True
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Skip very short sequences
            if inputs["input_ids"].size(1) < 10:
                continue

            # Calculate loss exactly like 05_evaluate
            labels = inputs["input_ids"].clone()

            # Create attention mask like 05_evaluate
            attention_mask = torch.ones_like(inputs["input_ids"])

            outputs = model(input_ids=inputs["input_ids"], attention_mask=attention_mask, labels=labels)

            loss = outputs.loss.item()
            valid_tokens = attention_mask.sum().item()

            # Same calculation as 05_evaluate
            total_loss += loss * valid_tokens
            total_tokens += valid_tokens
            processed_samples += 1

            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/50 samples...")

    # Calculate final metrics exactly like 05_evaluate
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
    else:
        avg_loss = float('inf')
        perplexity = float('inf')

    results = {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "processed_samples": processed_samples,
        "dataset": "wikitext_raw_text",
        "template": "raw_text"
    }

    print(f"\n📊 Raw Text Perplexity Results (same as 05_evaluate):")
    print(f"  Format: Raw text (no ChatML/conversational formatting)")
    print(f"  Samples: {processed_samples:,}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Note: Uses identical calculation to 05_evaluate script")

    return results


def evaluate_boolq(model, tokenizer, prompt_template: str = "chatml",
                   max_samples: int = 100) -> Dict:
    """Evaluate model on BoolQ dataset for yes/no reasoning."""

    print(f"\n🤔 Evaluating BoolQ accuracy (max {max_samples} samples)...")

    # Load BoolQ dataset
    try:
        dataset = load_dataset("boolq", split="validation")
    except:
        print("  WARNING: BoolQ dataset not available, skipping...")
        return {"accuracy": 0.0, "samples": 0, "correct": 0}

    if len(dataset) > max_samples:
        # Random sample for fair evaluation
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = dataset.select(indices)

    print(f"  Processing {len(dataset)} BoolQ samples...")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, example in enumerate(dataset):
            passage = example["passage"]
            question = example["question"]
            label = example["answer"]  # True/False

            # Format prompt based on template
            if prompt_template == "chatml":
                prompt = f"<|im_start|>user\nBased on this passage: {passage}\n\nQuestion: {question}\nAnswer with only 'Yes' or 'No'.\n<|im_end|>\n<|im_start|>assistant\n"
            elif prompt_template == "chat":
                prompt = f"Human: Based on this passage: {passage}\n\nQuestion: {question}\nAnswer with only 'Yes' or 'No'.\n\nAssistant: "
            else:
                prompt = f"### Instruction:\nBased on this passage: {passage}\n\nQuestion: {question}\nAnswer with only 'Yes' or 'No'.\n\n### Response:\n"

            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Use the SAME method as 05_evaluate - check token probabilities!
            try:
                yes_token = tokenizer.encode("Yes", add_special_tokens=False)[0]
                no_token = tokenizer.encode("No", add_special_tokens=False)[0]
            except:
                # Fallback if encoding fails
                yes_token = tokenizer.encode("yes", add_special_tokens=False)[0]
                no_token = tokenizer.encode("no", add_special_tokens=False)[0]

            # Forward pass to get logits (like 05_evaluate)
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]  # Last token logits

            # Get probabilities for Yes/No tokens
            yes_prob = torch.softmax(logits, dim=0)[yes_token].item()
            no_prob = torch.softmax(logits, dim=0)[no_token].item()

            # Prediction based on probabilities (like 05_evaluate)
            predicted_answer = yes_prob > no_prob

            if predicted_answer == label:
                correct += 1
            total += 1

            if (i + 1) % 25 == 0:
                print(f"    Processed {i + 1}/{len(dataset)} BoolQ samples...")

    accuracy = correct / total if total > 0 else 0.0

    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "samples": total
    }

    print(f"\n📊 BoolQ Results:")
    print(f"  Samples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

    return results


def run_smoke_tests(model, tokenizer, prompt_template: str = "chatml",
                   mode: str = "standard") -> Dict:
    """Run comprehensive smoke tests using evaluation/smoke_prompts.json."""

    print(f"\n💨 Running smoke tests ({mode} mode)...")

    # Load smoke test prompts
    smoke_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation", "smoke_prompts.json")
    if not os.path.exists(smoke_file):
        print(f"  WARNING: Smoke test file not found: {smoke_file}")
        return {"summary_stats": {"avg_quality_score": 0.0}}

    with open(smoke_file, 'r', encoding='utf-8') as f:
        smoke_data = json.load(f)

    categories = smoke_data["categories"]

    # Determine how many prompts to test based on mode
    if mode == "quick":
        prompts_per_category = 1
    elif mode == "standard":
        prompts_per_category = 3
    else:  # thorough
        prompts_per_category = None  # All prompts

    all_results = []
    category_results = {}

    model.eval()

    with torch.no_grad():
        for category, prompts in categories.items():
            print(f"  Testing category: {category}")

            if prompts_per_category:
                test_prompts = prompts[:prompts_per_category]
            else:
                test_prompts = prompts

            category_scores = []

            for prompt in test_prompts:
                # Format prompt based on template
                if prompt_template == "chatml":
                    formatted_prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
                elif prompt_template == "chat":
                    formatted_prompt = f"Human: {prompt}\n\nAssistant: "
                else:
                    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

                # Generate response
                inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].size(1):],
                    skip_special_tokens=True
                ).strip()

                # Simple quality scoring
                quality_score = assess_response_quality(prompt, response)
                category_scores.append(quality_score)

                result = {
                    "category": category,
                    "prompt": prompt,
                    "response": response,
                    "quality_score": quality_score
                }
                all_results.append(result)

            category_avg = sum(category_scores) / len(category_scores) if category_scores else 0.0
            category_results[category] = {
                "avg_score": category_avg,
                "samples": len(category_scores)
            }
            print(f"    {category}: {category_avg:.2f} avg quality")

    # Calculate overall statistics
    all_scores = [r["quality_score"] for r in all_results]
    summary_stats = {
        "avg_quality_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "total_prompts": len(all_results),
        "categories_tested": len(category_results)
    }

    results = {
        "summary_stats": summary_stats,
        "category_results": category_results,
        "individual_results": all_results
    }

    print(f"\n📊 Smoke Test Results:")
    print(f"  Categories: {len(category_results)}")
    print(f"  Total prompts: {len(all_results)}")
    print(f"  Average quality: {summary_stats['avg_quality_score']:.3f}")

    return results


def assess_response_quality(prompt: str, response: str) -> float:
    """Simple heuristic quality assessment for generated responses."""

    if not response or len(response.strip()) == 0:
        return 0.0

    score = 0.5  # Base score

    # Length appropriateness
    if 10 <= len(response) <= 200:
        score += 0.1
    elif len(response) < 5:
        score -= 0.3
    elif len(response) > 300:
        score -= 0.1

    # Basic coherence checks
    if response.count('.') > 0:  # Has sentences
        score += 0.1

    # Repetition check
    words = response.lower().split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio > 0.7:
            score += 0.2
        elif unique_ratio < 0.5:
            score -= 0.2

    # Completeness check (doesn't end abruptly)
    if response.endswith(('.', '!', '?', '"', "'")):
        score += 0.1

    return min(1.0, max(0.0, score))


def run_inference(model, tokenizer, prompt: str, prompt_template: str = "chatml",
                 **generation_kwargs) -> str:
    """Run inference on a single prompt with the loaded model."""

    # Format prompt based on template
    if prompt_template == "chatml":
        formatted_prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    elif prompt_template == "chat":
        formatted_prompt = f"Human: {prompt}\n\nAssistant: "
    else:
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    # Default generation parameters
    default_kwargs = {
        "max_new_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.1
    }
    default_kwargs.update(generation_kwargs)

    # Generate
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = model.generate(
            inputs["input_ids"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **default_kwargs
        )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].size(1):],
            skip_special_tokens=True
        ).strip()

    return response


def generation_test(model, tokenizer, prompt_template: str = "chatml", num_samples: int = 5):
    """Run generation tests with sample prompts."""

    print(f"\n🎯 Generation Test ({num_samples} samples):")
    print("=" * 60)

    # Sample prompts
    test_prompts = [
        "Explain what machine learning is in simple terms.",
        "Write a short poem about the ocean.",
        "What are the benefits of renewable energy?",
        "How do you make a good first impression?",
        "Describe the process of photosynthesis."
    ]

    for i, prompt in enumerate(test_prompts[:num_samples]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {prompt}")

        # Use the inference function
        response = run_inference(model, tokenizer, prompt, prompt_template)

        print(f"Response: {response}")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive SFT Model Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to SFT model directory")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                       help="Path to tokenizer")
    parser.add_argument("--prompt_template", type=str, default="chatml",
                       choices=["instruct", "chat", "chatml"],
                       help="Prompt template used during training")

    # Model loading options
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Load LoRA adapters if available")
    parser.add_argument("--no_lora", action="store_true",
                       help="Don't load LoRA adapters")

    # Evaluation modes
    parser.add_argument("--mode", type=str, default="standard",
                       choices=["quick", "standard", "thorough"],
                       help="Evaluation mode (quick/standard/thorough)")

    # Component selection
    parser.add_argument("--perplexity", action="store_true", default=True,
                       help="Run perplexity evaluation")
    parser.add_argument("--boolq", action="store_true", default=True,
                       help="Run BoolQ evaluation")
    parser.add_argument("--smoke_tests", action="store_true", default=True,
                       help="Run smoke tests")
    parser.add_argument("--generation_test", action="store_true", default=True,
                       help="Run generation tests")
    parser.add_argument("--skip_perplexity", action="store_true",
                       help="Skip perplexity evaluation")
    parser.add_argument("--skip_boolq", action="store_true",
                       help="Skip BoolQ evaluation")
    parser.add_argument("--skip_smoke", action="store_true",
                       help="Skip smoke tests")
    parser.add_argument("--skip_generation", action="store_true",
                       help="Skip generation tests")

    # Sample limits
    parser.add_argument("--max_perplexity_samples", type=int, default=None,
                       help="Maximum samples for perplexity (None = all)")
    parser.add_argument("--max_boolq_samples", type=int, default=100,
                       help="Maximum samples for BoolQ evaluation")
    parser.add_argument("--num_gen_samples", type=int, default=5,
                       help="Number of generation test samples")

    # Inference mode
    parser.add_argument("--inference", type=str, default=None,
                       help="Run inference on a single prompt")

    # Output options
    parser.add_argument("--output_file", type=str, default=None,
                       help="Save results to JSON file")
    parser.add_argument("--assess", action="store_true",
                       help="Run performance assessment after evaluation")

    args = parser.parse_args()

    # Determine LoRA usage
    use_lora = not args.no_lora

    # Configure evaluation components based on mode and args
    if args.mode == "quick":
        args.max_perplexity_samples = args.max_perplexity_samples or 1000
        args.max_boolq_samples = min(args.max_boolq_samples, 20)
        args.num_gen_samples = min(args.num_gen_samples, 3)
    elif args.mode == "thorough":
        args.max_boolq_samples = min(args.max_boolq_samples, 500)

    # Handle inference mode
    if args.inference:
        print("=== SFT Model Inference ===")
        print(f"Model: {args.model_path}")
        print(f"Prompt: {args.inference}")

        model, tokenizer = load_sft_model(args.model_path, args.tokenizer_path, use_lora)
        response = run_inference(model, tokenizer, args.inference, args.prompt_template)

        print(f"\nResponse: {response}")
        return

    print("=== SFT Model Comprehensive Evaluation ===")
    print(f"Model: {args.model_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Template: {args.prompt_template}")
    print(f"Mode: {args.mode}")
    print(f"LoRA: {use_lora}")

    start_time = time.time()

    # Load model
    model, tokenizer = load_sft_model(args.model_path, args.tokenizer_path, use_lora)

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Results dictionary
    results = {
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "prompt_template": args.prompt_template,
        "evaluation_mode": args.mode,
        "model_parameters": total_params,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_time_seconds": None
    }

    # Run evaluations based on arguments
    if args.perplexity and not args.skip_perplexity:
        results["perplexity_metrics"] = calculate_sft_adapted_perplexity(
            model, tokenizer, args.prompt_template, max_samples=args.max_perplexity_samples
        )

    if args.boolq and not args.skip_boolq:
        results["boolq_metrics"] = evaluate_boolq(
            model, tokenizer, args.prompt_template, args.max_boolq_samples
        )
        results["boolq_accuracy"] = results["boolq_metrics"]["accuracy"]

    if args.smoke_tests and not args.skip_smoke:
        results["smoke_tests"] = run_smoke_tests(
            model, tokenizer, args.prompt_template, args.mode
        )

    if args.generation_test and not args.skip_generation:
        print("\n" + "="*60)
        generation_test(model, tokenizer, args.prompt_template, args.num_gen_samples)

    # Calculate total evaluation time
    total_time = time.time() - start_time
    results["evaluation_time_seconds"] = total_time

    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {args.output_file}")

    # Run assessment if requested
    if args.assess and args.output_file:
        try:
            assess_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation", "assess_performance.py")
            if os.path.exists(assess_script):
                print("\n" + "="*60)
                print("Running performance assessment...")
                os.system(f'python "{assess_script}" "{args.output_file}"')
        except Exception as e:
            print(f"Assessment failed: {e}")

    print(f"\n✅ Evaluation completed in {total_time:.1f} seconds!")

    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)

    if "perplexity_metrics" in results:
        ppl = results["perplexity_metrics"]["perplexity"]
        print(f"📈 Perplexity: {ppl:.2f}")

    if "boolq_accuracy" in results:
        acc = results["boolq_accuracy"]
        print(f"🤔 BoolQ Accuracy: {acc:.3f} ({acc*100:.1f}%)")

    if "smoke_tests" in results:
        quality = results["smoke_tests"]["summary_stats"]["avg_quality_score"]
        print(f"💨 Smoke Test Quality: {quality:.3f}")

    print(f"⏱️ Total Time: {total_time:.1f}s")
    print(f"🏷️ Model Size: {total_params:,} parameters")


if __name__ == "__main__":
    main()