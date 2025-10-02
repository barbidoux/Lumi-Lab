#!/usr/bin/env python3
"""
Final model evaluation script.
Calculates perplexity, performs zero-shot benchmarks and smoke-tests.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sentencepiece as spm

from utils.dataset_utils import PackedDataset
from utils.model_utils import load_pretrained_model
from utils.tokenizer_utils import SentencePieceTokenizerWrapper


def validate_tokenizer_consistency_eval(data_dirs: List[str]) -> Dict:
    """Validate that all datasets use the same tokenizer via SHA256 hash (modern approach)."""
    if not data_dirs:
        return {}

    print("🔍 Validating tokenizer consistency across datasets...")

    reference_hash = None
    reference_dir = None

    for data_dir in data_dirs:
        # Use new final_manifest.json format
        manifest_path = Path(data_dir) / 'final_manifest.json'

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"❌ Final manifest file not found: {manifest_path}\n"
                f"This dataset may not have been processed with the packed format.\n"
                f"Please re-process the dataset."
            )

        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load manifest {manifest_path}: {e}")

        if 'tokenizer_config_hash' not in manifest:
            raise ValueError(
                f"❌ No tokenizer_config_hash in manifest: {manifest_path}\n"
                f"This dataset was processed with an older version.\n"
                f"Please re-process the dataset."
            )

        tokenizer_hash = manifest['tokenizer_config_hash']

        if reference_hash is None:
            reference_hash = tokenizer_hash
            reference_dir = data_dir
            print(f"📊 Reference tokenizer hash (from {data_dir}): {reference_hash[:16]}...")
            if 'statistics' in manifest and 'vocab_size' in manifest['statistics']:
                print(f"   Vocab size: {manifest['statistics']['vocab_size']:,}")
        else:
            if tokenizer_hash != reference_hash:
                raise ValueError(
                    f"❌ Tokenizer mismatch between {reference_dir} and {data_dir}:\n"
                    f"   Reference hash: {reference_hash}\n"
                    f"   Current hash:   {tokenizer_hash}\n\n"
                    f"All datasets must use the same tokenizer. Please re-process datasets."
                )

            print(f"✅ {data_dir}: tokenizer hash matches reference")

    print(f"✅ All {len(data_dirs)} datasets use consistent tokenizer")
    return {"tokenizer_config_hash": reference_hash}


def evaluate_packed_perplexity(model, tokenizer, data_dir: str, split: str = "val", batch_size: int = 8) -> Dict[str, float]:
    """Efficiently calculate perplexity on packed dataset format (.bin/.idx)."""
    print(f"Calculating perplexity on packed dataset: {data_dir} ({split} split)...")

    # Load packed dataset
    dataset = PackedDataset(data_dir, split=split)

    # Create efficient dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Memory-mapped files work best with single process
        pin_memory=True
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Perplexity {Path(data_dir).name}"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Count valid tokens (non-padding)
            valid_tokens = attention_mask.sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        print(f"   └─ Perplexity: {perplexity:.2f} ({total_tokens:,} tokens)")
        return {
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "avg_loss": avg_loss
        }
    else:
        print(f"   └─ No valid tokens found")
        return {"perplexity": float('inf'), "total_tokens": 0, "avg_loss": float('inf')}



def calculate_perplexity_wikitext(model, tokenizer, max_samples: int = 100) -> float:
    """Calculate perplexity on Wikitext-2 with memory-efficient processing."""
    print(f"Calculating perplexity on Wikitext-2 (max {max_samples} samples)...")

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for example in tqdm(dataset, desc="Wikitext Perplexity"):
            text = example["text"].strip()
            if len(text) < 20:  # Skip very short texts
                continue

            # Tokenize with truncation
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )

            input_ids = inputs['input_ids'].to(model.device)
            if input_ids.shape[1] < 10:  # Skip very short sequences
                continue

            attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids)).to(model.device)
            labels = input_ids.clone()

            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                valid_tokens = attention_mask.sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

            except torch.cuda.OutOfMemoryError:
                print("GPU memory limit reached, stopping evaluation")
                break

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        print(f"   └─ Wikitext Perplexity: {perplexity:.2f} ({total_tokens:,} tokens)")
        return perplexity
    else:
        print("   └─ No valid tokens found")
        return float('inf')


def calculate_multi_dataset_perplexity(
    model,
    tokenizer,
    data_dirs: List[str],
    weights: Optional[List[float]] = None
) -> Dict[str, float]:
    """Calculate per-dataset and weighted perplexity across multiple packed datasets."""
    print(f"📊 Multi-dataset perplexity evaluation on {len(data_dirs)} datasets...")

    # Normalize weights
    if weights is None:
        weights = [1.0] * len(data_dirs)
    elif len(weights) != len(data_dirs):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of data_dirs ({len(data_dirs)})")

    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    results = {}
    total_weighted_loss = 0.0
    total_weighted_tokens = 0

    # Calculate perplexity for each dataset using efficient packed format
    for i, (data_dir, weight) in enumerate(zip(data_dirs, normalized_weights)):
        dataset_name = Path(data_dir).name
        print(f"   Dataset {i+1}/{len(data_dirs)}: {dataset_name} (weight: {weight:.3f})")

        # Use the new efficient evaluate_packed_perplexity function
        ppl_result = evaluate_packed_perplexity(model, tokenizer, data_dir, split="val")

        perplexity = ppl_result["perplexity"]
        total_tokens = ppl_result["total_tokens"]
        avg_loss = ppl_result["avg_loss"]

        results[f"ppl_{dataset_name}"] = perplexity

        if total_tokens > 0 and math.isfinite(avg_loss):
            # Add to weighted average calculation
            total_weighted_loss += avg_loss * total_tokens * weight
            total_weighted_tokens += total_tokens * weight

    # Calculate weighted perplexity
    if total_weighted_tokens > 0:
        weighted_avg_loss = total_weighted_loss / total_weighted_tokens
        weighted_perplexity = math.exp(weighted_avg_loss)
        results["ppl_weighted"] = weighted_perplexity
        print(f"📈 Weighted perplexity: {weighted_perplexity:.2f}")
    else:
        results["ppl_weighted"] = float('inf')
        print(f"❌ No valid tokens found across datasets")

    return results


def evaluate_boolq(model, tokenizer, max_samples: int = 100) -> Dict[str, float]:
    """Evaluate model on BoolQ (Yes/No questions) with token validation."""
    print(f"BoolQ evaluation (max {max_samples} samples)...")

    # VALIDATION: Ensure "Yes" and "No" are single tokens
    yes_tokens = tokenizer.encode("Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("No", add_special_tokens=False)

    if len(yes_tokens) != 1:
        raise ValueError(f"\"Yes\" tokenizes to {len(yes_tokens)} tokens: {yes_tokens}. BoolQ requires single token.")
    if len(no_tokens) != 1:
        raise ValueError(f"\"No\" tokenizes to {len(no_tokens)} tokens: {no_tokens}. BoolQ requires single token.")

    yes_token = yes_tokens[0]
    no_token = no_tokens[0]
    print(f"✅ Token validation passed: Yes={yes_token}, No={no_token}")

    # Load dataset
    dataset = load_dataset("boolq")["validation"]
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for example in tqdm(dataset, desc="BoolQ Evaluation"):
            question = example["question"]
            passage = example["passage"]
            answer = example["answer"]  # True/False

            # Format prompt
            prompt = f"Passage: {passage[:500]}...\nQuestion: {question}\nAnswer: "

            # Tokenization
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]  # [vocab_size]

            # Probabilities for Yes/No
            yes_prob = torch.softmax(logits, dim=0)[yes_token].item()
            no_prob = torch.softmax(logits, dim=0)[no_token].item()

            # Prediction
            predicted_answer = yes_prob > no_prob

            if predicted_answer == answer:
                correct += 1
            total += 1

    accuracy = correct / total
    return {"boolq_accuracy": accuracy, "boolq_correct": correct, "boolq_total": total}


def run_smoke_tests(model, tokenizer, test_prompts: List[str]) -> List[Dict[str, str]]:
    """Robust smoke test function with proper response extraction."""
    print("Running smoke-tests...")

    results = []
    model.eval()

    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            print(f"\nTest {i+1}/{len(test_prompts)}")
            print(f"Prompt: {prompt}")

            try:
                # Tokenization
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                input_length = inputs['input_ids'].shape[1]

                # Generation
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=96,
                    temperature=0.8,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )

                # ROBUST: Extract only the newly generated tokens
                generated_ids = outputs[0, input_length:]  # Only new tokens
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                result = {
                    "prompt": prompt,
                    "response": response.strip(),
                    "full_text": full_text
                }
                results.append(result)

                print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")

            except Exception as e:
                print(f"Error in test {i+1}: {e}")
                result = {
                    "prompt": prompt,
                    "response": f"Error: {str(e)}",
                    "full_text": f"Error: {str(e)}"
                }
                results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Complete model evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model to evaluate")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer/spm32k.model",
                       help="Path to SentencePiece tokenizer (default: data/tokenizer/spm32k.model)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Directory to save results")
    
    # Multi-dataset evaluation
    parser.add_argument("--data_dirs", nargs='+', type=str,
                       help="Multiple data directories for multi-dataset perplexity evaluation")
    parser.add_argument("--data_weights", nargs='+', type=float,
                       help="Weights for multi-dataset evaluation (must match --data_dirs length)")
    parser.add_argument("--skip_perplexity", action="store_true",
                       help="Skip perplexity calculation")
    parser.add_argument("--skip_boolq", action="store_true",
                       help="Skip BoolQ evaluation")
    parser.add_argument("--max_boolq_samples", type=int, default=100,
                       help="Maximum number of BoolQ samples")
    parser.add_argument("--custom_prompts", type=str, default=None,
                       help="JSON file with custom prompts")
    parser.add_argument("--fast_mode", action="store_true",
                       help="Run in fast mode with reduced samples")
    parser.add_argument("--detailed_output", action="store_true",
                       help="Generate detailed output files")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model = load_pretrained_model(args.model_path)
    
    # Load tokenizer
    if args.tokenizer_path:
        # Handle SentencePiece tokenizer files
        if args.tokenizer_path.endswith('.model'):
            # Load SentencePiece model directly
            tokenizer = SentencePieceTokenizerWrapper(args.tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {model.device}")
    
    # Structure to store all results
    evaluation_results = {
        "model_path": args.model_path,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(model.device)
    }
    
    # 1. Calculate perplexity (multi-dataset or single dataset)
    if not args.skip_perplexity:
        if args.data_dirs:
            # Multi-dataset perplexity evaluation
            try:
                # Validate arguments
                if args.data_weights and len(args.data_weights) != len(args.data_dirs):
                    raise ValueError(f"Number of weights ({len(args.data_weights)}) must match number of data_dirs ({len(args.data_dirs)})")
                
                # CRITICAL: Validate tokenizer consistency across all datasets
                tokenizer_metadata = validate_tokenizer_consistency_eval(args.data_dirs)
                evaluation_results["tokenizer_metadata"] = tokenizer_metadata
                
                multi_ppl_results = calculate_multi_dataset_perplexity(model, tokenizer, args.data_dirs, args.data_weights)
                evaluation_results.update(multi_ppl_results)
                
                # Print summary
                for key, value in multi_ppl_results.items():
                    if key.startswith("ppl_"):
                        dataset_name = key[4:]  # Remove 'ppl_' prefix
                        print(f"Perplexity ({dataset_name}): {value:.2f}")
                
            except Exception as e:
                print(f"Error calculating multi-dataset perplexity: {e}")
                evaluation_results["multi_perplexity_error"] = str(e)
        else:
            # Wikitext-2 perplexity evaluation
            try:
                wikitext_samples = 50 if args.fast_mode else 100
                perplexity = calculate_perplexity_wikitext(model, tokenizer, wikitext_samples)
                evaluation_results["perplexity_wikitext"] = perplexity
                print(f"Perplexity (Wikitext-2): {perplexity:.2f}")
            except Exception as e:
                print(f"Error calculating Wikitext perplexity: {e}")
                evaluation_results["perplexity_wikitext_error"] = str(e)
    
    # 2. BoolQ benchmark (adapted according to mode)
    if not args.skip_boolq:
        boolq_samples = min(args.max_boolq_samples, 20 if args.fast_mode else args.max_boolq_samples)
        try:
            boolq_results = evaluate_boolq(model, tokenizer, boolq_samples)
            evaluation_results.update(boolq_results)
            print(f"BoolQ Accuracy: {boolq_results['boolq_accuracy']:.3f} ({boolq_results['boolq_correct']}/{boolq_results['boolq_total']})")
        except Exception as e:
            print(f"Error during BoolQ evaluation: {e}")
            evaluation_results["boolq_error"] = str(e)
    
    # 3. Smoke-tests
    # Load prompts from file or use defaults
    default_prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "How does a neural network work?",
        "What are the benefits of renewable energy?",
        "Describe the water cycle.",
        "What is the capital of France?",
        "Write a short poem about nature.",
        "How do you make chocolate chip cookies?",
        "What causes seasons on Earth?",
        "Explain photosynthesis to a child."
    ]
    
    # Load custom prompts if provided, otherwise use defaults
    if args.custom_prompts:
        try:
            with open(args.custom_prompts, 'r', encoding='utf-8') as f:
                custom_data = json.load(f)
                if isinstance(custom_data, list):
                    test_prompts = custom_data
                else:
                    test_prompts = custom_data.get("prompts", default_prompts)
        except Exception as e:
            print(f"Error loading custom prompts: {e}. Using defaults.")
            test_prompts = default_prompts
    else:
        test_prompts = default_prompts[:5]  # Use first 5 default EN prompts
    
    # Run smoke-tests
    try:
        smoke_results = run_smoke_tests(model, tokenizer, test_prompts)
        evaluation_results["smoke_tests"] = smoke_results
        print(f"\nSmoke-tests completed: {len(smoke_results)} prompts tested")
    except Exception as e:
        print(f"Error during smoke-tests: {e}")
        evaluation_results["smoke_tests_error"] = str(e)
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    # Detailed save if requested
    if args.detailed_output and "smoke_tests" in evaluation_results:
        detailed_file = output_dir / "detailed_smoke_tests.md"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            f.write("# Smoke Tests - Detailed Results\n\n")

            # Fixed: smoke_tests is now a list directly
            for i, result in enumerate(evaluation_results["smoke_tests"], 1):
                f.write(f"## Test {i}\n\n")
                f.write(f"**Prompt:** {result['prompt']}\n\n")
                f.write(f"**Response:** {result['response']}\n\n")
                f.write(f"**Full Text:** {result['full_text']}\n\n")
                f.write("---\n\n")

        print(f"Detailed results in: {detailed_file}")
    
    # Summary report
    print(f"\n{'='*50}")
    print("EVALUATION REPORT")
    print(f"{'='*50}")
    print(f"Model: {args.model_path}")
    print(f"Parameters: {evaluation_results['model_parameters']:,}")
    
    # Display results summary
    if "perplexity_wikitext" in evaluation_results:
        print(f"Perplexity (Wikitext-2): {evaluation_results['perplexity_wikitext']:.2f}")

    # Multi-dataset perplexity results
    for key, value in evaluation_results.items():
        if key.startswith("ppl_") and key != "ppl_weighted":
            dataset_name = key[4:]  # Remove 'ppl_' prefix
            print(f"Perplexity ({dataset_name}): {value:.2f}")

    if "ppl_weighted" in evaluation_results:
        print(f"Weighted Perplexity: {evaluation_results['ppl_weighted']:.2f}")

    if "boolq_accuracy" in evaluation_results:
        print(f"BoolQ Accuracy: {evaluation_results['boolq_accuracy']:.3f}")

    if "smoke_tests" in evaluation_results:
        print(f"Smoke-tests: {len(evaluation_results['smoke_tests'])} prompts tested")
    
    print(f"\nResults saved to: {results_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()