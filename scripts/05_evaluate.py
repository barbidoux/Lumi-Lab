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
from evaluate import load
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sentencepiece as spm

from utils.dataset_utils import TokenizedDataset, create_dataloader
from utils.model_utils import load_pretrained_model


def validate_tokenizer_consistency_eval(data_dirs: List[str]) -> Dict:
    """Validate that all datasets use the same tokenizer and return metadata (evaluation version)."""
    if not data_dirs:
        return {}
    
    print("🔍 Validating tokenizer consistency across datasets...")
    
    reference_metadata = None
    reference_dir = None
    
    for data_dir in data_dirs:
        manifest_path = Path(data_dir) / 'manifest.json'
        
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"❌ Manifest file not found: {manifest_path}\n"
                f"This dataset may not have been processed with the new tokenizer system.\n"
                f"Please re-process using: make reencode-dataset DIR={data_dir}"
            )
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load manifest {manifest_path}: {e}")
        
        if 'tokenizer_metadata' not in manifest:
            raise ValueError(
                f"❌ No tokenizer metadata in manifest: {manifest_path}\n"
                f"This dataset was processed with an older version.\n"
                f"Please re-process using: make reencode-dataset DIR={data_dir}"
            )
        
        tokenizer_metadata = manifest['tokenizer_metadata']
        
        # Validate required fields
        required_fields = ['tokenizer_sha256', 'tokenizer_vocab_size', 'special_tokens', 'normalizer', 'byte_fallback']
        for field in required_fields:
            if field not in tokenizer_metadata:
                raise ValueError(f"❌ Missing tokenizer metadata field '{field}' in {manifest_path}")
        
        if reference_metadata is None:
            reference_metadata = tokenizer_metadata
            reference_dir = data_dir
            print(f"📊 Reference tokenizer (from {data_dir}):")
            print(f"   SHA256: {reference_metadata['tokenizer_sha256'][:16]}...")
            print(f"   Vocab size: {reference_metadata['tokenizer_vocab_size']:,}")
            print(f"   Special tokens: {reference_metadata['special_tokens']}")
        else:
            # Compare with reference
            mismatches = []
            
            if tokenizer_metadata['tokenizer_sha256'] != reference_metadata['tokenizer_sha256']:
                mismatches.append(f"SHA256: {tokenizer_metadata['tokenizer_sha256'][:16]}... != {reference_metadata['tokenizer_sha256'][:16]}...")
            
            if tokenizer_metadata['tokenizer_vocab_size'] != reference_metadata['tokenizer_vocab_size']:
                mismatches.append(f"Vocab size: {tokenizer_metadata['tokenizer_vocab_size']} != {reference_metadata['tokenizer_vocab_size']}")
            
            if tokenizer_metadata['special_tokens'] != reference_metadata['special_tokens']:
                mismatches.append(f"Special tokens: {tokenizer_metadata['special_tokens']} != {reference_metadata['special_tokens']}")
            
            if tokenizer_metadata['normalizer'] != reference_metadata['normalizer']:
                mismatches.append(f"Normalizer: {tokenizer_metadata['normalizer']} != {reference_metadata['normalizer']}")
            
            if tokenizer_metadata['byte_fallback'] != reference_metadata['byte_fallback']:
                mismatches.append(f"Byte fallback: {tokenizer_metadata['byte_fallback']} != {reference_metadata['byte_fallback']}")
            
            if mismatches:
                raise ValueError(
                    f"❌ Tokenizer mismatch between {reference_dir} and {data_dir}:\\n" +
                    "\\n".join(f"   {mismatch}" for mismatch in mismatches) +
                    f"\\n\\nAll datasets must use the same tokenizer. Solutions:\\n"
                    f"1. Re-process inconsistent dataset: make reencode-dataset DIR={data_dir}\\n"
                    f"2. Use tokenizer-reset + data-rebuild-all to restart with consistent tokenizer"
                )
            
            print(f"✅ {data_dir}: tokenizer matches reference")
    
    print(f"✅ All {len(data_dirs)} datasets use consistent tokenizer")
    return reference_metadata


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


def calculate_perplexity(model, tokenizer, dataset_name: str = "wikitext-2-raw-v1") -> float:
    """Calculate perplexity on a reference dataset like Wikitext-2."""
    print(f"Calculating perplexity on {dataset_name}...")
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")["test"]
    
    # Data preparation
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding=False,
            return_overflowing_tokens=False
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenization for perplexity"
    )
    
    # Filter empty sequences
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 10)
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: {
            'input_ids': torch.tensor(x[0]['input_ids']).unsqueeze(0),
            'attention_mask': torch.ones(1, len(x[0]['input_ids']))
        }
    )
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating perplexity"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            # Labels are the same as input_ids for perplexity
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Compter les tokens valides (non-padding)
            valid_tokens = attention_mask.sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


def calculate_multi_dataset_perplexity(
    model, 
    tokenizer, 
    data_dirs: List[str], 
    weights: Optional[List[float]] = None
) -> Dict[str, float]:
    """Calculate per-dataset and weighted perplexity across multiple datasets."""
    print(f"📊 Multi-dataset perplexity evaluation on {len(data_dirs)} datasets...")
    
    # Import here to avoid circular imports
    from utils.dataset_utils import WeightedMultiDatasetSampler
    
    # Initialize sampler for validation data
    multi_sampler = WeightedMultiDatasetSampler(
        data_dirs=data_dirs,
        weights=weights,
        seed=42,
        batch_size=1,  # For perplexity calculation
        split="val"  # Use validation split
    )
    
    results = {}
    total_weighted_loss = 0.0
    total_weighted_tokens = 0
    
    model.eval()
    
    # Calculate perplexity for each dataset
    for i, (dataset, dataset_name, weight) in enumerate(zip(multi_sampler.datasets, multi_sampler.dataset_names, multi_sampler.weights)):
        print(f"   Dataset {i+1}/{len(data_dirs)}: {dataset_name}")
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for sample_idx in tqdm(range(len(dataset)), desc=f"Perplexity {dataset_name}"):
                sample = dataset[sample_idx]
                
                input_ids = sample['input_ids'].unsqueeze(0).to(model.device)
                attention_mask = (input_ids != 0).long().to(model.device)
                labels = sample['labels'].unsqueeze(0).to(model.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Count valid tokens (non-padding)
                valid_tokens = attention_mask.sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
        
        # Calculate perplexity for this dataset
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
            results[f"ppl_{dataset_name}"] = perplexity
            print(f"   └─ Perplexity: {perplexity:.2f}")
            
            # Add to weighted average
            total_weighted_loss += total_loss * weight
            total_weighted_tokens += total_tokens * weight
        else:
            print(f"   └─ No valid tokens found")
            results[f"ppl_{dataset_name}"] = float('inf')
    
    # Calculate weighted perplexity
    if total_weighted_tokens > 0:
        weighted_avg_loss = total_weighted_loss / total_weighted_tokens
        weighted_perplexity = math.exp(weighted_avg_loss)
        results["ppl_weighted"] = weighted_perplexity
        print(f"📈 Weighted perplexity: {weighted_perplexity:.2f}")
    else:
        results["ppl_weighted"] = float('inf')
    
    return results


def evaluate_boolq(model, tokenizer, max_samples: int = 100) -> Dict[str, float]:
    """Evaluate model on BoolQ (Yes/No questions)."""
    print(f"BoolQ evaluation (max {max_samples} samples)...")
    
    # Chargement du dataset BoolQ
    dataset = load_dataset("boolq")["validation"]
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    correct = 0
    total = 0
    
    model.eval()
    
    with torch.no_grad():
        for example in tqdm(dataset, desc="Évaluation BoolQ"):
            question = example["question"]
            passage = example["passage"]
            answer = example["answer"]  # True/False
            
            # Format du prompt
            prompt = f"Passage: {passage[:500]}...\nQuestion: {question}\nAnswer: "
            
            # Tokenisation
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800)
            # Move to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate probabilities for "Yes" and "No"
            yes_token = tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_token = tokenizer.encode("No", add_special_tokens=False)[0]
            
            # Forward pass
            outputs = model(**inputs)
            # Get logits for the last token - shape should be [batch, seq_len, vocab_size]
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
    """Simple, working smoke test function."""
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
                
                # Generation with default parameters
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
                
                # Decoding - outputs should be [batch_size, sequence_length]
                generated_ids = outputs[0]  # Take first (and only) sample from batch
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Extract just the generated part
                generated_text = response[len(prompt):].strip()
                
                result = {
                    "prompt": prompt,
                    "response": generated_text,
                    "full_text": response
                }
                results.append(result)
                
                print(f"Response: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
                
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
                       help="Chemin vers le tokenizer SentencePiece (par défaut: data/tokenizer/spm32k.model)")
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
    
    # Chargement du tokenizer
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
            # Traditional single dataset perplexity
            try:
                perplexity = calculate_perplexity(model, tokenizer)
                evaluation_results["perplexity"] = perplexity
                print(f"Perplexity (Wikitext-2): {perplexity:.2f}")
            except Exception as e:
                print(f"Error calculating perplexity: {e}")
                evaluation_results["perplexity_error"] = str(e)
    
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
        print(f"Erreur lors des smoke-tests: {e}")
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
            
            for i, result in enumerate(evaluation_results["smoke_tests"]["individual_results"], 1):
                f.write(f"## Test {i}\n\n")
                f.write(f"**Prompt:** {result['prompt']}\n\n")
                f.write(f"**Response:** {result['response']}\n\n")
                f.write(f"**Quality:** {result['quality_scores']['overall_quality']:.3f}/1.0\n")
                f.write(f"**Longueur:** {result['word_count']} mots\n\n")
                f.write("---\n\n")
        
        print(f"Detailed results in: {detailed_file}")
    
    # Summary report
    print(f"\n{'='*50}")
    print("RAPPORT D'ÉVALUATION")
    print(f"{'='*50}")
    print(f"Model: {args.model_path}")
    print(f"Parameters: {evaluation_results['model_parameters']:,}")
    
    # Handle compatibility with old format
    if "perplexity_metrics" in evaluation_results:
        ppl_metrics = evaluation_results["perplexity_metrics"]
        print(f"Perplexity (WikiText-2): {ppl_metrics['perplexity']:.2f}")
        print(f"  └─ {ppl_metrics['total_tokens']:,} tokens @ {ppl_metrics['tokens_per_second']:.0f} tok/sec")
    elif "perplexity" in evaluation_results:
        print(f"Perplexity (Wikitext-2): {evaluation_results['perplexity']:.2f}")
    
    if "boolq_accuracy" in evaluation_results:
        print(f"BoolQ Accuracy: {evaluation_results['boolq_accuracy']:.3f}")
    
    if "smoke_tests" in evaluation_results:
        if "summary_stats" in evaluation_results["smoke_tests"]:
            smoke_summary = evaluation_results["smoke_tests"]["summary_stats"]
            print(f"Smoke-tests: {smoke_summary['total_prompts']} prompts")
            print(f"  └─ Average quality: {smoke_summary['avg_quality_score']:.3f}/1.0")
        else:
            # Compatibility with old format
            print(f"Smoke-tests: {len(evaluation_results['smoke_tests'])} prompts tested")
    
    print(f"\nResults saved to: {results_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()