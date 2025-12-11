#!/usr/bin/env python3
"""
DPO evaluation script for Lumi-Lab pipeline.

This script performs comprehensive evaluation of DPO-trained models:
- Reward margin and win rate
- Perplexity comparison (chosen vs rejected)
- BoolQ accuracy
- Smoke tests
- Generation quality assessment
- JSON/CSV output with plots
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import PeftModel

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dpo_evaluation import evaluate_dpo_model
from utils.model_utils import load_pretrained_model


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=numeric_level,
        handlers=[logging.StreamHandler(sys.stdout)]
    )


logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load evaluation configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    logger.info(f"📋 Loaded evaluation config from {config_path}")
    return config


def load_dpo_model(
    model_path: str,
    device: torch.device
) -> torch.nn.Module:
    """
    Load DPO model (with or without LoRA adapters).

    Args:
        model_path: Path to DPO model checkpoint
        device: Torch device

    Returns:
        Loaded model
    """
    model_path = Path(model_path)

    # Check if model has LoRA adapters
    has_lora = (model_path / "adapter_config.json").exists()

    if has_lora:
        logger.info("🔧 Loading DPO model with LoRA adapters...")

        # Find base model
        if (model_path / "base_model").exists():
            base_model_path = model_path / "base_model"
        else:
            # Try parent directory
            base_model_path = model_path.parent

        # Load base model
        base_model = load_pretrained_model(str(base_model_path))

        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, str(model_path))

        # Merge for faster inference
        logger.info("   Merging LoRA adapters for faster inference...")
        model = model.merge_and_unload()

        logger.info(f"   ✅ Model loaded with merged LoRA adapters")
    else:
        logger.info("📦 Loading DPO model (standard)...")
        model = load_pretrained_model(str(model_path))
        logger.info("   ✅ Model loaded")

    model = model.to(device)
    model.eval()

    return model


def load_eval_dataset(
    data_dir: str,
    max_samples: int = 1000
) -> Any:
    """
    Load DPO evaluation dataset from processed corpus.

    Args:
        data_dir: Directory containing DPO corpus
        max_samples: Maximum samples to load

    Returns:
        Loaded dataset
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"DPO corpus directory not found: {data_path}")

    # Load manifest
    manifest_path = data_path / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        logger.info(f"📂 Loading evaluation dataset: {manifest['dataset_name']}")
        logger.info(f"   Total samples: {manifest['total_samples']:,}")

    # Load JSONL shards
    shard_files = sorted(data_path.glob("shard_*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {data_path}")

    logger.info(f"   Loading from {len(shard_files)} shard(s)...")

    # Load dataset
    dataset = load_dataset(
        'json',
        data_files=[str(f) for f in shard_files],
        split='train'
    )

    # Subsample if needed
    if max_samples and len(dataset) > max_samples:
        logger.info(f"   Subsampling {max_samples:,} from {len(dataset):,}")
        indices = list(range(len(dataset)))
        import random
        random.shuffle(indices)
        dataset = dataset.select(indices[:max_samples])

    logger.info(f"   ✅ Loaded {len(dataset):,} samples for evaluation")

    return dataset


def save_results(
    results: Dict[str, Any],
    output_file: str,
    config: Dict[str, Any],
    model_path: str,
    tokenizer_dir: str
):
    """Save evaluation results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create full results structure
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'tokenizer_dir': str(tokenizer_dir),
        'evaluation_config': config.get('name', 'unknown'),
        'metrics': {
            k: v for k, v in results.items()
            if k not in ['generation_samples', 'smoke_test_results']
        },
        'generation_samples': results.get('generation_samples', []),
        'smoke_test_results': results.get('smoke_test_results', [])
    }

    # Save JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    logger.info(f"💾 Results saved to {output_path}")

    # Also save CSV summary
    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("metric,value\n")
        for key, value in full_results['metrics'].items():
            if isinstance(value, (int, float)):
                f.write(f"{key},{value}\n")

    logger.info(f"💾 CSV summary saved to {csv_path}")


def print_results_summary(results: Dict[str, Any]):
    """Print a formatted summary of evaluation results."""
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 80 + "\n")

    # DPO-specific metrics
    if 'reward_margin' in results:
        logger.info("🎯 DPO Metrics:")
        logger.info(f"   Reward margin: {results['reward_margin']:.4f}")
        logger.info(f"   Win rate: {results['win_rate']:.2%}")
        logger.info(f"   Accuracy (threshold 0.0): {results.get('accuracy_threshold_0.0', 0.0):.2%}")
        logger.info("")

    # Perplexity comparison
    if 'chosen_perplexity_mean' in results:
        logger.info("📈 Perplexity Comparison:")
        logger.info(f"   Chosen responses: {results['chosen_perplexity_mean']:.2f}")
        logger.info(f"   Rejected responses: {results['rejected_perplexity_mean']:.2f}")
        logger.info(f"   Improvement: {results.get('perplexity_improvement', 0.0):.2f}")
        logger.info("")

    # BoolQ
    if 'boolq_accuracy' in results:
        logger.info("🎯 BoolQ (Question Answering):")
        logger.info(f"   Accuracy: {results['boolq_accuracy']:.2%} ({results.get('boolq_correct', 0)}/{results.get('boolq_total', 0)})")
        logger.info("")

    # Smoke tests
    if 'smoke_tests_pass_rate' in results:
        logger.info("🔬 Smoke Tests:")
        logger.info(f"   Pass rate: {results['smoke_tests_pass_rate']:.2%} ({results['smoke_tests_passed']}/{results['smoke_tests_total']})")
        logger.info("")

    # Generation samples
    if 'generation_samples' in results:
        logger.info("💬 Generation Samples:")
        for i, sample in enumerate(results['generation_samples'][:3]):
            logger.info(f"\n   Sample {i+1}:")
            logger.info(f"   Prompt: {sample['prompt']}")
            logger.info(f"   Response: {sample['response'][:200]}...")

    logger.info("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="DPO model evaluation for Lumi-Lab pipeline")

    parser.add_argument("--config", type=str, required=True,
                       help="Path to evaluation configuration file (e.g., config/evaluation/dpo_standard.json)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to DPO model checkpoint")
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                       help="Path to tokenizer directory")
    parser.add_argument("--eval_data_dir", type=str, default=None,
                       help="Path to DPO evaluation corpus (optional, uses config if not provided)")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output file for results (JSON)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger.info("=" * 80)
    logger.info("DPO MODEL EVALUATION - Lumi-Lab Pipeline")
    logger.info("=" * 80 + "\n")

    try:
        # Load config
        config = load_config(args.config)

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️ Device: {device}")

        # Load tokenizer
        logger.info(f"📖 Loading tokenizer from {args.tokenizer_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"   Vocabulary size: {len(tokenizer)}")

        # Load model
        logger.info(f"🤖 Loading DPO model from {args.model_path}...")
        model = load_dpo_model(args.model_path, device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

        # Load evaluation dataset (if provided)
        eval_dataset = None
        if args.eval_data_dir:
            eval_dataset = load_eval_dataset(
                args.eval_data_dir,
                max_samples=config.get('eval_dataset', {}).get('max_samples', 1000)
            )
        else:
            logger.warning("⚠️ No evaluation dataset provided. Only running generation tests.")

        # Run evaluation
        results = evaluate_dpo_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            device=device,
            eval_config=config
        )

        # Save results
        save_results(
            results=results,
            output_file=args.output_file,
            config=config,
            model_path=args.model_path,
            tokenizer_dir=args.tokenizer_dir
        )

        # Print summary
        print_results_summary(results)

        logger.info("✅ DPO EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"❌ Error during DPO evaluation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
