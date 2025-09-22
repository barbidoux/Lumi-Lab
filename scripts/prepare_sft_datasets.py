#!/usr/bin/env python3
"""
Generic SFT Dataset preparation script.
Uses configuration files from config/sft_datasets/ to prepare datasets.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import glob

from datasets import load_dataset, Dataset


def load_dataset_config(config_path: str) -> Dict[str, Any]:
    """Load dataset configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_available_datasets(config_dir: str = "config/sft_datasets") -> List[str]:
    """List available dataset configurations."""
    config_files = glob.glob(os.path.join(config_dir, "*.json"))
    # Exclude README and mixed configs
    datasets = []
    for config_file in config_files:
        name = os.path.basename(config_file).replace('.json', '')
        if name not in ['mixed_sft', 'README']:
            datasets.append(name)
    return sorted(datasets)


def prepare_dataset_generic(config: Dict[str, Any], max_samples: int = None) -> int:
    """Generic dataset preparation based on configuration."""

    dataset_name = config['name']
    source = config['source']
    output_file = config['output_file']
    format_config = config.get('format', {})
    preprocessing = config.get('preprocessing', {})

    print(f"Loading {dataset_name} dataset from {source}...")

    try:
        # Load dataset
        if dataset_name == "dolly15k":
            dataset = load_dataset(source)['train']
        elif dataset_name == "oasst1":
            dataset = load_dataset(source)['train']
        elif dataset_name == "alpaca":
            dataset = load_dataset(source)['train']
        else:
            # Try generic loading
            try:
                dataset = load_dataset(source)['train']
            except:
                dataset = load_dataset(source, split="train")

    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return 0

    print(f"Loaded {len(dataset)} samples")

    # Convert to our standard format
    converted_samples = []

    if dataset_name == "dolly15k":
        converted_samples = prepare_dolly15k_format(dataset, format_config, preprocessing)
    elif dataset_name == "oasst1":
        converted_samples = prepare_oasst1_format(dataset, format_config, preprocessing, config.get('language', 'en'))
    elif dataset_name == "alpaca":
        converted_samples = prepare_alpaca_format(dataset, format_config, preprocessing)
    else:
        # Generic preparation
        converted_samples = prepare_generic_format(dataset, format_config, preprocessing, dataset_name)

    print(f"Converted {len(converted_samples)} samples")

    # Apply preprocessing filters
    if preprocessing:
        converted_samples = apply_preprocessing_filters(converted_samples, preprocessing)
        print(f"After preprocessing: {len(converted_samples)} samples")

    # Limit samples if requested
    if max_samples and max_samples < len(converted_samples):
        converted_samples = converted_samples[:max_samples]
        print(f"Limited to {max_samples} samples")

    # Save as JSONL
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Saved {len(converted_samples)} samples to {output_file}")
    return len(converted_samples)


def prepare_dolly15k_format(dataset, format_config: Dict, preprocessing: Dict) -> List[Dict]:
    """Prepare Dolly 15k dataset format."""
    converted_samples = []

    for example in dataset:
        instruction = example['instruction'].strip()
        context = example.get('context', '').strip()
        response = example['response'].strip()

        # Combine instruction and context if requested
        if preprocessing.get('combine_instruction_context', True) and context:
            prompt = f"{instruction}\n\nContext: {context}"
        else:
            prompt = instruction

        converted_samples.append({
            "prompt": prompt,
            "response": response,
            "source": "dolly15k"
        })

    return converted_samples


def prepare_oasst1_format(dataset, format_config: Dict, preprocessing: Dict, lang: str = "en") -> List[Dict]:
    """Prepare OASST1 dataset format."""

    # The dataset is already the train split
    conversations = dataset
    print(f"Processing {len(conversations)} conversation messages...")

    # Build conversation trees
    print("Building conversation trees...")
    message_dict = {}
    for msg in conversations:
        message_dict[msg['message_id']] = msg

    # Extract assistant responses with their prompts
    converted_samples = []

    for msg in conversations:
        if (msg['role'] == 'assistant' and
            msg['lang'] == lang and
            msg['parent_id'] in message_dict):

            # Get the user prompt
            parent_msg = message_dict[msg['parent_id']]
            if parent_msg['role'] == 'prompter':
                prompt = parent_msg['text'].strip()
                response = msg['text'].strip()

                # Skip if empty
                if not prompt or not response:
                    continue

                # Handle None rank values
                rank = msg.get('rank', 0)
                if rank is None:
                    rank = 0

                converted_samples.append({
                    "prompt": prompt,
                    "response": response,
                    "source": "oasst1",
                    "rank": rank
                })

    # Sort by rank if requested
    if preprocessing.get('sort_by_rank', True):
        converted_samples.sort(key=lambda x: x.get('rank', 0), reverse=True)

    # Remove rank field from output
    for sample in converted_samples:
        sample.pop('rank', None)

    return converted_samples


def prepare_alpaca_format(dataset, format_config: Dict, preprocessing: Dict) -> List[Dict]:
    """Prepare Stanford Alpaca dataset format."""
    converted_samples = []

    for example in dataset:
        instruction = example['instruction'].strip()
        input_text = example.get('input', '').strip()
        output_text = example['output'].strip()

        # Combine instruction and input if requested
        if preprocessing.get('combine_instruction_input', True) and input_text:
            prompt = f"{instruction}\n\nInput: {input_text}"
        else:
            prompt = instruction

        converted_samples.append({
            "prompt": prompt,
            "response": output_text,
            "source": "alpaca"
        })

    return converted_samples


def prepare_generic_format(dataset, format_config: Dict, preprocessing: Dict, source_name: str) -> List[Dict]:
    """Generic dataset preparation."""
    converted_samples = []

    prompt_key = format_config.get('prompt_key', 'prompt')
    response_key = format_config.get('response_key', 'response')

    for example in dataset:
        if prompt_key in example and response_key in example:
            prompt = str(example[prompt_key]).strip()
            response = str(example[response_key]).strip()

            if prompt and response:
                converted_samples.append({
                    "prompt": prompt,
                    "response": response,
                    "source": source_name
                })

    return converted_samples


def apply_preprocessing_filters(samples: List[Dict], preprocessing: Dict) -> List[Dict]:
    """Apply preprocessing filters to samples."""
    filtered_samples = []

    min_prompt_length = preprocessing.get('min_prompt_length', 10)
    min_response_length = preprocessing.get('min_response_length', 20)
    max_prompt_length = preprocessing.get('max_prompt_length', 2000)
    max_response_length = preprocessing.get('max_response_length', 4000)

    for sample in samples:
        prompt = sample['prompt']
        response = sample['response']

        # Length filters
        if (len(prompt) < min_prompt_length or
            len(response) < min_response_length or
            len(prompt) > max_prompt_length or
            len(response) > max_response_length):
            continue

        # Clean text
        prompt = ''.join(char for char in prompt if ord(char) >= 32 or char in '\n\t')
        response = ''.join(char for char in response if ord(char) >= 32 or char in '\n\t')

        sample['prompt'] = prompt.strip()
        sample['response'] = response.strip()

        if sample['prompt'] and sample['response']:
            filtered_samples.append(sample)

    return filtered_samples


def main():
    parser = argparse.ArgumentParser(description="Generic SFT dataset preparation from configs")
    parser.add_argument("--dataset", type=str, required=False,
                       help="Dataset name (based on config files) or 'all'")
    parser.add_argument("--config_dir", type=str, default="config/sft_datasets",
                       help="Directory containing dataset configurations")
    parser.add_argument("--output_dir", type=str, default="data/sft",
                       help="Output directory (can be overridden by config)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples per dataset")
    parser.add_argument("--list", action="store_true",
                       help="List available dataset configurations")

    args = parser.parse_args()

    # List available datasets
    available_datasets = list_available_datasets(args.config_dir)

    if args.list:
        print("Available datasets:")
        for dataset in available_datasets:
            config_path = os.path.join(args.config_dir, f"{dataset}.json")
            if os.path.exists(config_path):
                config = load_dataset_config(config_path)
                print(f"  - {dataset}: {config.get('description', 'No description')}")
        return

    if not args.dataset:
        print("Available datasets:")
        for dataset in available_datasets:
            print(f"  - {dataset}")
        print("\nUse --dataset <name> or --dataset all")
        return

    print(f"=== SFT Dataset Preparation ===")
    print(f"Config directory: {args.config_dir}")
    print(f"Output directory: {args.output_dir}")

    total_samples = 0
    datasets_to_prepare = []

    if args.dataset == "all":
        datasets_to_prepare = available_datasets
    else:
        if args.dataset in available_datasets:
            datasets_to_prepare = [args.dataset]
        else:
            print(f"Error: Dataset '{args.dataset}' not found in {args.config_dir}")
            print(f"Available datasets: {', '.join(available_datasets)}")
            return

    # Prepare each dataset
    prepared_files = []

    for dataset_name in datasets_to_prepare:
        config_path = os.path.join(args.config_dir, f"{dataset_name}.json")

        if not os.path.exists(config_path):
            print(f"Warning: Config file not found: {config_path}")
            continue

        print(f"\n--- Preparing {dataset_name} ---")
        config = load_dataset_config(config_path)

        # Override output directory if specified
        if args.output_dir != "data/sft":
            output_file = os.path.join(args.output_dir, os.path.basename(config['output_file']))
            config['output_file'] = output_file

        try:
            samples = prepare_dataset_generic(config, args.max_samples)
            total_samples += samples
            prepared_files.append(config['output_file'])
        except Exception as e:
            print(f"Error preparing {dataset_name}: {e}")
            continue

    print(f"\n✅ Preparation completed!")
    print(f"Total samples prepared: {total_samples}")
    print(f"Files created:")
    for file_path in prepared_files:
        print(f"  - {file_path}")

    # Generate example usage
    if len(prepared_files) > 0:
        print(f"\n📋 Example SFT command:")

        if len(prepared_files) == 1:
            datasets_str = prepared_files[0]
            weights_str = ""
        else:
            datasets_str = " ".join(prepared_files)
            # Default weights
            if len(prepared_files) == 3:
                weights_str = "--dataset_weights 0.4 0.4 0.2"
            elif len(prepared_files) == 2:
                weights_str = "--dataset_weights 0.6 0.4"
            else:
                weights_str = f"--dataset_weights {' '.join(['0.33'] * len(prepared_files))}"

        print(f"""
accelerate launch --mixed_precision bf16 scripts/03_sft.py \\
  --model_path checkpoints/mix-tiny-fa2-75k/tiny/final \\
  --tokenizer_path data/tokenizer/spm32k.model \\
  --dataset_paths {datasets_str} \\
  {weights_str} \\
  --prompt_template chatml \\
  --output_dir checkpoints/sft/tiny-mixed \\
  --config_path config/sft_tiny.json \\
  --use_lora --do_gen_test
""")


if __name__ == "__main__":
    main()