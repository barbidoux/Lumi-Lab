# SFT Dataset Configurations

This directory contains configurations for preparing and using instruction-following datasets for supervised fine-tuning (SFT).

## Available Datasets

### 1. Dolly 15k (`dolly15k.json`)
- **Source**: Databricks Dolly 15k
- **Size**: ~15k high-quality instruction-response pairs
- **Quality**: High
- **Recommended weight**: 0.4
- **Description**: Human-generated instruction following examples

### 2. OpenAssistant OASST1 (`oasst1.json`)
- **Source**: OpenAssistant Conversations v1
- **Size**: ~35k samples (English only)
- **Quality**: Medium-High
- **Recommended weight**: 0.4
- **Description**: Multi-turn conversational responses, ranked by quality

### 3. Stanford Alpaca (`alpaca.json`)
- **Source**: Stanford Alpaca
- **Size**: ~52k samples
- **Quality**: Medium
- **Recommended weight**: 0.2
- **Description**: GPT-generated instruction following examples

## Quick Start

### 1. Prepare Datasets

```bash
# Prepare individual datasets
python scripts/prepare_sft_datasets.py --dataset dolly15k --output_dir data/sft
python scripts/prepare_sft_datasets.py --dataset oasst1 --output_dir data/sft
python scripts/prepare_sft_datasets.py --dataset alpaca --output_dir data/sft

# Or prepare all at once
python scripts/prepare_sft_datasets.py --dataset all --output_dir data/sft
```

### 2. Run SFT Training

#### Single Dataset
```bash
accelerate launch --mixed_precision bf16 scripts/03_sft.py \\
  --model_path checkpoints/mix-tiny-fa2-75k/tiny/final \\
  --tokenizer_path data/tokenizer/spm32k.model \\
  --dataset_paths data/sft/dolly15k.jsonl \\
  --prompt_template chatml \\
  --output_dir checkpoints/sft/tiny-dolly \\
  --config_path config/sft_tiny.json \\
  --use_lora --do_gen_test
```

#### Multi-Dataset with Weighting
```bash
accelerate launch --mixed_precision bf16 scripts/03_sft.py \\
  --model_path checkpoints/mix-tiny-fa2-75k/tiny/final \\
  --tokenizer_path data/tokenizer/spm32k.model \\
  --dataset_paths data/sft/dolly15k.jsonl data/sft/oasst1_en.jsonl data/sft/alpaca.jsonl \\
  --dataset_weights 0.4 0.4 0.2 \\
  --prompt_template chatml \\
  --output_dir checkpoints/sft/tiny-mixed \\
  --config_path config/sft_tiny.json \\
  --use_lora --merge_adapters --do_gen_test
```

## Dataset Weighting

The `--dataset_weights` parameter allows you to control the relative contribution of each dataset:

- **Weights are relative**: `[0.4, 0.4, 0.2]` means 40%/40%/20% split
- **Upsampling**: If a dataset gets higher weight than its natural proportion, examples are repeated
- **Downsampling**: If a dataset gets lower weight, random subset is selected
- **Example**: With datasets of [15k, 35k, 52k] samples and weights [0.4, 0.4, 0.2]:
  - Dolly: 15k → ~41k samples (upsampled)
  - OASST1: 35k → ~41k samples (upsampled)
  - Alpaca: 52k → ~20k samples (downsampled)
  - Total: ~102k samples

## Recommended Configurations

### For Tiny Models (< 1B parameters)
```bash
--dataset_weights 0.5 0.3 0.2  # Focus on high-quality Dolly
--config_path config/sft_tiny.json
--val_split_ratio 0.05  # More validation data
```

### For Small Models (1-3B parameters)
```bash
--dataset_weights 0.4 0.4 0.2  # Balanced mix
--config_path config/sft_small.json
--val_split_ratio 0.02  # Standard validation split
```

### For Base Models (7B+ parameters)
```bash
--dataset_weights 0.3 0.4 0.3  # More diverse mix
--config_path config/sft.json
--val_split_ratio 0.01  # Minimal validation split
```

## Template Support

All datasets support these prompt templates:

- **`chatml`**: ChatML format (`<|im_start|>user...assistant<|im_end|>`)
- **`instruct`**: Instruction format (`### Instruction: ... ### Response:`)
- **`chat`**: Simple chat format (`Human: ... Assistant:`)

## Quality Assessment

Use the evaluation script to assess model quality:

```bash
python scripts/06_eval_sft.py \\
  --model_path checkpoints/sft/tiny-mixed \\
  --tokenizer_path data/tokenizer/spm32k.model \\
  --dataset_paths data/sft/dolly15k.jsonl \\
  --prompt_template chatml \\
  --max_eval_samples 500
```

## File Format

All datasets are converted to JSONL format with standardized fields:

```json
{"prompt": "Explain quantum computing", "response": "Quantum computing is...", "source": "dolly15k"}
{"prompt": "What is photosynthesis?", "response": "Photosynthesis is...", "source": "oasst1"}
```

## Advanced Usage

### Custom Sampling Limits
```bash
python scripts/prepare_sft_datasets.py --dataset all --max_samples 10000 --output_dir data/sft_small
```

### Multi-language Support (OASST1)
```bash
python scripts/prepare_sft_datasets.py --dataset oasst1 --lang fr --output_dir data/sft_french
```

### Resume Training
```bash
# Auto-resume from latest checkpoint
accelerate launch scripts/03_sft.py [args] --resume_from_checkpoint

# Resume from specific checkpoint
accelerate launch scripts/03_sft.py [args] --resume_from_checkpoint checkpoints/sft/tiny-mixed/checkpoint-500
```