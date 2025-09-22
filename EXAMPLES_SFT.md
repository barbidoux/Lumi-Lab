# SFT Training Examples

## Dataset Preparation

First, prepare your datasets:

```bash
# Prepare all popular SFT datasets
python scripts/prepare_sft_datasets.py --dataset all --output_dir data/sft

# Or prepare individual datasets
python scripts/prepare_sft_datasets.py --dataset dolly15k --output_dir data/sft
python scripts/prepare_sft_datasets.py --dataset oasst1 --output_dir data/sft
python scripts/prepare_sft_datasets.py --dataset alpaca --output_dir data/sft
```

## Configuration Commands

### Example 1: Tiny model with ChatML + Dataset Weighting
```bash
accelerate launch --mixed_precision bf16 scripts/03_sft.py \
  --model_path checkpoints/mix-tiny-fa2-75k/tiny/final \
  --tokenizer_path data/tokenizer/spm32k.model \
  --dataset_paths data/sft/dolly15k.jsonl data/sft/oasst1_en.jsonl data/sft/alpaca.jsonl \
  --dataset_weights 0.4 0.4 0.2 \
  --prompt_template chatml \
  --output_dir checkpoints/sft/tiny-mixed \
  --config_path config/sft_tiny.json \
  --use_lora --lora_r 16 --packing --do_gen_test
```

### Example 2: Small model with balanced weighting
```bash
accelerate launch --mixed_precision bf16 scripts/03_sft.py \
  --model_path checkpoints/mix-small-fa2-150k/small/final \
  --tokenizer_path data/tokenizer/spm32k.model \
  --dataset_paths data/sft/dolly15k.jsonl data/sft/oasst1_en.jsonl data/sft/alpaca.jsonl \
  --dataset_weights 0.33 0.33 0.34 \
  --prompt_template instruct \
  --output_dir checkpoints/sft/small-balanced \
  --config_path config/sft_small.json \
  --use_lora --merge_adapters
```

### Example 3: High-quality focus (Dolly-heavy)
```bash
accelerate launch --mixed_precision bf16 scripts/03_sft.py \
  --model_path checkpoints/mix-tiny-fa2-75k/tiny/final \
  --tokenizer_path data/tokenizer/spm32k.model \
  --dataset_paths data/sft/dolly15k.jsonl data/sft/oasst1_en.jsonl \
  --dataset_weights 0.7 0.3 \
  --prompt_template chatml \
  --output_dir checkpoints/sft/tiny-quality \
  --config_path config/sft_tiny.json \
  --use_lora --merge_adapters
```

### Example 3: Resume training from checkpoint
```bash
accelerate launch --mixed_precision bf16 scripts/03_sft.py \
  --model_path checkpoints/mix-tiny-fa2-75k/tiny/final \
  --tokenizer_path data/tokenizer/spm32k.model \
  --dataset_paths data/sft/mixed_dataset.jsonl \
  --prompt_template chatml \
  --output_dir checkpoints/sft/tiny-chatml \
  --config_path config/sft_tiny.json \
  --resume_from_checkpoint checkpoints/sft/tiny-chatml/checkpoint-500
```

### Example 4: Full fine-tuning (no LoRA)
```bash
accelerate launch --mixed_precision bf16 scripts/03_sft.py \
  --model_path checkpoints/mix-tiny-fa2-75k/tiny/final \
  --tokenizer_path data/tokenizer/spm32k.model \
  --dataset_paths data/sft/high_quality_dataset.jsonl \
  --prompt_template chatml \
  --output_dir checkpoints/sft/tiny-full-ft \
  --config_path config/sft_tiny.json \
  --no_lora --no_packing
```

## Evaluation Examples

### Quick evaluation on validation set
```bash
python scripts/06_eval_sft.py \
  --model_path checkpoints/sft/tiny-chatml \
  --tokenizer_path data/tokenizer/spm32k.model \
  --dataset_paths data/sft/eval_dataset.jsonl \
  --prompt_template chatml \
  --max_eval_samples 200
```

### Generation-only test
```bash
python scripts/06_eval_sft.py \
  --model_path checkpoints/sft/tiny-chatml \
  --tokenizer_path data/tokenizer/spm32k.model \
  --dataset_paths data/sft/eval_dataset.jsonl \
  --prompt_template chatml \
  --gen_only --num_gen_samples 10
```

### Evaluate merged model
```bash
python scripts/06_eval_sft.py \
  --model_path checkpoints/sft/tiny-chatml \
  --tokenizer_path data/tokenizer/spm32k.model \
  --dataset_paths data/sft/eval_dataset.jsonl \
  --prompt_template chatml \
  --no_lora
```

## Dataset Weighting

The `--dataset_weights` parameter controls the relative contribution of each dataset:

### How it works:
- **Relative weights**: `[0.4, 0.4, 0.2]` = 40%/40%/20% split
- **Automatic resampling**: Datasets are upsampled or downsampled to match target weights
- **Preserves diversity**: All datasets contribute, but in controlled proportions

### Examples:

```bash
# Equal weighting (33.3% each)
--dataset_weights 0.33 0.33 0.34

# High-quality focus (70% Dolly, 30% OASST1)
--dataset_weights 0.7 0.3

# Balanced mix with some Alpaca (40% Dolly, 40% OASST1, 20% Alpaca)
--dataset_weights 0.4 0.4 0.2

# Conversation-heavy (20% Dolly, 60% OASST1, 20% Alpaca)
--dataset_weights 0.2 0.6 0.2
```

### Recommended weights by model size:

| Model Size | Dolly15k | OASST1 | Alpaca | Reasoning |
|------------|----------|---------|---------|-----------|
| Tiny (<1B) | 0.5 | 0.3 | 0.2 | Focus on high-quality |
| Small (1-3B) | 0.4 | 0.4 | 0.2 | Balanced mix |
| Base (7B+) | 0.3 | 0.4 | 0.3 | More diversity |

## Dataset Format

The script expects datasets in JSONL format with `prompt` and `response` fields:

```json
{"prompt": "Explain quantum computing", "response": "Quantum computing is a revolutionary approach..."}
{"prompt": "What is photosynthesis?", "response": "Photosynthesis is the process by which plants..."}
```

### Available prepared datasets:
- **dolly15k.jsonl**: ~15k high-quality instruction-response pairs
- **oasst1_en.jsonl**: ~35k conversational responses (English)
- **alpaca.jsonl**: ~52k instruction-following examples

## Key Features

### ✅ Fixed Issues
- **Tokenizer validation**: SP32k tokenizer with correct special tokens (pad:0, unk:1, bos:2, eos:3)
- **ChatML support**: Proper EOS tokens and special token handling
- **Evaluation**: Train/val split with early stopping
- **Multi-dataset**: Support for multiple dataset concatenation
- **Flexible training**: LoRA/full-FT, packing, checkpoint resume
- **Output management**: Respects --output_dir, cosine scheduler

### 🚀 New Capabilities
- Multi-dataset training with automatic concatenation
- Robust tokenizer validation and error handling
- ChatML special token management with embedding resize
- Early stopping and best model loading
- LoRA adapter merging option
- Comprehensive generation testing
- Automatic checkpoint resuming
- Configuration override via CLI arguments

### 📋 Requirements
- **Tokenizer**: Must be SP32k compatible with correct special token mapping
- **Model**: Pre-trained model checkpoint from step 02
- **Datasets**: JSONL format with prompt/response fields
- **GPU**: Recommended for training (supports bf16 on Ampere+)