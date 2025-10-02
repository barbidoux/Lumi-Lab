# GEMINI.md

This file provides guidance to Gemini AI when working with code in this repository.

## Overview

Lumi-Lab is a complete mini-LLM training pipeline implementing a modern LLaMA-like decoder-only transformer architecture. It provides a modular, production-grade system for pre-training, supervised fine-tuning (SFT), and direct preference optimization (DPO) of language models, optimized for RTX 4090 GPUs.

## Setup Instructions

Install all required Python packages:

```bash
pip install -r requirements.txt
```

*Note: For GPU usage with FlashAttention-2, you may need to install it separately.*

## Core Workflows

The project follows a **9-step modular pipeline** driven by JSON configuration files.

### Step 1: Prepare Tokenizer Corpus (100M tokens)

Process raw data from various sources into a clean, sharded corpus for tokenizer training.

**Command:**
```bash
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/tokenizer_training_mix.json \
    --output-dir data/datasets/tokenizer_corpus \
    --shard-size 10000 \
    --use-cache
```

**Key Arguments:**
- `--config`: Path to corpus configuration JSON
- `--output-dir`: Where to save the processed corpus
- `--use-cache`: Enables caching for faster, resumable processing
- `--analyze-only`: Preview data sources and statistics without processing

### Step 2: Train Tokenizer (ONCE!)

Train a single global SentencePiece tokenizer (vocab_size=32768) that will be used for ALL datasets.

**Command:**
```bash
python scripts/20_train_tokenizer.py \
    --config config/pretrain/tokenizer/spm32k.json \
    --output-dir data/models/tokenizers/spm_32k
```

**Important:** Train the tokenizer ONCE. All datasets must use this same frozen tokenizer.

### Step 3: Prepare Pre-training Corpus (600M tokens)

Prepare the main training corpus with the target token budget.

**Command:**
```bash
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/tiny_23M_chinchilla_500M.json \
    --output-dir data/datasets/tiny_23M_corpus \
    --shard-size 10000 \
    --use-cache
```

### Step 4: Pack Dataset (Tokenize + Sequence Packing)

Tokenize the corpus and pack into fixed-length sequences (1024 tokens).

**Command:**
```bash
python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/tiny_23M_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/tiny_23M_1024
```

**Key Features:**
- Tokenizes using the frozen tokenizer
- Packs sequences to fixed length
- Creates train/val split (95/5)
- Stores tokenizer SHA256 for validation

### Step 5: Pre-train Model (~24h on RTX 4090)

Train a model from scratch using `accelerate` for distributed training.

**Command:**
```bash
accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/tiny_23M_1024 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain \
    --num_workers 4 \
    --log-level INFO 2>&1 | tee logs/pretrain_tiny.log
```

**Key Arguments:**
- `--config`: Training configuration (includes architecture reference)
- `--data_dirs`: One or more packed dataset directories
- `--data_weights`: (Optional) Weights for multi-dataset sampling
- `--tokenizer_dir`: Path to tokenizer directory
- `--output_dir`: Where to save checkpoints
- `--resume_from_checkpoint auto`: Auto-resume from latest checkpoint
- `--log-level INFO`: Detailed logging

**Multi-Dataset Training:**
```bash
accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/c4_1024 data/processed/gutenberg_1024 \
    --data_weights 0.5 0.5 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain
```

### Step 6: Evaluate Pre-training

Assess the pretrained model's performance on perplexity, BoolQ, and generation quality.

**Command:**
```bash
python scripts/50_evaluate_pretrain.py \
    --model_path checkpoints/pretrain/tiny/final \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir evaluation_results/pretrain/tiny
```

### Step 7: Prepare SFT Corpus

Prepare conversational datasets with template formatting (ChatML recommended).

**Command:**
```bash
python scripts/60_prepare_sft_corpus.py \
    --config config/sft/datasets/alpaca_chatml.json \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir data/sft_processed/alpaca_chatml

python scripts/60_prepare_sft_corpus.py \
    --config config/sft/datasets/oasst1_chatml.json \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir data/sft_processed/oasst1_chatml
```

**Templates Available:**
- ChatML (recommended): `<|im_start|>user\n...<|im_end|>`
- Instruct: `### Instruction:\n...\n### Response:`
- Chat: `Human: ...\nAssistant:`
- Alpaca: Full instruction format

### Step 8: SFT Training (~2-3h on RTX 4090)

Fine-tune with LoRA adapters for parameter-efficient training.

**Command:**
```bash
accelerate launch --mixed_precision bf16 scripts/70_train_sft.py \
    --config config/sft/training/lora_optimal_tiny_23m.json \
    --model_path checkpoints/pretrain/tiny/final \
    --data_dirs data/sft_processed/alpaca_chatml data/sft_processed/oasst1_chatml \
    --data_weights 0.7 0.3 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/sft/tiny 2>&1 | tee logs/sft_tiny.log
```

**Key Features:**
- LoRA adapters (~95% memory reduction)
- Multi-dataset weighted sampling
- Generation evaluation during training
- Complete state preservation

### Step 9: Evaluate SFT

Comprehensive SFT evaluation with metrics and generation quality.

**Command:**
```bash
python scripts/80_evaluate_sft.py \
    --config config/evaluation/sft_standard.json \
    --model_path checkpoints/sft/tiny/final \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_file evaluation_results/sft/tiny/results.json
```

### Step 10: Serve Model (Optional)

Launch interactive or API serving.

**Command:**
```bash
python scripts/100_serve.py \
    --model_path checkpoints/sft/tiny/final \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --port 8000
```

## Critical Design Principles

### 1. Tokenizer Consistency
- **ONE global tokenizer** trained once, reused everywhere
- SHA256 verification on ALL dataset loads
- Blocking errors prevent silent failures

### 2. Deterministic Training
- Full RNG state preservation (Python, NumPy, PyTorch, CUDA)
- Complete checkpoint state (model, optimizer, scheduler, scaler, RNG)
- Resume produces identical loss curves

### 3. Multi-Dataset Support
- Weighted sampling for curriculum learning
- Chinchilla-optimal token budgets (20 tokens per parameter)
- Phase-based training (high quality → diversity)

### 4. Progress Tracking
- Equivalent epochs displayed during evaluation
- Progress percentage tracking
- Helps detect overfitting early

### 5. Production Robustness
- Sharded data loading (unlimited dataset sizes)
- Memory-efficient streaming
- Comprehensive error handling

## Configuration Files

### Architecture Configs
- `config/architectures/tiny.json`: 23M params (n_layer=6, d_model=256, n_head=4)
- `config/architectures/small.json`: 42M params (n_layer=12, d_model=512, n_head=8)
- `config/architectures/base.json`: 124M params (n_layer=24, d_model=768, n_head=12)

### Corpus Configs
- `config/pretrain/corpus/tokenizer_training_mix.json`: Tokenizer corpus (100M tokens)
- `config/pretrain/corpus/tiny_23M_chinchilla_500M.json`: 23M model corpus (600M tokens)
- `config/pretrain/sources/*.json`: Individual source configs

### Training Configs
- `config/pretrain/training/chinchilla_tiny_500m.json`: Complete training config
- `config/sft/training/lora_optimal_tiny_23m.json`: Optimal SFT config

### Evaluation Config
- `config/evaluation/sft_standard.json`: Evaluation metrics + generation params

## Verification and Testing

**Primary verification tool:** `scripts/50_evaluate_pretrain.py` and `scripts/80_evaluate_sft.py`

When making changes to training or model architecture, run evaluation scripts to ensure:
- Perplexity has not regressed
- BoolQ accuracy is maintained
- Generation quality is acceptable

**Smoke tests:** Evaluation scripts include basic generation capability checks.

## Common Pitfalls

1. **Forgetting to train tokenizer first**: Always run `20_train_tokenizer.py` before datasets
2. **Inconsistent tokenizers**: Never train multiple tokenizers
3. **Missing `--tokenizer_dir`**: All training and evaluation requires explicit tokenizer path
4. **Ignoring tokenizer mismatch errors**: These are BLOCKING - fix immediately
5. **Not using multi-dataset sampling**: Curriculum learning improves quality significantly
6. **Not monitoring progress tracking**: Watch equivalent epochs to detect overfitting
7. **Not using accelerate**: Always use `accelerate launch --mixed_precision bf16`
8. **Forgetting --log-level**: Use `--log-level INFO` for detailed logs

## Troubleshooting

### Error: "Tokenizer mismatch between datasets"
```bash
# Check all SHA256 hashes
cat data/models/tokenizers/spm_32k/tokenizer_config.json | jq '.sha256_hash'
cat data/processed/*/final_manifest.json | jq '.tokenizer_config_hash'
# ALL must be IDENTICAL

# If mismatch: re-pack the problematic dataset
python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/problem_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/problem_dataset \
    --force
```

### Error: "No tokenizer found"
```bash
# Train the tokenizer first
python scripts/20_train_tokenizer.py \
    --config config/pretrain/tokenizer/spm32k.json \
    --output-dir data/models/tokenizers/spm_32k
```

## Performance Targets

| Model | Perplexity (WikiText-2) | BoolQ Accuracy | Tokens/sec (RTX 4090) | VRAM (BF16) |
|-------|------------------------|----------------|----------------------|-------------|
| Tiny (23M) | 45-70 | 58-68% (pretrain)<br>68-75% (SFT) | ~600-800 | ~8-12 GB (pretrain)<br>~6-8 GB (SFT LoRA) |
| Small (42M) | 30-50 | 65-72% | ~400-600 | ~12-16 GB |
| Base (124M) | 20-35 | 70-78% | ~200-300 | ~16-24 GB |

## Memory Optimization

| Technique | VRAM Reduction | Speed Impact | When to Use |
|-----------|----------------|--------------|-------------|
| FlashAttention-2 | ~50% | +30% | Always (if available) |
| Gradient Checkpointing | ~40% | -10% | Large models or OOM |
| LoRA (r=32) | ~95% | +200% | SFT stage |
| BF16 Mixed Precision | ~50% | +20% | Always (default) |

## File Structure

```
Lumi-Lab/
├── scripts/
│   ├── 10_prepare_corpus.py       # Corpus preparation
│   ├── 20_train_tokenizer.py      # Tokenizer training
│   ├── 30_pack_dataset.py         # Tokenize and pack
│   ├── 40_pretrain.py             # Pre-training
│   ├── 50_evaluate_pretrain.py    # Pre-training evaluation
│   ├── 60_prepare_sft_corpus.py   # SFT corpus prep
│   ├── 70_train_sft.py            # SFT training
│   ├── 80_evaluate_sft.py         # SFT evaluation
│   ├── 90_train_dpo.py            # DPO training
│   └── 100_serve.py               # Model serving
├── config/
│   ├── architectures/             # Model architectures
│   ├── pretrain/                  # Pre-training configs
│   ├── sft/                       # SFT configs
│   └── evaluation/                # Evaluation configs
├── utils/
│   ├── dataset_utils.py           # Dataset loaders
│   ├── model_utils.py             # Model creation
│   ├── sft_templates.py           # Conversation templates
│   └── tokenizer_utils.py         # Tokenizer validation
├── data/
│   ├── datasets/                  # Corpus (text)
│   ├── models/tokenizers/         # Global tokenizer
│   ├── processed/                 # Packed datasets
│   └── sft_processed/             # SFT corpus
├── checkpoints/
│   ├── pretrain/                  # Pre-training checkpoints
│   └── sft/                       # SFT checkpoints
└── evaluation_results/            # Evaluation outputs
```

## Additional Documentation

- [README.md](README.md): Complete usage guide and tutorials
- [TEST_PLAN_FINAL_23M.md](TEST_PLAN_FINAL_23M.md): Complete testing guide for 23M model
- [CLAUDE.md](CLAUDE.md): Claude Code assistant guide (similar to this file)
- [MIGRATION.md](MIGRATION.md): Migration guide from old script naming

## Script Naming Convention

- **10-series**: Data preparation (corpus, tokenizer, packing)
- **40-series**: Training (pretrain, SFT, DPO)
- **50-80-series**: Evaluation
- **100-series**: Serving

## Best Practices

1. **Start with tiny model**: Validate pipeline before scaling
2. **Monitor VRAM**: Use `nvidia-smi` or `watch -n 1 nvidia-smi`
3. **Checkpoint frequently**: Set `save_steps` to 10-15% of total steps
4. **Test with --analyze-only**: Preview corpus stats before processing
5. **Validate tokenizer**: Check SHA256 hash consistency
6. **Use weighted sampling**: Multi-dataset > single dataset
7. **Use accelerate with bf16**: Always use mixed precision
8. **Track with logs**: Use `2>&1 | tee logs/{step}.log`
9. **Watch progress**: Monitor equivalent epochs for overfitting
10. **Enable deterministic mode**: Critical for reproducibility
