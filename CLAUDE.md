# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Lumi-Lab is a complete mini-LLM training pipeline implementing a modern LLaMA-like decoder-only transformer architecture. It provides a modular, production-grade system for pre-training, supervised fine-tuning (SFT), and direct preference optimization (DPO) of language models, optimized for RTX 4090 GPUs.

**Key Architecture Principles**:
- ✅ **Config-Driven**: ALL hyperparameters in JSON configs (no hardcoded values)
- ✅ **WebUI-Ready**: CLI accepts only paths and runtime flags
- ✅ **Modular Pipeline**: Corpus → Tokenizer → Packing → Training → Evaluation
- ✅ **Multi-Dataset**: Weighted sampling for curriculum learning
- ✅ **Deterministic**: Full reproducibility with RNG state preservation

## Essential Commands

### Complete Pipeline (Example: Tiny 23M Model)

#### Step 1: Tokenizer Corpus (100M tokens)
```bash
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/tokenizer_training_mix.json \
    --output-dir data/datasets/tokenizer_corpus \
    --shard-size 10000 \
    --use-cache
```

#### Step 2: Train Tokenizer (32K vocab - ONCE!)
```bash
python scripts/20_train_tokenizer.py \
    --config config/pretrain/tokenizer/spm32k.json \
    --output-dir data/models/tokenizers/spm_32k
```

#### Step 3: Pretrain Corpus (600M tokens)
```bash
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/tiny_23M_chinchilla_500M.json \
    --output-dir data/datasets/tiny_23M_corpus \
    --shard-size 10000 \
    --use-cache
```

#### Step 4: Pack Dataset (Tokenize + Sequences)
```bash
python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/tiny_23M_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/tiny_23M_1024
```

#### Step 5: Pre-training (23M model, ~24h on RTX 4090)
```bash
accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/tiny_23M_1024 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain \
    --num_workers 4 \
    --log-level INFO 2>&1 | tee logs/pretrain_tiny.log
```

#### Step 6: Evaluate Pre-training
```bash
python scripts/50_evaluate_pretrain.py \
    --model_path checkpoints/pretrain/tiny/final \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir evaluation_results/pretrain/tiny
```

#### Step 7: SFT Corpus Preparation
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

#### Step 8: SFT Training (LoRA, ~2-3h on RTX 4090)
```bash
accelerate launch --mixed_precision bf16 scripts/70_train_sft.py \
    --config config/sft/training/lora_optimal_tiny_23m.json \
    --model_path checkpoints/pretrain/tiny/final \
    --data_dirs data/sft_processed/alpaca_chatml data/sft_processed/oasst1_chatml \
    --data_weights 0.7 0.3 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/sft/tiny 2>&1 | tee logs/sft_tiny.log
```

#### Step 9: Evaluate SFT
```bash
python scripts/80_evaluate_sft.py \
    --config config/evaluation/sft_standard.json \
    --model_path checkpoints/sft/tiny/final \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_file evaluation_results/sft/tiny/results.json
```

#### Step 10: DPO Corpus Preparation (Optional - ⚠️ EXPERIMENTAL - UNTESTED)
```bash
python scripts/90_prepare_dpo_corpus.py \
    --config config/dpo/datasets/orca_dpo.json \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir data/dpo_processed/orca
```

#### Step 11: DPO Training (⚠️ EXPERIMENTAL - UNTESTED)
```bash
accelerate launch --mixed_precision bf16 scripts/95_train_dpo.py \
    --config config/dpo/training/dpo_tiny.json \
    --model_path checkpoints/sft/tiny/final \
    --data_dirs data/dpo_processed/orca \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/dpo/tiny 2>&1 | tee logs/dpo_tiny.log
```

#### Step 12: Evaluate DPO (⚠️ EXPERIMENTAL - UNTESTED)
```bash
python scripts/98_evaluate_dpo.py \
    --config config/evaluation/dpo_standard.json \
    --model_path checkpoints/dpo/tiny/final \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --eval_data_dir data/dpo_processed/orca \
    --output_file evaluation_results/dpo/tiny/results.json
```

## Architecture Overview

### Modular Pipeline Design

Lumi-Lab implements a **12-step modular pipeline** that separates concerns for maximum flexibility:

#### Step 1: Corpus Preparation (`10_prepare_corpus.py`)
- **Purpose**: Transform raw datasets into cleaned, validated text corpus
- **Config**: `config/pretrain/corpus/*.json` or `config/pretrain/sources/*.json`
- **Operations**: Loading, cleaning, deduplication (MinHash), quality filtering, sharding
- **Output**: `data/datasets/{name}/` with JSONL shards + manifest + data card
- **Key**: Tokenizer-independent, pure text processing

#### Step 2: Tokenizer Training (`20_train_tokenizer.py`)
- **Purpose**: Train a single global SentencePiece tokenizer (vocab_size=32768)
- **Config**: `config/pretrain/tokenizer/spm32k.json`
- **Critical**: Only run ONCE. All datasets must use the same frozen tokenizer
- **Output**: `data/models/tokenizers/spm_32k/` with SHA256 verification
- **Safety**: Built-in consistency validation prevents tokenizer mismatches

#### Step 3: Dataset Packing (`30_pack_dataset.py`)
- **Purpose**: Tokenize corpus and pack into fixed-length sequences
- **Config**: `config/pretrain/packing/default.json`
- **Operations**: Token encoding, sequence packing (1024), train/val split, shuffle
- **Output**: `data/processed/{name}/` with .bin/.idx files + manifest
- **Validation**: Tokenizer SHA256 stored in manifest

#### Step 4: Pre-training (`40_pretrain.py`)
- **Purpose**: Train models from scratch with multi-dataset support
- **Config**: `config/pretrain/training/*.json` (includes architecture reference)
- **Features**: Weighted sampling, deterministic training, auto-resume, complete RNG state preservation
- **Optimization**: FlashAttention-2, gradient checkpointing, mixed precision
- **Progress Tracking**: Equivalent epochs displayed during evaluation for overfitting detection
- **Output**: `checkpoints/pretrain/{model}/` with complete state

#### Step 5: Pre-training Evaluation (`50_evaluate_pretrain.py`)
- **Purpose**: Evaluate pretrained models (perplexity, BoolQ, generations)
- **Output**: `evaluation_results/pretrain/{model}/`

#### Step 6: SFT Corpus Preparation (`60_prepare_sft_corpus.py`)
- **Purpose**: Prepare conversational datasets with template formatting
- **Config**: `config/sft/datasets/*.json`
- **Templates**: ChatML (recommended), Instruct, Chat, Alpaca
- **Output**: `data/sft_processed/{name}/` with tokenizer validation

#### Step 7: SFT Training (`70_train_sft.py`)
- **Purpose**: Fine-tune with LoRA adapters (95% memory reduction)
- **Config**: `config/sft/training/*.json`
- **Features**: Multi-dataset weighted sampling, generation evaluation, full state checkpointing
- **Output**: `checkpoints/sft/{model}/` with LoRA adapters

#### Step 8: SFT Evaluation (`80_evaluate_sft.py`)
- **Purpose**: Comprehensive SFT evaluation
- **Config**: `config/evaluation/sft_standard.json`
- **Metrics**: Perplexity, BoolQ, smoke tests, generation quality
- **Output**: `evaluation_results/sft/{model}/results.json`

#### Step 9a: DPO Corpus Preparation (`90_prepare_dpo_corpus.py`) - ⚠️ EXPERIMENTAL
- **Purpose**: Prepare DPO datasets with chosen/rejected response pairs
- **Config**: `config/dpo/datasets/*.json` (orca_dpo.json, hh_rlhf.json)
- **Operations**: Loading, validation, quality filtering, sharding with tokenizer verification
- **Output**: `data/dpo_processed/{name}/` with JSONL shards + manifest
- **Key**: Validates triplets (prompt, chosen, rejected), ensures tokenizer consistency

#### Step 9b: DPO Training (`95_train_dpo.py`) - ⚠️ EXPERIMENTAL
- **Purpose**: Align model with human preferences using DPO loss
- **Config**: `config/dpo/training/*.json` (dpo_tiny.json, dpo_small.json, dpo_base.json)
- **Features**: Multi-dataset weighted sampling, LoRA adapters, config-driven, tokenizer SHA256 verification
- **Key parameters**: beta=0.1 (KL penalty), learning_rate=5e-7 (much lower than SFT)
- **Output**: `checkpoints/dpo/{model}/` with LoRA adapters + full state
- **Note**: Run AFTER SFT, not on pretrained model

#### Step 9c: DPO Evaluation (`98_evaluate_dpo.py`) - ⚠️ EXPERIMENTAL
- **Purpose**: Comprehensive DPO model evaluation
- **Config**: `config/evaluation/dpo_standard.json`
- **Metrics**: Reward margin, win rate, perplexity comparison (chosen vs rejected), BoolQ, generation quality
- **Output**: `evaluation_results/dpo/{model}/results.json` + CSV summary

### Critical Design Principles

1. **Tokenizer Consistency**: One global tokenizer prevents subtle bugs
   - SHA256 verification on all dataset loads
   - Blocking errors instead of silent failures
   - Automatic validation in training scripts

2. **Deterministic Training**: Complete reproducibility
   - Full RNG state preservation (Python, NumPy, PyTorch, CUDA)
   - Checkpoint includes global_step, optimizer, scheduler, scaler states
   - Resume training produces identical loss curves

3. **Multi-Dataset Support**: Weighted sampling for curriculum learning
   - Chinchilla-optimal token budgets (20 tokens per parameter)
   - Phase-based training (high quality → diversity)
   - Real-time dataset mix monitoring

4. **Progress Tracking**: Overfitting detection
   - Equivalent epochs displayed during evaluation steps
   - Progress percentage tracking (global_step/max_steps)
   - Helps detect when to stop early if perplexity increases

5. **Production-Grade Robustness**:
   - Comprehensive error handling and validation
   - Sharded data loading (unlimited dataset sizes)
   - Memory-efficient streaming from disk
   - Complete logging and metrics

## Key Configuration Files

### Architecture Configs
- `config/architectures/tiny.json`: 23M params (n_layer=6, d_model=256, n_head=4)
- `config/architectures/small.json`: 42M params (n_layer=12, d_model=512, n_head=8)
- `config/architectures/base.json`: 124M params (n_layer=24, d_model=768, n_head=12)
- All use: head_dim=64, ffn_ratio=4.0, vocab_size=32768

### Corpus Configs
- `config/pretrain/corpus/tokenizer_training_mix.json`: Tokenizer training corpus (100M tokens)
- `config/pretrain/corpus/tiny_23M_chinchilla_500M.json`: 23M model corpus (600M tokens)
- `config/pretrain/sources/*.json`: Individual source configs (C4, Gutenberg, FineWeb, Wikipedia)

### Tokenizer Config
- `config/pretrain/tokenizer/spm32k.json`: SentencePiece 32k config with special tokens

### Packing Config
- `config/pretrain/packing/default.json`: Sequence packing params (length=1024, split=0.95, shuffle=true)

### Training Configs
- `config/pretrain/training/chinchilla_tiny_500m.json`: Complete training config (architecture + hyperparams)
  - Includes: learning_rate, batch_size, optimizer (betas, eps, weight_decay), scheduler, hardware params

### SFT Configs
- `config/sft/datasets/*.json`: Dataset configs (sources, templates, quality filters)
- `config/sft/training/*.json`: Training configs (LoRA settings, hyperparameters, eval prompts)
- Key configs:
  - `lora_optimal_tiny_23m.json`: Optimal config for 23M model (r=32, alpha=64, lr=8e-5)
  - `alpaca_chatml.json`: ChatML template for Alpaca dataset
  - `oasst1_chatml.json`: ChatML template for OASST1 dataset

### Evaluation Config
- `config/evaluation/sft_standard.json`: Evaluation metrics + generation params

## Critical Workflows

### Complete Pipeline from Scratch (23M Model, 4 Datasets)
See **TEST_PLAN_FINAL_23M.md** for the complete step-by-step guide.

**Summary**:
1. Prepare tokenizer corpus (100M tokens: C4 28M + Gutenberg 24M + FineWeb 24M + Wikipedia 24M)
2. Train tokenizer (32k vocab, SentencePiece Unigram)
3. Prepare pretrain corpus (600M tokens: 4 datasets × 150M)
4. Pack dataset (tokenize + sequence packing 1024)
5. Pre-train model (30K steps, Chinchilla-optimal)
6. Evaluate pretrain (perplexity, BoolQ)
7. Prepare SFT corpus (Alpaca + OASST1, ChatML template)
8. SFT training (LoRA r=32, 2K steps)
9. Evaluate SFT (full metrics + generation quality)

### Multi-Dataset Training Strategy
```bash
# Phase A: High-quality foundations (C4 + Gutenberg)
accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_phase_a.json \
    --data_dirs data/processed/c4_1024 data/processed/gutenberg_1024 \
    --data_weights 0.67 0.33 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain

# Phase B: Diversity (FineWeb + Wikipedia) - resume from Phase A
accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_phase_b.json \
    --data_dirs data/processed/fineweb_1024 data/processed/wikipedia_1024 \
    --data_weights 0.50 0.50 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain \
    --resume_from_checkpoint checkpoints/pretrain/tiny/step_15000
```

### Troubleshooting Tokenizer Issues

**Error: "Tokenizer mismatch between datasets"**
```bash
# This is a BLOCKING error - do not ignore!
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

**Error: "No tokenizer found"**
```bash
# You forgot to train the tokenizer first
python scripts/20_train_tokenizer.py \
    --config config/pretrain/tokenizer/spm32k.json \
    --output-dir data/models/tokenizers/spm_32k
```

## Important Implementation Details

### Model Architecture (LLaMA-like)
- **Components**: RMSNorm, SwiGLU activation, RoPE positional embeddings, no bias terms
- **Implementation**: `utils/model_utils.py:create_model()`
- **Attention**: FlashAttention-2 by default, automatic fallback to PyTorch SDPA
- **Scaling**: Consistent head_dim=64 across all model sizes

### Dataset Loading
- **Pre-training**: `utils/dataset_utils.py:ShardedTokenizedDataset` (memory-efficient streaming)
- **SFT**: `utils/dataset_utils.py:SFTDataset` (conversation formatting + caching)
- **Sampling**: `WeightedMultiDatasetSampler` for deterministic multi-dataset mixing
- **Validation**: SHA256 tokenizer verification on every dataset load

### Checkpointing Strategy
- **Location**: `checkpoints/{stage}/{model_size}/step_{N}` or `final`
- **Contents**: Model weights, optimizer state, scheduler state, scaler state, RNG states, global_step, loss history
- **Auto-resume**: Scripts detect latest checkpoint automatically
- **Validation**: Restarting from checkpoint produces identical loss/LR curves

**IMPORTANT - Path Conventions by Script:**
| Script | Output Path Behavior | Example |
|--------|---------------------|---------|
| `40_pretrain.py` | Appends `{model_name}/` subdirectory | `--output_dir checkpoints/pretrain/micro` → saves to `checkpoints/pretrain/micro/micro/` |
| `70_train_sft.py` | Uses path as-is (no subdirectory) | `--output_dir checkpoints/sft/micro` → saves to `checkpoints/sft/micro/` |
| `95_train_dpo.py` | Uses path as-is (no subdirectory) | `--output_dir checkpoints/dpo/micro` → saves to `checkpoints/dpo/micro/` |

**Chaining Pretrain → SFT:**
```bash
# Pretrain saves to: checkpoints/pretrain/micro/micro/final
accelerate launch scripts/40_pretrain.py \
    --output_dir checkpoints/pretrain/micro ...

# SFT must reference the ACTUAL path (with double nesting):
accelerate launch scripts/70_train_sft.py \
    --model_path checkpoints/pretrain/micro/micro/final \
    --output_dir checkpoints/sft/micro ...
```

### SFT Templates
- **ChatML** (recommended): `<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n...\n<|im_end|>`
- **Instruct**: `### Instruction:\n...\n\n### Response:\n...`
- **Chat**: `Human: ...\n\nAssistant: ...`
- **Alpaca**: Full instruction format with task description
- **Location**: `utils/sft_templates.py`

### Memory Optimization Techniques
| Technique | VRAM Reduction | Speed Impact | When to Use |
|-----------|----------------|--------------|-------------|
| FlashAttention-2 | ~50% | +30% | Always (if available) |
| Gradient Checkpointing | ~40% | -10% | Large models or OOM |
| LoRA (r=32) | ~95% | +200% | SFT stage |
| BF16 Mixed Precision | ~50% | +20% | Always (default) |
| Batch Size Tuning | Variable | Neutral | Match VRAM capacity |

## Common Pitfalls

1. **Forgetting to train tokenizer first**: Always run tokenizer training before preparing datasets
2. **Inconsistent tokenizers**: Never train multiple tokenizers. Use frozen tokenizer for all datasets.
3. **Missing `--tokenizer_dir` or `--tokenizer_path`**: Training requires tokenizer_dir, evaluation requires explicit tokenizer_dir
4. **Ignoring tokenizer mismatch errors**: These are BLOCKING for good reason - fix immediately
5. **Not using multi-dataset weighted sampling**: Curriculum learning significantly improves model quality
6. **Insufficient warmup steps**: Use 10% of max_steps as warmup (critical for stability)
7. **Not monitoring progress tracking**: Watch equivalent epochs in evaluation logs to detect overfitting
8. **Not using accelerate**: Always use `accelerate launch --mixed_precision bf16` for training
9. **Forgetting --log-level**: Use `--log-level INFO` for detailed training logs
10. **Log buffering issues (WSL2)**: Logs may appear in bursts instead of real-time. Use unbuffered output:
    ```bash
    # Option 1: Environment variable (recommended)
    PYTHONUNBUFFERED=1 python scripts/40_pretrain.py ... 2>&1 | tee logs/pretrain.log

    # Option 2: stdbuf for line buffering
    stdbuf -oL python scripts/40_pretrain.py ... 2>&1 | tee logs/pretrain.log
    ```

## File Structure Reference

```
Lumi-Lab/
├── scripts/
│   ├── 10_prepare_corpus.py       # Corpus preparation (streaming)
│   ├── 20_train_tokenizer.py      # Tokenizer training
│   ├── 30_pack_dataset.py         # Tokenize and pack sequences
│   ├── 40_pretrain.py             # Pre-training
│   ├── 50_evaluate_pretrain.py    # Pre-training evaluation
│   ├── 60_prepare_sft_corpus.py   # SFT corpus preparation
│   ├── 70_train_sft.py            # SFT training (LoRA)
│   ├── 80_evaluate_sft.py         # SFT evaluation
│   ├── 90_prepare_dpo_corpus.py   # DPO corpus preparation (⚠️ experimental)
│   ├── 95_train_dpo.py            # DPO training (⚠️ experimental)
│   ├── 98_evaluate_dpo.py         # DPO evaluation (⚠️ experimental)
│   └── 100_serve.py               # Model serving
│
├── config/
│   ├── architectures/             # Model architectures
│   │   ├── tiny.json              # 23M params
│   │   ├── small.json             # 42M params
│   │   └── base.json              # 124M params
│   │
│   ├── pretrain/
│   │   ├── corpus/                # Corpus configs
│   │   │   ├── tokenizer_training_mix.json
│   │   │   └── tiny_23M_chinchilla_500M.json
│   │   ├── sources/               # Individual sources
│   │   │   ├── c4_english.json
│   │   │   ├── gutenberg_books.json
│   │   │   ├── fineweb_edu.json
│   │   │   └── vietgpt_wikipedia.json
│   │   ├── tokenizer/
│   │   │   └── spm32k.json
│   │   ├── packing/
│   │   │   └── default.json
│   │   └── training/              # Training configs
│   │       └── chinchilla_tiny_500m.json
│   │
│   ├── sft/
│   │   ├── datasets/              # SFT dataset configs
│   │   │   ├── alpaca_chatml.json
│   │   │   └── oasst1_chatml.json
│   │   └── training/              # SFT training configs
│   │       └── lora_optimal_tiny_23m.json
│   │
│   └── evaluation/
│       └── sft_standard.json
│
├── utils/
│   ├── dataset_utils.py           # Dataset loaders and samplers
│   ├── model_utils.py             # Model creation and checkpointing
│   ├── sft_templates.py           # Conversation templates
│   ├── sft_evaluation.py          # SFT-specific evaluation
│   ├── tokenizer_utils.py         # Tokenizer validation
│   └── corpus_cache.py            # TRUE STREAMING cache system
│
├── data/
│   ├── datasets/                  # Corpus (text only)
│   │   ├── tokenizer_corpus/
│   │   └── tiny_23M_corpus/
│   ├── models/
│   │   └── tokenizers/
│   │       └── spm_32k/           # Global tokenizer
│   ├── processed/                 # Packed datasets (.bin/.idx)
│   └── sft_processed/             # SFT corpus
│
├── checkpoints/
│   ├── pretrain/{model}/          # Pre-training checkpoints
│   └── sft/{model}/               # SFT checkpoints
│
├── evaluation_results/
│   ├── pretrain/{model}/
│   └── sft/{model}/
│
├── TEST_PLAN_FINAL_23M.md         # Complete testing guide
├── CONFIG_VALUES_VERIFICATION_REPORT.md  # Config validation report
└── CLAUDE.md                      # This file
```

## Development Best Practices

1. **Start with tiny model**: Validate pipeline and hyperparameters before scaling
2. **Monitor VRAM**: Use `nvidia-smi` or `watch -n 1 nvidia-smi` during training
3. **Checkpoint frequently**: Set `save_steps` in config to 10-15% of total steps
4. **Test with `--analyze-only`**: Preview corpus stats before full processing
5. **Validate tokenizer consistency**: Check `data/models/tokenizers/spm_32k/tokenizer_config.json` for SHA256 hash
6. **Use weighted sampling**: Multi-dataset training > single dataset for quality
7. **Enable deterministic mode**: Critical for reproducibility and debugging
8. **Use accelerate with bf16**: `accelerate launch --mixed_precision bf16`
9. **Track with logs**: Always use `2>&1 | tee logs/{step}.log`
10. **Watch progress tracking**: Monitor equivalent epochs during evaluation to catch overfitting early

## Performance Targets

| Model | Perplexity (WikiText-2) | BoolQ Accuracy | Tokens/sec (RTX 4090) | VRAM (BF16) |
|-------|------------------------|----------------|----------------------|-------------|
| Tiny (23M) | 45-70 | 58-68% (pretrain)<br>68-75% (SFT) | ~600-800 | ~8-12 GB (pretrain)<br>~6-8 GB (SFT LoRA) |
| Small (42M) | 30-50 | 65-72% | ~400-600 | ~12-16 GB |
| Base (124M) | 20-35 | 70-78% | ~200-300 | ~16-24 GB |

## Additional Documentation

- [README.md](README.md): Complete usage guide and tutorials
- [TEST_PLAN_FINAL_23M.md](TEST_PLAN_FINAL_23M.md): Complete step-by-step testing guide for 23M model
- [MIGRATION.md](MIGRATION.md): Migration guide from old script naming to new numbering system

## Notes on Script Organization

**Script Naming Convention**:
- 10-series: Data preparation (corpus, tokenizer, packing)
- 40-series: Training (pretrain, SFT, DPO)
- 50-80-series: Evaluation
- 100-series: Serving

**Recent Changes** (2025-01):
- ✅ Scripts renumbered: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
- ✅ Configs: Hierarchical structure (architectures/, pretrain/, sft/, evaluation/)
- ✅ Tokenizer: SHA256 validation everywhere
- ✅ Multi-dataset: Weighted sampling with deterministic reproducibility
- ✅ Progress tracking: Equivalent epochs displayed during evaluation
- ✅ Complete state preservation: Perfect resumability from checkpoints

**Key Implementation Details**:
- All scripts use `--config` parameter for main configuration
- Arguments use underscores: `--data_dirs`, `--model_path`, `--tokenizer_dir`
- Training scripts use accelerate: `accelerate launch --mixed_precision bf16`
- All hyperparameters loaded from JSON configs (no hardcoded values)
- Logging with `--log-level INFO` for detailed output
