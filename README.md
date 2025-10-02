# 🤖 Lumi-Lab - Production-Grade Mini-LLM Training Pipeline

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/barbidou)

A complete, **config-driven** codebase for training mini-LLMs (decoder-only transformers) optimized for personal machines with RTX 4090 GPUs. Implements a modular pipeline: **corpus preparation**, **tokenizer training**, **pre-training**, **SFT (Supervised Fine-Tuning)**, and **DPO (Direct Preference Optimization)**.

> **💡 Works with more modest GPUs too!** While optimized for RTX 4090 (16GB), this pipeline works on any NVIDIA GPU with CUDA support (even 8GB cards like RTX 3060). You'll just need more time and patience - reduce batch sizes, use gradient accumulation, and maybe grab an extra coffee ☕ while training.

## ✨ Key Features

- ⚙️ **100% Config-Driven**: All hyperparameters in JSON configs (WebUI-ready architecture)
- 🔧 **Modular Pipeline**: Corpus → Tokenizer → Packing → Training → Evaluation
- 🌐 **Multi-Dataset Training**: Weighted sampling for curriculum learning
- 🎯 **Chinchilla-Optimal**: Scientifically-based token budgets (20 tokens/parameter)
- 🔒 **Frozen Tokenizer System**: One global tokenizer prevents subtle bugs
- 📊 **Industrial SFT**: Two-stage approach with LoRA for memory efficiency
- ⚡ **FlashAttention-2**: ~50% VRAM reduction, automatic fallback to SDPA
- 🔄 **Production Checkpoints**: Complete state preservation (RNG, optimizer, scheduler)
- 🎲 **Deterministic Training**: Perfect reproducibility for research

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [⚙️ Installation](#️-installation)
- [🔒 One Tokenizer to Rule Them All](#-one-tokenizer-to-rule-them-all)
- [🏗️ Project Structure](#️-project-structure)
- [🧠 Architecture](#-architecture)
- [📊 Model Configurations](#-model-configurations)
- [🔄 Complete Training Pipeline](#-complete-training-pipeline)
- [📖 Detailed Usage Guide](#-detailed-usage-guide)
- [🎯 RTX 4090 Optimizations](#-rtx-4090-optimizations)
- [🔧 Troubleshooting](#-troubleshooting)
- [📈 Monitoring and Evaluation](#-monitoring-and-evaluation)

## 🚀 Quick Start

### Complete Pipeline - Tiny 23M Model (Chinchilla-Optimal)

```bash
# 0. Setup environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
mkdir -p logs

# 1. Prepare tokenizer corpus (100M tokens from 4 datasets)
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/tokenizer_training_mix.json \
    --output-dir data/datasets/tokenizer_corpus 

# 2. Train global tokenizer (32K vocab - ONLY ONCE!)
python scripts/20_train_tokenizer.py \
    --config config/pretrain/tokenizer/spm32k.json \
    --output-dir data/models/tokenizers/spm_32k

# 3. Prepare pretrain corpus (600M tokens, Chinchilla-optimal)
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/tiny_23M_chinchilla_500M.json \
    --output-dir data/datasets/tiny_23M_corpus 

# 4. Pack dataset (tokenize + create sequences)
python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/tiny_23M_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/tiny_23M_1024

# 5. Pre-train model (23M params, ~2-3 hours on RTX 4090)
accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/tiny_23M_1024 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain

# 6. Evaluate
python scripts/50_evaluate_pretrain.py \
    --model_path checkpoints/pretrain/tiny/final \
    --tokenizer_path data/models/tokenizers/spm_32k \
    --output_dir evaluation_results/pretrain
```

**Expected Results**:
- Loss: ~3.0-3.5 (600M tokens)
- Perplexity: ~20-35 on validation
- Training time: ~2-3 hours (RTX 4090, FlashAttention-2)
- VRAM: ~6-8 GB

### Fast Testing Pipeline (Demo Dataset)

```bash
# Quick validation with tiny dataset (~10M tokens, 15 minutes)
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/demo_fast.json \
    --output-dir data/datasets/demo_corpus

python scripts/20_train_tokenizer.py \
    --config config/pretrain/tokenizer/spm32k.json \
    --output-dir data/models/tokenizers/spm_32k

python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/demo_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/demo_1024

accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/demo_1024 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/demo \
    --num_workers 2
```

## ⚙️ Installation

### System Requirements

- **GPU**: RTX 4090 (24 GB VRAM) recommended
  - Also works on **more modest GPUs** (RTX 3060 8GB, RTX 3070 8GB, etc.)
  - You'll just need to reduce batch sizes and be patient ☕
- **RAM**: 32 GB recommended (minimum 16 GB)
- **Python**: 3.11+
- **CUDA**: 11.8+ or 12.1+
- **Disk Space**: 100 GB free (datasets + checkpoints)

### Quick Install

```bash
git clone https://github.com/your-username/Lumi-Lab.git
cd Lumi-Lab
pip install -r requirements.txt
```

### FlashAttention-2 Installation (Recommended)

FlashAttention-2 provides **~50% VRAM reduction** and **~30% speedup**:

```bash
# For CUDA 12.1+
pip install flash-attn --no-build-isolation

# For CUDA 11.8
pip install flash-attn==2.5.8 --no-build-isolation
```

**Automatic Fallback**: If FlashAttention-2 is not available, Lumi-Lab automatically falls back to PyTorch SDPA (Scaled Dot-Product Attention). No code changes needed.

**Requirements**:
- CUDA 11.6+ or CUDA 12.x
- PyTorch 2.0+ with CUDA
- GPU architecture SM 8.0+ (RTX 30/40 series)

**Installation Help**: [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import __version__; print(f'Transformers: {__version__}')"
python -c "from accelerate import __version__; print(f'Accelerate: {__version__}')"
```

**Expected output**:
```
PyTorch: 2.3.0+cu121
CUDA: True
Transformers: 4.40.0+
Accelerate: 0.30.0+
```

## 🔒 One Tokenizer to Rule Them All

Lumi-Lab implements a **frozen tokenizer system** that ensures perfect consistency across all datasets and prevents subtle bugs from tokenizer mismatches.

### 🎯 Why This Matters

Traditional multi-dataset training often fails silently due to:
- ❌ Different tokenizers across datasets → incompatible token spaces
- ❌ Vocabulary size mismatches → index out of bounds errors
- ❌ Special token inconsistencies → corrupted training signals
- ❌ Silent degradation → models that "work" but perform poorly

**Our Solution**: Train one global tokenizer once, freeze it, use it everywhere.

### 🔨 Workflow

```bash
# 1. Train the global tokenizer (ONLY ONCE at project start)
python scripts/20_train_tokenizer.py \
    --config config/pretrain/tokenizer/spm32k.json \
    --output-dir data/models/tokenizers/spm_32k

# 2. Prepare all datasets with the frozen tokenizer
python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/corpus_A \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/corpus_A_1024

python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/corpus_B \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/corpus_B_1024

# 3. Train with guaranteed consistency
accelerate launch scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/corpus_A_1024 data/processed/corpus_B_1024 \
    --data_weights 0.5 0.5 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain
```

### 🛡️ Built-in Safety Features

- **SHA256 Verification**: Every dataset includes tokenizer hash
- **Blocking Errors**: Training fails immediately on mismatch (no silent bugs)
- **Automatic Validation**: Checked at dataset load time
- **Clear Error Messages**: Exact fix instructions provided

### 📄 Tokenizer Metadata

Every packed dataset includes verification metadata in `final_manifest.json`:

```json
{
  "tokenizer_config_hash": "a1b2c3d4e5f6...",
  "statistics": {
    "vocab_size": 32768,
    "total_tokens": 600000824,
    "total_sequences": 585938
  }
}
```

### ❌ Common Mistakes (Now Prevented!)

```bash
# ❌ OLD WAY: Inconsistent tokenizers
python scripts/10_prepare_corpus.py --config dataset_A.json  # Creates tokenizer A
python scripts/10_prepare_corpus.py --config dataset_B.json  # Creates tokenizer B
# → Silent incompatibility, degraded training

# ✅ NEW WAY: One frozen tokenizer
python scripts/20_train_tokenizer.py --config tokenizer_config.json  # THE tokenizer
python scripts/30_pack_dataset.py --tokenizer-dir tokenizer/ ...     # Uses THE tokenizer
python scripts/30_pack_dataset.py --tokenizer-dir tokenizer/ ...     # Uses THE tokenizer
# → Guaranteed consistency
```

## 🏗️ Project Structure

```
Lumi-Lab/
├── config/                                # All configurations (100% config-driven)
│   ├── architectures/                     # Model architectures
│   │   ├── tiny.json                     # 23M params (6 layers, 256 dim)
│   │   ├── small.json                    # 42M params (12 layers, 512 dim)
│   │   └── base.json                     # 124M params (24 layers, 768 dim)
│   ├── pretrain/                          # Pre-training configs
│   │   ├── corpus/                        # Corpus preparation configs
│   │   │   ├── tokenizer_training_mix.json      # 100M tokens for tokenizer
│   │   │   ├── tiny_23M_chinchilla_500M.json    # 600M tokens (Chinchilla-optimal)
│   │   │   ├── model_124M_phase_a.json          # Phase A: high-quality (1.5B tokens)
│   │   │   └── model_124M_phase_b.json          # Phase B: diversity (1.5B tokens)
│   │   ├── sources/                       # Data source configs
│   │   │   ├── c4_english.json
│   │   │   ├── gutenberg_books.json
│   │   │   ├── fineweb_edu.json
│   │   │   └── vietgpt_wikipedia.json
│   │   ├── tokenizer/                     # Tokenizer training
│   │   │   └── spm32k.json               # 32K vocab SentencePiece
│   │   ├── packing/                       # Dataset packing
│   │   │   └── default.json              # Sequence length, split ratio
│   │   └── training/                      # Training hyperparameters
│   │       └── chinchilla_tiny_500m.json # Complete training config
│   ├── sft/                               # Supervised fine-tuning
│   │   ├── datasets/                      # SFT dataset configs
│   │   │   ├── alpaca.json
│   │   │   ├── dolly15k.json
│   │   │   ├── oasst1.json
│   │   │   └── mixed_sft.json
│   │   └── training/                      # SFT training configs (LoRA)
│   │       ├── lora_tiny_23m.json
│   │       ├── lora_small.json
│   │       └── lora_base.json
│   ├── evaluation/                        # Evaluation configs
│   │   ├── sft_standard.json
│   │   └── smoke_test_prompts.json
│   └── dpo/                               # DPO configs (future)
│
├── scripts/                               # Training pipeline scripts
│   ├── 10_prepare_corpus.py              # Clean, deduplicate, shard corpus
│   ├── 20_train_tokenizer.py             # Train SentencePiece tokenizer (32K vocab)
│   ├── 30_pack_dataset.py                # Tokenize + pack into sequences
│   ├── 40_pretrain.py                    # Pre-train from scratch (multi-dataset)
│   ├── 50_evaluate_pretrain.py           # Evaluate pretrained model
│   ├── 60_prepare_sft_corpus.py          # Prepare SFT corpus (templates)
│   ├── 70_train_sft.py                   # SFT with LoRA adapters
│   ├── 80_evaluate_sft.py                # Evaluate SFT model
│   ├── 90_train_dpo.py                   # DPO alignment (future)
│   └── 100_serve.py                      # Inference server (CLI + API)
│
├── utils/                                 # Shared utilities
│   ├── dataset_utils.py                  # PackedDataset, MultiDatasetSampler
│   ├── model_utils.py                    # Model creation, checkpoints
│   ├── sft_templates.py                  # ChatML, Instruct, Alpaca templates
│   ├── sft_evaluation.py                 # SFT-specific evaluation
│   └── debug/                            # Debug utilities
│       ├── corpus_cache.py
│       ├── tokenizer_validation.py
│       └── adaptive_estimation.py
│
├── data/                                  # Data directory (created automatically)
│   ├── datasets/                          # Raw corpus (cleaned, sharded)
│   │   ├── tokenizer_corpus/             # 100M tokens for tokenizer training
│   │   └── tiny_23M_corpus/              # 600M tokens for pretraining
│   ├── models/                            # Trained models
│   │   └── tokenizers/                   # Tokenizers
│   │       └── spm_32k/                  # 32K vocab SentencePiece
│   │           ├── spm32k.model
│   │           ├── spm32k.vocab
│   │           ├── tokenizer_config.json
│   │           └── TOKENIZER_CARD.md     # Tokenizer documentation
│   └── processed/                         # Packed datasets (tokenized sequences)
│       ├── tiny_23M_1024/                # 585K sequences, 1024 tokens each
│       └── demo_1024/                    # Demo dataset
│
├── checkpoints/                           # Model checkpoints
│   ├── pretrain/                          # Pretrained models
│   │   └── tiny/
│   │       ├── step_5000/
│   │       ├── step_10000/
│   │       └── final/
│   └── sft/                               # Fine-tuned models
│       └── tiny_lora/
│
├── evaluation_results/                    # Evaluation outputs
│   ├── pretrain/
│   └── sft/
│
├── logs/                                  # Training logs
│
├── CLAUDE.md                              # Claude Code integration guide
├── ARCHITECTURE.md                        # Technical architecture details
├── CHINCHILLA_PIPELINE.md                 # Chinchilla-optimal training strategy
├── SFT_INDUSTRIAL_PIPELINE.md             # Industrial SFT documentation
├── Makefile                               # Convenience commands
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## 🧠 Architecture

### LLaMA-like Decoder-Only Transformer

Lumi-Lab implements a modern decoder-only transformer following **LLaMA design principles**:

#### Core Components

- **🔧 RMSNorm**: Root Mean Square normalization (faster, more stable than LayerNorm)
- **⚡ SwiGLU**: Swish-Gated Linear Units activation (superior to ReLU/GELU)
- **📐 RoPE**: Rotary Position Embeddings (better length extrapolation)
- **🚫 No Bias Terms**: Cleaner scaling, fewer parameters
- **🎯 Causal Attention**: Autoregressive text generation
- **💾 FlashAttention-2**: Memory-efficient attention with ~50% VRAM savings

#### Why LLaMA Architecture?

- ✅ **Proven**: State-of-the-art results from 6M to 70B+ parameters
- ✅ **Efficient**: Optimized for modern GPU training
- ✅ **Scalable**: Consistent design principles across all model sizes
- ✅ **Compatible**: Full HuggingFace Transformers integration
- ✅ **Memory-Optimal**: RMSNorm + no bias + FlashAttention-2

**📖 For complete architectural details, see [ARCHITECTURE.md](ARCHITECTURE.md)**

## 📊 Model Configurations

| Model | Layers | d_model | Heads | head_dim | FFN | Parameters | VRAM (bf16) | Chinchilla Tokens |
|-------|--------|---------|-------|----------|-----|------------|-------------|-------------------|
| **tiny** | 6 | 256 | 4 | 64 | 1024 | 23M | ~6-8 GB | ~500M (20×) |
| **small** | 12 | 512 | 8 | 64 | 2048 | 42M | ~8-12 GB | ~840M (20×) |
| **base** | 24 | 768 | 12 | 64 | 3072 | 124M | ~12-18 GB | ~2.5B (20×) |

### Scaling Strategy

**Consistent design principles**:
- `head_dim = 64` (constant across all sizes)
- `d_ff = d_model × 4` (FFN ratio = 4.0)
- `vocab_size = 32768` (SentencePiece)
- `sequence_length = 1024` (adjustable)

### Chinchilla-Optimal Training

Based on [Chinchilla paper](https://arxiv.org/abs/2203.15556), optimal token count ≈ **20 tokens per parameter**:

- **23M model**: ~500M tokens (600M for safety margin)
- **42M model**: ~840M tokens (~1B recommended)
- **124M model**: ~2.5B tokens (~3B recommended)

Training with fewer tokens = underfitting. Training with more = diminishing returns.

**📖 For complete Chinchilla strategy, see [CHINCHILLA_PIPELINE.md](CHINCHILLA_PIPELINE.md)**

## 🔄 Complete Training Pipeline

### Pipeline Overview

```
┌─────────────────────┐
│   10_prepare_corpus │  ← Clean, deduplicate, shard
│   (Raw Text → JSONL)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 20_train_tokenizer  │  ← Train SentencePiece (ONCE!)
│   (JSONL → .model)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  30_pack_dataset    │  ← Tokenize + pack sequences
│ (JSONL → .bin+.idx) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    40_pretrain      │  ← Pre-train model (multi-dataset)
│   (Data → Model)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│50_evaluate_pretrain │  ← Perplexity, benchmarks
│  (Model → Metrics)  │
└─────────────────────┘
```

### Stage 1: Corpus Preparation

**Purpose**: Transform raw datasets into cleaned, deduplicated, sharded text corpus.

```bash
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/tiny_23M_chinchilla_500M.json \
    --output-dir data/datasets/tiny_23M_corpus \
    --shard-size 10000 \
    --log-level INFO
```

**Operations**:
- Loading from HuggingFace datasets
- Text cleaning (control chars, URLs, duplicates)
- MinHash deduplication (configurable threshold)
- Quality filtering (length, language detection)
- Sharding (memory-efficient processing)

**Output**: `data/datasets/tiny_23M_corpus/`
- Sharded JSONL files (`shard_0000.jsonl`, `shard_0001.jsonl`, ...)
- `manifest.json` (metadata, statistics)
- `DATA_CARD.md` (documentation)

### Stage 2: Tokenizer Training

**Purpose**: Train a single global SentencePiece tokenizer (32K vocabulary).

```bash
python scripts/20_train_tokenizer.py \
    --config config/pretrain/tokenizer/spm32k.json \
    --output-dir data/models/tokenizers/spm_32k \
    --log-level INFO
```

**Critical**: Run this **ONLY ONCE** at project start. All datasets must use the same frozen tokenizer.

**Output**: `data/models/tokenizers/spm_32k/`
- `spm32k.model` (SentencePiece model)
- `spm32k.vocab` (vocabulary)
- `tokenizer_config.json` (HuggingFace format)
- `TOKENIZER_CARD.md` (documentation + SHA256 hash)

**Quality Metrics**:
- Compression ratio: ~3.5-4.5 chars/token
- UNK rate: <0.1%
- Overall quality: 95-100/100

### Stage 3: Dataset Packing

**Purpose**: Tokenize corpus and pack into fixed-length sequences for training.

```bash
python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/tiny_23M_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/tiny_23M_1024 \
    --log-level INFO
```

**Operations**:
- Token encoding with frozen tokenizer
- Sequence packing (length=1024)
- Train/val split (95/5 default)
- Binary format (.bin + .idx) for memory-mapped loading
- Tokenizer SHA256 verification

**Output**: `data/processed/tiny_23M_1024/`
- `train.bin` + `train.idx` (training data)
- `val.bin` + `val.idx` (validation data)
- `final_manifest.json` (metadata + tokenizer hash)

### Stage 4: Pre-training

**Purpose**: Train the language model from scratch.

```bash
accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/tiny_23M_1024 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain \
    --num_workers 4 \
    --log-level INFO
```

**Features**:
- **Multi-dataset training**: Weighted sampling for curriculum learning
- **Deterministic training**: Full RNG state preservation
- **Auto-resume**: Resumes from latest checkpoint automatically
- **Progress tracking**: Shows equivalent epochs, overfitting detection
- **Evaluation**: Validation perplexity every `eval_steps`
- **Checkpointing**: Complete state saved every `save_steps`

**Expected Metrics** (23M model, 600M tokens):
- Training time: ~2-3 hours (RTX 4090, FlashAttention-2)
- Final loss: ~3.0-3.5
- Validation perplexity: ~20-35
- VRAM usage: ~6-8 GB (bf16 + FlashAttention-2)

**Multi-Dataset Example**:
```bash
accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/corpus_A_1024 data/processed/corpus_B_1024 \
    --data_weights 0.7 0.3 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain \
    --num_workers 4
```

This trains with **70% corpus A, 30% corpus B** (deterministic weighted sampling).

**Resume Training**:
```bash
# Automatic: finds latest checkpoint
accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/tiny_23M_1024 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain \
    --resume_from_checkpoint auto

# Manual: specify checkpoint path
accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/tiny_23M_1024 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain \
    --resume_from_checkpoint checkpoints/pretrain/tiny/step_10000
```

### Stage 5: Evaluation

**Purpose**: Evaluate pretrained model with perplexity and benchmarks.

```bash
python scripts/50_evaluate_pretrain.py \
    --model_path checkpoints/pretrain/tiny/final \
    --tokenizer_path data/models/tokenizers/spm_32k \
    --output_dir evaluation_results/pretrain/tiny \
    --log-level INFO
```

**Metrics**:
- **WikiText-2 Perplexity**: Standard language modeling metric
- **BoolQ Zero-shot**: Yes/No reasoning capability
- **Generation Quality**: Coherence, fluency, factuality

**Expected Results** (23M model):
- Perplexity: ~20-35 (WikiText-2)
- BoolQ Accuracy: ~55-65% (random = 50%)

## 📖 Detailed Usage Guide

### Config-Driven Philosophy

**All hyperparameters are in JSON configs**. The CLI only accepts:
- ✅ **Paths**: `--config`, `--data_dirs`, `--tokenizer_dir`, `--output_dir`
- ✅ **Runtime flags**: `--num_workers`, `--log-level`, `--resume_from_checkpoint`
- ❌ **No hyperparameters**: `--learning_rate`, `--batch_size`, etc. → Use config files!

**Why?**
- **WebUI-Ready**: Easy to build UI on top (just file pickers + buttons)
- **Reproducible**: All experiments fully documented in configs
- **Version-Controlled**: Configs tracked in git, hyperparameters never lost
- **No CLI Hell**: No 50-argument commands

### Multi-Dataset Training Strategy

**Curriculum Learning**: Train on multiple datasets with different characteristics.

```bash
# 1. Prepare multiple corpuses
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/high_quality.json \
    --output-dir data/datasets/high_quality_corpus

python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/diverse.json \
    --output-dir data/datasets/diverse_corpus

# 2. Pack both with the same tokenizer
python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/high_quality_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/high_quality_1024

python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/diverse_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/diverse_1024

# 3. Train with weighted sampling (70% high-quality, 30% diverse)
accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/high_quality_1024 data/processed/diverse_1024 \
    --data_weights 0.7 0.3 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain_multidataset \
    --num_workers 4
```

**Benefits**:
- Better generalization
- Reduced overfitting
- Curriculum learning (easy → hard)
- Domain adaptation

### Supervised Fine-Tuning (SFT)

**Two-stage industrial SFT pipeline** with LoRA for memory efficiency.

#### Stage 1: Prepare SFT Corpus

```bash
python scripts/60_prepare_sft_corpus.py \
    --config config/sft/datasets/alpaca.json \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/sft_processed/alpaca_1024 \
    --log-level INFO
```

**Features**:
- Conversation templates (ChatML, Instruct, Alpaca)
- Quality filtering
- Tokenization with frozen global tokenizer
- Proper label masking (only train on assistant responses)

#### Stage 2: Train with LoRA

```bash
accelerate launch --mixed_precision bf16 scripts/70_train_sft.py \
    --config config/sft/training/lora_tiny_23m.json \
    --model_path checkpoints/pretrain/tiny/final \
    --data_dirs data/sft_processed/alpaca_1024 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/sft/tiny_alpaca \
    --log-level INFO
```

**LoRA Benefits**:
- ~95% VRAM reduction vs full fine-tuning
- Fast training (10-20x speedup)
- Modular adapters (swap for different use cases)
- No catastrophic forgetting of pretrain knowledge

**Multi-Dataset SFT**:
```bash
accelerate launch --mixed_precision bf16 scripts/70_train_sft.py \
    --config config/sft/training/lora_tiny_23m.json \
    --model_path checkpoints/pretrain/tiny/final \
    --data_dirs data/sft_processed/alpaca_1024 data/sft_processed/oasst1_1024 \
    --data_weights 0.6 0.4 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/sft/tiny_mixed \
    --num_workers 4
```

**📖 For complete SFT guide, see [SFT_INDUSTRIAL_PIPELINE.md](SFT_INDUSTRIAL_PIPELINE.md)**

### Inference and Serving

#### Interactive CLI

```bash
python scripts/100_serve.py \
    --model_path checkpoints/sft/tiny_alpaca/final \
    --tokenizer_path data/models/tokenizers/spm_32k \
    --mode chat \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_length 256
```

**Modes**:
- `chat`: Interactive conversational interface
- `generate`: Single prompt generation
- `api`: REST API server (FastAPI)

#### REST API Server

```bash
python scripts/100_serve.py \
    --model_path checkpoints/sft/tiny_alpaca/final \
    --tokenizer_path data/models/tokenizers/spm_32k \
    --mode api \
    --port 8000
```

**Example Request**:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms.",
    "max_length": 200,
    "temperature": 0.7
  }'
```

## 🎯 GPU Optimizations & Lower-End Hardware

Lumi-Lab is optimized for RTX 4090 (24 GB VRAM) but **works on any NVIDIA GPU with CUDA support**. Even 8GB cards like RTX 3060 or RTX 3070 will work - you'll just need to adjust batch sizes, use gradient accumulation, and grab more coffee ☕ while training takes longer.

### Memory Optimization Techniques

| Technique | VRAM Reduction | Speed Impact | When to Use |
|-----------|----------------|--------------|-------------|
| **FlashAttention-2** | ~50% | +30% speed | Always (if available) |
| **bf16 Mixed Precision** | ~50% | +20% speed | Always (default) |
| **Gradient Checkpointing** | ~40% | -10% speed | Large models or OOM |
| **LoRA (SFT)** | ~95% | +200% speed | SFT stage |
| **Gradient Accumulation** | Variable | Neutral | Increase effective batch size |

### VRAM Usage by Model Size

| Model | Parameters | Batch=8 (bf16 + Flash) | Batch=4 (bf16 + Flash) | Batch=2 (bf16 + Flash) | Works on 8GB GPU? |
|-------|------------|------------------------|------------------------|------------------------|-------------------|
| **tiny** | 23M | ~6-8 GB | ~4-5 GB | ~3-4 GB | ✅ Yes |
| **small** | 42M | ~8-12 GB | ~5-7 GB | ~4-5 GB | ✅ Yes (batch=2) |
| **base** | 124M | ~12-18 GB | ~8-10 GB | ~6-7 GB | ⚠️ Tight (batch=1-2) |

**Note for 8GB GPUs**: Start with `tiny` model, batch_size=2-4, gradient_accumulation=8-16. It will work, just take longer!

### Troubleshooting OOM (Out of Memory)

If you encounter OOM errors:

1. **Enable FlashAttention-2** (if not already):
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Reduce batch size** in config:
   ```json
   {
     "training_params": {
       "per_device_train_batch_size": 4,  // Was 8
       "gradient_accumulation_steps": 8   // Was 4 (keeps effective batch size)
     }
   }
   ```

3. **Enable gradient checkpointing** in config:
   ```json
   {
     "hardware_params": {
       "gradient_checkpointing": true
     }
   }
   ```

4. **Use smaller sequence length**:
   ```json
   {
     "packing_params": {
       "sequence_length": 512  // Was 1024
     }
   }
   ```

5. **Reduce num_workers** (dataloader workers use RAM):
   ```bash
   --num_workers 2  # Or even 1
   ```

### Performance Tuning

**For Maximum Speed**:
```json
{
  "hardware_params": {
    "use_flash_attn": true,
    "bf16": true,
    "gradient_checkpointing": false,
    "dataloader_num_workers": 4
  },
  "training_params": {
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 2
  }
}
```

**For Minimum VRAM**:
```json
{
  "hardware_params": {
    "use_flash_attn": true,
    "bf16": true,
    "gradient_checkpointing": true,
    "dataloader_num_workers": 1
  },
  "training_params": {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8
  }
}
```

**Effective Batch Size** = `batch_size × gradient_accumulation_steps × num_gpus`

Example: `4 × 8 × 1 = 32` (same as batch=32 in one step, but uses 1/8th the memory)

## 🔧 Troubleshooting

### Common Issues

#### 1. Tokenizer Mismatch Error

```
ValueError: Tokenizer mismatch between datasets
   Dataset hash:   a1b2c3d4...
   Tokenizer hash: e5f6g7h8...
```

**Fix**:
```bash
# Re-pack the dataset with the correct tokenizer
python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/problem_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/problem_dataset_1024
```

#### 2. CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Fix**: See [RTX 4090 Optimizations](#-rtx-4090-optimizations) section above.

#### 3. FlashAttention Not Available

```
WARNING: FlashAttention-2 not available, falling back to SDPA
```

This is OK! SDPA (PyTorch native) works fine, just uses more VRAM.

**To fix** (optional):
```bash
pip install flash-attn --no-build-isolation
```

#### 4. NaN/Inf Loss During Training

```
Step 1234: loss=nan, lr=3.0e-04
```

**Causes**:
- Learning rate too high
- Mixed precision instability
- Bad data

**Fix**:
1. **Reduce learning rate** in config:
   ```json
   {
     "training_params": {
       "learning_rate": 1e-4  // Was 3e-4
     }
   }
   ```

2. **Increase gradient clipping**:
   ```json
   {
     "training_params": {
       "max_grad_norm": 0.5  // Was 1.0
     }
   }
   ```

3. **Check data quality**: Inspect corpus for corrupted samples

#### 5. Resume Training Not Working

```
FileNotFoundError: checkpoint not found
```

**Fix**:
```bash
# Use auto-detection
accelerate launch scripts/40_pretrain.py \
    --config config.json \
    --data_dirs data/processed/dataset \
    --tokenizer_dir tokenizer/ \
    --output_dir checkpoints/pretrain \
    --resume_from_checkpoint auto  # Finds latest automatically

# Or specify exact path
--resume_from_checkpoint checkpoints/pretrain/tiny/step_10000
```

### Getting Help

If you encounter issues:

1. **Check logs**: `logs/` directory contains detailed training logs
2. **Check configs**: Verify all paths and hyperparameters
3. **Check GPU**: `nvidia-smi` to monitor VRAM/utilization
4. **Check documentation**: `CLAUDE.md`, `ARCHITECTURE.md`, `CHINCHILLA_PIPELINE.md`
5. **Open an issue**: [GitHub Issues](https://github.com/your-username/Lumi-Lab/issues)

## 📈 Monitoring and Evaluation

### Training Metrics

Monitor these metrics during training:

- **Loss**: Should decrease steadily (target: ~3.0-3.5 for 600M tokens)
- **Learning Rate**: Should increase during warmup, then decay
- **Perplexity**: Should decrease (target: ~20-35 for 23M model)
- **Equivalent Epochs**: Watch for overfitting after ~3-4 epochs
- **VRAM Usage**: Should be stable (no memory leaks)

### Progress Tracking During Training

During training, you'll see:

```
Step 2500: loss=4.23, lr=2.5e-04
📊 Evaluation...
   Progress: 2500/30000 steps (8.3%), ~1.28 epochs
   Validation perplexity: 45.32

Step 5000: loss=3.87, lr=3.0e-04
💾 Checkpoint saved at step 5000 -> checkpoints/pretrain/tiny/step_5000
📊 Evaluation...
   Progress: 5000/30000 steps (16.7%), ~2.56 epochs
   Validation perplexity: 38.15
```

**Progress Information**:
- Current step / total steps
- Progress percentage
- **Equivalent epochs** (for overfitting detection!)
- Validation perplexity

**Overfitting Detection**:
- If perplexity **increases** after ~3-4 equivalent epochs → stop early
- If perplexity **plateaus** → consider more data or longer training
- If perplexity **decreases steadily** → continue training

### Evaluation Benchmarks

**Perplexity (WikiText-2)**:
- Random model: >10,000
- Undertrained: 100-200
- Good: 20-50
- Excellent: <20

**BoolQ Zero-shot**:
- Random: 50%
- Undertrained: 50-55%
- Good: 60-70%
- Excellent: >70%

### Expected Performance Targets

| Model | Training Time | Final Loss | Perplexity | BoolQ | VRAM |
|-------|--------------|------------|------------|-------|------|
| **tiny** (23M, 600M tokens) | 2-3h | 3.0-3.5 | 20-35 | 55-65% | 6-8 GB |
| **small** (42M, 1B tokens) | 8-12h | 2.5-3.0 | 15-25 | 65-75% | 8-12 GB |
| **base** (124M, 3B tokens) | 24-48h | 2.0-2.5 | 10-20 | 70-80% | 12-18 GB |

## 📚 Additional Documentation

- **[CLAUDE.md](CLAUDE.md)**: Complete command reference for Claude Code integration
- **[GEMINI.md](GEMINI.md)**: Complete command reference for Gemini CLI integration
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Technical architecture details
- **[CHINCHILLA_PIPELINE.md](CHINCHILLA_PIPELINE.md)**: Chinchilla-optimal training strategy
- **[SFT_INDUSTRIAL_PIPELINE.md](SFT_INDUSTRIAL_PIPELINE.md)**: Industrial SFT guide
- **[BREAKING_CHANGES.md](BREAKING_CHANGES.md)**: Migration guide from old versions

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Good Luck With That Public License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LLaMA** (Meta AI): Architecture inspiration
- **FlashAttention-2** (Dao et al.): Memory-efficient attention
- **Chinchilla** (DeepMind): Optimal scaling laws
- **HuggingFace**: Transformers library
- **PyTorch Team**: PyTorch framework

## ☕ Support the Project

If you find Lumi-Lab useful and want to support continued development, consider [buying me a coffee ☕](https://ko-fi.com/barbidou)!

This project is built with limited resources (one laptop and lots of coffee) and your support helps me spend more time:
- 🔧 Improving the pipeline and documentation
- ⚡ Adding new features and optimizations
- 📚 Creating educational content and tutorials
- ☕ Keeping the coffee machine running

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/barbidou)

## 📧 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/babridoux/Lumi-Lab/issues)
- **Discussions**: [Join the community](https://github.com/barbidoux/Lumi-Lab/discussions)
- **Ko-fi**: [Support the project ☕](https://ko-fi.com/barbidou)

---

**Built with ❤️ (and ☕) for the ML community**

> *"I don't have an excavator, I have a fork. But technical complexity has never been an obstacle, only an exciting puzzle. I would rather dig happily with a fork than wait for someone to hand me a bulldozer."*

*Train your own mini-LLM, learn the fundamentals, and have fun!*
