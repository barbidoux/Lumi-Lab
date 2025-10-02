# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Lumi-Lab is a complete mini-LLM training pipeline implementing a modern LLaMA-like decoder-only transformer architecture. It provides a modular, production-grade system for pre-training, supervised fine-tuning (SFT), and direct preference optimization (DPO) of language models, optimized for RTX 4090 GPUs.

## Essential Commands

### Setup & Environment
```bash
make install                    # Install all dependencies
make check-env                  # Verify CUDA, PyTorch, Transformers
```

### Data Preparation Pipeline
```bash
# CRITICAL: The "One Tokenizer to Rule Them All" approach
make tokenizer-train-mix        # Train global tokenizer ONCE (required first step)
make prepare-wiki-with-tokenizer   # Prepare Wikipedia with frozen tokenizer
make prepare-owt-with-tokenizer    # Prepare OpenWebText with frozen tokenizer
make data-rebuild-all           # Full pipeline: tokenizer + all datasets

# Individual corpus preparation (modular pipeline)
python scripts/01_prepare_corpus.py --config config/datasets/sources/c4_english.json --output_dir data/corpus/c4
python scripts/02_train_tokenizer.py --config config/datasets/training_configs/tokenizer_training_mix.json
python scripts/03_pack_dataset.py --corpus_dir data/corpus/c4 --tokenizer_path data/tokenizer/spm32k.model --output_dir data/processed/c4_32k_1024
```

### Pre-training
```bash
# Single dataset training
make pretrain-tiny              # Train tiny model (6M params, ~2-4h)
make pretrain-small             # Train small model (42M params, ~8-12h)
make pretrain-base              # Train base model (124M params, ~24-48h)

# Multi-dataset training with weighted sampling
accelerate launch scripts/04_pretrain.py \
    --config config/tiny.json \
    --data_dirs data/processed/wiki_32k_1024 data/processed/owt_32k_1024 \
    --data_weights 0.3 0.7 \
    --max_steps 20000

# Resume from checkpoint
accelerate launch scripts/04_pretrain.py \
    --config config/tiny.json \
    --data_dirs data/processed/wiki_32k_1024 \
    --resume_from_checkpoint checkpoints/pretrain/tiny/step_10000
```

### SFT (Supervised Fine-Tuning)
```bash
# Industrial SFT pipeline (two-stage approach)
make prepare-sft-corpus         # Stage 1: Prepare SFT corpus with templates
make sft-train-tiny             # Stage 2: Train with LoRA

# Multi-dataset SFT with weighted sampling
accelerate launch scripts/03_sft_industrial.py \
    --config config/sft_training/lora_tiny.json \
    --model_path checkpoints/pretrain/tiny/final \
    --data_dirs data/sft_processed/alpaca_chatml_32k_1024 data/sft_processed/oasst1_chatml_32k_1024 \
    --data_weights 0.7 0.3 \
    --tokenizer_path data/tokenizer/spm32k.model \
    --output_dir checkpoints/sft_industrial/tiny
```

### Evaluation
```bash
make evaluate                   # Complete evaluation with all metrics
make evaluate-quick             # Fast evaluation for development
make assess-performance         # Automated performance analysis

python scripts/05_evaluate.py \
    --model_path checkpoints/pretrain/tiny/final \
    --tokenizer_path data/tokenizer/spm32k.model \
    --output_dir evaluation_results
```

### Testing
```bash
make test-pipeline              # Quick test with synthetic data (standard attention)
make test-pipeline-flash        # Quick test with FlashAttention-2
```

## Architecture Overview

### Modular Pipeline Design

Lumi-Lab implements a **three-stage modular pipeline** that separates concerns for maximum flexibility:

#### Stage 1: Corpus Preparation (`01_prepare_corpus.py`)
- **Purpose**: Transform raw datasets into cleaned, validated text corpus
- **Operations**: Loading, cleaning, deduplication (MinHash), quality filtering, sharding
- **Output**: `data/corpus/{name}/` with JSONL shards + manifest + data card
- **Key**: Tokenizer-independent, pure text processing

#### Stage 2: Tokenizer Training (`02_train_tokenizer.py`)
- **Purpose**: Train a single global SentencePiece tokenizer (vocab_size=32768)
- **Critical**: Only run ONCE. All datasets must use the same frozen tokenizer
- **Output**: `data/tokenizer/spm32k.model` with SHA256 verification
- **Safety**: Built-in consistency validation prevents tokenizer mismatches

#### Stage 3: Dataset Packing (`03_pack_dataset.py`)
- **Purpose**: Tokenize corpus and pack into fixed-length sequences
- **Operations**: Token encoding, sequence packing, train/val split, sharding
- **Output**: `data/processed/{name}_32k_1024/` ready for training
- **Validation**: Tokenizer SHA256 stored in manifest for consistency checks

### Two-Stage SFT Pipeline

The Industrial SFT pipeline mirrors the pre-training architecture:

#### Stage 1: SFT Corpus Preparation (`02_prepare_sft_corpus.py`)
- **Purpose**: Process conversational datasets with template formatting
- **Templates**: ChatML (recommended), Instruct, Chat, Alpaca
- **Output**: Sharded conversation corpus with tokenizer validation

#### Stage 2: SFT Training (`03_sft_industrial.py`)
- **Purpose**: Fine-tune with LoRA adapters for memory efficiency
- **Features**: Multi-dataset weighted sampling, generation evaluation, full state checkpointing
- **Memory**: ~95% reduction vs full fine-tuning (LoRA magic)

### Pre-training Script (`04_pretrain.py`)
- **Purpose**: Train models from scratch with multi-dataset support
- **Features**: Weighted sampling, deterministic training, auto-resume, complete RNG state preservation
- **Optimization**: FlashAttention-2, gradient checkpointing, mixed precision

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

4. **Production-Grade Robustness**:
   - Comprehensive error handling and validation
   - Sharded data loading (unlimited dataset sizes)
   - Memory-efficient streaming from disk
   - Complete logging and metrics

## Key Configuration Files

### Model Architectures
- `config/tiny.json`: 6M params (n_layer=6, d_model=256, n_head=4)
- `config/small.json`: 42M params (n_layer=12, d_model=512, n_head=8)
- `config/base.json`: 124M params (n_layer=24, d_model=768, n_head=12)
- All use: head_dim=64, ffn_ratio=4.0, vocab_size=32768

### SFT Configurations
- `config/sft_datasets/`: Dataset configs (sources, templates, quality filters)
- `config/sft_training/`: Training configs (LoRA settings, hyperparameters, eval prompts)
- Key configs:
  - `lora_optimal_tiny_23m.json`: Optimal config for 23M model (r=32, alpha=64, lr=8e-5)
  - `lora_tiny.json`: Standard tiny model (r=16, alpha=16)
  - `alpaca_chatml.json`: ChatML template for Alpaca dataset

### Dataset Configurations
- `config/datasets/sources/`: Raw corpus sources (C4, Wikipedia, Gutenberg, FineWeb-Edu)
- `config/datasets/training_configs/`: Pre-training configs with token budgets
- `config/datasets/training_configs/tokenizer_training_mix.json`: Global tokenizer config

## Critical Workflows

### Complete Pipeline from Scratch
```bash
# 1. Train global tokenizer (ONCE)
make tokenizer-train-mix

# 2. Prepare all datasets with frozen tokenizer
make prepare-wiki-with-tokenizer
make prepare-owt-with-tokenizer

# 3. Pre-train model
make pretrain-tiny

# 4. Prepare SFT data
make prepare-sft-corpus

# 5. Fine-tune with LoRA
make sft-train-tiny

# 6. Evaluate
make evaluate
```

### Multi-Dataset Training Strategy
```bash
# Phase A: High-quality foundations (1.5B tokens)
accelerate launch scripts/04_pretrain.py \
    --config config/base.json \
    --data_dirs data/processed/wiki_32k_1024 data/processed/books_32k_1024 \
    --data_weights 0.67 0.33 \
    --max_steps 30000

# Phase B: Diversity and robustness (1.5B tokens)
accelerate launch scripts/04_pretrain.py \
    --config config/base.json \
    --data_dirs data/processed/c4_32k_1024 data/processed/forums_32k_1024 \
    --data_weights 0.67 0.33 \
    --max_steps 30000 \
    --resume_from_checkpoint checkpoints/pretrain/base/step_30000
```

### Troubleshooting Tokenizer Issues

**Error: "Tokenizer mismatch between datasets"**
```bash
# This is a BLOCKING error - do not ignore!
# Re-encode the problematic dataset
make reencode-dataset DIR=data/processed/problem_dataset

# Or reset and rebuild everything
make tokenizer-reset  # Type 'YES' to confirm
make data-rebuild-all
```

**Error: "No tokenizer found"**
```bash
# You forgot to train the tokenizer first
make tokenizer-train-mix
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
| LoRA (r=16) | ~95% | +200% | SFT stage |
| FP16 Mixed Precision | ~50% | +20% | Always (default) |
| Batch Size Tuning | Variable | Neutral | Match VRAM capacity |

## Common Pitfalls

1. **Forgetting to train tokenizer first**: Always run `make tokenizer-train-mix` before preparing datasets
2. **Inconsistent tokenizers**: Never train multiple tokenizers. Use frozen tokenizer for all datasets.
3. **Missing `--tokenizer_path`**: Evaluation and serving require explicit tokenizer path for clean text output
4. **Ignoring tokenizer mismatch errors**: These are BLOCKING for good reason - fix immediately
5. **Not using multi-dataset weighted sampling**: Curriculum learning significantly improves model quality
6. **Insufficient warmup steps**: Use 10% of max_steps as warmup (critical for stability)

## File Structure Reference

```
Lumi-Lab/
├── scripts/
│   ├── 01_prepare_corpus.py       # Stage 1: Clean raw data into corpus
│   ├── 02_train_tokenizer.py      # Stage 2: Train global tokenizer
│   ├── 03_pack_dataset.py         # Stage 3: Tokenize and pack sequences
│   ├── 02_prepare_sft_corpus.py   # SFT Stage 1: Prepare conversations
│   ├── 03_sft_industrial.py       # SFT Stage 2: Train with LoRA
│   ├── 04_pretrain.py             # Pre-train from scratch
│   ├── 05_evaluate.py             # Evaluation with metrics
│   └── 06_serve.py                # Interactive/API inference
├── utils/
│   ├── dataset_utils.py           # Dataset loaders and samplers
│   ├── model_utils.py             # Model creation and checkpointing
│   ├── sft_templates.py           # Conversation templates
│   ├── sft_evaluation.py          # SFT-specific evaluation
│   ├── tokenizer_metrics.py       # Tokenizer quality analysis
│   └── precise_token_counter.py   # Token budget calculation
├── config/
│   ├── tiny.json, small.json, base.json  # Model architectures
│   ├── sft_datasets/              # SFT dataset configurations
│   ├── sft_training/              # SFT training configurations
│   └── datasets/
│       ├── sources/               # Raw corpus sources
│       └── training_configs/      # Pre-training dataset configs
├── data/
│   ├── tokenizer/                 # Global frozen tokenizer (spm32k)
│   ├── corpus/                    # Stage 1 output (cleaned text)
│   ├── processed/                 # Stage 3 output (tokenized, packed)
│   └── sft_processed/             # SFT corpus output
├── checkpoints/
│   ├── pretrain/{size}/           # Pre-training checkpoints
│   └── sft_industrial/{size}/     # SFT checkpoints
├── evaluation_results/            # Evaluation outputs
├── Makefile                       # All workflow commands
├── README.md                      # Complete user documentation
├── CHINCHILLA_PIPELINE.md         # Chinchilla-optimal training strategy
└── SFT_INDUSTRIAL_PIPELINE.md     # Industrial SFT documentation
```

## Development Best Practices

1. **Always start with tiny model**: Validate pipeline and hyperparameters before scaling
2. **Use session commands**: `make session-quick`, `make session-prototype` for time-boxed development
3. **Monitor VRAM**: Use `make monitor` or `nvidia-smi` during training
4. **Checkpoint frequently**: Set `--save_steps` to 10% of total steps
5. **Test with synthetic data first**: `make test-pipeline` before training on real data
6. **Validate tokenizer consistency**: Check `data/tokenizer/TOKENIZER_CARD.md` for SHA256 hash
7. **Use weighted sampling**: Multi-dataset training > single dataset for quality
8. **Enable deterministic mode**: Critical for reproducibility and debugging

## Performance Targets

| Model | Perplexity (WikiText-2) | BoolQ Accuracy | Tokens/sec (RTX 4090) | VRAM (FP16) |
|-------|------------------------|----------------|----------------------|-------------|
| Tiny (6M) | 50-80 | 55-65% | ~2000 | ~2-6 GB |
| Small (42M) | 30-50 | 65-72% | ~800 | ~4-10 GB |
| Base (124M) | 20-35 | 70-78% | ~300 | ~8-14 GB |

## Additional Documentation

- [README.md](README.md): Complete usage guide and tutorials
- [CHINCHILLA_PIPELINE.md](CHINCHILLA_PIPELINE.md): Token budget optimization strategy
- [SFT_INDUSTRIAL_PIPELINE.md](SFT_INDUSTRIAL_PIPELINE.md): Detailed SFT pipeline documentation
- [Makefile](Makefile): All available commands with explanations

## Notes on TRL Integration

The SFT industrial script supports TRL's SFTTrainer with `--use_trl_trainer` flag. Key compatibility notes:
- Use `eval_strategy` (not `evaluation_strategy`)
- Use `processing_class` (not `tokenizer`) in TRL 0.23.0+
- `max_seq_length` handled via dataset preprocessing, not SFTTrainer parameter
- Reference the working `scripts/03_sft.py` for TRL integration patterns