# 🤖 Lumi - Mini-LLM Training Pipeline

A complete codebase for training mini-LLMs (decoder-only) optimized for personal machines with RTX 4090 GPU. Implements a complete modular pipeline: pre-training, SFT (Supervised Fine-Tuning), and DPO (Direct Preference Optimization).

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [⚙️ Installation](#️-installation)
- [🏗️ Architecture](#️-architecture)
- [📊 Model Configurations](#-model-configurations)
- [🔄 Training Pipeline](#-training-pipeline)
- [📖 Detailed Usage Guide](#-detailed-usage-guide)
- [🎯 RTX 4090 Optimizations](#-rtx-4090-optimizations)
- [🔧 Troubleshooting](#-troubleshooting)
- [📈 Monitoring and Evaluation](#-monitoring-and-evaluation)
- [🚀 Makefile Commands](#-makefile-commands)
- [📝 Complete Examples](#-complete-examples)

## 🚀 Quick Start

### Session-Based Learning (Recommended)

```bash
# 1. Installation
git clone https://github.com/barbidoux/Lumi.git
cd Lumi
make install

# 2. First session: 30min validation (perfect for first-time users)
make session-quick

# 3. Learning session: 2h complete tiny model (your first real model)
make session-prototype

# 4. Check what you've built
make session-status
```

### Traditional Pipeline (Alternative)

```bash
# If you prefer step-by-step control
make prepare-demo          # Fast dataset for testing
make pretrain-tiny
make evaluate

# Or for production
make prepare-owt           # OpenWebText dataset
make pretrain-tiny
make evaluate
```

## ⚙️ Installation

### System Requirements

- **GPU**: RTX 4090 (16 GB VRAM) or CUDA equivalent
- **RAM**: 32 GB recommended (minimum 16 GB)
- **Python**: 3.11+
- **CUDA**: 11.8+ or 12.1+
- **Disk Space**: 50 GB free (data + checkpoints)

### Automatic Installation

```bash
make install
```

### Manual Installation

```bash
pip install -r requirements.txt
```

### FlashAttention-2 Installation (Optional but Recommended)

FlashAttention-2 provides significant memory savings (~50%) and speed improvements for training:

```bash
# Install FlashAttention-2 for CUDA 12.1
pip install flash-attn --no-build-isolation

# Or for CUDA 11.8
pip install flash-attn==2.5.8 --no-build-isolation
```

**Requirements:**
- CUDA 11.6+ or CUDA 12.x
- PyTorch 2.0+ compiled with CUDA
- Compatible GPU architecture (SM 8.0+, e.g., RTX 30/40 series)

**Fallback behavior:**
- If FlashAttention-2 fails to install/load, Lumi automatically falls back to PyTorch SDPA
- Use `--no_flash_attn` flag to explicitly disable FlashAttention-2
- No code changes needed - fallback is automatic and transparent

**Installation help:**
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
- [Pre-compiled wheels](https://github.com/Dao-AILab/flash-attention/releases)

### Environment Verification

```bash
make check-env
```

**Expected output:**
```
PyTorch: 2.3.0+cu121
CUDA available: True
Transformers: 4.40.0
Accelerate: 0.30.1
GPU: NVIDIA GeForce RTX 4090
```

## 🏗️ Project Structure

```
Lumi/
├── config/                 # Model configurations
│   ├── tiny.json          # 6M parameters - ideal for testing
│   ├── small.json         # 42M parameters - balanced  
│   ├── base.json          # 124M parameters - performant
│   ├── sft.json           # LoRA/SFT configuration
│   └── advanced_example.json # GQA and advanced features
├── evaluation/             # Evaluation system
│   ├── smoke_prompts.json  # Categorized test prompts
│   ├── evaluation_config.json # Performance benchmarks
│   ├── assess_performance.py  # Automated assessment
│   └── quick_prompts.json  # Fast evaluation prompts
├── sessions/               # Session logs and management
│   └── (session logs created automatically)
├── data/                   # Training data
│   └── (created automatically)
├── scripts/                # Training scripts
│   ├── 01_prepare_data.py  # Data preparation and cleaning
│   ├── 02_pretrain.py      # Pre-training from scratch
│   ├── 03_sft.py          # Supervised fine-tuning with LoRA
│   ├── 04_dpo.py          # DPO alignment
│   ├── 05_evaluate.py     # Evaluation and benchmarks
│   └── 06_serve.py        # Model inference server (CLI + API)
├── utils/                  # Shared utilities
│   ├── dataset_utils.py   # Dataset management
│   └── model_utils.py     # Model creation/loading
│   └── validate_architecture.py # Architecture validation
├── ARCHITECTURE.md         # Detailed technical documentation
├── SESSIONS.md             # Session management guide
├── Makefile               # Automated commands with sessions
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```

## 🧠 LLaMA-like Architecture

**Lumi implements a modern decoder-only transformer following LLaMA design principles:**

### Core Architecture Components
- **🔧 RMSNorm**: Root Mean Square normalization (faster than LayerNorm)
- **⚡ SwiGLU**: Swish-Gated Linear Units activation (better than ReLU/GELU)  
- **📐 RoPE**: Rotary Position Embeddings (superior length extrapolation)
- **🚫 No Bias**: Cleaner scaling, fewer parameters
- **🎯 Causal Attention**: Autoregressive text generation
- **💾 GQA Ready**: Grouped Query Attention for memory efficiency

### Why LLaMA Architecture?
- ✅ **Proven**: State-of-the-art results across scales
- ✅ **Efficient**: Optimized for modern GPU training
- ✅ **Scalable**: Consistent performance from 6M to 124M+ parameters
- ✅ **Compatible**: Full HuggingFace Transformers support
- ✅ **Memory-Optimal**: RMSNorm + no bias + FlashAttention-2

**📖 For complete architectural details, see [ARCHITECTURE.md](ARCHITECTURE.md)**

## 📊 Model Configurations

| Model | Layers | Dimension | Heads | Parameters | VRAM (FP16) | Time (1 epoch) | Use Case |
|-------|--------|-----------|-------|------------|-------------|----------------|----------|
| **tiny** | 6 | 256 | 4 | ~6M | ~2 GB | 2-4h | Quick tests, prototyping |
| **small** | 12 | 512 | 8 | ~42M | ~4 GB | 8-12h | Development, validation |
| **base** | 24 | 768 | 12 | ~124M | ~8 GB | 24-48h | Production, performance |

### Architecture Scaling Strategy

**Consistent design principles across all sizes:**
- `head_dim = 64` (d_model / n_head)
- `ffn_ratio = 4.0` (d_ff / d_model)
- `vocab_size = 32768` (SentencePiece optimized)

### Configuration Examples

#### Standard Configuration
```json
{
  "model_name": "custom",
  "n_layer": 12,           // Transformer layers
  "d_model": 512,          // Hidden dimension  
  "n_head": 8,             // Attention heads
  "d_ff": 2048,            // FFN dimension (4x rule)
  "vocab_size": 32768,     // Fixed vocabulary
  "sequence_length": 1024, // Context length
  "dropout": 0.1           // Regularization
}
```

#### Advanced Configuration (GQA)
```json
{
  "model_name": "gqa_example",
  "n_layer": 16,
  "d_model": 640,
  "n_head": 10,
  "num_key_value_heads": 5,  // GQA: 2:1 ratio
  "d_ff": 2560,
  "sequence_length": 1536,
  "rope_theta": 10000.0      // RoPE base frequency
}
```

**🔧 See `config/advanced_example.json` for all available options**

## 🔄 Training Pipeline

### Process Overview

1. **Data Preparation** 📊
   - Raw dataset loading
   - Intelligent cleaning and filtering
   - MinHash deduplication (fuzzy duplicates)
   - SentencePiece tokenization

2. **Pre-training** 🧠
   - Language learning from scratch
   - Optimized Llama-like architecture
   - Automatic checkpointing and resumption

3. **Supervised Fine-tuning (SFT)** 🎯
   - Adaptation with conversational data
   - LoRA for memory efficiency
   - Multiple templates (ChatML, Instruct)

4. **DPO Alignment** ⚖️
   - Human preference optimization
   - Native PEFT adapter management
   - Advanced quality metrics

5. **Evaluation** 📈
   - WikiText-2 perplexity
   - Zero-shot benchmarks (BoolQ)
   - Custom generation tests

6. **Inference & Serving** 🚀
   - Interactive CLI chat interface
   - REST API server with FastAPI
   - Multiple prompt templates support

## 📊 Supported Datasets

### Pre-training Datasets

| Dataset | HuggingFace ID | License | Size | Language | Filters Applied | Notes |
|---------|----------------|---------|------|----------|-----------------|--------|
| **OpenWebText** | `openwebtext` | Public Domain | ~40GB, 8M docs | EN | Min 50 chars, max 10K chars, English detection, MinHash dedup | Primary recommendation |
| **WikiText-103** | `wikitext-103-raw-v1` | CC BY-SA 3.0 | ~500MB, 103M tokens | EN | Raw version, minimal processing | Good for smaller experiments |
| **WikiText-2** | `wikitext-2-raw-v1` | CC BY-SA 3.0 | ~12MB, 2M tokens | EN | Raw version, minimal processing | Quick testing only |

### Evaluation Datasets

| Dataset | HuggingFace ID | License | Size | Purpose | Usage in Lumi |
|---------|----------------|---------|------|---------|---------------|
| **WikiText-2** | `wikitext-2-raw-v1` | CC BY-SA 3.0 | 2M tokens | Perplexity | Primary evaluation metric |
| **BoolQ** | `boolq` | CC BY 4.0 | 15.9K questions | Yes/No reasoning | Zero-shot capability indicator |

### Fine-tuning Datasets (Optional)

| Dataset | HuggingFace ID | License | Notes |
|---------|----------------|---------|--------|
| **Alpaca** | `tatsu-lab/alpaca` | CC BY-NC 4.0 | Instruction following |
| **OpenAssistant** | `OpenAssistant/oasst1` | Apache 2.0 | Conversational data |
| **ShareGPT** | Various | Varies | Multiple derivatives available |

⚠️ **BookCorpus Note**: Avoid due to unclear licensing status. Use OpenWebText instead.

### Dataset Preprocessing Pipeline

**Applied filters for all pre-training data:**
- **Language**: English detection with 70% threshold
- **Length**: 50-10,000 characters (configurable)
- **Deduplication**: SHA256 exact + MinHash fuzzy (80% similarity)
- **Cleaning**: URLs, code blocks, control characters removed
- **Quality**: Basic heuristics for text quality

## 📖 Detailed Usage Guide

### 1. Data Preparation

Our robust, config-based data preparation pipeline supports multiple datasets with reproducible configurations.

#### Quick Start (Recommended)

```bash
# OpenWebText (production-ready, large dataset)
make prepare-owt

# Wikipedia EN (medium size, high quality)
make prepare-wiki

# WikiText-103 (small, fast for testing)
make prepare-wt103

# Demo tiny (very fast for development)
make prepare-demo
```

#### Available Dataset Configurations

| Config File | Dataset | Size | Use Case |
|-------------|---------|------|----------|
| `config/datasets/openwebtext.json` | OpenWebText | ~40GB | Production training |
| `config/datasets/wikipedia_en.json` | Wikipedia EN | ~20GB | High-quality corpus |
| `config/datasets/wikitext103.json` | WikiText-103 | ~500MB | Quick testing |
| `config/datasets/demo_tiny.json` | WikiText-2 | ~12MB | Development |

#### Custom Configuration

Create your own dataset config (JSON/YAML):

```json
{
  "input_path": "your-dataset-name",
  "output_dir": "data/processed/custom_32k_1024",
  "tokenizer_path": "data/tokenizer/spm32k.model",
  "vocab_size": 32768,
  "sequence_length": 1024,
  "min_length": 50,
  "max_length": 10000,
  "use_minhash": true,
  "minhash_threshold": 0.8,
  "train_ratio": 0.98,
  "shard_tokens": 5000000
}
```

Then use it:
```bash
make prepare CONFIG=config/datasets/your_config.json
```

#### Advanced Usage

```bash
# Override config values via CLI
python scripts/01_prepare_data.py \
    --config_path config/datasets/openwebtext.json \
    --vocab_size 16384 \
    --max_length 2048

# Use local files
python scripts/01_prepare_data.py \
    --config_path config/datasets/custom.json \
    --input_path ./data/raw/my_corpus.txt
```

**Pipeline Features:**
- ✅ **Config-Based**: Reproducible, versioned configurations
- ✅ **Multi-Source**: HuggingFace, local files, custom datasets
- ✅ **Advanced Deduplication**: SHA256 + MinHashLSH for fuzzy duplicates  
- ✅ **Smart Filtering**: Language detection, length filtering, text cleaning
- ✅ **Robust Processing**: ftfy encoding fix, URL/code removal
- ✅ **Sharded Output**: JSONL format with manifest and data cards
- ✅ **Quality Assurance**: Built-in validation and statistics

**Output Structure:**
```
data/processed/dataset_name/
├── train_00000.jsonl        # Training shards
├── train_00001.jsonl
├── val_00000.jsonl          # Validation shards  
├── manifest.json            # Shard registry
├── DATA_CARD.md            # Dataset documentation
└── stats.json              # Processing statistics
```

### 2. Pre-training from Scratch

#### Tiny Model (recommended for beginners)

```bash
# Via Makefile (recommended)
make pretrain-tiny

# Or manually with more control
accelerate launch scripts/02_pretrain.py \
    --config config/tiny.json \
    --data_path ./data/processed/tokenized_data.json \
    --learning_rate 3e-4 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 1000 \
    --save_steps 2000 \
    --logging_steps 50
```

#### Resume from Checkpoint

```bash
accelerate launch scripts/02_pretrain.py \
    --config config/tiny.json \
    --data_path ./data/processed/tokenized_data.json \
    --resume_from_checkpoint ./checkpoints/pretrain/tiny/step_10000
```

#### Deterministic Training for Reproducibility

```bash
# Enable deterministic training (default)
accelerate launch scripts/02_pretrain.py \
    --config config/tiny.json \
    --data_path ./data/processed/tokenized_data.json \
    --seed 42

# Disable deterministic training (faster but non-reproducible)
accelerate launch scripts/02_pretrain.py \
    --config config/tiny.json \
    --data_path ./data/processed/tokenized_data.json \
    --no_deterministic
```

**Deterministic training features:**
- 🎯 **Complete Seed Management**: Python, NumPy, PyTorch, CUDA
- 🔒 **CUDNN Deterministic**: Ensures exact reproducibility
- 💾 **RNG State Checkpoints**: Save/restore all random number generators
- ⚡ **Resumable**: Continue training with exact same randomness
- 🧪 **Validation**: Restart from checkpoint shows identical loss/LR (±1e-6)

**Training features:**
- ✅ **Accelerate**: Native multi-GPU distribution
- ✅ **FlashAttention-2**: 50% memory reduction
- ✅ **Gradient Checkpointing**: Memory/time optimization
- ✅ **Deterministic Training**: Complete reproducibility with seeds
- ✅ **Robust Checkpoints**: Complete state saving with model, optimizer, scheduler, scaler, global_step, and RNG states
- ✅ **Cosine Scheduler**: Warmup + optimal decay

### 3. Supervised Fine-tuning (SFT)

#### SFT Dataset Preparation

**Expected JSON format:**
```json
[
  {
    "prompt": "What is artificial intelligence?",
    "response": "Artificial intelligence (AI) is a technology that enables machines to mimic human intelligence to solve complex problems and make decisions."
  },
  {
    "prompt": "How does machine learning work?", 
    "response": "Machine learning uses algorithms to analyze data, identify patterns, and make predictions without explicit programming."
  }
]
```

#### Launch SFT with Templates

```bash
# ChatML template (recommended)
python scripts/03_sft.py \
    --model_path ./checkpoints/pretrain/tiny/final \
    --dataset_path ./data/sft_dataset.json \
    --prompt_template chatml \
    --output_dir ./checkpoints/sft

# Classic instruction template
python scripts/03_sft.py \
    --model_path ./checkpoints/pretrain/tiny/final \
    --dataset_path ./data/sft_dataset.json \
    --prompt_template instruct
```

**Available templates:**
- `chatml`: `<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n...\n<|im_end|>`
- `instruct`: `### Instruction:\n...\n\n### Response:\n...`
- `chat`: `Human: ...\n\nAssistant: ...`

### 4. DPO Alignment

#### DPO Dataset with Preferences

**Required format:**
```json
[
  {
    "prompt": "How to learn effectively?",
    "chosen": "To learn effectively, you need to be consistent, practice actively, use memorization techniques, and review regularly with spaced repetition.",
    "rejected": "Just read once and you're good."
  }
]
```

#### DPO Execution

```bash
python scripts/04_dpo.py \
    --model_path ./checkpoints/sft \
    --dataset_path ./data/dpo_dataset.json \
    --beta 0.1 \
    --learning_rate 5e-7 \
    --batch_size 2 \
    --gradient_accumulation_steps 8
```

**DPO optimizations:**
- ✅ **Native PEFT Management**: No costly adapter fusion
- ✅ **Automatic Reference**: DPOTrainer creates reference model
- ✅ **Early Stopping**: Intelligent metric-based stopping
- ✅ **VRAM Optimized**: Memory management for RTX 4090

### 5. Realistic Evaluation System

#### Standard Evaluation

```bash
# Complete evaluation with quality analysis
python scripts/05_evaluate.py \
    --model_path ./checkpoints/dpo \
    --output_dir ./evaluation_results \
    --max_boolq_samples 100 \
    --detailed_output
```

#### Fast Development Mode

```bash
# Quick evaluation for development iterations
python scripts/05_evaluate.py \
    --model_path ./checkpoints/pretrain/tiny/final \
    --fast_mode \
    --output_dir ./quick_eval
```

#### Performance Assessment

```bash
# Automatic performance assessment with recommendations
python evaluation/assess_performance.py ./evaluation_results/evaluation_results.json
```

**Enhanced evaluation system:**
- 🎯 **WikiText-2 Focus**: Primary benchmark on clean Wikipedia text
- 🧪 **Categorized Smoke-Tests**: 8 categories (knowledge, reasoning, creativity, etc.)
- 📈 **Quality Analysis**: Automatic response quality scoring (0-1.0)
- 📉 **Performance Assessment**: Compare against expected benchmarks per model size
- ⚡ **Fast Mode**: Reduced evaluation for rapid development
- 📝 **Detailed Reports**: Markdown output with all responses and scores

**Evaluation categories:**
- **Basic Knowledge**: Factual questions, simple completions
- **Language Understanding**: Sentence completion, synonyms, definitions
- **Reasoning**: Logical questions, cause-effect, problem-solving
- **Creativity**: Story writing, poetry, imaginative scenarios
- **Instruction Following**: Specific format requests, counting, formatting
- **Wikipedia Style**: Encyclopedic completions, factual continuations
- **Conversation**: Greetings, questions, social interaction
- **Multilingual**: Basic responses in different languages

**Expected performance ranges:**

| Model | Perplexity | BoolQ Accuracy | Smoke Quality | Status |
|-------|------------|----------------|---------------|--------|
| **tiny (6M)** | 40-80 | 55-65% | 0.35-0.60 | 🟡 Experimental |
| **small (42M)** | 25-50 | 60-72% | 0.50-0.75 | 🟢 Development |
| **base (124M)** | 15-35 | 65-78% | 0.60-0.85 | 🔵 Production |

### 6. Model Inference & Serving

#### Interactive CLI Mode

```bash
# Launch interactive chat with your trained model
python scripts/06_serve.py \
    --model_path ./checkpoints/dpo \
    --mode interactive \
    --template chatml \
    --temperature 0.7 \
    --max_new_tokens 150
```

#### API Server Mode

```bash
# Start FastAPI server
python scripts/06_serve.py \
    --model_path ./checkpoints/dpo \
    --mode api \
    --host 127.0.0.1 \
    --port 8000
```

**API Usage:**
```bash
# Test the API with curl
curl -X POST "http://127.0.0.1:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "What is machine learning?",
       "max_new_tokens": 100,
       "temperature": 0.7,
       "template": "chatml"
     }'
```

**Serving features:**
- ✅ **Interactive CLI**: Real-time chat interface
- ✅ **REST API**: FastAPI-based HTTP endpoints
- ✅ **Template Support**: ChatML, Instruct, Chat formats
- ✅ **Configurable Generation**: All parameters adjustable
- ✅ **Optimized Inference**: Model optimization for speed

## 🎯 RTX 4090 Optimizations

### Automatic GPU Configuration

```bash
make config-rtx4090
```

### Integrated Optimization Techniques

| Technique | VRAM Reduction | Performance Gain | Description |
|-----------|----------------|------------------|-------------|
| **FlashAttention-2** | ~50% | +30% speed | Memory-efficient attention |
| **Gradient Checkpointing** | ~40% | -10% speed | Time/memory trade-off |
| **FP16 Mixed Precision** | ~50% | +20% speed | Automatic mixed precision |
| **LoRA Fine-tuning** | ~90% | +200% speed | Low-rank adapters |
| **Gradient Accumulation** | Variable | Effective batch | Simulates large batch |

### Optimal Parameters per Model

| Configuration | tiny | small | base |
|---------------|------|-------|------|
| **Batch Size** | 16 | 8 | 4 |
| **Grad Accumulation** | 4 | 8 | 16 |
| **Sequence Length** | 1024 | 1024 | 2048 |
| **VRAM Usage** | ~6 GB | ~10 GB | ~14 GB |
| **Time/epoch** | 2-4h | 8-12h | 24-48h |

## 🔧 Troubleshooting

### Common Errors and Solutions

#### 🚨 Out Of Memory (OOM)

```bash
# Solution 1: Reduce batch size
export BATCH_SIZE=4
make pretrain-tiny

# Solution 2: Enable gradient checkpointing
python scripts/02_pretrain.py --gradient_checkpointing

# Solution 3: Reduce sequence length
python scripts/02_pretrain.py --max_length 512
```

#### 🚨 Missing Tokenizer/Dataset

```bash
# Create sample datasets for testing
make create-sample-datasets

# Verify data structure
python -c "import json; print(json.load(open('./data/sft_dataset.json'))[:2])"
```

#### 🚨 Corrupted Checkpoint

```bash
# Clean and restart
make clean-checkpoints
make pretrain-tiny

# Or resume from previous checkpoint
ls ./checkpoints/pretrain/tiny/  # List available checkpoints
```

#### 🚨 CUDA/GPU Errors

```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU if needed (slower)
export CUDA_VISIBLE_DEVICES=""
```

### Resource Monitoring

```bash
# Real-time GPU monitoring
make monitor

# Manual monitoring
watch -n 2 'nvidia-smi; echo ""; ps aux | grep python | head -3'

# Detailed debugging logs
export TRANSFORMERS_VERBOSITY=debug
export ACCELERATE_DEBUG_MODE=1
```

## 📈 Monitoring and Evaluation

### Training Metrics

- **Loss**: Main objective - should decrease steadily
- **Perplexity**: exp(loss) - lower = better (target <50 for tiny)
- **Learning Rate**: Cosine scheduler tracking with warmup
- **Gradient Norm**: Instability detection (normal <1.0)

### Wandb Integration (optional)

```bash
# Wandb setup
pip install wandb
export WANDB_API_KEY=your_api_key_here

# Automatic logging during training
make pretrain-tiny  # Logs sent automatically
```

### Continuous Evaluation

```bash
# Periodic evaluation during training
python scripts/05_evaluate.py \
    --model_path ./checkpoints/pretrain/tiny/step_5000 \
    --skip_perplexity \
    --max_boolq_samples 50
```

### Deterministic Training Validation

Test that training is fully reproducible:

```bash
# 1. Start training with specific seed
accelerate launch scripts/02_pretrain.py \
    --config config/tiny.json \
    --data_path ./data/processed/tokenized_data.json \
    --seed 1337 \
    --max_steps 100 \
    --save_steps 50

# 2. Note the loss/LR at step 100, then restart from step 50
accelerate launch scripts/02_pretrain.py \
    --config config/tiny.json \
    --data_path ./data/processed/tokenized_data.json \
    --resume_from_checkpoint ./checkpoints/pretrain/tiny/step_50 \
    --max_steps 100

# 3. Loss/LR should be identical at step 100 (±1e-6 precision)
```

**Expected behavior:**
- ✅ Identical loss values when restarting from checkpoint
- ✅ Identical learning rate progression
- ✅ Identical model weights and optimizer states
- ✅ Same validation perplexity when evaluated

## 🚀 Session-Based Development

### Focused Development Sessions

**Lumi provides time-boxed development sessions for efficient learning and iteration:**

| Session | Duration | Purpose | Command |
|---------|----------|---------|----------|
| **Quick** | 30 min | Pipeline testing & validation | `make session-quick` |
| **Prototype** | 2 hours | Complete tiny model development | `make session-prototype` |
| **Experiment** | 4 hours | Small model + fine-tuning | `make session-experiment` |
| **Evaluation** | 1 hour | Deep analysis of trained models | `make session-evaluation` |
| **Debug** | Interactive | Problem diagnosis & fixing | `make session-debug` |
| **Architecture** | 30 min | Configuration validation | `make session-architecture` |

#### Session Examples

```bash
# First-time user: validate setup and test pipeline
make session-quick

# Learning session: train your first model
make session-prototype

# Development session: serious model training
make session-experiment

# Check current status anytime
make session-status
```

**📝 For complete session guide, see [SESSIONS.md](SESSIONS.md)**

### Main Commands

| Command | Description | Estimated Time |
|---------|-------------|----------------|
| `make help` | Show complete help | Instant |
| `make install` | Install dependencies | 5-10 min |
| `make check-env` | Environment verification | 30 sec |
| `make prepare` | Data preparation | 1-2h |
| `make pretrain-tiny` | Tiny model pre-training | 2-4h |
| `make pretrain-small` | Small model pre-training | 8-12h |
| `make pretrain-base` | Base model pre-training | 24-48h |
| `make sft` | Supervised fine-tuning | 30min-2h |
| `make dpo` | DPO alignment | 1-4h |
| `make evaluate` | Complete realistic evaluation | 30min |
| `make evaluate-quick` | Fast evaluation for development | 5min |
| `make assess-performance` | Performance assessment with recommendations | Instant |
| `make serve` | Interactive model inference | Instant |

### Session Management Commands

| Command | Description | Session Type |
|---------|-------------|---------------|
| `make session-status` | Check current development state | Status |
| `make session-cleanup` | Clean temporary session files | Maintenance |
| `make session-quick` | 30min: rapid testing & validation | Development |
| `make session-prototype` | 2h: complete tiny model training | Development |
| `make session-experiment` | 4h: small model + fine-tuning | Development |
| `make session-evaluation` | 1h: deep model analysis | Analysis |
| `make session-debug` | Interactive problem solving | Debug |
| `make session-architecture` | 30min: config validation | Validation |

### Maintenance Commands

| Command | Description |
|---------|-------------|
| `make pipeline-full` | Complete automatic pipeline |
| `make resume-pretrain-tiny` | Resume training from checkpoint |
| `make create-sample-datasets` | Generate test datasets |
| `make test-pipeline` | Quick functionality test |
| `make evaluate-detailed` | Evaluation with detailed markdown report |
| `make backup` | Backup configs + checkpoints |
| `make clean` | Clean temporary files |
| `make clean-checkpoints` | Remove all checkpoints |
| `make monitor` | Real-time resource monitoring |

### Customizable Variables

```bash
# Customize datasets
make prepare RAW_DATASET="c4"
make sft SFT_DATASET="./my_conversations.json"
make dpo DPO_DATASET="./my_preferences.json"

# Adjust parameters
make pretrain-tiny BATCH_SIZE=8 LEARNING_RATE=1e-4
```

## 📝 Complete Examples

### Example 1: Quick Test (30 minutes)

```bash
# Minimal pipeline for validation
make install
make create-sample-datasets
make test-pipeline
make evaluate
make serve  # Interactive chat with your model
```

### Example 2: Complete Tiny Training

```bash
# 1. Preparation with real data
python scripts/01_prepare_data.py \
    --input_path "wikitext-103-raw-v1" \
    --output_dir ./data/processed \
    --use_minhash \
    --vocab_size 32768

# 2. Pre-training 
make pretrain-tiny

# 3. Intermediate evaluation
python scripts/05_evaluate.py \
    --model_path ./checkpoints/pretrain/tiny/final \
    --output_dir ./results_pretrain

# 4. SFT with conversations
python scripts/03_sft.py \
    --model_path ./checkpoints/pretrain/tiny/final \
    --dataset_path ./data/conversations.json \
    --prompt_template chatml

# 5. Final evaluation  
make evaluate

# 6. Chat with your model
python scripts/06_serve.py --model_path ./checkpoints/pretrain/tiny/final --mode interactive
```

### Example 3: Production Base Model

```bash
# 1. Massive data with optimizations
python scripts/01_prepare_data.py \
    --input_path "openwebtext" \
    --output_dir ./data/processed \
    --use_minhash \
    --minhash_threshold 0.85 \
    --min_length 100 \
    --max_length 5000

# 2. Distributed pre-training
accelerate config  # Multi-GPU configuration
make pretrain-base

# 3. High-quality SFT
python scripts/03_sft.py \
    --model_path ./checkpoints/pretrain/base/final \
    --dataset_path ./data/high_quality_conversations.json \
    --prompt_template chatml \
    --config_path config/sft.json

# 4. DPO alignment
python scripts/04_dpo.py \
    --model_path ./checkpoints/sft \
    --dataset_path ./data/human_preferences.json \
    --beta 0.1 \
    --num_train_epochs 1

# 5. Complete benchmark evaluation
python scripts/05_evaluate.py \
    --model_path ./checkpoints/dpo \
    --custom_prompts ./evaluation/comprehensive_prompts.json \
    --output_dir ./final_evaluation

# 6. Deploy production API
python scripts/06_serve.py \
    --model_path ./checkpoints/dpo \
    --mode api \
    --host 0.0.0.0 \
    --port 8000
```

## 🔒 Security and Best Practices

### Data Security
- ✅ **SHA256** instead of MD5 for hashing
- ✅ **Validation** of data format before processing  
- ✅ **Cleaning** removal of sensitive content
- ✅ **Checkpoints** optional encrypted backup

### Training Best Practices
- ✅ **Continuous monitoring** of metrics and resources
- ✅ **Frequent checkpoints** to avoid data loss
- ✅ **Regular validation** on hold-out data
- ✅ **Documentation** of hyperparameters and results

### Resource Management
- ✅ **VRAM monitoring** to prevent OOM crashes
- ✅ **Log rotation** to avoid disk saturation
- ✅ **Regular cleanup** of temporary files
- ✅ **Backup** of important configurations

## 📊 Expected Results

### Model Performance (RTX 4090)

| Metric | tiny | small | base |
|--------|------|-------|------|
| **WikiText-2 Perplexity** | 50-80 | 30-50 | 20-35 |
| **BoolQ Accuracy** | 55-65% | 65-70% | 70-75% |
| **Tokens/second** | 2000 | 800 | 300 |
| **VRAM Memory** | 6 GB | 10 GB | 14 GB |

### Typical Learning Curves

- **Loss**: Exponential decay then stabilization
- **Perplexity**: Reduction 80→40 for tiny (50K steps)
- **Gradient norm**: Stable around 0.5-1.0
- **Learning rate**: Warmup then cosine decay

### Complete Training Times

| Phase | tiny | small | base |
|-------|------|-------|------|
| **Data preparation** | 1h | 1-2h | 2-3h |
| **Pre-training** | 2-4h | 8-12h | 24-48h |
| **SFT** | 30min | 1h | 2h |
| **DPO** | 1h | 2h | 4h |
| **Total** | ~5-7h | ~12-16h | ~32-56h |

## 🤝 Contribution and Support

### Technical Architecture

**🏗️ LLaMA-based Decoder-Only Transformer**
- **Core**: RMSNorm + SwiGLU + RoPE + No Bias (exact LLaMA architecture)
- **Scales**: 6M → 42M → 124M parameters with consistent head_dim=64
- **Memory**: FlashAttention-2, Gradient Checkpointing, GQA support
- **Training**: Deterministic seeding, complete checkpoint states
- **Framework**: PyTorch 2.3+, Transformers 4.40+, Accelerate, TRL
- **Hardware**: RTX 4090 16GB optimized (works on smaller GPUs)

**📈 Performance Characteristics**
- **Tiny (6M)**: ~2GB VRAM, ~2000 tok/s, perfect for prototyping
- **Small (42M)**: ~4GB VRAM, ~800 tok/s, balanced development
- **Base (124M)**: ~8GB VRAM, ~300 tok/s, production quality

**📖 Complete Documentation**: [ARCHITECTURE.md](ARCHITECTURE.md) contains detailed technical specifications, architectural choices, performance benchmarks, and comparison with original LLaMA.

### Educational Focus
This project demonstrates modern LLM training techniques with complete transparency:
- **Reproducible**: Deterministic training with exact seed management
- **Well-documented**: Every architectural choice explained and justified
- **Scalable**: Same code trains 6M to 124M+ parameter models
- **Memory-efficient**: Multiple optimization techniques for personal hardware

---

## 📄 License

Educational and research project. Free to use for learning, research, and personal development.

---

**🎯 Lumi Pipeline**: Complete LLaMA-like architecture from zero to functional LLM in hours, with modern optimizations and educational transparency.

**🏗️ Architecture**: Precise LLaMA implementation with RMSNorm, SwiGLU, RoPE, and FlashAttention-2 - see [ARCHITECTURE.md](ARCHITECTURE.md) for full technical details.
