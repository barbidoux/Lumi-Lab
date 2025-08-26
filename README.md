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

```bash
# 1. Installation
git clone https://github.com/barbidoux/Lumi.git
cd Lumi
make install

# 2. Quick test with synthetic data
make test-pipeline

# 3. Complete pipeline with real data
make create-sample-datasets
make prepare
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

## 🏗️ Architecture

```
Lumi/
├── config/                 # Model configurations
│   ├── tiny.json          # 6M parameters - ideal for testing
│   ├── small.json         # 42M parameters - balanced  
│   ├── base.json          # 124M parameters - performant
│   └── sft.json           # LoRA/SFT configuration
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
├── Makefile               # Automated commands
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```

## 📊 Model Configurations

| Model | Layers | Dimension | Heads | Parameters | VRAM (FP16) | Time (1 epoch) | Use Case |
|-------|--------|-----------|-------|------------|-------------|----------------|----------|
| **tiny** | 6 | 256 | 4 | ~6M | ~2 GB | 2-4h | Quick tests, prototyping |
| **small** | 12 | 512 | 8 | ~42M | ~4 GB | 8-12h | Development, validation |
| **base** | 24 | 768 | 12 | ~124M | ~8 GB | 24-48h | Production, performance |

### Configuration Customization

Edit JSON files in `config/`:

```json
{
  "model_name": "custom",
  "n_layer": 12,           // Number of transformer layers
  "d_model": 512,          // Embedding dimension
  "n_head": 8,             // Multi-head attention heads
  "d_ff": 2048,            // Feed-forward dimension
  "vocab_size": 32768,     // Vocabulary size (fixed)
  "sequence_length": 1024, // Maximum context length
  "dropout": 0.1           // Dropout for regularization
}
```

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

## 📖 Detailed Usage Guide

### 1. Data Preparation

#### Option A: Hugging Face Dataset

```bash
python scripts/01_prepare_data.py \
    --input_path "openwebtext" \
    --output_dir ./data/processed \
    --use_minhash \
    --vocab_size 32768 \
    --min_length 50 \
    --max_length 10000
```

#### Option B: Local JSON Data

```bash
python scripts/01_prepare_data.py \
    --input_path ./data/raw/my_dataset.json \
    --output_dir ./data/processed \
    --use_minhash \
    --minhash_threshold 0.8
```

**Preparation features:**
- ✅ **MinHash Deduplication**: Detects duplicates with 80% similarity
- ✅ **Intelligent Cleaning**: Removes URLs, code, control characters  
- ✅ **Language Filtering**: Automatic English language detection
- ✅ **SentencePiece Tokenizer**: Optimized 32K token vocabulary
- ✅ **Secure Hashing**: SHA256 for data integrity

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

**Training features:**
- ✅ **Accelerate**: Native multi-GPU distribution
- ✅ **FlashAttention-2**: 50% memory reduction
- ✅ **Gradient Checkpointing**: Memory/time optimization
- ✅ **Robust Checkpoints**: Complete state saving
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

### 5. Complete Evaluation

```bash
python scripts/05_evaluate.py \
    --model_path ./checkpoints/dpo \
    --output_dir ./evaluation_results \
    --max_boolq_samples 100
```

**Automatic metrics:**
- 📊 **WikiText-2 Perplexity**: General language quality
- 🎯 **BoolQ Accuracy**: Yes/no question comprehension  
- 🧪 **Smoke Tests**: Generation on custom prompts
- 📈 **Detailed Statistics**: Time, memory, tokens/sec

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

## 🚀 Makefile Commands

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
| `make evaluate` | Complete evaluation | 30min |
| `make serve` | Interactive model inference | Instant |

### Maintenance Commands

| Command | Description |
|---------|-------------|
| `make pipeline-full` | Complete automatic pipeline |
| `make resume-pretrain-tiny` | Resume training from checkpoint |
| `make create-sample-datasets` | Generate test datasets |
| `make test-pipeline` | Quick functionality test |
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
- **Base**: Llama-like decoder-only transformer
- **Optimizations**: FlashAttention-2, LoRA, gradient checkpointing
- **Framework**: PyTorch 2.3+, Transformers, Accelerate, TRL
- **Target**: RTX 4090 16GB with VRAM optimizations

### Development
This project implements the latest LLM training techniques with a focus on memory efficiency and reproducibility on personal hardware.

---

## 📄 License

Educational and research project. Free to use for learning, research, and personal development.

---

**🎯 Lumi Pipeline**: From zero to a functional LLM in hours on RTX 4090 with all modern optimizations integrated.
