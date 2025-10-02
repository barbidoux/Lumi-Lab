# 🎓 Industrial SFT Pipeline Documentation

## Overview

The **Industrial SFT Pipeline** is a production-grade implementation of Supervised Fine-Tuning for Lumi-Lab models. It replaces the original monolithic `03_sft.py` script with a modular, scalable, and robust two-stage pipeline that mirrors the architecture excellence of the main corpus preparation and pre-training systems.

## 🏗️ Architecture Principles

### Two-Stage Pipeline Design

**Stage 1: Data Preparation** (`02_prepare_sft_corpus.py`)
- Conversation dataset loading and validation
- Template-based formatting (ChatML, Instruct, Chat, Alpaca)
- Quality filtering and cleaning
- Tokenization with frozen tokenizer validation
- Sharded output with manifest generation

**Stage 2: Training** (`03_sft_industrial.py`)
- Pure training script with accelerate integration
- Multi-dataset weighted sampling
- Robust checkpointing with full state preservation
- Advanced evaluation with generation tests
- LoRA/PEFT optimization for memory efficiency

### Core Design Philosophy

1. **Separation of Concerns**: Data preparation and training are completely separated
2. **Tokenizer Consistency**: SHA256 validation ensures perfect tokenizer alignment
3. **Reproducibility**: Complete deterministic state management
4. **Scalability**: Sharded data loading for unlimited dataset sizes
5. **Robustness**: Production-grade error handling and validation

## 📊 Features Comparison

| Feature | Original SFT | Industrial SFT |
|---------|--------------|----------------|
| **Data Loading** | In-memory, monolithic | Sharded, memory-mapped |
| **Tokenizer Validation** | None | SHA256 verification |
| **Multi-Dataset** | Basic | Weighted sampling with validation |
| **Checkpointing** | Transformer basic | Complete state + RNG preservation |
| **Evaluation** | Loss only | Loss + generation + quality metrics |
| **Reproducibility** | Partial | Complete deterministic training |
| **Scalability** | RAM-limited | Unlimited (streaming from disk) |
| **Configuration** | Simple JSON | Structured configs with validation |

## 🚀 Quick Start

### Prerequisites

1. **Pre-trained model** available in `checkpoints/pretrain/tiny/final`
2. **Global tokenizer** trained with `make tokenizer-train-mix`
3. **Accelerate configured** for your hardware setup

### Basic Usage

```bash
# 1. Prepare SFT corpus
make prepare-sft-corpus

# 2. Train tiny model with SFT
make sft-train-tiny

# 3. Evaluate the trained model
make sft-eval
```

### Complete Pipeline

```bash
# Run the full pipeline end-to-end
make sft-pipeline-complete
```

## 📋 Configuration System

### Dataset Configuration (`config/sft_datasets/`)

Example: `alpaca_chatml.json`
```json
{
  "name": "alpaca_chatml",
  "description": "Stanford Alpaca dataset formatted with ChatML template",
  "template": "chatml",

  "output_params": {
    "sequence_length": 1024,
    "shard_size": 1000,
    "train_ratio": 0.95
  },

  "quality_filters": {
    "min_prompt_length": 10,
    "max_prompt_length": 2000,
    "min_response_length": 10,
    "max_response_length": 2000,
    "filter_urls": true,
    "filter_code_blocks": false
  },

  "datasets": [
    {
      "name": "alpaca_gpt4",
      "type": "huggingface",
      "dataset_name": "tatsu-lab/alpaca",
      "text_fields": {
        "prompt": "instruction",
        "response": "output"
      },
      "weight": 1.0
    }
  ]
}
```

### Training Configuration (`config/sft_training/`)

Example: `lora_tiny.json`
```json
{
  "name": "lora_tiny",
  "description": "LoRA configuration for tiny model fine-tuning",

  "training_params": {
    "learning_rate": 1e-3,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 100,
    "save_steps": 200,
    "eval_steps": 100
  },

  "lora_config": {
    "r": 16,
    "lora_alpha": 16,
    "target_modules": [
      "q_proj", "k_proj", "v_proj", "o_proj",
      "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout": 0.05
  },

  "evaluation_config": {
    "eval_prompts": [
      "Hello, how are you?",
      "What is 2+2?",
      "Tell me a joke."
    ],
    "generation_config": {
      "max_new_tokens": 64,
      "temperature": 0.7,
      "top_p": 0.9
    }
  }
}
```

## 🔧 Detailed Usage

### Data Preparation

#### Basic Corpus Preparation
```bash
python scripts/02_prepare_sft_corpus.py \
    --config config/sft_datasets/alpaca_chatml.json \
    --output_dir data/sft_processed/alpaca_chatml_32k_1024 \
    --tokenizer_path data/tokenizer/spm32k.model
```

#### Multiple Dataset Preparation
```bash
# Prepare multiple datasets with different templates
python scripts/02_prepare_sft_corpus.py \
    --config config/sft_datasets/alpaca_chatml.json \
    --output_dir data/sft_processed/alpaca_chatml_32k_1024 \
    --tokenizer_path data/tokenizer/spm32k.model

python scripts/02_prepare_sft_corpus.py \
    --config config/sft_datasets/openassistant_instruct.json \
    --output_dir data/sft_processed/openassistant_instruct_32k_1024 \
    --tokenizer_path data/tokenizer/spm32k.model
```

### Training

#### Single Dataset Training
```bash
accelerate launch scripts/03_sft_industrial.py \
    --config config/sft_training/lora_tiny.json \
    --model_path checkpoints/pretrain/tiny/final \
    --data_dirs data/sft_processed/alpaca_chatml_32k_1024 \
    --tokenizer_path data/tokenizer/spm32k.model \
    --output_dir checkpoints/sft_industrial/tiny
```

#### Multi-Dataset Weighted Training
```bash
accelerate launch scripts/03_sft_industrial.py \
    --config config/sft_training/lora_small.json \
    --model_path checkpoints/pretrain/small/final \
    --data_dirs \
        data/sft_processed/alpaca_chatml_32k_1024 \
        data/sft_processed/openassistant_instruct_32k_1024 \
    --data_weights 0.7 0.3 \
    --tokenizer_path data/tokenizer/smp32k.model \
    --output_dir checkpoints/sft_industrial/small_multi
```

#### Resume Training from Checkpoint
```bash
accelerate launch scripts/03_sft_industrial.py \
    --config config/sft_training/lora_small.json \
    --model_path checkpoints/pretrain/small/final \
    --data_dirs data/sft_processed/alpaca_chatml_32k_1024 \
    --tokenizer_path data/tokenizer/spm32k.model \
    --output_dir checkpoints/sft_industrial/small \
    --resume_from_checkpoint checkpoints/sft_industrial/small/checkpoint-1000
```

## 🎨 Conversation Templates

The industrial SFT pipeline supports multiple conversation templates:

### ChatML (Recommended)
```
<|im_start|>user
What is artificial intelligence?
<|im_end|>
<|im_start|>assistant
Artificial intelligence (AI) is a technology that enables machines to mimic human intelligence...
<|im_end|>
```

### Instruct
```
### Instruction:
What is artificial intelligence?

### Response:
Artificial intelligence (AI) is a technology that enables machines to mimic human intelligence...
```

### Chat
```
Human: What is artificial intelligence?Assistant: Artificial intelligence (AI) is a technology that enables machines to mimic human intelligence...
```

### Alpaca
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is artificial intelligence?

### Response:
Artificial intelligence (AI) is a technology that enables machines to mimic human intelligence...
```

### Template Selection Guidelines

- **ChatML**: Recommended for modern LLMs, explicit role markers, good for multi-turn conversations
- **Instruct**: Classic format, good for instruction-following tasks
- **Chat**: Simple human-assistant format, easy to understand
- **Alpaca**: Formal instruction format with task description

## 📊 Data Output Structure

After preparation, SFT data is organized as follows:

```
data/sft_processed/alpaca_chatml_32k_1024/
├── train_00000.jsonl          # Training shards
├── train_00001.jsonl
├── val_00000.jsonl            # Validation shard
├── manifest.json              # Metadata and shard registry
├── DATA_CARD.md              # Dataset documentation
└── training_state.json       # Processing statistics
```

### Manifest Structure
```json
{
  "created_at": "2024-09-29T10:30:00",
  "tokenizer_metadata": {
    "tokenizer_sha256": "a1b2c3d4...",
    "vocab_size": 32768,
    "tokenizer_path": "data/tokenizer/spm32k.model"
  },
  "splits": {
    "train": {
      "num_shards": 25,
      "conversations": 48576,
      "total_tokens": 24288000
    },
    "val": {
      "num_shards": 2,
      "conversations": 2424,
      "total_tokens": 1212000
    }
  },
  "config": { "...": "original_config" }
}
```

## 🔄 Training Process

### Training Flow

1. **Initialization**
   - Load pre-trained model from checkpoint
   - Apply LoRA configuration to target modules
   - Setup optimizer and learning rate scheduler
   - Initialize accelerate for distributed training

2. **Data Loading**
   - Load SFT datasets from sharded files
   - Validate tokenizer consistency across datasets
   - Setup weighted sampling if multiple datasets
   - Create data loaders with proper collation

3. **Training Loop**
   - Forward pass through model
   - LoRA adapter updates (not base model)
   - Gradient accumulation and clipping
   - Learning rate scheduling
   - Periodic evaluation and generation tests

4. **Checkpointing**
   - Save LoRA adapters
   - Preserve optimizer and scheduler states
   - Save random number generator states
   - Store training metadata and loss history

### Memory Optimization

| Technique | Memory Reduction | Speed Impact |
|-----------|------------------|--------------|
| **LoRA r=16** | ~95% | +200% |
| **Gradient Checkpointing** | ~40% | -10% |
| **FP16 Mixed Precision** | ~50% | +20% |
| **Batch Size Optimization** | Variable | Neutral |

### Expected VRAM Usage

| Model Size | Base Training | LoRA Training | Max Seq Length |
|------------|---------------|---------------|----------------|
| **Tiny (6M)** | ~6 GB | ~2 GB | 1024 |
| **Small (42M)** | ~14 GB | ~4 GB | 1024 |
| **Base (124M)** | >16 GB | ~8 GB | 2048 |

## 📈 Evaluation System

### Automatic Evaluation

The industrial SFT pipeline includes comprehensive evaluation:

1. **Quantitative Metrics**
   - Training loss progression
   - Validation loss
   - Perplexity calculation
   - Generation success rate

2. **Qualitative Assessment**
   - Response diversity score
   - Coherence measurement
   - Repetition detection
   - Length distribution analysis

3. **Generation Tests**
   - Configurable evaluation prompts
   - Template-specific response extraction
   - Quality scoring with automated metrics

### Custom Evaluation Prompts

Create custom evaluation sets:

```json
{
  "eval_prompts": [
    "What is machine learning?",
    "Explain quantum computing",
    "Write a short poem about cats",
    "How do you make a sandwich?",
    "What are the benefits of exercise?"
  ]
}
```

## 🛠️ Advanced Usage

### Custom Dataset Integration

To add a new dataset source:

1. **Create Dataset Configuration**
```json
{
  "name": "my_custom_dataset",
  "template": "chatml",
  "datasets": [
    {
      "name": "custom_source",
      "type": "json",
      "dataset_name": "/path/to/my_data.json",
      "text_fields": {
        "prompt": "question",
        "response": "answer"
      }
    }
  ]
}
```

2. **Prepare the Data**
```bash
python scripts/02_prepare_sft_corpus.py \
    --config config/sft_datasets/my_custom_dataset.json \
    --output_dir data/sft_processed/my_custom_dataset_32k_1024 \
    --tokenizer_path data/tokenizer/spm32k.model
```

### Multi-GPU Training

The pipeline supports distributed training via accelerate:

```bash
# Configure accelerate for multi-GPU
accelerate config

# Launch distributed training
accelerate launch scripts/03_sft_industrial.py \
    --config config/sft_training/lora_base.json \
    --model_path checkpoints/pretrain/base/final \
    --data_dirs data/sft_processed/large_dataset_32k_1024 \
    --tokenizer_path data/tokenizer/spm32k.model \
    --output_dir checkpoints/sft_industrial/base_distributed
```

### Custom LoRA Configuration

Fine-tune LoRA parameters for your use case:

```json
{
  "lora_config": {
    "r": 64,                    // Higher rank = more parameters, better quality
    "lora_alpha": 16,           // Scaling factor, typically r or r/2
    "target_modules": [         // Which layers to adapt
      "q_proj", "k_proj", "v_proj", "o_proj",     // Attention layers
      "gate_proj", "up_proj", "down_proj"         // MLP layers
    ],
    "lora_dropout": 0.1,        // Regularization
    "bias": "none"              // Usually none for LLaMA models
  }
}
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Tokenizer Mismatch Error
```
ValueError: Tokenizer mismatch between datasets!
```

**Solution:**
```bash
# Re-encode problematic dataset
python scripts/02_prepare_sft_corpus.py \
    --config path/to/config.json \
    --output_dir data/sft_processed/fixed_dataset \
    --tokenizer_path data/tokenizer/spm32k.model \
    --force
```

#### 2. Out of Memory During Training
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size: `per_device_train_batch_size: 4 → 2`
- Increase gradient accumulation: `gradient_accumulation_steps: 4 → 8`
- Enable gradient checkpointing: `gradient_checkpointing: true`
- Reduce LoRA rank: `r: 64 → 32`

#### 3. Training Not Resuming Properly
```
FileNotFoundError: training_state.json not found
```

**Solution:**
```bash
# Check checkpoint directory structure
ls checkpoints/sft_industrial/tiny/checkpoint-1000/

# Should contain:
# - adapter_config.json
# - adapter_model.safetensors
# - training_state.json
```

#### 4. Poor Generation Quality

**Diagnostics:**
- Check training loss convergence
- Verify template formatting in data
- Review evaluation metrics
- Test with different generation parameters

**Solutions:**
- Increase training steps/epochs
- Adjust learning rate
- Improve data quality filtering
- Fine-tune generation parameters

### Performance Optimization

#### Hardware-Specific Settings

**RTX 4090 (16GB VRAM):**
```json
{
  "per_device_train_batch_size": 8,
  "gradient_accumulation_steps": 4,
  "gradient_checkpointing": true,
  "fp16": true,
  "dataloader_pin_memory": true
}
```

**RTX 3090 (24GB VRAM):**
```json
{
  "per_device_train_batch_size": 12,
  "gradient_accumulation_steps": 2,
  "gradient_checkpointing": false,
  "fp16": true
}
```

## 📚 Examples and Recipes

### Recipe 1: Quick Prototyping (30 minutes)
```bash
# Use tiny dataset and model for rapid iteration
SFT_DATASET_CONFIG=config/sft_datasets/alpaca_chatml.json \
SFT_TRAINING_CONFIG=config/sft_training/lora_tiny.json \
SFT_MODEL_PATH=checkpoints/pretrain/tiny/final \
make sft-custom
```

### Recipe 2: High-Quality Conversational Model
```bash
# Multi-dataset training with quality focus
make prepare-sft-corpus SFT_DATASET_CONFIG=config/sft_datasets/multi_dataset_chatml.json
make sft-train-small SFT_TRAINING_CONFIG=config/sft_training/lora_small.json
```

### Recipe 3: Production Deployment
```bash
# Full pipeline with base model
make prepare-sft-corpus SFT_DATASET_CONFIG=config/sft_datasets/production_mix.json
make sft-train-base
make sft-eval
```

## 🔄 Migration from Legacy SFT

### Key Differences

| Aspect | Legacy SFT | Industrial SFT |
|--------|------------|----------------|
| **Usage** | `python scripts/03_sft.py` | `make sft-train-tiny` |
| **Data Prep** | Inline during training | Separate preparation stage |
| **Configs** | Single file | Separate dataset/training configs |
| **Checkpoints** | Basic transformer | Complete state preservation |
| **Multi-Dataset** | Manual combination | Native weighted sampling |

### Migration Steps

1. **Extract your dataset configuration:**
```bash
# Old way
python scripts/03_sft.py --dataset_path ./data/my_conversations.json

# New way: Create config file
cat > config/sft_datasets/my_conversations.json << EOF
{
  "name": "my_conversations",
  "template": "chatml",
  "datasets": [
    {
      "name": "my_data",
      "type": "json",
      "dataset_name": "./data/my_conversations.json",
      "text_fields": {
        "prompt": "prompt",
        "response": "response"
      }
    }
  ]
}
EOF
```

2. **Prepare data with new pipeline:**
```bash
make prepare-sft-corpus SFT_DATASET_CONFIG=config/sft_datasets/my_conversations.json
```

3. **Train with industrial pipeline:**
```bash
make sft-train-tiny  # or sft-train-small, sft-train-base
```

## 🎯 Best Practices

### Data Quality
1. **Curate high-quality conversations** - Quality over quantity
2. **Balance prompt/response lengths** - Avoid extremely short or long examples
3. **Consistent formatting** - Use the same template throughout
4. **Filter inappropriate content** - Remove harmful or biased examples

### Training Strategy
1. **Start with tiny model** - Validate pipeline and hyperparameters
2. **Use appropriate learning rates** - Higher for smaller models (1e-3 → 3e-4 → 1e-4)
3. **Monitor generation quality** - Regular evaluation prevents overfitting
4. **Save multiple checkpoints** - Allow rollback to better states

### Resource Management
1. **Monitor VRAM usage** - Use `nvidia-smi` during training
2. **Optimize batch sizes** - Find the sweet spot for your hardware
3. **Use gradient checkpointing** - Trade compute for memory
4. **Regular cleanup** - Remove old checkpoints to save disk space

### Evaluation and Deployment
1. **Test on diverse prompts** - Ensure broad capability
2. **Compare with base model** - Measure improvement from SFT
3. **Validate template consistency** - Ensure proper formatting in generation
4. **Document model cards** - Track training details and performance

## 📖 References

- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **SFT Best Practices**: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- **Accelerate Documentation**: [HuggingFace Accelerate](https://huggingface.co/docs/accelerate)
- **PEFT Library**: [Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)

---

## 🏁 Conclusion

The Industrial SFT Pipeline represents a significant advancement in the robustness, scalability, and usability of supervised fine-tuning in Lumi-Lab. By adopting the same architectural principles as the corpus preparation and pre-training pipelines, it ensures consistency and reliability across the entire model development lifecycle.

**Key Benefits:**
- ✅ **Production-Ready**: Robust error handling and validation
- ✅ **Scalable**: Handles datasets of any size
- ✅ **Reproducible**: Complete deterministic training
- ✅ **Modular**: Clean separation of concerns
- ✅ **Extensible**: Easy to add new templates and datasets

The pipeline is designed to grow with your needs, from rapid prototyping with tiny models to production deployment with base models, all while maintaining the highest standards of code quality and engineering excellence.