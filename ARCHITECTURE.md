# 🏗️ Lumi Architecture Documentation

## Overview

Lumi implements a **decoder-only transformer architecture** closely following the **LLaMA (Large Language Model Meta AI)** design principles, optimized for personal GPU training. While optimized for RTX 4090, this architecture works on any NVIDIA GPU with CUDA support (even 8GB cards like RTX 3060) - you'll just need to adjust batch sizes and be patient. This document provides precise technical details of all architectural choices and their justifications.

## 🔧 Core Architecture Principles

### Decoder-Only Transformer
- **Design**: Unidirectional autoregressive language model
- **Purpose**: Next-token prediction for text generation
- **Advantage**: Simpler than encoder-decoder, better for causal language modeling
- **Implementation**: Uses HuggingFace `LlamaForCausalLM` as base architecture

### Key Design Choices (Following LLaMA)

1. **Pre-normalization with RMSNorm**
2. **SwiGLU activation function** 
3. **Rotary Position Embedding (RoPE)**
4. **No bias terms in linear layers**
5. **Grouped Query Attention (GQA) ready**
6. **FlashAttention-2 optimization**

---

## 📊 Model Configurations

### Parameter Scaling Strategy

| Model | Layers (L) | Hidden Size (d_model) | Heads (H) | FFN Size | Parameters | Memory (BF16) | Chinchilla Tokens |
|-------|------------|----------------------|-----------|----------|------------|---------------|-------------------|
| **tiny** | 6 | 256 | 4 | 1024 | ~23M | ~2-4GB | ~500M |
| **small** | 12 | 512 | 8 | 2048 | ~42M | ~4-6GB | ~840M |
| **base** | 24 | 768 | 12 | 3072 | ~124M | ~8-12GB | ~2.5B |

**Note**: Memory shown is base model memory. Training adds ~2-4x for gradients, optimizer states, and activations.

### Scaling Relationships

```
FFN_size = 4 × d_model (standard transformer ratio, SwiGLU)
head_dim = d_model / num_heads = 64 (consistent across all models)
vocab_size = 32768 (fixed, SentencePiece optimized)
sequence_length = 1024 (default, configurable)
```

### Chinchilla-Optimal Training
Following the Chinchilla scaling laws, optimal training uses **~20 tokens per parameter**:
- **tiny (23M)**: ~500M tokens
- **small (42M)**: ~840M tokens
- **base (124M)**: ~2.5B tokens

See training configs in `config/pretrain/training/*.json` for implementation.

---

## 🧠 Detailed Architecture Components

### 1. Token Embeddings
```yaml
Type: Learned embeddings
Vocabulary: 32,768 tokens (SentencePiece)  
Dimension: d_model
Initialization: Normal(mean=0, std=0.02)
Tied Weights: False (separate input/output embeddings)
```

**Justification**: 32K vocab provides good balance between efficiency and coverage for English text. No weight tying allows more flexibility during training.

### 2. Positional Encoding: RoPE (Rotary Position Embedding)
```yaml
Type: Rotary Position Embedding
Base Frequency (θ): 10000.0
Application: Applied to Q and K in attention
Max Context: 1024 (tiny/small), 2048 (base)
Extrapolation: Linear (no scaling by default)
```

**Technical Details**:
- Applies rotation matrices to query and key vectors
- Encodes relative positions naturally
- Better extrapolation to longer sequences than learned embeddings
- No additional parameters required

**Why RoPE over alternatives**:
- ✅ **vs Sinusoidal**: Better handling of relative positions
- ✅ **vs Learned**: No additional parameters, better length generalization  
- ✅ **vs ALiBi**: More established, better HuggingFace support

### 3. Transformer Layers

#### Layer Structure (Pre-LN variant):
```
Input
  ↓
RMSNorm → Multi-Head Attention → Residual Add
  ↓
RMSNorm → Feed-Forward Network → Residual Add
  ↓
Output
```

#### 3.1 RMSNorm (Root Mean Square Normalization)
```yaml
Type: RMSNorm (instead of LayerNorm)
Epsilon: 1e-5
Learnable Scale: Yes (γ parameter)
Bias: No bias term
Position: Pre-normalization
```

**Mathematical Formula**:
```
RMSNorm(x) = (x / RMS(x)) * γ
where RMS(x) = sqrt(mean(x²) + ε)
```

**Advantages over LayerNorm**:
- ✅ **Simplicity**: No mean subtraction, only RMS scaling
- ✅ **Efficiency**: ~7% faster computation
- ✅ **Stability**: Better numerical stability
- ✅ **Memory**: Slightly lower memory usage

#### 3.2 Multi-Head Attention (MHA)

```yaml
Attention Type: Scaled Dot-Product with Causal Masking
Heads: 4 (tiny), 8 (small), 12 (base)
Head Dimension: 64 (d_model / num_heads)
QKV Projection: Linear layers (no bias)
Output Projection: Linear layer (no bias)
Dropout: Applied to attention weights and output
```

**Attention Mechanism**:
```
Attention(Q,K,V) = softmax(QK^T / √d_k + mask) V
where mask prevents attention to future tokens
```

**Key Features**:
- **Causal Masking**: Lower triangular mask for autoregressive generation
- **Grouped Query Attention Ready**: `num_key_value_heads = num_heads` (can be reduced for GQA)
- **FlashAttention-2**: Automatic memory-efficient implementation when available
- **No Bias**: Following LLaMA design for better scaling

#### 3.3 Feed-Forward Network (SwiGLU)

```yaml
Architecture: SwiGLU (Swish-Gated Linear Unit)
Hidden Size: 4 × d_model (1024, 2048, 3072)
Activation: SiLU (Swish) with gating
Components: up_proj, gate_proj, down_proj
Bias: No bias terms
```

**SwiGLU Formula**:
```
SwiGLU(x) = SiLU(up_proj(x)) ⊙ gate_proj(x)
FFN(x) = down_proj(SwiGLU(x))
```

Where:
- `SiLU(x) = x * sigmoid(x)` (Swish activation)  
- `⊙` denotes element-wise multiplication (gating)

**Why SwiGLU over alternatives**:
- ✅ **vs ReLU**: Better gradient flow, no dead neurons
- ✅ **vs GELU**: Smoother gradients, better empirical results
- ✅ **vs Standard GLU**: SiLU activation works better than sigmoid
- ✅ **Performance**: Consistent improvements in language modeling

---

## ⚙️ Training-Specific Configurations

### Initialization Strategy
```yaml
Linear Layers: Normal(mean=0, std=0.02)
Embeddings: Normal(mean=0, std=0.02) 
RMSNorm Scale: Ones initialization
Attention Output: Scaled by 1/√(2*num_layers) for stability
```

### Dropout Configuration
```yaml
Attention Dropout: 0.1 (applied to attention weights)
Hidden Dropout: 0.1 (applied to FFN output)  
Embedding Dropout: 0.1 (applied after token embeddings)
Residual Dropout: Applied before residual connections
```

### Mixed Precision Training
```yaml
Primary Type: BF16 (bfloat16, recommended)
Alternative: FP16 (if BF16 not available)
Master Weights: FP32 (for optimizer states)
Loss Scaling: Not needed for BF16, dynamic for FP16
FlashAttention: Uses BF16/FP16 natively for efficiency
```

**Why BF16 over FP16**:
- ✅ Better numerical stability (same exponent range as FP32)
- ✅ No loss scaling needed
- ✅ Handles gradients better during training
- ✅ Supported on Ampere+ GPUs (RTX 30/40 series)

---

## 🚀 Optimization Choices

### Memory Optimizations

#### 1. FlashAttention-2
```yaml
Memory Reduction: ~50% during training
Speed Improvement: ~30% on RTX 4090
Fallback Chain: FlashAttention-2 → SDPA → Eager
Compatibility: CUDA 11.6+, SM 8.0+ (RTX 30/40 series)
```

#### 2. Gradient Checkpointing  
```yaml
Memory Reduction: ~40% (trades compute for memory)
Implementation: Recompute activations during backward pass
Trade-off: ~10% speed decrease for memory savings
```

#### 3. Parameter Efficiency
```yaml
No Bias Terms: ~10-15% parameter reduction
Tied Embeddings: Disabled (allows more flexibility)
Shared Layers: Not used (maintains full expressivity)
```

### Computational Optimizations

#### 1. Attention Mechanism Hierarchy
```
1. FlashAttention-2 (if available)
   ↓ (fallback if import/runtime fails)
2. PyTorch SDPA (Scaled Dot Product Attention)  
   ↓ (fallback if unsupported)
3. Manual Attention (always works)
```

#### 2. Activation Functions
- **SiLU (Swish)**: More efficient than GELU, better than ReLU
- **No Approximations**: Uses exact SiLU for numerical stability

#### 3. Linear Layer Optimizations
- **No Bias**: Reduces parameters and computation
- **Proper Initialization**: Prevents gradient explosion/vanishing

---

## 📏 Architecture Comparison

### Lumi vs Original LLaMA

| Component | Lumi | LLaMA | Notes |
|-----------|------|-------|--------|
| **Architecture** | Decoder-only | Decoder-only | ✅ Identical |
| **Normalization** | RMSNorm | RMSNorm | ✅ Identical |
| **Activation** | SwiGLU | SwiGLU | ✅ Identical |
| **Position** | RoPE | RoPE | ✅ Identical |
| **Attention** | MHA/GQA-ready | GQA | ⚠️ MHA by default, GQA available |
| **Vocab Size** | 32K | 32K | ✅ Identical |
| **Bias** | No bias | No bias | ✅ Identical |
| **Scale** | 6M-124M | 7B-70B+ | ⚠️ Much smaller for personal use |

### Lumi vs GPT-2/GPT-3

| Component | Lumi | GPT-2/3 | Advantage |
|-----------|------|---------|-----------|
| **Normalization** | RMSNorm | LayerNorm | ✅ Faster, more stable |
| **Activation** | SwiGLU | GELU/ReLU | ✅ Better empirical results |
| **Position** | RoPE | Learned/Sinusoidal | ✅ Better length extrapolation |
| **Attention** | Causal MHA | Causal MHA | ✅ Similar |
| **Bias** | No bias | With bias | ✅ Fewer parameters |

---

## 🎯 Design Justifications

### Why These Specific Configurations?

#### Tiny Model (23M parameters)
```yaml
Target: Fast iteration, proof-of-concept, learning
Layers: 6 (minimum for reasonable depth)
Hidden: 256 (smallest practical size)
Heads: 4 (maintains head_dim=64)
Context: 1024 (sufficient for most tasks)
Training: ~24h on RTX 4090, works on 8GB GPUs
Chinchilla: ~500M tokens
```

#### Small Model (42M parameters)
```yaml
Target: Balanced development, fine-tuning experiments
Layers: 12 (good depth-width balance)
Hidden: 512 (2x tiny for 7x parameters)
Heads: 8 (maintains head_dim=64)
Context: 1024 (memory-efficient)
Training: ~48h on RTX 4090, works on 8GB GPUs (batch=2)
Chinchilla: ~840M tokens
```

#### Base Model (124M parameters)
```yaml
Target: Production use, best quality
Layers: 24 (deeper for better representation)
Hidden: 768 (standard BERT-base size)
Heads: 12 (maintains head_dim=64)
Context: 1024 (default, expandable to 2048)
Training: ~120h on RTX 4090, tight on 8GB GPUs
Chinchilla: ~2.5B tokens
```

### Head Dimension Consistency
All models use `head_dim = 64` because:
- ✅ **Empirically optimal**: Best performance across many studies
- ✅ **Hardware efficient**: Aligns well with GPU memory/compute
- ✅ **RoPE compatibility**: Works well with rotary embeddings  
- ✅ **FlashAttention**: Optimal for memory-efficient attention

---

## 🔬 Advanced Architecture Features

### Grouped Query Attention (GQA) Support
```yaml
Current: MHA (num_kv_heads = num_heads)
Available: GQA (num_kv_heads < num_heads) 
Configuration: Set num_key_value_heads in config
Memory Savings: ~30% for inference with minimal quality loss
```

**GQA Example Configuration**:
```json
{
  "num_attention_heads": 12,
  "num_key_value_heads": 4,  // 3:1 ratio (typical)
  // ... other params
}
```

### FlashAttention-2 Integration
```yaml
Automatic Detection: Checks for flash_attn availability
Graceful Fallback: SDPA → Eager attention
Memory Benefits: ~50% reduction in attention memory
Speed Benefits: ~30% faster training on RTX 4090
Compatibility: Handles different CUDA versions gracefully
```

### Deterministic Training Support
```yaml
Seed Management: Complete RNG state control
CUDNN Settings: Deterministic mode available  
Checkpoint States: All random states preserved
Reproducibility: Bit-exact reproduction possible
```

---

## 🏎️ Performance Characteristics

### Memory Usage (with BF16 + FlashAttention-2)

| Model | Training (batch=8) | Training (batch=4) | Training (batch=2) | Inference | Works on 8GB GPU? |
|-------|-------------------|-------------------|-------------------|-----------|-------------------|
| **tiny (23M)** | ~6-8 GB | ~4-5 GB | ~3-4 GB | ~2 GB | ✅ Yes |
| **small (42M)** | ~8-12 GB | ~5-7 GB | ~4-5 GB | ~4 GB | ✅ Yes (batch=2) |
| **base (124M)** | ~12-18 GB | ~8-10 GB | ~6-7 GB | ~8 GB | ⚠️ Tight (batch=1-2) |

**For 8GB GPUs**: Use batch_size=2-4 with gradient_accumulation_steps=8-16 to maintain effective batch size.

### Training Speed (RTX 4090 with BF16 + FlashAttention-2)

| Model | Tokens/sec | Steps/hour | Chinchilla Training Time |
|-------|------------|-------------|--------------------------|
| **tiny (23M)** | ~600-800 | ~1000 | ~24h (500M tokens) |
| **small (42M)** | ~400-600 | ~600 | ~48h (840M tokens) |
| **base (124M)** | ~200-300 | ~300 | ~120h (2.5B tokens) |

**Note**: Times assume optimal batch size. Smaller GPUs will need smaller batches → longer training time but same final quality.

### Inference Speed (RTX 4090)

| Model | Tokens/sec (batch=1) | Throughput (batch=8) |
|-------|---------------------|---------------------|
| **tiny** | ~150 | ~800 |
| **small** | ~80 | ~400 |
| **base** | ~45 | ~200 |

---

## 🔧 Configuration Guidelines

### Choosing Model Size

**Use Tiny (23M) when**:
- 🧪 Prototyping new ideas
- ⚡ Need very fast training/iteration (~24h)
- 💾 Limited GPU memory (works great on 8GB)
- 🎯 Testing code/pipeline changes
- 📚 Learning LLM training from scratch
- 💰 Limited compute budget

**Use Small (42M) when**:
- 🔬 Balanced development work
- 📚 Fine-tuning experiments
- 🎯 Good quality without long training (~48h)
- 💾 Moderate GPU memory (8-16GB)
- 🏃 Want reasonable performance quickly

**Use Base (124M) when**:
- 🏆 Production deployment
- 📈 Best possible quality needed
- 💪 Have sufficient compute budget (~120h)
- 💾 Full GPU memory available (16GB+)
- 🎓 Research-grade results required

### Hyperparameter Recommendations

#### Learning Rates by Model Size
```yaml
tiny:  3e-4 to 5e-4  (higher for faster convergence)
small: 1e-4 to 3e-4  (balanced)  
base:  5e-5 to 1e-4  (lower for stability)
```

#### Batch Size Guidelines
```yaml
# RTX 4090 (16GB)
tiny:  8-16  (comfortable, can go higher)
small: 4-8   (optimal balance)
base:  2-4   (memory constrained)

# RTX 3060/3070 (8GB)
tiny:  2-4   (use gradient_accumulation=8-16)
small: 2     (use gradient_accumulation=16-32)
base:  1-2   (tight, use gradient_accumulation=32-64)
```

**Pro tip**: Use gradient accumulation to maintain effective batch size even with small per-device batches.

#### Context Length Trade-offs
```yaml
1024: Good for most tasks, memory efficient
2048: Better for long-form generation, 2x memory
4096: Experimental, requires gradient checkpointing
```

---

## 📚 References & Inspirations

### Primary References
1. **LLaMA Paper**: "LLaMA: Open and Efficient Foundation Language Models" (Meta, 2023)
2. **RMSNorm**: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)  
3. **SwiGLU**: "GLU Variants Improve Transformer" (Shazeer, 2020)
4. **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
5. **FlashAttention**: "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)

### Implementation Inspirations
- **HuggingFace Transformers**: LlamaForCausalLM implementation
- **Meta's LLaMA**: Official architecture choices
- **Andrej Karpathy's nanoGPT**: Clean minimal implementation
- **EleutherAI GPT-NeoX**: Training best practices

---

## 🎯 Architecture Evolution

This architecture represents the **current stable implementation**. Future considerations:

### Potential Improvements
- **Multi-Query Attention (MQA)**: Even more memory-efficient than GQA
- **RMSNorm variants**: QKNorm, LayerScale integration
- **Advanced position encodings**: ALiBi, rotary improvements
- **Mixture of Experts (MoE)**: Sparse parameter scaling
- **Better activation functions**: Recent Swish/GELU improvements

### Compatibility Commitment  
All changes will maintain **backward compatibility** with existing checkpoints and configurations. Architecture evolution will be **opt-in** through configuration flags.

---

## 🌍 Hardware Accessibility

This architecture is designed to be **democratically accessible**:

- **RTX 4090 (16GB)**: Optimal performance, all models comfortable
- **RTX 3080/3090 (10-12GB)**: All models work well with adjusted batch sizes
- **RTX 3060/3070 (8GB)**: Tiny and Small models work great, Base is possible with patience
- **Lower-end cards**: Tiny model works on even more modest hardware

**Philosophy**: You don't need a datacenter to train LLMs from scratch. A consumer GPU, patience, and good coffee ☕ are enough.

> *"I don't have an excavator, I have a fork. But technical complexity has never been an obstacle, only an exciting puzzle."*

---

*This architecture documentation is maintained alongside the codebase and updated with each major release. Last updated: January 2025*