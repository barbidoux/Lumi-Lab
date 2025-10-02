# Project Structure - Lumi-Lab v2.0

**Clean, logical, hierarchical organization**

---

## 📂 Complete Directory Tree

```
Lumi-Lab/
│
├── 📜 README.md                    # Main documentation
├── 📜 CLAUDE.md                    # Claude Code instructions
├── 📜 MIGRATION.md                 # Migration guide v1→v2
├── 📜 BREAKING_CHANGES.md          # Breaking changes list
├── 📜 CLEANUP_SUMMARY.md           # Cleanup summary
├── 📜 TODO_REMAINING.md            # Remaining tasks
├── 📜 PROJECT_STRUCTURE.md         # This file
├── 📜 Makefile                     # Build automation (⚠️ needs update)
├── 📜 requirements.txt             # Python dependencies
│
├── 📁 scripts/                     # Main pipeline scripts
│   ├── 📄 __init__.py
│   │
│   ├── 🔟 PRE-TRAINING PIPELINE (10-50)
│   ├── 📄 10_prepare_corpus.py              # Clean raw data → corpus
│   ├── 📄 20_train_tokenizer.py             # Train SentencePiece tokenizer
│   ├── 📄 30_pack_dataset.py                # Tokenize + pack sequences
│   ├── 📄 40_pretrain.py                    # Pre-train transformer
│   ├── 📄 50_evaluate_pretrain.py           # Evaluate pre-trained model
│   │
│   ├── 6️⃣0️⃣ SFT PIPELINE (60-80)
│   ├── 📄 60_prepare_sft_corpus.py          # Prepare SFT corpus
│   ├── 📄 70_train_sft.py                   # Train with LoRA (industrial)
│   ├── 📄 80_evaluate_sft.py                # Evaluate SFT model
│   │
│   ├── 9️⃣0️⃣ DPO PIPELINE (90)
│   ├── 📄 90_train_dpo.py                   # DPO training
│   │
│   ├── 💯 SERVING (100)
│   ├── 📄 100_serve.py                      # Interactive/API serving
│   │
│   └── 📁 helpers/                          # Helper scripts
│       └── 📄 merge_lora.py                 # Merge LoRA + base model
│
├── 📁 config/                      # Configuration files
│   │
│   ├── 📁 architectures/           # Model architectures
│   │   ├── 📄 tiny.json                     # 23M params (6 layers, 256 dim)
│   │   ├── 📄 small.json                    # 124M params (12 layers, 512 dim)
│   │   └── 📄 base.json                     # 350M params (24 layers, 768 dim)
│   │
│   ├── 📁 pretrain/                # Pre-training configs
│   │   ├── 📁 sources/                      # Raw corpus sources
│   │   │   ├── 📄 c4_english.json
│   │   │   ├── 📄 wikipedia_en.json
│   │   │   ├── 📄 openwebtext.json
│   │   │   ├── 📄 gutenberg.json
│   │   │   └── 📄 fineweb_edu.json
│   │   │
│   │   └── 📁 training/                     # Training configs
│   │       ├── 📄 tokenizer_training_mix.json
│   │       ├── 📄 chinchilla_tiny_500m.json
│   │       ├── 📄 chinchilla_small_2b.json
│   │       └── 📄 chinchilla_base_7b.json
│   │
│   ├── 📁 sft/                     # SFT configs
│   │   ├── 📁 datasets/                     # SFT dataset configs
│   │   │   ├── 📄 alpaca_dolly_oasst_instruct.json  ✅ USE THIS
│   │   │   ├── 📄 alpaca.json
│   │   │   ├── 📄 dolly15k.json
│   │   │   ├── 📄 oasst1.json
│   │   │   └── 📄 README.md
│   │   │
│   │   └── 📁 training/                     # LoRA training configs
│   │       ├── 📄 lora_tiny_23m.json        # Optimal for 23M
│   │       ├── 📄 lora_tiny.json            # Standard tiny
│   │       ├── 📄 lora_small.json
│   │       └── 📄 lora_base.json
│   │
│   ├── 📁 dpo/                     # DPO configs (future)
│   │   ├── 📁 datasets/
│   │   └── 📁 training/
│   │
│   ├── 📁 evaluation/              # Evaluation configs
│   │   └── 📄 smoke_test_prompts.json
│   │
│   └── 📄 tokens.json              # HuggingFace auth token (.gitignored)
│
├── 📁 utils/                       # Utility modules
│   ├── 📄 __init__.py
│   │
│   ├── ✅ ESSENTIAL UTILS
│   ├── 📄 model_utils.py                    # Model loading, checkpointing
│   ├── 📄 dataset_utils.py                  # StreamingSFTDataset, loaders
│   ├── 📄 sft_templates.py                  # Instruct/ChatML templates
│   ├── 📄 sft_evaluation.py                 # SFT evaluation metrics
│   ├── 📄 tokenizer_utils.py                # Tokenizer helpers + validate_sp32k_tokenizer()
│   ├── 📄 auth.py                           # HuggingFace authentication
│   │
│   └── 📁 debug/                            # Debug utilities (optional)
│       ├── 📄 adaptive_estimation.py
│       ├── 📄 corpus_cache.py
│       ├── 📄 tokenizer_validation.py
│       ├── 📄 tokenizer_metrics.py
│       ├── 📄 precise_token_counter.py
│       ├── 📄 robust_imports.py
│       └── 📄 validate_architecture.py
│
├── 📁 data/                        # Data directory (created during pipeline)
│   ├── 📁 raw/                              # Raw downloaded data
│   ├── 📁 corpus/                           # Stage 1: Cleaned corpus
│   ├── 📁 tokenizer/                        # Trained tokenizers
│   ├── 📁 processed/                        # Stage 3: Packed datasets
│   ├── 📁 sft_processed/                    # SFT processed data
│   └── 📁 models/                           # Downloaded models
│
├── 📁 checkpoints/                 # Training checkpoints
│   ├── 📁 pretrain/                         # Pre-training checkpoints
│   │   ├── 📁 tiny/
│   │   ├── 📁 small/
│   │   └── 📁 base/
│   │
│   ├── 📁 sft/                              # SFT checkpoints
│   │   └── 📁 {experiment_name}/
│   │       ├── 📁 final/                    # LoRA adapters
│   │       └── 📁 merged/                   # Merged model (if --merge_adapters)
│   │
│   └── 📁 dpo/                              # DPO checkpoints
│
├── 📁 evaluation_results/          # Evaluation outputs
│   └── 📄 *.json                            # Evaluation metrics
│
└── 📁 venv/                        # Python virtual environment (.gitignored)
```

---

## 🔢 Numbering Scheme

Scripts are numbered by **tens (10, 20, 30...)** for easy insertion of new stages:

| Range | Purpose | Current Scripts |
|-------|---------|-----------------|
| **10-50** | Pre-training pipeline | 5 scripts |
| **60-80** | SFT pipeline | 3 scripts |
| **90** | DPO pipeline | 1 script |
| **100+** | Serving/inference | 1 script |

**Benefits:**
- ✅ Clear ordering
- ✅ Room for expansion (e.g., add `35_validate_dataset.py` between 30 and 40)
- ✅ Easy to memorize
- ✅ Alphabetical = execution order

---

## 📊 Config Hierarchy

```
config/
├── architectures/      # WHAT to train (model size/architecture)
├── pretrain/          # HOW to pre-train
│   ├── sources/       #   - WHERE to get data
│   └── training/      #   - HOW MUCH data (Chinchilla budgets)
├── sft/               # HOW to fine-tune
│   ├── datasets/      #   - WHICH SFT datasets
│   └── training/      #   - WHICH LoRA config
├── dpo/               # HOW to align (future)
└── evaluation/        # HOW to evaluate
```

**Design Principles:**
1. **Separation of concerns** - Architecture vs data vs training
2. **Hierarchical** - Top-level = pipeline stage
3. **Self-documenting** - Directory name = purpose
4. **Scalable** - Easy to add new configs

---

## 🔄 Data Flow

### Pre-training Pipeline
```
Raw Data → [10_prepare_corpus] → Cleaned Corpus
                                      ↓
Corpus Mixture → [20_train_tokenizer] → SentencePiece Model (32k vocab)
                                              ↓
Corpus + Tokenizer → [30_pack_dataset] → Packed Sequences (1024 tokens)
                                              ↓
Packed Data → [40_pretrain] → Pre-trained Model
                                    ↓
Model + Tokenizer → [50_evaluate_pretrain] → Metrics
```

### SFT Pipeline
```
SFT Datasets → [60_prepare_sft_corpus] → Formatted Conversations
                                              ↓
Conversations + Tokenizer → [60 --enable_packing] → Packed SFT Data
                                                        ↓
Packed Data + Pre-trained Model → [70_train_sft] → LoRA Adapters
                                                        ↓
                                              [70 --merge_adapters]
                                                        ↓
                                                  Merged Model
                                                        ↓
Merged Model → [80_evaluate_sft] → SFT Metrics
```

---

## 🗂️ File Naming Conventions

### Scripts
```
{stage_number}_{action}_{target}.py

Examples:
- 10_prepare_corpus.py        # Stage 10, prepare corpus
- 70_train_sft.py              # Stage 70, train SFT
- 80_evaluate_sft.py           # Stage 80, evaluate SFT
```

### Configs
```
{category}/{subcategory}/{name}.json

Examples:
- architectures/tiny.json              # Tiny model architecture
- pretrain/sources/c4_english.json     # C4 corpus source
- sft/training/lora_tiny_23m.json      # LoRA config for 23M model
```

### Data Directories
```
data/{stage}/{name}_{vocab}_{seqlen}[_version]

Examples:
- data/corpus/c4/                                   # Stage 1: Cleaned corpus
- data/processed/c4_32k_1024/                       # Stage 3: Tokenized (32k vocab, 1024 seq)
- data/sft_processed/alpaca_instruct_32k_1024_v3/   # SFT packed (v3.0 format)
```

---

## 🎯 Quick Navigation Guide

**I want to...**

| Task | Location |
|------|----------|
| Change model size | `config/architectures/` |
| Add new corpus source | `config/pretrain/sources/` |
| Modify token budget | `config/pretrain/training/` |
| Configure SFT dataset | `config/sft/datasets/` |
| Tune LoRA hyperparams | `config/sft/training/` |
| Run pre-training | `scripts/40_pretrain.py` |
| Run SFT | `scripts/70_train_sft.py` |
| Evaluate model | `scripts/50_evaluate_pretrain.py` or `scripts/80_evaluate_sft.py` |
| Serve model | `scripts/100_serve.py` |
| Debug tokenizer | `utils/debug/tokenizer_metrics.py` |

---

## ✅ Design Principles

This structure follows:

1. **Modularity** - Each script has one clear responsibility
2. **Separation of Concerns** - Config vs code vs data
3. **Discoverability** - Numbered scripts = execution order
4. **Scalability** - Room to add new stages (15, 25, 35...)
5. **Clarity** - Self-documenting names and hierarchy
6. **Best Practices** - Industry-standard organization

---

## 🆚 Comparison: Old vs New

### Old Structure (v1.x)
```
❌ Flat config directory
❌ Scripts numbered 01-06 (ambiguous)
❌ ChatML mixed with Instruct
❌ Utils all in one place
❌ Cross-imports between scripts
```

### New Structure (v2.0)
```
✅ Hierarchical config (architectures/pretrain/sft)
✅ Scripts numbered 10-100 (clear stages)
✅ Instruct format only (works!)
✅ Essential vs debug utils separated
✅ Zero cross-imports (functions in utils/)
```

**Result**: -50% config files, clearer organization, easier maintenance

---

## 📚 Related Documentation

- [README.md](README.md) - Getting started guide
- [MIGRATION.md](MIGRATION.md) - v1→v2 migration guide
- [BREAKING_CHANGES.md](BREAKING_CHANGES.md) - Breaking changes list
- [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) - What changed in cleanup
- [TODO_REMAINING.md](TODO_REMAINING.md) - Remaining tasks
- [CLAUDE.md](CLAUDE.md) - Claude Code instructions

---

**Last Updated**: 2025-10-02
**Version**: 2.0.0
