# Plan de Test Complet - Modèle 23M (Architecture Réelle)
## Pipeline de A à Z: Zero → Évaluation SFT

**Date**: 2025-10-02
**Objectif**: Tester le pipeline complet avec l'architecture réelle refactorisée
**Modèle**: Tiny 23M - 600M tokens Chinchilla-optimal
**Durée estimée**: ~6-10 heures

---

## 📋 Architecture du Pipeline

### Datasets Utilisés

#### Pour le Tokenizer (100M tokens)
- **Config**: `config/pretrain/corpus/tokenizer_training_mix.json`
- **Sources**:
  - C4: 28M tokens (28%)
  - Gutenberg: 24M tokens (24%)
  - FineWeb-Edu: 24M tokens (24%)
  - Wikipedia: 24M tokens (24%)

#### Pour le Pré-training (600M tokens)
- **Config**: `config/pretrain/corpus/tiny_23M_chinchilla_500M.json`
- **Sources**:
  - C4: 150M tokens (25%)
  - Gutenberg: 150M tokens (25%)
  - FineWeb-Edu: 150M tokens (25%)
  - Wikipedia: 150M tokens (25%)

---

## 🗂️ Structure Attendue

```
data/
├── datasets/
│   ├── tokenizer_corpus/          # Étape 1 (100M tokens)
│   │   ├── cache/
│   │   ├── shards/
│   │   │   ├── shard_0000.jsonl.gz
│   │   │   └── ... (50 shards)
│   │   ├── manifest.json
│   │   └── plan.json
│   │
│   └── tiny_23M_corpus/            # Étape 3 (600M tokens)
│       ├── cache/
│       ├── shards/
│       ├── manifest.json
│       └── plan.json
│
├── models/
│   └── tokenizers/
│       └── spm_32k/                # Étape 2
│           ├── spm.model
│           ├── spm.vocab
│           ├── tokenizer_config.json
│           └── TOKENIZER_CARD.md
│
├── processed/
│   └── tiny_23M_1024/              # Étape 4
│       ├── train.bin
│       ├── train.idx
│       ├── val.bin
│       ├── val.idx
│       └── manifest.json
│
└── sft_processed/                  # Étape 7
    ├── alpaca_chatml/
    └── oasst1_chatml/

checkpoints/
├── pretrain/
│   └── tiny_23M/                   # Étape 5
│       ├── step_5000/
│       └── final/
└── sft/
    └── tiny_23M/                   # Étape 8
        └── final/

evaluation_results/
├── pretrain/tiny_23M/              # Étape 6
└── sft/tiny_23M/                   # Étape 9
```

---

## 📝 Plan de Test Détaillé

### ÉTAPE 1: Préparation Corpus Tokenizer (100M tokens)

#### 1.1 Analyse du Plan (Optionnel - Vérification)
```bash
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/tokenizer_training_mix.json \
    --output-dir data/datasets/tokenizer_corpus \
    --analyze-only \
    --log-level INFO
```

**Note**: `shard_size` est maintenant dans le config (`processing_params.shard_size`, défaut=50000), pas en CLI.

**Vérifications**:
- [ ] Plan généré: `data/datasets/tokenizer_corpus/plan.json`
- [ ] Rapport affiché:
  - C4: ~34K samples (28M tokens)
  - Gutenberg: ~118 samples (24M tokens)
  - FineWeb: ~13K samples (24M tokens)
  - Wikipedia: ~2.3K samples (24M tokens)
  - **TOTAL: ~50K samples, 100M tokens**

---

#### 1.2 Génération du Corpus Tokenizer
```bash
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/tokenizer_training_mix.json \
    --output-dir data/datasets/tokenizer_corpus \
    --use-cache \
    --log-level INFO \
    2>&1 | tee logs/01_tokenizer_corpus.log
```

**Vérifications en Cours**:
- [ ] **Phase 1 - Streaming Processing**:
  - 4 sources traitées séquentiellement
  - Chaque source: `🚀Streaming {source}: XXXtokens`
  - Memory: CONSTANT (~50MB max)
  - Token budget atteint pour chaque source

- [ ] **Phase 2 - Assembly**:
  - `🔧 Starting Phase 2: TRUE STREAMING ASSEMBLY`
  - 50 shards créés
  - Deduplication globale: ~0.3%
  - Target tokens: 100M

**Sortie Attendue**:
```
✅ TRUE STREAMING processing complete!
   📊 Final documents: ~83K
   🎯 Exact tokens: ~100M
   💾 Peak memory: <100MB
   📦 Output shards: 50
```

**Fichiers Créés**:
- [ ] `data/datasets/tokenizer_corpus/manifest.json`
- [ ] `data/datasets/tokenizer_corpus/shards/` (50 fichiers .jsonl.gz)
- [ ] `data/datasets/tokenizer_corpus/cache/` (fichiers temporaires)

**Durée estimée**: 10-15 min

---

### ÉTAPE 2: Training Tokenizer (32K vocab)

```bash
python scripts/20_train_tokenizer.py \
    --config config/pretrain/tokenizer/spm32k.json \
    --output-dir data/models/tokenizers/spm_32k \
    --log-level INFO \
    2>&1 | tee logs/02_tokenizer_training.log
```

**Vérifications en Cours**:
- [ ] **Integrity Verification**: Tous les shards vérifiés
- [ ] **Corpus Analysis**:
  - 50/50 shards traités
  - Stats: Docs, Sentences, Tokens affichés
- [ ] **Sentence Streaming**:
  - `📊 Progress bar` avec sentences/s
  - Target: 100M sentences max
  - Actual: ~2.8M sentences written
- [ ] **SentencePiece Training**:
  - EM training iterations
  - Shrinking vocabulary
  - Final vocab: 32,768

**Sortie Attendue**:
```
✅ Tokenizer training complete!
📁 Output: data/models/tokenizers/spm_32k/
   📝 spm.model (vocab_size=32768)
   📊 Compression ratio: ~4.1 chars/token
   🎯 Fertility: 1.15
   📈 Coverage: 99.95%
```

**Fichiers Créés**:
- [ ] `spm.model` (le tokenizer)
- [ ] `spm.vocab` (vocabulaire)
- [ ] `tokenizer_config.json` (config + SHA256)
- [ ] `TOKENIZER_CARD.md` (métriques)

**SHA256 CRITIQUE**:
```bash
# Noter le hash pour validation
cat data/models/tokenizers/spm_32k/tokenizer_config.json | jq '.sha256_hash'
# Exemple: "a1b2c3d4..."
```

**Durée estimée**: 5-10 min

---

### ÉTAPE 3: Préparation Corpus Pré-training (600M tokens)

#### 3.1 Analyse du Plan
```bash
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/tiny_23M_chinchilla_500M.json \
    --output-dir data/datasets/tiny_23M_corpus \
    --analyze-only \
    --log-level INFO
```

**Vérifications**:
- [ ] Plan avec 4 sources:
  - C4: 150M tokens (25%)
  - Gutenberg: 150M tokens (25%)
  - FineWeb: 150M tokens (25%)
  - Wikipedia: 150M tokens (25%)
  - **TOTAL: 600M tokens**

---

#### 3.2 Génération du Corpus Complet
```bash
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/tiny_23M_chinchilla_500M.json \
    --output-dir data/datasets/tiny_23M_corpus \
    --use-cache \
    --log-level INFO \
    2>&1 | tee logs/03_pretrain_corpus.log
```

**Vérifications**:
- [ ] 4 sources traitées (C4, Gutenberg, FineWeb, Wikipedia)
- [ ] Chaque source: 150M tokens
- [ ] Assembly: ~300 shards (600M tokens / 2M per shard)
- [ ] Deduplication globale appliquée

**Sortie Attendue**:
```
✅ TRUE STREAMING processing complete!
   📊 Final documents: ~500K
   🎯 Exact tokens: ~600M
   💾 Peak memory: <100MB
   📦 Output shards: ~300
```

**Durée estimée**: 30-60 min (dépend du streaming HuggingFace)

---

### ÉTAPE 4: Packing du Dataset (Tokenization + Séquences)

```bash
python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/tiny_23M_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/tiny_23M_1024 \
    --log-level INFO \
    2>&1 | tee logs/04_packing.log
```

**Vérifications**:
- [ ] Tokenizer chargé: SHA256 validé
- [ ] Corpus manifest chargé
- [ ] Tokenization streaming (shards traités un par un)
- [ ] Packing: séquences de 1024 tokens
- [ ] Train/val split: 98/2

**Sortie Attendue**:
```
✅ Sequence packing complete!
   📊 Total tokens: ~600M
   🎯 Sequences: ~586K (1024 tokens each)
   📂 Train: ~574K sequences (98%)
   📂 Val: ~12K sequences (2%)
   💾 Files:
      - train.bin (~1.1 GB)
      - train.idx
      - val.bin (~23 MB)
      - val.idx
```

**Fichiers Créés**:
- [ ] `train.bin` + `train.idx`
- [ ] `val.bin` + `val.idx`
- [ ] `manifest.json` (avec tokenizer_sha256)

**Validation Tokenizer**:
```bash
cat data/processed/tiny_23M_1024/manifest.json | jq '.tokenizer_sha256'
# DOIT être identique au tokenizer
```

**Durée estimée**: 20-40 min

---

### ÉTAPE 5: Pré-training (23M paramètres)

```bash
accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/tiny_23M_1024 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain \
    --num_workers 4 \
    2>&1 | tee logs/05_pretrain.log
```

**Config Utilisée** (`chinchilla_tiny_500m.json`):
- Architecture: `config/architectures/tiny.json` (23M params)
- Max steps: 30,000
- Batch size: 8 per device
- Gradient accumulation: 4
- Learning rate: 3e-4 (cosine + warmup 3K steps)
- Eval steps: 2,500
- Save steps: 5,000

**Vérifications en Cours**:
- [ ] **Démarrage**:
  - Tokenizer SHA256 validé
  - Model: "23M parameters"
  - FlashAttention-2 activé (ou SDPA fallback)
  - Dataset: 574K train sequences

- [ ] **Training**:
  - Loss diminue: ~10 → <3.0
  - Learning rate: warmup puis cosine decay
  - GPU util > 85%
  - VRAM: ~8-12 GB

- [ ] **Checkpoints**:
  - Tous les 5,000 steps
  - Contient: model, optimizer, scheduler, meta, rng_state

- [ ] **Évaluation** (tous les 2,500 steps):
  - Validation perplexity calculée
  - Devrait diminuer au fil du training

**Sortie Attendue**:
```
✅ Training complete!
   📊 Steps: 30,000
   📉 Final loss: 2.8-3.2
   ⏱️  Duration: 2-4h
   💾 Checkpoints: step_5000, 10000, ..., 30000, final
```

**Durée estimée**: 2-4 heures (RTX 4090)

---

### ÉTAPE 6: Évaluation Pré-training

```bash
python scripts/50_evaluate_pretrain.py \
    --model_path checkpoints/pretrain/tiny/final \
    --tokenizer_path data/models/tokenizers/spm_32k \
    --output_dir evaluation_results/pretrain/tiny \
    --log-level INFO \
    2>&1 | tee logs/06_eval_pretrain.log
```

**Métriques Attendues**:
- [ ] **WikiText-2 Perplexity**: 45-70
- [ ] **BoolQ Accuracy**: 58-68%
- [ ] **Générations**: Cohérentes mais pas conversationnelles

**Durée estimée**: 10-15 min

---

### ÉTAPE 7: Préparation Corpus SFT

#### 7.1 Alpaca (ChatML)
```bash
python scripts/60_prepare_sft_corpus.py \
    --config config/sft/datasets/alpaca_chatml.json \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir data/sft_processed/alpaca_chatml \
    --log-level INFO \
    2>&1 | tee logs/07a_sft_alpaca.log
```

#### 7.2 OASST1 (ChatML)
```bash
python scripts/60_prepare_sft_corpus.py \
    --config config/sft/datasets/oasst1_chatml.json \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir data/sft_processed/oasst1_chatml \
    --log-level INFO \
    2>&1 | tee logs/07b_sft_oasst1.log
```

**Vérifications**:
- [ ] Template ChatML appliqué:
  ```
  <|im_start|>user
  {instruction}
  <|im_end|>
  <|im_start|>assistant
  {response}
  <|im_end|>
  ```
- [ ] Tokenizer SHA256 vérifié (identique au pretrain)
- [ ] Train/val split créé
- [ ] Final manifest avec tokenizer_config_hash

**Durée estimée**: 5-10 min chacun

---

### ÉTAPE 8: SFT Training (LoRA)

```bash
accelerate launch --mixed_precision bf16 scripts/70_train_sft.py \
    --config config/sft/training/lora_optimal_tiny_23m.json \
    --model_path checkpoints/pretrain/tiny/final \
    --data_dirs data/sft_processed/alpaca_chatml data/sft_processed/oasst1_chatml \
    --data_weights 0.7 0.3 \
    --tokenizer_path data/models/tokenizers/spm_32k \
    --output_dir checkpoints/sft/tiny \
    2>&1 | tee logs/08_sft.log
```

**Config** (`lora_optimal_tiny_23m.json`):
- LoRA: r=32, alpha=64, dropout=0.05
- Learning rate: 8e-5
- Max steps: 2,000
- Batch size: 4, grad accum: 8

**Vérifications**:
- [ ] Base model chargé
- [ ] LoRA adapters: r=32, alpha=64
- [ ] Dataset mix: 70% Alpaca, 30% OASST1
- [ ] Loss SFT diminue: ~2.5 → <1.5
- [ ] VRAM < 8 GB

**Durée estimée**: 1-2 heures

---

### ÉTAPE 9: Évaluation SFT

```bash
python scripts/80_evaluate_sft.py \
    --config config/evaluation/sft_standard.json \
    --model_path checkpoints/sft/tiny/final \
    --tokenizer_path data/models/tokenizers/spm_32k \
    --output_file evaluation_results/sft/tiny/results.json \
    2>&1 | tee logs/09_eval_sft.log
```

**Métriques Attendues**:
- [ ] **SFT Perplexity**: 35-50
- [ ] **BoolQ Accuracy**: 68-75% (+10% vs pretrain)
- [ ] **Smoke Tests Quality**: 0.65-0.80
- [ ] **Générations**: Cohérentes, structurées (ChatML)

**Durée estimée**: 15-20 min

---

## 🔍 Validation Finale

### Tokenizer Consistency Check
```bash
echo "=== TOKENIZER SHA256 VALIDATION ==="
echo ""
echo "SOURCE:"
cat data/models/tokenizers/spm_32k/tokenizer_config.json | jq -r '.sha256_hash'
echo ""
echo "PRETRAIN DATASET:"
cat data/processed/tiny_23M_1024/manifest.json | jq -r '.tokenizer_sha256'
echo ""
echo "SFT DATASETS:"
cat data/sft_processed/alpaca_chatml/manifest.json | jq -r '.tokenizer_sha256'
cat data/sft_processed/oasst1_chatml/manifest.json | jq -r '.tokenizer_sha256'
echo ""
echo "⚠️ ALL MUST BE IDENTICAL!"
```

---

## 📊 Métriques de Référence

### Pré-training (23M - 600M tokens)
| Métrique | Attendu | Obtenu |
|----------|---------|--------|
| Final Loss | 2.8-3.2 | _____ |
| WikiText-2 PPL | 45-70 | _____ |
| BoolQ Accuracy | 58-68% | _____ |
| Tokens/sec (4090) | ~600-800 | _____ |
| VRAM (BF16) | ~8-12 GB | _____ |

### SFT (23M - LoRA r=32)
| Métrique | Attendu | Obtenu |
|----------|---------|--------|
| Final Loss | 1.5-2.0 | _____ |
| SFT Perplexity | 35-50 | _____ |
| BoolQ Accuracy | 68-75% | _____ |
| Smoke Quality | 0.65-0.80 | _____ |
| VRAM (LoRA) | ~6-8 GB | _____ |

---

## ✅ Critères de Succès

1. **Pipeline complet** sans erreur ✓
2. **Tokenizer SHA256** identique partout ✓
3. **Corpus streaming** avec <100MB memory ✓
4. **Pretrain loss** < 3.0 ✓
5. **SFT loss** < 2.0 ✓
6. **BoolQ amélioration** +10% après SFT ✓
7. **Générations SFT** cohérentes (ChatML) ✓
8. **Configs utilisés** (pas de hardcoded) ✓

---

## 🚀 Commandes Rapides (TL;DR)

```bash
# SETUP
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
mkdir -p logs

# Étape 1: Corpus Tokenizer (100M)
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/tokenizer_training_mix.json \
    --output-dir data/datasets/tokenizer_corpus \
    --use-cache \
    --log-level INFO

# Étape 2: Tokenizer Training (32K)
python scripts/20_train_tokenizer.py \
    --config config/pretrain/tokenizer/spm32k.json \
    --output-dir data/models/tokenizers/spm_32k \
    --log-level INFO

# Étape 3: Corpus Pretrain (600M)
python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/tiny_23M_chinchilla_500M.json \
    --output-dir data/datasets/tiny_23M_corpus \
    --use-cache \
    --log-level INFO

# Étape 4: Packing
python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/tiny_23M_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/tiny_23M_1024 \
    --log-level INFO

# Étape 5: Pretrain (avec accelerate)
accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/tiny_23M_1024 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain \
    --num_workers 4

# Étape 6: Eval Pretrain
python scripts/50_evaluate_pretrain.py \
    --model_path checkpoints/pretrain/tiny/final \
    --tokenizer_path data/models/tokenizers/spm_32k \
    --output_dir evaluation_results/pretrain/tiny

# Étape 7: SFT Corpus
python scripts/60_prepare_sft_corpus.py \
    --config config/sft/datasets/alpaca_chatml.json \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir data/sft_processed/alpaca_chatml

python scripts/60_prepare_sft_corpus.py \
    --config config/sft/datasets/oasst1_chatml.json \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir data/sft_processed/oasst1_chatml

# Étape 8: SFT Train (avec accelerate)
accelerate launch --mixed_precision bf16 scripts/70_train_sft.py \
    --config config/sft/training/lora_optimal_tiny_23m.json \
    --model_path checkpoints/pretrain/tiny/final \
    --data_dirs data/sft_processed/alpaca_chatml data/sft_processed/oasst1_chatml \
    --data_weights 0.7 0.3 \
    --tokenizer_path data/models/tokenizers/spm_32k \
    --output_dir checkpoints/sft/tiny

# Étape 9: Eval SFT
python scripts/80_evaluate_sft.py \
    --config config/evaluation/sft_standard.json \
    --model_path checkpoints/sft/tiny/final \
    --tokenizer_path data/models/tokenizers/spm_32k \
    --output_file evaluation_results/sft/tiny/results.json
```

---

**Prêt pour le test complet! 🚀**
