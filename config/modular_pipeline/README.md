# 📁 Configuration Pipeline Modulaire - Organisation

## 📋 Structure des Dossiers

```
config/
├── datasets/                    # 🟢 ANCIENNE CONFIGURATION (Legacy)
│   ├── c4_en.json              # Anciens configs individuels
│   ├── gutenberg_books.json
│   ├── stackexchange.json
│   └── ...                     # Autres anciens fichiers
│
├── meta_configs/               # 🟢 ANCIENNE CONFIGURATION (Legacy)
│   ├── phase_a.json           # Anciens meta-configs
│   ├── tiny_23M_chinchilla.json
│   └── ...
│
└── modular_pipeline/           # 🔵 NOUVELLE CONFIGURATION (Modular)
    ├── README.md              # Ce fichier d'organisation
    ├── sources/               # Configurations des sources individuelles
    │   ├── c4_english.json    # Source C4
    │   ├── gutenberg_books.json # Source Gutenberg
    │   ├── fineweb_edu.json   # Source FineWeb-Edu
    │   └── vietgpt_wikipedia.json # Source VietGPT Wikipedia
    │
    └── training_configs/       # Configurations complètes d'entraînement
        ├── tokenizer_training_mix.json    # Config tokenizer (7M tokens)
        ├── tiny_23M_chinchilla_500M.json  # Config dataset 23M (500M tokens)
        └── demo_fast.json                 # Config test rapide (100K tokens)
```

---

## 🎯 Configurations Disponibles

### **1. Entraînement Tokenizer OPTIMAL**
- **Fichier**: `training_configs/tokenizer_training_mix.json`
- **Usage**: Entraîner le tokenizer SentencePiece global (optimal absolu)
- **Tokens**: 25M tokens échantillonnés (4 sources équilibrées)
- **Distribution**: C4(7M), VietGPT(6M), FineWeb(6M), Gutenberg(6M)
- **Durée**: ~45-60 minutes
- **Réutilisable**: 23M ET 124M modèles

### **2. Dataset Tiny 23M Chinchilla**
- **Fichier**: `training_configs/tiny_23M_chinchilla_500M.json`
- **Usage**: Dataset complet pour modèle 23M paramètres
- **Tokens**: 500M tokens (Chinchilla-optimal 22:1)
- **Sources**: C4, Gutenberg, FineWeb-Edu, VietGPT-Wikipedia (25% chacune)
- **Durée**: ~3-4 heures

### **3. Modèle 124M - Phase A (Foundation)**
- **Fichier**: `training_configs/model_124M_phase_a.json`
- **Usage**: Phase A du curriculum learning pour 124M
- **Tokens**: 1.5B tokens (contenu structuré haute qualité)
- **Sources**: VietGPT-Wikipedia(50%), Gutenberg(33%), FineWeb-Edu(17%)
- **Stratégie**: Fondations linguistiques solides
- **Durée**: ~8-12 heures

### **4. Modèle 124M - Phase B (Robustness)**
- **Fichier**: `training_configs/model_124M_phase_b.json`
- **Usage**: Phase B du curriculum learning pour 124M
- **Tokens**: 1.5B tokens (contenu web diversifié)
- **Sources**: C4-WebCrawl(67%), FineWeb-Discussions(33%)
- **Stratégie**: Robustesse et généralisation
- **Durée**: ~8-12 heures

### **5. Test Rapide**
- **Fichier**: `training_configs/demo_fast.json`
- **Usage**: Validation rapide du pipeline
- **Tokens**: 100K tokens
- **Sources**: TinyStories + WikiText-2
- **Durée**: ~10-15 minutes

---

## 🚀 Commandes d'Utilisation

### **🔤 Tokenizer Optimal (Une Fois)**
```bash
# Entraîner le tokenizer réutilisable (25M tokens)
python run_modular_pipeline.py --config tokenizer --step all
```

### **🤖 Modèle Tiny 23M**
```bash
# Dataset complet 23M (500M tokens)
python run_modular_pipeline.py --config tiny_23M --step all
```

### **🎓 Modèle 124M - Curriculum Learning**
```bash
# Option 1: Pipeline automatique complet (2 phases)
python run_curriculum_124M.py --step all

# Option 2: Phase par phase
python run_curriculum_124M.py --step phase_a    # Foundation (1.5B tokens)
python run_curriculum_124M.py --step phase_b    # Robustness (1.5B tokens)

# Option 3: Phases individuelles avec contrôle granulaire
python run_modular_pipeline.py --config model_124M_phase_a --step all
python run_modular_pipeline.py --config model_124M_phase_b --step all
```

### **Test Rapide (Validation)**
```bash
# Pipeline complet de test
python scripts/01_prepare_corpus.py \
  --config config/modular_pipeline/training_configs/demo_fast.json \
  --output-dir data/modular/demo_corpus

python scripts/02_train_tokenizer.py \
  --manifest data/modular/demo_corpus/manifest.json \
  --output-dir data/modular/demo_tokenizer \
  --vocab-size 8192

python scripts/03_pack_sequences.py \
  --manifest data/modular/demo_corpus/manifest.json \
  --tokenizer-dir data/modular/demo_tokenizer \
  --output-dir data/modular/demo_packed \
  --sequence-length 512
```

---

## 🔧 Avantages de la Nouvelle Organisation

### ✅ **Séparation Claire**
- **Legacy**: `config/datasets/` et `config/meta_configs/`
- **Modular**: `config/modular_pipeline/`

### ✅ **Format JSON Uniforme**
- Fini les mélanges YAML/JSON
- Structure cohérente et lisible

### ✅ **Organisation Logique**
- **Sources individuelles**: descriptions et paramètres par dataset
- **Configs d'entraînement**: compositions complètes pour usage spécifique

### ✅ **Flexibilité Maximale**
- Réutilisation facile des sources
- Paramètres centralisés
- Traçabilité complète

---

## 📊 Résultats Attendus

### **Tokenizer (7M tokens)**
- Vocabulaire: 32,768 tokens
- Fichiers: `spm.model`, `spm.vocab`, `tokenizer_config.json`
- Temps: 20-30 minutes

### **Dataset 500M (Tiny 23M)**
- Séquences training: ~478,000 (1024 tokens chacune)
- Séquences validation: ~10,000
- Fichiers: `train.bin`, `val.bin`, `final_manifest.json`
- Taille: ~2GB compressé

### **Dataset Demo (100K)**
- Séquences training: ~190 (512 tokens chacune)
- Vocabulaire: 8,192 tokens
- Temps total: 10-15 minutes