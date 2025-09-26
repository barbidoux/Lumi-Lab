# 🚀 Procédure Complète : Entraînement Tokenizer + Dataset Tiny 23M Chinchilla

## 📋 Vue d'ensemble

Cette procédure vous guide pour :
1. **Entraîner un tokenizer SentencePiece** sur un mix représentatif de 4 sources (7M tokens)
2. **Générer un dataset complet** de 500M tokens pour un modèle 23M paramètres (ratio Chinchilla optimal)

## 🎯 Configuration Cible

- **Modèle** : 23M paramètres
- **Dataset** : 500M tokens (ratio Chinchilla 22:1)
- **Tokenizer** : SentencePiece 32K vocabulaire
- **Séquences** : 1024 tokens
- **Sources** : C4-EN, Gutenberg, FineWeb-Edu, VietGPT-Wikipedia (25% chacune)

---

## 📁 ÉTAPE 1 : Préparation des Répertoires

```bash
# Créer la structure de répertoires
mkdir -p data/tokenizer_corpus
mkdir -p data/full_dataset_corpus
mkdir -p data/tokenizer
mkdir -p data/packed
mkdir -p logs
```

---

## 🔤 ÉTAPE 2 : Entraînement du Tokenizer

### 2.1 Analyser les sources pour le tokenizer

```bash
python scripts/01_prepare_corpus.py \
  --config config/tokenizer_training_mix.yaml \
  --output-dir data/tokenizer_corpus \
  --analyze-only \
  --log-level INFO

# ✅ Vérifier les résultats dans data/tokenizer_corpus/plan.json
```

### 2.2 Préparer le corpus pour tokenizer (7M tokens)

```bash
python scripts/01_prepare_corpus.py \
  --config config/tokenizer_training_mix.yaml \
  --output-dir data/tokenizer_corpus \
  --shard-size 5000 \
  --log-level INFO

# ⏱️ Temps estimé: 15-30 minutes
# ✅ Vérifier: data/tokenizer_corpus/manifest.json et data/tokenizer_corpus/shards/
```

### 2.3 Entraîner le tokenizer SentencePiece

```bash
python scripts/02_train_tokenizer.py \
  --manifest data/tokenizer_corpus/manifest.json \
  --output-dir data/tokenizer \
  --vocab-size 32768 \
  --model-type unigram \
  --character-coverage 0.9995 \
  --max-training-sentences 2000000 \
  --log-level INFO

# ⏱️ Temps estimé: 10-20 minutes
# ✅ Vérifier: data/tokenizer/spm.model et data/tokenizer/tokenizer_config.json
```

---

## 📊 ÉTAPE 3 : Génération du Dataset Complet (500M tokens)

### 3.1 Analyser les sources complètes

```bash
python scripts/01_prepare_corpus.py \
  --config config/tiny_23M_full_dataset.yaml \
  --output-dir data/full_dataset_corpus \
  --analyze-only \
  --log-level INFO

# ✅ Vérifier le plan dans data/full_dataset_corpus/plan.json
# 🔍 Vérifier que chaque source planifie ~31M échantillons pour 125M tokens
```

### 3.2 Préparer le corpus complet

```bash
python scripts/01_prepare_corpus.py \
  --config config/tiny_23M_full_dataset.yaml \
  --output-dir data/full_dataset_corpus \
  --shard-size 10000 \
  --log-level INFO

# ⏱️ Temps estimé: 2-4 heures (500M tokens)
# ✅ Vérifier: data/full_dataset_corpus/manifest.json
# 📈 Surveiller les logs pour voir la progression par source
```

### 3.3 Tokeniser et packager les séquences

```bash
python scripts/03_pack_sequences.py \
  --manifest data/full_dataset_corpus/manifest.json \
  --tokenizer-dir data/tokenizer \
  --output-dir data/packed \
  --sequence-length 1024 \
  --train-val-split 0.98 \
  --log-level INFO

# ⏱️ Temps estimé: 30-60 minutes
# ✅ Vérifier: data/packed/final_manifest.json, train.bin, val.bin
```

---

## 🔍 ÉTAPE 4 : Validation et Vérification

### 4.1 Vérifier le tokenizer

```bash
# Le tokenizer a été testé automatiquement dans l'étape 2.3
# Vérifier les logs pour s'assurer que les tests ont réussi

# Test manuel optionnel:
python -c "
import sentencepiece as spm
sp = smp.SentencePieceProcessor()
sp.load('data/tokenizer/smp.model')
text = 'Hello world! This is a test.'
tokens = sp.encode(text)
decoded = sp.decode(tokens)
print(f'Original: {text}')
print(f'Tokens: {tokens}')
print(f'Decoded: {decoded}')
print(f'Vocab size: {sp.vocab_size()}')
"
```

### 4.2 Vérifier le dataset final

```bash
# Vérifier les statistiques finales
cat data/packed/final_manifest.json | jq '.statistics'

# Vérifications attendues:
# - total_tokens_used: ~500M
# - train_sequences: ~478K (500M / 1024 * 0.98)
# - val_sequences: ~10K (500M / 1024 * 0.02)
# - sequence_length: 1024
# - vocab_size: 32768
```

### 4.3 Test de chargement des données

```bash
python -c "
import numpy as np
import json

# Charger les infos
with open('data/packed/train.idx', 'r') as f:
    train_info = json.load(f)

# Charger les données via memmap
train_data = np.memmap(
    'data/packed/train.bin',
    dtype=np.uint16,
    mode='r',
    shape=tuple(train_info['shape'])
)

print(f'Dataset shape: {train_data.shape}')
print(f'Sample sequence: {train_data[0][:10].tolist()}')
print(f'Min token ID: {train_data.min()}')
print(f'Max token ID: {train_data.max()}')
"
```

---

## 📈 ÉTAPE 5 : Monitoring et Logs

### Fichiers de logs à surveiller :
- `logs/01_prepare_corpus.log` - Progression du traitement
- `logs/02_train_tokenizer.log` - Entraînement tokenizer
- `logs/03_pack_sequences.log` - Packaging des séquences

### Métriques clés à vérifier :
- **Tokenizer** : 7M tokens d'entraînement, 32K vocab
- **Corpus** : 500M tokens, ~488K séquences de 1024 tokens
- **Distribution** : 25% par source (125M tokens chacune)
- **Qualité** : Taux de déduplication, filtrage linguistique

---

## ✅ ÉTAPE 6 : Résultats Attendus

À la fin de cette procédure, vous devriez avoir :

```
data/
├── tokenizer/
│   ├── spm.model                    # Modèle SentencePiece
│   ├── spm.vocab                    # Vocabulaire
│   └── tokenizer_config.json        # Configuration
├── packed/
│   ├── train.bin                    # Données d'entraînement (478K séquences)
│   ├── train.idx                    # Index d'entraînement
│   ├── val.bin                      # Données de validation (10K séquences)
│   ├── val.idx                      # Index de validation
│   └── final_manifest.json          # Manifest final
└── [corpus directories avec shards compressés]
```

### Statistiques finales attendues :
- **Total tokens** : 500M (Chinchilla-optimal pour 23M paramètres)
- **Séquences d'entraînement** : ~478,000 (98%)
- **Séquences de validation** : ~10,000 (2%)
- **Taille sur disque** : ~2GB (format binaire compressé)

---

## 🚨 Dépannage

### Problèmes courants :

1. **Erreur d'authentification HF** :
   ```bash
   export HF_TOKEN="your_token_here"
   # ou utilisez huggingface-cli login
   ```

2. **Mémoire insuffisante** :
   - Réduire `shard_size` à 5000
   - Traiter les sources une par une

3. **Erreur de tokenizer** :
   - Vérifier que spm.model existe
   - Relancer l'étape 2.3 si nécessaire

4. **Données corrompues** :
   - Utiliser `--force` pour regénérer
   - Vérifier l'espace disque disponible

---

## 📞 Commandes de Diagnostic

```bash
# Vérifier l'espace disque
df -h

# Vérifier les processus Python
ps aux | grep python

# Taille des fichiers générés
du -sh data/*/

# Vérifier l'intégrité des manifests
python -m json.tool data/tokenizer_corpus/manifest.json
python -m json.tool data/full_dataset_corpus/manifest.json
python -m json.tool data/packed/final_manifest.json
```

Votre dataset sera prêt pour l'entraînement du modèle 23M paramètres ! 🎯