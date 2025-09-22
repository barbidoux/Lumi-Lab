# Audit du script SFT (scripts/03_sft.py)

## Problèmes identifiés

### 1. Tokenizer - Fallback incorrect (scripts/03_sft.py:134-135)
**Problème**: Fallback vers `microsoft/DialoGPT-medium` si `--tokenizer_path` non fourni
```python
# Use default tokenizer if not specified
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
```
**Impact**: Casse la continuité avec le prétrain SP32k et ses tokens spéciaux
**Solution**: Supprimer le fallback et échouer proprement si tokenizer SP32k non fourni

### 2. Absence d'eval_dataset (scripts/03_sft.py:197-205)
**Problème**: `eval_steps` configuré mais aucun `eval_dataset` fourni au SFTTrainer
```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Pas d'eval_dataset
    data_collator=collator,
    # ...
)
```
**Impact**: Aucune évaluation pendant l'entraînement malgré la configuration
**Solution**: Créer un split train/val et ajouter `eval_dataset`

### 3. ChatML sans EOS (scripts/03_sft.py:41)
**Problème**: Format ChatML ne se termine pas par EOS, contrairement aux autres templates
```python
formatted_text = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n{response}\n<|im_end|>"
```
**Impact**: Incohérence avec les autres formats qui ajoutent `<|endoftext|>`
**Solution**: Ajouter EOS à la fin de tous les templates

### 4. TrainingArguments non conformes (scripts/03_sft.py:156-178)
**Problèmes multiples**:
- `output_dir=config["output_dir"]` au lieu de `args.output_dir` (ligne 157)
- Scheduler par défaut (linéaire) au lieu de cosine
- `load_best_model_at_end=False` (ligne 177)
**Impact**: Configuration non respectée et optimisation sous-optimale
**Solution**: Utiliser `args.output_dir` et configurer cosine scheduler

### 5. Tokens spéciaux ChatML manquants
**Problème**: Aucune validation/ajout des tokens `<|im_start|>`, `<|im_end|>`
**Impact**: Tokens potentiellement non reconnus ou traités comme UNK
**Solution**: Ajouter ces tokens et redimensionner les embeddings

### 6. Packing désactivé (scripts/03_sft.py:204)
**Problème**: `packing=False` en dur
```python
packing=False,  # Disabled for better sequence control
```
**Impact**: Throughput réduit sur GPU
**Solution**: Rendre configurable via flag

### 7. Pas de validation SP32k
**Problème**: Aucune validation que le tokenizer chargé est bien SP32k avec les bons tokens spéciaux
**Impact**: Incompatibilité potentielle avec le prétrain
**Solution**: Valider `pad:0, unk:1, bos:2, eos:3`

### 8. Fonctionnalités manquantes
- Pas de support multi-datasets
- Pas de reprise depuis checkpoint
- Pas d'option de fusion LoRA
- Pas d'early stopping
- Pas de seed configuré

## Configuration recommandée

### Avant (problématique)
```bash
accelerate launch scripts/03_sft.py \
  --model_path checkpoints/pretrain/model \
  --dataset_path data/sft/dataset.jsonl \
  --prompt_template chatml
# → Utilise DialoGPT tokenizer si --tokenizer_path absent
# → Pas d'évaluation pendant training
# → ChatML sans EOS
```

### Après (corrigé)
```bash
accelerate launch --mixed_precision bf16 scripts/03_sft.py \
  --model_path checkpoints/mix-tiny-fa2-75k/tiny/final \
  --tokenizer_path data/tokenizer/spm32k.model \
  --dataset_paths data/sft/oasst1_en.jsonl data/sft/dolly15k.jsonl \
  --prompt_template chatml \
  --output_dir checkpoints/sft/tiny-chatml \
  --config_path config/sft_tiny.json \
  --use_lora --lora_r 16 --packing --resume_from_checkpoint
```

## Validation nécessaire
1. Chargement et validation tokenizer SP32k
2. Ajout eval_dataset avec early stopping
3. Correction formats pour EOS cohérent
4. Respect de tous les arguments CLI
5. Support multi-datasets et reprise training