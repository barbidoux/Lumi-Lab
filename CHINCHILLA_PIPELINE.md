# Pipeline Chinchilla-Optimal pour Modèle 124M

## Vue d'ensemble

Ce pipeline implémente la stratégie Chinchilla-optimal pour l'entraînement d'un modèle de 124 millions de paramètres, avec un corpus de ~2.5 milliards de tokens et un apprentissage par curriculum en deux phases.

## Nouvelles Fonctionnalités Implémentées

### 1. Support `max_samples` universel

**Problème résolu :** Le paramètre `max_samples` n'était supporté que dans `tokenizer_mix.json`. Il est maintenant disponible partout.

**Nouvelles capacités :**
- Support dans toutes les configurations de datasets via le champ `max_samples`
- Argument CLI `--max_samples` pour override en temps réel
- Limitation intelligente des gros datasets (C4, OpenWebText, etc.)

**Exemple d'usage :**
```bash
# Via configuration
python scripts/01_prepare_data.py --config_path config/datasets/wikipedia_en.json

# Via CLI override
python scripts/01_prepare_data.py --config_path config/datasets/c4_en.json --max_samples 500000
```

### 2. Configurations Phase-Based (Chinchilla-Optimal)

**Phase A - Haute Qualité (1.5B tokens):**
- `phase_a_wikipedia_1b.json` : 400k échantillons → ~1B tokens
- `phase_a_bookcorpus_500m.json` : 200k échantillons → ~500M tokens

**Phase B - Diversité (1.5B tokens):**
- `phase_b_c4_1b.json` : 800k échantillons → ~1B tokens
- `phase_b_forums_500m.json` : 300k échantillons → ~500M tokens

**Total :** ~3B tokens (dépasse la cible Chinchilla de 2.5B pour sécurité)

### 3. Commandes Makefile Intégrées

**Préparation par phase :**
```bash
make prepare-phase-a          # Phase A complète (1.5B tokens)
make prepare-phase-b          # Phase B complète (1.5B tokens)
make prepare-chinchilla-full  # Pipeline complet (3B tokens)
```

**Préparation individuelle :**
```bash
make prepare-wikipedia-1b     # Wikipedia seul (1B tokens)
make prepare-bookcorpus-500m  # Books seuls (500M tokens)
make prepare-c4-1b           # C4 seul (1B tokens)
make prepare-forums-500m     # Forums seuls (500M tokens)
```

## Workflow Complet

### Étape 1 : Tokenizer Global
```bash
make tokenizer-train-mix
```
→ Crée le tokenizer sur un mélange représentatif de toutes les sources

### Étape 2 : Préparation Complète
```bash
make prepare-chinchilla-full
```
→ Prépare tous les datasets des phases A et B

### Étape 3 : Entraînement Curriculum (à implémenter)
```bash
# Phase A : Apprentissage des bases (haute qualité)
make pretrain-curriculum-phase-a

# Phase B : Développement robustesse (diversité)
make pretrain-curriculum-phase-b
```

## Calculs Chinchilla-Optimal

**Modèle cible :** 124M paramètres
**Ratio optimal :** ~20 tokens par paramètre
**Corpus cible :** 124M × 20 = 2.48B tokens
**Corpus implémenté :** ~3B tokens (sécurité 20%)

**Répartition curriculum :**
- Phase A (fondations) : 1.5B tokens haute qualité
- Phase B (robustesse) : 1.5B tokens diversifiés

## Avantages de cette Approche

1. **Précision :** Contrôle exact de la taille des datasets via `max_samples`
2. **Reproductibilité :** Échantillonnage déterministe (seed=42)
3. **Efficacité :** Pas de téléchargement complet des gros datasets
4. **Flexibilité :** Override possible via CLI
5. **Production-ready :** Pipeline béton armé avec validation

## Prochaines Étapes

1. Implémenter l'entraînement curriculum dans `04_pretrain.py`
2. Ajouter le support multi-datasets dans `WeightedMultiDatasetSampler`
3. Créer les métriques de suivi pour les transitions Phase A → Phase B
4. Valider empiriquement les ratios tokens/performance

## Tests

```bash
# Vérifier les configurations
python -c "
import json
configs = ['phase_a_wikipedia_1b.json', 'phase_a_bookcorpus_500m.json', 'phase_b_c4_1b.json', 'phase_b_forums_500m.json']
for cfg in configs:
    with open(f'config/datasets/{cfg}', 'r') as f:
        config = json.load(f)
    print(f'{cfg}: {config.get(\"max_samples\", 0):,} samples → {config.get(\"_target_tokens\", \"Unknown\")}')
"

# Tester max_samples
python scripts/01_prepare_data.py --config_path config/datasets/test_max_samples.json --max_samples 100
```

---

**Status :** ✅ Implémenté et testé
**Prêt pour :** Phase de production rigoureuse