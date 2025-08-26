# Makefile pour l'entraînement d'un mini-LLM
# Commandes prêtes à l'emploi pour chaque phase du pipeline

# Variables par défaut
PYTHON := python
ACCELERATE := accelerate launch
DATA_DIR := ./data
CONFIG_DIR := ./config
SCRIPTS_DIR := ./scripts
CHECKPOINTS_DIR := ./checkpoints
EVALUATION_DIR := ./evaluation
SESSION_DIR := ./sessions

# Configuration des sessions
SESSION_TIME ?= auto
SESSION_NAME ?= $(shell date +%Y%m%d_%H%M%S)
SESSION_LOG := $(SESSION_DIR)/$(SESSION_NAME).log

# Datasets par défaut
RAW_DATASET := openwebtext
PROCESSED_DATA := $(DATA_DIR)/processed/tokenized_data.json
SFT_DATASET := $(DATA_DIR)/sft_dataset.json
DPO_DATASET := $(DATA_DIR)/dpo_dataset.json

# Configurations de modèle
TINY_CONFIG := $(CONFIG_DIR)/tiny.json
SMALL_CONFIG := $(CONFIG_DIR)/small.json
BASE_CONFIG := $(CONFIG_DIR)/base.json
SFT_CONFIG := $(CONFIG_DIR)/sft.json

# Aide
.PHONY: help
help:
	@echo "🤖 Makefile pour l'entraînement d'un mini-LLM"
	@echo ""
	@echo "📚 SESSIONS DE DÉVELOPPEMENT (courtes et focalisées):"
	@echo "  session-quick        - Session 30min: test rapide du pipeline"
	@echo "  session-prototype    - Session 2h: prototype avec tiny model"
	@echo "  session-experiment   - Session 4h: expérimentation avec small model"
	@echo "  session-evaluation   - Session 1h: évaluation et analyse"
	@echo "  session-debug        - Session interactive de debug"
	@echo "  session-architecture - Validation de l'architecture"
	@echo ""
	@echo "🚀 COMMANDES PRINCIPALES:"
	@echo "  install              - Installe les dépendances"
	@echo "  prepare              - Prépare les données d'entraînement"
	@echo "  pretrain-tiny        - Lance le pré-entraînement du modèle tiny"
	@echo "  pretrain-small       - Lance le pré-entraînement du modèle small"
	@echo "  pretrain-base        - Lance le pré-entraînement du modèle base"
	@echo "  sft                  - Lance le fine-tuning supervisé"
	@echo "  dpo                  - Lance l'alignement DPO"
	@echo ""
	@echo "📊 ÉVALUATION ET ANALYSE:"
	@echo "  evaluate             - Évaluation complète"
	@echo "  evaluate-quick       - Évaluation rapide pour développement"
	@echo "  assess-performance   - Analyse automatique des performances"
	@echo "  validate-architecture - Validation de la configuration"
	@echo ""
	@echo "🎯 INFÉRENCE ET SERVICES:"
	@echo "  serve                - Lance l'interface interactive"
	@echo "  serve-api            - Lance le serveur API"
	@echo ""
	@echo "🔧 MAINTENANCE:"
	@echo "  clean                - Nettoie les fichiers temporaires"
	@echo "  clean-checkpoints    - Supprime tous les checkpoints"
	@echo "  backup               - Sauvegarde configs et modèles"
	@echo "  monitor              - Surveillance des ressources"
	@echo ""
	@echo "⚙️ Variables configurables:"
	@echo "  RAW_DATASET         - Dataset brut à utiliser (défaut: openwebtext)"
	@echo "  SFT_DATASET         - Dataset pour SFT (défaut: $(SFT_DATASET))"
	@echo "  DPO_DATASET         - Dataset pour DPO (défaut: $(DPO_DATASET))
  MODEL_PATH          - Chemin du modèle pour évaluation
  CONFIG              - Configuration à valider"
	@echo "  SESSION_TIME        - Temps de session en minutes (défaut: auto)"

# Installation des dépendances
.PHONY: install
install:
	@echo "Installation des dépendances..."
	pip install -r requirements.txt
	@echo "Dépendances installées avec succès!"

# Préparation des données
.PHONY: prepare
prepare:
	@echo "Préparation des données d'entraînement..."
	@mkdir -p $(DATA_DIR)/processed
	$(PYTHON) $(SCRIPTS_DIR)/01_prepare_data.py \
		--input_path $(RAW_DATASET) \
		--output_dir $(DATA_DIR)/processed \
		--vocab_size 32768 \
		--min_length 50 \
		--max_length 10000
	@echo "Données préparées dans $(DATA_DIR)/processed"

# Pré-entraînement modèle tiny
.PHONY: pretrain-tiny
pretrain-tiny:
	@echo "Lancement du pré-entraînement du modèle tiny..."
	@mkdir -p $(CHECKPOINTS_DIR)/pretrain/tiny
	$(ACCELERATE) $(SCRIPTS_DIR)/02_pretrain.py \
		--config $(TINY_CONFIG) \
		--data_path $(PROCESSED_DATA) \
		--output_dir $(CHECKPOINTS_DIR)/pretrain/tiny \
		--learning_rate 3e-4 \
		--batch_size 16 \
		--gradient_accumulation_steps 4 \
		--num_train_epochs 1 \
		--warmup_steps 1000 \
		--save_steps 2000 \
		--logging_steps 50
	@echo "Pré-entraînement tiny terminé!"

# Pré-entraînement modèle small
.PHONY: pretrain-small
pretrain-small:
	@echo "Lancement du pré-entraînement du modèle small..."
	@mkdir -p $(CHECKPOINTS_DIR)/pretrain/small
	$(ACCELERATE) $(SCRIPTS_DIR)/02_pretrain.py \
		--config $(SMALL_CONFIG) \
		--data_path $(PROCESSED_DATA) \
		--output_dir $(CHECKPOINTS_DIR)/pretrain/small \
		--learning_rate 3e-4 \
		--batch_size 8 \
		--gradient_accumulation_steps 8 \
		--num_train_epochs 1 \
		--warmup_steps 2000 \
		--save_steps 5000 \
		--logging_steps 100
	@echo "Pré-entraînement small terminé!"

# Pré-entraînement modèle base
.PHONY: pretrain-base
pretrain-base:
	@echo "Lancement du pré-entraînement du modèle base..."
	@mkdir -p $(CHECKPOINTS_DIR)/pretrain/base
	$(ACCELERATE) $(SCRIPTS_DIR)/02_pretrain.py \
		--config $(BASE_CONFIG) \
		--data_path $(PROCESSED_DATA) \
		--output_dir $(CHECKPOINTS_DIR)/pretrain/base \
		--learning_rate 2e-4 \
		--batch_size 4 \
		--gradient_accumulation_steps 16 \
		--num_train_epochs 1 \
		--warmup_steps 4000 \
		--save_steps 10000 \
		--logging_steps 200
	@echo "Pré-entraînement base terminé!"

# Fine-tuning supervisé
.PHONY: sft
sft:
	@echo "Lancement du fine-tuning supervisé..."
	@mkdir -p $(CHECKPOINTS_DIR)/sft
	$(PYTHON) $(SCRIPTS_DIR)/03_sft.py \
		--model_path $(CHECKPOINTS_DIR)/pretrain/tiny/final \
		--dataset_path $(SFT_DATASET) \
		--config_path $(SFT_CONFIG) \
		--output_dir $(CHECKPOINTS_DIR)/sft
	@echo "Fine-tuning supervisé terminé!"

# Alignement DPO
.PHONY: dpo
dpo:
	@echo "Lancement de l'alignement DPO..."
	@mkdir -p $(CHECKPOINTS_DIR)/dpo
	$(PYTHON) $(SCRIPTS_DIR)/04_dpo.py \
		--model_path $(CHECKPOINTS_DIR)/sft \
		--dataset_path $(DPO_DATASET) \
		--output_dir $(CHECKPOINTS_DIR)/dpo \
		--beta 0.1 \
		--learning_rate 5e-7 \
		--batch_size 2 \
		--gradient_accumulation_steps 8 \
		--num_train_epochs 1 \
		--max_length 1024
	@echo "Alignement DPO terminé!"

# Évaluation
.PHONY: evaluate
evaluate:
	@echo "Évaluation du modèle final..."
	@mkdir -p ./evaluation_results
	$(PYTHON) $(SCRIPTS_DIR)/05_evaluate.py \
		--model_path $(CHECKPOINTS_DIR)/dpo \
		--output_dir ./evaluation_results \
		--max_boolq_samples 100
	@echo "Évaluation terminée! Résultats dans ./evaluation_results"

# Inférence interactive
.PHONY: serve
serve:
	@echo "Lancement du mode interactif..."
	$(PYTHON) $(SCRIPTS_DIR)/06_serve.py \
		--model_path $(CHECKPOINTS_DIR)/dpo \
		--mode interactive \
		--template chatml \
		--temperature 0.7 \
		--max_new_tokens 150

# Serveur API
.PHONY: serve-api
serve-api:
	@echo "Lancement du serveur API..."
	$(PYTHON) $(SCRIPTS_DIR)/06_serve.py \
		--model_path $(CHECKPOINTS_DIR)/dpo \
		--mode api \
		--host 127.0.0.1 \
		--port 8000

# Pipeline complet pour modèle tiny
.PHONY: pipeline-tiny
pipeline-tiny: prepare pretrain-tiny
	@echo "Pipeline complet tiny terminé!"

# Pipeline complet avec fine-tuning (nécessite les datasets SFT et DPO)
.PHONY: pipeline-full
pipeline-full: prepare pretrain-tiny sft dpo evaluate serve
	@echo "Pipeline complet terminé! Modèle prêt à l'usage."

# Commandes de reprise depuis checkpoint
.PHONY: resume-pretrain-tiny
resume-pretrain-tiny:
	@echo "Reprise du pré-entraînement tiny depuis le dernier checkpoint..."
	$(ACCELERATE) $(SCRIPTS_DIR)/02_pretrain.py \
		--config $(TINY_CONFIG) \
		--data_path $(PROCESSED_DATA) \
		--output_dir $(CHECKPOINTS_DIR)/pretrain/tiny \
		--resume_from_checkpoint $(CHECKPOINTS_DIR)/pretrain/tiny/step_* \
		--learning_rate 3e-4 \
		--batch_size 16 \
		--gradient_accumulation_steps 4

# Test rapide avec données synthétiques
.PHONY: test-pipeline
test-pipeline:
	@echo "Test du pipeline avec données synthétiques..."
	@mkdir -p $(DATA_DIR)/test
	@echo '[[1, 2, 3, 4, 5]]' > $(DATA_DIR)/test/tokenized_data.json
	$(PYTHON) $(SCRIPTS_DIR)/02_pretrain.py \
		--config $(TINY_CONFIG) \
		--data_path $(DATA_DIR)/test/tokenized_data.json \
		--output_dir $(CHECKPOINTS_DIR)/test \
		--max_steps 10 \
		--logging_steps 5
	@echo "Test terminé!"

# Génération d'exemples de datasets
.PHONY: create-sample-datasets
create-sample-datasets:
	@echo "Création d'exemples de datasets..."
	@mkdir -p $(DATA_DIR)
	@echo '[' > $(SFT_DATASET)
	@echo '  {"prompt": "Qu'\''est-ce que l'\''IA ?", "response": "L'\''intelligence artificielle est..."},' >> $(SFT_DATASET)
	@echo '  {"prompt": "Comment ça marche ?", "response": "Cela fonctionne grâce à..."}' >> $(SFT_DATASET)
	@echo ']' >> $(SFT_DATASET)
	
	@echo '[' > $(DPO_DATASET)
	@echo '  {"prompt": "Explique l'\''IA", "chosen": "L'\''IA est une technologie fascinante...", "rejected": "Je sais pas."},' >> $(DPO_DATASET)
	@echo '  {"prompt": "Comment apprendre ?", "chosen": "Il faut étudier régulièrement...", "rejected": "C'\''est facile."}' >> $(DPO_DATASET)
	@echo ']' >> $(DPO_DATASET)
	@echo "Datasets d'exemple créés!"

# Monitoring des ressources pendant l'entraînement
.PHONY: monitor
monitor:
	@echo "Surveillance des ressources système..."
	watch -n 2 'nvidia-smi | head -15; echo ""; ps aux | grep python | head -5; echo ""; df -h | head -5'

# Nettoyage
.PHONY: clean
clean:
	@echo "Nettoyage des fichiers temporaires..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Nettoyage terminé!"

.PHONY: clean-checkpoints
clean-checkpoints:
	@echo "Suppression des checkpoints..."
	rm -rf $(CHECKPOINTS_DIR)
	@echo "Checkpoints supprimés!"

# Vérification de l'environnement
.PHONY: check-env
check-env:
	@echo "Vérification de l'environnement..."
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
	@$(PYTHON) -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	@$(PYTHON) -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
	@echo "Vérification terminée!"

# Configuration pour différentes tailles de GPU
.PHONY: config-rtx4090
config-rtx4090:
	@echo "Configuration optimisée pour RTX 4090 (16 GB)..."
	@echo "Utilisation des paramètres recommandés pour votre GPU"

# Sauvegarde des configs et modèles
.PHONY: backup
backup:
	@echo "Sauvegarde des configurations et checkpoints importants..."
	@mkdir -p ./backups/$(shell date +%Y%m%d_%H%M%S)
	cp -r $(CONFIG_DIR) ./backups/$(shell date +%Y%m%d_%H%M%S)/
	@if [ -d "$(CHECKPOINTS_DIR)/pretrain/tiny/final" ]; then \
		cp -r $(CHECKPOINTS_DIR)/pretrain/tiny/final ./backups/$(shell date +%Y%m%d_%H%M%S)/model_final; \
	fi
	@echo "Sauvegarde terminée dans ./backups/"

# ============================================================================
# SESSIONS DE DÉVELOPPEMENT FOCALISÉES
# ============================================================================

# Session rapide (30 minutes) - Test et validation
.PHONY: session-quick
session-quick:
	@echo "🚀 SESSION RAPIDE (30 min) - Test pipeline"
	@echo "Début: $(shell date)"
	@mkdir -p $(SESSION_DIR)
	@echo "=== SESSION QUICK $(shell date) ===" > $(SESSION_LOG)
	$(MAKE) check-env 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) create-sample-datasets 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) test-pipeline 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) evaluate-quick 2>&1 | tee -a $(SESSION_LOG)
	@echo "🎉 Session rapide terminée! Log: $(SESSION_LOG)"
	@echo "Durée estimée: 30 minutes"

# Session prototype (2 heures) - Développement tiny model
.PHONY: session-prototype  
session-prototype:
	@echo "🛠️ SESSION PROTOTYPE (2h) - Tiny model complet"
	@echo "Début: $(shell date)"
	@mkdir -p $(SESSION_DIR)
	@echo "=== SESSION PROTOTYPE $(shell date) ===" > $(SESSION_LOG)
	$(MAKE) check-env 2>&1 | tee -a $(SESSION_LOG)
	@if [ ! -f "$(PROCESSED_DATA)" ]; then $(MAKE) prepare-quick 2>&1 | tee -a $(SESSION_LOG); fi
	$(MAKE) pretrain-tiny-quick 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) evaluate-quick 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) validate-architecture 2>&1 | tee -a $(SESSION_LOG)
	@echo "🎉 Session prototype terminée! Log: $(SESSION_LOG)"
	@echo "Modèle disponible: $(CHECKPOINTS_DIR)/pretrain/tiny/final"

# Session expérimentation (4 heures) - Small model avec fine-tuning
.PHONY: session-experiment
session-experiment:
	@echo "🧪 SESSION EXPÉRIMENTATION (4h) - Small model + SFT"
	@echo "Début: $(shell date)"
	@mkdir -p $(SESSION_DIR)
	@echo "=== SESSION EXPERIMENT $(shell date) ===" > $(SESSION_LOG)
	$(MAKE) check-env 2>&1 | tee -a $(SESSION_LOG)
	@if [ ! -f "$(PROCESSED_DATA)" ]; then $(MAKE) prepare 2>&1 | tee -a $(SESSION_LOG); fi
	$(MAKE) pretrain-small 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) create-sample-datasets 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) sft-small 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) evaluate 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) assess-performance 2>&1 | tee -a $(SESSION_LOG)
	@echo "🎉 Session expérimentation terminée! Log: $(SESSION_LOG)"
	@echo "Modèle disponible: $(CHECKPOINTS_DIR)/sft"

# Session évaluation (1 heure) - Analyse approfondie
.PHONY: session-evaluation
session-evaluation:
	@echo "📊 SESSION ÉVALUATION (1h) - Analyse complète"
	@echo "Début: $(shell date)"
	@mkdir -p $(SESSION_DIR)
	@echo "=== SESSION EVALUATION $(shell date) ===" > $(SESSION_LOG)
	@echo "Recherche du dernier modèle entraîné..."
	@if [ -d "$(CHECKPOINTS_DIR)/dpo" ]; then \
		echo "Évaluation du modèle DPO" | tee -a $(SESSION_LOG); \
		$(MAKE) MODEL_PATH=$(CHECKPOINTS_DIR)/dpo evaluate-detailed 2>&1 | tee -a $(SESSION_LOG); \
	elif [ -d "$(CHECKPOINTS_DIR)/sft" ]; then \
		echo "Évaluation du modèle SFT" | tee -a $(SESSION_LOG); \
		$(MAKE) MODEL_PATH=$(CHECKPOINTS_DIR)/sft evaluate-detailed 2>&1 | tee -a $(SESSION_LOG); \
	elif [ -d "$(CHECKPOINTS_DIR)/pretrain/tiny/final" ]; then \
		echo "Évaluation du modèle tiny" | tee -a $(SESSION_LOG); \
		$(MAKE) MODEL_PATH=$(CHECKPOINTS_DIR)/pretrain/tiny/final evaluate-detailed 2>&1 | tee -a $(SESSION_LOG); \
	else \
		echo "❌ Aucun modèle trouvé pour évaluation" | tee -a $(SESSION_LOG); \
	fi
	$(MAKE) assess-performance 2>&1 | tee -a $(SESSION_LOG) || true
	@echo "🎉 Session évaluation terminée! Log: $(SESSION_LOG)"

# Session debug interactive
.PHONY: session-debug
session-debug:
	@echo "🔧 SESSION DEBUG - Mode interactif"
	@echo "=== OPTIONS DISPONIBLES ==="
	@echo "1. Vérifier l'environnement: make check-env"
	@echo "2. Tester avec données synthétiques: make test-pipeline"
	@echo "3. Valider une architecture: make validate-architecture"
	@echo "4. Évaluation rapide: make evaluate-quick"
	@echo "5. Surveillance ressources: make monitor"
	@echo "6. Nettoyer et redémarrer: make clean"
	@echo "7. Lister les checkpoints: ls -la $(CHECKPOINTS_DIR)/*/"
	@echo ""
	@echo "📝 Pour des logs détaillés, ajoutez 2>&1 | tee debug.log"
	@echo "Exemple: make check-env 2>&1 | tee debug.log"

# Session validation architecture
.PHONY: session-architecture
session-architecture:
	@echo "🏗️ SESSION ARCHITECTURE - Validation des configurations"
	@mkdir -p $(SESSION_DIR)
	@echo "=== SESSION ARCHITECTURE $(shell date) ===" > $(SESSION_LOG)
	@echo "Validation de toutes les configurations..."
	@for config in $(CONFIG_DIR)/*.json; do \
		echo "Validation: $$config" | tee -a $(SESSION_LOG); \
		$(PYTHON) utils/validate_architecture.py "$$config" 2>&1 | tee -a $(SESSION_LOG); \
		echo "" | tee -a $(SESSION_LOG); \
	done
	@echo "🎉 Validation architecturale terminée! Log: $(SESSION_LOG)"

# ============================================================================
# COMMANDES DE SUPPORT POUR LES SESSIONS
# ============================================================================

# Préparation rapide des données (dataset plus petit)
.PHONY: prepare-quick
prepare-quick:
	@echo "Préparation rapide des données..."
	@mkdir -p $(DATA_DIR)/processed
	$(PYTHON) $(SCRIPTS_DIR)/01_prepare_data.py \
		--input_path "wikitext-2-raw-v1" \
		--output_dir $(DATA_DIR)/processed \
		--vocab_size 32768 \
		--min_length 50 \
		--max_length 1000
	@echo "Données rapides préparées!"

# Pré-entraînement tiny rapide (moins d'epochs)
.PHONY: pretrain-tiny-quick
pretrain-tiny-quick:
	@echo "Pré-entraînement tiny rapide (version courte)..."
	@mkdir -p $(CHECKPOINTS_DIR)/pretrain/tiny
	$(ACCELERATE) $(SCRIPTS_DIR)/02_pretrain.py \
		--config $(TINY_CONFIG) \
		--data_path $(PROCESSED_DATA) \
		--output_dir $(CHECKPOINTS_DIR)/pretrain/tiny \
		--learning_rate 5e-4 \
		--batch_size 16 \
		--gradient_accumulation_steps 4 \
		--max_steps 1000 \
		--warmup_steps 100 \
		--save_steps 500 \
		--logging_steps 50
	@echo "Pré-entraînement tiny rapide terminé!"

# SFT pour small model
.PHONY: sft-small
sft-small:
	@echo "Fine-tuning supervisé pour modèle small..."
	@mkdir -p $(CHECKPOINTS_DIR)/sft
	$(PYTHON) $(SCRIPTS_DIR)/03_sft.py \
		--model_path $(CHECKPOINTS_DIR)/pretrain/small/final \
		--dataset_path $(SFT_DATASET) \
		--config_path $(SFT_CONFIG) \
		--output_dir $(CHECKPOINTS_DIR)/sft
	@echo "Fine-tuning supervisé (small) terminé!"

# Évaluation rapide pour développement
.PHONY: evaluate-quick
evaluate-quick:
	@echo "Évaluation rapide..."
	@mkdir -p ./evaluation_results
	$(PYTHON) $(SCRIPTS_DIR)/05_evaluate.py \
		--model_path $(or $(MODEL_PATH),$(CHECKPOINTS_DIR)/pretrain/tiny/final) \
		--output_dir ./evaluation_results \
		--fast_mode \
		--max_boolq_samples 20
	@echo "Évaluation rapide terminée!"

# Évaluation détaillée avec rapport
.PHONY: evaluate-detailed
evaluate-detailed:
	@echo "Évaluation détaillée avec rapport..."
	@mkdir -p ./evaluation_results
	$(PYTHON) $(SCRIPTS_DIR)/05_evaluate.py \
		--model_path $(or $(MODEL_PATH),$(CHECKPOINTS_DIR)/dpo) \
		--output_dir ./evaluation_results \
		--detailed_output \
		--max_boolq_samples 100
	@echo "Évaluation détaillée terminée!"

# Analyse automatique des performances
.PHONY: assess-performance
assess-performance:
	@echo "Analyse des performances..."
	@if [ -f "./evaluation_results/evaluation_results.json" ]; then \
		$(PYTHON) $(EVALUATION_DIR)/assess_performance.py ./evaluation_results/evaluation_results.json; \
	else \
		echo "❌ Fichier d'évaluation non trouvé. Lancez 'make evaluate' d'abord."; \
	fi

# Validation de l'architecture
.PHONY: validate-architecture
validate-architecture:
	@echo "Validation de l'architecture..."
	@if [ -n "$(CONFIG)" ]; then \
		$(PYTHON) utils/validate_architecture.py $(CONFIG); \
	else \
		$(PYTHON) utils/validate_architecture.py $(TINY_CONFIG); \
	fi

# Statut de session - montre l'état actuel
.PHONY: session-status
session-status:
	@echo "📊 STATUT DE LA SESSION"
	@echo "========================================"
	@echo "Heure actuelle: $(shell date)"
	@echo "Dossier de travail: $(PWD)"
	@echo ""
	@echo "📁 Données disponibles:"
	@if [ -f "$(PROCESSED_DATA)" ]; then echo "  ✅ Données préparées"; else echo "  ❌ Données non préparées (make prepare)"; fi
	@echo ""
	@echo "🤖 Modèles disponibles:"
	@if [ -d "$(CHECKPOINTS_DIR)/pretrain/tiny/final" ]; then echo "  ✅ Tiny model"; else echo "  ❌ Tiny model non entraîné"; fi
	@if [ -d "$(CHECKPOINTS_DIR)/pretrain/small/final" ]; then echo "  ✅ Small model"; else echo "  ❌ Small model non entraîné"; fi
	@if [ -d "$(CHECKPOINTS_DIR)/pretrain/base/final" ]; then echo "  ✅ Base model"; else echo "  ❌ Base model non entraîné"; fi
	@if [ -d "$(CHECKPOINTS_DIR)/sft" ]; then echo "  ✅ Modèle SFT"; else echo "  ❌ Modèle SFT non entraîné"; fi
	@if [ -d "$(CHECKPOINTS_DIR)/dpo" ]; then echo "  ✅ Modèle DPO"; else echo "  ❌ Modèle DPO non entraîné"; fi
	@echo ""
	@echo "📊 Évaluations:"
	@if [ -f "./evaluation_results/evaluation_results.json" ]; then echo "  ✅ Résultats d'évaluation disponibles"; else echo "  ❌ Pas d'évaluation récente"; fi
	@echo ""
	@echo "💾 Espace disque:"
	@du -sh $(CHECKPOINTS_DIR) 2>/dev/null || echo "  Pas de checkpoints"
	@echo ""
	@echo "🔥 GPU Status:"
	@nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "  GPU non disponible"

# Nettoyage de session (garde les modèles importants)
.PHONY: session-cleanup
session-cleanup:
	@echo "🧹 Nettoyage de session..."
	@$(MAKE) clean
	@rm -rf $(DATA_DIR)/test
	@find $(SESSION_DIR) -name "*.log" -mtime +7 -delete 2>/dev/null || true
	@echo "Nettoyage terminé (modèles conservés)!"