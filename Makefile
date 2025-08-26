# Makefile pour l'entraînement d'un mini-LLM
# Commandes prêtes à l'emploi pour chaque phase du pipeline

# Variables par défaut
PYTHON := python
ACCELERATE := accelerate launch
DATA_DIR := ./data
CONFIG_DIR := ./config
SCRIPTS_DIR := ./scripts
CHECKPOINTS_DIR := ./checkpoints

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
	@echo "Makefile pour l'entraînement d'un mini-LLM"
	@echo ""
	@echo "Commandes disponibles:"
	@echo "  help                 - Affiche cette aide"
	@echo "  install              - Installe les dépendances"
	@echo "  prepare              - Prépare les données d'entraînement"
	@echo "  pretrain-tiny        - Lance le pré-entraînement du modèle tiny"
	@echo "  pretrain-small       - Lance le pré-entraînement du modèle small"
	@echo "  pretrain-base        - Lance le pré-entraînement du modèle base"
	@echo "  sft                  - Lance le fine-tuning supervisé"
	@echo "  dpo                  - Lance l'alignement DPO"
	@echo "  evaluate             - Évalue le modèle final"
	@echo "  serve                - Lance l'interface interactive"
	@echo "  serve-api            - Lance le serveur API"
	@echo "  clean                - Nettoie les fichiers temporaires"
	@echo "  clean-checkpoints    - Supprime tous les checkpoints"
	@echo ""
	@echo "Variables configurables:"
	@echo "  RAW_DATASET         - Dataset brut à utiliser (défaut: openwebtext)"
	@echo "  SFT_DATASET         - Dataset pour SFT (défaut: $(SFT_DATASET))"
	@echo "  DPO_DATASET         - Dataset pour DPO (défaut: $(DPO_DATASET))"

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