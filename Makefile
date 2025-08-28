# Makefile for mini-LLM training
# Ready-to-use commands for each pipeline phase

# Default variables
PYTHON := python
ACCELERATE := accelerate launch
DATA_DIR := ./data
CONFIG_DIR := ./config
SCRIPTS_DIR := ./scripts
CHECKPOINTS_DIR := ./checkpoints
EVALUATION_DIR := ./evaluation
SESSION_DIR := ./sessions

# Session configuration
SESSION_TIME ?= auto
SESSION_NAME ?= $(shell date +%Y%m%d_%H%M%S)
SESSION_LOG := $(SESSION_DIR)/$(SESSION_NAME).log

# Default datasets
RAW_DATASET := openwebtext
PROCESSED_DATA := $(DATA_DIR)/processed/tokenized_data.json
SFT_DATASET := $(DATA_DIR)/sft_dataset.json
DPO_DATASET := $(DATA_DIR)/dpo_dataset.json

# Model configurations
TINY_CONFIG := $(CONFIG_DIR)/tiny.json
SMALL_CONFIG := $(CONFIG_DIR)/small.json
BASE_CONFIG := $(CONFIG_DIR)/base.json
SFT_CONFIG := $(CONFIG_DIR)/sft.json

# Help
.PHONY: help
help:
	@echo "🤖 Makefile for mini-LLM training"
	@echo ""
	@echo "📚 DEVELOPMENT SESSIONS (short and focused):"
	@echo "  session-quick        - 30min session: rapid pipeline testing"
	@echo "  session-prototype    - 2h session: prototype with tiny model"
	@echo "  session-experiment   - 4h session: experimentation with small model"
	@echo "  session-evaluation   - 1h session: evaluation and analysis"
	@echo "  session-debug        - Interactive debugging session"
	@echo "  session-architecture - Architecture validation"
	@echo ""
	@echo "🚀 MAIN COMMANDS:"
	@echo "  install              - Install dependencies"
	@echo "  prepare              - Prepare dataset from CONFIG file"
	@echo "  prepare-owt          - Prepare OpenWebText (recommended)"
	@echo "  prepare-wiki         - Prepare Wikipedia EN"
	@echo "  prepare-wt103        - Prepare WikiText-103 (fast)"
	@echo "  prepare-demo         - Prepare demo tiny dataset (very fast)"
	@echo "  pretrain-tiny        - Launch tiny model pre-training"
	@echo "  pretrain-small       - Launch small model pre-training"
	@echo "  pretrain-base        - Launch base model pre-training"
	@echo "  sft                  - Launch supervised fine-tuning"
	@echo "  dpo                  - Launch DPO alignment"
	@echo ""
	@echo "📊 EVALUATION AND ANALYSIS:"
	@echo "  evaluate             - Complete evaluation"
	@echo "  evaluate-quick       - Quick evaluation for development"
	@echo "  assess-performance   - Automatic performance analysis"
	@echo "  validate-architecture - Configuration validation"
	@echo ""
	@echo "🎯 INFERENCE AND SERVICES:"
	@echo "  serve                - Launch interactive interface"
	@echo "  serve-api            - Launch API server"
	@echo ""
	@echo "🔧 MAINTENANCE:"
	@echo "  clean                - Clean temporary files"
	@echo "  clean-data           - Remove processed datasets"
	@echo "  clean-checkpoints    - Remove all checkpoints"
	@echo "  clean-all            - Complete cleanup (temp + data + checkpoints)"
	@echo "  clean-repo           - Repository cleanup for commits (keeps configs)"
	@echo "  clean-status         - Show what would be cleaned"
	@echo "  backup               - Backup configs and models"
	@echo "  monitor              - Resource monitoring"
	@echo ""
	@echo "⚙️ Configurable variables:"
	@echo "  CONFIG              - Dataset config file (required for prepare)"
	@echo "  SFT_DATASET         - Dataset for SFT (default: $(SFT_DATASET))"
	@echo "  DPO_DATASET         - Dataset for DPO (default: $(DPO_DATASET))"
	@echo "  MODEL_PATH          - Model path for evaluation"
	@echo "  SESSION_TIME        - Session time in minutes (default: auto)"
	@echo ""
	@echo "📁 Available dataset configs:"
	@echo "  config/datasets/openwebtext.json   - OpenWebText (large, production)"
	@echo "  config/datasets/wikipedia_en.json  - Wikipedia EN (medium)"
	@echo "  config/datasets/wikitext103.json   - WikiText-103 (small, fast)"
	@echo "  config/datasets/demo_tiny.json     - Demo dataset (tiny, very fast)"

# Dependencies installation
.PHONY: install
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Data preparation from config files
.PHONY: prepare
prepare:
	@echo "Preparing dataset from config: $(CONFIG)"
	$(PYTHON) $(SCRIPTS_DIR)/01_prepare_data.py --config_path $(CONFIG)

# OpenWebText preparation (recommended for production)
.PHONY: prepare-owt
prepare-owt:
	@echo "Preparing OpenWebText dataset..."
	$(MAKE) prepare CONFIG=config/datasets/openwebtext.json

# Wikipedia EN preparation
.PHONY: prepare-wiki
prepare-wiki:
	@echo "Preparing Wikipedia EN dataset..."
	$(MAKE) prepare CONFIG=config/datasets/wikipedia_en.json

# WikiText-103 preparation (fast for testing)
.PHONY: prepare-wt103
prepare-wt103:
	@echo "Preparing WikiText-103 dataset..."
	$(MAKE) prepare CONFIG=config/datasets/wikitext103.json

# Demo tiny preparation (very fast for development)
.PHONY: prepare-demo
prepare-demo:
	@echo "Preparing demo tiny dataset..."
	$(MAKE) prepare CONFIG=config/datasets/demo_tiny.json

# Legacy preparation (backward compatibility)
.PHONY: prepare-legacy
prepare-legacy:
	@echo "Preparing training data (legacy mode)..."
	@mkdir -p $(DATA_DIR)/processed
	$(PYTHON) $(SCRIPTS_DIR)/01_prepare_data.py \
		--input_path $(RAW_DATASET) \
		--output_dir $(DATA_DIR)/processed \
		--vocab_size 32768 \
		--min_length 50 \
		--max_length 10000
	@echo "Data prepared in $(DATA_DIR)/processed"

# Tiny model pre-training
.PHONY: pretrain-tiny
pretrain-tiny:
	@echo "Launching tiny model pre-training..."
	@mkdir -p $(CHECKPOINTS_DIR)/pretrain/tiny
	$(ACCELERATE) $(SCRIPTS_DIR)/02_pretrain.py \
		--config $(TINY_CONFIG) \
		--data_dir $(DATA_DIR)/processed \
		--output_dir $(CHECKPOINTS_DIR)/pretrain/tiny \
		--learning_rate 3e-4 \
		--batch_size 16 \
		--gradient_accumulation_steps 4 \
		--num_train_epochs 1 \
		--warmup_steps 1000 \
		--save_steps 2000 \
		--logging_steps 50
	@echo "Tiny pre-training completed!"

# Small model pre-training
.PHONY: pretrain-small
pretrain-small:
	@echo "Launching small model pre-training..."
	@mkdir -p $(CHECKPOINTS_DIR)/pretrain/small
	$(ACCELERATE) $(SCRIPTS_DIR)/02_pretrain.py \
		--config $(SMALL_CONFIG) \
		--data_dir $(DATA_DIR)/processed \
		--output_dir $(CHECKPOINTS_DIR)/pretrain/small \
		--learning_rate 3e-4 \
		--batch_size 8 \
		--gradient_accumulation_steps 8 \
		--num_train_epochs 1 \
		--warmup_steps 2000 \
		--save_steps 5000 \
		--logging_steps 100
	@echo "Small pre-training completed!"

# Base model pre-training
.PHONY: pretrain-base
pretrain-base:
	@echo "Launching base model pre-training..."
	@mkdir -p $(CHECKPOINTS_DIR)/pretrain/base
	$(ACCELERATE) $(SCRIPTS_DIR)/02_pretrain.py \
		--config $(BASE_CONFIG) \
		--data_dir $(DATA_DIR)/processed \
		--output_dir $(CHECKPOINTS_DIR)/pretrain/base \
		--learning_rate 2e-4 \
		--batch_size 4 \
		--gradient_accumulation_steps 16 \
		--num_train_epochs 1 \
		--warmup_steps 4000 \
		--save_steps 10000 \
		--logging_steps 200
	@echo "Base pre-training completed!"

# Supervised fine-tuning
.PHONY: sft
sft:
	@echo "Launching supervised fine-tuning..."
	@mkdir -p $(CHECKPOINTS_DIR)/sft
	$(PYTHON) $(SCRIPTS_DIR)/03_sft.py \
		--model_path $(CHECKPOINTS_DIR)/pretrain/tiny/final \
		--dataset_path $(SFT_DATASET) \
		--config_path $(SFT_CONFIG) \
		--output_dir $(CHECKPOINTS_DIR)/sft
	@echo "Supervised fine-tuning completed!"

# DPO alignment
.PHONY: dpo
dpo:
	@echo "Launching DPO alignment..."
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
	@echo "DPO alignment completed!"

# Evaluation
.PHONY: evaluate
evaluate:
	@echo "Evaluating final model..."
	@mkdir -p ./evaluation_results
	$(PYTHON) $(SCRIPTS_DIR)/05_evaluate.py \
		--model_path $(CHECKPOINTS_DIR)/dpo \
		--output_dir ./evaluation_results \
		--max_boolq_samples 100
	@echo "Evaluation completed! Results in ./evaluation_results"

# Interactive inference
.PHONY: serve
serve:
	@echo "Launching interactive mode..."
	$(PYTHON) $(SCRIPTS_DIR)/06_serve.py \
		--model_path $(CHECKPOINTS_DIR)/dpo \
		--mode interactive \
		--template chatml \
		--temperature 0.7 \
		--max_new_tokens 150

# API server
.PHONY: serve-api
serve-api:
	@echo "Launching API server..."
	$(PYTHON) $(SCRIPTS_DIR)/06_serve.py \
		--model_path $(CHECKPOINTS_DIR)/dpo \
		--mode api \
		--host 127.0.0.1 \
		--port 8000

# Complete pipeline for tiny model
.PHONY: pipeline-tiny
pipeline-tiny: prepare pretrain-tiny
	@echo "Complete tiny pipeline finished!"

# Complete pipeline with fine-tuning (requires SFT and DPO datasets)
.PHONY: pipeline-full
pipeline-full: prepare pretrain-tiny sft dpo evaluate serve
	@echo "Complete pipeline finished! Model ready for use."

# Resume commands from checkpoint
.PHONY: resume-pretrain-tiny
resume-pretrain-tiny:
	@echo "Resuming tiny pre-training from last checkpoint..."
	$(ACCELERATE) $(SCRIPTS_DIR)/02_pretrain.py \
		--config $(TINY_CONFIG) \
		--data_dir $(DATA_DIR)/processed \
		--output_dir $(CHECKPOINTS_DIR)/pretrain/tiny \
		--resume_from_checkpoint $(CHECKPOINTS_DIR)/pretrain/tiny/step_* \
		--learning_rate 3e-4 \
		--batch_size 16 \
		--gradient_accumulation_steps 4

# Quick test with synthetic data (standard attention)
.PHONY: test-pipeline
test-pipeline:
	@echo "Testing pipeline with synthetic data (standard attention)..."
	@mkdir -p $(DATA_DIR)/test
	@[ ! -f $(DATA_DIR)/test/tokenized_data.json ] && python -c "import json; tokens1 = list(range(1, 1251)); tokens2 = list(range(1251, 2501)); tokens3 = list(range(2501, 3751)); json.dump([tokens1, tokens2, tokens3], open('$(DATA_DIR)/test/tokenized_data.json', 'w'))" || true
	$(PYTHON) $(SCRIPTS_DIR)/02_pretrain.py \
		--config $(TINY_CONFIG) \
		--data_path $(DATA_DIR)/test/tokenized_data.json \
		--output_dir $(CHECKPOINTS_DIR)/test \
		--max_steps 10 \
		--logging_steps 5 \
		--no_flash_attn
	@echo "Test completed!"

# Quick test with FlashAttention-2
.PHONY: test-pipeline-flash
test-pipeline-flash:
	@echo "Testing pipeline with synthetic data (FlashAttention-2)..."
	@mkdir -p $(DATA_DIR)/test
	@[ ! -f $(DATA_DIR)/test/tokenized_data.json ] && python -c "import json; tokens1 = list(range(1, 1251)); tokens2 = list(range(1251, 2501)); tokens3 = list(range(2501, 3751)); json.dump([tokens1, tokens2, tokens3], open('$(DATA_DIR)/test/tokenized_data.json', 'w'))" || true
	$(PYTHON) $(SCRIPTS_DIR)/02_pretrain.py \
		--config $(TINY_CONFIG) \
		--data_path $(DATA_DIR)/test/tokenized_data.json \
		--output_dir $(CHECKPOINTS_DIR)/test-flash \
		--max_steps 10 \
		--logging_steps 5 \
		--use_flash_attn
	@echo "FlashAttention test completed!"

# Generate sample datasets
.PHONY: create-sample-datasets
create-sample-datasets:
	@echo "Creating sample datasets..."
	@mkdir -p $(DATA_DIR)
	@echo '[' > $(SFT_DATASET)
	@echo '  {"prompt": "What is AI?", "response": "Artificial intelligence is..."},' >> $(SFT_DATASET)
	@echo '  {"prompt": "How does it work?", "response": "It works thanks to..."}' >> $(SFT_DATASET)
	@echo ']' >> $(SFT_DATASET)
	
	@echo '[' > $(DPO_DATASET)
	@echo '  {"prompt": "Explain AI", "chosen": "AI is a fascinating technology...", "rejected": "I don't know."},' >> $(DPO_DATASET)
	@echo '  {"prompt": "How to learn?", "chosen": "You need to study regularly...", "rejected": "It's easy."}' >> $(DPO_DATASET)
	@echo ']' >> $(DPO_DATASET)
	@echo "Sample datasets created!"

# Resource monitoring during training
.PHONY: monitor
monitor:
	@echo "Monitoring system resources..."
	watch -n 2 'nvidia-smi | head -15; echo ""; ps aux | grep python | head -5; echo ""; df -h | head -5'

# Cleanup Commands
.PHONY: clean
clean:
	@echo "🧹 Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.log" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf __pycache__ 2>/dev/null || true
	@echo "✅ Temporary files cleaned!"

.PHONY: clean-data
clean-data:
	@echo "🗂️ Removing processed datasets..."
	@if [ -d "$(DATA_DIR)/processed" ]; then \
		echo "  Removing $(DATA_DIR)/processed..."; \
		rm -rf $(DATA_DIR)/processed; \
	fi
	@if [ -d "$(DATA_DIR)/test" ]; then \
		echo "  Removing $(DATA_DIR)/test..."; \
		rm -rf $(DATA_DIR)/test; \
	fi
	@if [ -d "$(DATA_DIR)/tokenizer" ]; then \
		echo "  Removing $(DATA_DIR)/tokenizer..."; \
		rm -rf $(DATA_DIR)/tokenizer; \
	fi
	@if [ -f "$(SFT_DATASET)" ]; then \
		echo "  Removing sample SFT dataset..."; \
		rm -f $(SFT_DATASET); \
	fi
	@if [ -f "$(DPO_DATASET)" ]; then \
		echo "  Removing sample DPO dataset..."; \
		rm -f $(DPO_DATASET); \
	fi
	@echo "✅ Datasets cleaned!"

.PHONY: clean-checkpoints
clean-checkpoints:
	@echo "🏋️ Removing checkpoints..."
	@if [ -d "$(CHECKPOINTS_DIR)" ]; then \
		echo "  Removing $(CHECKPOINTS_DIR)..."; \
		rm -rf $(CHECKPOINTS_DIR); \
	fi
	@if [ -d "./evaluation_results" ]; then \
		echo "  Removing evaluation results..."; \
		rm -rf ./evaluation_results; \
	fi
	@echo "✅ Checkpoints removed!"

.PHONY: clean-all
clean-all: clean clean-data clean-checkpoints
	@echo "🎉 Complete cleanup finished!"

.PHONY: clean-repo
clean-repo:
	@echo "🚀 Repository cleanup for commits..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@$(MAKE) clean
	@$(MAKE) clean-data
	@$(MAKE) clean-checkpoints
	@if [ -d "./sessions" ]; then \
		echo "🗂️ Removing session logs..."; \
		rm -rf ./sessions; \
	fi
	@if [ -d "./backups" ]; then \
		echo "💾 Removing backup files..."; \
		rm -rf ./backups; \
	fi
	@if [ -d "./wandb" ]; then \
		echo "📊 Removing wandb logs..."; \
		rm -rf ./wandb; \
	fi
	@if [ -d "./.accelerate" ]; then \
		echo "⚡ Removing accelerate cache..."; \
		rm -rf ./.accelerate; \
	fi
	@echo "✅ Repository is now clean for commit!"
	@echo "📦 Kept: config/, scripts/, utils/, requirements.txt, README.md"
	@echo "🗑️  Removed: data/, checkpoints/, logs, caches, backups"

.PHONY: clean-status
clean-status:
	@echo "📊 Cleanup Status - Current disk usage:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📁 DATASETS:"
	@if [ -d "$(DATA_DIR)/processed" ]; then \
		du -sh $(DATA_DIR)/processed 2>/dev/null | awk '{print "   Processed data: " $$1}' || echo "   Processed data: 0B"; \
	else \
		echo "   Processed data: None"; \
	fi
	@if [ -d "$(DATA_DIR)/tokenizer" ]; then \
		du -sh $(DATA_DIR)/tokenizer 2>/dev/null | awk '{print "   Tokenizers: " $$1}' || echo "   Tokenizers: 0B"; \
	else \
		echo "   Tokenizers: None"; \
	fi
	@echo ""
	@echo "🏋️ CHECKPOINTS:"
	@if [ -d "$(CHECKPOINTS_DIR)" ]; then \
		du -sh $(CHECKPOINTS_DIR) 2>/dev/null | awk '{print "   Models: " $$1}' || echo "   Models: 0B"; \
	else \
		echo "   Models: None"; \
	fi
	@if [ -d "./evaluation_results" ]; then \
		du -sh ./evaluation_results 2>/dev/null | awk '{print "   Eval results: " $$1}' || echo "   Eval results: 0B"; \
	else \
		echo "   Eval results: None"; \
	fi
	@echo ""
	@echo "🗂️ OTHER:"
	@if [ -d "./sessions" ]; then \
		du -sh ./sessions 2>/dev/null | awk '{print "   Session logs: " $$1}' || echo "   Session logs: 0B"; \
	else \
		echo "   Session logs: None"; \
	fi
	@if [ -d "./backups" ]; then \
		du -sh ./backups 2>/dev/null | awk '{print "   Backups: " $$1}' || echo "   Backups: 0B"; \
	else \
		echo "   Backups: None"; \
	fi
	@if [ -d "./wandb" ]; then \
		du -sh ./wandb 2>/dev/null | awk '{print "   W&B logs: " $$1}' || echo "   W&B logs: 0B"; \
	else \
		echo "   W&B logs: None"; \
	fi
	@echo ""
	@echo "💾 TOTAL REPOSITORY SIZE:"
	@du -sh . 2>/dev/null | awk '{print "   " $$1}' || echo "   Unknown"
	@echo ""
	@echo "🧹 CLEANUP COMMANDS:"
	@echo "   make clean          - Remove temp files only"
	@echo "   make clean-data     - Remove datasets only"  
	@echo "   make clean-all      - Remove everything except configs"
	@echo "   make clean-repo     - Full cleanup for git commits"

# Environment check
.PHONY: check-env
check-env:
	@echo "Checking environment..."
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@$(PYTHON) -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	@$(PYTHON) -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
	@echo "Environment check completed!"

# Configuration for different GPU sizes
.PHONY: config-rtx4090
config-rtx4090:
	@echo "Optimized configuration for RTX 4090 (16 GB)..."
	@echo "Using recommended parameters for your GPU"

# Backup configs and models
.PHONY: backup
backup:
	@echo "Backing up important configurations and checkpoints..."
	@mkdir -p ./backups/$(shell date +%Y%m%d_%H%M%S)
	cp -r $(CONFIG_DIR) ./backups/$(shell date +%Y%m%d_%H%M%S)/
	@if [ -d "$(CHECKPOINTS_DIR)/pretrain/tiny/final" ]; then \
		cp -r $(CHECKPOINTS_DIR)/pretrain/tiny/final ./backups/$(shell date +%Y%m%d_%H%M%S)/model_final; \
	fi
	@echo "Backup completed in ./backups/"

# ============================================================================
# FOCUSED DEVELOPMENT SESSIONS
# ============================================================================

# Quick session (30 minutes) - Testing and validation
.PHONY: session-quick
session-quick:
	@echo "🚀 QUICK SESSION (30 min) - Pipeline testing"
	@echo "Start: $(shell date)"
	@mkdir -p $(SESSION_DIR)
	@echo "=== SESSION QUICK $(shell date) ===" > $(SESSION_LOG)
	$(MAKE) check-env 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) create-sample-datasets 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) test-pipeline 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) evaluate-quick 2>&1 | tee -a $(SESSION_LOG)
	@echo "🎉 Quick session completed! Log: $(SESSION_LOG)"
	@echo "Estimated duration: 30 minutes"

# Prototype session (2 hours) - Tiny model development
.PHONY: session-prototype  
session-prototype:
	@echo "🛠️ PROTOTYPE SESSION (2h) - Complete tiny model"
	@echo "Start: $(shell date)"
	@mkdir -p $(SESSION_DIR)
	@echo "=== SESSION PROTOTYPE $(shell date) ===" > $(SESSION_LOG)
	$(MAKE) check-env 2>&1 | tee -a $(SESSION_LOG)
	@if [ ! -f "$(PROCESSED_DATA)" ]; then $(MAKE) prepare-quick 2>&1 | tee -a $(SESSION_LOG); fi
	$(MAKE) pretrain-tiny-quick 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) evaluate-quick 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) validate-architecture 2>&1 | tee -a $(SESSION_LOG)
	@echo "🎉 Prototype session completed! Log: $(SESSION_LOG)"
	@echo "Model available: $(CHECKPOINTS_DIR)/pretrain/tiny/final"

# Experiment session (4 hours) - Small model with fine-tuning
.PHONY: session-experiment
session-experiment:
	@echo "🧪 EXPERIMENT SESSION (4h) - Small model + SFT"
	@echo "Start: $(shell date)"
	@mkdir -p $(SESSION_DIR)
	@echo "=== SESSION EXPERIMENT $(shell date) ===" > $(SESSION_LOG)
	$(MAKE) check-env 2>&1 | tee -a $(SESSION_LOG)
	@if [ ! -f "$(PROCESSED_DATA)" ]; then $(MAKE) prepare 2>&1 | tee -a $(SESSION_LOG); fi
	$(MAKE) pretrain-small 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) create-sample-datasets 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) sft-small 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) evaluate 2>&1 | tee -a $(SESSION_LOG)
	$(MAKE) assess-performance 2>&1 | tee -a $(SESSION_LOG)
	@echo "🎉 Experiment session completed! Log: $(SESSION_LOG)"
	@echo "Model available: $(CHECKPOINTS_DIR)/sft"

# Evaluation session (1 hour) - In-depth analysis
.PHONY: session-evaluation
session-evaluation:
	@echo "📊 EVALUATION SESSION (1h) - Complete analysis"
	@echo "Start: $(shell date)"
	@mkdir -p $(SESSION_DIR)
	@echo "=== SESSION EVALUATION $(shell date) ===" > $(SESSION_LOG)
	@echo "Looking for last trained model..."
	@if [ -d "$(CHECKPOINTS_DIR)/dpo" ]; then \
		echo "Evaluating DPO model" | tee -a $(SESSION_LOG); \
		$(MAKE) MODEL_PATH=$(CHECKPOINTS_DIR)/dpo evaluate-detailed 2>&1 | tee -a $(SESSION_LOG); \
	elif [ -d "$(CHECKPOINTS_DIR)/sft" ]; then \
		echo "Evaluating SFT model" | tee -a $(SESSION_LOG); \
		$(MAKE) MODEL_PATH=$(CHECKPOINTS_DIR)/sft evaluate-detailed 2>&1 | tee -a $(SESSION_LOG); \
	elif [ -d "$(CHECKPOINTS_DIR)/pretrain/tiny/final" ]; then \
		echo "Evaluating tiny model" | tee -a $(SESSION_LOG); \
		$(MAKE) MODEL_PATH=$(CHECKPOINTS_DIR)/pretrain/tiny/final evaluate-detailed 2>&1 | tee -a $(SESSION_LOG); \
	else \
		echo "❌ No model found for evaluation" | tee -a $(SESSION_LOG); \
	fi
	$(MAKE) assess-performance 2>&1 | tee -a $(SESSION_LOG) || true
	@echo "🎉 Evaluation session completed! Log: $(SESSION_LOG)"

# Interactive debug session
.PHONY: session-debug
session-debug:
	@echo "🔧 DEBUG SESSION - Interactive mode"
	@echo "=== AVAILABLE OPTIONS ==="
	@echo "1. Check environment: make check-env"
	@echo "2. Test with synthetic data: make test-pipeline"
	@echo "3. Validate an architecture: make validate-architecture"
	@echo "4. Quick evaluation: make evaluate-quick"
	@echo "5. Resource monitoring: make monitor"
	@echo "6. Clean and restart: make clean"
	@echo "7. List checkpoints: ls -la $(CHECKPOINTS_DIR)/*/"
	@echo ""
	@echo "📝 For detailed logs, add 2>&1 | tee debug.log"
	@echo "Example: make check-env 2>&1 | tee debug.log"

# Architecture validation session
.PHONY: session-architecture
session-architecture:
	@echo "🏗️ ARCHITECTURE SESSION - Configuration validation"
	@mkdir -p $(SESSION_DIR)
	@echo "=== SESSION ARCHITECTURE $(shell date) ===" > $(SESSION_LOG)
	@echo "Validating all configurations..."
	@for config in $(CONFIG_DIR)/*.json; do \
		echo "Validating: $$config" | tee -a $(SESSION_LOG); \
		$(PYTHON) utils/validate_architecture.py "$$config" 2>&1 | tee -a $(SESSION_LOG); \
		echo "" | tee -a $(SESSION_LOG); \
	done
	@echo "🎉 Architecture validation completed! Log: $(SESSION_LOG)"

# ============================================================================
# SUPPORT COMMANDS FOR SESSIONS
# ============================================================================

# Quick data preparation (smaller dataset)
.PHONY: prepare-quick
prepare-quick:
	@echo "Quick data preparation..."
	@mkdir -p $(DATA_DIR)/processed
	$(PYTHON) $(SCRIPTS_DIR)/01_prepare_data.py \
		--input_path "wikitext-2-raw-v1" \
		--output_dir $(DATA_DIR)/processed \
		--vocab_size 32768 \
		--min_length 50 \
		--max_length 1000
	@echo "Quick data prepared!"

# Quick tiny pre-training (fewer epochs)
.PHONY: pretrain-tiny-quick
pretrain-tiny-quick:
	@echo "Quick tiny pre-training (short version)..."
	@mkdir -p $(CHECKPOINTS_DIR)/pretrain/tiny
	$(ACCELERATE) $(SCRIPTS_DIR)/02_pretrain.py \
		--config $(TINY_CONFIG) \
		--data_dir $(DATA_DIR)/processed \
		--output_dir $(CHECKPOINTS_DIR)/pretrain/tiny \
		--learning_rate 5e-4 \
		--batch_size 16 \
		--gradient_accumulation_steps 4 \
		--max_steps 1000 \
		--warmup_steps 100 \
		--save_steps 500 \
		--logging_steps 50
	@echo "Quick tiny pre-training completed!"

# SFT for small model
.PHONY: sft-small
sft-small:
	@echo "Supervised fine-tuning for small model..."
	@mkdir -p $(CHECKPOINTS_DIR)/sft
	$(PYTHON) $(SCRIPTS_DIR)/03_sft.py \
		--model_path $(CHECKPOINTS_DIR)/pretrain/small/final \
		--dataset_path $(SFT_DATASET) \
		--config_path $(SFT_CONFIG) \
		--output_dir $(CHECKPOINTS_DIR)/sft
	@echo "Supervised fine-tuning (small) completed!"

# Quick evaluation for development
.PHONY: evaluate-quick
evaluate-quick:
	@echo "Quick evaluation..."
	@mkdir -p ./evaluation_results
	$(PYTHON) $(SCRIPTS_DIR)/05_evaluate.py \
		--model_path $(or $(MODEL_PATH),$(CHECKPOINTS_DIR)/pretrain/tiny/final) \
		--output_dir ./evaluation_results \
		--fast_mode \
		--max_boolq_samples 20
	@echo "Quick evaluation completed!"

# Detailed evaluation with report
.PHONY: evaluate-detailed
evaluate-detailed:
	@echo "Detailed evaluation with report..."
	@mkdir -p ./evaluation_results
	$(PYTHON) $(SCRIPTS_DIR)/05_evaluate.py \
		--model_path $(or $(MODEL_PATH),$(CHECKPOINTS_DIR)/dpo) \
		--output_dir ./evaluation_results \
		--detailed_output \
		--max_boolq_samples 100
	@echo "Detailed evaluation completed!"

# Automatic performance analysis
.PHONY: assess-performance
assess-performance:
	@echo "Analyzing performance..."
	@if [ -f "./evaluation_results/evaluation_results.json" ]; then \
		$(PYTHON) $(EVALUATION_DIR)/assess_performance.py ./evaluation_results/evaluation_results.json; \
	else \
		echo "❌ Evaluation file not found. Run 'make evaluate' first."; \
	fi

# Architecture validation
.PHONY: validate-architecture
validate-architecture:
	@echo "Validating architecture..."
	@if [ -n "$(CONFIG)" ]; then \
		$(PYTHON) utils/validate_architecture.py $(CONFIG); \
	else \
		$(PYTHON) utils/validate_architecture.py $(TINY_CONFIG); \
	fi

# Session status - shows current state
.PHONY: session-status
session-status:
	@echo "📊 SESSION STATUS"
	@echo "========================================"
	@echo "Current time: $(shell date)"
	@echo "Working directory: $(PWD)"
	@echo ""
	@echo "📁 Available data:"
	@if [ -f "$(PROCESSED_DATA)" ]; then echo "  ✅ Data prepared"; else echo "  ❌ Data not prepared (make prepare)"; fi
	@echo ""
	@echo "🤖 Available models:"
	@if [ -d "$(CHECKPOINTS_DIR)/pretrain/tiny/final" ]; then echo "  ✅ Tiny model"; else echo "  ❌ Tiny model not trained"; fi
	@if [ -d "$(CHECKPOINTS_DIR)/pretrain/small/final" ]; then echo "  ✅ Small model"; else echo "  ❌ Small model not trained"; fi
	@if [ -d "$(CHECKPOINTS_DIR)/pretrain/base/final" ]; then echo "  ✅ Base model"; else echo "  ❌ Base model not trained"; fi
	@if [ -d "$(CHECKPOINTS_DIR)/sft" ]; then echo "  ✅ SFT model"; else echo "  ❌ SFT model not trained"; fi
	@if [ -d "$(CHECKPOINTS_DIR)/dpo" ]; then echo "  ✅ DPO model"; else echo "  ❌ DPO model not trained"; fi
	@echo ""
	@echo "📊 Evaluations:"
	@if [ -f "./evaluation_results/evaluation_results.json" ]; then echo "  ✅ Evaluation results available"; else echo "  ❌ No recent evaluation"; fi
	@echo ""
	@echo "💾 Disk space:"
	@du -sh $(CHECKPOINTS_DIR) 2>/dev/null || echo "  No checkpoints"
	@echo ""
	@echo "🔥 GPU Status:"
	@nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "  GPU not available"

# Session cleanup (keeps important models)
.PHONY: session-cleanup
session-cleanup:
	@echo "🧹 Session cleanup..."
	@$(MAKE) clean
	@rm -rf $(DATA_DIR)/test 2>/dev/null || true
	@find $(SESSION_DIR) -name "*.log" -mtime +7 -delete 2>/dev/null || true
	@echo "✅ Session cleanup completed (models preserved)!"