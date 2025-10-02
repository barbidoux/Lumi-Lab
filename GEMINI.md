# GEMINI Project Guide for Lumi-Lab

This document provides instructions for the Gemini AI assistant to effectively interact with the `Lumi-Lab` codebase by using the core Python scripts directly.

## 1. Project Overview

`Lumi-Lab` is a sophisticated environment for creating and fine-tuning smaller language models. The pipeline is modular, allowing for detailed control over data preparation, pre-training, supervised fine-tuning (SFT), and evaluation. The project relies on configuration files (`.json`) to define every aspect of the workflow.

## 2. Setup Instructions

To set up the development environment, install all required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```
*Note: For GPU usage with FlashAttention-2, you may need to install it separately after the other requirements.*

## 3. Core Workflows

The project follows a clear, multi-step pipeline. All scripts are driven by JSON configuration files.

### Step 1: Prepare Corpus

This step processes raw data from various sources into a clean, sharded, and ready-to-use format for the tokenizer and trainer.

- **Command:**
  ```bash
  python scripts/01_prepare_corpus.py --config [DATASET_CONFIG_PATH] --output-dir [OUTPUT_DIR] --use-cache
  ```
- **Example:**
  ```bash
  python scripts/01_prepare_corpus.py --config config/datasets/training_configs/tiny_23M_chinchilla_500M.json --output-dir data/datasets/tiny_23M_prepared --use-cache
  ```
- **Key Arguments:**
  - `--config`: Path to the dataset configuration JSON (e.g., `config/datasets/training_configs/tiny_23M_chinchilla_500M.json`).
  - `--output-dir`: Where to save the processed corpus.
  - `--use-cache`: Enables the caching system for faster, resumable processing.
  - `--analyze-only`: To inspect the data sources and plan without processing.

### Step 2: Pre-train Model

This step trains a model from scratch on one or more prepared datasets. It uses `accelerate` for distributed training.

- **Command:**
  ```bash
  accelerate launch scripts/04_pretrain.py \
    --config [MODEL_CONFIG_PATH] \
    --data_dirs [PROCESSED_DATA_DIR_1] [PROCESSED_DATA_DIR_2] \
    --tokenizer_dir [TOKENIZER_DIR_PATH] \
    --output_dir [CHECKPOINT_OUTPUT_DIR]
  ```
- **Example:**
  ```bash
  accelerate launch scripts/04_pretrain.py \
    --config config/tiny.json \
    --data_dirs data/datasets/tiny_23M_prepared \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain
  ```
- **Key Arguments:**
  - `--config`: Path to the model architecture JSON (e.g., `config/tiny.json`).
  - `--data_dirs`: One or more directories containing the processed corpus from Step 1.
  - `--tokenizer_dir`: Path to the tokenizer model directory.
  - `--output_dir`: Where to save model checkpoints.
  - `--resume_from_checkpoint 'auto'`: To automatically resume from the latest checkpoint.

### Step 3: Supervised Fine-Tuning (SFT)

This step fine-tunes a pre-trained model on a specific task using instruction-based datasets. It uses LoRA for parameter-efficient fine-tuning (PEFT). The `--use_trl_trainer` flag is recommended for faster execution.

- **Command:**
  ```bash
  accelerate launch scripts/03_sft_industrial.py \
    --config [SFT_CONFIG_PATH] \
    --model_path [BASE_MODEL_PATH] \
    --data_dirs [SFT_DATA_DIR] \
    --tokenizer_path [TOKENIZER_PATH] \
    --output_dir [SFT_OUTPUT_DIR] \
    --use_trl_trainer
  ```
- **Example:**
  ```bash
  accelerate launch scripts/03_sft_industrial.py \
    --config config/sft_training/lora_tiny.json \
    --model_path checkpoints/pretrain/tiny/final \
    --data_dirs data/sft_processed/alpaca_chatml_32k_1024 \
    --tokenizer_path data/tokenizer/spm32k.model \
    --output_dir checkpoints/sft/tiny_alpaca \
    --use_trl_trainer
  ```
- **Key Arguments:**
  - `--config`: Path to the SFT training configuration (e.g., `config/sft_training/lora_tiny.json`).
  - `--model_path`: Path to the pre-trained model checkpoint from Step 2.
  - `--data_dirs`: Directory of the prepared SFT dataset.
  - `--tokenizer_path`: Path to the tokenizer model file.
  - `--output_dir`: Where to save the fine-tuned adapter and checkpoints.

### Step 4: Evaluate Model

This final step assesses the model's performance on perplexity benchmarks (like Wikitext or custom data) and qualitative smoke tests.

- **Command:**
  ```bash
  python scripts/05_evaluate.py \
    --model_path [TRAINED_MODEL_PATH] \
    --tokenizer_path [TOKENIZER_PATH] \
    --output_dir [EVAL_RESULTS_DIR] \
    --data_dirs [PROCESSED_DATA_DIR_FOR_PPL]
  ```
- **Example:**
  ```bash
  python scripts/05_evaluate.py \
    --model_path checkpoints/sft/tiny_alpaca/final_model \
    --tokenizer_path data/tokenizer/spm32k.model \
    --output_dir evaluation_results/tiny_alpaca_eval \
    --data_dirs data/datasets/tiny_23M_prepared
  ```
- **Key Arguments:**
  - `--model_path`: Path to the final trained model (pre-trained or SFT).
  - `--tokenizer_path`: Path to the tokenizer used during training.
  - `--output_dir`: Directory to save `evaluation_results.json`.
  - `--data_dirs`: (Optional) Path to prepared data to calculate perplexity against.

## 4. Verification and Testing

The project does not contain a formal test suite (e.g., `pytest`). Model verification is performed via the **evaluation script**:

- **`scripts/05_evaluate.py`**: This is the primary tool for verifying model quality. When I make changes to the training or model architecture, I should run this script to ensure performance (perplexity, BoolQ accuracy) has not regressed.
- **Smoke Tests**: The evaluation script also runs smoke tests to check for basic generation capabilities.

