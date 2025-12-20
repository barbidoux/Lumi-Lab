# Create logs directory
mkdir -p logs/medium

# ============================================================
# STEP 1: Prepare Corpus (1.8B tokens, ~2-3 hours)
# ============================================================
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/medium_84M_chinchilla_1800M.json \
    --output-dir data/datasets/medium_84M_corpus \
    --use-cache \
    --log-level INFO 2>&1 | tee logs/medium/01_prepare_corpus.log

# ============================================================
# STEP 2: Pack Dataset (~30-45 minutes)
# ============================================================
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/medium_84M_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/medium_84M_1024 \
    --skip-tokenizer-check 2>&1 | tee logs/medium/02_pack_dataset.log

# ============================================================
# STEP 3: Pretrain (~40-50 hours on RTX 4090)
# ============================================================
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_medium_1800m.json \
    --data_dirs data/processed/medium_84M_1024 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain/medium \
    --num_workers 4 \
    --resume_from_checkpoint auto \
    --log-level INFO 2>&1 | tee -a logs/medium/03_pretrain.log

# ============================================================
# STEP 4: Evaluate Pretrain
# ============================================================
mkdir -p evaluation_results/pretrain/medium && \
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/50_evaluate_pretrain.py \
    --model_path checkpoints/pretrain/medium/medium/final \
    --tokenizer_path data/models/tokenizers/spm_32k \
    --output_dir evaluation_results/pretrain/medium \
    --log-level INFO 2>&1 | tee logs/medium/04_evaluate_pretrain.log

# ============================================================
# STEP 5: Prepare SFT Corpus (55M tokens, ~2-3 hours)
# ============================================================
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/60_prepare_sft_corpus.py \
    --config config/sft/datasets/medium_84m_sft_balanced.json \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir data/sft_processed/medium_84m_balanced \
    --log-level INFO 2>&1 | tee logs/medium/05_sft_corpus.log

# ============================================================
# STEP 6: SFT Training (~6-8 hours on RTX 4090)
# ============================================================
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab accelerate launch --mixed_precision bf16 scripts/70_train_sft.py \
    --config config/sft/training/lora_medium_84m.json \
    --model_path checkpoints/pretrain/medium/medium/final \
    --data_dirs data/sft_processed/medium_84m_balanced \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --resume_from_checkpoint auto \
    --output_dir checkpoints/sft/medium 2>&1 | tee logs/medium/06_sft_training.log

# ============================================================
# STEP 7: Evaluate SFT
# ============================================================
mkdir -p evaluation_results/sft/medium && \
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/80_evaluate_sft.py \
    --config config/evaluation/sft_standard.json \
    --model_path checkpoints/sft/medium/final \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_file evaluation_results/sft/medium/results.json 2>&1 | tee logs/medium/07_evaluate_sft.log
