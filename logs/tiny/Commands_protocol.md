Step 3: Pretrain Corpus (600M tokens) - ALREADY RUNNING
# Already in progress - monitor with:
tail -f logs/03_pretrain_corpus.log
Step 4: Dataset Packing
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/tiny_23M_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/tiny_23M_1024 \
    --log-level INFO 2>&1 | tee logs/04_pack_dataset.log
Step 5: Pre-training (23M model, ~24h on RTX 4090)
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_tiny_500m.json \
    --data_dirs data/processed/tiny_23M_1024 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain/tiny \
    --num_workers 4 \
    --log-level INFO 2>&1 | tee logs/05_pretrain.log
Step 6: Evaluate Pre-training
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/50_evaluate_pretrain.py \
    --model_path checkpoints/pretrain/tiny/tiny/final \
    --tokenizer_path data/models/tokenizers/spm_32k \
    --output_dir evaluation_results/pretrain/tiny \
    --log-level INFO 2>&1 | tee logs/06_evaluate_pretrain.log
Step 7a: SFT Corpus Preparation (Alpaca)
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/60_prepare_sft_corpus.py \
    --config config/sft/datasets/tiny_23m_sft_balanced.json \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir data/sft_processed/tiny_23m_balanced \
    --force 2>&1 | tee logs/07a_sft_corpus.log
Step 7b: SFT Training (LoRA, ~2-3h on RTX 4090)
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab accelerate launch --mixed_precision bf16 scripts/70_train_sft.py \
    --config config/sft/training/lora_tiny_23m.json \
    --model_path checkpoints/pretrain/tiny/tiny/final \
    --data_dirs data/sft_processed/tiny_23m_balanced \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/sft/tiny 2>&1 | tee logs/07b_sft_training.log
Step 8: Final SFT Evaluation
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/80_evaluate_sft.py \
    --config config/evaluation/sft_standard.json \
    --model_path checkpoints/sft/tiny/final \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_file evaluation_results/sft/tiny/results.json 2>&1 | tee logs/08_evaluate_sft.log
Important Notes:
Step 5 pretrain saves to checkpoints/pretrain/tiny/tiny/ (double nesting)
Step 7b SFT saves to checkpoints/sft/tiny/ (no double nesting)
All logs go to logs/ directory
Pre-training is the longest step (~24h on RTX 4090)