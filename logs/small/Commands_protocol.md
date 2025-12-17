# Step 1: Corpus (reuse existing tokenizer!)
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/10_prepare_corpus.py \
    --config config/pretrain/corpus/small_42M_chinchilla_900M.json \
    --output-dir data/datasets/small_42M_corpus \
    --log-level INFO 2>&1 | tee logs/small/01_corpus.log

# Step 2: Pack
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/30_pack_dataset.py \
    --config config/pretrain/packing/default.json \
    --corpus-dir data/datasets/small_42M_corpus \
    --tokenizer-dir data/models/tokenizers/spm_32k \
    --output-dir data/processed/small_42M_1024 \
    --skip-tokenizer-check \
    --force \
    --log-level INFO 2>&1 | tee logs/small/02_pack.log
# Step 3: Pretrain
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab accelerate launch --mixed_precision bf16 scripts/40_pretrain.py \
    --config config/pretrain/training/chinchilla_small_900m.json \
    --data_dirs data/processed/small_42M_1024 \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/pretrain/small \
    --num_workers 4 \
    --log-level INFO 2>&1 | tee logs/small/03_pretrain.log

# Step 4: Evaluate Pretrain
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/50_evaluate_pretrain.py \
    --model_path checkpoints/pretrain/small/small/final \
    --tokenizer_path data/models/tokenizers/spm_32k \
    --output_dir evaluation_results/pretrain/small \
    --log-level INFO 2>&1 | tee logs/small/04_evaluate_pretrain.log

# Step 5: SFT Corpus
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/60_prepare_sft_corpus.py \
    --config config/sft/datasets/small_42m_sft_balanced.json \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir data/sft_processed/small_42m_balanced \
    --force 2>&1 | tee logs/small/05_sft_corpus.log

# Step 6: SFT Training
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab accelerate launch --mixed_precision bf16 scripts/70_train_sft.py \
    --config config/sft/training/lora_small_42m.json \
    --model_path checkpoints/pretrain/small/small/final \
    --data_dirs data/sft_processed/small_42m_balanced \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_dir checkpoints/sft/small 2>&1 | tee logs/small/06_sft_training.log

# Step 7: Evaluate SFT
mkdir -p evaluation_results/sft/small && \
PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/c/Users/matth/GITREPOS/Lumi-Lab python scripts/80_evaluate_sft.py \
    --config config/evaluation/sft_standard.json \
    --model_path checkpoints/sft/small/final \
    --tokenizer_dir data/models/tokenizers/spm_32k \
    --output_file evaluation_results/sft/small/results.json 2>&1 | tee logs/small/07_evaluate_sft.log