#!/usr/bin/env bash
# Launch MerMED-FM self-supervised pretraining with torchrun (one process per GPU).
# Edit the CONFIG block below, then run:  bash train_mermed.sh
set -euo pipefail

# ----------------------------- CONFIG (edit me) -----------------------------
NUM_GPUS=8
# Training manifest CSV (columns: image_id, image_path, modality).
DATA_PATH="/path/to/MedFM/MerMED_Mix4.csv"
# Optional backbone checkpoint to initialize from (leave empty to train from scratch).
PRETRAINED_PATH=""
# Optional directory holding a checkpoint.pth to resume from (leave empty for a fresh run).
RESUME_FROM_DIR=""
# Where checkpoints and run metadata are written.
OUTPUT_DIR="./output_mermed"
# Set to 1 to train without Weights & Biases (checkpoints still go to OUTPUT_DIR).
NO_WANDB=0
# 100-epoch schedule used for the released MerMED.pth (its teacher snapshot is epoch 50).
EPOCHS=100
# ---------------------------------------------------------------------------

extra_args=()
if [[ "${NO_WANDB}" -eq 1 ]]; then
    extra_args+=(--no_wandb)
fi

torchrun --nproc-per-node "${NUM_GPUS}" main_mermed.py \
    --arch vit_base \
    --patch_size 16 \
    --batch_size_per_gpu 128 \
    --num_workers 10 \
    --local_crops_number 10 \
    --global_crops_scale 0.2 1 \
    --local_crops_scale 0.05 0.2 \
    --out_dim 131072 \
    --partition_size 16384 \
    --optimizer adamw \
    --lr 5e-5 \
    --min_lr 1e-06 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --layer_decay 0.8 \
    --warmup_epochs 10 \
    --momentum_teacher 0.9995 \
    --warmup_teacher_temp 0.04 \
    --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 10 \
    --drop_path_rate 0.1 \
    --use_bn_in_head true \
    --clip_grad 1 \
    --print_freq 50 \
    --epochs "${EPOCHS}" \
    --data_path "${DATA_PATH}" \
    --pretrained_path "${PRETRAINED_PATH}" \
    --resume_from_dir "${RESUME_FROM_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    "${extra_args[@]+"${extra_args[@]}"}"
