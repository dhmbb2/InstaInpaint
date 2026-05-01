#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$(dirname "$0")/.."

EXP_NAME="${1:?Usage: $0 EXP_NAME}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-11451}"
export WORLD_SIZE="${WORLD_SIZE:-1}"

DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/dl3dv_960/}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/training_logs}"
MASK_CACHE_PATH="${MASK_CACHE_PATH:-${ROOT_DIR}/data/mask_cache.h5}"
DEPTH_CACHE_PATH="${DEPTH_CACHE_PATH:-${ROOT_DIR}/data/depth_cache.h5}"
PRETRAIN_CKPT_PATH="${PRETRAIN_CKPT_PATH:-${ROOT_DIR}/checkpoints/pretraining_ckpt.pth}"

torchrun \
    --master-port="${MASTER_PORT}" \
    --nnodes=1 \
    --nproc_per_node="${WORLD_SIZE}" \
    -m instainpaint.train \
    --dataset_type dl3dv_dataset \
    --dataset_formulation scene \
    --data_path "${DATA_PATH}" \
    --exp_root "${EXP_ROOT}" \
    --exp_name "${EXP_NAME}" \
    --pretrain_ckpt_path "${PRETRAIN_CKPT_PATH}" \
    --loss_weights_file aegaussian_scene \
    --epochs 100 \
    --warmup_iters 200 \
    --lr 8e-5 \
    --min_lr 1e-6 \
    --batch_size_per_gpu 8 \
    --num_workers 8 \
    --saveimg_iter_freq 300 \
    --saveckp_epoch_freq 5 \
    --backup_ckp_epoch_freq 10 \
    --image_num_per_batch 4 \
    --output_image_num 8 \
    --input_image_res 512,512 \
    --output_image_res 512,512 \
    --seed 1024 \
    --transformer_depth 24 \
    --upsampler_mlp_dim 1024 \
    --patch_size 8 \
    --depth_bias=-4 \
    --weight_norm \
    --remove_attn_bias \
    --remove_emb_bias \
    --remove_norm_bias \
    --remove_norm_affine \
    --centralized_cropping \
    --interactive_session \
    --mask_cache_path "${MASK_CACHE_PATH}" \
    --stereo_depth_cache_path "${DEPTH_CACHE_PATH}" \
    --clip_len 15 \
    --mask_mode "instance+random+3dconsistent" \
    --mask_prob 1 2 1 \
    --project_gaussian_mode all \
    --mask_multiple_objects 
