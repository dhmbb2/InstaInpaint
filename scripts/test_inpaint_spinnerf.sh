#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

EXP_NAME="${1:-spinnerf_eval}"

BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
FOV="${FOV:-57.5270,88.3793}"
INPUT_IMAGE_RES="${INPUT_IMAGE_RES:-512,904}"
OUTPUT_IMAGE_RES="${OUTPUT_IMAGE_RES:-512,904}"
IMAGE_NUM_PER_BATCH="${IMAGE_NUM_PER_BATCH:-4}"
OUTPUT_IMAGE_NUM="${OUTPUT_IMAGE_NUM:-40}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/outputs}"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/spinnerf_dataset}"
CONTEXT_JSON="${CONTEXT_JSON:-${ROOT_DIR}/eval_json/spinnerf_context_triangle.json}"
VIS_CENTER_FILE="${VIS_CENTER_FILE:-${ROOT_DIR}/eval_json/spinnerf_vis_centroid.json}"
PREINPAINT_JSON="${PREINPAINT_JSON:-${ROOT_DIR}/eval_json/spinnerf_inpaint_triangle.json}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/checkpoints/exp_ins+random+3d_121_multimask.pth}"

python -m instainpaint.evaluate \
    --exp_root "${EXP_ROOT}" \
    --exp_name "${EXP_NAME}" \
    --batch_size_per_gpu "${BATCH_SIZE_PER_GPU}" \
    --num_workers "${NUM_WORKERS}" \
    --data_path "${DATA_PATH}" \
    --fov "${FOV}" \
    --image_num_per_batch "${IMAGE_NUM_PER_BATCH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --input_image_res "${INPUT_IMAGE_RES}" \
    --output_image_res "${OUTPUT_IMAGE_RES}" \
    --output_image_num "${OUTPUT_IMAGE_NUM}" \
    --dataset_type spin_nerf_dataset \
    --dataset_formulation scene \
    --interactive_session \
    --centralized_cropping \
    --save_model_num 30 \
    --save_video \
    --render_depth \
    --camera_trajectory circle \
    --context_json "${CONTEXT_JSON}" \
    --vis_center_file "${VIS_CENTER_FILE}" \
    --preinpaint_json "${PREINPAINT_JSON}" \
    --vis_scale 2.5 \
    --output_has_gt \
    --depth_vis_mode inpaint \
    --cal_metric_mode \
    --skip_image_saving
