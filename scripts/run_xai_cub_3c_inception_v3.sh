#!/usr/bin/env bash
set -euo pipefail

# Base settings
export PYTHON_SCRIPT="..\xai_visualization.py"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech_3class/inception_v3/inception_v3.pth"
export MODEL_NAME="inception_v3"
export CONFIG_FILE='../config_cub_3classes.yaml'
export RESULTS="/mnt/sdb2/cub_3c_xai/xai_images_cub_3c_${MODEL_NAME}_"
# Define array of model paths to iterate
MODELS=(
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch1x1_0.5.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch1x1.conv_0.5.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch1x1_0.6.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch1x1.conv_0.6.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch1x1_0.7.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch1x1.conv_0.7.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch1x1_0.8.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch1x1.conv_0.8.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch1x1.conv_0.5.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3_1.conv_0.5.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch1x1.conv_0.6.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3_1.conv_0.6.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch1x1.conv_0.7.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3_1.conv_0.7.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch1x1.conv_0.8.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3_1.conv_0.8.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_1.conv_0.5.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3_2a.conv_0.5.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_1.conv_0.6.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3_2a.conv_0.6.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_1.conv_0.7.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3_2a.conv_0.7.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_1.conv_0.8.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3_2a.conv_0.8.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_2.conv_0.5.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3_2b.conv_0.5.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_2.conv_0.6.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3_2b.conv_0.6.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_2.conv_0.7.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3_2b.conv_0.7.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_2.conv_0.8.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3_2b.conv_0.8.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_3.conv_0.5.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3dbl_3a_0.5.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_3.conv_0.6.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3dbl_3a_0.6.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_3.conv_0.7.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3dbl_3a_0.7.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_3.conv_0.8.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch3x3dbl_3a_0.8.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7dbl_1.conv_0.5.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch_pool.conv_0.5.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7dbl_1.conv_0.6.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch_pool.conv_0.6.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7dbl_1.conv_0.7.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch_pool.conv_0.7.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_6e.branch7x7dbl_1.conv_0.8.pth"
"../results_3classes/inception_v3/loss_inception_v3_Mixed_7b.branch_pool.conv_0.8.pth"

)

# Iterate through each model
for MODIFIED_MODEL in "${MODELS[@]}"; do
    echo "Processing model: $MODIFIED_MODEL"

    # Derive suffix for save directory (use basename to keep clean)
    BASE_NAME=$(basename "$MODIFIED_MODEL" .pth)
    SAVE_DIR="${RESULTS}${BASE_NAME}"
    mkdir -p "${SAVE_DIR}"

    python ${PYTHON_SCRIPT} \
        --org_model_path "${ORIGINAL_MODEL}" \
        --modified_model_path "${MODIFIED_MODEL}" \
        --model_name "${MODEL_NAME}" \
        --config_file "${CONFIG_FILE}" \
        --save_dir "${SAVE_DIR}"

    echo "? Finished processing ${BASE_NAME}"
    echo "-----------------------------------------"
done
