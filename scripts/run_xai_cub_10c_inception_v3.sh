#!/usr/bin/env bash
set -euo pipefail

export PYTHON_SCRIPT="../xai_visualization.py"
# Base settings
export MODEL_NAME="inception_v3"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech_multiclass/${MODEL_NAME}/${MODEL_NAME}.pth"

export CONFIG_FILE='../config_cub_multiclass.yaml'
export RESULTS="/mnt/sdb2/cub_10c_xai/${MODEL_NAME}/"
# Define array of model paths to iterate
MODELS=(
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch1x1.conv_0.5.pth"
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch1x1.conv_0.6.pth"
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch1x1.conv_0.7.pth"
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch1x1.conv_0.8.pth"
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_1.conv_0.5.pth"
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_1.conv_0.6.pth"
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_1.conv_0.7.pth"
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_1.conv_0.8.pth"
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_2.conv_0.5.pth"
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_2.conv_0.6.pth"
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_2.conv_0.7.pth"
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_2.conv_0.8.pth"
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_3.conv_0.5.pth"
"/home/srikanth/study1/RecalibrateNN/results_10classes/results/inception_v3/loss_inception_v3_Mixed_6e.branch7x7_3.conv_0.6.pth"
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
