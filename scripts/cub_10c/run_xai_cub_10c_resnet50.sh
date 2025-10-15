#!/usr/bin/env bash
set -euo pipefail
export PYTHON_SCRIPT="../../xai_visualization.py"
# Base settings
export MODEL_NAME="resnet50"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech_10c_models/${MODEL_NAME}/${MODEL_NAME}.pth"
export CONFIG_FILE='../../config_cub_multiclass.yaml'
export RESULTS="/mnt/sdb2/cub_10c_xai/${MODEL_NAME}/"

# Define array of model paths to iterate
MODELS=(
    "/mnt/sdd/cub_10c/resnet50/loss_resnet50_layer1.0.conv1_0.5.pth"
    "/mnt/sdd/cub_10c/resnet50/loss_resnet50_layer1.0.conv1_0.6.pth"
    "/mnt/sdd/cub_10c/resnet50/loss_resnet50_layer1.1.conv3_0.5.pth"
    "/mnt/sdd/cub_10c/resnet50/loss_resnet50_layer1.1.conv3_0.6.pth"
    "/mnt/sdd/cub_10c/resnet50/loss_resnet50_layer2.0.conv1_0.5.pth"
    "/mnt/sdd/cub_10c/resnet50/loss_resnet50_layer2.0.conv1_0.6.pth"
    "/mnt/sdd/cub_10c/resnet50/loss_resnet50_layer3.0.conv1_0.6.pth"
    "/mnt/sdd/cub_10c/resnet50/loss_resnet50_layer3.0.conv2_0.5.pth"
    "/mnt/sdd/cub_10c/resnet50/loss_resnet50_layer4.1.conv1_0.5.pth"
    "/mnt/sdd/cub_10c/resnet50/loss_resnet50_layer4.1.conv1_0.6.pth"


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
