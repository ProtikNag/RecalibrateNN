#!/usr/bin/env bash
set -euo pipefail

# Base settings
export PYTHON_SCRIPT="../../xai_visualization.py"
export MODEL_NAME="mobilenet_v3_small"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/legacy/${MODEL_NAME}/${MODEL_NAME}.pth"
export CONFIG_FILE='../../config_legacy_3classes.yaml'
export RESULTS="/mnt/sdb2/legacy_3c_xai/${MODEL_NAME}/xai_images_cub_3c"
# Define array of model paths to iterate
MODELS=(
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.10.block.3.0_0.5.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.10.block.3.0_0.6.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.8.block.0.0_0.5.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.8.block.0.0_0.6.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.11.block.3.0_0.5.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.11.block.3.0_0.6.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.8.block.2.fc2_0.5.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.8.block.2.fc2_0.6.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.1.block.0.0_0.5.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.1.block.0.0_0.6.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.8.block.3.0_0.5.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.8.block.3.0_0.6.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.5.block.0.0_0.5.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.5.block.0.0_0.6.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.9.block.1.0_0.5.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.9.block.1.0_0.6.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.5.block.3.0_0.5.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.5.block.3.0_0.6.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.9.block.2.fc1_0.5.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.9.block.2.fc1_0.6.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.6.block.3.0_0.5.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.6.block.3.0_0.6.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.9.block.3.0_0.5.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.9.block.3.0_0.6.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.7.block.3.0_0.5.pth"
    "/mnt/sdd/basics/mobilenet_v3_small/loss_mobilenet_v3_small_features.7.block.3.0_0.6.pth"
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
