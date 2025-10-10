#!/usr/bin/env bash
set -euo pipefail

# Base settings
export BASEPATH_RESULTS='/mnt/sdd/basics/'
export LAYER='features.21'
export LAMBDA='0.6'
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech_multiclass/vgg16/vgg16.pth"
export MODEL_NAME="vgg16"
export CONFIG_FILE='./config_cub_3classes.yaml'
export RESULTS="/mnt/sdb2/cub_3c_xai/xai_images_cub_3c_vgg16_"
# Define array of model paths to iterate
MODELS=(
      "./results_3classes/vgg16/loss_vgg16_features.19_0.5.pth "
      "./results_3classes/vgg16/loss_vgg16_features.24_0.5.pth "
      "./results_3classes/vgg16/loss_vgg16_features.28_0.5.pth "
      "./results_3classes/vgg16/loss_vgg16_features.19_0.6.pth "
      "./results_3classes/vgg16/loss_vgg16_features.24_0.6.pth "
      "./results_3classes/vgg16/loss_vgg16_features.28_0.6.pth "
      "./results_3classes/vgg16/loss_vgg16_features.19_0.7.pth "
      "./results_3classes/vgg16/loss_vgg16_features.24_0.7.pth "
      "./results_3classes/vgg16/loss_vgg16_features.28_0.7.pth "
      "./results_3classes/vgg16/loss_vgg16_features.19_0.8.pth "
      "./results_3classes/vgg16/loss_vgg16_features.24_0.8.pth "
      "./results_3classes/vgg16/loss_vgg16_features.28_0.8.pth "
      "./results_3classes/vgg16/loss_vgg16_features.21_0.5.pth "
      "./results_3classes/vgg16/loss_vgg16_features.26_0.5.pth "
      "./results_3classes/vgg16/loss_vgg16_features.7_0.5.pth  "
      "./results_3classes/vgg16/loss_vgg16_features.21_0.6.pth "
      "./results_3classes/vgg16/loss_vgg16_features.26_0.6.pth "
      "./results_3classes/vgg16/loss_vgg16_features.7_0.6.pth  "
      "./results_3classes/vgg16/loss_vgg16_features.21_0.7.pth "
      "./results_3classes/vgg16/loss_vgg16_features.26_0.7.pth "
      "./results_3classes/vgg16/loss_vgg16_features.7_0.7.pth  "
      "./results_3classes/vgg16/loss_vgg16_features.21_0.8.pth "
      "./results_3classes/vgg16/loss_vgg16_features.26_0.8.pth "
      "./results_3classes/vgg16/loss_vgg16_features.7_0.8.pth  "
)

# Iterate through each model
for MODIFIED_MODEL in "${MODELS[@]}"; do
    echo "Processing model: $MODIFIED_MODEL"

    # Derive suffix for save directory (use basename to keep clean)
    BASE_NAME=$(basename "$MODIFIED_MODEL" .pth)
    SAVE_DIR="${RESULTS}${BASE_NAME}"
    mkdir -p "${SAVE_DIR}"

    python xai_visualization.py \
        --org_model_path "${ORIGINAL_MODEL}" \
        --modified_model_path "${MODIFIED_MODEL}" \
        --model_name "${MODEL_NAME}" \
        --config_file "${CONFIG_FILE}" \
        --save_dir "${SAVE_DIR}"

    echo "? Finished processing ${BASE_NAME}"
    echo "-----------------------------------------"
done
