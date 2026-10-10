#!/bin/bash
LIBRARY_PATH="/home/srikanth/study1/RecalibrateNN"
export PATH="$LIBRARY_PATH:$PATH"
export PYTHONPATH="$LIBRARY_PATH:$PYTHONPATH"

PYTHON_SCRIPT="/home/srikanth/study1/RecalibrateNN/lean_recalibration/main_store_cav.py"
# Configuration variables for models
MODELS=("vgg16" "resnet50" "inception_v3" "mobilenet_v3_small" "mobilenet_v3_large")
MODEL_PATH="/mnt/sdd/basics/balanced_training/"
CONFIG_FILE="/home/srikanth/study1/RecalibrateNN/config/legacy/config_legacy_3classes_ver4.yaml"
CAV_STORE="/mnt/sdd/basics/cavstore"


# Execute for each model
for model in "${MODELS[@]}"; do
    echo "Processing model: $model"
    python "$PYTHON_SCRIPT" \
        --model_name "$model" \
        --model_path "$MODEL_PATH" \
        --config_file "$CONFIG_FILE" \
        --cav_store "$CAV_STORE"

done

# Execute for each model
#for model in "${MODELS[@]}"; do
#    echo "Processing model: $model"
#    python "$PYTHON_SCRIPT" \
#        --model_name "$model" \
#        --model_path "$MODEL_PATH" \
#        --config_file "$CONFIG_FILE" \
#        --cav_store "$CAV_STORE" \
#        --store_multiconcept_cav
#
#done