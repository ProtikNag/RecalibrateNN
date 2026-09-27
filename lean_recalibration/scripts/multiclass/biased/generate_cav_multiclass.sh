#!/bin/bash
LIBRARY_PATH="/home/srikanth/study1/RecalibrateNN"
export PATH="$LIBRARY_PATH:$PATH"
export PYTHONPATH="$LIBRARY_PATH:$PYTHONPATH"

PYTHON_SCRIPT="/home/srikanth/study1/RecalibrateNN/lean_recalibration/main_store_cav.py"
# Configuration variables for models
MODELS=("vgg16" "resnet50" "inception_v3" "mobilenet_v3_small" "mobilenet_v3_large")
MODEL_PATH="/mnt/sdd/biased_models/base_model"
CONFIG_FILE="/home/srikanth/study1/RecalibrateNN/config/multiclass/biased/config_biased_3classes.yaml"

CAV_STORE="/mnt/sdd/biased_models/multiclass_cav_biased"

# Execute for each model
for model in "${MODELS[@]}"; do
    echo "Processing model: $model"
    python "$PYTHON_SCRIPT" \
        --model_name "$model" \
        --model_path "$MODEL_PATH" \
        --config_file "$CONFIG_FILE" \
        --cav_store "$CAV_STORE" \
        --store_multiconcept_cav
done
