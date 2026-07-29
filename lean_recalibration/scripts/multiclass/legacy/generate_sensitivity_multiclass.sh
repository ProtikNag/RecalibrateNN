#!/bin/bash
LIBRARY_PATH="/home/srikanth/study1/RecalibrateNN"
export PATH="$LIBRARY_PATH:$PATH"
export PYTHONPATH="$LIBRARY_PATH:$PYTHONPATH"

PYTHON_SCRIPT="/home/srikanth/study1/RecalibrateNN/lean_recalibration/utils_sensitivity_multiclass.py"
# Configuration variables for models
MODELS=("vgg16" "resnet50" "inception_v3" "mobilenet_v3_small" "mobilenet_v3_large")
MODEL_PATH="/mnt/sdd/biased_models/base_model"
RECALIB_PATH=""
CONFIG_FILE="/home/srikanth/study1/RecalibrateNN/config/multiclass/legacy/config_legacy_3classes.yaml"
SENSITIVITY_STORE="/mnt/sdd/biased_models/multiclass_sensitivity"
CAV_STORE="/mnt/sdd/biased_models/multiclass_cav/"
# Execute for each model
for model in "${MODELS[@]}"; do
    echo "Processing model: $model"
    python "$PYTHON_SCRIPT" \
        --model_name "$model" \
        --org_model_path "$MODEL_PATH/${model}/${model}.pth" \
        --recal_model_basepath "$RECALIB_PATH" \
        --config "$CONFIG_FILE" \
        --store_results "$SENSITIVITY_STORE" \
        --concept_mode "multiclass" \
        --manifest "$CAV_STORE/${model}_manifest.json"
done
