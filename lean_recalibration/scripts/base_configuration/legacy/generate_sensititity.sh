#!/bin/bash
LIBRARY_PATH="/home/srikanth/study1/RecalibrateNN"
export PATH="$LIBRARY_PATH:$PATH"
export PYTHONPATH="$LIBRARY_PATH:$PYTHONPATH"

PYTHON_SCRIPT="/home/srikanth/study1/RecalibrateNN/lean_recalibration/utils_sensitivity_multiclass.py"
# Configuration variables for models
MODELS=("vgg16" "resnet50" "inception_v3" "mobilenet_v3_small" "mobilenet_v3_large")
#MODELS=("inception_v3")
MODEL_PATH="/mnt/sdd/basics/balanced_training"
CONFIG_FILE="/home/srikanth/study1/RecalibrateNN/config/legacy/config_legacy_3classes_ver4.yaml"
CAV_STORE="/mnt/sdd/basics/cavstore"
RECALIB_PASE_PATH="/mnt/sdc/try_stored_cav"
STORE_RESULTS="/mnt/sdc/try_stored_cav/sensitivity"
MANIFEST_FILE="/mnt/sdd/basics/cavstore"

# Execute for each model
for model in "${MODELS[@]}"; do
    echo "Processing model: $model"
    python "$PYTHON_SCRIPT" \
        --org_model_path "$MODEL_PATH"/${model}/${model}.pth \
        --recal_model_basepath "$RECALIB_PASE_PATH" \
        --model_name "$model" \
        --before_after \
        --config_file "$CONFIG_FILE" \
        --store_results "$STORE_RESULTS" \
        --concept_mode "multiclass" \
        --manifest_file "$MANIFEST_FILE"/${model}_manifest.json

done

