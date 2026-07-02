#!/bin/bash
#Usage ./run_all_predictions.sh or ./run_all_predictions.sh vgg16 resnet50 inception_v3 
# Configuration variables
IMAGE_FOLDER="/home/biased_dataset/baseline_data/deer/biased_data/biased_valid_dataset/valid"
CLASS_NAMES="deer_neverseen,horse,zebra"
DESTINATION_DIR="/mnt/sdd/biased_models/biased_prediction/before_after_all"
BASE_MODEL_PATH="/mnt/sdd/biased_models/base_model"
RECALIB_BASE_PATH="/mnt/sdd/biased_models/recalib_selected"

# ============================================================
# SELECT MODELS TO RUN (comment/uncomment as needed)
# ============================================================
MODELS=(
    "vgg16"
    #"resnet50"
    #"inception_v3"
    #"mobilenet_v3_small"
    #"mobilenet_v3_large"
)

# ============================================================
# OR pass model as argument: ./script.sh vgg16 resnet50
# ============================================================
if [ $# -gt 0 ]; then
    MODELS=("$@")
fi

# ============================================================
# Process each selected model
# ============================================================
echo "============================================"
echo "Models to process: ${MODELS[@]}"
echo "============================================"

for model in "${MODELS[@]}"; do

    echo ""
    echo "--------------------------------------------"
    echo "Processing model: $model"
    echo "--------------------------------------------"

    # Build paths
    MODEL_PATH="$BASE_MODEL_PATH/$model/$model.pth"
    MODEL_DIR="$DESTINATION_DIR/$model"
    RECALIB_DIR="$RECALIB_BASE_PATH/$model"
    OUTPUT_EXCEL="$MODEL_DIR/predictions_${model}.xlsx"

    # Check if base model exists
    if [ ! -f "$MODEL_PATH" ]; then
        echo "[ERROR] Base model not found: $MODEL_PATH"
        continue
    fi

    # Check if recalibrated model directory exists
    if [ ! -d "$RECALIB_DIR" ]; then
        echo "[ERROR] Recalibrated model directory not found: $RECALIB_DIR"
        continue
    fi

    # Create output directory
    mkdir -p "$MODEL_DIR"

    # Run prediction
    echo "Running predictions for $model..."
    echo "  Base model     : $MODEL_PATH"
    echo "  Recalib dir    : $RECALIB_DIR"
    echo "  Output Excel   : $OUTPUT_EXCEL"

    python predict_before_after_all.py \
        --base_model_path "$MODEL_PATH" \
        --class_names "$CLASS_NAMES" \
        --image_folder "$IMAGE_FOLDER" \
        --dest_dir "$MODEL_DIR" \
        --recalibrated_model_directory "$RECALIB_DIR" \
        --output_excel "$OUTPUT_EXCEL"

    # Check if python script succeeded
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Completed processing for model: $model"
    else
        echo "[ERROR] Failed processing for model: $model"
    fi

done

echo ""
echo "============================================"
echo "All selected models processed!"
echo "============================================"
