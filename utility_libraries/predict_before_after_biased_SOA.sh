#!/bin/bash

# Configuration variables
#IMAGE_FOLDER="/home/biased_dataset/baseline_data/deer/biased_data/biased_dataset/valid"
IMAGE_FOLDER="/home/biased_dataset/baseline_data/deer/biased_data/biased_valid_dataset/valid"
CLASS_NAMES="deer_neverseen,horse,zebra"

BASE_MODEL_PATH="/mnt/sdd/biased_models/base_model"
#METHOD=("erm" "group_dro" "jtt" "lss")
METHOD=("lss")
# Array of models
MODELS=("vgg16" "inception_v3" "resnet50" "mobilenet_v3_small" "mobilenet_v3_large")

# Process all models
for method in "${METHOD[@]}"; do
    for model in "${MODELS[@]}"; do
      if [[ "$METHOD" == "lss" ]]; then 
        RECALIBRATED_MODELS="/mnt/sdd/biased_models/base_model/state_of_art/${method}/${model}/${model}_full.pth"
      else
        RECALIBRATED_MODELS="/mnt/sdd/biased_models/base_model/state_of_art/${method}/${model}/${model}.pth"
      fi
    echo "Processing model: $model"
    echo "Recalibrated models: $RECALIBRATED_MODELS"
    DESTINATION_DIR="/mnt/sdd/biased_models/biased_prediction/SOA/${method}/before_after/${model}"
    mkdir -p "$DESTINATION_DIR"

    # Create subdirectory for each model
    MODEL_PATH="$BASE_MODEL_PATH/$model/$model.pth"
    
    echo "Destination directory: $DESTINATION_DIR"
    echo "Base model path: $MODEL_PATH"
    echo "Recalibrated model path: $RECALIBRATED_MODELS"
    echo "Method: $method"
    echo "Image folder: $IMAGE_FOLDER"
    echo "Class names: $CLASS_NAMES"
    echo "Output Excel: $DESTINATION_DIR/predictions_${model}.xlsx"
    
    if [[ "$METHOD" == "lss" ]]; then
        EXTRA_ARGS="--state_of_art_method 'lss' "
    else
        EXTRA_ARGS=""
    fi
    

    # Run predictions with recalibrated models
    echo "Running predictions for $model..."
    python predict_data.py \
       --base_model_path "$MODEL_PATH" \
       --recalibrated_model_path "$RECALIBRATED_MODELS" \
       --class_names "$CLASS_NAMES" \
       --image_folder "$IMAGE_FOLDER" \
       --dest_dir "$DESTINATION_DIR" \
       --output_excel "$DESTINATION_DIR/predictions_${model}.xlsx" \
       $EXTRA_ARGS

    echo "Completed processing for model: $model"
    done
done

echo "All models processed successfully!"
