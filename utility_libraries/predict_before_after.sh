#!/bin/bash

# Configuration variables
IMAGE_FOLDER="/home/datasets/valid"
CLASS_NAMES="deer,horse,zebra"
DESTINATION_DIR="/mnt/sdc/sensitivity_try2/predictions_valid_1"
BASE_MODEL_PATH="/mnt/sdd/basics/balanced_training/"

# Array of models
#MODELS=("vgg16" "inception_v3" "resnet50" "mobilenet_v3_small" "mobilenet_v3_large")
#MODELS=("vgg16")
#MODELS=("inception_v3")
#MODELS=("resnet50") 
#MODELS=("mobilenet_v3_small")
MODELS=("mobilenet_v3_large")
# Array of recalibrated models (one for each model)
RECALIBRATED_MODELS=(

    "/mnt/sdc/try2/mobilenet_v3_large/loss_mobilenet_v3_large_features.6.block.2.fc2_0.5.pth"
    "/mnt/sdc/try2/inception_v3/loss_inception_v3_Mixed_6e.branch1x1.conv_0.5.pth"
    "/mnt/sdc/try2/inception_v3/loss_inception_v3_Mixed_6b.branch7x7dbl_4.conv_0.6.pth"
    "/mnt/sdc/try2/mobilenet_v3_small/loss_mobilenet_v3_small_features.8.block.1.0_0.6.pth"
    
    "/mnt/sdc/try2/mobilenet_v3_large/loss_mobilenet_v3_large_features.11.block.2.fc2_0.5.pth"

    "/mnt/sdc/try2/vgg16/loss_vgg16_features.12_0.6.pth"
    "/mnt/sdc/try2/inception_v3/loss_inception_v3_Mixed_6b.branch7x7dbl_4.conv_0.6.pth"
    "/mnt/sdc/try2/resnet50/loss_resnet50_layer4.0.conv1_0.5.pth"
    "/mnt/sdc/try2/mobilenet_v3_small/loss_mobilenet_v3_small_features.6.block.2.fc2_0.5.pth"

    
    
    
     
    
    )

# Iterate over each model
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    recalibrated_model="${RECALIBRATED_MODELS[$i]}"
 
    echo "Processing model: $model"
    
    # Create subdirectory for each model
    MODEL_PATH="$BASE_MODEL_PATH/$model/$model.pth"
    MODEL_DIR="$DESTINATION_DIR/$model"
    mkdir -p "$MODEL_DIR"
    
    # Run predictions before recalibration
    echo "Running predictions for $model (before recalibration)..."
    python predict_data.py \
        --base_model_path "$MODEL_PATH" \
        --recalibrated_model_path "$recalibrated_model" \
        --class_names "$CLASS_NAMES" \
        --image_folder "$IMAGE_FOLDER" \
        --dest_dir "$MODEL_DIR" \
        --output_excel "$MODEL_DIR/predictions_${model}.xlsx"
    
    
    echo "Completed processing for model: $model"
done

echo "All models processed successfully!"
