#!/bin/bash

# Configuration variables
IMAGE_FOLDER="/mnt/sdc/data/caltech/CUB_200_2011/cub_dataset/threeclasses/valid"
CLASS_NAMES="001.Black_footed_Albatross,002.Laysan_Albatross,003.Sooty_Albatross"
DESTINATION_DIR="/mnt/sdc/cub_sensitivity/cub_3c_models/predictions"
BASE_MODEL_PATH="/mnt/sdc/cub_sensitivity/cub_3c_models/cub3c_models/"

# Array of models
#MODELS=("vgg16" "inception_v3" "resnet50" "mobilenet_v3_small" "mobilenet_v3_large")
#MODELS=("vgg16")
#MODELS=("inception_v3")
#MODELS=("resnet50") 
#MODELS=("mobilenet_v3_small")
MODELS=("mobilenet_v3_large")
# Array of recalibrated models (one for each model)
RECALIBRATED_MODELS=(
     "/mnt/sdc/cub_sensitivity/cub_3c_models/recalib/mobilenet_v3_large/loss_mobilenet_v3_large_features.6.block.2.fc2_0.5.pth"

     "/mnt/sdc/cub_sensitivity/cub_3c_models/recalib/mobilenet_v3_small/loss_mobilenet_v3_small_features.6.block.2.fc2_0.5.pth"
     "/mnt/sdc/cub_sensitivity/cub_3c_models/recalib/resnet50/loss_resnet50_layer4.1.conv1_0.5.pth"
     "/mnt/sdc/cub_sensitivity/cub_3c_models/recalib/vgg16/loss_vgg16_features.10_0.5.pth"


    
    )
#     "/mnt/sdc/cub_sensitivity/cub_3c_models/recalib/inception_v3/loss_inception_v3_Mixed_6b.branch7x7_1.conv_0.5.pth"

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
