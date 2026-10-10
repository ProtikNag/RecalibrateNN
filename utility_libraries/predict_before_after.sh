#!/bin/bash

# Configuration variables
IMAGE_FOLDER="/home/datasets/valid"
CLASS_NAMES="deer,horse,zebra"
DESTINATION_DIR="/mnt/sdc/try3/predictions"
BASE_MODEL_PATH="/mnt/sdd/basics/balanced_training"

# Array of models
#MODELS=("vgg16" "inception_v3" "resnet50" "mobilenet_v3_small" "mobilenet_v3_large")
MODELS=("vgg16")
#MODELS=("inception_v3")
#MODELS=("resnet50") 
#MODELS=("mobilenet_v3_small")
#MODELS=("mobilenet_v3_large")
# Array of recalibrated models (one for each model)

RECALIBRATED_MODELS=(
  "/mnt/sdc/try3/inception_v3/loss_inception_v3_Mixed_6b.branch7x7_3.conv_0.0.pth"
  "/mnt/sdc/try3/inception_v3/loss_inception_v3_Mixed_6b.branch7x7_3.conv_0.4.pth"
  "/mnt/sdc/try3/inception_v3/loss_inception_v3_Mixed_6b.branch7x7_3.conv_0.5.pth"
  "/mnt/sdc/try3/inception_v3/loss_inception_v3_Mixed_6b.branch7x7_3.conv_0.6.pth"
  "/mnt/sdc/try3/inception_v3/loss_inception_v3_Mixed_6b.branch7x7_3.conv_0.7.pth"
  "/mnt/sdc/try3/inception_v3/loss_inception_v3_Mixed_6b.branch7x7_3.conv_0.8.pth"
  
  "/mnt/sdc/try3/inception_v3/loss_inception_v3_Mixed_6b.branch7x7dbl_1.conv_0.0.pth"
  "/mnt/sdc/try3/inception_v3/loss_inception_v3_Mixed_6b.branch7x7dbl_1.conv_0.4.pth"
  "/mnt/sdc/try3/inception_v3/loss_inception_v3_Mixed_6b.branch7x7dbl_1.conv_0.5.pth"
  "/mnt/sdc/try3/inception_v3/loss_inception_v3_Mixed_6b.branch7x7dbl_1.conv_0.6.pth"
  "/mnt/sdc/try3/inception_v3/loss_inception_v3_Mixed_6b.branch7x7dbl_1.conv_0.7.pth"  
  "/mnt/sdc/try3/inception_v3/loss_inception_v3_Mixed_6b.branch7x7dbl_1.conv_0.8.pth"
  
)


RECALIBRATED_MODELS_RNET50=(
  "/mnt/sdc/try3/resnet50/loss_resnet50_layer4.1.conv1_0.0.pth",
  "/mnt/sdc/try3/resnet50/loss_resnet50_layer4.1.conv1_0.4.pth",
  "/mnt/sdc/try3/resnet50/loss_resnet50_layer4.1.conv1_0.5.pth",
  "/mnt/sdc/try3/resnet50/loss_resnet50_layer4.1.conv1_0.6.pth",
  "/mnt/sdc/try3/resnet50/loss_resnet50_layer4.1.conv1_0.7.pth",
  "/mnt/sdc/try3/resnet50/loss_resnet50_layer4.1.conv1_0.8.pth",       
  "/mnt/sdc/try3/resnet50/loss_resnet50_layer4.2.conv2_0.0.pth",
  "/mnt/sdc/try3/resnet50/loss_resnet50_layer4.2.conv2_0.4.pth",
  "/mnt/sdc/try3/resnet50/loss_resnet50_layer4.2.conv2_0.5.pth",
  "/mnt/sdc/try3/resnet50/loss_resnet50_layer4.2.conv2_0.6.pth",
  "/mnt/sdc/try3/resnet50/loss_resnet50_layer4.2.conv2_0.7.pth",
  "/mnt/sdc/try3/resnet50/loss_resnet50_layer4.2.conv2_0.8.pth"
)


RECALIBRATED_MODELS_VGG16=(

     "/mnt/sdc/try3/vgg16/loss_vgg16_features.10_0.0.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.10_0.4.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.10_0.5.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.10_0.6.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.10_0.7.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.10_0.8.pth",

     "/mnt/sdc/try3/vgg16/loss_vgg16_features.12_0.0.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.12_0.4.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.12_0.5.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.12_0.6.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.12_0.7.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.12_0.8.pth",

     "/mnt/sdc/try3/vgg16/loss_vgg16_features.14_0.0.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.14_0.4.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.14_0.5.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.14_0.6.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.14_0.7.pth",
     "/mnt/sdc/try3/vgg16/loss_vgg16_features.14_0.8.pth"
    
    )

# Iterate over each model
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"


    echo "Processing model: $model"
    
    # Create subdirectory for each model
    MODEL_PATH="$BASE_MODEL_PATH/$model/$model.pth"
    MODEL_DIR="$DESTINATION_DIR/$model"
    echo "Base model path: $MODEL_PATH"
    mkdir -p "$MODEL_DIR"
    RECALIBRATION_DIR="/mnt/sdc/try3/${model}"    
    # Run predictions before recalibration
    echo "Running predictions for $model (before recalibration)..."
    echo " python predict_before_after_all.py \
        --base_model_path "$MODEL_PATH" \
        --recalibrated_model_directory ${RECALIBRATION_DIR} \
        --class_names '$CLASS_NAMES' \
        --image_folder '$IMAGE_FOLDER' \
        --dest_dir '$MODEL_DIR' \
        --output_excel '$MODEL_DIR/predictions_${model}.xlsx'
    "
    
    python predict_before_after_all.py \
        --base_model_path "$MODEL_PATH" \
        --recalibrated_model_directory ${RECALIBRATION_DIR} \
        --class_names "$CLASS_NAMES" \
        --image_folder "$IMAGE_FOLDER" \
        --dest_dir "$MODEL_DIR" \
        --output_excel "$MODEL_DIR/predictions_${model}.xlsx"
    
    
    echo "Completed processing for model: $model"
done

echo "All models processed successfully!"
