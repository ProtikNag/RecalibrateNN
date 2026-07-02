#!/bin/bash

# Configuration variables
#IMAGE_FOLDER="/home/biased_dataset/baseline_data/deer/biased_data/biased_dataset/valid"
IMAGE_FOLDER="/home/biased_dataset/baseline_data/deer/biased_data/biased_valid_dataset/valid"
CLASS_NAMES="deer_neverseen,horse,zebra"
DESTINATION_DIR="/mnt/sdd/biased_models/biased_prediction/before_after"
BASE_MODEL_PATH="/mnt/sdd/biased_models/base_model/"

# Array of models
#MODELS=("vgg16" "inception_v3" "resnet50" "mobilenet_v3_small" "mobilenet_v3_large")
#MODELS=("vgg16")
#MODELS=("inception_v3")
#MODELS=("resnet50") 
#MODELS=("mobilenet_v3_small")
MODELS=("mobilenet_v3_large")
# Recalibrated models (comma-separated)
: << 'COMMENT'
RECALIBRATED_MODELS="\
/mnt/sdd/biased_models/recalib_selected/vgg16/model0_combo_0.pth,\
/mnt/sdd/biased_models/recalib_selected/vgg16/model0_combo_1.pth,\
/mnt/sdd/biased_models/recalib_selected/vgg16/model0_combo_2.pth,\
/mnt/sdd/biased_models/recalib_selected/vgg16/model0_combo_3.pth,\
/mnt/sdd/biased_models/recalib_selected/vgg16/model0_combo_4.pth,\
/mnt/sdd/biased_models/recalib_selected/vgg16/model0_combo_5.pth,\
/mnt/sdd/biased_models/recalib_selected/vgg16/model0_combo_6.pth"


#RESNET50
RECALIBRATED_MODELS="\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_0.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_1.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_2.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_3.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_4.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_5.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_6.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_7.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_8.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_9.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_10.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_11.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_12.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_13.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_14.pth,\
/mnt/sdd/biased_models/recalib_selected/resnet50/model0_combo_15.pth"


RECALIBRATED_MODELS="\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_small/model0_combo_0.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_small/model0_combo_1.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_small/model0_combo_2.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_small/model0_combo_3.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_small/model0_combo_10.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_small/model0_combo_11.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_small/model0_combo_12.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_small/model0_combo_13.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_small/model0_combo_14.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_small/model0_combo_15.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_small/model0_combo_20.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_small/model0_combo_21.pth "
COMMENT

RECALIBRATED_MODELS="\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_large/model0_combo_0.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_large/model0_combo_1.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_large/model0_combo_2.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_large/model0_combo_3.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_large/model0_combo_10.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_large/model0_combo_11.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_large/model0_combo_12.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_large/model0_combo_13.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_large/model0_combo_14.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_large/model0_combo_15.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_large/model0_combo_20.pth,\
/mnt/sdd/biased_models/recalib_selected/mobilenet_v3_large/model0_combo_21.pth "

# Process model
model="${MODELS[0]}"

echo "Processing model: $model"
echo "Recalibrated models: $RECALIBRATED_MODELS"

# Create subdirectory for each model
MODEL_PATH="$BASE_MODEL_PATH/$model/$model.pth"
MODEL_DIR="$DESTINATION_DIR/$model"
mkdir -p "$MODEL_DIR"

# Run predictions with recalibrated models
echo "Running predictions for $model..."
python predict_data.py \
    --base_model_path "$MODEL_PATH" \
    --recalibrated_model_path "$RECALIBRATED_MODELS" \
    --class_names "$CLASS_NAMES" \
    --image_folder "$IMAGE_FOLDER" \
    --dest_dir "$MODEL_DIR" \
    --output_excel "$MODEL_DIR/predictions_${model}.xlsx"

echo "Completed processing for model: $model"
echo "All models processed successfully!"
