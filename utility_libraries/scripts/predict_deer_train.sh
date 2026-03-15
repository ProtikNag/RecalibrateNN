#!/bin/bash

CLASS_NAMES="deer,horse,zebra"
MODE="valid"
CONFIG="patched_in"

models=("vgg16" "inception_v3" "resnet50" "mobilenet_v3_small" "mobilenet_v3_large")
BASE_MODEL_PATH="/mnt/sdd/basics/balanced_training"
BASE_ANIMAL="zebra"
BASE_IMAGE_FOLDER="/home/prediction_dataset/${BASE_ANIMAL}"
OUTPUT_DIR_BASE="/mnt/sdc/prediction_results/${CONFIG}"

PATCHED_IMAGE_FOLDERS=(
    "/home/morphed_data/${CONFIG}/morphed_${MODE}_${BASE_ANIMAL}_deer_background"
    "/home/morphed_data/${CONFIG}/morphed_${MODE}_${BASE_ANIMAL}_deer_coat"
    "/home/morphed_data/${CONFIG}/morphed_${MODE}_${BASE_ANIMAL}_deer_face"
    "/home/morphed_data/${CONFIG}/morphed_${MODE}_${BASE_ANIMAL}_deer_legs"
    "/home/morphed_data/${CONFIG}/morphed_${MODE}_${BASE_ANIMAL}_horse_background"
    "/home/morphed_data/${CONFIG}/morphed_${MODE}_${BASE_ANIMAL}_horse_coat"
    "/home/morphed_data/${CONFIG}/morphed_${MODE}_${BASE_ANIMAL}_horse_face"
    "/home/morphed_data/${CONFIG}/morphed_${MODE}_${BASE_ANIMAL}_horse_legs"
)


for PATCHED_IMAGE_FOLDER in "${PATCHED_IMAGE_FOLDERS[@]}"; do
    echo "Processing $PATCHED_IMAGE_FOLDER..."

    # Remove the soft link to the deer folder in train
    rm -rf "/home/prediction_dataset/${BASE_ANIMAL}/${MODE}/${BASE_ANIMAL}"
    ln -sf "${PATCHED_IMAGE_FOLDER}" "/home/prediction_dataset/${BASE_ANIMAL}/${MODE}/${BASE_ANIMAL}"

    # Loop through each model
    for model in "${models[@]}"; do
        output_dir="$OUTPUT_DIR_BASE/$model/$(basename ${PATCHED_IMAGE_FOLDER})"
        mkdir -p "${output_dir}"
        output_excel="${output_dir}/predictions.xlsx"

        echo "Running prediction for $model..."
        python ../predict_data.py \
            --base_model_path "$BASE_MODEL_PATH/$model/$model.pth" \
            --class_names "$CLASS_NAMES" \
            --image_folder "$BASE_IMAGE_FOLDER/${MODE}" \
            --dest_dir "$output_dir" \
            --output_excel "$output_excel"
        mv audit_log.log "${output_dir}/audit_log.log"
        echo "Completed $model"
    done

    echo "All models processed for $PATCHED_IMAGE_FOLDER!"
done

echo "All results saved at $OUTPUT_DIR_BASE"
