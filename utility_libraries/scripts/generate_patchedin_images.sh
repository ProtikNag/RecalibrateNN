#!/bin/bash
SRC_ANIMALS=("deer" "horse" "zebra")
DEST_ANIMALS=("deer" "horse" "zebra")
CONCEPTS=("background" "face" "coat" "legs")
MODES=("train" "valid")
OUTPUT_BASE="/home/morphed_data/patched_in"

for MODE in "${MODES[@]}"; do
    for SRC_ANIMAL in "${SRC_ANIMALS[@]}"; do
        for DEST_ANIMAL in "${DEST_ANIMALS[@]}"; do
            if [ "$SRC_ANIMAL" != "$DEST_ANIMAL" ]; then
                for CONCEPT in "${CONCEPTS[@]}"; do
                    OUTPUT_MORPHED_IMAGES="$OUTPUT_BASE/morphed_${MODE}_${SRC_ANIMAL}_${DEST_ANIMAL}_${CONCEPT}"
                    mkdir -p "${OUTPUT_MORPHED_IMAGES}"
                    ORG_FOLDER="/home/balanced_dataset/${MODE}/${SRC_ANIMAL}"
                    SEG_FOLDER="/home/segmented_data/output_${MODE}_segmented_${SRC_ANIMAL}"
                    BB_DIR="${SEG_FOLDER}/bbox"
                    BG_FOLDER="/mnt/sdc/concepts/concepts_links/concept_150/${DEST_ANIMAL}/${CONCEPT}"
                    echo "Processing ${SRC_ANIMAL} to ${DEST_ANIMAL} for concept ${CONCEPT} in mode ${MODE}"
                    echo "Original image folder: ${ORG_FOLDER}"
                    echo "Segmentation folder: ${SEG_FOLDER}"
                    echo "Bounding box folder: ${BB_DIR}"
                    echo "Background folder: ${BG_FOLDER}"
                    echo "Output folder: ${OUTPUT_MORPHED_IMAGES}"
                    echo "python ../patched_background.py --original-image-dir \"${ORG_FOLDER}\" --bb-dir \"${BB_DIR}\" --concept-image-dir \"${BG_FOLDER}\" --output-dir \"${OUTPUT_MORPHED_IMAGES}\""
                    python ../patched_background.py --original-image-dir "${ORG_FOLDER}" --bb-dir "${BB_DIR}" --concept-image-dir "${BG_FOLDER}" --output-dir "${OUTPUT_MORPHED_IMAGES}"
                done
            fi
        done
    done
done
