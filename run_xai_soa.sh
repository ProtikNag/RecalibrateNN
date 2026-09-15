#!/bin/bash
models=("inception_v3" "resnet50" "vgg16"  "mobilenet_v3_small" "mobilenet_v3_large")
#methods=("group_dro" "erm" "jtt" "lss")
methods=("lss")

XAI_METHOD=gradcam
CONFIG_FILE="/home/srikanth/study1/RecalibrateNN/config/biased/config_biased_3classes.yaml"

for method in "${methods[@]}"
do
    for model in "${models[@]}"
    do
        SAVE_DIR="/mnt/sdd/biased_models/biased_prediction/SOA/${method}/XAI/${model}/gradcam"
        if [[ "$method" == "lss" ]]; then
            EXTRA_ARGS="--method lss"
            MODEL_PATH="/mnt/sdd/biased_models/base_model/state_of_art/${method}/${model}/${model}_full.pth"
        else
            EXTRA_ARGS=""
            MODEL_PATH="/mnt/sdd/biased_models/base_model/state_of_art/${method}/${model}/${model}.pth"
        fi

        mkdir -p /mnt/sdd/biased_models/biased_prediction/SOA/${method}/XAI/${model}/$XAI_METHOD 
        python xai_after.py  --model_name ${model} --config_file ${CONFIG_FILE} --save_dir ${SAVE_DIR}  \
        --modified_model_path ${MODEL_PATH} --before_after  ${EXTRA_ARGS}
    done
done    
