#!/bin/bash
MODELS=("inception_v3" "resnet50" "vgg16"  "mobilenet_v3_small" "mobilenet_v3_large")
METHODS=("group_dro" "erm" "jtt" "lss")
#METHODS=("lss")

XAI_METHOD=("gradcam" "integrated_gradients")
CONFIG_FILE="/home/srikanth/study1/RecalibrateNN/config/biased/config_biased_3classes.yaml"


for method in "${METHODS[@]}"
do
    for xai_method in "${XAI_METHOD[@]}"
    do
        for model in "${MODELS[@]}"
        do
            SAVE_DIR="/mnt/sdd/biased_models/biased_prediction/SOA/${method}/XAI/${model}/${xai_method}"
            if [[ "$method" == "lss" ]]; then
                EXTRA_ARGS="--method lss"
                MODEL_PATH="/mnt/sdd/biased_models/base_model/state_of_art/${method}/${model}/${model}_full.pth"
            else
                EXTRA_ARGS=""
                MODEL_PATH="/mnt/sdd/biased_models/base_model/state_of_art/${method}/${model}/${model}.pth"
            fi
  
            mkdir -p /mnt/sdd/biased_models/biased_prediction/SOA/${method}/XAI/${model}/$xai_method 
            echo "python xai_after.py  --model_name ${model} --config_file ${CONFIG_FILE} --save_dir ${SAVE_DIR} --override_xai ${xai_method} \
            --modified_model_path ${MODEL_PATH} --before_after  ${EXTRA_ARGS}"
            python xai_after.py  --model_name ${model} --config_file ${CONFIG_FILE} --save_dir ${SAVE_DIR} --override_xai ${xai_method} \
            --modified_model_path ${MODEL_PATH} --before_after  ${EXTRA_ARGS}
        done
    done
done    

