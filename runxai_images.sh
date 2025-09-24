#!/bin/sh
#VGG16 example

#export Platform="Srikanth"
#ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/vgg16/vgg16.pth"
#MODIFIED_MODEL="/mnt/sdd/results/loss_vgg16_features.12_0.5.pth"
#MODEL_NAME="vgg16"

#CALTECH

export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech/vgg16/vgg16.pth"
export MODIFIED_MODEL="./reslts/loss_vgg16_features.5_0.5.pth"
export MODEL_NAME="vgg16"
export CONFIG_FILE='./config_cub_3classes.yaml'

export SAVE_DIR="./xai_images/"

python xai_visualization.py --org_model_path "${ORIGINAL_MODEL}" --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE} --save_dir ${SAVE_DIR}


#python xai_visualization.py --org_model_path /home/srikanth/trained_models/pytorch/caltech/vgg16/vgg16.pt --modified_model_path ./reslts/loss_vgg16_features.5_0.5.pth  --model_name vgg16 --config_file './config_cub_3classes.yaml' --save_dir ./xai_images/
