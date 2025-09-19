#!/bin/sh
#VGG16 example

#export Platform="Srikanth"
#ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/vgg16/vgg16.pth"
#MODIFIED_MODEL="/mnt/sdd/results/loss_vgg16_features.12_0.5.pth"
#MODEL_NAME="vgg16"

#CALTECH
export PLATFORM="CUB"
ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech/vgg16/vgg16.pt"
MODIFIED_MODEL="/mnt/sdd/caltech/results/vgg16//loss_vgg16_features.12_0.5.pth"
MODEL_NAME="vgg16"


SAVE_DIR="./xai_images/"

python xai_visualization.py --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --save_dir ${SAVE_DIR}
