#!/bin/sh
#VGG16 example

#export Platform="Srikanth"
#ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/vgg16/vgg16.pth"
#MODIFIED_MODEL="/mnt/sdd/results/loss_vgg16_features.12_0.5.pth"
#MODEL_NAME="vgg16"

#CALTECH
export BASEPATH_RESULTS='./results'

#VGG16
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech/vgg16/vgg16.pth"
export MODEL_NAME="vgg16"
export MODIFIED_MODEL="$BASEPATH_RESULTS/$MODEL_NAME/loss_vgg16_features.21_0.5.pth"
export CONFIG_FILE='./config_cub_3classes.yaml'
export SAVE_DIR="./xai_images"
mkdir -p ${SAVE_DIR}


python xai_visualization.py --org_model_path "${ORIGINAL_MODEL}" --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE} --save_dir ${SAVE_DIR}


#RESNET50
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech/resnet50/resnet50.pth"
export MODEL_NAME="resnet50"
export MODIFIED_MODEL="$BASEPATH_RESULTS/$MODEL_NAME/loss_resnet50_layer4.0.conv1_0.5.pth"
export CONFIG_FILE='./config_cub_3classes.yaml'
export SAVE_DIR="./xai_images"
mkdir -p ${SAVE_DIR}

python xai_visualization.py --org_model_path "${ORIGINAL_MODEL}" --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE} --save_dir ${SAVE_DIR}



#inception_v3
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech/inception_v3/inception_v3.pth"
export MODEL_NAME="inception_v3"
export MODIFIED_MODEL="$BASEPATH_RESULTS/$MODEL_NAME/loss_inception_v3_Mixed_6e.branch1x1_0.6.pth"
export CONFIG_FILE='./config_cub_3classes.yaml'
export SAVE_DIR="./xai_images"
mkdir -p ${SAVE_DIR}

python xai_visualization.py --org_model_path "${ORIGINAL_MODEL}" --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE} --save_dir ${SAVE_DIR}


#mobilenet_v3_large
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech/mobilenet_v3_small/mobilenet_v3_small.pth"
export MODEL_NAME="mobilenet_v3_large"
export MODIFIED_MODEL="$BASEPATH_RESULTS/$MODEL_NAME/loss_mobilenet_v3_large_features.12.block.2.fc1_0.5.pth"
export CONFIG_FILE='./config_cub_3classes.yaml'
export SAVE_DIR="./xai_images"
mkdir -p ${SAVE_DIR}

python xai_visualization.py --org_model_path "${ORIGINAL_MODEL}" --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE} --save_dir ${SAVE_DIR}


#mobilenet_v3_small
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech/mobilenet_v3_small/mobilenet_v3_small.pth"
export MODEL_NAME="mobilenet_v3_small"
export MODIFIED_MODEL="$BASEPATH_RESULTS/$MODEL_NAME/loss_mobilenet_v3_small_features.5.block.0.0_0.5.pth"
export CONFIG_FILE='./config_cub_3classes.yaml'
export SAVE_DIR="./xai_images"
mkdir -p ${SAVE_DIR}

python xai_visualization.py --org_model_path "${ORIGINAL_MODEL}" --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE} --save_dir ${SAVE_DIR}

