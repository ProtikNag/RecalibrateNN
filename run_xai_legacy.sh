#!/bin/sh
#VGG16 example


#Legacy
#--------------------------------------   VGG16  --------------------------------------
export BASEPATH_RESULTS='/mnt/sdd/basics/'
export LAYER='features.21'
export LAMBDA='0.6'
#VGG16
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/vgg16/vgg16.pth"
export MODEL_NAME="vgg16"
export MODIFIED_MODEL="$BASEPATH_RESULTS/$MODEL_NAME/loss_${MODEL_NAME}_${LAYER}_${LAMBDA}.pth"
export CONFIG_FILE='./config_legacy_3classes.yaml'
export SAVE_DIR="./xai_images_${MODEL_NAME}_${LAYER}_${LAMBDA}"
mkdir -p ${SAVE_DIR}


python xai_visualization.py --org_model_path "${ORIGINAL_MODEL}" --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE} --save_dir ${SAVE_DIR}

#--------------------------------------   Incepion_V3  --------------------------------------
export BASEPATH_RESULTS='/mnt/sdd/basics/'
export LAYER='Mixed_6e.branch1x1.conv'
export LAMBDA='0.5'
export MODEL_NAME="inception_v3"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/${MODEL_NAME}/${MODEL_NAME}.pth"

export MODIFIED_MODEL="$BASEPATH_RESULTS/$MODEL_NAME/loss_${MODEL_NAME}_${LAYER}_${LAMBDA}.pth"
export CONFIG_FILE='./config_legacy_3classes.yaml'
export SAVE_DIR="./xai_images_${MODEL_NAME}_${LAYER}_${LAMBDA}"
mkdir -p ${SAVE_DIR}

python xai_visualization.py --org_model_path "${ORIGINAL_MODEL}" --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE} --save_dir ${SAVE_DIR}
