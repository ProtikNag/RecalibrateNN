#!/usr/bin/env bash
set -euo pipefail

#    parser.add_argument("--org_model_path", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--modified_model_path", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--config_file", type=str, default=None, help="Configuration file")
#    parser.add_argument("--save_dir", type=str, default=None, help="Specify a save directory to save the results")


export MODEL_NAME="vgg16"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/legacy/${MODEL_NAME}/${MODEL_NAME}.pth"
export CONFIG_FILE='../../config_legacy_3classes.yaml'
export RESULTS="/mnt/sdb2/legacy_unseen/${MODEL_NAME}"
export MODIFIED_MODEL="/mnt/sdd/basics/vgg16/loss_vgg16_features.24_0.5.pth"
python ../../xai_visualize_all.py --override_image_path  --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE}  --save_dir ${RESULTS}


export MODEL_NAME="mobilenet_v3_small"
export RESULTS="/mnt/sdb2/legacy_unseen/${MODEL_NAME}"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/legacy/${MODEL_NAME}/${MODEL_NAME}.pth"
export MODIFIED_MODEL="/mnt/sdd/basics/${MODEL_NAME}/loss_${MODEL_NAME}_features.7.block.3.0_0.5.pth"

python ../../xai_visualize_all.py --override_image_path --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE}  --save_dir ${RESULTS}

export MODEL_NAME="mobilenet_v3_large"
export RESULTS="/mnt/sdb2/legacy_unseen/${MODEL_NAME}"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/legacy/${MODEL_NAME}/${MODEL_NAME}.pth"
export MODIFIED_MODEL="/mnt/sdd/basics/${MODEL_NAME}/loss_${MODEL_NAME}_features.9.block.2.0_0.5.pth"
python ../../xai_visualize_all.py --override_image_path --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE}  --save_dir ${RESULTS}


export MODEL_NAME="resnet50"
export RESULTS="/mnt/sdb2/legacy_unseen/${MODEL_NAME}"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/legacy/${MODEL_NAME}/${MODEL_NAME}.pth"
export MODIFIED_MODEL="/mnt/sdd/basics/${MODEL_NAME}/loss_${MODEL_NAME}_layer3.0.conv3_0.5.pth"
python ../../xai_visualize_all.py --override_image_path --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE}  --save_dir ${RESULTS}


export MODEL_NAME="inception_v3"
export RESULTS="/mnt/sdb2/legacy_unseen/${MODEL_NAME}"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/legacy/${MODEL_NAME}/${MODEL_NAME}.pth"
export MODIFIED_MODEL="/mnt/sdd/basics/${MODEL_NAME}/loss_${MODEL_NAME}_Mixed_6e.branch1x1.conv_0.5.pth"
python ../../xai_visualize_all.py --override_image_path --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODIFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE}  --save_dir ${RESULTS}


