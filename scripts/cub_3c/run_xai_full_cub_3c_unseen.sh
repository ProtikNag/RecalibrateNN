#!/usr/bin/env bash
set -euo pipefail

#    parser.add_argument("--org_model_path", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--modified_model_path", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--config_file", type=str, default=None, help="Configuration file")
#    parser.add_argument("--save_dir", type=str, default=None, help="Specify a save directory to save the results")

export BASE_PATH_RES="/mnt/sdb2/cub_3c_full/"
export CONFIG_FILE='../../config_cub_3classes.yaml'
export RESULTS="/mnt/sdb2/legacy_unseen/cub_3c/"


export MODEL_NAME="vgg16"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech_3class/${MODEL_NAME}/${MODEL_NAME}.pth"
export MODEFIED_MODEL="/mnt/sdd/cub_3c/${MODEL_NAME}/loss_${MODEL_NAME}_features.24_0.5.pth"
python ../../xai_visualize_all.py --override_image_path  --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODEFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE}  --save_dir ${RESULTS}/${MODEL_NAME}}



export MODEL_NAME="mobilenet_v3_small"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech_3class/${MODEL_NAME}/${MODEL_NAME}.pth"
export RESULTS="${BASE_PATH_RES}${MODEL_NAME}"
export MODEFIED_MODEL="/mnt/sdd/cub_3c/${MODEL_NAME}/loss_${MODEL_NAME}_features.7.block.3.0_0.5.pth"
python ../../xai_visualize_all.py --override_image_path --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODEFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE}  --save_dir ${RESULTS}/${MODEL_NAME}

export MODEL_NAME="mobilenet_v3_large"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech_3class/${MODEL_NAME}/${MODEL_NAME}.pth"
export MODEFIED_MODEL="/mnt/sdd/cub_3c/${MODEL_NAME}/loss_${MODEL_NAME}_features.13.block.2.fc2_0.5.pth"
python ../../xai_visualize_all.py  --override_image_path --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODEFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE}  --save_dir ${RESULTS}/${MODEL_NAME}


export MODEL_NAME="resnet50"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech_3class/${MODEL_NAME}/${MODEL_NAME}.pth"
export MODEFIED_MODEL="/mnt/sdd/cub_3c/${MODEL_NAME}/loss_${MODEL_NAME}_layer4.2.conv2_0.5.pth"
python ../../xai_visualize_all.py  --override_image_path --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODEFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE}  --save_dir ${RESULTS}/${MODEL_NAME}


export MODEL_NAME="inception_v3"
export ORIGINAL_MODEL="/home/srikanth/trained_models/pytorch/caltech_3class/${MODEL_NAME}/${MODEL_NAME}.pth"
export MODEFIED_MODEL="/mnt/sdd/cub_3c/${MODEL_NAME}/loss_${MODEL_NAME}_Mixed_6e.branch7x7_1.conv_0.5.pth"
python ../../xai_visualize_all.py  --override_image_path --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODEFIED_MODEL}  --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE}  --save_dir ${RESULTS}/${MODEL_NAME}
