#!/usr/bin/env bash
set -euo pipefail

#    parser.add_argument("--org_model_path", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--modified_model_path", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--config_file", type=str, default=None, help="Configuration file")
#    parser.add_argument("--save_dir", type=str, default=None, help="Specify a save directory to save the results")


####################################################################################################################
#  Commom variables
# ####################################################################################################################
PYTHON_SCRIPT="../../xai_visualize_all.py"
BASE_MODEL_DIR="/home/srikanth/trained_models/pytorch/legacy"
CONFIG_FILE='../../config_legacy_3classes.yaml'
RECALIBRATED_MODELS_BASE="/mnt/sdd/basics"
RESULTS_BASE="/tmp/legacy_full"
MODELS_ARRAY=("vgg16" "mobilenet_v3_small" "mobilenet_v3_large" "resnet50") 
####################################################################################################################
VGG_XAI=false
RNET_XAI=false
IV3_XAI=true
MET_S_XAI=true
MET_L_XAI=true

source ../utility_scripts.sh
# Call the function to display parameters and get confirmation
print_parameters

#process_xai_selected models
if [ "$VGG_XAI" = true ]; then
    MODEL="vgg16"
    RECALIBRATED_MODELS=("loss_vgg16_features.14_1.0.pth"
                         "loss_vgg16_features.14_0.6.pth"
                         "loss_vgg16_features.21_0.2.pth"
                         "loss_vgg16_features.12_0.6.pth"
                         "loss_vgg16_features.26_0.4.pth"
                         "loss_vgg16_features.17_0.2.pth"
                         "loss_vgg16_features.7_0.4.pth"
                         "loss_vgg16_features.14_0.3.pth")
fi
################ DO NOT EDIT BEYOND THIS POINT ##############################
  # Takes 2 parameters Model name and pth file array
  process_xai_for_selected_model "${MODEL}" "${RECALIBRATED_MODELS[@]}"
################ DO NOT EDIT BEYOND THIS POINT ##############################

#process_xai_selected models
if [ "$RNET_XAI" = true ]; then
    MODEL="resnet50"
    RECALIBRATED_MODELS=("loss_resnet50_layer4.1.conv1_0.5.pth"
                         "loss_resnet50_layer4.1.conv1_0.6.pth"
                         "loss_resnet50_layer4.2.conv3_0.6.pth"
                         "loss_resnet50_layer2.3.conv1_0.6.pth")
fi
################ DO NOT EDIT BEYOND THIS POINT ##############################
  # Takes 2 parameters Model name and pth file array
  process_xai_for_selected_model "${MODEL}" "${RECALIBRATED_MODELS[@]}"
################ DO NOT EDIT BEYOND THIS POINT ##############################



#process_xai_selected models
if [ "$IV3_XAI" = true ]; then
    MODEL="inception_v3"
    RECALIBRATED_MODELS=("loss_inception_v3_Mixed_6e.branch1x1.conv_0.5.pth"
                         "loss_inception_v3_Mixed_6e.branch1x1.conv_0.6.pth"
                         "loss_inception_v3_Mixed_6e.branch7x7_1.conv_0.6.pth"
                         "loss_inception_v3_Mixed_6e.branch7x7_1.conv_0.7.pth")
fi
################ DO NOT EDIT BEYOND THIS POINT ##############################
  # Takes 2 parameters Model name and pth file array
  process_xai_for_selected_model "${MODEL}" "${RECALIBRATED_MODELS[@]}"
################ DO NOT EDIT BEYOND THIS POINT ##############################


#process_xai_selected models
if [ "$MET_S_XAI" = true ]; then
    MODEL="mobilenet_v3_small"
    RECALIBRATED_MODELS=("loss_mobilenet_v3_small_features.10.block.3.0_0.5.pth"
                         "loss_mobilenet_v3_small_features.10.block.3.0_0.7.pth"
                         "loss_mobilenet_v3_small_features.8.block.3.0_0.7.pth"
                         "loss_mobilenet_v3_small_features.8.block.0.0_0.5.pth")
fi
################ DO NOT EDIT BEYOND THIS POINT ##############################
  # Takes 2 parameters Model name and pth file array
  process_xai_for_selected_model "${MODEL}" "${RECALIBRATED_MODELS[@]}"
################ DO NOT EDIT BEYOND THIS POINT ##############################


#process_xai_selected models
if [ "$MET_L_XAI" = true ]; then
    MODEL="mobilenet_v3_large"
    RECALIBRATED_MODELS=("loss_mobilenet_v3_large_features.11.block.2.fc1_0.5.pth "
                         "loss_mobilenet_v3_large_features.14.block.2.fc1_0.5.pth"
                         "loss_mobilenet_v3_large_features.15.block.2.fc2_0.6.pth"
                         "loss_mobilenet_v3_large_features.16.0_0.6.pth")
fi
################ DO NOT EDIT BEYOND THIS POINT ##############################
  # Takes 2 parameters Model name and pth file array
  process_xai_for_selected_model "${MODEL}" "${RECALIBRATED_MODELS[@]}"
################ DO NOT EDIT BEYOND THIS POINT ##############################



################ End of script ##############################


################ End of script ##############################
