#!/usr/bin/env bash
set -euo pipefail

#    parser.add_argument("--org_model_path", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--modified_model_path", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--config_file", type=str, default=None, help="Configuration file")
#    parser.add_argument("--save_dir", type=str, default=None, help="Specify a save directory to save the results")


####################################################################################################################

source ./common_params.sh
source ../utility_scripts.sh
PYTHON_SCRIPT="../../xai_visualize_all.py"

VGG_XAI=false
RNET_XAI=false
IV3_XAI=true
MET_S_XAI=false
MET_L_XAI=false


# Call the function to display parameters and get confirmation
print_parameters

echo -e "\033[32m XAI FOR VGG16 \033[0m $VGG_XAI"
echo -e "\033[32m XAI FOR RNET \033[0m $RNET_XAI"
echo -e "\033[32m XAI FOR IV3 \033[0m $IV3_XAI"
echo -e "\033[32m XAI FOR MNET_S \033[0m $MET_S_XAI"
echo -e "\033[32m XAI FOR MNET_L \033[0m $MET_L_XAI"

wait_for_user

####################################################################################################################
#process_xai_selected models
if [ "$VGG_XAI" = true ]; then
    MODEL="vgg16"
    RECALIBRATED_MODELS=( "/mnt/sdc/try2/vgg16/loss_vgg16_features.12_0.6.pth"
          
                        )
  # Takes 2 parameters Model name and pth file array
  process_xai_for_selected_model "${MODEL}" "${RECALIBRATED_MODELS[@]}"

fi
################ DO NOT EDIT BEYOND THIS POINT #####################################################################
################ DO NOT EDIT BEYOND THIS POINT #####################################################################

#process_xai_selected models
if [ "$RNET_XAI" = true ]; then
    MODEL="resnet50"
    RECALIBRATED_MODELS=("/mnt/sdc/try2/resnet50/loss_resnet50_layer4.0.conv1_0.5.pth"
          
                        )
  # Takes 2 parameters Model name and pth file array
  process_xai_for_selected_model "${MODEL}" "${RECALIBRATED_MODELS[@]}"

fi
################ DO NOT EDIT BEYOND THIS POINT #####################################################################
################ DO NOT EDIT BEYOND THIS POINT #####################################################################



#process_xai_selected models
if [ "$IV3_XAI" = true ]; then
    MODEL="inception_v3"
    RECALIBRATED_MODELS=("/mnt/sdc/try2/inception_v3/loss_inception_v3_Mixed_6b.branch7x7_1.conv_0.6.pth"
          
                        )
  # Takes 2 parameters Model name and pth file array
  process_xai_for_selected_model "${MODEL}" "${RECALIBRATED_MODELS[@]}"

fi
################ DO NOT EDIT BEYOND THIS POINT #####################################################################
################ DO NOT EDIT BEYOND THIS POINT #####################################################################


#process_xai_selected models
if [ "$MET_S_XAI" = true ]; then
    MODEL="mobilenet_v3_small"
    #RECALIBRATED_MODELS=("/mnt/sdc/try2/mobilenet_v3_small/loss_mobilenet_v3_small_features.6.block.2.fc2_0.5.pth"
    RECALIBRATED_MODELS=("/mnt/sdc/try2/mobilenet_v3_small/loss_mobilenet_v3_small_features.8.block.1.0_0.6.pth"      
                        )
  # Takes 2 parameters Model name and pth file array
  process_xai_for_selected_model "${MODEL}" "${RECALIBRATED_MODELS[@]}"

fi
################ DO NOT EDIT BEYOND THIS POINT #####################################################################
################ DO NOT EDIT BEYOND THIS POINT #####################################################################


#process_xai_selected models
if [ "$MET_L_XAI" = true ]; then
    MODEL="mobilenet_v3_large"
    RECALIBRATED_MODELS=("/mnt/sdc/try2/mobilenet_v3_large/loss_mobilenet_v3_large_features.11.block.2.fc2_0.5.pth"
          
                        )
  # Takes 2 parameters Model name and pth file array
  process_xai_for_selected_model "${MODEL}" "${RECALIBRATED_MODELS[@]}"

fi
################ DO NOT EDIT BEYOND THIS POINT #####################################################################
################ DO NOT EDIT BEYOND THIS POINT #####################################################################



################ End of script #####################################################################################



