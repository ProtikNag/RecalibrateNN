#!/usr/bin/env bash
set -euo pipefail

export PATH=../../../:${PATH}

### All parameters over written here
MODELS_ARRAY=("vgg16" "resnet50" "inception_v3" "mobilenet_v3_small" "mobilenet_v3_large") 
BASE_MODEL_DIR="/mnt/sdd/biased_models"
CONFIG_FILE="/home/srikanth/study1/RecalibrateNN/config/biased/config_biased_3classes.yaml"
# Call the function to display parameters and get confirmation
BEFORE_AFTER=false
echo -e "\033[32mRecalibration Before and after Flag:\033[0m $BEFORE_AFTER"

RECALIBRATED_MODELS_BASE="/mnt/sdd/biased_models/recalib_results"
RECALIBRATED_RESULTS_LOCATION="/mnt/sdd/biased_models/recalib_results"
SENSITIVITY_RESULTS_LOCATION="/mnt/sdd/biased_models/sensitivity"
XAI_RESULTS_LOCATION="/mnt/sdd/biased_models/xai_results"


####################################################################################################################

print_parameters() {
    echo "======================================================"
    echo -e "\033[36mCONFIGURABLE PARAMETERS:\033[0m"
    echo "======================================================"
    echo -e "\033[33mBase Model Directory:\033[0m $BASE_MODEL_DIR"
    echo -e "\033[32mConfig File:\033[0m $CONFIG_FILE"
    echo -e "\033[33mModels to Process:\033[0m ${MODELS_ARRAY[*]}"
    echo

    echo "======================================================"
    echo -e "\033[36mRecalibration related data:\033[0m"
    echo "======================================================"
    echo -e "\033[33mRecalibrated Models Results Location:\033[0m $RECALIBRATED_RESULTS_LOCATION"
    echo

    echo "======================================================"
    echo -e "\033[36mSensitivity related data:\033[0m"
    echo "======================================================"
    echo -e "\033[33mSensitivity Results Location:\033[0m $SENSITIVITY_RESULTS_LOCATION"
    echo -e "\033[33mRecalibrated Models Base:\033[0m $RECALIBRATED_MODELS_BASE"

    echo


    echo "======================================================"
    echo -e "\033[36mXAI  related data:\033[0m"
    echo "======================================================"
    echo -e "\033[33mXAI Results Location:\033[0m $XAI_RESULTS_LOCATION"
    echo
    echo "======================================================"

}
