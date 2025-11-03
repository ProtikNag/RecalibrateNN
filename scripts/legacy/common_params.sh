####################################################################################################################
#  Commom variables
# ####################################################################################################################

BASE_MODEL_DIR="/mnt/sdd/basics/base_models"
CONFIG_FILE='../../config/legacy/config_legacy_3classes.yaml'
MODELS_ARRAY=("vgg16" "mobilenet_v3_small" "mobilenet_v3_large" "resnet50")

#Recalibration Results location
RECALIBRATED_RESULTS_LOCATION="/tmp/legacy_full"

#Sensitivity and XAI Results location 
RECALIBRATED_MODELS_BASE="/mnt/sdd/basics"
SENSITIVITY_RESULTS_LOCATION="/tmp/sensitivity"
XAI_RESULTS_LOCATION="/tmp/legacy_full"

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
