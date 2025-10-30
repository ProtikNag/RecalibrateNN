####################################################################################################################
#  Commom variables
# ####################################################################################################################

BASE_MODEL_DIR="/home/srikanth/trained_models/pytorch/legacy"
CONFIG_FILE='../../config_legacy_3classes.yaml'
RECALIBRATED_MODELS_BASE="/mnt/sdd/basics"
RESULTS_BASE="/tmp/legacy_full"
MODELS_ARRAY=("vgg16" "mobilenet_v3_small" "mobilenet_v3_large" "resnet50") 
SENSITIVITY_RESULTS_LOCATION="/tmp/sensitivity"
####################################################################################################################

print_parameters() {
    echo "======================================================"
    echo -e "\033[36mCONFIGURABLE PARAMETERS:\033[0m"
    echo "======================================================"
    echo -e "\033[32mPython Script:\033[0m $PYTHON_SCRIPT"
    echo -e "\033[33mBase Model Directory:\033[0m $BASE_MODEL_DIR"
    echo -e "\033[32mConfig File:\033[0m $CONFIG_FILE"
    echo -e "\033[33mRecalibrated Models Base:\033[0m $RECALIBRATED_MODELS_BASE"
    echo -e "\033[32mResults Base Directory:\033[0m $RESULTS_BASE"
    echo -e "\033[33mModels to Process:\033[0m ${MODELS_ARRAY[*]}"
    echo -e "\033[32mSensitivity results stored at :\033[0m $SENSITIVITY_RESULTS_LOCATION"
    echo "======================================================"
}
