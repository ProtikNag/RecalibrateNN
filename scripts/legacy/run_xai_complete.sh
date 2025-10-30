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

####################################################################################################################
# Function to return all .pth files from a given directory
# 
get_pth_files() {
    local dir_path="$1"
    if [ -d "$dir_path" ]; then
        find "$dir_path" -name "*.pth" -type f
    else
        echo "Directory $dir_path does not exist" >&2
        return 1
    fi
}

# Function to process a single pth file for a given model
process_pth_file() {
    local pth_file="$1"
    local model_name="$2"
    if [ -n "$pth_file" ] && [ -n "$model_name" ]; then
        export MODEL_NAME="$model_name"
        export ORIGINAL_MODEL="${BASE_MODEL_DIR}/${MODEL_NAME}/${MODEL_NAME}.pth"
        export RESULTS="${RESULTS_BASE}/${MODEL_NAME}/$(basename "${pth_file%.*}")"
        mkdir -p "$RESULTS"
        export MODEFIED_MODEL="$pth_file"
        echo "command: python ${PYTHON_SCRIPT} --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODEFIED_MODEL} --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE} --save_dir ${RESULTS}"
        python ${PYTHON_SCRIPT} --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODEFIED_MODEL} --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE} --save_dir ${RESULTS}
    fi
}

# Function to display all configurable parameters and get user confirmation
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
    echo "======================================================"
    echo
    read -p "Do you want to proceed with these settings? (y/N): " confirmation
    if [[ ! "$confirmation" =~ ^[Yy]$ ]]; then
        echo "Execution cancelled by user."
        exit 0
    fi
    echo "Proceeding with execution..."
    echo
}

####################################################################################################################

# Call the function to display parameters and get confirmation
print_parameters

# Iterate through each model in MODELS_ARRAY
for model in "${MODELS_ARRAY[@]}"; do
    echo "Processing model: $model"
    # Get all .pth files from the recalibrated models directory
    pth_files_list=$(get_pth_files "${RECALIBRATED_MODELS_BASE}/${model}")
    echo "PTH files in ${RECALIBRATED_MODELS_BASE}/${model}:"
    echo "$pth_files_list"
    echo "Executing command for model: $model"
    # Loop through each .pth file and execute the python command
    while IFS= read -r pth_file; do
        process_pth_file "$pth_file" "$model"
    done <<< "$pth_files_list"
done

################ End of script ##############################
