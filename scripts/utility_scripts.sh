#!/usr/bin/env bash
set -euo pipefail

#    parser.add_argument("--org_model_path", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--modified_model_path", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--config_file", type=str, default=None, help="Configuration file")
#    parser.add_argument("--save_dir", type=str, default=None, help="Specify a save directory to save the results")

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
####################################################################################################################
# Function to process a single pth file for a given model
process_pth_file() {
    local pth_file="$1"
    local model_name="$2"
    if [ -n "$pth_file" ] && [ -n "$model_name" ]; then
        export MODEL_NAME="$model_name"
        export ORIGINAL_MODEL="${BASE_MODEL_DIR}/${MODEL_NAME}/${MODEL_NAME}.pth"
        export RESULTS="${XAI_RESULTS_LOCATION}/${MODEL_NAME}/$(basename "${pth_file%.*}")"
        mkdir -p "$RESULTS"
        export MODEFIED_MODEL="$pth_file"
        echo "command: python ${PYTHON_SCRIPT} --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODEFIED_MODEL} --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE} --save_dir ${RESULTS}"
        python ${PYTHON_SCRIPT} --org_model_path ${ORIGINAL_MODEL} --modified_model_path ${MODEFIED_MODEL} --model_name ${MODEL_NAME} --config_file ${CONFIG_FILE} --save_dir ${RESULTS}
    fi
}
####################################################################################################################
# Waits with a prompt for user to read all the parameters and acknowledge
wait_for_user() {
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
# Function to process selected model with specific pth file
# Takes 2 parameters Model name and pth file array
process_xai_for_selected_model() {
    local model_name="$1"
    shift
    local pth_file_array=("$@")
    
    echo "Processing model: $model_name"
    echo "Processing files: ${pth_file_array[@]}"
    echo "Executing command for model: $model_name"
    for pth_file in "${pth_file_array[@]}"; do
        echo "Processing file: $pth_file"
        process_pth_file "$pth_file" "$model_name"
    done
}

####################################################################################################################

####################################################################################################################

# Function to process all models
process_xai_forall_models() {
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
}
