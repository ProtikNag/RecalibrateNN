#!/usr/bin/env bash
set -euo pipefail
export PATH=../../../:${PATH}


source ./common_params.sh
source ../utility_scripts.sh

MODELS_ARRAY=("vgg16" "resnet50" "inception_v3" "mobilenet_v3_small" "mobilenet_v3_large")

PYTHON_SCRIPT="../../main_store_cav.py"


#MODELS_ARRAY=("inception_v3")

print_parameters
#wait_for_user

###########################################################################################



for model in "${MODELS_ARRAY[@]}"; do
    echo "Processing sensitivity for model: ${model}"

    COMMAND="python ${PYTHON_SCRIPT} \
    --model_path ${BASE_MODEL_DIR}/\
    --model_name ${model} \
    --store_results ${CAV_RESULTS_LOCATION} \
    --config ${CONFIG_FILE}"
    echo -e "\033[32m Command to execute:\033[0m $COMMAND"
    export model && $COMMAND 
done


###########################################################################################
