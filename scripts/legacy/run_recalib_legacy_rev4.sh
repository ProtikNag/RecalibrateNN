#!/usr/bin/env bash
set -euo pipefail
export PATH=../../../:${PATH}


source ./common_params.sh
source ../utility_scripts.sh

PYTHON_SCRIPT="../../main.py"
CONFIG_FILE="../../config/legacy/config_legacy_3classes_ver4.yaml"
#export RECALIBRATED_RESULTS_LOCATION='/mnt/sdh/basics/recalib'
export RECALIBRATED_RESULTS_LOCATION='/mnt/sdc/try3'
print_parameters
RUN_SPECIFIC_MODEL=true 

#wait_for_user


if [ "$RUN_SPECIFIC_MODEL" = true ]; then
  MODELS_ARRAY=("vgg16" "inception_v3")
fi



###########################################################################################
for model in "${MODELS_ARRAY[@]}"; do
    echo "Processing Recalibration for model: ${model}"

COMMAND="python ${PYTHON_SCRIPT} \
    --model_name ${model} \
    --model_path ${BASE_MODEL_DIR} \
    --store_results ${RECALIBRATED_RESULTS_LOCATION} \
    --config ${CONFIG_FILE}"

    echo -e "\033[32m Command to execute:\033[0m $COMMAND"
    export model && $COMMAND 
done
###########################################################################################
