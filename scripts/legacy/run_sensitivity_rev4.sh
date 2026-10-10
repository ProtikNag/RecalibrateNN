#!/usr/bin/env bash
set -euo pipefail
export PATH=../../../:${PATH}


source ./common_params.sh
source ../utility_scripts.sh

PYTHON_SCRIPT="../../util_sensitivity_compute.py"

# Call the function to display parameters and get confirmation

BEFORE_AFTER=true
echo -e "\033[32mRecalibration Before and after Flag:\033[0m $BEFORE_AFTER"
CONFIG_FILE="../../config/legacy/config_legacy_3classes_ver4.yaml"
RECALIBRATED_MODELS_BASE="/mnt/sdc/try2/"
SENSITIVITY_RESULTS_LOCATION="/mnt/sdc/sensitivity_try3/randomseed_132"
if [ "$BEFORE_AFTER" = true ]; then
  BEFORE_AFTER_OPTION="--before_after"
else
  BEFORE_AFTER_OPTION=""
fi

MODELS_ARRAY=("vgg16")

print_parameters
#wait_for_user

###########################################################################################



for model in "${MODELS_ARRAY[@]}"; do
    echo "Processing sensitivity for model: ${model}"

    COMMAND="python ${PYTHON_SCRIPT} \
    --org_model_path ${BASE_MODEL_DIR}/${model}/${model}.pth \
    ${BEFORE_AFTER_OPTION} \
    --model_name ${model} \
    --recal_model_basepath ${RECALIBRATED_MODELS_BASE} \
    --store_results ${SENSITIVITY_RESULTS_LOCATION} \
    --config ${CONFIG_FILE}"

    echo -e "\033[32m Command to execute:\033[0m $COMMAND"
    export model && $COMMAND 
done


###########################################################################################
