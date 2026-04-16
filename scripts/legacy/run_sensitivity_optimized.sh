#!/usr/bin/env bash
set -euo pipefail
export PATH=../../../:${PATH}


source ./common_params.sh
source ../utility_scripts.sh

PYTHON_SCRIPT="../../util_compute_sensitivity_optimized.py"

# Call the function to display parameters and get confirmation

BEFORE_AFTER=true
echo -e "\033[32mRecalibration Before and after Flag:\033[0m $BEFORE_AFTER"

RECALIBRATED_MODELS_BASE="/mnt/sdc/try2/"
SENSITIVITY_RESULTS_LOCATION="/mnt/sdc/sensitivity_try2"
if [ "$BEFORE_AFTER" = true ]; then
  BEFORE_AFTER_OPTION="--before_after"
else
  BEFORE_AFTER_OPTION="/mnt/sdc/try2"
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
