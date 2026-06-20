#!/usr/bin/env bash
set -euo pipefail
export PATH=../../../:${PATH}


source ./common_params.sh
source ../utility_scripts.sh

PYTHON_SCRIPT="../../util_sensitivity_compute.py"

if [ "$#" -eq 0 ]; then
    echo "Error: No argument provided. Expected arguments are beak, forehead, breast"
    echo "Usage: $0 <argument>"
    exit 1
fi

echo "Argument provided: $1"


ARTIFACT=$1
#force update sensitivity result location
SENSITIVITY_RESULTS_LOCATION="/mnt/sdc/cub_sensitivity/cub_3c_models/sensitivity/sensitivity_config_cub_3c_${ARTIFACT}"
mkdir -p ${SENSITIVITY_RESULTS_LOCATION}
# Call the function to display parameters and get confirmation
#2
CONFIG_FILE="../../config/cub_3c/config_cub_3c_${ARTIFACT}.yaml"
#Only for overriding it
MODELS_ARRAY=("vgg16")
BEFORE_AFTER=true
echo -e "\033[32mRecalibration Before and after Flag:\033[0m $BEFORE_AFTER"

print_parameters

if [ "$BEFORE_AFTER" = true ]; then
  BEFORE_AFTER_OPTION="--before_after"
else
  BEFORE_AFTER_OPTION=""
fi

#wait_for_user

###########################################################################################

# Core script


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

