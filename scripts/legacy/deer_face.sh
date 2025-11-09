#!/usr/bin/env bash
set -euo pipefail
export PATH=../../../:${PATH}


source ./common_params.sh
source ../utility_scripts.sh

PYTHON_SCRIPT="../../util_sensitivity_compute.py"

#force update sensitivity result location
SENSITIVITY_RESULTS_LOCATION="/mnt/sdb2/sensitivity_analysis_paper/run2/sensitivity_config_s_deer_face"
#CONFIG_FILES=(   "../../config/legacy/config_s_deer_all.yaml"  
#   "../../config/legacy/config_s_deer_coat_new.yaml"  
# OK  "../../config/legacy/config_s_deer_face.yaml"
# OK "../../config/legacy/config_s_deer_leg.yaml"
#"../../config/legacy/config_s_deer_coat.yaml" 
#)
# Call the function to display parameters and get confirmation
#2
CONFIG_FILE="../../config/legacy/config_s_deer_face.yaml"
MODELS_ARRAY=("inception_v3")
print_parameters
BEFORE_AFTER=false
echo -e "\033[32mRecalibration Before and after Flag:\033[0m $BEFORE_AFTER"

if [ "$BEFORE_AFTER" = true ]; then
  BEFORE_AFTER_OPTION="--before_after"
else
  BEFORE_AFTER_OPTION=""
fi

wait_for_user

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
