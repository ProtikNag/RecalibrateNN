#!/usr/bin/env bash
#!/usr/bin/env bash
set -euo pipefail
export PATH=../../../:${PATH}

################ Common methods ##########################
source ../utility_scripts.sh

#################### COmmon parameters ###################
PYTHON_SCRIPT="../../util_sensitivity_compute.py"

BASE_MODEL_DIR="/mnt/sdc/data/caltech/CUB_200_2011/cub_trained_cpu/threeclasses"
MODELS_ARRAY=("vgg16" "mobilenet_v3_small" "mobilenet_v3_large" "resnet50" "inception_v3")


#Sensitivity and XAI Results location 
RECALIBRATED_MODELS_BASE="/mnt/sdd/cub_3c"
#Recalibration Results location
RECALIBRATED_RESULTS_LOCATION="/tmp/cub_3c_full"
XAI_RESULTS_LOCATION="/tmp/cub_3c_full"


BEFORE_AFTER=false
echo -e "\033[32mRecalibration Before and after Flag:\033[0m $BEFORE_AFTER"

if [ "$BEFORE_AFTER" = true ]; then
  BEFORE_AFTER_OPTION="--before_after"
else
  BEFORE_AFTER_OPTION=""
fi


ARTIFACT=beak
CONFIG_FILE="../../config/cub_3c/config_cub_3c_${ARTIFACT}.yaml"
#force update sensitivity result location
SENSITIVITY_RESULTS_LOCATION="/mnt/sdc/sensitivity_analysis_paper/cub_3c/sensitivity_config_s_${ARTIFACT}"
#force update sensitivity result location
SENSITIVITY_RESULTS_LOCATION="/mnt/sdc/sensitivity_analysis_paper/cub_3c/sensitivity_config_s_${ARTIFACT}"

print_parameters

#################### COmmon parameters ###################

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
