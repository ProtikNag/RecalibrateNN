#!/usr/bin/env bash
set -euo pipefail

#    parser.add_argument("--org_model_path", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--modified_model_path", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
#    parser.add_argument("--config_file", type=str, default=None, help="Configuration file")
#    parser.add_argument("--save_dir", type=str, default=None, help="Specify a save directory to save the results")


source ./common_params.sh
source ../utility_scripts.sh
PYTHON_SCRIPT="../../xai_visualize_all.py"
# Call the function to display parameters and get confirmation
print_parameters
echo -e "\033[32mPython Script:\033[0m $PYTHON_SCRIPT"
wait_for_user

process_xai_forall_models ()
for model in "${MODELS_ARRAY[@]}"; do

  python $PYTHON_SCRIPT  --org_model_path "${BASE_MODEL_DIR}${model}/${model}.pth" --model_name $model --config_file $CONFIG_FILE --save_dir     "${XAI_RESULTS_LOCATION}/${model}"

done

process_xai_forall_models

 
################ End of script ##############################
