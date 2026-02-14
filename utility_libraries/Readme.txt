#Readme
------------------------------
Step 1: Run sensitivity compuration
------------------------------
script: /home/srikanth/study1/RecalibrateNN/scripts/legacy/generate_concept_sensitivity.sh
Executes the base script util_sensitivity_compute.py with the before option only 
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

------------------------------
Step 2: Summarize sensitivity
------------------------------ 
Program 
python sensitivity_summariser.py /mnt/sdc/sensitivity_analysis_paper/sensitivity_summary/sensitivity_config_s_coat coat
Expectation : 
Folder : sensitivity_summary/sensitivity_config_s_coat 
inception_v3  mobilenet_v3_large  mobilenet_v3_small  resnet50 
Result : sensitivity_config_s_coat_consolidated_results.xlsx

------------------------------------------------------------------------------------------
Step 3: Generate the p values and Z test and computes the final results
------------------------------------------------------------------------------------------
sensitivity_summariser.py: This program does the Z test and Binomial  Test 
 python sensitivity_summariser.py /mnt/sdc/sensitivity_analysis_paper/sensitivity_summary/sensitivity_config_s_coat/ coat
Found 5 CSV files
Copied Class 0 from rnet50 as rnet50_Class_0_coat
Copied Class 1 from rnet50 as rnet50_Class_1_coat
Copied Class 2 from rnet50 as rnet50_Class_2_coat
Copied Class 0 from iv3 as iv3_Class_0_coat
Copied Class 1 from iv3 as iv3_Class_1_coat
Copied Class 2 from iv3 as iv3_Class_2_coat
Copied Class 0 from mnet_v3_large as mnet_v3_large_Class_0_coat
Copied Class 1 from mnet_v3_large as mnet_v3_large_Class_1_coat
Copied Class 2 from mnet_v3_large as mnet_v3_large_Class_2_coat
Copied Class 0 from vgg16 as vgg16_Class_0_coat
Copied Class 1 from vgg16 as vgg16_Class_1_coat
Copied Class 2 from vgg16 as vgg16_Class_2_coat
Copied Class 0 from mnet_v3_small as mnet_v3_small_Class_0_coat
Copied Class 1 from mnet_v3_small as mnet_v3_small_Class_1_coat
Copied Class 2 from mnet_v3_small as mnet_v3_small_Class_2_coat
Consolidated results saved to: /mnt/sdc/sensitivity_analysis_paper/sensitivity_summary/sensitivity_config_s_coat/sensitivity_config_s_coat_consolidated_results.xlsx
Analysis complete!
------------------------------------------------------------------------------------------
Step 4: Summarize P Values 
------------------------------------------------------------------------------------------
Pre requisite: Run the step 3 and get all the excel files in one folder  the folder should contain sensitivity_config_s_coat_consolidated_results.xlsx
sensitivity_config_s_legs_consolidated_results.xlsx sensitivity_config_s_face_consolidated_results.xlsx sensitivity_config_s_bg_consolidated_results.xlsx sensitivity_config_s_all_consolidated_results.xlsx
ls /home/srikanth/study1/RecalibrateNN/sensitivity_results/legacy/multiconcept/
This generates a summary file as well with positive and negative sensitivities

------------------------------------------------------------------------------------------
Step 4: Generate neuron perturbation 
------------------------------------------------------------------------------------------
pertubate_neurons.py: Pertubate neurons with gaussian and mean uses pertubation_utilities.py:layer_config.yaml and logger.py

Perturb neurons in a neural network model

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Name of the model (e.g., vgg16, resnet50, inception_v3)
  --config CONFIG       Path to the yaml file
  --layers_to_pertubate LAYERS_TO_PERTUBATE (Overrides config file)
                        list of layers to pertubate
  --saveas SAVEAS       Destination excel file name

generate_summary_perturbation.py: Summarize mean and delta logits
generate_summary_perturbation.py <directory_path>

==========================================================================================
python generate_summary_perturbation.py /mnt/sdc/sensitivity_analysis_paper/perturbation/train/iv3_new
/mnt/sdc/sensitivity_analysis_paper/perturbation/train/mnet_large
/mnt/sdc/sensitivity_analysis_paper/perturbation/train/mnet_small
/mnt/sdc/sensitivity_analysis_paper/perturbation/train/rnet50
/mnt/sdc/sensitivity_analysis_paper/perturbation/train/iv3
Create summary file in same directory 

python generate_summary_excel.py

==========================================================================================
------------------------------------------------------------------------------------------
Generic scripts: 
------------------------------------------------------------------------------------------
1. image_topdf.py: Convert images to pdf for paper
2. Silhoutte.py: Generate Silhoutte.py of image2. Silhoutte.py: Generate Silhoutte.py of image 
 

==========================================================================================


------------------------------------------------------------------------------------------
Paper 1 Step 3: Create Corelation matrix 
------------------------------------------------------------------------------------------
Hard coded paths 
Folder path is hard coded in the script
summarise_data.py
Create summary of the corellation sheets
        r'/mnt/sdb2/sensitivity_analysis_paper/corelation_all',
        r'/mnt/sdb2/sensitivity_analysis_paper/corelation_bg',
        r'/mnt/sdb2/sensitivity_analysis_paper/corelation_coat',
        r'/mnt/sdb2/sensitivity_analysis_paper/corelation_face',
        r'/mnt/sdb2/sensitivity_analysis_paper/corelation_leg'
Creates 1 sheet containint 1 artifact per for all models


