import os
import sys
from pathlib import Path
import pandas as pd
import re
####################################################################################################
# # This script is designed to extract file paths of specific metric files from a structured directory.
#################################################################################################### 
def extract_metric_filepath(base_path):
    # Ensure base_path is a Path object
    if isinstance(base_path, str):
        base_path = Path(base_path)
    files_list = []
    model_dict = {}
    for model_dir in base_path.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            model_dict[model_name] = {"train": [], "valid": []}
            for split in ["train", "valid"]:
                split_pattern = f"morphed_{split}_"
                for folder in model_dir.iterdir():
                    if folder.is_dir() and folder.name.startswith(split_pattern):
                        stats_file = folder / "prediction_statistics_predictions.txt"
                        if stats_file.exists():
                            files_list.append(str(stats_file))
                            model_dict[model_name][split].append(str(stats_file))
    return {"files_list": files_list, "model_dict": model_dict}
####################################################################################################
# # This script is designed to format the content of a confusion matrix file.
####################################################################################################    
def getparameters_from_filepath(file_path):
    parts = file_path.split(os.sep)
    
    found_model = False
    models = ['vgg16', 'resnet50', 'mobilenet_v3_small', 'mobilenet_v3_large', 'inception_v3']
    model_folder = None
    found_model = False
    for m in models:
        if m in parts:
            model_folder = m
            found_model = True
            break    
    result['model'] = model_folder
    mode = ["patched_out", "patched_in"]
    patching_option = None
    for m in mode:
        if m in parts:
            patching_option = m
            break
    result['patching_option'] = patching_option
    idx = next((i for i, p in enumerate(parts) if p.startswith("morphed")), -1)
    experiment_folder = parts[idx] if idx != -1 else None
    experiment_folder = experiment_folder.split("_")
    result['mode'] = experiment_folder[1]
    result['src_class'] = experiment_folder[2]
    result['concept_class'] = experiment_folder[3]
    result['concept'] = experiment_folder[4]
    return result
####################################################################################################
# # This script is designed to format the content of a confusion matrix file.
####################################################################################################    
def format_confusion_matrix(input_file):
    df_metrics = None
    with open(input_file, 'r') as f:
        content = f.read()
    # Extract confusion matrix
    matrix_pattern = r'\[\[(.*?)\]\]'
    matrix_match = re.search(matrix_pattern, content, re.DOTALL)
    confusion_matrix = None
    if matrix_match:
        matrix_str = matrix_match.group(1)
        rows = []
        for line in matrix_str.strip().split('\n'):
            line = line.strip()
            if line:
                # Remove any brackets and split by whitespace
                line = line.replace('[', '').replace(']', '')
                row = list(map(int, line.split()))
                if row:
                    rows.append(row)
        confusion_matrix = rows
    # Extract overall accuracy
    accuracy_pattern = r'Overall Accuracy:\s+([\d.]+)'
    accuracy_match = re.search(accuracy_pattern, content)
    overall_accuracy = float(accuracy_match.group(1)) if accuracy_match else None
    # Extract per-class statistics
    per_class_statistics = {}
    class_pattern = r'(\w+):\n\s+Precision:\s+([\d.]+)\n\s+Recall:\s+([\d.]+)\n\s+F1-Score:\s+([\d.]+)\n\s+Support:\s+([\d.]+)'
    for match in re.finditer(class_pattern, content):
        class_name = match.group(1)
        per_class_statistics[class_name] = {
            'Precision': float(match.group(2)),
            'Recall': float(match.group(3)),
            'F1-Score': float(match.group(4)),
            'Support': float(match.group(5))
        }
    result = {
        'confusion_matrix': confusion_matrix,
        'overall_accuracy': overall_accuracy,
        'per_class_statistics': per_class_statistics
    }
    try:
        metrics = result['per_class_statistics']
        df_metrics = pd.DataFrame(metrics).T
        cm = result['confusion_matrix']
        df_metrics['Class_Name'] = df_metrics.index
        class_map = {'deer':0, 'horse':1, 'zebra':2}
        df_metrics['Class_ID'] = df_metrics['Class_Name'].map(class_map)
        df_metrics['Pred_0'] = [row[0] for row in cm]
        df_metrics['Pred_1'] = [row[1] for row in cm]
        df_metrics['Pred_2'] = [row[2] for row in cm]
        df_metrics = df_metrics.reset_index(drop=True)
    except Exception as e:
        print(f"Failed to create DataFrame for file {input_file}: {e}")
    return df_metrics

####################################################################################################
# # This script is designed to format the content of a confusion matrix file.
####################################################################################################    
result = extract_metric_filepath("/mnt/sdc/prediction_results/patched_in")
all_data = []
for file in result['files_list']:
    print(file)
    experiment_results = format_confusion_matrix(file)
    required_cols = ["model", "patching_option", "mode", "src_class", "concept_class", "concept"]
    if isinstance(experiment_results, pd.DataFrame) and not experiment_results.empty:
        experiment_configuration = getparameters_from_filepath(file) or {}
    else:
        # one fallback row
        experiment_results = pd.DataFrame([{}])
        experiment_configuration = {}
    experiment_configuration = {col: experiment_configuration.get(col, "Unknown") for col in required_cols}
    experiment_results = experiment_results.assign(**experiment_configuration)    
    #print(f"Formatted Data for {file}:\n{experiment_results}\n")
    all_data.append(experiment_results)
final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv("experiment_results.csv", index=False)
with pd.ExcelWriter("experiment_results_grouped.xlsx") as writer:
    for model_name, model_df in final_df.groupby("model"):
        for animal, animal_df in model_df.groupby("src_class"):
            sheet_name = f"{model_name}_{animal}"
            # Excel sheet names have a max length of 31 characters
            sheet_name = sheet_name[:31]
            animal_df.to_excel(writer, sheet_name=sheet_name, index=False)
print(final_df)
