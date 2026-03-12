import os
import sys
from pathlib import Path
import pandas as pd
import re
####################################################################################################
# # This script is designed to extract file paths of specific metric files from a structured directory.
####################################################################################################    
def extract_metric_filepath(base_path):
    results = []
    for split in ["train", "valid"]:
        split_path = base_path / split
        for main_folder in split_path.iterdir():
            if main_folder.is_dir():
                main_folder_name = main_folder.name
                for model_folder in main_folder.iterdir():
                    if model_folder.is_dir():
                        model_folder_name = model_folder.name
                        text_file = model_folder / split.lower() / f"prediction_statistics_predictions_{split}.txt"
                        if text_file.exists():
                            results.append({
                                "main_folder": main_folder_name,
                                "full_path": str(text_file),
                                "model_folder": model_folder_name
                            })
    for item in results:
        print(f"Main Folder: {item['main_folder']}")
        print(f"File Path: {item['full_path']}\n")
        print(f"Model Folder: {item['model_folder']}\n")
    return results

####################################################################################################
# # This script is designed to format the content of a confusion matrix file.
####################################################################################################    
def format_confusion_matrix(input_file):
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
    return result


####################################################################################################
# # This script is designed to format the content of a confusion matrix file.
####################################################################################################    
if __name__ == "__main__":
    base_path = Path("/mnt/sdc/prediction_results/patched_out")
    results = extract_metric_filepath(base_path)
    all_data = []
    for item in results:
        formatted_data = format_confusion_matrix(item['full_path'])
        """
        print("--------------------------------------------------\n")
        print(f"Formatted Data for {item['full_path']}:\n{formatted_data}\n")
        print(f"Model Folder: {item['model_folder']}\n")
        print(f"Main Folder: {item['main_folder']}\n")
        print("Confusion Matrix:")
        for row in formatted_data['confusion_matrix']:
            print(row)
        print(f"Overall Accuracy: {formatted_data['overall_accuracy']}\n")
        print("Per Class Statistics:")
        for class_name, stats in formatted_data['per_class_statistics'].items():
            print(f"{class_name}: {stats}")
            print("\n")
        print("--------------------------------------------------\n")
        """
        metrics = formatted_data['per_class_statistics']
        try:
          cm = formatted_data['confusion_matrix']
          df_metrics = pd.DataFrame(metrics).T
          df_metrics['Class_Name'] = df_metrics.index
          class_map = {'deer':0, 'horse':1, 'zebra':2}
          df_metrics['Class_ID'] = df_metrics['Class_Name'].map(class_map)
          df_metrics['Pred_0'] = [row[0] for row in cm]
          df_metrics['Pred_1'] = [row[1] for row in cm]
          df_metrics['Pred_2'] = [row[2] for row in cm]
          df_metrics = df_metrics.reset_index(drop=True)
          df_metrics['model_folder'] =  item['model_folder']
          df_metrics['Configuration'] =  item['main_folder']
          df_metrics['Filename'] = item['full_path'].split(os.sep)[-1]
          experiment = item['main_folder'].split('_')
          Class_animal = experiment[0]
          mode  = experiment[1]
          overlay_concept = experiment[2] + '_' + experiment[3]
          df_metrics['Class_animal'] = Class_animal
          df_metrics['mode'] = mode
          df_metrics['overlay_concept'] = overlay_concept
          df_metrics = df_metrics.reset_index(drop=True)
          all_data.append(df_metrics)
        except Exception as e:
          print(f"Failed to compute the metric for file ",item['full_path'])
    # Convert all_data to DataFrame and save to CSV
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        df.to_csv(f'results_{mode}.csv', index=False)
        print(f"Results saved to results.csv")
        output_file = f'results.xlsx'
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
          for model_name, group in df.groupby('model_folder'):
            sheet_name = str(model_name)[:31]  # Excel sheet name limit
            group.to_excel(writer, sheet_name=sheet_name, index=False)
