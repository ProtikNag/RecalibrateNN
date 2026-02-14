import os
import sys
import pandas as pd
from pathlib import Path
import openpyxl
from openpyxl import Workbook

#3 Parse all excel files and create a list
def get_allexcelfiles(directory):
    excel_files = [f for f in os.listdir(directory) if f.endswith(('.xlsx', '.xls'))]
    gaussian_files = [f for f in excel_files if 'gaussian' in f.lower()]
    mean_files = [f for f in excel_files if 'mean' in f.lower()]
    print(f"Found {len(gaussian_files)} gaussian files and {len(mean_files)} mean files")
    return gaussian_files, mean_files

# 4. Create destination file
def write_summary_to_excel(summary_data, output_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary_gaussian"
    
    # Write headers
    headers = ['File','Method Used', 'Artifact used', 'Class Used', 'Combination', 'Combination Layer', 'Method', 'Class', 'Mean_Original_Prob',   
    'Mean_Perturbed_Prob', 'Mean_Delta_Logits_Class0', 'Mean_Delta_Logits_Class1', 'Mean_Delta_Logits_Class2']
    ws.append(headers)
    
    # Write summary data
    for row in summary_data:
        ws.append(row)
    
    # Save the workbook
    wb.save(output_path)
    print(f"Summary saved to: {output_path}")

def create_desitnation_file(directory):
    temp = directory.split('/')[-1]
    print(temp)
    summary_path = os.path.join(directory,  temp + '_summary.xlsx')
    wb = Workbook()
    # Remove default sheet
    if 'Sheet' in wb.sheetnames:
        del wb['Sheet']
    
    # Create gaussian worksheets
    wb.create_sheet('Summary_gaussian_class0')
    wb.create_sheet('Summary_gaussian_class1')
    wb.create_sheet('Summary_gaussian_class2')
    
    # Create mean worksheets
    wb.create_sheet('Summary_mean_class0')
    wb.create_sheet('Summary_mean_class1')
    wb.create_sheet('Summary_mean_class2')
    # Write headers to all worksheets
    headers = ['File', 'Combination', 'Combination Layer', 'Method', 'Class', 'Mean_Original_Prob', 'Mean_Perturbed_Prob', 
               'Mean_Delta_Logits_Class0', 'Mean_Delta_Logits_Class1', 'Mean_Delta_Logits_Class2']
    
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        ws.append(headers)
    
    # Save empty workbook
    wb.save(summary_path)
    print(f"Created empty workbook with worksheets at: {summary_path}")
    return summary_path    

def process_files(perturbation_list,method_name, directory, wb):
    for filename in  perturbation_list:
        file_path = os.path.join(directory, filename)
        # List all sheets in the excel file
        xl_file = pd.ExcelFile(file_path)
        sheet_names = xl_file.sheet_names
        sheet_names.pop(0)  # Remove Summary sheet
        print(f"Sheets in {filename}: {sheet_names}")      
        method_used = filename.split('_')[-1]
        artifact_used = filename.split('_')[-2]
        class_used = filename.split('_')[-3]
        
        # Read the summary sheet and create a list of the first column
        try:
            summary_df = pd.read_excel(file_path, sheet_name='Summary')
            combination_layernames = summary_df.iloc[:, 0].tolist()
            print(f"First column values from Summary sheet: {combination_layernames}")
            print("Number of sheets in the file:", len(sheet_names))
        except Exception as e:
            print(f"Could not read Summary sheet from {filename}: {str(e)}")
        for worksheet, combination_layer in zip(sheet_names, combination_layernames):
            try:
                # 6. Navigate to sheet name 'combination'
                df = pd.read_excel(file_path, sheet_name=worksheet)
                # 7. Data already read into dataframe
                # 8. Group by class_id (0, 1, 2)
                # Replace class_id values: 'd' -> 0, 'h' -> 1, 'z' -> 2
                #mapping = {'d': 0, 'h': 1, 'z': 2}
                #df['class_id'] = df['class_id'].map(mapping).fillna(df['class_id'])
                
                for class_id in [0, 1, 2]:
                    class_df = df[df['class_id'] == class_id]
                    if class_df.empty:
                        continue
                    # 9. Compute mean of original_prob
                    mean_original_prob = class_df['original_prob'].mean()
                   
                    # 10. Compute mean of perturbed_prob
                    mean_perturbed_prob = class_df['perturbed_prob'].mean()
                    
                    # 11. Compute mean of delta_logits columns
                    mean_delta_class0 = class_df['delta_logits_class0'].mean()
                    mean_delta_class1 = class_df['delta_logits_class1'].mean()
                    mean_delta_class2 = class_df['delta_logits_class2'].mean()
                    
                    # 12 & 13. Store results in summary file
                    row_data = [
                        filename,
                        method_used,
                        artifact_used,
                        class_used,
                        worksheet,
                        combination_layer,
                        class_id,
                        mean_original_prob,
                        mean_perturbed_prob,
                        mean_delta_class0,
                        mean_delta_class1,
                        mean_delta_class2
                    ]
                    ws = wb[f"Summary_{method_name}_class{class_id}"]
                    ws.append(row_data)
                    print(f"Processed {filename} - Sheet {worksheet} - Class {class_id}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
                    
                    
def main():
    # 1. Get directory name from command line
    if len(sys.argv) < 2:
        print("Usage: python generate_summary_perturbation.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)
    # 2. Parse all excel files and create a list
    gaussian_files, mean_files = get_allexcelfiles(directory)
    print(f"Gaussian files: {gaussian_files}")
    print(f"Mean files: {mean_files}")
    # 4. Create destination file
    summary_path = create_desitnation_file(directory)
    wb = openpyxl.load_workbook(summary_path)
    process_files(mean_files, 'mean', directory, wb)
    wb.save(summary_path)
    process_files(gaussian_files, 'gaussian', directory, wb)
    wb.save(summary_path)

    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()