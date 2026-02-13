import os
import sys
import pandas as pd
from pathlib import Path
import openpyxl
from openpyxl import Workbook
def get_allexcelfiles(directory):
    excel_files = [f for f in os.listdir(directory) if f.endswith(('.xlsx', '.xls'))]
    gaussian_files = [f for f in excel_files if 'gaussian' in f.lower()]
    mean_files = [f for f in excel_files if 'mean' in f.lower()]
    print(f"Found {len(gaussian_files)} gaussian files and {len(mean_files)} mean files")
    return gaussian_files, mean_files

def modify_columns_mean_files(mean_files, directory, reference_df=None):
    # Get the first 3 columns from df to use as reference
    first_three_cols = reference_df.iloc[:, :3]
    for filename in mean_files:
        file_path = os.path.join(directory, filename)
        # Load the entire workbook
        xl_file = pd.ExcelFile(file_path)
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            for sheet_name in xl_file.sheet_names:
                # Only process sheets that start with 'combination'
                if not str(sheet_name).startswith('combination'):
                    continue
                # Read the current sheet
                sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Replace first 3 columns with df's first 3 columns
                if len(sheet_df) == len(first_three_cols):
                    sheet_df.iloc[:, :3] = first_three_cols.values
                # Write back to the same sheet
                sheet_df.to_excel(writer, sheet_name=str(sheet_name), index=False)
        print(f"Modified {filename}")

if(__name__ == "__main__"):
    if len(sys.argv) != 2:
        print("Usage: python modify_means_files.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)
    gaussian_files, mean_files = get_allexcelfiles(directory)
    if not gaussian_files:
        print("Error: No gaussian files found in the directory.")
        sys.exit(1)
    if not mean_files:
        print("Error: No mean files found in the directory.")
        sys.exit(1)
    reference_df = pd.read_excel(os.path.join(directory, gaussian_files[0]), sheet_name='combination_0')
    modify_columns_mean_files(mean_files, directory, reference_df)