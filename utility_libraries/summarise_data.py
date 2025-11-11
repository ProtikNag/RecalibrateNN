import pandas as pd
import os
from pathlib import Path
import sys

# Define the directories
directories = [
    r'/mnt/sdb2/sensitivity_analysis_paper/corelation_all',
    r'/mnt/sdb2/sensitivity_analysis_paper/corelation_bg',
    r'/mnt/sdb2/sensitivity_analysis_paper/corelation_coat',
    r'/mnt/sdb2/sensitivity_analysis_paper/corelation_face',
    r'/mnt/sdb2/sensitivity_analysis_paper/corelation_leg'
]



def summarise_tcav_scores(
    directories,
    model_name,
    filename_pattern,
    destination_file,
    summarysheet_name
):
    all_dataframes = []
    # Process each directory
    for directory in directories:
        directory = os.path.join(directory, model_name)
        if not os.path.exists(directory):
            print(f"Directory {directory} not found, skipping...")
            continue
        # Find the Excel file starting with 'anova'
        excel_files = list(Path(directory).glob(filename_pattern))
        if not excel_files:
            print(f"No anova*.xlsx file found in {directory}, skipping...")
            continue
        # Use the first matching file
        excel_file = excel_files[0]
        print(f"Processing: {excel_file}")
        try:
            # Read the TCAVScores sheet
            df = pd.read_excel(excel_file, sheet_name='TCAVScores')
            # Add a column to identify the source directory
            df['Source'] = directory
            # Append to list
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error processing {excel_file}: {e}")
    # Combine all dataframes
    if all_dataframes:
        summary_df = pd.concat(all_dataframes, ignore_index=True)
        # Write to summary Excel file
        with pd.ExcelWriter(destination_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name=summarysheet_name, index=False)
        print(f"Summary file created successfully: {summarysheet_name}")
    else:
        print("No data found to create summary file")    
        
        
if(__name__ == '__main__'):
    if(sys.argv.__len__() > 2):
        model = sys.argv[1]
        destination_dir = sys.argv[2] 
    else:
        print("Please provide the model name as a command-line argument. and the destination directory")
        print("Example: python anova.py resnet50 /mnt/sdb2/sensitivity_analysis_paper/")
        sys.exit(1)
    filename_pattern = f'anova*{model}*.xlsx'
    summarysheet_name = f'summary_{model}.xlsx'
    
    destination_file = os.path.join(destination_dir, summarysheet_name)
    print(f"Model: {model}")
    print(f"Filename pattern: {filename_pattern}")
    print(f"Summary sheet name: {summarysheet_name}")
    print(f"Destination file: {destination_file}")
    
    # List to store all dataframes
    summarise_tcav_scores(directories, model, filename_pattern, destination_file, summarysheet_name) 

