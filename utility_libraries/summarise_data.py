import pandas as pd
import os
from pathlib import Path
import sys
import numpy as np

if(os.name == 'posix'):
    # Define the directories
    directories = [
        r'/mnt/sdb2/sensitivity_analysis_paper/corelation_all',
        r'/mnt/sdb2/sensitivity_analysis_paper/corelation_bg',
        r'/mnt/sdb2/sensitivity_analysis_paper/corelation_coat',
        r'/mnt/sdb2/sensitivity_analysis_paper/corelation_face',
        r'/mnt/sdb2/sensitivity_analysis_paper/corelation_leg'
    ]
if(os.name == 'nt'):
    directories = [
        r'c:\temp\corelation_all',
        r'c:\temp\corelation_bg',
        r'c:\temp\corelation_coat',
        r'c:\temp\corelation_face',
        r'c:\temp\corelation_leg'
    ]


def get_all_excelfiles(directories,model_name, filename_pattern):
    filenames = []
    for directory in directories:
        directory = os.path.join(directory, model_name)
        if not os.path.exists(directory):
            print(f"Directory {directory} not found, skipping...")
            continue
        filenames.extend(list(Path(directory).glob(filename_pattern)))
    return filenames


def summarize_tcav_scores(file_names, destination_file, model_name, summarysheet_name):
    all_dataframes = []
    for excel_file in file_names:
        print(f"Processing: {excel_file}")
        try:
            # Read the TCAVScores sheet
            df = pd.read_excel(excel_file, sheet_name='TCAVScores')
            #Store the first column once then drop it from subsequent dataframes
            if not all_dataframes:
                all_dataframes.append(df.iloc[:, [0]].rename(columns={df.columns[0]: f"{model_name}"}))
                df = df.drop(df.columns[0], axis=1)
            else:
                df = df.drop(df.columns[0], axis=1)
            
            # Add a column to identify the source directory
            excel_file = str(excel_file)
            #print(excel_file.split(os.sep)[-3])
            #print("_____",excel_file.split(os.sep))
            for columns in df.columns:
              temp = f"{excel_file.split(os.sep)[-3]}_{columns}"
              temp = temp.replace("corelation_", "").strip()
              df = df.rename(columns={columns: temp})
            if not all_dataframes:
                # If this is the first dataframe, use it as base
                all_dataframes.append(df)
            else:
                # Merge with existing dataframe by adding new columns
                all_dataframes[0] = pd.concat([all_dataframes[0], df], axis=1)
                
            for columns in df.columns:
                print(columns)
                excel_file = str(excel_file)
                #print(temp)
                    
                    
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

def summarize_feature_corelation(file_names, src_sheet_name, destination_file, model_name, summarysheet_name):
    all_dataframes = []
    for excel_file in file_names:
        print(f"Processing: {excel_file}")
        try:
            # Read the Melted class scores to compute sensitivity
            df = pd.read_excel(excel_file, sheet_name=src_sheet_name)
            #get the correlation feature name
            temp = str(excel_file).split(os.sep)[-3]
            temp = temp.replace("corelation_", "").strip()
            print(f"Correlation Feature: {temp}")
            df = df.rename(columns={'Sensitivity': temp})
            if not all_dataframes:
                all_dataframes.append(df[['Layer', temp]])
            else:
                all_dataframes[0] = pd.concat([all_dataframes[0], df[[temp]]], axis=1)
        except Exception as e:
            print(f"Error processing {excel_file}: {e}")
            continue
    print(all_dataframes)
    all_dataframes = pd.DataFrame(all_dataframes[0])
    subdfs = {layer: all_dataframes[all_dataframes["Layer"] == layer] for layer in all_dataframes["Layer"].unique()}
    correlations = {
    layer: subdf.drop(columns=["Layer"]).corr()
    for layer, subdf in subdfs.items()
    }
    for layer in correlations:
        mask = np.triu(np.ones_like(correlations[layer], dtype=bool))
        correlations[layer] = correlations[layer].mask(mask)

    correlations = pd.concat(correlations).reset_index().rename(columns={'level_0': 'Layer'})
    print(correlations)
    # Write to summary Excel file
    with pd.ExcelWriter(destination_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        for layer, corr_df in correlations.groupby('Layer'):
            sheet_name = str(layer).replace('/', '_')[:31]  # Excel sheet name limit is 31 chars
            corr_df.to_excel(writer, sheet_name=sheet_name, index=False)
    #with pd.ExcelWriter(destination_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    #    correlations.to_excel(writer, sheet_name='All_Layers', index=False)

    return all_dataframes        
    
#Class0_Melted_Melted
        
if(__name__ == '__main__'):
    if(sys.argv.__len__() > 1):
        destination_dir = sys.argv[1] 
    else:
        print("Please provide the model name as a command-line argument. and the destination directory")
        print("Example: python summarize_data.py  /mnt/sdb2/sensitivity_analysis_paper/")
        if(os.name == 'nt'):
            destination_dir = r'c:\temp\summary_output'
        else:
            destination_dir = f'/mnt/sdb2/sensitivity_analysis_paper/'
        #sys.exit(1)
    models = ['vgg16','inception_v3', 'resnet50', 'mobilenet_v3_small', 'mobilenet_v3_large']
    #models = ['vgg16']
    for model in models:
        summarysheet_name = f'summary_{model}.xlsx'
        destination_file = os.path.join(destination_dir, summarysheet_name)
        filename_pattern = f'anova*{model}*.xlsx'
        excel_files = get_all_excelfiles(directories, model, filename_pattern)
        # Create an empty destination file to clear up old contents
        if os.path.exists(destination_file):
            os.remove(destination_file)
        summarize_tcav_scores(excel_files, destination_file,model, summarysheet_name)
        a = summarize_feature_corelation(excel_files, 'Class0_Melted_Melted', destination_file, model, summarysheet_name)
        


