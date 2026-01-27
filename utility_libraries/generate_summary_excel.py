import pandas as pd
from pathlib import Path
from openpyxl.styles import Alignment
import re

def create_model_summaries(folder_path=".", sheet_names=None):
    """Create summary files for each model by extracting sheets from Excel files."""
    if sheet_names is None:
        sheet_names = ["resnet50", "vgg16", "inception_v3", "mobilenet_v3_small", "mobilenet_v3_large"]
    
    # Create a summary file for each model
    for target_name in sheet_names:
        summary_file = f"summary_{target_name}.xlsx"
        
        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            print(f"\nCreating summary for {target_name}...")
            
            # Iterate through all xlsx files in the folder
            for file in Path(folder_path).glob("*.xlsx"):
                if file.name.startswith("summary_"):
                    continue                   
                print(f"  Processing {file.name}...")
                
                # Get all sheet names from the Excel file
                excel_file = pd.ExcelFile(file)
                
                # Check each sheet in the file
                for sheet in excel_file.sheet_names:
                    #print("all sheets in file:", excel_file.sheet_names)
                    # Check if sheet starts with the target name
                    if str(sheet).lower().startswith(target_name.lower()):
                        print(f"    - Found matching sheet: {sheet}")
                        # Read the sheet
                        df = pd.read_excel(file, sheet_name=sheet)
                        summary_sheet_name  = sheet
                        # Write to summary file
                        print(f"      - Writing to summary sheet: {summary_sheet_name}")
                        df.to_excel(writer, sheet_name=summary_sheet_name, index=False)
                        print(f"    - Extracted sheet: {sheet}")
            
            # Apply formatting after all sheets are written
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                
                # Left align all cells and autofit columns
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        cell.alignment = Alignment(horizontal='left')
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"  Summary file created: {summary_file}")
    
    print("\nAll summary files created!")

def add_summary_sheet(folder_path=".", sheet_names=None):
    """Add a summary sheet to each model summary file that consolidates Hypothesis Test Results."""
    if sheet_names is None:
        sheet_names = ["resnet50", "vgg16", "inception_v3", "mobilenet_v3_small", "mobilenet_v3_large"]
      
    for model_name in sheet_names:
        summary_file = Path(folder_path) / f"summary_{model_name}.xlsx"
            
        if not summary_file.exists():
            print(f"Summary file not found: {summary_file}")
            continue
            
        print(f"\nAdding summary sheet to {summary_file.name}...")
            
        # Read the existing workbook
        excel_file = pd.ExcelFile(summary_file)
            
        if len(excel_file.sheet_names) == 0:
            print(f"  No sheets found in {summary_file.name}")
            continue
            
        # Read the first sheet to get LayerName column
        first_sheet = excel_file.sheet_names[0]
        df_first = pd.read_excel(summary_file, sheet_name=first_sheet)
            
        if 'Layer' not in df_first.columns:
            print(f"  'Layer' column not found in {summary_file.name}")
            continue
            
        # Create summary dataframe with LayerName as first column
        summary_df = pd.DataFrame({'Layer': df_first['Layer']})
            
        # Iterate through all sheets and collect Hypothesis Test Result columns
        for sheet_name in excel_file.sheet_names:
            df_sheet = pd.read_excel(summary_file, sheet_name=sheet_name)
            
            if 'Hypothesis Test Result' in df_sheet.columns:
                # Extract class and artifact type from sheet name
                class_match = re.search(r'(Class_[0-2])', str(sheet_name), re.IGNORECASE)
                artifact_match = re.search(r'(coat|legs|face|back|all)', str(sheet_name), re.IGNORECASE)
                
                class_name = class_match.group(1) if class_match else "Unknown"
                artifact_name = artifact_match.group(1) if artifact_match else "Unknown"
                
                column_name = f"{class_name}_{artifact_name}_Hypothesis Test Result"
                summary_df[column_name] = df_sheet['Hypothesis Test Result']
                print(f"  Added column: {column_name}")
            
        # Write back to the Excel file with the new summary sheet
        with pd.ExcelWriter(summary_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            # Write the summary sheet
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
            # Apply formatting to the summary sheet
            worksheet = writer.sheets['Summary']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                   
                for cell in column:
                    cell.alignment = Alignment(horizontal='left')
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                    
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
          
        print(f"  Summary sheet added to {summary_file.name}")
        
    print("\nAll summary sheets added!")


def add_pvalue_sheet(folder_path=".", sheet_names=None):
    """Add a P-value sheet to each model summary file that consolidates all P-values."""
    if sheet_names is None:
        sheet_names = ["resnet50", "vgg16", "inception_v3", "mobilenet_v3_small", "mobilenet_v3_large"]
      
    for model_name in sheet_names:
        summary_file = Path(folder_path) / f"summary_{model_name}.xlsx"
            
        if not summary_file.exists():
            print(f"Summary file not found: {summary_file}")
            continue
            
        print(f"\nAdding P-value sheet to {summary_file.name}...")
            
        # Read the existing workbook
        excel_file = pd.ExcelFile(summary_file)
            
        if len(excel_file.sheet_names) == 0:
            print(f"  No sheets found in {summary_file.name}")
            continue
            
        # Read the first sheet to get Layer column
        first_sheet = excel_file.sheet_names[0]
        df_first = pd.read_excel(summary_file, sheet_name=first_sheet)
            
        if 'Layer' not in df_first.columns:
            print(f"  'Layer' column not found in {summary_file.name}")
            continue
            
        # Create P-value dataframe with Layer as first column
        pvalue_df = pd.DataFrame({'Layer': df_first['Layer']})
            
        # Iterate through all sheets and collect P-value columns
        for sheet_name in excel_file.sheet_names:
            df_sheet = pd.read_excel(summary_file, sheet_name=sheet_name)
            
            # Extract class and artifact type from sheet name
            class_match = re.search(r'(Class_[0-2])', str(sheet_name), re.IGNORECASE)
            artifact_match = re.search(r'(coat|legs|face|back|all)', str(sheet_name), re.IGNORECASE)
            
            class_name = class_match.group(1) if class_match else "Unknown"
            artifact_name = artifact_match.group(1) if artifact_match else "Unknown"
            
            # Add Binomial P-value columns
            if 'Binomial P-value (Positive)' in df_sheet.columns:
                column_name = f"{class_name}_{artifact_name}_Binomial P-value (Positive)"
                pvalue_df[column_name] = df_sheet['Binomial P-value (Positive)']
                print(f"  Added column: {column_name}")
            
            if 'Binomial P-value (Negative)' in df_sheet.columns:
                column_name = f"{class_name}_{artifact_name}_Binomial P-value (Negative)"
                pvalue_df[column_name] = df_sheet['Binomial P-value (Negative)']
                print(f"  Added column: {column_name}")
            
            # Add Z-test P-value columns
            if 'Z-test P-value (Positive)' in df_sheet.columns:
                column_name = f"{class_name}_{artifact_name}_Z-test P-value (Positive)"
                pvalue_df[column_name] = df_sheet['Z-test P-value (Positive)']
                print(f"  Added column: {column_name}")
            
            if 'Z-test P-value (Negative)' in df_sheet.columns:
                column_name = f"{class_name}_{artifact_name}_Z-test P-value (Negative)"
                pvalue_df[column_name] = df_sheet['Z-test P-value (Negative)']
                print(f"  Added column: {column_name}")
            
        # Write back to the Excel file with the new P-value sheet
        with pd.ExcelWriter(summary_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            # Write the P-value sheet
            pvalue_df.to_excel(writer, sheet_name='PValue', index=False)
                
            # Apply formatting to the P-value sheet
            worksheet = writer.sheets['PValue']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                   
                for cell in column:
                    cell.alignment = Alignment(horizontal='left')
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                    
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
          
        print(f"  P-value sheet added to {summary_file.name}")
        
    print("\nAll P-value sheets added!")


def main():
    # Define the folder path containing xlsx files
    folder_path = "."  # Change this to your folder path
    
    # Define the sheet names to extract
    sheet_names = ["resnet50", "vgg16", "inception_v3", "mobilenet_v3_small", "mobilenet_v3_large"]
    
    create_model_summaries(folder_path, sheet_names)
    add_summary_sheet(folder_path, sheet_names)
    add_pvalue_sheet(folder_path, sheet_names)

if __name__ == "__main__":
    main()