import pandas as pd
import os
from pathlib import Path

# -------------------------
# Configuration
# -------------------------
base_dir = input("Enter the base directory path: ").strip()
sheets = ["Class_0_Corr", "Class_1_Corr", "Class_2_Corr"]
output_file = "top5_last_row_correlations.txt"

# -------------------------
# Processing
# -------------------------
# Find all Excel files starting with "sensitivity"
excel_files = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().startswith("sensitivity") and file.endswith((".xlsx", ".xls")):
            excel_files.append(os.path.join(root, file))

if not excel_files:
    print("No Excel files starting with 'sensitivity' found.")
    exit()

print(f"Found {len(excel_files)} file(s):")
for f in excel_files:
    print(f"  {f}")

with open(output_file, "w") as f:
    for excel_file in excel_files:
        f.write(f"{'='*60}\n")
        f.write(f"File: {excel_file}\n")
        f.write(f"{'='*60}\n\n")
        
        for sheet in sheets:
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet, index_col=0)
                
                # Get the last row (as a Series)
                last_row = df.iloc[-1]
                
                # Get top 5 values and corresponding column names
                top5 = last_row.sort_values(ascending=False).head(6)
                
                # Write results
                # Extract directory name and file name
                dir_name = os.path.basename(os.path.dirname(excel_file))
                file_name = os.path.basename(excel_file)
                
                f.write(f"Directory: {dir_name}\n")
                f.write(f"File name: {file_name}\n")
                f.write(f"Sheet: {sheet}\n")
                f.write(f"Last row name: {last_row.name}\n")
                f.write("Top 5 correlations:\n")
                f.write(f"Sheet: {sheet}\n")
                f.write(f"Last row name: {last_row.name}\n")
                f.write("Top 5 correlations:\n")
                
                for col, val in top5.items():
                    f.write(f"  {col}: {val:.4f}\n")
                
                f.write("\n" + "-"*40 + "\n\n")
            
            except Exception as e:
                f.write(f"Sheet: {sheet}\n")
                f.write(f"Error: {str(e)}\n\n")
                f.write("-"*40 + "\n\n")

print("Results saved to:", output_file)
