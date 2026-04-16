import pandas as pd
import os
import re
import sys
from pathlib import Path

# Read CSV file
csv_file = sys.argv[1]
csv_dir = os.path.dirname(os.path.abspath(csv_file))

model_name = sys.argv[2]
df = pd.read_csv(csv_file)

# Drop columns
columns_to_drop = [col for col in df.columns 
                   if col == 'Full filepath' or col.startswith('sensitivityscore_before_')]
df = df.drop(columns=columns_to_drop)

# Extract unique alpha values from column names
alpha_pattern = r'sensitivityscore_After_.*_(\d\.\d)$'
alpha_values = set()

for col in df.columns:
    match = re.search(alpha_pattern, col)
    if match:
        alpha_values.add(match.group(1))

alpha_values = sorted(list(alpha_values))


# Create separate dataframes for each alpha value and save
for alpha in alpha_values:
    # Create output directory if it doesn't exist
    output_dir = f'after_{model_name}_{alpha}'
    output_dir = os.path.join(csv_dir, output_dir)
    Path(output_dir).mkdir(exist_ok=True)

    # Get columns matching this alpha value and 'Full Class Index'
    cols_for_alpha = ['Full Class Index'] if 'Full Class Index' in df.columns else []
    cols_for_alpha += [col for col in df.columns 
                       if col.endswith(f'_{alpha}') and col.startswith('sensitivityscore_After_')]
    
    if cols_for_alpha:
        df_alpha = df[cols_for_alpha]
        output_file = os.path.join(output_dir, f'sensitivity_{alpha}.csv')
        df_alpha.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
