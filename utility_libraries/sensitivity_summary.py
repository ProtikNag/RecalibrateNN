import pandas as pd
import sys
import os
import numpy as np
from scipy import stats
image_counts = {"0": 704, "1": 984, "2": 1633}
# 1. Open the CSV file
# Get CSV filename from command line argument
if len(sys.argv) < 2:
    print("Usage: python sensitivity_summary.py <root_directory>")
    sys.exit(1)
# Get source directory from command line argument
source_dir = sys.argv[1]
print(f"Source directory: {source_dir}")
# Find all CSV files in subdirectories
csv_files = []
for root, dirs, files in os.walk(source_dir):
    print(f"Searching in: {root}")
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

print(csv_files)
if not csv_files:
    print(f"No CSV files found in {source_dir}")
    sys.exit(1)

print(f"Found {len(csv_files)} CSV file(s):")
for csv_file in csv_files:
    print(f"  - {csv_file}")
# Process the first CSV file (or modify to loop through all)
# Create a list to store all dataframes
all_results = []

for csv_file in csv_files:
    print(f"\nProcessing: {csv_file}")   
    df = pd.read_csv(csv_file)
    # 2. Drop the column 'Full filepath' 
    df = df.drop(columns=['Full filepath'])
    # Get sensitivity columns
    sensitivity_cols = [col for col in df.columns if col.startswith('sensitivityscore_before_') ]
    print("Sensitivity columns found:", sensitivity_cols)
    # 3 & 4. Group by "Full Class Index" and count entries
    class_counts = df.groupby('Full Class Index').size().to_dict()
    print("Class counts:", class_counts)
    # Prepare results list
    results = []
    # Create a dictionary to store data for each layer
    layer_data = {}

    for col in sensitivity_cols:
        grouped = df.groupby('Full Class Index')[col]
        print(f"Analyzing column: {col}")
        # 4. Count values > 0 for each class
        positive_counts = grouped.apply(lambda x: (x > 0).sum()).to_dict()
        print("Positive counts:", positive_counts)
        
        # 5. Count values < 0 for each class
        negative_counts = grouped.apply(lambda x: (x < 0).sum()).to_dict()
        
        # 7. Remove 'sensitivity_before' prefix
        layer_name = col.replace('sensitivityscore_before_', '').replace('sensitivityscore_before_', '')
        
        # Store sensitivity index for each class in this layer
        sensitivity_indices = {}
        for class_idx in class_counts.keys():
            total_images = class_counts[class_idx]
            num_positive = positive_counts.get(class_idx, 0)
            print(f"Class {class_idx}: {num_positive} positive out of {total_images} images")
            sensitivity_index = num_positive / total_images if total_images > 0 else 0
            
            # Binomial hypothesis test (H0: p = 0.5)
            
            p0 = 0.5  # null hypothesis probability
            n = total_images
            x = num_positive
            
            if n > 0:
                # Z-score for proportion test
                p_hat = x / n
                se = np.sqrt(p0 * (1 - p0) / n)
                z_score = (p_hat - p0) / se if se > 0 else 0
                
                # Two-tailed p-value
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                # 95% confidence interval for the proportion
                ci_se = np.sqrt(p_hat * (1 - p_hat) / n)
                ci_lower = p_hat - 1.96 * ci_se
                ci_upper = p_hat + 1.96 * ci_se
                
                print(f"Class {class_idx}, Layer {col}:")
                print(f"  Sensitivity Index = {sensitivity_index:.4f} ({num_positive}/{total_images})")
                print(f"  Z-score = {z_score:.4f}, p-value = {p_value:.4f}")
                print(f"  95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
                
                # Store as tuple or dict for later use
                sensitivity_indices[class_idx] = {
                    'sensitivity_index': sensitivity_index,
                    'z_score': z_score,
                    'p_value': p_value,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                }
            else:
                sensitivity_indices[class_idx] = {
                    'sensitivity_index': 0,
                    'z_score': 0,
                    'p_value': 1.0,
                    'ci_lower': 0,
                    'ci_upper': 0
                }
        
        layer_data[layer_name] = sensitivity_indices

    # Create results dataframe with classes as columns
    results_df = pd.DataFrame(layer_data).T
    results_df.index.name = 'Layer Name'
    results_df.reset_index(inplace=True)
    
    # Store dataframe with source filename
    results_df['Source File'] = os.path.basename(csv_file)
    results_df['Source Directory'] = os.path.basename(os.path.dirname(csv_file))
    results_df['Parent Directory'] = os.path.basename(os.path.dirname(os.path.dirname(csv_file)))
    all_results.append(results_df)

# Combine all results into a single Excel file with different worksheets
if not all_results:
    print("No results to save")
    sys.exit(1)

# Create output filename based on source directory or first CSV file
if csv_files:
    out_filename = os.path.join(source_dir, "sensitivity_analysis_summary.xlsx")
else:
    out_filename = "sensitivity_analysis_summary.xlsx"

# Save all results to a single Excel file with multiple sheets
with pd.ExcelWriter(out_filename, engine='openpyxl') as writer:
    for i, result_df in enumerate(all_results):
        # Use the source filename as sheet name (without extension)
        source_name = os.path.splitext(result_df['Source File'].iloc[0])[0]
        sheet_name = source_name[:31]  # Excel sheet name limit is 31 characters
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Analysis complete. Results saved to '{out_filename}' with {len(all_results)} sheet(s)")