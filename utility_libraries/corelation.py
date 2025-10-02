import pandas as pd
import sys
if len(sys.argv) < 2:
    print("Usage: python corelation.py <csv_file>")
    sys.exit(1)
csv_file = sys.argv[1]


#csv_file = 'sensitivity_audit_trail_vgg16_20250927_222941.csv'
df = pd.read_csv(csv_file)

# Dictionary to hold DataFrames for each class index
dfs_by_class = {}

# Split the data based on "Full Class Index"
for class_index in range(10):
    dfs_by_class[class_index] = df[df['Full Class Index'] == class_index]

# Compute Pearson correlation for each DataFrame
correlations = {}
for class_index, class_df in dfs_by_class.items():
    #class_df = class_df.drop(columns=['Full FilePath', 'Full Class Index'])
    
    # Drop non-numeric columns for correlation calculation
    numeric_df = class_df.select_dtypes(include='number').drop(columns=['Full Class Index'])
    correlations[class_index] = numeric_df.corr(method='pearson')
    numeric_df['class'] = class_index
    # Example: print correlation for class 0
    print(f"Pearson correlation for class {class_index}:")
    print(correlations[class_index])
    with pd.ExcelWriter('numeric_dfs_by_class.xlsx', mode='a' if class_index > 0 else 'w') as writer:
        numeric_df.to_excel(writer, sheet_name=f'class_{class_index}', index=False)