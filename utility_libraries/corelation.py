

"""
Correlation Analysis and Visualization Tool
This script performs correlation analysis on CSV data grouped by class indices.
It generates correlation matrices, heatmaps, and subset analyses for each class group.
Features:
- Loads CSV data and groups by 'Full Class Index' column
- Cleans column names by removing 'sensitivityscore_before_' prefix
- Calculates correlation matrices for each class group
- Creates masked correlation matrices (lower triangle only)
- Generates correlation heatmaps with proper formatting
- Identifies top 10 positive correlations with the last row/column
- Creates subset heatmaps for the most correlated features
- Exports all correlation matrices to Excel with separate sheets
Output Files:
- correlation_matrices.xlsx: Excel file containing all correlation matrices
- correlation_heatmap_class_{i}.png: Full correlation heatmaps for each class
- subset_heatmap_class_{i}.png: Subset heatmaps showing top correlations
Requirements:
- Input CSV file must contain 'Full filepath' and 'Full Class Index' columns
- Data columns should be prefixed with 'sensitivityscore_before_'
Dependencies:
- pandas: Data manipulation and analysis
- scipy.stats: Statistical functions (imported but not used in current implementation)
- seaborn: Statistical data visualization
- matplotlib.pyplot: Plotting and visualization
- numpy: Numerical operations
Usage:
Run the script and provide the path to your CSV file when prompted.
The script will process each class group and generate visualizations and Excel output.
"""

import pandas as pd
from scipy.stats import shapiro, normaltest
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Replace 'your_file.csv' with the path to your CSV file
#file_name = input("Enter the path to your CSV file: ")
# Replace 'your_file.csv' with the path to your CSV file
if len(sys.argv) != 2:
    print("Usage: python corelation.py <csv_file_path>")
    sys.exit(1)

file_name = sys.argv[1]

df = pd.read_csv(file_name)
df = df.drop(columns=['Full filepath'])
class_groups = dict(tuple(df.groupby('Full Class Index')))

# Display the first few rows
print(df.head())
print(class_groups)
for i in class_groups:
    class_groups[i] = class_groups[i].drop(columns=['Full Class Index'])
    class_groups[i].columns = [col.replace('sensitivityscore_before_', '') for col in class_groups[i].columns]
    print(class_groups[i].head())  # Display first few rows of each class group
    # Drop the 'sensitivity' column for group i and store in a new DataFrame
    col_names = class_groups[i].columns.tolist()
    correlation_matrix = class_groups[i].corr()
    print(f"Correlation matrix for class group {i}:")
    print(correlation_matrix)
    # Create mask for upper triangle and apply it to correlation matrix
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    masked_correlation = correlation_matrix.mask(mask)
    print(f"Masked correlation matrix for class group {i} (lower triangle only):")    
    with pd.ExcelWriter('correlation_matrices.xlsx', mode='a' if i != list(class_groups.keys())[0] else 'w') as writer:
        masked_correlation.to_excel(writer, sheet_name=f'Class_{i}')
    

    # Plotting the correlation heatmap (lower triangle only)
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, 
                cbar_kws={"shrink": .8}, mask=mask)
    plt.title(f'Correlation Heatmap for Class Group {i}')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'correlation_heatmap_class_{i}.png')
    plt.close()
    print(f"Correlation heatmap for class group {i} saved as 'correlation_heatmap_class_{i}.png'")
    # Get the last row of the correlation matrix
    last_row = correlation_matrix.iloc[-1]

    # Find columns with positive correlation in the last row (excluding the last column itself)
    positive_corr_cols = last_row[last_row > 0].index.tolist()
    # Create subset matrix with only positive correlation columns (top 10)
    if len(positive_corr_cols) > 1:
        # Sort by correlation values and take top 10
        last_row_sorted = last_row[last_row > 0].sort_values(ascending=False)
        top_10_cols = last_row_sorted.head(10).index.tolist()
        subset_matrix = correlation_matrix.loc[top_10_cols, top_10_cols]
        print(f"Subset matrix for class group {i} (top 10 positive correlations with last row):")
        print(subset_matrix)
        # Plot heatmap for subset matrix with upper triangle mask
        plt.figure(figsize=(8, 6))
        mask = np.triu(np.ones_like(subset_matrix, dtype=bool))
        sns.heatmap(subset_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, 
                    cbar_kws={"shrink": 1.0}, mask=mask)
        plt.title(f'Top 15 Positive Correlation Subset Heatmap for Class Group {i}')
        #plt.xticks(rotation=45)
        #plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'subset_heatmap_class_{i}.png')
        plt.close()
        print(f"Subset heatmap for class group {i} saved as 'subset_heatmap_class_{i}.png'")
    else:
        print(f"No positive correlations found in last row for class group {i}")
    # Save subset correlation matrix to Excel file
    with pd.ExcelWriter('correlation_matrices.xlsx', mode='a') as writer:
        # Create mask for upper triangle and apply it to subset matrix
        subset_mask = np.triu(np.ones_like(subset_matrix, dtype=bool))
        masked_subset_matrix = subset_matrix.mask(subset_mask)
        masked_subset_matrix.to_excel(writer, sheet_name=f'Subset_Class_{i}')