"""
Performs comprehensive ANOVA analysis and post-hoc testing on sensitivity analysis data.
This module implements a complete statistical analysis workflow including:
1. Data Preprocessing:
    - Loads CSV data containing sensitivity analysis results
    - Groups data by 'Full Class Index' 
    - Cleans column names by removing 'sensitivityscore_before_' prefix
    - Removes non-numeric columns ('Full filepath', 'sensitivity')
2. Descriptive Statistics:
    - Calculates summary statistics (mean, std, min, max, quartiles) for each class group
    - Transposes results for better readability in Excel format
    - Adds class identifiers to facilitate comparison across groups
3. One-Way ANOVA Implementation:
    - Performs one-way ANOVA using scipy.stats.f_oneway() for each variable
    - Tests the null hypothesis that all class groups have equal means
    - Compares each variable across all available class groups
    - Calculates F-statistic and p-values to determine statistical significance
    - Sorts results by p-value to identify most significant differences
4. Post-Hoc Analysis (Tukey's HSD):
    - Conducts pairwise comparisons using Tukey's Honestly Significant Difference test
    - Applied only to variables showing significant ANOVA results (p < 0.05)
    - Controls family-wise error rate for multiple comparisons
    - Identifies which specific class pairs differ significantly
    - Focuses on top 3 most significant variables for detailed analysis
The ANOVA method tests whether there are statistically significant differences in 
sensitivity scores between different class groups, while Tukey's HSD determines 
which specific pairs of classes differ when overall significance is detected.
Input Requirements:
- CSV file with 'Full Class Index' column for grouping
- Numeric columns prefixed with 'sensitivityscore_before_'
- Minimum 2 class groups for meaningful ANOVA analysis
Output:
- Descriptive statistics for each class group
- ANOVA F-statistics and p-values for all variables
- Tukey's HSD pairwise comparison results for significant variables
"""
import pandas as pd
from scipy.stats import shapiro, normaltest
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import jarque_bera
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

# Replace 'your_file.csv' with the path to your CSV file
file_name = input("Enter the path to your CSV file: ")
df = pd.read_csv(file_name)
df = df.drop(columns=['Full filepath'])
# Create three separate dataframes, one per class
class_0_df = df[df['Full Class Index'] == 0].copy()
class_1_df = df[df['Full Class Index'] == 1].copy()
class_2_df = df[df['Full Class Index'] == 2].copy()

class_0_df.columns = [col.replace('sensitivityscore_before_', '') for col in class_0_df.columns]
class_0_df = class_0_df.drop(columns=['Full Class Index'])
class_1_df.columns = [col.replace('sensitivityscore_before_', '') for col in class_1_df.columns]
class_1_df = class_1_df.drop(columns=['Full Class Index'])
class_2_df.columns = [col.replace('sensitivityscore_before_', '') for col in class_2_df.columns]
class_2_df = class_2_df.drop(columns=['Full Class Index'])

# Melt the dataframe to create two columns: feature names and their values
melted_data = class_0_df.melt(var_name='Feature', value_name='Value')
# Perform one-way ANOVA comparing the two features
feature_0_values = melted_data[melted_data['Feature'] == 'feature_0']['Value']
feature_1_values = melted_data[melted_data['Feature'] == 'feature_1']['Value']
f_stat, p_value = f_oneway(*melted_data)
print(f"ANOVA results for Class {i}: F-statistic = {f_stat}, p-value = {p_value}")

# Check your melted_data structure
print("Number of groups:", len(melted_data))
for i, group in enumerate(melted_data):
    print(f"Group {i}: length={len(group)}, unique_values={len(set(group))}")
    print(f"  Values: {group[:5]}...")  # Show first 5 values
    
# Check if all values are the same across groups
all_values = [val for group in melted_data for val in group]
print(f"Total unique values across all groups: {len(set(all_values))}")


print(f"Class 0 DataFrame shape: {class_0_df.shape}")
print(f"Class 1 DataFrame shape: {class_1_df.shape}")
print(f"Class 2 DataFrame shape: {class_2_df.shape}")
class_groups = dict(tuple(df.groupby('Full Class Index')))
create_plots=False
# Display the first few rows
print(df.head())
print(class_groups)
excel_filename = "anova_results_all_classes.xlsx"
for i in class_groups:
    class_groups[i] = class_groups[i].drop(columns=['Full Class Index'])
    class_groups[i].columns = [col.replace('sensitivityscore_before_', '') for col in class_groups[i].columns]
    #print(class_groups[i].head())  # Display first few rows of each class group
    # Drop the 'sensitivity' column for group i and store in a new DataFrame
    col_names = class_groups[i].columns.tolist()
    data = class_groups[i][col_names]
    data = data.drop(columns=['sensitivity'], errors='ignore')
    print(data.head())  # Display first few rows of the new DataFrame
    print(data.keys())
    # Calculate and print descriptive statistics
    desc_stats = data.describe()
    print(f"\nDescriptive Statistics for Class {i}:")
    print(desc_stats)
    # Prepare descriptive statistics for Excel (transpose for better readability)
    desc_stats_transposed = desc_stats.T
    desc_stats_transposed.reset_index(inplace=True)
    desc_stats_transposed.rename(columns={'index': 'Column'}, inplace=True)
    desc_stats_transposed['Class'] = i
    # Reorder columns to have Class first
    cols = ['Class'] + [col for col in desc_stats_transposed.columns if col != 'Class']
    desc_stats_transposed = desc_stats_transposed[cols]
    # Write descriptive statistics to Excel file
    with pd.ExcelWriter(excel_filename, mode='a' if i != list(class_groups.keys())[0] else 'w', engine='openpyxl') as writer:
        desc_stats_transposed.to_excel(writer, sheet_name=f'Class_{i}_Stats', index=False)
    # Prepare data for ANOVA - melt the dataframe to long format
    melted_data = data.melt(var_name='Feature', value_name='Value')
    # --- 1️⃣ One-Way ANOVA ---
    groups = [group["Value"].values for _, group in melted_data.groupby("Feature")]
    f_stat, p_value = f_oneway(*groups)
    print(f"\nClass {i} → One-Way ANOVA: F = {f_stat:.4f}, p = {p_value:.6f}")
    # Save ANOVA summary (using statsmodels for a detailed table)
    model = ols('Value ~ C(Feature)', data=melted_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table.reset_index(inplace=True)
    anova_table.rename(columns={'index': 'Source'}, inplace=True)
    anova_table['Class'] = i
    cols = ['Class'] + [c for c in anova_table.columns if c != 'Class']
    anova_table = anova_table[cols]
    
    
    
    # --- 2️⃣ Tukey HSD Post-Hoc Test ---
    if p_value < 0.05:
        tukey = pairwise_tukeyhsd(endog=melted_data['Value'],
                                  groups=melted_data['Feature'],
                                  alpha=0.05)
        tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_df['Class'] = i

        print(f"Tukey HSD Results for Class {i}:")
        print(tukey_df.head())

        # --- 3️⃣ Save ANOVA + Tukey Results to Excel ---
        with pd.ExcelWriter(excel_filename, mode='a' if i != list(class_groups.keys())[0] else 'w',
                            engine='openpyxl') as writer:
            anova_table.to_excel(writer, sheet_name=f'Class_{i}_ANOVA', index=False)
            tukey_df.to_excel(writer, sheet_name=f'Class_{i}_Tukey', index=False)

        # --- 4️⃣ Plot Tukey Simultaneous Confidence Intervals ---
        fig = tukey.plot_simultaneous(comparison_name=tukey.groupsunique[0],
                                      xlabel='Mean difference',
                                      ylabel='Feature')
        plt.title(f"Tukey HSD 95% CI for Class {i}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Save each class Tukey plot
        plot_path = f"Tukey_Class_{i}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved Tukey plot for Class {i} → {plot_path}")

    else:
        print(f"No significant differences found for Class {i} (p = {p_value:.4f}). Skipping Tukey test.")
        with pd.ExcelWriter(excel_filename, mode='a' if i != list(class_groups.keys())[0] else 'w',
                            engine='openpyxl') as writer:
            anova_table.to_excel(writer, sheet_name=f'Class_{i}_ANOVA', index=False)
