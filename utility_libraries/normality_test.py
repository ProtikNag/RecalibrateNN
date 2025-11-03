# Copyright (c) 2025 Srikanth K S
# All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Normality Testing Analysis Script
This script performs comprehensive normality testing on sensitivity score data grouped by class indices.
It applies multiple statistical tests to assess whether data distributions follow a normal distribution
and generates visualizations and Excel reports for analysis.
Overview:
--------
The script reads CSV data containing sensitivity scores, groups the data by 'Full Class Index',
and performs normality testing on each numeric column within each class group. It generates
histograms, Q-Q plots, and saves detailed statistical results to Excel files.
Normality Tests Implemented:
---------------------------
1. Shapiro-Wilk Test:
    - Purpose: Tests the null hypothesis that data comes from a normal distribution
    - Best for: Small to medium sample sizes (n < 5000)
    - Statistic: W-statistic ranges from 0 to 1, where 1 indicates perfect normality
    - Interpretation: Higher W values suggest more normal distribution
    - p-value > 0.05: Data is likely normally distributed
    - p-value ≤ 0.05: Data significantly deviates from normal distribution
2. D'Agostino's Normality Test:
    - Purpose: Combines tests for skewness and kurtosis to assess normality
    - Best for: Larger sample sizes where Shapiro-Wilk may be less reliable
    - Statistic: Chi-square statistic based on skewness and kurtosis
    - Interpretation: Tests whether skewness and kurtosis match normal distribution
    - p-value > 0.05: Data is consistent with normal distribution
    - p-value ≤ 0.05: Data shows significant departure from normality
3. Jarque-Bera Test:
    - Purpose: Tests normality based on sample skewness and kurtosis
    - Best for: Large sample sizes (asymptotically valid)
    - Statistic: JB statistic follows chi-square distribution with 2 degrees of freedom
    - Theory: Normal distribution has skewness = 0 and kurtosis = 3
    - p-value > 0.05: Data is consistent with normal distribution
    - p-value ≤ 0.05: Data significantly deviates from normal distribution
Statistical Parameters Explained:
--------------------------------
Shapiro_Stat (W-statistic):
     - Range: 0 to 1
     - Higher values indicate closer approximation to normal distribution
     - Values close to 1 suggest normality
Shapiro_p_value:
     - Probability of observing the test statistic under null hypothesis
     - Null hypothesis: Data comes from normal distribution
     - Threshold: 0.05 for significance testing
DAgostino_Stat:
     - Chi-square statistic combining skewness and kurtosis tests
     - Larger values indicate greater deviation from normality
     - Based on standardized measures of distribution shape
DAgostino_p_value:
     - Combined p-value from skewness and kurtosis tests
     - Tests whether distribution shape matches normal distribution
     - Threshold: 0.05 for significance testing
Normal_Shapiro & Normal_DAgostino:
     - Boolean indicators of normality based on p-value > 0.05
     - True: Data passes normality test
     - False: Data fails normality test (significantly non-normal)
Jarque-Bera test_Stat:
     - JB = n/6 * (S² + (K-3)²/4) where:
        - n = sample size
        - S = skewness
        - K = kurtosis
     - Follows chi-square distribution with 2 degrees of freedom
     - Higher values indicate greater deviation from normality
Jarque-Bera test_p_value:
     - Probability of observing JB statistic under normality assumption
     - Low p-values (≤ 0.05) indicate rejection of normality
     - High p-values (> 0.05) suggest data is consistent with normal distribution
Output Files:
------------
- PNG files: Histograms and Q-Q plots for visual normality assessment
- Excel file: Comprehensive normality test results for all classes and columns
- Console output: Detailed results and data previews
Usage:
------
Run the script and provide the path to a CSV file containing:
- 'Full Class Index': Column for grouping data
- 'Full filepath': Column to be dropped
- Numeric columns with 'sensitivityscore_before_' prefix for analysis
Dependencies:
------------
- pandas: Data manipulation and analysis
- scipy.stats: Statistical tests (shapiro, normaltest, jarque_bera)
- matplotlib.pyplot: Plotting histograms and Q-Q plots
- seaborn: Statistical data visualization
- numpy: Numerical computing
- openpyxl: Excel file operations
"""
import pandas as pd
from scipy.stats import shapiro, normaltest
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import jarque_bera

# Replace 'your_file.csv' with the path to your CSV file
file_name = input("Enter the path to your CSV file: ")
df = pd.read_csv(file_name)
df = df.drop(columns=['Full filepath'])
class_groups = dict(tuple(df.groupby('Full Class Index')))
create_plots=False
# Display the first few rows
print(df.head())
print(class_groups)
for i in class_groups:
    class_groups[i] = class_groups[i].drop(columns=['Full Class Index'])
    class_groups[i].columns = [col.replace('sensitivityscore_before_', '') for col in class_groups[i].columns]
    print(class_groups[i].head())  # Display first few rows of each class group
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
    # Normality check for each column in data
    normality_results = []
    for col in data.columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(data[col]):
            continue
        # Remove NaN values for testing
        col_data = data[col].dropna()
        if len(col_data) < 3:  # Need at least 3 values for normality tests
            continue
        # Perform Shapiro-Wilk test
        #The Shapiro-Wilk test is a statistical test used to determine if a data sample comes from a normally distributed population
        shapiro_stat, shapiro_p = shapiro(col_data)
        # Perform D'Agostino's normality test
        dagostino_stat, dagostino_p = normaltest(col_data)
        # Perform Jarque-Bera test
        jb_stat, jb_p = jarque_bera(col_data)
        print(col)
        
        # Store results
        # Find the descriptive stats for this column
        desc_row = desc_stats_transposed.loc[desc_stats_transposed['Column'] == col]
        if not desc_row.empty:
            desc_dict = desc_row.iloc[0].to_dict()
        else:
            desc_dict = {}
        
        normality_results.append({
            'Class': i,
            'Column': col,
            'Shapiro_Stat': shapiro_stat,
            'Shapiro_p_value': shapiro_p,
            'DAgostino_Stat': dagostino_stat,
            'DAgostino_p_value': dagostino_p,
            'Normal_Shapiro': shapiro_p > 0.05,
            'Normal_DAgostino': dagostino_p > 0.05,
            'Jarque-Bera test_Stat': jb_stat,
            'Jarque-Bera test_p_value': jb_p,
            'Normal_Jarque-Bera': jb_p > 0.05,
            'count': desc_dict.get('count', np.nan),
            'mean': desc_dict.get('mean', np.nan),
            'std': desc_dict.get('std', np.nan),
            'min': desc_dict.get('min', np.nan),
            '25%': desc_dict.get('25%', np.nan),
            '50%': desc_dict.get('50%', np.nan),
            '75%': desc_dict.get('75%', np.nan),
            'max': desc_dict.get('max', np.nan)
        })
        if(create_plots==True):
            # Create plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            ax1.hist(col_data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'Histogram - Class {i}, Column {col}')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Density')
            
            # Q-Q plot for normality
            stats.probplot(col_data, dist="norm", plot=ax2)
            ax2.set_title(f'Q-Q Plot - Class {i}, Column {col}')
            
            plt.tight_layout()
            plt.savefig(f'histogram_class_{i}_column_{col}.png', dpi=300, bbox_inches='tight')
    # Convert results to DataFrame
    normality_df = pd.DataFrame(normality_results)
    print(f"\nNormality test results for Class {i}:")
    print(normality_df)
    # Save the normality results to Excel file - create or append to existing file
    excel_filename = 'normality_results_all_classes.xlsx'
    if i == list(class_groups.keys())[0]:  # First class - create new file
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            normality_df.to_excel(writer, sheet_name=f'Class_{i}', index=False)
    else:  # Subsequent classes - append to existing file
        with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a') as writer:
            normality_df.to_excel(writer, sheet_name=f'Class_{i}', index=False)
