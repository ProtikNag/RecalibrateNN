"""
ANOVA Analysis Module for Sensitivity Score Data
This module provides comprehensive statistical analysis functionality for sensitivity score data
across different neural network layers and classes. It performs ANOVA tests, post-hoc comparisons,
normality testing, and homogeneity of variance testing.
The module is designed to analyze CSV data containing sensitivity scores from neural network layers
and performs statistical comparisons between different classes and layers.
Main Functions:
- clean_data: Preprocesses the input dataframe and separates it by class
- create_melted_data: Transforms wide-format data to long-format for statistical analysis
- perform_anova_and_tukey: Conducts one-way ANOVA and Tukey's HSD post-hoc test
- perform_anova_games_howell: Performs pairwise t-tests as substitute for Games-Howell test
- perform_levenes_test: Tests homogeneity of variance assumption
- identify_suitable_anova: Selects appropriate ANOVA method based on variance homogeneity
- verify_normality: Tests normality assumptions using multiple statistical tests
Statistical Tests Implemented:
- One-way ANOVA for comparing means across groups
- Tukey's HSD for post-hoc pairwise comparisons (equal variances)
- Welch's t-test for pairwise comparisons (unequal variances)
- Levene's test for homogeneity of variance
- Shapiro-Wilk test for normality
- D'Agostino's normality test
- Jarque-Bera test for normality
Input Requirements:
- CSV file with sensitivity scores across neural network layers
- 'Full Class Index' column to separate classes
- 'Full filepath' column (removed during preprocessing)
- Layer columns with 'sensitivityscore_before_' prefix
Output:
- Excel file with multiple sheets containing:
    - Descriptive statistics for each class
    - ANOVA results and post-hoc comparisons
    - Normality test results
    - Variance homogeneity test results
    - Melted data for visualization
Usage:
        python anova.py <input_csv_file>
The script automatically generates 'anova_results.xlsx' with all statistical analysis results.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway
from itertools import combinations
import numpy as np
import sys
from scipy.stats import shapiro, normaltest, jarque_bera


"""
Clean and preprocess the input dataframe by removing unnecessary columns and splitting by class.

This function performs data cleaning operations including dropping the 'Full filepath' column,
renaming columns by removing the 'sensitivityscore_before_' prefix, and separating the data
into three distinct dataframes based on the 'Full Class Index' values (0, 1, 2).

Args:
    df (pandas.DataFrame): Input dataframe containing sensitivity score data with columns
                          including 'Full filepath', 'Full Class Index', and columns with
                          'sensitivityscore_before_' prefix.

Returns:
    tuple: A tuple containing three pandas DataFrames:
        - class_0_df (pandas.DataFrame): Data for class 0 (Full Class Index == 0)
        - class_1_df (pandas.DataFrame): Data for class 1 (Full Class Index == 1)  
        - class_2_df (pandas.DataFrame): Data for class 2 (Full Class Index == 2)

Side Effects:
    Prints the count of records in each class to the console.

Note:
    The 'Full Class Index' column is removed from the returned dataframes.
    Column names are modified to remove the 'sensitivityscore_before_' prefix.
"""
def clean_data(df):
    df = df.drop(columns=['Full filepath'])
    df.columns = [col.replace('sensitivityscore_before_', '') for col in df.columns]
    # Create three separate dataframes, one per class
    class_0_df = df[df['Full Class Index'] == 0].drop(columns=['Full Class Index']).copy()
    class_1_df = df[df['Full Class Index'] == 1].drop(columns=['Full Class Index']).copy()
    class_2_df = df[df['Full Class Index'] == 2].drop(columns=['Full Class Index']).copy()
    print(f"Class 0 count: {len(class_0_df)}")
    print(f"Class 1 count: {len(class_1_df)}")
    print(f"Class 2 count: {len(class_2_df)}")
    return class_0_df, class_1_df, class_2_df


"""
Transform a DataFrame from wide to long format using pandas melt operation.

This function converts a DataFrame from wide format to long format, making it suitable
for statistical analysis. Optionally saves the melted data to an Excel file.

Parameters
----------
df : pandas.DataFrame
    The input DataFrame to be melted from wide to long format.
var_name : str, default 'Layer'
    Name to use for the variable column in the melted DataFrame.
value_name : str, default 'Sensitivity'
    Name to use for the value column in the melted DataFrame.
sheet_name : str, optional
    Base name for the Excel sheet. If provided with output_xls, the sheet
    will be named '{sheet_name}_Melted'.
output_xls : str, optional
    Path to the Excel file where the melted DataFrame should be saved.
    If None, no file will be written.

Returns
-------
pandas.DataFrame
    The melted DataFrame in long format.

Notes
-----
- Prints the count of rows in the melted DataFrame for verification
- If output_xls is provided, appends the melted data to the Excel file
- Uses openpyxl engine for Excel writing operations
"""
def create_melted_data(df,  var_name='Layer', value_name='Sensitivity', sheet_name=None, output_xls=None):
    df_melted = pd.melt(df, var_name=var_name, value_name=value_name)
    df_count = len(df_melted)
    print(f"Count of melted df: {df_count}")
    if output_xls is not None:
        with pd.ExcelWriter(output_xls, engine='openpyxl', mode='a') as writer:
            df_melted.to_excel(writer, sheet_name=f'{sheet_name}_Melted', index=False)
    return df_melted
"""
Perform ANOVA and Tukey's HSD post-hoc test on sensitivity data across different layers.

Tukey's Honestly Significant Difference (HSD) test is a post-hoc multiple comparison 
procedure used after ANOVA to determine which specific groups differ significantly 
from each other. It controls the family-wise error rate, making it suitable for 
comparing all possible pairs of group means while maintaining statistical rigor.

When to use Tukey's HSD:
- After a significant ANOVA F-test indicates differences between groups
- When you have 3 or more groups to compare
- When sample sizes are approximately equal across groups
- When data meets ANOVA assumptions (normality, homogeneity of variance, independence)
- When you want to compare ALL pairwise combinations of groups
- When you need to control Type I error across multiple comparisons

This function fits an ordinary least squares (OLS) model to test for differences in 
sensitivity across layers, performs ANOVA to test the overall significance, and 
conducts Tukey's HSD test for pairwise comparisons between layers. Results are 
saved to an Excel file.

Parameters
----------
df_melted : pandas.DataFrame
    Melted dataframe containing 'Sensitivity' and 'Layer' columns.
    'Sensitivity' should contain numeric values representing the dependent variable.
    'Layer' should contain categorical values representing different groups/layers.
sheet_name : str
    Base name for the Excel sheets. Two sheets will be created:
    '{sheet_name}_ANOVA' for ANOVA results and '{sheet_name}_Tukey' for Tukey results.
output_xls : str
    Path to the output Excel file where results will be appended.

Returns
-------
tuple
    A tuple containing:
    - anova_results : pandas.DataFrame
        ANOVA table with sum of squares, degrees of freedom, F-statistic, and p-values.
    - tukey_df : pandas.DataFrame
        Tukey HSD results showing pairwise comparisons between layers with 
        mean differences, confidence intervals, and significance indicators.

Notes
-----
- The function uses a significance level of α = 0.05 for Tukey's HSD test.
- Results are appended to the existing Excel file specified by output_xls.
- The Excel file must be writable and accessible at the specified path.
- If ANOVA assumptions are violated (especially homogeneity of variance), 
  consider using Games-Howell test instead of Tukey's HSD.
"""
def perform_anova_and_tukey(df_melted, sheet_name, output_xls):
    model = ols('Sensitivity ~ C(Layer)', data=df_melted).fit()
    anova_results = anova_lm(model)
    tukey = pairwise_tukeyhsd(endog=df_melted['Sensitivity'], groups=df_melted['Layer'], alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    with pd.ExcelWriter(output_xls, engine='openpyxl', mode='a') as writer:
        anova_results.to_excel(writer, sheet_name=f'{sheet_name}_ANOVA')
        tukey_df.to_excel(writer, sheet_name=f'{sheet_name}_Tukey', index=False)
    return anova_results, tukey_df

"""
**When to Use This Test:**
- One-way ANOVA: Use when comparing means across 3+ independent groups
- Games-Howell (approximated by Welch's t-test): Use when group variances are unequal (heteroscedasticity)
- No assumption of equal sample sizes required
- Data should be approximately normally distributed within each group
- Groups should be independent of each other
- Post-hoc tests are only meaningful if ANOVA shows significant differences (p < 0.05)
Examples
--------
>>> df = pd.DataFrame({
...     'Layer': ['A', 'A', 'B', 'B', 'C', 'C'],
...     'Sensitivity': [1.2, 1.5, 2.1, 2.3, 3.1, 3.4]
... })
>>> f_stat, p_val = perform_anova_games_howell(df, 'test', 'output.xlsx')

Perform ANOVA analysis followed by pairwise comparisons using Welch's t-test as a substitute for Games-Howell post-hoc test.
This function conducts a one-way ANOVA to test for significant differences between groups,
then performs pairwise t-tests with unequal variances (Welch's t-test) between all layer
combinations. It calculates confidence intervals for mean differences and determines
statistical significance.
Parameters
----------
df_melted : pandas.DataFrame
    A melted dataframe containing 'Layer' and 'Sensitivity' columns.
    'Layer' represents the grouping variable and 'Sensitivity' contains the values to compare.
sheet_name : str
    Base name for the Excel sheet where results will be saved.
    The actual sheet name will be '{sheet_name}_Games_Howell'.
output_xls : str or None
    Path to the output Excel file. If None, no file output is generated.
    Results are appended to existing file if it exists.
Returns
-------
tuple
    A tuple containing:
    - gh_stat (float): F-statistic from the one-way ANOVA
    - gh_p_values (float): p-value from the one-way ANOVA
Notes
-----
- Uses Welch's t-test (equal_var=False) instead of true Games-Howell test
- Calculates 95% confidence intervals using Welch's degrees of freedom
- Results include comparison names, p-values, confidence intervals, and significance flags
- Significance threshold is set to α = 0.05
"""
def perform_anova_games_howell(df_melted, sheet_name, output_xls):
    grouped_data = [group['Sensitivity'].values for name, group in df_melted.groupby('Layer')]
    gh_stat, gh_p_values = stats.f_oneway(*grouped_data)
    
    # Perform pairwise t-tests as a substitute for Games-Howell
    layer_names = df_melted['Layer'].unique()
    pairwise_comparisons = list(combinations(layer_names, 2))
    gh_results = []
    for layer1, layer2 in pairwise_comparisons:
        group1 = df_melted[df_melted['Layer'] == layer1]['Sensitivity']
        group2 = df_melted[df_melted['Layer'] == layer2]['Sensitivity']
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        comparison_name = f"{layer1} - {layer2}"
        # Calculate confidence interval for the difference in means
        n1, n2 = len(group1), len(group2)
        pooled_se = np.sqrt((group1.var(ddof=1)/n1) + (group2.var(ddof=1)/n2))
        df_welch = ((group1.var(ddof=1)/n1) + (group2.var(ddof=1)/n2))**2 / ((group1.var(ddof=1)/n1)**2/(n1-1) + (group2.var(ddof=1)/n2)**2/(n2-1))
        t_critical = stats.t.ppf(0.975, df_welch)
        mean_diff = group1.mean() - group2.mean()
        margin_error = t_critical * pooled_se
        lower_ci = mean_diff - margin_error
        upper_ci = mean_diff + margin_error
        gh_results.append({
            'Comparison': comparison_name,
            'p-value': p_value,
            'CI_Lower': lower_ci,
            'CI_Upper': upper_ci,
            'Significant': p_value < 0.05
        })
    gh_df = pd.DataFrame(gh_results)
    if(output_xls is not None):
        with pd.ExcelWriter(output_xls, engine='openpyxl', mode='a') as writer:
            gh_df.to_excel(writer, sheet_name=f'{sheet_name}_Games_Howell', index=False)
    return gh_stat, gh_p_values

"""
Perform Levene's test for homogeneity of variance across groups.

Levene's test is used to test the null hypothesis that all input samples 
are from populations with equal variances. It is an alternative to Bartlett's 
test when the data may not be normally distributed.

When to use Levene's test:
- To test the assumption of homogeneity of variance (homoscedasticity)
- Before performing ANOVA to determine if equal variance assumption is met
- When data may not be normally distributed (more robust than Bartlett's test)
- To decide between standard ANOVA (equal variances) vs. Welch's ANOVA (unequal variances)

Parameters
----------
df_melted : pandas.DataFrame
    A melted dataframe containing 'Layer' and 'Sensitivity' columns.
    'Layer' represents the grouping variable and 'Sensitivity' contains the values to test.
sheet_name : str, optional
    Base name for the Excel sheet where results will be saved.
    The actual sheet name will be '{sheet_name}_Levenes_Test'.
output_xls : str, optional
    Path to the output Excel file. If None, no file output is generated.
    Results are appended to existing file if it exists.

Returns
-------
tuple
    A tuple containing:
    - levene_stat (float): The test statistic
    - levene_p_value (float): The p-value for the test

Notes
-----
- Null hypothesis: All groups have equal variances
- Alternative hypothesis: At least one group has different variance
- If p-value < 0.05, reject null hypothesis (variances are not equal)
- If variances are unequal, consider using Games-Howell or Welch's test instead of standard ANOVA
"""
def perform_levenes_test(df_melted, sheet_name=None, output_xls=None):
    grouped_data = [group['Sensitivity'].values for name, group in df_melted.groupby('Layer')]
    levene_stat, levene_p_value = stats.levene(*grouped_data)
    
    if output_xls is not None and sheet_name is not None:
        levene_results = pd.DataFrame({
            'Levene_Statistic': [levene_stat],
            'Levene_p_value': [levene_p_value],
            'Equal_Variances': [levene_p_value > 0.05],
            'Interpretation': ['Equal variances assumed' if levene_p_value > 0.05 else 'Unequal variances - use Games-Howell']
        })
        
        with pd.ExcelWriter(output_xls, engine='openpyxl', mode='a') as writer:
            levene_results.to_excel(writer, sheet_name=f'{sheet_name}_Levenes_Test', index=False)
    
    return levene_stat, levene_p_value

"""
Determine and perform the appropriate ANOVA test based on homogeneity of variance.
This function performs Levene's test to check for homogeneity of variance and selects
the appropriate post-hoc analysis method. If variances are not homogeneous (p < 0.05),
it performs ANOVA with Games-Howell post-hoc test. Otherwise, it performs ANOVA with
Tukey's HSD post-hoc test.
Parameters
----------
df_melted : pandas.DataFrame
    Melted dataframe containing the data for ANOVA analysis
sheet_name : str
    Name of the Excel sheet where results will be written
output_xls : str or ExcelWriter
    Output Excel file path or ExcelWriter object for saving results
Returns
-------
anova_results : dict or pandas.DataFrame
    Results of the ANOVA analysis and post-hoc tests
Notes
-----
- Uses Levene's test with alpha = 0.05 as the threshold for variance homogeneity
- Games-Howell test is used when variances are unequal (robust to heteroscedasticity)
- Tukey's HSD test is used when variances are equal (assumes homoscedasticity)
"""
def identify_suitable_anova(df_melted, sheet_name, output_xls):
    levene_stat, levene_p_value = perform_levenes_test(df_melted)
    
    if(levene_p_value < 0.05):
        anova_results = perform_anova_games_howell(df_melted, sheet_name, output_xls)
    else:
        anova_results = perform_anova_and_tukey(df_melted, sheet_name, output_xls)
    return anova_results
"""
Verify normality of numeric columns in a DataFrame using multiple statistical tests.

This function performs three normality tests (Shapiro-Wilk, D'Agostino's, and Jarque-Bera) 
on each numeric column of the input DataFrame to determine if the data follows a normal 
distribution. Results can optionally be saved to an Excel file.

Parameters:
-----------
df : pandas.DataFrame
    The input DataFrame containing data to test for normality
sheet_name : str
    Name identifier for the class/sheet, used for labeling results
output_xls : str or None
    Path to Excel file where results should be saved. If None, no file output is generated

Returns:
--------
list of dict
    A list of dictionaries containing normality test results for each numeric column.
    Each dictionary contains:
    - 'Class': The sheet_name identifier
    - 'Column': Column name being tested
    - 'Shapiro_Stat': Shapiro-Wilk test statistic
    - 'Shapiro_p_value': Shapiro-Wilk p-value
    - 'DAgostino_Stat': D'Agostino test statistic  
    - 'DAgostino_p_value': D'Agostino p-value
    - 'Normal_Shapiro': Boolean indicating normality (p > 0.05) for Shapiro-Wilk
    - 'Normal_DAgostino': Boolean indicating normality (p > 0.05) for D'Agostino
    - 'Jarque-Bera test_Stat': Jarque-Bera test statistic
    - 'Jarque-Bera test_p_value': Jarque-Bera p-value
    - 'Normal_Jarque-Bera': Boolean indicating normality (p > 0.05) for Jarque-Bera

Notes:
------
- Only numeric columns are processed; non-numeric columns are skipped
- Columns with fewer than 3 non-NaN values are skipped
- NaN values are automatically removed before testing
- Uses significance level of 0.05 for normality determination
- If output_xls is provided, results are appended to the Excel file
"""
def verify_normality(df,sheet_name, output_xls):
    normality_results = []
    for col in df.columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Remove NaN values for testing
        col_data = df[col].dropna()
        if len(col_data) < 3:  # Need at least 3 values for normality tests
            continue
        # Perform Shapiro-Wilk test
        #The Shapiro-Wilk test is a statistical test used to determine if a data sample comes from a normally distributed population
        shapiro_stat, shapiro_p = shapiro(col_data)
        # Perform D'Agostino's normality test
        dagostino_stat, dagostino_p = normaltest(col_data)
        # Perform Jarque-Bera test
        jb_stat, jb_p = jarque_bera(col_data)
        normality_results.append({
            'Class': sheet_name,
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
        })
    if(output_xls is not None):
        normality_df = pd.DataFrame(normality_results)
        with pd.ExcelWriter(output_xls, engine='openpyxl', mode='a') as writer:
            normality_df.to_excel(writer, sheet_name=sheet_name, index=False)
    return normality_results

def save_bargraph_data(desc_stats_transposed, i):
    pass

if(__name__ == '__main__'):
    if len(sys.argv) < 3:
        print("Usage: python anova.py <input_csv_file example input.csv> <output_xls_file example output.xlsx>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_xls = sys.argv[2]
    # Create an empty Excel file
    with pd.ExcelWriter(output_xls, engine='openpyxl') as writer:
        pd.DataFrame().to_excel(writer, sheet_name='Sheet1', index=False)
    df = pd.read_csv(input_csv)
    print(df.head())
    class_0_df, class_1_df, class_2_df = clean_data(df)
    
    #Get descriptive statistics of each class and store it in the excel file
    with pd.ExcelWriter(output_xls, engine='openpyxl', mode='a') as writer:
        class_0_df.describe().to_excel(writer, sheet_name='Class_0_Descriptive_Stats')
        class_1_df.describe().to_excel(writer, sheet_name='Class_1_Descriptive_Stats')
        class_2_df.describe().to_excel(writer, sheet_name='Class_2_Descriptive_Stats')
        average_df = pd.DataFrame({
            'Class 0 Mean': class_0_df.mean(),
            'Class 1 Mean': class_1_df.mean(),
            'Class 2 Mean': class_2_df.mean()
        })
        average_df.to_excel(writer, sheet_name='Class_Averages')

    class_0_melted = create_melted_data(class_0_df, sheet_name='Class0_Melted', output_xls=output_xls)
    
    class_1_melted = create_melted_data(class_1_df, sheet_name='Class1_Melted', output_xls=output_xls)
    class_2_melted = create_melted_data(class_2_df, sheet_name='Class2_Melted', output_xls=output_xls)


    #Perform ANOVA and Tukey's HSD test for each class and store the results in the excel file

    identify_suitable_anova(class_0_melted, 'Class0_Anova', output_xls)
    identify_suitable_anova(class_1_melted, 'Class1_Anova', output_xls)
    identify_suitable_anova(class_2_melted, 'Class2_Anova', output_xls)
    verify_normality(class_0_df, 'Class0_Normality', output_xls)
    verify_normality(class_1_df, 'Class1_Normality', output_xls)
    verify_normality(class_2_df, 'Class2_Normality', output_xls)
