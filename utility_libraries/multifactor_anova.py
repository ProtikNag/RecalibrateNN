"""
Multifactor ANOVA Analysis for Neural Network Layer Sensitivity Scores
This script performs comprehensive statistical analysis on sensitivity scores across different
neural network layers using ANOVA (Analysis of Variance) techniques. It handles both equal
and unequal variance scenarios with appropriate statistical tests and post-hoc analyses.
Key Features:
- Loads and preprocesses sensitivity score data from CSV files
- Performs data transformation from wide to long format for ANOVA analysis
- Conducts Levene's test to check homogeneity of variances
- Applies appropriate ANOVA methods based on variance equality:
    * Standard ANOVA with Tukey HSD for equal variances
    * Welch's ANOVA with Bonferroni-corrected pairwise t-tests for unequal variances
- Validates ANOVA assumptions (normality and homoscedasticity)
- Generates visualizations and exports results to CSV files
Statistical Tests Performed:
- Levene's Test: Checks equality of variances across groups
- One-way ANOVA or Welch's ANOVA: Tests for significant differences between layer means
- Tukey HSD or Bonferroni-corrected t-tests: Post-hoc pairwise comparisons
- Shapiro-Wilk Test: Validates normality assumption of residuals
Input Requirements:
- CSV file containing sensitivity scores with columns for different layers
- 'Full Class Index' column for class filtering
- 'Full filepath' column (will be dropped during preprocessing)
Output Files Generated:
- class_0_melted.csv: Melted data in long format
- grouped_data_by_layer.csv: Data grouped by layer
- tukey_hsd_results.csv: Tukey HSD pairwise comparison results (if applicable)
Dependencies:
- pandas: Data manipulation and analysis
- seaborn, matplotlib: Data visualization
- scipy.stats: Statistical tests
- statsmodels: ANOVA and regression analysis
- numpy: Numerical computations
- itertools: Combination generation for pairwise tests
Usage:
Modify the 'file_name' variable to point to your sensitivity score CSV file.
Set _DEBUG to True for additional output files and debugging information.
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
_DEBUG = True
# Replace 'your_file.csv' with the path to your CSV file
#file_name = input("Enter the path to your CSV file: ")
file_name = r'C:\Users\srikant1\Downloads\paper2\sensitivity_all\vgg16\sensitivity_audit_trail_vgg16_20251027_074803.csv'
df = pd.read_csv(file_name)
df = df.drop(columns=['Full filepath'])
df.columns = [col.replace('sensitivityscore_before_', '') for col in df.columns]

# Create three separate dataframes, one per class
class_0_df = df[df['Full Class Index'] == 0].drop(columns=['Full Class Index']).copy()

# Melt class_0_df to convert from wide to long format
class_0_melted = pd.melt(class_0_df, var_name='Layer', value_name='Sensitivity')
class_0_count = len(class_0_melted)
print(f"Count of melted class_0_df: {class_0_count}")

if(_DEBUG):
    class_0_melted.to_csv('class_0_melted.csv', index=False)

#Layer is treated as catogorical variable    
#ols: This stands for Ordinary Least Squares, which is the "engine" used to find the best-fitting line.
#'Sensitivity ~ C(Layer)': This is the model's formula, which you read as "Sensitivity is predicted by Layer."
#Sensitivity: This is your dependent variable—the numerical value you are measuring. ~: The tilde symbol means "is predicted by."
#C(Layer): This is your independent variable or factor. The C() wrapper is very important: it tells the model to treat your Layer column 
# as a Categorical variable (e.g., 'conv1', 'conv2', 'fc1'), not a continuous number. This is what makes it an ANOVA.
#data=class_0_melted: This tells the model to find the Sensitivity and Layer columns inside your melted DataFrame.
#anova_results = anova_lm(model, typ=2)
#anova_lm(model, ...): This function is "Analysis of Variance for a Linear Model." It takes the model you just created and calculates the key ANOVA statistics.
#typ=2: This specifies the "Type 2 Sum of Squares." In simple terms, this is the standard and correct method to use for a model like yours 
# (a one-way ANOVA with no interaction terms). It tests the main effect of Layer on Sensitivity
#If PR(>F) is very small (e.g., < 0.05): It means there is a statistically significant difference in mean sensitivity between at least some of your layers.


model = ols('Sensitivity ~ C(Layer)', data=class_0_melted).fit()
anova_results = anova_lm(model, typ=2)

# IN case the variance are unequal
# Get the data for each group (layer)
# This assumes your 'class_0_melted' has columns 'Sensitivity' and 'Layer'
grouped_data = [group['Sensitivity'] for name, group in class_0_melted.groupby('Layer')]
# Store grouped_data in a CSV file
grouped_df = pd.DataFrame(dict([(name, pd.Series(group['Sensitivity'])) for name, group in class_0_melted.groupby('Layer')]))
grouped_df.to_csv('grouped_data_by_layer.csv', index=False)
print(f"Grouped data saved to 'grouped_data_by_layer.csv'")

# --- 1. Prepare the Data ---
# Levene's test requires each group's data as a separate argument.
# We can create a list of arrays, where each array is the 
# 'Sensitivity' data for one 'Layer'.

try:
    grouped_data = [group['Sensitivity'].values for name, group in class_0_melted.groupby('Layer')]

    # --- 2. Run Levene's Test ---
    # The * operator unpacks the list, so each group's array
    # is passed as a separate argument to the function.
    levene_stat, levene_p_value = stats.levene(*grouped_data)

    print(f"Levene's Test Statistic: {levene_stat:.4f}")
    print(f"P-value: {levene_p_value:.4g}") # .4g formats the p-value nicely

    # --- 3. Interpretation ---
    if levene_p_value < 0.05:
        print("\nConclusion: The p-value is less than 0.05.")
        print("The variances are significantly DIFFERENT (unequal).")
        print("Recommendation: Use Welch's ANOVA or a non-parametric test.")
        # Run Welch's ANOVA
        f_stat, p_value = anova_oneway(grouped_data, use_var="unequal")
        # Interpretation
        if p_value < 0.05:
            print("Result: Significant difference between layers (p < 0.05)")
        else:
            print("Result: No significant difference between layers (p >= 0.05)")
        # For unequal variances, we need to use a non-parametric approach for post-hoc testing
        # since Tukey HSD assumes equal variances
        print("\n========== Games-Howell Post-hoc Test (for unequal variances) ==========")

        # Since statsmodels doesn't have Games-Howell, we'll use a pairwise approach
        # or inform the user about the limitation

        # Get unique layers
        layers = class_0_melted['Layer'].unique()
        layer_pairs = list(combinations(layers, 2))

        print("Pairwise comparisons using Welch's t-test (Bonferroni corrected):")
        print("Layer1\t\tLayer2\t\tt-stat\t\tp-value\t\tAdjusted p\tSignificant")
        print("-" * 80)

        # Calculate number of comparisons for Bonferroni correction
        n_comparisons = len(layer_pairs)
        alpha_corrected = 0.05 / n_comparisons

        for layer1, layer2 in layer_pairs:
            group1 = class_0_melted[class_0_melted['Layer'] == layer1]['Sensitivity']
            group2 = class_0_melted[class_0_melted['Layer'] == layer2]['Sensitivity']
            
            # Welch's t-test (assumes unequal variances)
            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
            
            # Bonferroni correction
            p_adjusted = min(p_val * n_comparisons, 1.0)
            
            # Significance test
            significant = "Yes" if p_adjusted < 0.05 else "No"
            
            print(f"{layer1:<12}\t{layer2:<12}\t{t_stat:.4f}\t\t{p_val:.4f}\t\t{p_adjusted:.4f}\t\t{significant}")

        print(f"\nNote: Using Bonferroni correction with {n_comparisons} comparisons")
        print(f"Adjusted alpha level: {alpha_corrected:.4f}")

    else:
        print("\nConclusion: The p-value is greater than 0.05.")
        print("The variances are NOT significantly different (equal).")
        print("Recommendation: This assumption is met for a standard ANOVA.")
        # Also get counts by variable to check balance
        #class_0_var_counts = class_0_melted['Layer'].value_counts()
        #print(f"Count by variable in class_0_melted:\n{class_0_var_counts}")
        # Here we treat 'Layer' as a categorical factor affecting 'Sensitivity'
        # If you have other factors later (like ModelType, ClassIndex, etc.), you can add them like:
        # model = ols('Sensitivity ~ C(Layer) * C(AnotherFactor)', data=class_0_melted).fit()
        model = ols('Sensitivity ~ C(Layer)', data=class_0_melted).fit()
        anova_results = anova_lm(model, typ=2)
        
        # Extract F-statistic and p-value from ANOVA results
        f_stat = anova_results.loc['C(Layer)', 'F']
        p_value = anova_results.loc['C(Layer)', 'PR(>F)']
        
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4g}")
        
        # Interpretation
        if (p_value < 0.05):
            print("Result: Significant difference between layers (p < 0.05)")
        else:
            print("Result: No significant difference between layers (p >= 0.05)")
        print("\n========== Tukey HSD Post-hoc Test (for equal variances) ==========")
        tukey_hsd = pairwise_tukeyhsd(endog=class_0_melted['Sensitivity'],
                                      groups=class_0_melted['Layer'],
                                      alpha=0.05)
        print(tukey_hsd)

        # Optional: Save Tukey HSD results to CSV
        tukey_df = pd.DataFrame(data=tukey_hsd._results_table.data[1:], 
                               columns=tukey_hsd._results_table.data[0])
        tukey_df.to_csv('tukey_hsd_results.csv', index=False)
        print(f"\nTukey HSD results saved to 'tukey_hsd_results.csv'")
except NameError:
    print("Error: The DataFrame 'class_0_melted' was not found.")
except KeyError:
    print("Error: Make sure your DataFrame has columns named 'Layer' and 'Sensitivity'.")
    


print("========== Multifactor ANOVA Results ==========")
print(anova_results, "\n")

# === Post-hoc analysis: Tukey HSD ===
print("========== Tukey HSD Pairwise Comparison ==========")
tukey = pairwise_tukeyhsd(endog=class_0_melted['Sensitivity'],
                          groups=class_0_melted['Layer'],
                          alpha=0.05)
print(tukey)

# === Visualization ===
plt.figure(figsize=(10, 6))
sns.boxplot(x='Layer', y='Sensitivity', data=class_0_melted)
plt.title('Distribution of Sensitivity Scores Across Layers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === Optional: Check ANOVA assumptions ===
# 1. Normality check on residuals
residuals = model.resid
shapiro_test = stats.shapiro(residuals)
print("\nShapiro-Wilk Test for Residual Normality:")
print(f"Statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")

# 2. Homogeneity of variances (Levene’s test)
grouped = [group["Sensitivity"].values for name, group in class_0_melted.groupby("Layer")]
levene_test = stats.levene(*grouped)
print("\nLevene’s Test for Homogeneity of Variances:")
#print(f"Statistic={levene_test.statistic:.4f}, p-value={levene_test.pvalue:.4f}")grouped = [group["Sensitivity"].values for _, group in class_0_melted.groupby("Layer")]
