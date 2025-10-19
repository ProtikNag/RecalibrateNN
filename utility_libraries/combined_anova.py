import pandas as pd
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt

# Get file path from user
#file_path = input("Please enter the path to your CSV file: ")
file_path = 'temp.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Split data by Full Class Index
classes = df['Full Class Index'].unique()
class_dfs = {cls: df[df['Full Class Index'] == cls] for cls in classes}

# Process each class
results = []
for cls in classes:
    class_df = class_dfs[cls]
    
    # Get before and after columns
    before_cols = [col for col in class_df.columns if col.startswith('sensitivityscore_before')]
    after_cols = [col for col in class_df.columns if col.startswith('sensitivityscore_After')]
    before_scores = class_df[before_cols].values.flatten()
    after_scores = class_df[after_cols].values.flatten()
    combined_df = pd.DataFrame({
        'Score': pd.concat([pd.Series(before_scores), pd.Series(after_scores)]),
        'Anova': ['Before'] * len(before_scores) + ['After'] * len(after_scores),
        'Class': cls
    })
    
    results.append(combined_df)

# Combine all results
final_df = pd.concat(results, ignore_index=True)

# Perform one-way ANOVA for each class and generate plots
for cls in classes:
    class_data = final_df[final_df['Class'] == cls]
    before_group = class_data[class_data['Anova'] == 'Before']['Score']
    after_group = class_data[class_data['Anova'] == 'After']['Score']
    # Calculate means to check the direction of the difference
    mean_before = before_group.mean()
    mean_after = after_group.mean()
        
    f_stat, p_val = stats.f_oneway(before_group, after_group)
    # --- One-Tailed Test Adjustment ---
    # Null Hypothesis (H0): mean_before <= mean_after
    # Alternative Hypothesis (Ha): mean_before > mean_after (i.e., Before - After > 0)
    p_val_two_tailed = p_val
    
    if mean_before > mean_after:
        # If the sample mean difference is in the hypothesized direction (Before > After),
        # we halve the two-tailed p-value to get the one-tailed p-value.
        p_val_one_tailed = p_val_two_tailed / 2
        test_conclusion = f"The mean of 'Before' ({mean_before:.4f}) is greater than 'After' ({mean_after:.4f}). Testing H_a: Before > After."
    else:
        # If the sample mean difference is NOT in the hypothesized direction (Before <= After),
        # the one-tailed p-value is close to 1 (or 0.5 depending on the test), but since we're
        # interested in the *significance* of the difference in the *correct* direction,
        # we can state it's not significant for the one-tailed test.
        # A conservative approach is to set the one-tailed p-value to 1 or simply note
        # that the difference is in the opposite direction.
        p_val_one_tailed = 1.0 - (p_val_two_tailed / 2) # A more formal two-sample t-test approach
        # For simplicity, let's just use the two-tailed p-value and note the direction
        test_conclusion = f"The mean of 'Before' ({mean_before:.4f}) is NOT greater than 'After' ({mean_after:.4f}). Testing H_a: Before > After."
        p_val_one_tailed = 1.0 # This indicates no evidence to support H_a
    print(test_conclusion)
    print(f"One-tailed p-value: {p_val_one_tailed:.4f}")
    print(f"\nOne-way ANOVA results for Class {cls}:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_val:.4f}")

# Create plot
plt.figure(figsize=(8, 8))
sns.boxplot(
    data=final_df, 
    x='Class', 
    y='Score', 
    hue='Anova', 
    dodge=True,
    width=0.4,
    gap=0.3,
    linewidth=0.5
)
plt.legend(loc='upper left')
#plt.title('Before vs After Scores by Class')
plt.xlabel('Class')
plt.ylabel('Sensitivity Score')
# Remove top and right spines
sns.despine(top=True, right=True)

plt.savefig(f'class_{cls}_before_after.png', bbox_inches='tight', dpi=300)
plt.show()