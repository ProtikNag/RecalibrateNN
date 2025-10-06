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
    
    # Create DataFrame with reshaped data
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
    f_stat, p_val = stats.f_oneway(before_group, after_group)
    print(f"\nOne-way ANOVA results for Class {cls}:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_val:.4f}")

# Create plot
plt.figure(figsize=(7, 6))
sns.boxplot(
    data=final_df, 
    x='Class', 
    y='Score', 
    hue='Anova', 
    dodge=True, # Reduced dodge to bring the groups closer
    width=0.4, # Increased width to fill the space
    gap=0.3,
    linewidth=0.5
)
#plt.title('Before vs After Scores by Class')
plt.xlabel('Class')
plt.ylabel('Score')
# Remove top and right spines
sns.despine(top=True, right=True)

plt.savefig(f'class_{cls}_before_after.png', bbox_inches='tight', dpi=300)
plt.show()