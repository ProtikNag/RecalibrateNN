import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import sys

# a) Read the CSV file into a dataframe
if len(sys.argv) < 3:
    print("Usage: python anova_boxplots.py <input_csv_file> <output_folder>")
    sys.exit(1)
csv_file = sys.argv[1]
output_folder = sys.argv[2]

df = pd.read_csv(csv_file)

# b) Drop the 'Full filepath' column if it exists
if 'Full filepath' in df.columns:
    df = df.drop('Full filepath', axis=1)

# c) Split the dataframe based on unique values in 'Full Class Index'
unique_classes = df['Full Class Index'].unique()
df_by_class = {class_idx: df[df['Full Class Index'] == class_idx] for class_idx in unique_classes}



# Get all sensitivity columns
sensitivity_before_cols = [col for col in df.columns if col.startswith('sensitivityscore_before_')]
sensitivity_after_cols = [col for col in df.columns if col.startswith('sensitivityscore_After_')]

# Create ordered pairs: each before column paired with each after column
from itertools import product
sensitivity_pairs = list(product(sensitivity_before_cols, sensitivity_after_cols))

# Create output folders if they don't exist
os.makedirs(os.path.join(output_folder, 'Images'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'Results'), exist_ok=True)

sorted_classes = sorted(unique_classes)

# For each ordered pair in sensitivity_pairs, perform ANOVA per class and generate plots + results
for before_col, after_col in sensitivity_pairs:
    before_feature = before_col.replace('sensitivityscore_before_', '')
    after_feature = after_col.replace('sensitivityscore_After_', '')
    pair_name = f'{before_feature}_vs_{after_feature}'

    results_lines = []
    results_lines.append(f"Ordered Pair: {before_col}  <-->  {after_col}")
    results_lines.append("=" * 90)

    # Collect data per class
    class_before_data = {}
    class_after_data = {}
    for class_idx in sorted_classes:
        class_before_data[class_idx] = df_by_class[class_idx][before_col].dropna().values
        class_after_data[class_idx] = df_by_class[class_idx][after_col].dropna().values

    # --- One-way ANOVA across classes for before and after ---
    groups_before = [class_before_data[c] for c in sorted_classes]
    groups_after = [class_after_data[c] for c in sorted_classes]
    

    f_before, p_before = stats.f_oneway(*groups_before)
    f_after, p_after = stats.f_oneway(*groups_after)

    results_lines.append(f"\nANOVA across classes (Before - {before_col}):")
    results_lines.append(f"  F-statistic: {f_before:.4f}, p-value: {p_before:.4e}")
    results_lines.append(f"\nANOVA across classes (After - {after_col}):")
    results_lines.append(f"  F-statistic: {f_after:.4f}, p-value: {p_after:.4e}")

    # --- Per-class ANOVA (before vs after within each class) + stats + 95% CI ---
    for class_idx in sorted_classes:
        bdata = class_before_data[class_idx]
        adata = class_after_data[class_idx]
        results_lines.append(f"\n{'─' * 70}")
        results_lines.append(f"Class {class_idx}:")
        # One-way ANOVA: before vs after for this class
        if len(bdata) > 1 and len(adata) > 1:
            f_stat, p_val = stats.f_oneway(bdata, adata)
        else:
            f_stat, p_val = float('nan'), float('nan')
        results_lines.append(f"  ANOVA (before vs after): F={f_stat:.4f}, p={p_val:.4e}")
        # Basic statistics and 95% CI for before
        for label, data in [("Before", bdata), ("After", adata)]:
            n = len(data)
            mean = np.mean(data) if n > 0 else float('nan')
            std = np.std(data, ddof=1) if n > 1 else float('nan')
            median = np.median(data) if n > 0 else float('nan')
            min_val = np.min(data) if n > 0 else float('nan')
            max_val = np.max(data) if n > 0 else float('nan')
            if n > 1:
                se = std / np.sqrt(n)
                ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
            else:
                ci = (float('nan'), float('nan'))
            results_lines.append(f"  {label}:")
            results_lines.append(f"    N={n}, Mean={mean:.4f}, Std={std:.4f}, Median={median:.4f}, Min={min_val:.4f}, Max={max_val:.4f}")
            results_lines.append(f"    95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # Save results text file
    results_filename = os.path.join(output_folder, 'Results', f'{pair_name}_results.txt')
    with open(results_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results_lines))
    print(f"Results saved: {results_filename}")

    # --- Consolidated box plot: before and after for each class ---
    # Calculate y_limits from all data to ensure consistent scaling
    all_data = [class_before_data[c] for c in sorted_classes] + [class_after_data[c] for c in sorted_classes]
    all_values = np.concatenate([d for d in all_data if len(d) > 0])
    y_min = np.min(all_values)
    y_max = np.max(all_values)
    y_margin = (y_max - y_min) * 0.1
    y_limits = (y_min - y_margin, y_max + y_margin)
    
    num_classes = len(sorted_classes)
    fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 6))
    if num_classes == 1:
        axes = [axes]

    for i, class_idx in enumerate(sorted_classes):
        bdata = class_before_data[class_idx]
        adata = class_after_data[class_idx]
        axes[i].boxplot([bdata, adata], tick_labels=['Before', 'After'])
        axes[i].set_title(f'Class {class_idx}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Sensitivity Score')
        axes[i].set_ylim(y_limits)
        axes[i].grid(False)

    #fig.suptitle(f'Before vs After: {before_feature} / {after_feature}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    image_filename = f'Images/{pair_name}_boxplots.png'
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    print(f"Image saved: {image_filename}")
    plt.close()

print("\nProcessing complete!")

