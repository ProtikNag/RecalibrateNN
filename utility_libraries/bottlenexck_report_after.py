import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

############################################################
# DATA PREPARATION
############################################################

def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    df = df.drop(columns=['Full filepath'])
    df = df.drop(columns=[c for c in df.columns
                          if c.startswith('sensitivityscore_before_')])
    class_groups = dict(tuple(df.groupby('Full Class Index')))
    for k in class_groups:
        g = class_groups[k].drop(columns=['Full Class Index'])
        # Retain only columns that end with '_0.7'
        g = g.loc[:, [c for c in g.columns if c.endswith('_0.7')]]
        g.columns = [c.replace('sensitivityscore_After_', '') for c in g.columns]
        class_groups[k] = g
    return class_groups


############################################################
# GRADIENT ALIGNMENT (CORRELATION)
############################################################

def compute_gradient_alignment(df):
    """
    Computes gradient alignment matrix
    Equivalent to correlation but written explicitly
    """
    S = df.values
    N = S.shape[0]
    numerator = (S.T @ S) / N
    second_moment = np.mean(S**2, axis=0)
    denom = np.sqrt(np.outer(second_moment, second_moment))
    alignment = numerator / denom
    alignment = pd.DataFrame(alignment,
                             index=df.columns,
                             columns=df.columns)
    mask = np.triu(np.ones_like(alignment, dtype=bool), k=0)
    alignment_masked = alignment.mask(mask)
    return alignment_masked


############################################################
# CONCEPT SENSITIVITY PROPAGATION INDEX
############################################################

def compute_cspi(alignment_matrix):
    layers = alignment_matrix.columns
    n = len(layers)
    scores = []
    for i in range(n):
        downstream = alignment_matrix.iloc[i, i+1:]
        if len(downstream) > 0:
            score = np.mean(np.abs(downstream))
        else:
            score = 0
        scores.append(score)
    result = pd.DataFrame({
        "Layer": layers,
        "CSPI": scores
    })
    result = result.sort_values("CSPI", ascending=False)
    return result


############################################################
# BOTTLENECK DETECTION
############################################################

def detect_bottlenecks(alignment_matrix, threshold=0.2):
    layers = alignment_matrix.columns
    n = len(layers)
    bottlenecks = []
    for i in range(n):
        downstream = alignment_matrix.iloc[i, i+1:]
        if len(downstream) == 0:
            continue
        mean_corr = np.mean(np.abs(downstream))
        if mean_corr >= threshold:
            bottlenecks.append({
                "Layer": layers[i],
                "Mean_Alignment": mean_corr
            })
    return pd.DataFrame(bottlenecks)


############################################################
# HEATMAP PLOTTING
############################################################

def plot_heatmap(matrix, title, save_path):
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    plt.figure(figsize=(10,8))
    sns.heatmap(matrix,
                annot=True,
                fmt=".2f",
                cmap='coolwarm',
                mask=mask,
                square=True,
                cbar_kws={"shrink":0.8})
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


############################################################
# SUBSET HEATMAP
############################################################

def compute_top_subset(corr_matrix, top_k=10):
    last_row = corr_matrix.iloc[-1]
    positive = last_row[last_row > 0]
    if len(positive) <= 1:
        return None
    top_cols = positive.sort_values(ascending=False).head(top_k).index.tolist()
    subset = corr_matrix.loc[top_cols, top_cols]
    return subset


############################################################
# PROCESS EACH CLASS
############################################################

def process_class(class_id, df, output_xlsx, image_dir):
    print(f"\nProcessing class {class_id}")
    corr = df.corr()
    alignment = compute_gradient_alignment(df)
    cspi = compute_cspi(alignment)
    bottlenecks = detect_bottlenecks(alignment)
    ##################################################
    # Save correlation matrices
    ##################################################
    mask = np.triu(np.ones_like(corr, dtype=bool))
    masked_corr = corr.mask(mask)
    # Create Excel file if it doesn't exist
    if not os.path.exists(output_xlsx):
        pd.DataFrame().to_excel(output_xlsx, engine="openpyxl")
    
    with pd.ExcelWriter(output_xlsx,
                        mode='a',
                        if_sheet_exists='replace',
                        engine="openpyxl") as writer:
        masked_corr.to_excel(writer,
                             sheet_name=f'Class_{class_id}_corr')
        alignment.to_excel(writer,
                           sheet_name=f'Class_{class_id}_alignment')
        cspi.to_excel(writer,
                      sheet_name=f'Class_{class_id}_cspi')
    ##################################################
    # Heatmap
    ##################################################
    heatmap_path = os.path.join(image_dir,
                                f'correlation_heatmap_class_{class_id}.png')
    plot_heatmap(corr,
                 f'Correlation Heatmap Class {class_id}',
                 heatmap_path)
    ##################################################
    # Subset heatmap
    ##################################################
    subset = compute_top_subset(corr)
    if subset is not None:
        subset_path = os.path.join(image_dir,
                                   f'subset_heatmap_class_{class_id}.png')
        plot_heatmap(subset,
                     f'Top Correlation Subset Class {class_id}',
                     subset_path)
    ##################################################
    # Return bottlenecks
    ##################################################
    bottlenecks["Class"] = class_id
    return bottlenecks


############################################################
# MAIN
############################################################
def main():
    if len(sys.argv) != 3:
        print("Usage: python correlation.py <csv_file> <output_xlsx>")
        sys.exit(1)
    csv_file = sys.argv[1]
    output_xlsx = sys.argv[2]
    image_dir = os.path.dirname(output_xlsx)
    class_groups = load_and_prepare_data(csv_file)
    all_bottlenecks = []
    for class_id, df in class_groups.items():
        bn = process_class(class_id,
                           df,
                           output_xlsx,
                           image_dir)
        all_bottlenecks.append(bn)

    ##################################################
    # Save bottleneck report
    ##################################################
    bottleneck_report = pd.concat(all_bottlenecks)
    bottleneck_csv = os.path.join(image_dir,
                                  "bottleneck_report.csv")
    bottleneck_report.to_csv(bottleneck_csv,
                             index=False)
    print("\nBottleneck report saved to:", bottleneck_csv)

############################################################

if __name__ == "__main__":
    main()