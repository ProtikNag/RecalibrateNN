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
    try:
        df = df.drop(columns=['Full filepath'])
    except KeyError:
        pass
    try:
        df = df.drop(columns=[c for c in df.columns
                              if c.startswith('sensitivityscore_After_')])
    except KeyError:
        pass    
    class_groups = dict(tuple(df.groupby('Full Class Index')))
    for k in class_groups:
        g = class_groups[k].drop(columns=['Full Class Index'])
        g.columns = [c.replace('sensitivityscore_before_', '') for c in g.columns]
        try:
            g.columns = [c.replace('features.', 'Layer.') for c in g.columns]
        except Exception as e:
            pass
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
    return pd.DataFrame(alignment,
                        index=df.columns,
                        columns=df.columns)


############################################################
# CONCEPT SENSITIVITY PROPAGATION INDEX
############################################################
def compute_cspi(alignment_matrix, tau_abs=0.1, tau_rel=0.1):
    layers = alignment_matrix.columns
    n = len(layers)

    scores_absolute = []
    scores_relative = []
    layer_type = []
    is_bottleneck = []

    for i in range(n):
        downstream = alignment_matrix.iloc[i, i+1:]

        if len(downstream) > 0:
            score_abs = np.mean(np.abs(downstream))
            score_rel = np.mean(downstream)
        else:
            score_abs = 0.0
            score_rel = 0.0

        if score_abs < tau_abs:
            label = "Weak"
            bottleneck = 0
        elif score_rel > tau_rel:
            label = "Positive_Aligned"
            bottleneck = 0
        elif score_rel < -tau_rel:
            label = "Negative_Aligned"
            bottleneck = 1
        else:
            label = "Mixed_Strong"
            bottleneck = 1

        scores_absolute.append(score_abs)
        scores_relative.append(score_rel)
        layer_type.append(label)
        is_bottleneck.append(bottleneck)

    result = pd.DataFrame({
        "Layer": layers,
        "CSPI_Absolute": scores_absolute,
        "CSPI_Directional": scores_relative,
        "Layer_Type": layer_type,
        "Is_Bottleneck": is_bottleneck
    })
    """
    result["Bottleneck_Rank"] = result["CSPI_Absolute"].rank(
        ascending=False, method="dense"
    )
    """
    return result

############################################################
# MOVING AVERAGE
############################################################

def compute_moving_average(deltas, column_names, window_size=3):
    """
    Compute moving average of deltas
    
    Parameters:
    -----------
    deltas : array-like
        The delta values to smooth
    window_size : int
        Size of the moving window (default: 3)
    column_names : list
        List of column names corresponding to the deltas
    Returns:
    --------
    pd.DataFrame
        DataFrame containing moving average of deltas and corresponding layer names
    """
    deltas = np.array(deltas)
    column_names = np.array(column_names)
    if len(deltas) < window_size:
        return pd.DataFrame({
            'Layer': column_names,
            'Avg(d_i, d_i+1, d_i+2)': deltas,
            'Window_Layers': column_names
        })
    
    moving_avg = np.convolve(deltas, np.ones(window_size)/window_size, mode='valid')
    # Pad the beginning to maintain the same length
    pad_size = len(deltas) - len(moving_avg)
    moving_avg = np.concatenate([deltas[:pad_size], moving_avg])
    window_layers = []
    for i in range(len(deltas)):
        if i < pad_size:
            # For averaged values, concatenate the layer names in the window
            window_layers.append(column_names[i])
        else:
            window_idx = i - pad_size
            layer_window = column_names[window_idx:window_idx + window_size]
            window_layers.append(' -> '.join(layer_window))
    return pd.DataFrame({      
        'Moving_Avg': moving_avg,
        'Window_Layers': window_layers
    })


############################################################
# BOTTLENECK DETECTION
############################################################

def detect_bottlenecks_old(alignment_matrix, threshold=0.2):
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

# =========================
# 2. COMPUTE R(i, i+1)
# =========================
def compute_adjacent_correlation(corr_matrix):
    n = corr_matrix.shape[0]
    R = []
    print("Correlation matrix:\n", corr_matrix)

    for i in range(n - 1):
        R.append(corr_matrix[i, i + 1])
    print("Adjacent Correlations R(i, i+1):", R)
    return np.array(R)

# =========================
# 3. COMPUTE DELTA
# =========================
def compute_delta(R, column_names):
    delta = np.diff(R)
    layer_pairs = [(column_names[i], column_names[i+1]) for i in range(len(delta))]
    print("Delta values:", delta)

    return delta, layer_pairs

# =========================
# 5. BOTTLENECK DETECTION
# =========================
def detect_bottlenecks(R, delta, cspi=None,
                       tau_delta=0.2,
                       tau_corr=0.2,
                       tau_cspi=0.6):

    bottlenecks = []

    for i in range(len(R)):

        cond1 = False
        cond2 = False
        cond3 = False

        # Condition 1: sharp drop
        if i < len(delta):
            cond1 = delta[i] < -tau_delta

        # Condition 2: low correlation
        cond2 = R[i] < tau_corr

        # Condition 3: direction flip
        cond3 = R[i] < 0

        # Optional CSPI condition
        cond4 = False
        if cspi is not None:
            cond4 = cspi[i] < tau_cspi

        # Combine logic
        if (cond1 or cond2 or cond3) and (cspi is None or cond4):
            bottlenecks.append(1)
        else:
            bottlenecks.append(0)
    print("Bottleneck condition np.array(bottlenecks)", np.array(bottlenecks))

    return np.array(bottlenecks)
        

############################################################
# PROCESS EACH CLASS
############################################################

def process_class(class_id, df, output_xlsx, result_dir):
    print(f"\nProcessing class {class_id}")
    corr = df.corr()
    column_names = df.columns.tolist()
    print("Dataframe:\n", df.head())
    # Compute metrics
    R = compute_adjacent_correlation(corr.values)
    delta, layer_pairs = compute_delta(R, column_names)
    print(delta)
    # Save deltas and layer pairs in a DataFrame
    delta_df = pd.DataFrame({
        'R_i+1 - R_i': delta,
        'Layer_Pairs': layer_pairs
    })
    
    # Save delta_df to CSV file
    delta_csv_path = os.path.join(result_dir, f'delta_class_{class_id}.csv')
    delta_df.to_csv(delta_csv_path, index=False)
    
    print("Delta DataFrame:\n", delta_df)
    
    moving_avg_df = compute_moving_average(delta, column_names)
    print("Moving average of delta:", moving_avg_df.head())
    
    # Save moving_avg_df to CSV file
    moving_avg_csv_path = os.path.join(result_dir, f'moving_avg_class_{class_id}.csv')
    moving_avg_df.to_csv(moving_avg_csv_path, index=False)
    
    alignment = compute_gradient_alignment(df)
    cspi = compute_cspi(alignment)
    
    bottlenecks = detect_bottlenecks_old(alignment)
    """
    bottlenecks = detect_bottlenecks(R, delta, cspi=cspi["CSPI"].values,
                                     tau_delta=0.2,
                                     tau_corr=0.2,
                                     tau_cspi=0.6)

    """
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
        cspi.to_excel(writer,sheet_name=f'Class_{class_id}_cspi')
        delta_df.to_excel(writer,
                          sheet_name=f'Class_{class_id}_delta',
                          index=False)
        moving_avg_df.to_excel(writer,
                               sheet_name=f'Class_{class_id}_moving_avg',
                               index=False)
    ##################################################
    # Heatmap
    ##################################################
    heatmap_path = os.path.join(result_dir,
                                f'correlation_heatmap_class_{class_id}.png')
    plot_heatmap(corr,
                 f'Correlation Heatmap Class {class_id}',
                 heatmap_path)
    ##################################################
    # Subset heatmap
    ##################################################
    subset = compute_top_subset(corr)
    if subset is not None:
        subset_path = os.path.join(result_dir,
                                   f'subset_heatmap_class_{class_id}.png')
        plot_heatmap(subset,
                     f'Top Correlation Subset Class {class_id}',
                     subset_path)
    ##################################################
    # Return bottlenecks
    ##################################################
    return bottlenecks


############################################################
# MAIN
############################################################
def main():
    if len(sys.argv) != 3:
        print("Usage: python bottleneck.py <full path csv_file> <full path output_xlsx>")
        sys.exit(1)
    csv_file = sys.argv[1]
    output_xlsx = sys.argv[2]
    result_dir = os.path.dirname(output_xlsx)
    os.makedirs(result_dir, exist_ok=True)
    class_groups = load_and_prepare_data(csv_file)
    all_bottlenecks = {}
    for class_id, df in class_groups.items():
        bn = process_class(class_id,
                           df,
                           output_xlsx,
                           result_dir)
        all_bottlenecks[class_id] = bn

############################################################

if __name__ == "__main__":
    main()