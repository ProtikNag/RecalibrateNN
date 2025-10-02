import pandas as pd
import sys
import argparse
import matplotlib.pyplot as plt

columns_to_exclude = ['Full filepath', 'Full Class Index','Index','ImageName']
columns_to_rename = ['sensitivityscore_before_']

def read_csv(csv_file):
    try:
        df = pd.read_csv(csv_file)
        df = df.drop(columns=columns_to_exclude)
        df.columns = [col.replace(name, '') if name in col else col for col in df.columns for name in columns_to_rename]
        #df.columns = [col.replace('sensitivityscore_before_', '') for col in df.columns]
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

def extract_box_plot(df):
    # Exclude the 'Full Class index' column from boxplot data
    data_columns = [col for col in df.columns if col != 'Full Class Index']
    # Create a boxplot for each class index
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    # Exclude the 'Full Class index' column from boxplot data
    data_columns = [col for col in df.columns if col != 'ClassName']
    # Create a separate boxplot for each class index
    for idx, class_value in enumerate(['Deer', 'Horse', 'Zebra']):
        subset = df[df['ClassName'] == class_value]
        subset[data_columns].boxplot(ax=axes[idx], grid=False)
        axes[idx].set_title(
            f'Box Plot for Class Name = {class_value}',
            fontname='Times New Roman',
            fontsize=8
        )
        axes[idx].set_ylabel('Values')
        axes[idx].set_xticklabels(data_columns, rotation=90)
    fig.savefig('boxplot_potter.png', dpi=300)
    fig.savefig('boxplot_potter.pdf')
    plt.tight_layout()
    plt.show()

def extract_box_plot_individual(df):
    # Exclude the 'Full Class index' column from boxplot data
    data_columns = [col for col in df.columns if col != 'ClassName']
    # Create a separate boxplot for each class index
    for class_value in ['Deer', 'Horse', 'Zebra']:
        subset = df[df['ClassName'] == class_value]
        plt.figure(figsize=(6, 6))
        box = subset[data_columns].boxplot(grid=False, showcaps=True, patch_artist=True, boxprops=dict(linewidth=1), whiskerprops=dict(linewidth=1), medianprops=dict(linewidth=1), flierprops=dict(marker='o', markersize=3))
        # Remove the right border
        ax = plt.gca()
        ax.set_ylim([df[data_columns].min().min(), df[data_columns].max().max()])
        plt.gcf().set_size_inches(6, 3)  # Reduce height to 3 inches
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.gca().set_xticklabels(data_columns, fontsize=8, fontname='Times New Roman')
        plt.ylabel('Values')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'boxplot_potter_class_{class_value}.png', dpi=300)
        plt.savefig(f'boxplot_potter_class_{class_value}.pdf')
        plt.show()

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Boxplot for Potter data')
    parser.add_argument('csv_file', help='Path to the CSV file')
    args = parser.parse_args()
    csv_file = args.csv_file
    df = read_csv(csv_file)
    #extract_box_plot(df)
    extract_box_plot_individual(df)



