import pandas as pd
import sys
import argparse
import matplotlib.pyplot as plt

columns_to_exclude = ['Full filepath']
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
    for idx, class_value in enumerate([0, 1, 2]):    
        subset = df[df['Full Class Index'] == class_value]
        subset[data_columns].boxplot(ax=axes[idx], grid=False)
        axes[idx].set_title(
            f'Box Plot for Full Class index = {class_value}',
            fontname='Times New Roman',
            fontsize=12
        )
        axes[idx].set_ylabel('Values')
        axes[idx].set_xticklabels(data_columns, rotation=90)
    fig.savefig('boxplot_potter.png', dpi=300)
    fig.savefig('boxplot_potter.pdf')
    plt.tight_layout()
    plt.show()


def extract_box_plot_individual(df):
    # Exclude the 'Full Class index' column from boxplot data
    data_columns = [col for col in df.columns if col != 'Full Class Index']
    # Create a separate boxplot for each class index
    for class_value in [0, 1, 2]:
        subset = df[df['Full Class Index'] == class_value]
        plt.figure(figsize=(6, 6))
        subset[data_columns].boxplot(grid=False)
        plt.title(
            f'Box Plot for Full Class index = {class_value}',
            fontname='Times New Roman',
            fontsize=12
        )
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
    extract_box_plot(df)
    extract_box_plot_individual(df)



