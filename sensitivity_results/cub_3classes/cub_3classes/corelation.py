import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
file_name = r'C:\Users\srikant1\OneDrive - Intel Corporation\Documents\MobaXterm\slash\srikant1_soc5cg44242y1\RemoteFiles\1250082_3_41\sensitivity_audit_trail_vgg16_20250925_234304.csv'
file_name = input("Enter the path to your CSV file: ")
df = pd.read_csv(file_name)
df = df.drop(columns=['Full filepath'])
class_groups = dict(tuple(df.groupby('Full Class Index')))

# Display the first few rows
print(df.head())
print(class_groups)
for i in class_groups:
    class_groups[i] = class_groups[i].drop(columns=['Full Class Index'])
    class_groups[i].columns = [col.replace('sensitivityscore_before_', '') for col in class_groups[i].columns]
    print(class_groups[i].head())  # Display first few rows of each class group
    # Drop the 'sensitivity' column for group i and store in a new DataFrame
    col_names = class_groups[i].columns.tolist()
    correlation_matrix = class_groups[i].corr()
    print(f"Correlation matrix for class group {i}:")
    print(correlation_matrix)
    with pd.ExcelWriter('correlation_matrices.xlsx', mode='a' if i != list(class_groups.keys())[0] else 'w') as writer:
        correlation_matrix.to_excel(writer, sheet_name=f'Class_{i}')