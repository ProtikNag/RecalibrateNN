import pandas as pd
import os
from pathlib import Path

def process_image_paths(folder_path):
    """
    Process image_paths.csv file:
    - Extract mode, class_name, and file_name from image_path
    - Create unique image_name index per class (0-indexed, resets per class)
    - Save updated CSV
    """
    csv_path = os.path.join(folder_path, 'image_paths.csv')
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Extract mode, class_name, and file_name from image_path
    def extract_path_components(image_path):
        path_parts = Path(image_path).parts
        # Find mode (valid/train/test) and class_name
        mode = None
        class_name = None
        
        for i, part in enumerate(path_parts):
            if part in ['valid', 'train', 'test']:
                mode = part
                if i + 1 < len(path_parts):
                    class_name = path_parts[i + 1]
                break
        
        file_name = Path(image_path).name
        
        return pd.Series([mode, class_name, file_name])
    
    # Apply extraction
    df[['mode', 'class_name', 'file_name']] = df['image_path'].apply(extract_path_components)
    
    # Create image_name column: unique index per class, reset to 0 for each class
    df['image_name'] = df.groupby('class_name').cumcount()
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    print(f"Updated CSV saved to: {csv_path}")
    print(f"\nDataFrame preview:\n{df.head(10)}")

if __name__ == "__main__":
    folder_path = input("Enter folder path: ")
    process_image_paths(folder_path)
