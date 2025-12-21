import torch
from torchvision import models, transforms
from PIL import Image
import csv
import numpy as np
import os
import pandas as pd
import argparse

import torch.nn as nn

activations = {}
parser = argparse.ArgumentParser(description='Extract layer activations from a model')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model (e.g., vgg16, resnet50)')
parser.add_argument('--model_path', type=str, default=None, help='Path to the model file (default: {model_name}.pth)')
args = parser.parse_args()

model_name = args.model_name
model_path = args.model_path if args.model_path else f'{model_name}.pth'
model_base_path = f'/mnt/sdd/basics/base_models/{model_name}/{model_path}'  # Replace with your model path
base_image_dir = '/home/datasets/train'  # Replace with your base image directory
destination_csv = f'layer_activations_{model_name}.csv'  # Output CSV file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Prepare image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

# Hook to capture activations
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def load_model(model_path=None):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    # Register hooks for all layers
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d)):
            hooks.append(layer.register_forward_hook(get_activation(name)))
    return model, hooks


if(__name__ == "__main__"):
    model, hooks = load_model(model_base_path)
    
    # Register hooks for all layers
    #hooks = []
    #for name, layer in model.named_modules():
    #    if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d)):
    #        hooks.append(layer.register_forward_hook(get_activation(name)))

    # Define class directories
    class_dirs = {
        'deer': 0,
        'horse': 1,
        'zebra': 2
    }

    base_dir = base_image_dir  # Replace with your base directory path

    # Collect all images with their classes
    image_list = []
    for class_name, class_label in class_dirs.items():
        class_path = os.path.join(base_dir, class_name)
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_list.append((os.path.join(class_path, img_file), class_label, class_name))

    # Save results to CSV
    with open(destination_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Path', 'Class Label', 'Class Name', 'Layer Name', 'Layer Type', 'Transfer Function', 'Output Shape', 
                         'Mean Activation', 'Std Activation', 'Min Activation', 'Max Activation',' Pos activations', 
                         'Neg activations', 'Q1' , 'Q2', 'Q3','Kernel Shape', 'Kernel Count', 'Predicted Class', 'Predicted Probability'])
        
        for image_path, class_label, class_name in image_list:
            print(f"Processing: {image_path}")
            activations.clear()
            # Variable to store prediction
            predicted_class = None
            predicted_prob = None
            
            # Process image
            input_tensor = preprocess_image(image_path)
            # Forward pass and get prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_prob, predicted_class = torch.max(probabilities, dim=1)
                predicted_class = predicted_class.item()
                predicted_prob = predicted_prob.item()
            # Forward pass
            with torch.no_grad():
                output = model(input_tensor)
            
            for name, layer in model.named_modules():
                if name in activations:
                    act = activations[name]
                    layer_type = type(layer).__name__
                    
                    # Get transfer function
                    if isinstance(layer, nn.ReLU):
                        transfer_fn = 'ReLU: max(0, x)'
                    elif isinstance(layer, nn.Conv2d):
                        transfer_fn = 'Linear (before activation)'
                    elif isinstance(layer, nn.Linear):
                        transfer_fn = 'Linear (before activation)'
                    elif isinstance(layer, nn.MaxPool2d):
                        transfer_fn = 'MaxPool'
                    else:
                        transfer_fn = 'N/A'
                    
                    # Get kernel info
                    kernel_shape = 'N/A'
                    kernel_count = 'N/A'
                    if isinstance(layer, nn.Conv2d):
                        kernel_shape = f'{layer.kernel_size}'
                        kernel_count = layer.out_channels
                    elif isinstance(layer, nn.Linear):
                        kernel_shape = f'({layer.in_features}, {layer.out_features})'
                        kernel_count = layer.out_features
                    # Statistics
                    writer.writerow([
                        image_path,
                        class_label,
                        class_name,
                        name,
                        layer_type,
                        transfer_fn,
                        str(tuple(act.shape)),
                        f'{act.mean().item():.6f}',
                        f'{act.std().item():.6f}',
                        f'{act.min().item():.6f}',
                        f'{act.max().item():.6f}',
                        f'{int((act > 0).sum().item())}',  # Positive Count
                        f'{int((act <= 0).sum().item())}',  # Non-Positive Count
                        f'{torch.quantile(act.float().flatten(), 0.25).item():.6f}',
                        f'{torch.median(act.flatten()).item():.6f}',
                        f'{torch.quantile(act.float().flatten(), 0.75).item():.6f}',
                        kernel_shape,
                        kernel_count,
                        f'{predicted_class}',
                        f'{predicted_prob:.6f}'
                    ])

    # Remove hooks
    for hook in hooks:
        hook.remove()
    print(f"Activations saved to layer_activations.csv")
    # Read the CSV file
    df = pd.read_csv(destination_csv)

    # Get unique class names
    unique_classes = df['Class Name'].unique()

    # Create separate dataframes for each class
    class_dataframes = {}
    for class_name in unique_classes:
        class_dataframes[class_name] = df[df['Class Name'] == class_name].copy()

    # Access individual dataframes
    df_deer = class_dataframes.get('deer', pd.DataFrame())
    df_horse = class_dataframes.get('horse', pd.DataFrame())
    df_zebra = class_dataframes.get('zebra', pd.DataFrame())
    # Compute mean and std of Mean Activation for each unique Layer Name per class
    for class_name, class_df in class_dataframes.items():
        if not class_df.empty:
            layer_stats = class_df.groupby('Layer Name')['Mean Activation'].agg(['mean', 'std']).reset_index()
            layer_stats.columns = ['Layer Name', 'Mean of Mean Activation', 'Std of Mean Activation']
            print(f"\n{class_name} - Layer Statistics:")
            print(layer_stats.to_string(index=False))
    # Create Excel writer with the same base name as destination_csv but with .xlsx extension
    excel_file = destination_csv.replace('.csv', '.xlsx')
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Write the full dataframe to a sheet
        df.to_excel(writer, sheet_name='All_Data', index=False)
        
        # Write deer data and statistics
        if not df_deer.empty:
            df_deer.to_excel(writer, sheet_name='deer_data', index=False)
            deer_stats = df_deer.groupby('Layer Name')['Mean Activation'].agg(['mean', 'std']).reset_index()
            deer_stats.columns = ['Layer Name', 'Mean of Mean Activation', 'Std of Mean Activation']
            deer_stats.to_excel(writer, sheet_name='deer_stats', index=False)
        
        # Write horse data and statistics
        if not df_horse.empty:
            df_horse.to_excel(writer, sheet_name='horse_data', index=False)
            horse_stats = df_horse.groupby('Layer Name')['Mean Activation'].agg(['mean', 'std']).reset_index()
            horse_stats.columns = ['Layer Name', 'Mean of Mean Activation', 'Std of Mean Activation']
            horse_stats.to_excel(writer, sheet_name='horse_stats', index=False)
        
        # Write zebra data and statistics
        if not df_zebra.empty:
            df_zebra.to_excel(writer, sheet_name='zebra_data', index=False)
            zebra_stats = df_zebra.groupby('Layer Name')['Mean Activation'].agg(['mean', 'std']).reset_index()
            zebra_stats.columns = ['Layer Name', 'Mean of Mean Activation', 'Std of Mean Activation']
            zebra_stats.to_excel(writer, sheet_name='zebra_stats', index=False)
    
    print(f"Excel workbook saved to {excel_file}")

    print(f"Deer samples: {len(df_deer)}")
    print(f"Horse samples: {len(df_horse)}")
    print(f"Zebra samples: {len(df_zebra)}")
    exit()
