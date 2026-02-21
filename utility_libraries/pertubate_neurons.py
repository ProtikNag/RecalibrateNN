import itertools
import torch
from torchvision import models, transforms
from PIL import Image
import csv
import numpy as np
import os
import pandas as pd
import argparse
import torch.nn as nn
from  pertubation_utilities import NeuronPerturbationUtilities
import yaml
from logger import Logger_Singleton
import gc
from datetime import datetime


#activations
#model_name = args.model_name
#Usage 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perturb neurons in a neural network model')
    parser.add_argument('--model_name', type=str, default='inception_v3', help='Name of the model (e.g., vgg16, resnet50, inception_v3)')
    parser.add_argument('--config', type=str, help='Path to the yaml file')
    parser.add_argument('--layers_to_pertubate', default=None, type=str, help='list of layers to pertubate') 
    
    parser.add_argument('--saveas', type=str, help='Destination excel file name')

    return parser.parse_args()

def create_combinations(layers_to_modify):
    np.random.seed(42)
    combinations = []
    for i in range(1, len(layers_to_modify) + 1):
        combinations.extend(itertools.combinations(layers_to_modify, i))
    print(f"Total combinations of layers to perturb: {len(combinations)}")
    single_combos = [c for c in combinations if len(c) == 1]
    two_combo_utils = None
    three_combo_utils = None
    four_combos_utils = None
    if len(combinations) > 11:
        two_combos    = [c for c in combinations if len(c) == 2]
        three_combos  = [c for c in combinations if len(c) == 3]
        four_combos  = [c for c in combinations if len(c) == 4]
        #multi_combos = [c for c in combinations if len(c) > 1]
        print(f"single_combos combinations of layers to perturb: {len(single_combos)}")
        two_combo_utils = list(np.random.choice(len(two_combos), min(2, len(two_combos)), replace=False))
        three_combo_utils = list(np.random.choice(len(three_combos), min(1, len(three_combos)), replace=False))
        four_combos_utils = list(np.random.choice(len(four_combos), min(1, len(four_combos)), replace=False))
    combinations = single_combos
    
    if(two_combo_utils is not None):
        combinations += [two_combos[i] for i in two_combo_utils]
    if(three_combo_utils is not None):
        combinations += [three_combos[i] for i in three_combo_utils]
    if(four_combos_utils is not None):        
        combinations += [four_combos[i] for i in four_combos_utils]
    log_util.log(f"Total combinations before sampling: {len(combinations)}")
    log_util.log(f"Selected {len(combinations) - len(single_combos)} combinations after sampling: {combinations[len(single_combos):]}")
    print("Total numbe of combinations to perturb: ", len(combinations))
    print(f"Selected {len(combinations) - len(single_combos)} combinations after sampling: {combinations[len(single_combos):]}")
    return combinations

    
def perturbate_neurons(layers_to_modify, neuronPerturbation, method='gaussian', pertubation_value = None , saveas='perturbation_results', image_list=None):
    layers_to_modify = create_combinations(layers_to_modify)
        
    print(f"Created {len(layers_to_modify)} combinations of layers to perturb.")
    print(f"Layers to modify: {layers_to_modify}")
    filename_prefix = neuronPerturbation.getmodel_name()
    # Create Excel writer once for all combinations
    if(method == 'alpha'):
        rounded_value = round(pertubation_value, 2)
        filename = f'{filename_prefix}_{saveas}_{method}_{str(rounded_value)}.xlsx'    
    else:
        filename = f'{filename_prefix}_{saveas}_{method}.xlsx'
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Write summary sheet first
        layers_column = [" | ".join(tup) for tup in layers_to_modify]
        summary_df = pd.DataFrame(layers_column, columns=['Layers Perturbed'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        for idx, layer in enumerate(layers_to_modify):
            print(f"Perturbing layer: {layer}")
            layers_to_perturb = list(layer)
            
            if method == 'gaussian':
                print(f"Pertubating with method {method}")
                print(f"layers to pertubate {layers_to_perturb}")
                results = neuronPerturbation.compute_delta_logits_with_perturbation(method_name = "gaussian", layer_names=layers_to_perturb)
            elif method == 'mean':
                print(f"Pertubating with method {method}")
                results = neuronPerturbation.compute_delta_logits_with_perturbation(method_name = "mean", layer_names=layers_to_perturb)
            elif method == 'alpha':
                print(f"Pertubating with method {method}")
                results = neuronPerturbation.compute_delta_logits_with_perturbation(method_name = "alpha", pertubation_value= pertubation_value, layer_names=layers_to_perturb)
            else:
                raise ValueError(f"Unsupported method: {method}. Use 'gaussian', 'mean', or 'alpha'.")
            
            print(f"length of the original predictions are {len(results['original_predicted_prob'])} and perturbed predictions are {len(results['perturbed_predicted_prob'])}")
            
            # Save immediately to Excel
            save_single_result_to_sheet(writer, results, image_list, idx)
            
            # Clean up immediately
            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    print(f"Results saved to {filename}")
    return filename


def save_single_result_to_sheet(writer, results, image_list, combination_idx):
    """
    Save a single perturbation result to an Excel sheet immediately.
    Args:
        writer: pandas ExcelWriter object
        results: Dictionary containing single perturbation result
        image_list: List of tuples containing (image_path, class_id, class_name)
        combination_idx: Index of the current combination
    """
    
    image_paths = [x[0] for x in image_list]
    class_ids = [x[1] for x in image_list]
    class_names = [x[2] for x in image_list]
    
    # Extract tensors and convert to numpy
    original_prob_tensor = np.concatenate([t.detach().cpu().numpy() for t in results['original_predicted_prob']])
    pertubrated_prob_tensor = np.concatenate([t.detach().cpu().numpy() for t in results['perturbed_predicted_prob']])
    original_class_tensor = np.concatenate([t.detach().cpu().numpy() for t in results['original_predicted_class']])
    pertubrated_class_tensor = np.concatenate([t.detach().cpu().numpy() for t in results['perturbed_predicted_class']])
    delta_logits_tensor = np.concatenate([t.detach().cpu().numpy() for t in results['delta_logits']])
    oritinal_logits_tensor = np.concatenate([t.detach().cpu().numpy() for t in results['original_logits']])
    perturbed_logits_tensor = np.concatenate([t.detach().cpu().numpy() for t in results['perturbed_logits']])
    # Split delta_logits into separate columns for each class
    delta_logits_class0 = delta_logits_tensor[:, 0]
    delta_logits_class1 = delta_logits_tensor[:, 1]
    delta_logits_class2 = delta_logits_tensor[:, 2]
    
    oritinal_logits_tensor_class0 = oritinal_logits_tensor[:, 0]  
    oritinal_logits_tensor_class1 = oritinal_logits_tensor[:, 1]  
    oritinal_logits_tensor_class2 = oritinal_logits_tensor[:, 2] 
    perturbed_logits_tensor_class0 = perturbed_logits_tensor[:, 0]
    perturbed_logits_tensor_class1 = perturbed_logits_tensor[:, 1]
    perturbed_logits_tensor_class2 = perturbed_logits_tensor[:, 2]
    # Create dataframe
    sheet_data = {
        'image_list': image_paths,
        'class_id': class_ids,
        'class_name': class_names,
        'method_name': results['method_name'],
        'pertubation_value': results['pertubation_value'],
        'original_prob': original_prob_tensor,
        'original_predicted_class': original_class_tensor,
        'perturbed_prob': pertubrated_prob_tensor,
        'perturbed_predicted_class': pertubrated_class_tensor,
        'oritinal_logits_tensor_class0': oritinal_logits_tensor_class0,
        'oritinal_logits_tensor_class1': oritinal_logits_tensor_class1,
        'oritinal_logits_tensor_class2': oritinal_logits_tensor_class2,
        'perturbed_logits_tensor_class0': perturbed_logits_tensor_class0,
        'perturbed_logits_tensor_class1': perturbed_logits_tensor_class1,
        'perturbed_logits_tensor_class2': perturbed_logits_tensor_class2,
        
        'delta_logits_class0': delta_logits_class0,
        'delta_logits_class1': delta_logits_class1,
        'delta_logits_class2': delta_logits_class2
    }
    
    df = pd.DataFrame(sheet_data)
    sheet_name = f'combination_{combination_idx}'
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Saved combination {combination_idx} to sheet {sheet_name}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_layer_config(config_path='layer_config.yaml'):
    """
    Load layer configuration from a YAML file.
    Args:
        config_path: Path to the YAML configuration file
    Returns:
        Dictionary containing layer configurations for different models
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Using default configuration.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

def get_layers_from_config(model_name, config_path='layer_config.yaml'):
    """
    Get layers to perturb for a specific model from YAML configuration.
    Args:
        model_name: Name of the model (e.g., 'inception_v3', 'resnet50')
        config_path: Path to the YAML configuration file
    Returns:
        List of layer names to perturb, or empty list if not found
    """
    config = load_layer_config(config_path)
    if config and 'models' in config and model_name in config['models']:
        layers = config['models'][model_name].get('layers_to_perturb', [])
        print(f"Loaded {len(layers)} layers for {model_name} from config file.")
        return layers
    else:
        print(f"No configuration found for model {model_name}. Returning empty list.")
        return []

def get_model_path_from_config(model_name, config_path='layer_config.yaml', default_path=None):
    """
    Get model path for a specific model from YAML configuration.
    Args:
        model_name: Name of the model (e.g., 'inception_v3', 'resnet50')
        config_path: Path to the YAML configuration file
        default_path: Default path to use if not found in config
    Returns:
        Model path as string
    """
    config = load_layer_config(config_path)
    if config and 'models' in config and model_name in config['models']:
        model_path = config['models'][model_name].get('model_path', default_path)
        print(f"Model path for {model_name}: {model_path}")
        return model_path
    else:
        print(f"No model path found for {model_name}. Using default: {default_path}")
        return default_path

    
def get_dataset_path_from_config(model_name, config_path='layer_config.yaml', default_path=None):
    """
    Get dataset path for a specific model from YAML configuration.
    Args:
        model_name: Name of the model (e.g., 'inception_v3', 'resnet50')
        config_path: Path to the YAML configuration file
        default_path: Default path to use if not found in config
    Returns:
        Dataset path as string
    """
    config = load_layer_config(config_path)
    batch_size = 32
    if config and 'models' in config and model_name in config['models']:
        batch_size = config['general'].get('batch_size', 64)
        dataset_path = config['general'].get('dataset_path', default_path)
        print(f"Dataset path for {model_name}: {dataset_path}")
        return dataset_path, batch_size
    else:
        print(f"No dataset path found for {model_name}. Using default: {default_path}")
        return default_path, batch_size

def get_perturbation_methods_from_config(model_name, config_path='layer_config.yaml'):
    """
    Get perturbation methods for a specific model from YAML configuration.
    Args:
        model_name: Name of the model (e.g., 'inception_v3', 'resnet50')
        config_path: Path to the YAML configuration file
    Returns:
        List of perturbation methods, defaults to ['gaussian', 'mean', 'alpha'] if not found in config
    """
    config = load_layer_config(config_path)
    if config and 'models' in config and model_name in config['models']:
        methods = config['models'][model_name].get('perturbation_methods', ['gaussian', 'mean', 'alpha'])
        print(f"Perturbation methods for {model_name}: {methods}")
        return methods
    else:
        print(f"No perturbation methods found for {model_name}. Using defaults: ['gaussian', 'mean', 'alpha']")
        return ['gaussian', 'mean', 'alpha']
   

if(__name__ == "__main__"):
    #base_dir = base_image_dir  # Replace with your base directory path
    
    # Define class directories
    class_dirs = {
        'deer': 0,
        'horse': 1,
        'zebra': 2
    }  
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_util = Logger_Singleton(f"perturbation_log_{datetime_str}.log")

    args = parse_arguments()
    if(os.name == 'posix'):
        model_name = args.model_name if args.model_name else 'inception_v3'
        config_path = args.config if args.config else 'layer_config.yaml'
        model_base_path = get_model_path_from_config(model_name, config_path, default_path=f'/mnt/sdd/basics/base_models/{model_name}.pth')
        base_image_dir, batch_size = get_dataset_path_from_config(model_name, config_path, default_path='/home/datasets/train')
        #Limit batch size to 32 to save memory
        batch_size = min(batch_size, 32)
        method = get_perturbation_methods_from_config(model_name, config_path)
        saveas = args.saveas if args.saveas else 'perturbation_results.xlsx'
    else:
        model_name = 'inception_v3'
        model_base_path = f'C:\\Users\\srikant1\\Downloads\\results\\neuronpertubation\\{model_name}.pth'  # Replace with your model path
        base_image_dir = r'C:\Users\srikant1\Downloads\results\neuronpertubation\train'  # Replace with your base image directory
        batch_size = 32
        config_path = os.path.join(os.getcwd(), 'layer_config.yaml')
        saveas = 'perbutation_results.xlsx'
    log_util.log("="*80)
    log_util.log(f"Perturbation method and options {saveas}")
    log_util.log("="*80)
    
    log_util.log(f"Starting perturbation process at {datetime_str}")
    log_util.log(f"Model Name: {model_name}")
    log_util.log(f"Model Path: {model_base_path}")
    log_util.log(f"Base Image Directory: {base_image_dir}")
    log_util.log(f"Batch Size: {batch_size}")    
    neuronPerturbation = NeuronPerturbationUtilities(device)
    neuronPerturbation.setmodel_name(model_name)
    image_list = []
    # Collect all images with their classes
    for class_name, class_label in class_dirs.items():
        class_path = os.path.join(base_image_dir, class_name)
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_list.append((os.path.join(class_path, img_file), class_label, class_name))
    images = [i[0] for i in image_list]
    classes = [(i[1], i[2]) for i in image_list]
    print(f"Image list collected with {len(image_list)} images.")
    print(f"Path of images are: image_list[0:5]: ", image_list[0:5])
    print(f"Path to images are {images[0:10]}")
    log_util.log(f"Collected {len(image_list)} images for analysis.")
    
    neuronPerturbation.batch_process_images(images, batch_size=batch_size)
    print("Images loaded into memory ")
    neuronPerturbation.print_parameters()
    neuronPerturbation.load_model(model_base_path)
    print("Main model loaded into memory ")
    if(args.layers_to_pertubate is not None):
        temp = args.layers_to_pertubate
        layers_to_perturb = [item.strip() for item in temp.split(',')]
    else:
        layers_to_perturb = get_layers_from_config(model_name, config_path = config_path)
        print(f"Layers to pertubate from config file {layers_to_perturb}")
    
    neuronPerturbation.set_layers_to_perturb(layers_to_perturb)
    print(f"Layers to perturb: {layers_to_perturb}")
    log_util.log(f"Options class and artifact {saveas}")
    log_util.log(f"List of layers to pertubate {layers_to_perturb}")
    #Perform Image pertubation on these layers
    #define a function to get layers to perturb
    pertubation_method = neuronPerturbation.getperturbation_methods(model_name)
    print(f"Pertubating with method Alpha Noise ")
    print(f"layers to pertubate {layers_to_perturb} ")
    
    
    def pertubate_store_results(layers_to_perturb, neuronPerturbation, method=None,pertubation_value=None , image_list = image_list, saveas=saveas):
        consolidated_results = perturbate_neurons(layers_to_perturb, neuronPerturbation, method=method,pertubation_value=pertubation_value , image_list = image_list, saveas=saveas)    
        # Clear memory before next perturbation method
        del consolidated_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for alpha in np.linspace(0, 1, 4):
        print(f"Pertubating with method Alpha Noise : {alpha} ")
        print("value of alpha is ", alpha)
        pertubate_store_results(layers_to_perturb, neuronPerturbation, method='alpha',pertubation_value=alpha , image_list = image_list, saveas=saveas)
    print(f"Pertubating with method Gaussian Noise ")
    print(f"layers to pertubate {layers_to_perturb} ")
    pertubate_store_results(layers_to_perturb, neuronPerturbation, method='gaussian',image_list = image_list, saveas=saveas)
    print(f"Pertubating with method Means {layers_to_perturb}",)
    pertubate_store_results(layers_to_perturb, neuronPerturbation, method='mean',  image_list = image_list, saveas=saveas)
    log_util.log(f"Perturbation process completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")