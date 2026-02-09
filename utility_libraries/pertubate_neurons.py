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


#parser = argparse.ArgumentParser(description='Extract layer activations from a model')
#parser.add_argument('--model_name', type=str, required=True, help='Name of the model (e.g., vgg16, resnet50)')
#parser.add_argument('--model_path', type=str, default=None, help='Path to the model file (default:/mnt/sdd/basics/base_models/{model_name}.pth)')
#parser.add_argument('--activation_path', type=str, default=None, help='Path to the model file (default:/mnt/sdd/basics/activations/{model_name}_activations/ )')
#parser.add_argument('--dataset', type=str, default=None, help='Path to the dataset  file (default:/home/datasets/train )')
#parser.add_argument('--saveactivations',action='store_true', help='Save activations (default: False)')
#args = parser.parse_args()
#activations
#model_name = args.model_name
#Usage 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perturb neurons in a neural network model')
    parser.add_argument('--model_name', type=str, default='inception_v3', help='Name of the model (e.g., vgg16, resnet50, inception_v3)')
    parser.add_argument('--model_path', type=str, help='Path to the model file')
    parser.add_argument('--dataset', type=str, help='Path to the dataset directory')
    parser.add_argument('--method', type=str, choices=['gaussian', 'mean'], default='gaussian', help='Perturbation method to use')
    return parser.parse_args()

def create_combinations(layers_to_modify):
    combinations = []
    for i in range(1, len(layers_to_modify) + 1):
        combinations.extend(itertools.combinations(layers_to_modify, i))
    return combinations

def perturbate_neurons(layers_to_modify , neuromaperturbation, method='gaussian'):
    results = {}
    return_consolidated_results = []
    layers_to_modify = create_combinations(layers_to_modify)
        
    print(f"Created {len(layers_to_modify)} combinations of layers to perturb.")
    print(f"Layers to modify: {layers_to_modify}")
    for layer in layers_to_modify:
        print(f"Perturbing layer: {layer}")
        layers_to_perturb = list(layer)
        if(method == 'gaussian'):
            print(f"Pertubating with method {method} ")
            print(f"layers to pertubate {layers_to_perturb} ")
            results = neuronPerturbation.compute_delta_logits_with_perturbation(gausian_noise=True, layer_names=layers_to_perturb)
        elif(method == 'mean'):
            print(f"Pertubating with method {method} ")
            results = neuronPerturbation.compute_delta_logits_with_perturbation(layer_names=layers_to_perturb)
        #Append the layers to modify to the results dictionary
        results['layers_to_modify'] = layers_to_modify
        return_consolidated_results.append(results)
        #print(f"Delta logits for layer {layer}: {results['delta_logits']}")
        #print(f"Original predicted probability: {results['original_predicted_prob']}, Perturbed predicted probability: {results['perturbed_predicted_prob']}")
        #print(f"Original predicted class: {results['original_predicted_class']}, Perturbed predicted class: {results['perturbed_predicted_class']}")
        print(f"length of the original predictions are {len(results['original_predicted_prob'])} and perturbed predictions are {len(results['perturbed_predicted_prob'])}")
        original_probs = []
        predicted_probs = []
        for i in range(len(results['original_predicted_prob'])):
            #print(f"Original predicted probability for image {i}: {results['original_predicted_prob'][i]}, Perturbed predicted probability for image {i}: {results['perturbed_predicted_prob'][i]}")
            #print(f"Original predicted class for image {i}: {results['original_predicted_class'][i]}, Perturbed predicted class for image {i}: {results['perturbed_predicted_class'][i]}")
            pass
    return return_consolidated_results


def save_results_to_excel(consolidated_results, filename, image_list):
    """
    Save perturbation results to an Excel file with multiple sheets.
    Args:
        consolidated_results: List of dictionaries containing perturbation results
        filename: Name of the output Excel file
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for i in range(len(consolidated_results)):
            # Reset tensors for each combination
            original_prob_tensor = []
            pertubrated_prob_tensor = []
            pertubrated_class_tensor = []
            original_class_tensor = []
            delta_logits_tensor = []
            image_paths  = [x[0] for x in image_list]
            class_ids    = [x[1] for x in image_list]
            class_names  = [x[2] for x in image_list]
            tensor_list = consolidated_results[i]['original_predicted_prob']
            original_prob_tensor.extend(np.concatenate([t.detach().cpu().numpy() for t in tensor_list]))
            tensor_list = consolidated_results[i]['perturbed_predicted_prob']
            pertubrated_prob_tensor.extend(np.concatenate([t.detach().cpu().numpy() for t in tensor_list]))
            tensor_list = consolidated_results[i]['original_predicted_class']
            original_class_tensor.extend(np.concatenate([t.detach().cpu().numpy() for t in tensor_list]))
            tensor_list = consolidated_results[i]['perturbed_predicted_class']
            pertubrated_class_tensor.extend(np.concatenate([t.detach().cpu().numpy() for t in tensor_list]))

            tensor_list = consolidated_results[i]['delta_logits']
            delta_logits_tensor.extend(np.concatenate([t.detach().cpu().numpy() for t in tensor_list]))
            # Create a dataframe for the current combination
            sheet_data = {
                'image_list':  image_paths,
                'class_id': class_ids,
                'class_name': class_names,
                'original_prob': original_prob_tensor,
                'original_predicted_class': original_class_tensor,
                'perturbed_prob': pertubrated_prob_tensor,
                'perturbed_predicted_class': pertubrated_class_tensor,
                'delta_logits': delta_logits_tensor
                
            }
            df = pd.DataFrame(sheet_data)
            sheet_name = f'combination_{i}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        #writing the summary sheet
        affected_layers = consolidated_results[0]['layers_to_modify']
        layers_column = [ " | ".join(tup)   for tup in affected_layers]
        summary_df = pd.DataFrame(layers_column, columns=['Layers Perturbed'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
            #'class0_delta_logits': delta_logit_c0,
            #'class1_delta_logits': delta_logit_c1,
            #'class2_delta_logits': delta_logit_c2,
            
 
            
    print(f"Results saved to {filename}")



#model_path = os.path.join(args.model_path , model_name, f'{model_name}.pth') if args.model_path else model_base_path
#base_image_dir = args.dataset if args.dataset else base_image_dir
#if(args.saveactivations):
#    save_activations = args.saveactivations
#else:
#    save_activations = False



#os.makedirs(activations_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

   

if(__name__ == "__main__"):
    #base_dir = base_image_dir  # Replace with your base directory path
    
    # Define class directories
    class_dirs = {
        'deer': 0,
        'horse': 1,
        'zebra': 2
    }

    args = parse_arguments()
    if(os.name == 'posix'):
        model_name = args.model_name if args.model_name else 'inception_v3'
        model_path = args.model_path if args.model_path  else '/mnt/sdd/basics/balanced_training' 
        #f'/mnt/sdd/basics/balanced_training/{model_name}/{model_path}'
        model_base_path = os.path.join(model_path, model_name , model_name+'.pth')
        base_image_dir = args.dataset if args.dataset else '/home/balanced_dataset/train'  
        #base_image_dir = '/home/balanced_dataset/'
        method = args.method   
    else:
        model_name = 'inception_v3'
        model_base_path = f'C:\\Users\\srikant1\\Downloads\\results\\neuronpertubation\\{model_name}.pth'  # Replace with your model path
        base_image_dir = r'C:\Users\srikant1\Downloads\results\neuronpertubation\train'  # Replace with your base image directory
    
    
    neuronPerturbation = NeuronPerturbationUtilities(device)
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
    neuronPerturbation.batch_process_images(images)
    neuronPerturbation.print_parameters()
    neuronPerturbation.load_model(model_base_path)
    layers_to_perturb = ['Mixed_6e.branch_pool.conv','Mixed_6e.branch7x7dbl_3.conv' ]
    neuronPerturbation.set_layers_to_perturb(layers_to_perturb)

    layers_to_perturb = neuronPerturbation.get_layerstoPerturb(model_name)
    

    print(f"Layers to perturb: {layers_to_perturb}")    
    #Perform Image pertubation on these layers
    #define a function to get layers to perturb
    pertubation_method = neuronPerturbation.getperturbation_methods(model_name)
    print(f"Pertubating with method Gaussian Noise ")
    print(f"layers to pertubate {layers_to_perturb} ")
    consolidated_results = perturbate_neurons(layers_to_perturb, neuronPerturbation, method='gaussian')
    save_results_to_excel(consolidated_results, f'perturbation_results_{model_name}_gaussian.xlsx', image_list)
    print(f"Pertubating with method Means {layers_to_perturb}",)
    consolidated_results = perturbate_neurons(layers_to_perturb, neuronPerturbation, method='mean')
    save_results_to_excel(consolidated_results, f'perturbation_results_{model_name}_means.xlsx', image_list)
    







"""
    for layer in layers_to_perturb:
        print(f"Perturbing layer: {layer}")
        layer = [layer]
        results = neuronPerturbation.compute_delta_logits_with_perturbation(gausian_noise=True, layer_names=layers_to_perturb)
        print(f"Delta logits for layer {layer}: {results['delta_logits']}")
        print(f"Original predicted probability: {results['original_predicted_prob']}, Perturbed predicted probability: {results['perturbed_predicted_prob']}")
        print(f"Original predicted class: {results['original_predicted_class']}, Perturbed predicted class: {results['perturbed_predicted_class']}")

        #print(f"Pertubating with method Gaussian Noise ")
        #results = neuronPerturbation.compute_delta_logits_with_perturbation(gausian_noise=True, layer_names=layers_to_perturb)
        #print(f"Delta logits with Gaussian Noise: {results['delta_logits']}")
        #print(f"Original predicted probability: {results['original_predicted_prob']}, Perturbed predicted probability: {results['perturbed_predicted_prob']}")
        #print(f"Original predicted class: {results['original_predicted_class']}, Perturbed predicted class: {results['perturbed_predicted_class']}")



    print(f"Pertubating with method Means ")
    for layer in layers_to_perturb:
        print(f"Perturbing layer: {layer}")
        layer = [layer]
        results = neuronPerturbation.compute_delta_logits_with_perturbation(layer_names=layer)
        print(f"Delta logits for layer {layer}: {results['delta_logits']}")
        print(f"Original predicted probability: {results['original_predicted_prob']}, Perturbed predicted probability: {results['perturbed_predicted_prob']}")
        print(f"Original predicted class: {results['original_predicted_class']}, Perturbed predicted class: {results['perturbed_predicted_class']}")

    
    
    
    
    #print(delta_logits_list, original_logits_list, perturbed_logits_list)

"""