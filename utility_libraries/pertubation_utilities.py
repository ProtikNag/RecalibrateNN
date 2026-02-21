import torch
from torchvision import models, transforms
from PIL import Image
import csv
import numpy as np
import os
import pandas as pd
import argparse
from scipy import stats

import torch.nn as nn

from logger import Logger_Singleton

class NeuronPerturbationUtilities:
    activations = {}
    image_tensor = []
    model = None
    hooks = None
    device = None
    model_name = None
    log_util = None
    def __init__(self, device=None) -> None :
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_util = Logger_Singleton()

    def print_parameters(self):
        print("Input image shapes")
        print(f"Number of batches: {len(self.image_tensor)}")
        print(f"Batch size: {self.image_tensor[0].shape[0]}")
        print(f"Image tensor shape: {self.image_tensor[0].shape}")
        self.log_util.log(f"Input image shapes: Number of batches: {len(self.image_tensor)}, \
                          Batch size: {self.image_tensor[0].shape[0]}, \
                          Image tensor shape: {self.image_tensor[0].shape}")    

    def getmodel_name(self):
        return  self.model_name
        
    def setmodel_name(self,model_name):
        self.model_name = model_name
        return
        
    def preprocess_image(self, image_path):
        """
        Preprocess an image for CNN models.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            image_tensor: Preprocessed image tensor with shape [1, 3, 224, 224]
        """
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)
    
    def batch_process_images(self, image_path_list, batch_size=32):
        """
        Batch process a list of image file paths and create tensors as batches.
        
        Args:
            image_path_list: List of image file paths
            batch_size: Number of images per batch (default: 32)
            
        Returns:
            batches: List of batched tensors, each with shape [batch_size, 3, 224, 224]
        """
        batches = []
        current_batch = []
        
        for image_path in image_path_list:
            # Preprocess and add to current batch
            image_tensor = self.preprocess_image(image_path)
            current_batch.append(image_tensor)
            
            # When batch is full, stack and add to batches list
            if len(current_batch) == batch_size:
                batch_tensor = torch.cat(current_batch, dim=0)
                batches.append(batch_tensor)
                current_batch = []
        
        # Handle remaining images in the last batch
        if len(current_batch) > 0:
            batch_tensor = torch.cat(current_batch, dim=0)
            batches.append(batch_tensor)
        self.image_tensor = batches
        self.log_util.log(f"Batch processed {len(image_path_list)} \
                          images into {len(batches)} batches with batch size {batch_size}.")
        return batches


    # Hook to capture activations
    def get_activation(self,name):
        def hook(_model, _input, output):
            self.activations[name] = output.detach()
            self.log_util.log(f"Captured activation for layer: {name}, \
                activation shape: {output.shape}")
        return hook

    def load_model(self, model_path=None):
        if model_path is None:
            raise ValueError("model_path cannot be None")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.log_util.log(f"Model loaded from {model_path} and set to evaluation mode.")
        return 
    
    def set_layers_to_perturb(self, layers):
        self.layers_to_perturb = layers
        self.log_util.log(f"Custom layers to perturb set: {layers}")
        return self.layers_to_perturb

    def getperturbation_methods(self, _model_name):
        perturbation_methods = ['mean_replacement', 'tcav_concept_subspacermoval', 'gaussian_noise_injection']
        return perturbation_methods

    def compute_perturbated_neuron_activation(self, model= None, image_list= None, layer_name= None, method_name = None, pertubation_value=1e-3):
        """
        Computes the mean neuron activation for a specified layer across multiple images.
        Optionally adds Gaussian noise to the activations.
            
        Args:
            model: PyTorch model
            image_list: List of preprocessed image tensors
            layer_name: Name of the layer to compute mean activations for
            method_name: Method to use for perturbation ('mean', 'gaussian', or 'alpha')
            pertubation_value: Value used for perturbation (standard deviation for 'gaussian', alpha for 'alpha') (default: 1e-3)
            
        Returns:
            perturbed_activation: Tensor with the same shape as the layer's output, 
                                    containing mean activations or activations with Gaussian noise
        """
        layer_activations = []
        model = self.model if model is None else model
        if model is None:
            raise ValueError("Model cannot be None. Please load a model first using load_model().")
        image_tensor_list = self.image_tensor if image_list is None else image_list
        
        # Find the target layer and register a hook
        target_layer = []
        for name, layer in model.named_modules():
            if name == layer_name:
                target_layer.append(layer)
                break
                
        self.log_util.log(f"Target layer found: {layer_name}")
        if len(target_layer) == 0:
            raise ValueError(f"Layer '{layer_name}' not found in model")
        
        # Register hook to capture activations
        hooks = []
        for i in range(len(target_layer)):
            temp_hook = target_layer[i].register_forward_hook(self.get_activation(layer_name))
            hooks.append(temp_hook)

        for image_tensor in image_tensor_list:
            # Forward pass to populate activations
            with torch.no_grad():
                _ = model(image_tensor)
                
            # Get activation for the specified layer
            if layer_name in self.activations:
                self.log_util.log(f"Captured activation for layer: {layer_name}, activation shape: {self.activations[layer_name].shape}")
                layer_activations.append(self.activations[layer_name].clone())
        
        # Remove the hook
        for hook in hooks:
            hook.remove()
            
        # Stack all activations and compute mean across batch dimension
        if len(layer_activations) > 0:
            stacked_activations = torch.cat(layer_activations, dim=0)
            self.log_util.log(f"Stacked activations shape for layer {layer_name}: {stacked_activations.shape}") 
            if method_name == 'gaussian':
                # Add Gaussian noise to original activations
                #noise = gaussian_value * torch.randn_like(stacked_activations) * stacked_activations.std(dim=0, keepdim=True) + stacked_activations.mean(dim=0, keepdim=True)
                noise =  pertubation_value * stacked_activations.std(dim=0, keepdim=True) * torch.randn_like(stacked_activations)
                noise = torch.full_like(stacked_activations, pertubation_value)
                perturbed_activation = stacked_activations + noise
                self.log_util.log(f"Added Gaussian noise with std: {pertubation_value}")
                #print(f"Added Gaussian noise with std: {gaussian_value}")
                #print(f"perurbed activation shape: {perturbed_activation.shape}, sample values: {perturbed_activation.view(-1)}")
            elif method_name == 'mean':
                # Use mean activation
                self.log_util.log(f"Using mean replacement for perturbation.")
                perturbed_activation = stacked_activations.mean(dim=0, keepdim=True)
            elif method_name == 'alpha':
                alpha = pertubation_value     
                print(f"Using alpha scaling for perturbation with alpha: {alpha}")           
                self.log_util.log(f"Using alpha scaling for perturbation with alpha: {alpha}")
                perturbed_activation = stacked_activations + stacked_activations * alpha
                print(f"Applied alpha scaling with alpha: {alpha}, sample values: {perturbed_activation.view(-1)[:5]}")
                print(f"Stacked activations sample values: {stacked_activations.view(-1)[:5]}")
            else:
                raise ValueError(f"Unsupported method_name: {method_name}. Use 'mean', 'gaussian', or 'alpha'.")
            
            self.log_util.log(f"Perturbed activation  values: {perturbed_activation[0].view(-1)[:]}")  # Log sample values for debugging
            
            #Checking normality of activations using Shapiro-Wilk test and printing standard deviation of activations for debugging purposes. This can help identify if the activations are heavily skewed or have outliers which might affect the perturbation results.
            self.log_util.log(f"Standard deviation of activations for layer {layer_name}: {stacked_activations.std(dim=0).mean().item():.4f}")  
            self.log_util.log(f"Checking normality of activations for layer {layer_name} using Shapiro-Wilk test.") 
            # Check if activations are normally distributed using Shapiro-Wilk test
            flattened = stacked_activations.cpu().numpy().flatten()
            # Sample if too large (Shapiro-Wilk limited to 5000 samples)
            if len(flattened) > 5000:
                sample = np.random.choice(flattened, 5000, replace=False)
            else:
                sample = flattened
            stat, p_value = stats.shapiro(sample)
            self.log_util.log(f"Shapiro-Wilk test: statistic={stat:.4f}, p-value={p_value:.4e}")
            self.log_util.log(f"Normally distributed: {p_value > 0.05}")
            stdev_activation = stacked_activations.std(dim=0, keepdim=True)
            self.log_util.log(f"Perturbed activation shape: {perturbed_activation.shape}, Std activation shape: {stdev_activation.shape}")
            self.log_util.log(f"Std activation sample values: {stdev_activation.view(-1)[:5]}")
            return perturbed_activation
        else:
            raise ValueError(f"Layer '{layer_name}' not found in activations")


            

    def compute_delta_logits_with_perturbation(self, method_name = None, pertubation_value = 1e-3,model = None, image_list = None, layer_names = None):
        """
        Computes delta logits between original and perturbed predictions using mean replacement.
        
        Args:
            model: PyTorch model
            image_list: List of preprocessed image tensors
            layer_names: List of layer names to perturb
            
        Returns:
            delta_logits_list: List of delta logits (original - perturbed) for each image
            original_logits_list: List of original logits for each image
            perturbed_logits_list: List of perturbed logits for each image
        """
        delta_logits_list = []
        original_logits_list = []
        perturbed_logits_list = []
        org_predicted_prob = None
        perturbed_predicted_prob = None
        self.log_util.log(f"="*80)
        model = self.model if model is None else model
        if model is None:
            raise ValueError("Model cannot be None. Please load a model first using load_model().")
        image_list = self.image_tensor if image_list is None else image_list
        
        if layer_names is None or not isinstance(layer_names, list):
            raise ValueError("layer_names must be a list of layer names")
        
        # Compute mean activations for all layers
        perturbed_activation = {}
        target_layers = {}
        for layer_name in layer_names:
            perturbed_activation[layer_name] = self.compute_perturbated_neuron_activation(model=model, image_list=image_list, layer_name=layer_name, method_name=method_name, pertubation_value=pertubation_value)              
            # Find the layer module
            for name, layer in model.named_modules():
                if name == layer_name:
                    target_layers[layer_name] = layer
                    break
            if layer_name not in target_layers:
                raise ValueError(f"Layer '{layer_name}' not found in model")
        org_predicted_prob = []
        perturbed_predicted_prob = []
        org_predicted_class = []
        perturbedpredicted_class = []
        
        for image_tensor in image_list:
            # Get original logits
            with torch.no_grad():
                original_logits = model(image_tensor)
                org_probabilities = torch.nn.functional.softmax(original_logits, dim=1)
                pred_prob, pred_class = torch.max(org_probabilities, dim=1)
                org_predicted_prob.append(pred_prob)
                org_predicted_class.append(pred_class)
                #print(f"Original Predicted class: {org_predicted_class}, Original Probability: {org_predicted_prob}")
            
            # Define hooks for all layers
            hooks = []
            for layer_name in layer_names:
                def perturbation_replacement_hook(_module, _input, output, ln=layer_name):
                    # Broadcast the perturbed activation to match current output batch size
                    self.log_util.log(f"Applying perturbation for layer: {ln}, original output shape: {output.shape}, perturbed activation shape: {perturbed_activation[ln].shape}")
                    perturb = perturbed_activation[ln]
                    #Get the batch size from the output
                    batch_size = output.shape[0]
                    if perturb.shape[0] > batch_size:
                        perturb = perturb[:batch_size]  # Use only the first 'batch_size' samples if perturbed activation has more samples than current output
                        #pop out the perturbed activation for the current layer to free up memory
                        perturbed_activation[ln] = perturbed_activation[ln][batch_size:]
                    print(output.shape, perturb.shape, perturbed_activation.keys())
                    return perturb.expand_as(output).clone()
                
                hook = target_layers[layer_name].register_forward_hook(perturbation_replacement_hook)
                hooks.append(hook)
            
            # Get perturbed logits with all layers perturbed
            with torch.no_grad():
                perturbed_logits = model(image_tensor)

                perturb_probs = torch.nn.functional.softmax(perturbed_logits, dim=1)
                pred_prob, pred_class = torch.max(perturb_probs, dim=1)
                #print(f"Perturbed Predicted class: {perturbed_predicted_class}, Perturbed Probability: {perturbed_predicted_prob}")
                
                pred_prob, pred_class = torch.max(org_probabilities, dim=1)
                perturbed_predicted_prob.append(pred_prob)
                perturbedpredicted_class.append(pred_class)


            # Remove all hooks
            for hook in hooks:
                hook.remove()
            
            # Compute delta
            delta_logits = original_logits - perturbed_logits
            
            delta_logits_list.append(delta_logits)
            original_logits_list.append(original_logits)
            perturbed_logits_list.append(perturbed_logits)
        
        results = {
            "method_name": method_name,
            "pertubation_value": pertubation_value,
            "delta_logits": delta_logits_list,
            "original_logits": original_logits_list,
            "perturbed_logits": perturbed_logits_list,
            "original_predicted_prob": org_predicted_prob,
            "perturbed_predicted_prob": perturbed_predicted_prob,
            "original_predicted_class": org_predicted_class,
            "perturbed_predicted_class": perturbedpredicted_class
        }
        return results
