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

class NeuronPerturbationUtilities:
    activations = {}
    image_tensor = []
    model = None
    hooks = None
    device = None
    model_name = None
    def __init__(self, device=None) -> None :
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def print_parameters(self):
        print("Input image shapes")
        print(f"Number of batches: {len(self.image_tensor)}")
        print(f"Batch size: {self.image_tensor[0].shape[0]}")
        print(f"Image tensor shape: {self.image_tensor[0].shape}")
    
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
        return batches


    # Hook to capture activations
    def get_activation(self,name):
        def hook(_model, _input, output):
            self.activations[name] = output.detach()
        return hook

    def load_model(self, model_path=None):
        if model_path is None:
            raise ValueError("model_path cannot be None")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        
        # Register hooks for all layers
        #hooks = []
        #for name, layer in self.model.named_modules():
        #    if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d)):
        #        hooks.append(layer.register_forward_hook(self.get_activation(name)))
        
        return 

    def get_layerstoPerturb(self, model_name):
        layers_to_perturb = []
        if(model_name == "vgg16"):
            layers_to_perturb = ['features.2']
        if(model_name == "inception_v3"):
            layers_to_perturb = ['Conv2d_3b_1x1.conv','Mixed_5b.branch1x1.conv','Mixed_6a.branch3x3.conv','Mixed_6e.branch_pool.conv','Mixed_7a.branch7x7x3_3.conv','Mixed_7c.branch_pool.conv']
            layers_to_perturb = ['Conv2d_1a_3x3.conv',
'Mixed_5b.branch1x1.conv',
'Mixed_5c.branch3x3dbl_1.conv',
'Mixed_5d.branch5x5_2.conv',
'Mixed_6a.branch3x3dbl_1.conv',
'Mixed_6b.branch7x7_3.conv']
        #Using overriding methods here to make it easier to test with different layers. If layers_to_perturb is set using set_layers_to_perturb, it will override the default layers for the model.
        layers_to_perturb = self.layers_to_perturb if hasattr(self, 'layers_to_perturb') and self.layers_to_perturb is not None else layers_to_perturb
        return layers_to_perturb
    
    
    def set_layers_to_perturb(self, layers):
        self.layers_to_perturb = layers
        return self.layers_to_perturb

    def getperturbation_methods(self, _model_name):
        perturbation_methods = ['mean_replacement', 'tcav_concept_subspacermoval', 'gaussian_noise_injection']
        return perturbation_methods

    def compute_perturbated_neuron_activation(self, model= None, image_list= None, layer_name= None, gaussian_noise=False, gaussian_value=1e-3):
        """
        Computes the mean neuron activation for a specified layer across multiple images.
        Optionally adds Gaussian noise to the activations.
            
        Args:
            model: PyTorch model
            image_list: List of preprocessed image tensors
            layer_name: Name of the layer to compute mean activations for
            gaussian_noise: If True, adds Gaussian noise instead of using mean replacement
            gaussian_value: Standard deviation of Gaussian noise (default: 1e-3)
            
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
                layer_activations.append(self.activations[layer_name].clone())
        
        # Remove the hook
        for hook in hooks:
            hook.remove()
            
        # Stack all activations and compute mean across batch dimension
        if len(layer_activations) > 0:
            stacked_activations = torch.cat(layer_activations, dim=0)
            
            if gaussian_noise:
                # Add Gaussian noise to original activations
                noise = gaussian_value * torch.randn_like(stacked_activations) * stacked_activations.std(dim=0, keepdim=True) + stacked_activations.mean(dim=0, keepdim=True)
                
                noise =  gaussian_value * stacked_activations.std(dim=0, keepdim=True) * torch.randn_like(stacked_activations)

                #noise = gaussian_value * torch.randn_like(stacked_activations)
                noise = torch.full_like(stacked_activations, gaussian_value)
                perturbed_activation = stacked_activations + noise
                #print(f"Added Gaussian noise with std: {gaussian_value}")
                #print(f"perurbed activation shape: {perturbed_activation.shape}, sample values: {perturbed_activation.view(-1)[:5]}")
            else:
                # Use mean activation
                perturbed_activation = stacked_activations.mean(dim=0, keepdim=True)
            
            # Check if activations are normally distributed using Shapiro-Wilk test
            flattened = stacked_activations.cpu().numpy().flatten()
            # Sample if too large (Shapiro-Wilk limited to 5000 samples)
            if len(flattened) > 5000:
                sample = np.random.choice(flattened, 5000, replace=False)
            else:
                sample = flattened
            stat, p_value = stats.shapiro(sample)
            #print(f"Shapiro-Wilk test: statistic={stat:.4f}, p-value={p_value:.4e}")
            #print(f"Normally distributed: {p_value > 0.05}")
            stdev_activation = stacked_activations.std(dim=0, keepdim=True)
            #print(f"Perturbed activation shape: {perturbed_activation.shape}, Std activation shape: {stdev_activation.shape}")
            #print(f"Perturbed activation sample values: {perturbed_activation.view(-1)[:5]}")
            #print(f"Std activation sample values: {stdev_activation.view(-1)[:5]}")
            return perturbed_activation
        else:
            raise ValueError(f"Layer '{layer_name}' not found in activations")



    def compute_delta_logits_with_perturbation(self, gausian_noise = False, gausian_value = 1e-3,model = None, image_list = None, layer_names = None):
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
            perturbed_activation[layer_name] = self.compute_perturbated_neuron_activation( model = model, gaussian_noise = gausian_noise, gaussian_value = gausian_value, image_list = image_list, layer_name = layer_name)
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
                    perturb = perturbed_activation[ln]
                    #print(output.shape, perturb.shape, perturbed_activation.keys())
                    #return perturb.expand_as(output).clone()
                
                    if perturb.shape[0] == 1:
                        # If perturbed activation has batch size 1, expand it to match output
                        return perturb.expand_as(output).clone()
                    else:
                        # Otherwise, use only the first sample and expand
                        return perturb[0:1].expand_as(output).clone()
                
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
            "delta_logits": delta_logits_list,
            "original_logits": original_logits_list,
            "perturbed_logits": perturbed_logits_list,
            "original_predicted_prob": org_predicted_prob,
            "perturbed_predicted_prob": perturbed_predicted_prob,
            "original_predicted_class": org_predicted_class,
            "perturbed_predicted_class": perturbedpredicted_class
            
            
        }
        
        return results
