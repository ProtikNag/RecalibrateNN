import torch
import csv
import sys

import torch.nn as nn
import torchvision.models as models

def get_layer_shapes(model, input_size):
    layer_info = []
    hooks = []
    def register_hook(module):
        def hook(module, input, output):
            # Get the module name (e.g., features.0)
            for name, mod in model.named_modules():
                layer_name = name            
                if mod is module:
                    layer_name = name
                    break
            else:
                layer_mod_name = module.__class__.__name__
            layer_info.append({
                'layer': f"{layer_name}, ({module.__class__.__name__})",
                'input_shape': tuple(input[0].size()) if input else None,
                'output_shape': tuple(output.size()) if isinstance(output, torch.Tensor) else None
            })
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))
    model.apply(register_hook)
    dummy_input = torch.randn(*input_size)
    model(dummy_input)
    for h in hooks:
        h.remove()
    return layer_info

def save_to_csv(layer_info, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['layer', 'input_shape', 'output_shape']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for info in layer_info:
            writer.writerow({
                'layer': info['layer'],
                'input_shape': info['input_shape'],
                'output_shape': info['output_shape']
            })

if __name__ == "__main__":
    # Example: Load a pre-trained ResNet18
    if len(sys.argv) < 3:
        print("Usage: python Untitled-1 <model_path> <image_shape>")
        print("Example: python Untitled-1 model.pth 1,3,224,224")
        sys.exit(1)

    model_path = sys.argv[1]
    image_shape = tuple(map(int, sys.argv[2].split(',')))
    model_path = sys.argv[1]
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    input_size = image_shape  # Batch size 1, 3 channels, 224x224 image

    layer_info = get_layer_shapes(model, input_size)

    # Print layer info
    for idx, info in enumerate(layer_info):
        print(f"Layer {idx+1}: {info['layer']}")
        print(f"  Input shape: {info['input_shape']}")
        print(f"  Output shape: {info['output_shape']}")

    # Export to CSV
    save_to_csv(layer_info, 'model_layers.csv')
    print("Layer information exported to model_layers.csv")