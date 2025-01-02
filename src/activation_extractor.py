import torch.nn as nn


class ActivationExtractor(nn.Module):
    def __init__(self, model, layer_name):
        super().__init__()
        self.model = model
        self.activations = None
        self.hook = None
        self.layer_name = layer_name


    def forward_hook(self, _, __, output):
        self.activations = output


    def register_hook(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self.hook = module.register_forward_hook(self.forward_hook)
                break


    def unregister_hook(self):
        if self.hook:
            self.hook.remove()
            self.hook = None


    def forward(self, x):
        _ = self.model(x)
        return self.activations