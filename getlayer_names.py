import torch
import torchvision.models as models
from torchinfo import summary

# Load the pre-trained VGG16 model
model = models.resnet50(pretrained=True)
print(model)
# Extract and print all layer names
for name, module in model.named_modules():
    print(name)

#summary(model, input_size=(1, 224, 224))
