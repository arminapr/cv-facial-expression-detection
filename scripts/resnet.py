"""
resnet.py

returns the specified resnet model
"""

import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet(num_classes=7, pretrained=True, freeze_layers=False):
    """"
    Returns a ResNet model to work for FER2013

    This is similar to PA3 2.8, training using ResNet weights
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

    # Replace fc to match num_classes (6 or 7 depending on if it's balanced or not)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Modify the first convolutional layer to accept 1-channel input
    if model.conv1.in_channels == 3:
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=model.conv1.bias is not None
        )

    return model
