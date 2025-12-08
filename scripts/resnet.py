import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet(num_classes=7, pretrained=True, freeze_layers=False):
    """"
    Returns a ResNet-18 model to work for FER2013

    This is similar to PA3 2.8, training using ResNet weights
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

    # Replace fc to match num_classes = 7
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
