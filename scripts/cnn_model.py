"""
cnn_model.py

This script builds our custom CNN architecture used for facial emotion recognition.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# written with help of Claude Opus 4.1
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EfficientFERNet(nn.Module):
    def __init__(self, num_classes=7, width_mult=1.0):
        super().__init__()
        
        def ch(channels):
            return int(channels * width_mult)
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, ch(32), 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch(32)),
            nn.ReLU(inplace=True),
        )
        
        self.blocks = nn.Sequential(
            # Block 1: 48x48 -> 24x24
            DepthwiseSeparableConv(ch(32), ch(64), 3),
            SEBlock(ch(64), reduction=8),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2: 24x24 -> 12x12
            DepthwiseSeparableConv(ch(64), ch(128), 3),
            DepthwiseSeparableConv(ch(128), ch(128), 3),
            SEBlock(ch(128), reduction=8),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            
            # Block 3: 12x12 -> 6x6
            DepthwiseSeparableConv(ch(128), ch(256), 3),
            DepthwiseSeparableConv(ch(256), ch(256), 3),
            SEBlock(ch(256), reduction=8),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(ch(256), ch(128)),
            nn.BatchNorm1d(ch(128)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(ch(128), num_classes)
        )
        
        self._initialize_weights()
        
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def get_efficient_fer_model(num_classes=7, width_mult=0.75, pretrained=False):
    model = EfficientFERNet(num_classes=num_classes, width_mult=width_mult)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model