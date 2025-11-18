import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from resnet import get_resnet

if __name__ == "__main__":
    print("Loading ResNet model...")
    model = get_resnet(num_classes=7, pretrained=True)
    # Test a fake batch of 4 rgb images, 244x244
    x = torch.randn(4, 3, 224, 224)  
    out = model(x)
    print("Model forward pass successful!")
    print("Output shape:", out.shape) # Should be [4,7]