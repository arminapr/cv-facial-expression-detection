"""
test_data_loader.py

This script is used to test our data loader and ensure it is loading the proper data.

"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_loader import get_dataloaders

if __name__ == "__main__":
    train_loader, val_loader, test_loader, num_classes = get_dataloaders()

    print(f"Data loaded successfully!")
    print(f"Classes: {num_classes}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Testing batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
