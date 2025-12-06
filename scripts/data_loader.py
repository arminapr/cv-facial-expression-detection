import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

# with help from Claude Opus 4.1, rewrote this function to include a more efficient version
def get_dataloaders(data_dir="datasets/fer2013", batch_size=64, val_split=0.1, 
                   model_type='resnet', augmentation=True):
    """
    Get data loaders for FER2013 dataset.
    
    Args:
        model_type: 'efficient' for our CNN, 'resnet' for ResNet
        augmentation: Whether to use data augmentation for training
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "fer2013")
        data_dir = os.path.abspath(data_dir)
    
    if model_type == 'resnet':
        # original ResNet configuration
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=3), # ResNet expects 3 channels
            transforms.Resize((224, 224)), # Resnet input size is (224, 224)
            transforms.ToTensor(),
            # Mean and std come from ImageNet dataset used to train ResNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_test = transform_train
        
    else: # our CNN model
        if augmentation:
            transform_train = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((48, 48)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(degrees=10),
                # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalization
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))  # Cutout augmentation
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        
        # Test/val without augmentation
        transform_test = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    # Load datasets
    training_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
    testing_data = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform_test)
    # training data validation
    validation_size = int(len(training_data) * val_split)
    training_size = len(training_data) - validation_size
    
    # Use a fixed seed for reproducible splits
    generator = torch.Generator().manual_seed(42)
    training_data, validation_data = random_split(
        training_data, 
        [training_size, validation_size],
        generator=generator
    )
    
    # For validation data, we need to override the transform to remove augmentation
    if model_type != 'resnet':
        validation_data.dataset.transform = transform_test
    
    # Create data loaders with optimized settings
    training_loader = DataLoader(
        training_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # Increase if you have more CPU cores
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    validation_loader = DataLoader(
        validation_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    testing_loader = DataLoader(
        testing_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    num_classes = len(testing_data.classes)
    print(f"Dataset loaded: {training_size} train, {validation_size} val, {len(testing_data)} test samples")
    print(f"Number of classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    
    return training_loader, validation_loader, testing_loader, num_classes