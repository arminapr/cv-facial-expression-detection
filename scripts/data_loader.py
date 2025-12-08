import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch
import numpy as np
from collections import Counter

'''
the goal here is to balance the dataset so that each cateogru has the same number of photos
'''
def balance_dataset(dataset):    
    # Get all labels
    targets = np.array([dataset.targets[i] for i in range(len(dataset))])
    
    # find the minimum class count
    class_counts = Counter(targets)
    min_count = min(class_counts.values())
    
    # find the minimum sample
    balanced_indices = []
    
    for class_idx in range(len(dataset.classes)):
        class_indices = np.where(targets == class_idx)[0]
        sampled_indices = np.random.choice(class_indices, size=min_count, replace=False)
        balanced_indices.extend(sampled_indices)
    
    # take a random shuffle of the indices
    np.random.shuffle(balanced_indices)
    return Subset(dataset, balanced_indices)

# using this to remove any data labels we won't use
# help 
def remove_class(dataset, classname):
    orig_classes = list(dataset.classes)
    # new classes and mapping without the removed class
    new_classes = [c for c in orig_classes if c != classname]
    new_class_to_idx = {c: i for i, c in enumerate(new_classes)}
    # rebuild samples/targets with remapped labels
    new_samples = []
    for path, label in dataset.samples:
        orig_class = orig_classes[label]
        if orig_class == classname:
            continue
        new_label = new_class_to_idx[orig_class]
        new_samples.append((path, new_label))
    dataset.samples = new_samples
    dataset.imgs = new_samples
    dataset.targets = [s[1] for s in new_samples]
    dataset.classes = new_classes
    dataset.class_to_idx = new_class_to_idx
            
def get_dataloaders(data_dir="datasets/fer2013", batch_size=64, val_split=0.1, test_split=0.1, 
                   model_type='efficient', augmentation=True, balance_classes=True):
    """
    Get data loaders for FER2013 dataset.
    
    Args:
        model_type: 'efficient' for our CNN, 'resnet' for ResNet, 'vgg' for Custom VGG
        augmentation: Whether to use data augmentation for training
        balance_classes: Whether to undersample to balance classes
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
        
    elif model_type == 'vgg':
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)), # vgg input is 224 x 224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
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
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5], std=[0.5])
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
    #  disgust has much less samples so we will remove when balancing
    if balance_classes:
        remove_class(training_data, "disgust")
        remove_class(testing_data, "disgust")
    if balance_classes:
        training_data_balanced = balance_dataset(training_data)
        total_train_samples = len(training_data_balanced)
    else:
        training_data_balanced = training_data
        total_train_samples = len(training_data)
    
    # training data validation split
    validation_size = int(total_train_samples * val_split)
    training_size = total_train_samples - validation_size
    
    generator = torch.Generator().manual_seed(42)
    training_data_final, validation_data = random_split(
        training_data_balanced, 
        [training_size, validation_size],
        generator=generator
    )
    
    # For validation data, override transform to remove augmentation
    if model_type != 'resnet':
        # Need to access the root dataset through the Subset
        if balance_classes:
            training_data_balanced.dataset.transform = transform_test
        else:
            validation_data.dataset.transform = transform_test
    
    # Create data loaders
    training_loader = DataLoader(
        training_data_final, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
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