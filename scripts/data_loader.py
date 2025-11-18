import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir="datasets/fer2013", batch_size=64, val_split=0.1):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "fer2013")
        data_dir = os.path.abspath(data_dir)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # Data set has grayscale images and we need RGB
        transforms.Resize((224,224)), # Resnet input size is (224,224)
        transforms.ToTensor(),
        # Mean and std come from ImageNet dataset used to train ResNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]) 
    ])

    training_data = datasets.ImageFolder(os.path.join(data_dir,"train"), transform=transform)
    testing_data = datasets.ImageFolder(os.path.join(data_dir,"test"), transform=transform)

    # Use 10% of training data as validation dataset
    validation_size = int(len(training_data) * val_split)
    training_size = len(training_data) - validation_size
    training_data, validation_data = random_split(training_data, [training_size, validation_size])

    # NOTE for future: Set num_workers to 0 if running slow/dataloader stuck
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=2)
    testing_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = len(testing_data.classes)
    return training_loader, validation_loader, testing_loader, num_classes