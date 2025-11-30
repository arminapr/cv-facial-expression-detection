import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.utils.data import DataLoader
import os

def main():
    # switch to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # dataset paths
    train_dir = "./datasets/fer2013/train"
    test_dir = "./datasets/fer2013/test"
    
    # training with different hyperparameters

    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10
    num_classes = 7

    # trnasform the data for VIT
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # required by VIT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("Preparing datasets and dataloaders")
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # load pre-trained vit model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes
    )
    model.to(device)

    # loss functions
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # forward pass
            outputs = model(images).logits
            loss = criterion(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            running_loss += loss.item()
            _loc, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

    # test eval
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # saving the model
    os.makedirs("../checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../checkpoints/vit_fer_model.pth")

if __name__ == "__main__":
    main()