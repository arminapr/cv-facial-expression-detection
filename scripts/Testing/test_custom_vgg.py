import torch
from torch import nn, optim
from scripts.data_loader import get_dataloaders
from scripts.custom_vgg import CustomVGG
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, average_precision_score
from datetime import datetime
import numpy as np

# hyperparameters
batch_size = 64
learning_rate = 1e-4
num_epochs = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load the dataset
train_loader, val_loader, test_loader, num_classes = get_dataloaders(
    batch_size=batch_size,
    val_split=0.1,
    test_split=0.1,
    model_type='vgg', 
    augmentation=True,
    balance_classes=True
)

# initialize the model, loss function, and optimizer
model = CustomVGG(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
# test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(f"Test Accuracy: {100 * correct / total:.2f}%")

def evaluate_full_metrics(model, test_loader, device="cpu", class_names=None, save_prefix="results"):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ===== Confusion Matrix =====
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    if class_names:
        plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
        plt.yticks(np.arange(len(class_names)), class_names)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix_{datetime.now()}.png")
    plt.close()

    # ===== Average Precision (AP) per class =====
    num_classes = all_probs.shape[1]
    ap_scores = []

    print("\nAverage Precision (AP) per class:")
    for c in range(num_classes):
        binary_labels = (all_labels == c).astype(int)
        ap = average_precision_score(binary_labels, all_probs[:, c])
        ap_scores.append(ap)
        class_name = class_names[c] if class_names else f"Class {c}"
        print(f"{class_name}: AP = {ap:.4f}")

    mAP = np.mean(ap_scores)
    print(f"\nMean Average Precision (mAP): {mAP:.4f}")

    return cm, ap_scores, mAP

model_path = "custom_vgg_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

class_names = test_loader.dataset.classes

evaluate_full_metrics(model, test_loader, device=device, class_names=class_names, save_prefix=f"results_{num_epochs}_{learning_rate}")
