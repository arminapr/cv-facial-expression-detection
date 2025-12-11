"""
vit_pipeline.py

Trains, evaluates, and analyzes a Vision Transformer on the FER2013 dataset.
Generates metrics such as accuracy, mAP, and confusion matrix
"""

import os
import pickle
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, average_precision_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import ViTForImageClassification

# Custom Dataset Wrapper
class RemappedDataset(Dataset):
    """Dataset wrapper that remaps class labels"""
    def __init__(self, base_dataset, indices, label_mapping):
        self.base_dataset = base_dataset
        self.indices = indices
        self.label_mapping = label_mapping
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, label = self.base_dataset[actual_idx]
        return image, self.label_mapping[label]

# Utility Functions
def filter_and_balance_dataset(dataset: datasets.ImageFolder, exclude_class_idx: int = 1) -> RemappedDataset:
    """
    Remove samples from a specific class and balance remaining classes to minimum count.

    Args:
        dataset: ImageFolder dataset.
        exclude_class_idx: Index of class to remove (1 = Disgust).

    Returns:
        RemappedDataset with balanced classes and remapped labels.
    """
    targets = np.array(dataset.targets)
    valid_mask = targets != exclude_class_idx
    valid_indices = np.where(valid_mask)[0]
    valid_targets = targets[valid_mask]

    label_mapping = {old_label: new_label for new_label, old_label in enumerate(range(len(dataset.classes))) if old_label != exclude_class_idx}
    remapped_targets = np.array([label_mapping[t] for t in valid_targets])

    class_counts = Counter(remapped_targets)
    min_count = min(class_counts.values())

    balanced_indices = []
    for class_idx in sorted(class_counts.keys()):
        class_mask = remapped_targets == class_idx
        class_indices = valid_indices[class_mask]
        sampled_indices = np.random.choice(class_indices, size=min_count, replace=False)
        balanced_indices.extend(sampled_indices)

    np.random.shuffle(balanced_indices)
    return RemappedDataset(dataset, balanced_indices, label_mapping)

def compute_map(y_true: np.ndarray, y_scores: np.ndarray, num_classes: int) -> Tuple[float, List[float]]:
    """Compute mean Average Precision (mAP) for multi-class classification."""
    y_true_onehot = np.zeros((len(y_true), num_classes))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    return np.mean([average_precision_score(y_true_onehot[:, i], y_scores[:, i]) for i in range(num_classes)]), []

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], epoch: int, save_dir: str) -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}.png'))
    plt.close()

def evaluate_with_metrics(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int) -> Dict[str, float]:
    """Evaluate model and compute confusion matrix and mAP."""
    model.eval()
    all_labels, all_predictions, all_scores = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            total_loss += criterion(outputs, labels).item()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    accuracy = 100 * np.sum(all_predictions == all_labels) / len(all_labels)
    map_score, class_aps = compute_map(np.array(all_labels), np.array(all_scores), num_classes)

    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'map': map_score,
        'class_aps': class_aps
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir, test_dir = "./datasets/fer2013/train", "./datasets/fer2013/test"
    batch_size, learning_rate, num_epochs, weight_decay = 32, 1e-4, 7, 0.01
    num_classes, class_names = 6, ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    history = {
        'train_loss': np.zeros(num_epochs),
        'train_acc': np.zeros(num_epochs),
        'val_loss': np.zeros(num_epochs),
        'val_acc': np.zeros(num_epochs),
        'val_map': np.zeros(num_epochs),
        'val_confusion_matrices': []
    }

    cm_dir = "./confusion_matrices"
    os.makedirs(cm_dir, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.1)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = filter_and_balance_dataset(datasets.ImageFolder(train_dir, transform=train_transform))
    val_dataset = filter_and_balance_dataset(datasets.ImageFolder(test_dir, transform=val_transform))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_map, patience, patience_counter = 0.0, 3, 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        history['train_loss'][epoch] = running_loss / len(train_loader)
        history['train_acc'][epoch] = 100 * correct / total

        val_metrics = evaluate_with_metrics(model, val_loader, criterion, device, num_classes)
        history['val_loss'][epoch] = val_metrics['loss']
        history['val_acc'][epoch] = val_metrics['accuracy']
        history['val_map'][epoch] = val_metrics['map'] * 100
        history['val_confusion_matrices'].append(val_metrics['confusion_matrix'])
        plot_confusion_matrix(val_metrics['confusion_matrix'], class_names, epoch + 1, cm_dir)

        scheduler.step()

        if history['val_map'][epoch] > best_val_map:
            best_val_map = history['val_map'][epoch]
            patience_counter = 0
            os.makedirs("./checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_map': best_val_map,
                'history': history
            }, "./checkpoints/vit_fer_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    np.savez("./checkpoints/training_history.npz", **history)
    return history

if __name__ == "__main__":
    history = main()
    with open("training_metrics/metrics_VIT.pkl", "wb") as f:
        pickle.dump(history, f)
