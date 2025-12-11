"""
vit_pipeline.py

Trains, evaluates, and analyzes a Vision Transformer on the FER2013 dataset.
Generates metrics such as accuracy, mAP, and confusion matrix
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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
        # Remap the label
        remapped_label = self.label_mapping[label]
        return image, remapped_label

def filter_and_balance_dataset(dataset, exclude_class_idx=1):
    """
    Remove samples from a specific class and balance remaining classes to minimum count
    
    Args:
        dataset: ImageFolder dataset
        exclude_class_idx: Index of class to remove (1 = Disgust)
    
    Returns:
        RemappedDataset with balanced classes and remapped labels
    """
    # Get all targets
    targets = np.array(dataset.targets)
    
    # Create mask for samples NOT in the excluded class
    valid_mask = targets != exclude_class_idx
    valid_indices = np.where(valid_mask)[0]
    valid_targets = targets[valid_mask]
    
    # Create label mapping (shift down classes after excluded one)
    label_mapping = {}
    new_label = 0
    for old_label in range(len(dataset.classes)):
        if old_label == exclude_class_idx:
            continue
        label_mapping[old_label] = new_label
        new_label += 1
    
    # Remap targets for counting
    remapped_targets = np.array([label_mapping[t] for t in valid_targets])
    
    # Count samples per class
    class_counts = Counter(remapped_targets)
    print(f"Class distribution before balancing: {dict(class_counts)}")
    
    # Find minimum class count
    min_count = min(class_counts.values())
    print(f"Balancing to minimum count: {min_count}")
    
    # Sample min_count samples from each class
    balanced_indices = []
    for class_idx in sorted(class_counts.keys()):
        class_mask = remapped_targets == class_idx
        class_indices = valid_indices[class_mask]
        
        # Randomly sample min_count samples
        sampled_indices = np.random.choice(class_indices, size=min_count, replace=False)
        balanced_indices.extend(sampled_indices)
    
    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)
    
    print(f"Total samples after filtering and balancing: {len(balanced_indices)}")
    
    return RemappedDataset(dataset, balanced_indices, label_mapping)

def compute_map(y_true, y_scores, num_classes):
    """Compute mean Average Precision (mAP) for multi-class classification"""
    y_true_onehot = np.zeros((len(y_true), num_classes))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    
    aps = []
    for i in range(num_classes):
        ap = average_precision_score(y_true_onehot[:, i], y_scores[:, i])
        aps.append(ap)
    
    return np.mean(aps), aps

def plot_confusion_matrix(cm, class_names, epoch, save_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}.png'))
    plt.close()

def evaluate_with_metrics(model, data_loader, criterion, device, num_classes):
    """Evaluate model and compute confusion matrix and mAP"""
    model.eval()
    all_labels = []
    all_predictions = []
    all_scores = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_scores = np.array(all_scores)
    
    cm = confusion_matrix(all_labels, all_predictions)
    accuracy = 100 * np.sum(all_predictions == all_labels) / len(all_labels)
    avg_loss = total_loss / len(data_loader)
    
    map_score, class_aps = compute_map(all_labels, all_scores, num_classes)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'map': map_score,
        'class_aps': class_aps,
        'all_labels': all_labels,
        'all_predictions': all_predictions,
        'all_scores': all_scores
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dir = "./datasets/fer2013/train"
    test_dir = "./datasets/fer2013/test"
    
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 7
    num_classes = 6  # Changed from 7 to 6 (removed Disgust)
    weight_decay = 0.01
    
    # Updated class names (removed Disgust)
    class_names = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    history = {
        'train_loss': np.zeros(num_epochs),
        'train_acc': np.zeros(num_epochs),
        'train_map': np.zeros(num_epochs),
        'val_loss': np.zeros(num_epochs),
        'val_acc': np.zeros(num_epochs),
        'val_map': np.zeros(num_epochs),
        'val_confusion_matrices': [],
        'val_class_aps': []
    }
    
    cm_dir = "./confusion_matrices"
    os.makedirs(cm_dir, exist_ok=True)
    
    print("Loading data...")
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
    
    # Load full datasets
    train_dataset_full = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(test_dir, transform=val_transform)
    
    train_dataset = filter_and_balance_dataset(train_dataset_full, exclude_class_idx=1)
    val_dataset = filter_and_balance_dataset(val_dataset_full, exclude_class_idx=1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print("\nSetting up model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    best_val_map = 0.0
    patience = 3
    patience_counter = 0
        
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
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
        history['val_class_aps'].append(val_metrics['class_aps'])
        plot_confusion_matrix(val_metrics['confusion_matrix'], class_names, epoch + 1, cm_dir)
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {history['train_loss'][epoch]:.4f}, Train Acc: {history['train_acc'][epoch]:.2f}% | "
              f"Val Loss: {history['val_loss'][epoch]:.4f}, Val Acc: {history['val_acc'][epoch]:.2f}%, Val mAP: {history['val_map'][epoch]:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        print("  Per-class AP: ", end="")
        for i, (name, ap) in enumerate(zip(class_names, val_metrics['class_aps'])):
            print(f"{name}: {ap*100:.1f}%", end="  " if i < len(class_names)-1 else "\n")
        
        # saves the best model based on validation mAP
        if history['val_map'][epoch] > best_val_map:
            best_val_map = history['val_map'][epoch]
            best_val_acc = history['val_acc'][epoch]
            patience_counter = 0
            os.makedirs("./checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_map': best_val_map,
                'history': history
            }, "./checkpoints/vit_fer_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # print best validation results
    print(f"\nBest Validation mAP: {best_val_map:.2f}%, Best Validation Acc: {best_val_acc:.2f}%")
    
    os.makedirs("./checkpoints", exist_ok=True)
    np.savez("./checkpoints/training_history.npz", 
             train_loss=history['train_loss'],
             train_acc=history['train_acc'],
             train_map=history['train_map'],
             val_loss=history['val_loss'],
             val_acc=history['val_acc'],
             val_map=history['val_map'])
    
    return history

if __name__ == "__main__":
    history = main()
    os.makedirs("training_metrics", exist_ok=True)
    with open(os.path.join("training_metrics", f"metrics_VIT.pkl"), "wb") as f:
        pickle.dump(history, f)
