import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
import os
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def compute_map(y_true, y_scores, num_classes):
    """Compute mean Average Precision (mAP) for multi-class classification"""
    # Convert to one-hot encoding for true labels
    y_true_onehot = np.zeros((len(y_true), num_classes))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    
    # Compute AP for each class
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
            
            # Get probabilities using softmax
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_scores = np.array(all_scores)
    
    # Compute metrics
    cm = confusion_matrix(all_labels, all_predictions)
    accuracy = 100 * np.sum(all_predictions == all_labels) / len(all_labels)
    avg_loss = total_loss / len(data_loader)
    
    # Compute mAP
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
    # switch to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # dataset paths
    train_dir = "./datasets/fer2013/train"
    test_dir = "./datasets/fer2013/test"
    
    # training with different hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 7
    num_classes = 7
    weight_decay = 0.01  # L2 regularization
    
    # class names
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # track the metric
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
    
    # save to the confusion matrices
    cm_dir = "./confusion_matrices"
    os.makedirs(cm_dir, exist_ok=True)
    
    print("Loading data...")
    # trnasform the data for VIT
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
    
    print("Preparing datasets and dataloaders...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    
    # add more workers for faster data loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print("Setting up model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes,
        hidden_dropout_prob=0.1,  # add dropout for regularization
        attention_probs_dropout_prob=0.1
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing helps generalization
    # optimizer uses weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Cosine annealing with warm restarts or simple cosine decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # we stop early if no improvement in validation accuracy
    best_val_acc = 0.0
    best_val_map = 0.0
    patience = 3
    patience_counter = 0
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # forward pass
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Store training metrics
        history['train_loss'][epoch] = running_loss / len(train_loader)
        history['train_acc'][epoch] = 100 * correct / total
        
        
        # val metrics
        val_metrics = evaluate_with_metrics(model, val_loader, criterion, device, num_classes)
        history['val_loss'][epoch] = val_metrics['loss']
        history['val_acc'][epoch] = val_metrics['accuracy']
        history['val_map'][epoch] = val_metrics['map'] * 100
        history['val_confusion_matrices'].append(val_metrics['confusion_matrix'])
        history['val_class_aps'].append(val_metrics['class_aps'])
        plot_confusion_matrix(val_metrics['confusion_matrix'], class_names, epoch + 1, cm_dir)
        
        # step the scheduler
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {history['train_loss'][epoch]:.4f}, Train Acc: {history['train_acc'][epoch]:.2f}% | "
              f"Val Loss: {history['val_loss'][epoch]:.4f}, Val Acc: {history['val_acc'][epoch]:.2f}%, Val mAP: {history['val_map'][epoch]:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # print per-class AP
        print("  Per-class AP: ", end="")
        for i, (name, ap) in enumerate(zip(class_names, val_metrics['class_aps'])):
            print(f"{name}: {ap*100:.1f}%", end="  " if i < len(class_names)-1 else "\n")
        
        # save the best model based on mAP
        if history['val_map'][epoch] > best_val_map:
            best_val_map = history['val_map'][epoch]
            best_val_acc = history['val_acc'][epoch]
            patience_counter = 0
            os.makedirs("../checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_map': best_val_map,
                'history': history
            }, "./checkpoints/vit_fer_best.pth")
            print(f"  → Saved new best model with val_mAP: {best_val_map:.2f}%, val_acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # final results
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation mAP: {best_val_map:.2f}%")
    print("="*50)
    
    # Save history
    os.makedirs("../checkpoints", exist_ok=True)
    np.savez("../checkpoints/training_history.npz", 
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
    print("Saved metrics.")