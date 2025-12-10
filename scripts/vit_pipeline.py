import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
import os
import numpy as np
import pickle

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
    
    # track the metric
    history = {
        'train_loss': np.zeros(num_epochs),
        'train_acc': np.zeros(num_epochs),
        'val_loss': np.zeros(num_epochs),
        'val_acc': np.zeros(num_epochs)
    }
    
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
        
        # validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # store validation metrics
        history['val_loss'][epoch] = val_loss / len(val_loader)
        history['val_acc'][epoch] = 100 * val_correct / val_total
        
        # step the scheduler
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {history['train_loss'][epoch]:.4f}, Train Acc: {history['train_acc'][epoch]:.2f}% | "
              f"Val Loss: {history['val_loss'][epoch]:.4f}, Val Acc: {history['val_acc'][epoch]:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # save the best model
        if history['val_acc'][epoch] > best_val_acc:
            best_val_acc = history['val_acc'][epoch]
            patience_counter = 0
            os.makedirs("../checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }, "../checkpoints/vit_fer_best.pth")
            print(f"  → Saved new best model with val_acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # final results
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*50)
    np.savez("../checkpoints/training_history.npz", **history)

    
    return history

if __name__ == "__main__":
    history = main()
    os.makedirs("training_metrics", exist_ok=True)
    with open(os.path.join("training_metrics", f"metrics_VIT.pkl"), "wb") as f:
        pickle.dump(history, f)
    print("Saved metrics.")

    