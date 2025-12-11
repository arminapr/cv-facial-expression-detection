import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from train_resnet import train_model, test_model
from cnn_model import get_efficient_fer_model
from data_loader import get_dataloaders
import pickle
from datetime import datetime
from sklearn.metrics import confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

def set_up_training(model, learning_rate=0.001, weight_decay=0.0001, num_epochs=10, steps_per_epoch=None):
    """
    Optimized setup for efficient CNN
    """
    # Use AdamW for better weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
    )

    if steps_per_epoch:
        T_max = num_epochs * steps_per_epoch
    else:
        T_max = num_epochs
    
    # Cosine annealing for smooth learning rate decay
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max # change with number of epochs
    )
    
    # Standard cross entropy (you could also use label smoothing here)
    loss_fn = nn.CrossEntropyLoss()
    
    return loss_fn, optimizer, lr_scheduler

if __name__ == "__main__":
    # === Device setup ===
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    
    
    # === Load data with proper configuration ===
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        batch_size=128,  # Can use larger batch with efficient model
        val_split=0.1,
        model_type='efficient',  # Important: use efficient settings
        augmentation=True  # Enable data augmentation
    )
    
    # === Create efficient model ===
    model = get_efficient_fer_model(
        num_classes=num_classes, 
        width_mult=0.75  # Good balance of size/performance
    )

    model = model.to(device)
    
    # === Training hyperparameters ===
    num_epochs = 20 
    learning_rate = 0.001
    weight_decay = 0.0001
    steps_per_epoch = len(train_loader)
    
    # === Setup optimizer and scheduler ===
    loss_fn, optimizer, lr_scheduler = set_up_training(
        model, learning_rate, weight_decay, num_epochs, steps_per_epoch
    )
    
    # === Train model ===
    print(f"\nStarting training for {num_epochs} epochs...")
    model, metrics = train_model(
        model, train_loader, val_loader,
        loss_fn, optimizer, lr_scheduler,
        num_epochs, 
        print_freq=50,  
        device=device
    )
    
    # === Test final performance ===
    test_loss, test_acc = test_model(model, test_loader, loss_fn=loss_fn, device=device)
    print(f"\n{'='*50}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*50}")

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
        plt.savefig(f"{save_prefix}_confusion_matrix.png")
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


    
    class_names = test_loader.dataset.classes

    evaluate_full_metrics(model, test_loader, device=device, class_names=class_names, save_prefix=f"results_{num_epochs}_{learning_rate}")
    
    # === Save model ===
    model_path = f"efficient_fer_{num_epochs}_{learning_rate}_{weight_decay}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_acc,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'width_mult': 0.75
    }, model_path)
    print(f"\nModel saved as {model_path}")
    
    # === Save metrics ===
    metrics_dir = './training_metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(metrics_dir, f'efficient_fer_{num_epochs}_{learning_rate}.pkl')
    
    with open(metrics_path, 'wb') as f:
        pickle.dump({
            **metrics,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'model_params': sum(p.numel() for p in model.parameters())
        }, f)
    print(f"Metrics saved to {metrics_path}")