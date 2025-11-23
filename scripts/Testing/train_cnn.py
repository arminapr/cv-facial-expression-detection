import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from train_resnet import train_model, test_model
from cnn_model import get_efficient_fer_model
from data_loader import get_dataloaders
import pickle
from datetime import datetime

def set_up_training(model, learning_rate=0.001, weight_decay=0.0001):
    """
    Optimized setup for efficient CNN
    """
    # Use AdamW for better weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
    )
    
    # Cosine annealing for smooth learning rate decay
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=10 # change with number of epochs
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
    
    # === Training hyperparameters ===
    num_epochs = 10 
    learning_rate = 0.001
    weight_decay = 0.0001
    
    # === Setup optimizer and scheduler ===
    loss_fn, optimizer, lr_scheduler = set_up_training(
        model, learning_rate, weight_decay
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
    metrics_path = os.path.join(metrics_dir, f'efficient_fer_{timestamp}.pkl')
    
    with open(metrics_path, 'wb') as f:
        pickle.dump({
            **metrics,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'model_params': sum(p.numel() for p in model.parameters())
        }, f)
    print(f"Metrics saved to {metrics_path}")