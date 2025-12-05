import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from train_resnet import train_model, test_model, set_up_loss_optimizer_lr_scheduler
from resnet import get_resnet
from data_loader import get_dataloaders
import pickle
from datetime import datetime

if __name__ == "__main__":
    # === Setup ===
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # === Load data ===
    train_loader, val_loader, test_loader, num_classes = get_dataloaders()
    print(f"Loaded FER2013 dataset with {num_classes} emotion classes")

    # === Load model ===
    model = get_resnet(num_classes=num_classes, pretrained=True, freeze_layers=False)
    print("Loaded pretrained ResNet18 model")

    # === Hyperparameters ===
    num_epochs = 50   # (low for testing, increase for final run)
    batch_size = 1 # (not used here, but set in data_loader)
    learning_rate = 0.001
    momentum = 0.9
    lr_step_size = 2
    lr_gamma = 0.2

    # === Loss, optimizer, and LR scheduler ===
    loss_fn, optimizer, lr_scheduler = set_up_loss_optimizer_lr_scheduler(
        model, learning_rate, momentum, lr_step_size, lr_gamma
    )

    # === Train and validate ===
    model, metrics = train_model(
        model, train_loader, val_loader,
        loss_fn, optimizer, lr_scheduler,
        num_epochs, print_freq=50, device=device
    )

    # === Evaluate on test set ===
    # test_model now returns (loss, acc) — unpack both for clarity
    test_loss, test_acc = test_model(model, test_loader, loss_fn=loss_fn, device=device)
    print(f"Final Test Loss: {test_loss:.4f} Final Test Accuracy: {test_acc:.2f}%")

    # === Save trained model ===
    torch.save(model.state_dict(), "resnet18_fer2013.pth")
    print("Model saved as resnet18_fer2013.pth")

    # save the training and validation losses and accuracies
    metrics_dir = './training_metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f'{timestamp}.pkl'
    
    metrics_save_path = os.path.join(metrics_dir, base_filename)
    with open(metrics_save_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Saved training metrics to {metrics_save_path}")