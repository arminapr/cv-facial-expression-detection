import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_resnet import (
    train_val_model,
    test_model,
    set_up_loss_optimizer_lr_scheduler
)
from resnet import get_resnet
from data_loader import get_dataloaders
import torch


if __name__ == "__main__":
    # === Setup ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")

    # Load FER2013 data
    train_loader, val_loader, test_loader, num_classes = get_dataloaders()

    # Load ResNet model (like Task 2.8)
    model = get_resnet(num_classes=num_classes, pretrained=True, freeze_layers=False)

    # Hyperparameters (same naming as PA3)
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001
    momentum = 0.9
    lr_step_size = 2
    lr_gamma = 0.2

    # Setup loss, optimizer, lr scheduler
    loss_fn, optimizer, lr_scheduler = set_up_loss_optimizer_lr_scheduler(
        model, learning_rate, momentum, lr_step_size, lr_gamma
    )

    # Train & validate
    model = train_val_model(
        model, train_loader, val_loader,
        loss_fn, optimizer, lr_scheduler,
        num_epochs, print_freq=50, device=device
    )

    # Final test accuracy
    test_acc = test_model(model, test_loader, device=device)
    print(f"🧪 Test Accuracy: {test_acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), "resnet18_fer2013.pth")
    print("💾 Model saved to resnet18_fer2013.pth")
