import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloaders
from resnet import get_resnet

def train_model(model, train_data_loader, val_data_loader, loss_fn, optimizer, lr_scheduler, num_epochs, print_freq=50, device="cuda"):
    """
    Train and validate a ResNet model.

    Similar to PA3 2.3 (train_val_model)
    """

    model.to(device)
    print(f"Traiing started on {device.upper()} for {num_epochs} epochs")

    for epoch_i in range(num_epochs):
        model.train()

        running_loss = 0.0
        running_total = 0.0
        running_correct = 0.0

        for i, batch_data in enumerate(train_data_loader):
            images, labels = batch_dataimages = images.to(device)
            labels = labels.to(device).long()

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs,labels)

            # Backward pass
            optimizer.zer_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)

            running_loss += loss.item()
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            if i % print_freq == 0:
                avg_loss = running_loss / print_freq
                avg_acc = running_correct / running_total * 100
                last_lr = lr_scheduler.get_last_lr()[0]
                print(f"[{epoch_i + 1}/{num_epochs}, {i + 1:5d}/{len(train_data_loader)}] "
                      f"loss: {avg_loss:.3f} acc: {avg_acc:.2f}% lr: {last_lr:.5f}")
                
                running_loss = 0.0
                running_total = 0.0
                running_correct = 0.0

            lr_scheduler.step()
            val_acc = test_model(model, val_data_loader, device=device)
            print(f"[{epoch_i + 1}/{num_epochs}] val acc: {val_acc:.2f}%")
        print("Training complete!")
        return model

