"""
train_resnet.py

Implements training, validation, and evaluation for resnet models.
"""


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
    print(f"Training started on {device.upper()} for {num_epochs} epochs")

    # storing metrics for each epoch
    training_loss = []
    training_acc = []
    val_loss_history = []  # Changed from val_loss to avoid conflict
    val_acc_history = []   # Changed from val_acc to avoid conflict

    for epoch_i in range(num_epochs):
        model.train()

        # storing running metrics and epoch-level metrics
        running_loss = 0.0
        running_total = 0.0
        running_correct = 0.0

        epoch_loss_sum = 0.0
        epoch_total = 0
        epoch_correct = 0

        for i, batch_data in enumerate(train_data_loader):
            images, labels = batch_data
            images = images.to(device)
            labels = labels.to(device).long()

            # Forward pass
            # outputs = model(images)
            # loss = loss_fn(outputs, labels)

            # GPT said to use this in order to improve training speed
            if device in ["cuda", "mps"]:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)

            batch_size = labels.size(0)
            epoch_loss_sum += loss.item()
            epoch_total += batch_size
            epoch_correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            
            if i % print_freq == 0 and running_total > 0:
                avg_loss = running_loss / running_total
                avg_acc = running_correct / running_total * 100
                last_lr = lr_scheduler.get_last_lr()[0]
                print(f"[{epoch_i + 1}/{num_epochs}, {i + 1:5d}/{len(train_data_loader)}] "
                      f"loss: {avg_loss:.3f} acc: {avg_acc:.2f}% lr: {last_lr:.5f}")

                running_loss = 0.0
                running_total = 0.0
                running_correct = 0.0

        lr_scheduler.step()

        # compute epoch-level training metrics
        epoch_avg_loss = epoch_loss_sum / epoch_total if epoch_total > 0 else 0.0
        epoch_avg_acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0.0
        training_loss.append(epoch_avg_loss)
        training_acc.append(epoch_avg_acc)

        # validation metrics (now returns loss and acc)
        val_loss_epoch, val_acc_epoch = test_model(model, val_data_loader, loss_fn=loss_fn, device=device)
        val_loss_history.append(val_loss_epoch)
        val_acc_history.append(val_acc_epoch)

        print(f"[{epoch_i + 1}/{num_epochs}] train loss: {epoch_avg_loss:.4f} train acc: {epoch_avg_acc:.2f}% | val loss: {val_loss_epoch:.4f} val acc: {val_acc_epoch:.2f}%")

    print("Training complete!")
    metrics = {
        'train_loss': training_loss,
        'train_acc': training_acc,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history,
    }
    return model, metrics


def test_model(model, data_loader, loss_fn=None, device="cuda"):
    """
    Evaluate model loss and accuracy
    Based on PA3 2.2 (test_model)
    """

    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch_data in data_loader:
            images, labels = batch_data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            batch_size = labels.size(0)
            total += batch_size
            correct += (predicted == labels).sum().item()
            if loss_fn is not None:
                loss = loss_fn(outputs, labels)
                loss_sum += loss.item() # TODO: might need to add weight by batch size later

    acc = 100 * correct / total
    if loss_fn is not None:
        val_loss_avg = loss_sum / total
    else:  
        val_loss_avg = 0.0
    return val_loss_avg, acc


def set_up_loss_optimizer_lr_scheduler(model, learning_rate, momentum, lr_step_size, lr_gamma):
    """
    Based on PA3 2.4
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step_size, lr_gamma)
    return loss_fn, optimizer, lr_scheduler