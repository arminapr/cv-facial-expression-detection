"""
fixed_cnn.py

Provides a full training pipeline for our EfficientFER CNN and final model saving.

"""

# train_pipeline.py
import os
import pickle
from datetime import datetime
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from data_loader import get_dataloaders
from cnn_model import get_efficient_fer_model 

def set_up_training(
    model: torch.nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 30,
    scheduler_config: dict = None,
    use_adamw: bool = True,
):
    """
    Returns: loss_fn, optimizer, lr_scheduler
    scheduler_config (optional) keys:
      - 'type': 'cosine' or 'step' or 'cosine_restart'
      - 'T_max': (for cosine) number of epochs
      - 'step_size', 'gamma' (for step)
      - 'T_0' (for cosine_restart, in epochs)
    IMPORTANT: T_max / T_0 are specified in epochs (not steps)
    """
    loss_fn = nn.CrossEntropyLoss()

    if use_adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    if scheduler_config is None:
        scheduler_config = {'type': 'cosine', 'T_max': num_epochs}

    stype = scheduler_config.get('type', 'cosine')
    if stype == 'cosine':
        T_max = scheduler_config.get('T_max', num_epochs)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif stype == 'cosine_restart':
        T_0 = scheduler_config.get('T_0', 10)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0)
    elif stype == 'step':
        step_size = scheduler_config.get('step_size', 10)
        gamma = scheduler_config.get('gamma', 0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    return loss_fn, optimizer, lr_scheduler


def test_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: nn.Module = None, device: str = "cuda") -> Tuple[float, float]:
    """
    Evaluate model: returns (avg_loss_per_sample, accuracy_percent)
    Loss is averaged correctly over samples (weighted by batch size).
    """
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            bs = labels.size(0)
            total += bs
            correct += (preds == labels).sum().item()
            if loss_fn is not None:
                loss = loss_fn(outputs, labels)
                # weight by batch size
                loss_sum += loss.item() * bs

    acc = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = loss_sum / total if (loss_fn is not None and total > 0) else 0.0
    return avg_loss, acc


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    num_epochs: int,
    device: str = "cuda",
    print_freq: int = 50,
    use_amp: bool = True,
    save_dir: str = "./checkpoints",
):
    """
    Modern training loop:
      - moves model to device
      - uses autocast + GradScaler if available
      - aggregates loss properly (weighted by batch size)
      - steps lr_scheduler per epoch (or per batch if using warm restarts and user desires)
      - saves the final model
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    scaler = GradScaler() if use_amp and device.startswith("cuda") else None

    best_val_acc = 0.0
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            bs = labels.size(0)

            optimizer.zero_grad()
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            # metrics (weighted)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * bs
            running_correct += (preds == labels).sum().item()
            running_samples += bs

            if batch_idx % print_freq == 0 or batch_idx == len(train_loader):
                avg_batch_loss = running_loss / running_samples
                avg_batch_acc = 100.0 * running_correct / running_samples
                last_lr = optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch}/{num_epochs}] Batch {batch_idx}/{len(train_loader)}  | loss: {avg_batch_loss:.4f}  acc: {avg_batch_acc:.2f}%  lr: {last_lr:.6f}")

        # epoch-level train metrics
        epoch_train_loss = running_loss / running_samples if running_samples > 0 else 0.0
        epoch_train_acc = 100.0 * running_correct / running_samples if running_samples > 0 else 0.0
        metrics['train_loss'].append(epoch_train_loss)
        metrics['train_acc'].append(epoch_train_acc)

        # Validation
        val_loss, val_acc = test_model(model, val_loader, loss_fn=loss_fn, device=device)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)

        # Step scheduler: step per epoch (this works for CosineAnnealingLR and StepLR).
        # If you used CosineAnnealingWarmRestarts and want per-batch restarts, change logic accordingly.
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

        print(f"=== Epoch {epoch} Summary: train_loss: {epoch_train_loss:.4f} train_acc: {epoch_train_acc:.2f}% | val_loss: {val_loss:.4f} val_acc: {val_acc:.2f}%")

    # Save last model
    torch.save(model.state_dict(), os.path.join(save_dir, f"efficientFER_epoch{num_epochs}.pth"))
    print(f"Saved final model as efficientFER_epoch{num_epochs}.pth")

    print("Training complete.")
    return model, metrics


if __name__ == "__main__":
    # --- Config (tune these) ---
    batch_size = 64
    num_epochs = 30
    learning_rate = 0.001
    weight_decay = 0.0001
    width_mult = 1
    use_adamw = True
    use_amp = True

    # device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # data
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        batch_size=batch_size,
        val_split=0.1,
        model_type='efficient',
        augmentation=True
    )

    # use efficient fer model
    model = get_efficient_fer_model(num_classes=num_classes, width_mult=width_mult)

    # optimizer/scheduler
    scheduler_cfg = {'type': 'cosine', 'T_max': num_epochs}
    loss_fn, optimizer, lr_scheduler = set_up_training(model, learning_rate, weight_decay, num_epochs, scheduler_cfg, use_adamw=use_adamw)

    # training the model
    model, metrics = train_model(
        model, train_loader, val_loader,
        loss_fn, optimizer, lr_scheduler,
        num_epochs=num_epochs,
        device=device,
        print_freq=100,
        use_amp=use_amp,
        save_dir="./checkpoints"
    )

    # test final performance
    final_loss, final_acc = test_model(model, test_loader, loss_fn=loss_fn, device=device)
    print(f"Final test loss: {final_loss:.4f}, test acc: {final_acc:.2f}%")

    # save training metrics
    os.makedirs("training_metrics", exist_ok=True)
    with open(os.path.join("training_metrics", f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    print("Saved metrics.")
