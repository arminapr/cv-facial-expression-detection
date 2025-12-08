# GPT used for plotting/graphs
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from train_resnet import train_model, test_model, set_up_loss_optimizer_lr_scheduler
from resnet import get_resnet
from data_loader import get_dataloaders
import pickle
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np

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
    print("Loaded pretrained model")

    # === Hyperparameters ===
    num_epochs = 10   # (low for testing, increase for final run)
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


    # === Plots ===
    def evaluate_full_metrics(model, test_loader, device="cpu", class_names=None, save_prefix="results"):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

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
        plt.savefig(f"{save_prefix}_confusion_matrix.png")
        plt.close()

        # ===== Precision, Recall, F1 =====
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )

        print("\nPer-Class Metrics:")
        for i in range(len(precision)):
            print(f"Class {i}:  Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}")

        # Save bar plots
        x = np.arange(len(precision))
        plt.figure(figsize=(10, 5))
        plt.bar(x, precision)
        plt.xticks(x, class_names, rotation=45)
        plt.title("Precision per Class")
        plt.savefig(f"{save_prefix}_precision.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(x, recall)
        plt.xticks(x, class_names, rotation=45)
        plt.title("Recall per Class")
        plt.savefig(f"{save_prefix}_recall.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(x, f1)
        plt.xticks(x, class_names, rotation=45)
        plt.title("F1 Score per Class")
        plt.savefig(f"{save_prefix}_f1.png")
        plt.close()

        return precision, recall, f1, cm
    
    class_names = test_loader.dataset.classes

    evaluate_full_metrics(model, test_loader, device=device, class_names=class_names, save_prefix=f"results_{num_epochs}_{learning_rate}")

    # === Save trained model ===
    torch.save(model.state_dict(), f"resnet34_fer2013_{num_epochs}_{learning_rate}.pth")
    print("Model saved as resnet34_fer2013.pth")

    # save the training and validation losses and accuracies
    metrics_dir = './training_metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f'resnet34_{num_epochs}_{learning_rate}.pkl'
    
    metrics_save_path = os.path.join(metrics_dir, base_filename)
    with open(metrics_save_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Saved training metrics to {metrics_save_path}")