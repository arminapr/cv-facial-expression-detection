import pickle
import matplotlib.pyplot as plt
import os

# retrieve the training metrics
metrics_save_path = 'training_metrics/ResNet15_5epoch.pkl'
with open(metrics_save_path, 'rb') as f:
    metrics = pickle.load(f)
    
file_name = metrics_save_path.split('/')[-1]

# retrieve the training/validation losses and accuracies
train_losses = metrics['train_loss']
train_accuracies = metrics['train_acc']
val_losses = metrics['val_loss']
val_accuracies = metrics['val_acc']

# plot for losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
# plt.show()

# save the training and validation losses and accuracies
plot_dir = './plots'
os.makedirs(plot_dir, exist_ok=True)

# save the plot in plots directory
plt.savefig(f'{plot_dir}/{file_name}_training_validation_loss.png')

# plot for accuracies
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
# plt.show()

# save the plots
plt.savefig(f'{plot_dir}/{file_name}_training_validation_accuracy.png')