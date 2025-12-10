# Import necessary packages and libraries
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import matplotlib.pyplot as plt
import os
from cnn_model import EfficientFERNet

# load the model
def find_model_and_tranform(model_type='efficient', model_path='our_cnn_50_epoch.pth'):
    model = None
    transform = None
    if model_type == 'efficient':
        model = EfficientFERNet(width_mult=0.75)  # Update width_mult to match the checkpoint
        # Adjust the classifier layer for the custom model
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 6)

        # load the weights
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Extract the model weights from the state_dict
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        # Load the extracted weights into the model
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Model weights loaded successfully.")
        except RuntimeError as e:
            print("Error loading state_dict:", e)

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    elif model_type == 'resnet':
        # used this as inspiration and used copilot's assistanc ewith the code
        # load the model
        model = models.resnet50(pretrained=False)
        # make sure the last layer we have matches the output
        model.fc = nn.Linear(model.fc.in_features, 6)

        # load the weights
        state_dict = torch.load('resnet50_5step_.1.pth', map_location='cpu')
        model.load_state_dict(state_dict)

        # transform the input images 
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    model.eval()
    return model, transform
    
# class names for expressions
class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_handle = target_layer.register_forward_hook(self.forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get activations and gradients
        activations = self.activations
        gradients = self.gradients
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        gradcam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU
        gradcam = torch.relu(gradcam)
        
        # Normalize
        gradcam = gradcam.squeeze().cpu().numpy()
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
        
        return gradcam, target_class
    
    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def generate_gradcam(image_path, model, transform, target_layer, target_class=None):
    """Generate Grad-CAM heatmap for a given image."""
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Initialize Grad-CAM
    gradcam_generator = GradCAM(model, target_layer)
    
    # Generate Grad-CAM
    gradcam, pred_class = gradcam_generator.generate(input_tensor, target_class)
    
    # Resize to original image size
    gradcam = cv2.resize(gradcam, (image.width, image.height))
    
    # Clean up hooks
    gradcam_generator.remove_hooks()
    
    return gradcam, pred_class, image


def overlay_gradcam(image, gradcam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay Grad-CAM heatmap on original image."""
    # Convert PIL image to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    overlayed_image = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    
    return overlayed_image


def process_class_images(test_dir, class_name, model, transform, target_layer, output_dir, num_images=100):
    """Process 10 images from a class, generate Grad-CAM, and save results."""
    
    # Create output directory for this class
    class_output_dir = os.path.join(output_dir, f'gradcam_analysis_{class_name}')
    os.makedirs(class_output_dir, exist_ok=True)
    
    # Get image paths for this class
    class_dir = os.path.join(test_dir, class_name)
    image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_files = image_files[:num_images]  
    
    overlayed_images = []
    predictions = []
    
    print(f"\nProcessing class: {class_name}")
    print("-" * 50)
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(class_dir, image_file)
        
        # Generate Grad-CAM
        gradcam, pred_class, original_image = generate_gradcam(
            image_path, model, transform, target_layer
        )
        
        # Create overlay
        overlayed = overlay_gradcam(original_image, gradcam)
        overlayed_images.append(overlayed)
        predictions.append(pred_class)
    
    # Create a grid visualization
    fig, axes = plt.subplots(5, 6, figsize=(20, 8))
    fig.suptitle(f'Grad-CAM Analysis - True Class: {class_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    for idx, (ax, img, pred) in enumerate(zip(axes.flat, overlayed_images, predictions)):
        ax.imshow(img)
        ax.set_title(f'Pred: {class_names[pred]}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the grid
    grid_path = os.path.join(class_output_dir, f'gradcam_grid_{class_name}.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Grid saved to: {grid_path}")
    
    return overlayed_images, predictions


def analyze_all_classes(test_dir, model, transform, target_layer, output_base_dir='gradcam_analysis', num_images=30):
    """Process all classes and generate Grad-CAM visualizations."""
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each class
    all_results = {}
    
    for class_name in class_names:
        overlayed_images, predictions = process_class_images(
            test_dir, class_name, model, transform, target_layer, output_base_dir, num_images
        )
        
        all_results[class_name] = {
            'images': overlayed_images,
            'predictions': predictions
        }
    
    # Generate summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for class_name in class_names:
        preds = all_results[class_name]['predictions']
        true_class_idx = class_names.index(class_name)
        correct = sum(1 for p in preds if p == true_class_idx)
        accuracy = (correct / len(preds)) * 100
        
        print(f"{class_name.capitalize():10s}: {correct}/{len(preds)} correct ({accuracy:.1f}%)")
    
    print("\nAll Grad-CAM analyses saved to:", output_base_dir)
    
    return all_results


# Main execution
if __name__ == "__main__":
    model, transform = find_model_and_tranform(model_type='efficient', model_path='our_cnn_50_epoch.pth')
    
    # set the target layer based on model type
    if isinstance(model, EfficientFERNet):
        target_layer = model.blocks[-1]
    elif isinstance(model, models.ResNet):
        target_layer = model.layer4[-1]
    else:
        raise ValueError("Unsupported model type for Grad-CAM.")
    
    # Base directory containing test images organized by class
    test_dir = '/Users/armina/Documents/GitHub/cv-facial-expression-detection/datasets/fer2013/test'
    
    # Output directory for Grad-CAM results
    output_dir = 'gradcam_analysis'
    
    # Process all classes
    results = analyze_all_classes(
        test_dir=test_dir,
        model=model,
        transform=transform,
        target_layer=target_layer,
        output_base_dir=output_dir,
        num_images=30
    )
    
    print("\nDone! Check the 'gradcam_analysis' folder for results.")