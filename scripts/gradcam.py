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
from transformers import ViTForImageClassification

# load the model
def find_model_and_tranform(model_type='efficient', model_path='our_cnn_50_epoch.pth'):
    model = None
    transform = None
    # class names for expressions
    class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
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
        state_dict = torch.load(model_path, map_location='cpu')
        # Filter out mismatched layers
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(filtered_state_dict, strict=False)

        # transform the input images 
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif model_type == 'vit':
        vit_checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        # the transformer doesn't use the balanced dataset so we keep all 7 classes
        class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # we need eager attention implementation from the start because of hooks and since we need to store attentions
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=7,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            attn_implementation='eager'
        )
        model.load_state_dict(vit_checkpoint['model_state_dict'])
        model.config.output_attentions = True
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    model.eval()
    return model, transform, class_names

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
        self.activations = output[0].detach()  # output is a tuple for ViT
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        from transformers import ViTForImageClassification
        
        # Forward pass
        output = self.model(input_tensor)
        
        if hasattr(output, 'logits'):
            output = output.logits
        predicted_class = output.argmax(dim=1).item()
        
        if target_class is None:
            target_class = predicted_class
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get activations and gradients
        activations = self.activations  # Shape: [batch, num_patches + 1, hidden_dim]
        gradients = self.gradients      # Shape: [batch, num_patches + 1, hidden_dim]
        
        # if it' a vit model we need to change things because of the lack of conv layers
        if isinstance(self.model, ViTForImageClassification):
            # remove the cls token
            activations = activations[:, 1:, :] 
            gradients = gradients[:, 1:, :]
            
            # find weights by averaging gradients across the hidden dimension
            weights = torch.mean(gradients, dim=-1, keepdim=True)  # [batch, num_patches, 1]
            
            # weighted sum of activations
            gradcam = torch.sum(weights * activations, dim=-1)  # Shape: [batch, num_patches]
            gradcam = gradcam.squeeze(0).cpu().numpy()  # Shape: [num_patches]
            
            # reshape to 2D grid
            num_patches = int(np.sqrt(len(gradcam)))
            gradcam = gradcam.reshape(num_patches, num_patches)
            
            # use relu to focus on positive influences and normalize
            gradcam = np.maximum(gradcam, 0)
            gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
        else:
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            gradcam = torch.sum(weights * activations, dim=1, keepdim=True)
            gradcam = torch.relu(gradcam)
            gradcam = gradcam.squeeze().cpu().numpy()
            gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
        
        return gradcam, target_class
    
    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

# similar version of gradcam but for vit
class AttentionRollout:
    def __init__(self, model, head_fusion='mean', discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
    
    def generate(self, input_tensor):
        # enable the attention outputs
        with torch.no_grad():
            outputs = self.model(input_tensor, output_attentions=True)
        
        # get all attention maps
        attentions = outputs.attentions  # tuple of [batch, num_heads, seq_len, seq_len]
        attentions = torch.stack(attentions)  # [num_layers, batch, num_heads, seq_len, seq_len]
        
        # take average of heads
        if self.head_fusion == 'mean':
            attentions = attentions.mean(dim=2)  # [num_layers, batch, seq_len, seq_len]
        elif self.head_fusion == 'max':
            attentions = attentions.max(dim=2)[0]
        
        # remove batch
        attentions = attentions.squeeze(1)  # [num_layers, seq_len, seq_len]
        # find the residual connections
        num_tokens = attentions.shape[-1]
        eye = torch.eye(num_tokens).to(attentions.device)
        attentions = attentions + eye
        attentions = attentions / attentions.sum(dim=-1, keepdim=True)
        
        joint_attentions = attentions[0]
        for i in range(1, len(attentions)):
            joint_attentions = torch.matmul(attentions[i], joint_attentions)
        
        # get CLS token attention (this is the first token)
        cls_attention = joint_attentions[0, 1:] 
        
        num_patches = int(np.sqrt(len(cls_attention)))
        attention_map = cls_attention.reshape(num_patches, num_patches).cpu().numpy()
        # normalize the attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        pred_class = outputs.logits.argmax(dim=1).item()
        
        return {
            'attention_map': attention_map,
            'pred_class': pred_class
        }


def generate_gradcam(image_path, model, transform, target_layer, target_class=None, use_attention_rollout=False):
    """Generate Grad-CAM or Attention Rollout heatmap for a given image."""    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    if isinstance(model, ViTForImageClassification) and use_attention_rollout:
        # use attention rollout for vit since we don't have convolutional layers
        rollout = AttentionRollout(model)
        outputs = rollout.generate(input_tensor)
        heatmap = outputs['attention_map']
        pred_class = outputs['pred_class']
    else:
        # Initialize Grad-CAM
        gradcam_generator = GradCAM(model, target_layer)
        heatmap, pred_class = gradcam_generator.generate(input_tensor, target_class)
        gradcam_generator.remove_hooks()
    
    # Resize to original image size
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    
    return heatmap, pred_class, image


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


def process_class_images(test_dir, class_name, model, transform, target_layer, output_dir, num_images=30):
    """process 30 images from a class, generate Grad-CAM, and save results."""
    
    # Create output directory for this class
    class_output_dir = os.path.join(output_dir, f'gradcam_analysis_{class_name}')
    os.makedirs(class_output_dir, exist_ok=True)
    
    # Get image paths for this class
    class_dir = os.path.join(test_dir, class_name)
    image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_files = image_files[:num_images]  
    print(image_files)
    
    overlayed_images = []
    predictions = []
    
    for _, image_file in enumerate(image_files):
        image_path = os.path.join(class_dir, image_file)
        
        # generate Grad-CAM
        gradcam, pred_class, original_image = generate_gradcam(
            image_path, model, transform, target_layer, use_attention_rollout=isinstance(model, ViTForImageClassification)
        )
        overlayed = overlay_gradcam(original_image, gradcam)
        overlayed_images.append(overlayed)
        predictions.append(pred_class)
    
    # make a grid visualization
    fig, axes = plt.subplots(5, 6, figsize=(20, 8))
    fig.suptitle(f'Grad-CAM Analysis - True Class: {class_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    for _, (ax, img, pred) in enumerate(zip(axes.flat, overlayed_images, predictions)):
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


def analyze_all_classes(test_dir, model, transform, target_layer, class_names, output_base_dir='gradcam_analysis', num_images=30):
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
    
    return all_results


if __name__ == "__main__":
    # model, transform, class_names = find_model_and_tranform(model_type='resnet', model_path='resnet18_fer2013.pth')
    model, transform, class_names = find_model_and_tranform(model_type='resnet', model_path='resnet50_5step_.1.pth')
    # model, transform, class_names = find_model_and_tranform(model_type='vit', model_path='checkpoints/vit_fer_best.pth')
    # set the target layer based on model type
    if isinstance(model, EfficientFERNet):
        target_layer = model.blocks[-1]
    elif isinstance(model, models.ResNet):
        target_layer = model.layer4[-1]
    elif isinstance(model, ViTForImageClassification):
        target_layer = model.vit.encoder.layer[-1]  # Use the last encoder layer for Grad-CAM
    else:
        raise ValueError("Unsupported model type for Grad-CAM.")
    
    # Base directory containing test images organized by class
    test_dir = 'datasets/fer2013/test'
    output_dir = 'gradcam_analysis'
    
    # Process all classes
    results = analyze_all_classes(
        test_dir=test_dir,
        model=model,
        transform=transform,
        target_layer=target_layer,
        class_names=class_names,
        output_base_dir=output_dir,
        num_images=30
    )
