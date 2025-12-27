import cv2
import torch
import numpy as np
from PIL import Image
from gradcam import find_model_and_tranform, GradCAM, overlay_gradcam
from torchvision.transforms import functional as F
from gradcam import AttentionRollout

# set the model type and weights
MODEL_TYPE = 'vit'
MODEL_PATH = 'checkpoints/vit_fer_best.pth'
model, transform, class_names = find_model_and_tranform(model_type=MODEL_TYPE, model_path=MODEL_PATH)

# set the target layer based on the model type
if MODEL_TYPE == 'efficient':
    from cnn_model import SEBlock
    for layer in reversed(model.blocks):
        if isinstance(layer, SEBlock):
            target_layer = layer
            break
elif MODEL_TYPE == 'resnet':
    from torchvision.models import resnet50
    target_layer = model.layer4[-1]
elif MODEL_TYPE == 'vit':
    from transformers import ViTForImageClassification
    target_layer = model.vit.encoder.layer[-1]
elif MODEL_TYPE == 'custom_vgg':
    from custom_vgg import CustomVGG
    for layer in reversed(model.features):
        if isinstance(layer, torch.nn.Conv2d):
            target_layer = layer
            break
else:
    raise ValueError("Unsupported model type for Grad-CAM.")    

# Works on a Mac for camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()


if MODEL_TYPE != 'vit':
    gradcam_generator = GradCAM(model, target_layer)
else:
    rollout = AttentionRollout(model)
    
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # use open cv to detect faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # find what the face ROI is
        face_bgr = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        input_tensor = transform(face_pil).unsqueeze(0)

        # Generate Grad-CAM heatmap
        if MODEL_TYPE == 'vit':
            outputs = rollout.generate(input_tensor)
            heatmap = outputs['attention_map']
            pred_class = outputs['pred_class']
        else:
            heatmap, pred_class = gradcam_generator.generate(input_tensor, target_class=None)

        # make the heatmap the same size as the face and put it over the face
        heatmap_resized = cv2.resize(heatmap, (w, h))
        overlayed_face_rgb = overlay_gradcam(face_rgb, heatmap_resized)
        
        # have to change it for open cv
        overlayed_face_bgr = cv2.cvtColor(overlayed_face_rgb, cv2.COLOR_RGB2BGR)
        frame[y:y+h, x:x+w] = overlayed_face_bgr

        # show the predicted class label on top of the heatmap
        cv2.putText(frame, f"Pred: {class_names[pred_class]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('GradCAM/Attn Rollout Inference', frame)

    # if the user presses q, break the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

if MODEL_TYPE != 'vit':
    gradcam_generator.remove_hooks()
# release resources 
cap.release()
cv2.destroyAllWindows()