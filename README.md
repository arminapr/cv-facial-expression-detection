# cv-facial-expression-detection
Facial Expression Detection using Deep Learning

# [Demo](https://northeastern-my.sharepoint.com/:f:/r/personal/parvareshrizi_a_northeastern_edu/Documents/CV-FER-AP%20and%20EA?csf=1&web=1&e=BQkTZX)


## Overview
This repository contains scripts and resources for training, evaluating, and visualizing various deep learning models for facial expression detection. The models are trained on the FER2013 dataset and include architectures such as ResNet, VGG, and Vision Transformers (ViT).

## Repository Structure
- **checkpoints/**: Pre-trained model weights for various architectures
- **confusion_matrices/**: Confusion matrices generated during evaluation
- **datasets/**: FER2013 dataset organized into training and testing splits
- **gradcam/**: Grad-CAM visualizations for different models and expressions
- **plots/**: Visualization plots for training metrics
- **scripts/**: Main files for model training, evaluation, and utilities
  - `cnn_model.py`: Our custom CNN model 
  - `custom_vgg.py`: Implements a custom VGG model
  - `data_collection.py`: Handles data collection and preprocessing
  - `data_loader.py`: Data loading utilities used by other CNN files
  - `gradcam.py`: Generates Grad-CAM visualizations
  - `plot.py`: Generates plots for metrics
  - `resnet.py`: Defines ResNet architectures
  - `train_resnet.py`: Script for training ResNet models
  - `vit_pipeline.py`: Pipeline for Vision Transformer (ViT) models
  - **Testing/**: Contains test scripts for various modules to run and evaluate the models

## Running the Models
### Scripts to Run as Standalone Files
The following scripts can be executed directly as standalone files (using python <filename>):
- `gradcam.py`: Generates Grad-CAM visualizations
- `train_resnet.py`: Trains ResNet models
- `vit_pipeline.py`: Runs the Vision Transformer pipeline
- any file under `Testing/` can be run standalone with the exception of the custom vgg file below

### Scripts to Run as Modules
The following file needs to be run as a module for the pipeline to work:
- `Testing/test_custom_vgg.py`

The following scripts are designed to be imported as modules:
- `cnn_model.py`
- `custom_vgg.py`
- `data_loader.py`
- `plot.py`
- `resnet.py`

## Requirements
Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Dataset
The FER2013 dataset is used for training and evaluation. Ensure the dataset is placed in the `datasets/fer2013/` directory with the following structure:
```
datasets/
  fer2013/
    train/
      angry/
      disgust/
      fear/
      happy/
      neutral/
      sad/
      surprise/
    test/
      angry/
      disgust/
      fear/
      happy/
      neutral/
      sad/
      surprise/
```

## Pre-trained Models
Pre-trained model weights are available in the `checkpoints/` directory. Use these weights for evaluation or fine-tuning. More weights that are larger than git allows can be found here: https://northeastern-my.sharepoint.com/:f:/r/personal/parvareshrizi_a_northeastern_edu/Documents/CV-FER-AP%20and%20EA?csf=1&web=1&e=BQkTZX
