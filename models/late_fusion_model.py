import torch
import torch.nn as nn
from torchvision import models

def initialize_model(config):
    if "swin" in config.model_name.lower():
        # Dynamically determine the Swin model size from config (tiny, small, base)
        swin_version = config.model_name.split("_")[-1]  # Extract the last part (e.g., 't', 's', 'b')
        
        if swin_version == "t":
            base_model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)  # Swin-Tiny
        elif swin_version == "s":
            base_model = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)  # Swin-Small
        elif swin_version == "b":
            base_model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)  # Swin-Base
        else:
            raise ValueError(f"Unsupported Swin version: {swin_version}. Use 'swin_t', 'swin_s', or 'swin_b'.")

        num_ftrs = base_model.head.in_features
        base_model.head = nn.Identity()  # Remove classification head
        
    elif "vit" in config.model_name.lower():
        # Initialize the Vision Transformer model
        base_model = models.__dict__[config.model_name](weights='IMAGENET1K_V1')
        
        # Remove the classifier head by replacing it with an identity layer
        num_ftrs = base_model.heads.head.in_features  # ViT head input features
        base_model.heads.head = nn.Identity()  # Replace classifier with identity layer
        
        # Construct the weights key after initialization
        weights_key = f"{config.model_name.capitalize()}_Weights.IMAGENET1K_V1"
        try:
            weights = getattr(models, weights_key, None)
        except AttributeError:
            raise ValueError(f"Weights not found for model: {config.model_name}")
        
    elif 'resnet' in config.model_name.lower():
        # Dynamically load the ResNet model and weights
        model_class = getattr(models, config.model_name)  # e.g., models.resnet34, models.resnet50
        weights_key = f"{config.model_name.capitalize()}_Weights.DEFAULT"
        try:
            weights = getattr(models, weights_key, None)
        except AttributeError:
            raise ValueError(f"Weights not found for model: {config.model_name}")
        
        base_model = model_class(weights=weights)
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Identity()  # Remove original fully connected layer

    elif 'efficientnet' in config.model_name.lower():
        # Handle EfficientNet models
        model_class = getattr(models, config.model_name)  # e.g., models.efficientnet_b0
        weights_key = f"{config.model_name.capitalize()}_Weights.DEFAULT"
        try:
            weights = getattr(models, weights_key, None)
        except AttributeError:
            raise ValueError(f"Weights not found for model: {config.model_name}")

        base_model = model_class(weights=weights)
        num_ftrs = base_model.classifier[1].in_features  # EfficientNet's final layer
        base_model.classifier = nn.Identity()  # Remove original classifier

    else:
        raise ValueError(f"Unsupported model: {config.model_name}")

    # Define the Late Fusion Model
    class LateFusionModel(nn.Module):
        def __init__(self, base_model, num_ftrs):
            super(LateFusionModel, self).__init__()
            self.base_model = base_model
            self.fc = nn.Sequential(
                nn.Linear(num_ftrs * 6, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            batch_size = x.size(0)
            x = x.view(batch_size * 6, 3, config.img_size[0], config.img_size[1])  # Reshape for 6 images
            features = self.base_model(x)  # Extract features
            features = features.view(batch_size, 6, -1)
            features = torch.flatten(features, 1)
            output = self.fc(features)
            return output

    # Instantiate the model
    model = LateFusionModel(base_model, num_ftrs)
    return model
