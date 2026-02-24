import torch
import torch.nn as nn
from torchvision import models

def initialize_model(config):
        
    if 'resnet' in config.model_name.lower():
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
