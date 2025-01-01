import torch
from torchvision import transforms

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

def get_data_transforms(config):
    """
    Returns the data transformations for train, validation, and test sets.
    :param config: Config object containing image size and other settings.
    """
    return {
        'train': transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Uncomment the following lines if you want additional augmentations:
            # transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
            transforms.ToTensor(),
            # AddGaussianNoise(mean=0.0, std=1),  # Uncomment to add Gaussian noise
            transforms.Normalize([-0.0437, -0.0380, -0.1277], [0.2313, 0.2045, 0.0707])
        ]),
        'val': transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize([-0.0437, -0.0380, -0.1277], [0.2313, 0.2045, 0.0707])
        ]),
        'test': transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize([-0.0437, -0.0380, -0.1277], [0.2313, 0.2045, 0.0707])
        ])
    }
