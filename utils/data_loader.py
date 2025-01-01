import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from sklearn.model_selection import train_test_split

class LeafDataset(Dataset):
    def __init__(self, data, config, transform=None):
        self.data = data
        self.transform = transform
        self.config = config  # Pass config explicitly
        self.image_label_pairs = self._create_image_label_pairs()

    def _create_image_label_pairs(self):
        pairs = []
        for _, record in self.data.iterrows():
            label_value = record[self.config.output_feature]
            try:
                label = float(label_value)
            except ValueError as e:
                print(f"Processing label: {label_value}")
                print(f"Error converting label to float: {e}")
                continue

            img_paths = []
            for img_name in record['images']:
                if "cam0" in img_name:
                    img_path = os.path.join(self.config.data_dir, 'cam0', img_name + ".jpg")
                else:
                    img_path = os.path.join(self.config.data_dir, 'cam1', img_name + ".jpg")
                img_paths.append(img_path)
            
            pairs.append((img_paths, label))
        return pairs

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        img_paths, label = self.image_label_pairs[idx]
        images = []
        for img_path in img_paths:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images)  # Stack the images into a single tensor
        return images, label


def prepare_datasets(config, get_data_transforms):
    data = pd.read_json(config.json_file)
    train_data, test_data = train_test_split(
        data, 
        test_size=(1 - config.train_split), 
        random_state=config.seed
    )
    val_data, test_data = train_test_split(
        test_data, 
        test_size=(config.test_split / (config.test_split + config.val_split)), 
        random_state=config.seed
    )
    
    transforms = get_data_transforms(config)  # Explicitly call the function for transforms
    return {
        'train': LeafDataset(train_data, config, transform=transforms['train']),
        'val': LeafDataset(val_data, config, transform=transforms['val']),
        'test': LeafDataset(test_data, config, transform=transforms['test'])
    }
