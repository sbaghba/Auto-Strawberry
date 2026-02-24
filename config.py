import os

class Config:
    # Get the base directory of the project (where config.py is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the config.py file
    
    # Path to the datasets folder (relative to the project root)
    datasets_dir = os.path.join(base_dir, 'datasets')  # Example: /project_root/datasets
    
    # Set the path to the data directory and CSV file relative to the datasets folder
    data_dir = os.path.join(datasets_dir, 'Auto-Strawberry')  # Example: /project_root/datasets/Auto-Strawberry
    json_file = os.path.join(datasets_dir, 'annotations.json')  # Example: /project_root/datasets/annotations.json
    save_dir = os.path.join(base_dir, 'visualizations')  # Example: /project_root/visualizations
    
    # Size to which images will be resized
    img_size = (224, 224) # Adjust based on the backbone model
    
    # Number of samples per batch
    batch_size = 8
    
    # Learning rate for the optimizer
    learning_rate = 1e-4
    
    # Number of epochs for training
    num_epochs = 100
    
    # Weight decay (L2 regularization) for the optimizer
    weight_decay = 1e-4
    
    # Number of worker threads for data loading
    num_workers = 8
    
    # Proportion of data to be used for training
    train_split = 0.8
    
    # Proportion of data to be used for validation
    val_split = 0.1
    
    # Proportion of data to be used for testing
    test_split = 0.1
    
    # Random seed for reproducibility
    seed = 42
    
    # Name of the model to be used
    model_name = 'resnet34' # Example: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', etc.    
    
    # Output feature to be predicted
    output_feature = 'Total_leaf_area_of_DP'  # Example: "Total_leaf_area_of_DP", "largest_Petiol_length_cm", "fresh_mass_g", "Avg_crown_diameter", etc.

# Instantiate the Config class
config = Config()
