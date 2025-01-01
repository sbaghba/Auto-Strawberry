import argparse
import torch
import wandb
from config import Config
from models.late_fusion_model import initialize_model
from utils.data_loader import prepare_datasets
from utils.transforms import get_data_transforms
from utils.training import train_model, get_loss_function, get_optimizer, get_scheduler
from utils.evaluation import evaluate_model, evaluate_training_data
from utils.visualization import visualize_model_predictions

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train, Evaluate, or Visualize the model")
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'visualize'], default='train',
                        help="Mode to run: 'train' (training), 'eval' (evaluation), 'visualize' (visualization)")
    parser.add_argument('--weights', type=str, default=None, 
                        help="Path to the model weights file to load (for evaluation or resuming training)")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Initialize configuration
    config = Config()

    # Initialize model
    model = initialize_model(config).cuda()

    # Prepare datasets and dataloaders
    datasets = prepare_datasets(config, get_data_transforms)
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=config.batch_size, shuffle=True)
                   for x in ['train', 'val', 'test']}

    # Initialize optimizer, scheduler, and loss function
    loss_fn = get_loss_function()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Initialize Weights & Biases (optional)
    wandb.init(project="Auto-Strawberry", config=config)
    wandb_logger = wandb

    if args.mode == 'train':
        # Training mode
        model = train_model(model, dataloaders, loss_fn, optimizer, scheduler, config.num_epochs, wandb_logger)
        torch.save(model.state_dict(), "best_model.pth")  # Save trained model
        wandb.save("best_model.pth")

    elif args.mode == 'eval':
        # Evaluation mode
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights))  # Load weights if provided
            print(f"Loaded model weights from {args.weights}")
        else:
            print("No weights file provided, using the current model.")
        
        evaluate_model(model, dataloaders, wandb_logger=wandb_logger)
        evaluate_training_data(model, dataloaders, wandb_logger=wandb_logger)

    elif args.mode == 'visualize':
        # Visualization mode
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights))  # Load weights if provided
            print(f"Loaded model weights from {args.weights}")
        else:
            print("No weights file provided, using the current model.")
        
        visualize_model_predictions(model, dataloaders)

if __name__ == "__main__":
    main()
