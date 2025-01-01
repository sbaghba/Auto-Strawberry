import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def get_loss_function():
    """Returns the loss function to be used during training."""
    return nn.MSELoss()

def get_optimizer(model, config):
    """
    Returns the optimizer for the model.
    :param model: PyTorch model whose parameters will be optimized.
    :param config: Config object containing optimizer parameters like learning rate and weight decay.
    """
    return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

def get_scheduler(optimizer, config):
    """
    Returns the learning rate scheduler.
    :param optimizer: Optimizer to which the scheduler will be applied.
    :param config: Config object for scheduler hyperparameters.
    """
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold=1e-4)

def train_model(model, dataloaders, loss_fn, optimizer, scheduler, num_epochs, wandb_logger=None):
    """
    Trains the model and performs validation.
    :param model: PyTorch model to train.
    :param dataloaders: Dictionary of dataloaders for 'train' and 'val' phases.
    :param loss_fn: Loss function to be used during training.
    :param optimizer: Optimizer to update model weights.
    :param scheduler: Learning rate scheduler to adjust learning rate.
    :param num_epochs: Number of epochs to train the model.
    :param wandb_logger: Weights & Biases logger object for tracking metrics (optional).
    :return: Trained model with the best weights.
    """
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    break_counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_description(f"{phase} Epoch {epoch}")

                    inputs = inputs.cuda()
                    labels = labels.float().cuda()

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = loss_fn(outputs.squeeze(), labels)

                        if phase == 'train':
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    tepoch.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if wandb_logger:
                wandb_logger.log({f'{phase}_loss': epoch_loss})

            if phase == 'val':
                scheduler.step(epoch_loss)
                print(f'Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]:.10f}')
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    break_counter = 0  # Reset counter if validation loss improves
                else:
                    break_counter += 1
                    if break_counter > 10:
                        print("Validation loss did not improve. Stopping early.")
                        print(f'Best val loss: {best_loss:.4f}')
                        model.load_state_dict(best_model_wts)
                        return model

    print(f'Best val loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model
