import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_training_data(model, dataloader, tolerance=0.1, wandb_logger=None):
    """
    Evaluate the model on the training data.
    :param model: Trained PyTorch model.
    :param dataloader: Dictionary containing the training dataloader.
    :param tolerance: Tolerance for calculating accuracy (default: 0.1 or 10%).
    :param wandb_logger: Optional Weights & Biases logger for tracking metrics.
    """
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader['train']:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays for calculations
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Calculate evaluation metrics
    accuracy = np.mean(np.abs((predictions - true_labels) / true_labels) <= tolerance) * 100
    mae = mean_absolute_error(true_labels, predictions)
    rmse = np.sqrt(mean_squared_error(true_labels, predictions))
    r2 = r2_score(true_labels, predictions)
    variance = np.var(true_labels - predictions)

    # Print metrics
    print(f'Training Accuracy within ±{tolerance*100}% tolerance: {accuracy:.2f}%')
    print(f'Training Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Training Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Training Variance of Errors: {variance:.4f}')
    print(f'Training R-squared (R2): {r2:.4f}')

    # Log metrics to Weights & Biases if logger is provided
    if wandb_logger:
        wandb_logger.log({
            "Training Accuracy": accuracy,
            "Training MAE": mae,
            "Training RMSE": rmse,
            "Training Variance_of_Errors": variance,
            "Training R2": r2
        })


def evaluate_model(model, dataloader, tolerance=0.1, wandb_logger=None):
    """
    Evaluate the model on the test data.
    :param model: Trained PyTorch model.
    :param dataloader: Dictionary containing the test dataloader.
    :param tolerance: Tolerance for calculating accuracy (default: 0.1 or 10%).
    :param wandb_logger: Optional Weights & Biases logger for tracking metrics.
    """
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader['test']:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays for calculations
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Calculate evaluation metrics
    accuracy = np.mean(np.abs((predictions - true_labels) / true_labels) <= tolerance) * 100
    mae = mean_absolute_error(true_labels, predictions)
    rmse = np.sqrt(mean_squared_error(true_labels, predictions))
    r2 = r2_score(true_labels, predictions)
    variance = np.var(true_labels - predictions)

    # Print metrics
    print(f'Accuracy within ±{tolerance*100}% tolerance: {accuracy:.2f}%')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Variance of Errors: {variance:.4f}')
    print(f'R-squared (R2): {r2:.4f}')

    # Log metrics to Weights & Biases if logger is provided
    if wandb_logger:
        wandb_logger.log({
            "Accuracy": accuracy,
            "MAE": mae,
            "RMSE": rmse,
            "Variance_of_Errors": variance,
            "R2": r2
        })
