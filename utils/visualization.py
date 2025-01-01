import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torchviz import make_dot
import matplotlib.cm as cm
from config import Config

def plot_predictions(images, measured_labels, estimated_labels):
    """
    Plot images with measured and estimated labels for comparison.
    :param images: Batch of images.
    :param measured_labels: Ground truth labels.
    :param estimated_labels: Predicted labels.
    """
    num_samples = len(measured_labels)
    fig, axs = plt.subplots(num_samples, 6, figsize=(18, num_samples * 3))
    fig.suptitle('Measured vs Estimated Labels', fontsize=16)

    for i in range(num_samples):
        for j in range(6):
            # Convert image from tensor to numpy
            img = images[i, j].permute(1, 2, 0).cpu().numpy()
            # Denormalize the image
            img = denormalize_image(img)

            # Display the image
            axs[i, j].imshow(img)
            axs[i, j].axis('off')

            # Display the measured and estimated labels below the first image in the row
            if j == 0:
                measured_label = measured_labels[i]
                estimated_label = estimated_labels[i]
                axs[i, j].set_title(f'Measured: {measured_label:.2f}\nEstimate: {estimated_label:.2f}', fontsize=12, color='blue')

def visualize_model_predictions(model, dataloader, num_samples=1000):
    """
    Visualize model predictions by generating various plots.
    :param model: Trained model to make predictions.
    :param dataloader: DataLoader for the test set.
    :param num_samples: Number of samples to visualize.
    """
    model.eval()
    measured_labels_list, estimated_labels_list = [], []

    with torch.no_grad():
        for inputs, measured_labels in dataloader['test']:
            inputs = inputs.cuda()
            measured_labels = measured_labels.cuda()

            outputs = model(inputs)
            estimated_labels = outputs.squeeze().cpu().numpy()
            measured_labels = measured_labels.cpu().numpy()

            # Append measured and estimated labels for plotting
            measured_labels_list.extend(measured_labels)
            estimated_labels_list.extend(estimated_labels)

            if len(measured_labels_list) >= num_samples:
                break  # Limit number of samples

    # Convert lists to numpy arrays for plotting
    measured_labels_array = np.array(measured_labels_list[:num_samples])
    estimated_labels_array = np.array(estimated_labels_list[:num_samples])

    # Plot measured vs estimated leaf areas
    plot_true_vs_predicted(measured_labels_array, estimated_labels_array)

    # Plot residuals histogram
    residuals = measured_labels_array - estimated_labels_array
    plot_residuals_histogram(residuals)

    # Plot leaf area distribution
    plot_leaf_area_distribution(measured_labels_array, estimated_labels_array)

    # Generate and save model architecture diagram
    generate_model_architecture_diagram(model)

    # Visualize saliency maps
    visualize_saliency_maps(model, dataloader, num_samples=5, view_index=4)

    print("All visualizations have been generated and saved.")

def plot_true_vs_predicted(measured_labels, estimated_labels):
    """
    Plot the true vs predicted leaf area.
    :param measured_labels: True leaf area labels.
    :param estimated_labels: Predicted leaf area labels.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(measured_labels, estimated_labels, color='green', alpha=0.8, edgecolors='w', s=50)

    # Fit a linear regression line
    regressor = LinearRegression()
    regressor.fit(measured_labels.reshape(-1, 1), estimated_labels)
    a = regressor.coef_[0]
    b = regressor.intercept_

    # Plot the best-fit line
    x_vals = np.linspace(min(measured_labels), max(measured_labels), 100)
    y_vals = a * x_vals + b
    plt.plot(x_vals, y_vals, 'r--', lw=2, label=f'Best Fit: y = {a:.2f}x + {b:.2f}')

    # Diagonal line for reference
    plt.plot([min(measured_labels), max(measured_labels)], [min(measured_labels), max(measured_labels)], 'b--', lw=2, label='y = x')
    
    # Add plot title and labels
    plt.title('Measured vs Estimated Leaf Area (cm²)')
    plt.xlabel('Measured Leaf Area (cm²)')
    plt.ylabel('Estimated Leaf Area (cm²)')
    plt.legend()

    # Display RMSE and R² as text
    r2 = r2_score(measured_labels, estimated_labels)
    mae = mean_absolute_error(measured_labels, estimated_labels)
    rmse = np.sqrt(mean_squared_error(measured_labels, estimated_labels))
    plt.text(0.05, 0.95, f'R²: {r2:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}', transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=12, color='blue')
    
    save_path = os.path.join(Config.save_dir, "measured_vs_estimated_leaf_area.png")
    os.makedirs(Config.save_dir, exist_ok=True)  # Ensure the directory exists
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_residuals_histogram(residuals):
    """
    Plot the histogram of residuals.
    :param residuals: Residuals between true and predicted labels.
    """
    plt.figure(figsize=(8, 6))

    # Define bin edges with a width of 2.5
    min_residual = np.floor(residuals.min() / 2.5) * 2.5  # Round down to the nearest multiple of 2.5
    max_residual = np.ceil(residuals.max() / 2.5) * 2.5   # Round up to the nearest multiple of 2.5
    bins = np.arange(min_residual, max_residual + 2.5, 2.5)  # Bin edges from min to max with 2.5 width

    # Plot histogram with specified bins
    plt.hist(residuals, bins=bins, color='blue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Residuals Histogram')
    plt.xlabel('Residuals (Measured - Estimated)')
    plt.ylabel('Frequency')
    save_path = os.path.join(Config.save_dir, "residuals_histogram.png")
    os.makedirs(Config.save_dir, exist_ok=True)  # Ensure the directory exists
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_leaf_area_distribution(measured_labels, estimated_labels):
    """
    Plot the distribution of measured and estimated leaf areas.
    :param measured_labels: True leaf area labels.
    :param estimated_labels: Predicted leaf area labels.
    """
    plt.figure(figsize=(8, 6))
    # Histogram for measured and estimated leaf areas
    # Define bin edges with a width of 5
    min_residual = np.floor(measured_labels.min() / 5) * 5  # Round down to the nearest multiple of 5
    max_residual = np.ceil(measured_labels.max() / 5) * 5   # Round up to the nearest multiple of 5
    bins = np.arange(min_residual, max_residual + 5, 5)  # Bin edges from min to max with 5 width
    plt.hist(measured_labels, bins, alpha=0.5, label='Measured Leaf Area', color='green', edgecolor='black')
    plt.hist(estimated_labels, bins, alpha=0.5, label='Estimated Leaf Area', color='blue', edgecolor='black')
    plt.title('Distribution of Measured and Estimated Leaf Areas')
    plt.xlabel('Leaf Area')
    plt.ylabel('Frequency')
    plt.legend()
    save_path = os.path.join(Config.save_dir, "leaf_area_distribution.png")
    os.makedirs(Config.save_dir, exist_ok=True)  # Ensure the directory exists
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def generate_model_architecture_diagram(model):
    """
    Generate and save the model architecture diagram.
    :param model: PyTorch model.
    """
    # Create a dummy input that matches the input size for your model
    dummy_input = torch.randn(1, 6, 3, 224, 224).cuda()  # Example input size for the late fusion model

    # Generate the computation graph using torchviz
    dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))

    # Save the model architecture diagram
    diagram_path = os.path.join(Config.save_dir, "model_architecture_diagram")
    os.makedirs(Config.save_dir, exist_ok=True)  # Ensure the directory exists
    dot.render(diagram_path, format="png")

def compute_saliency_maps(model, inputs, labels, view_index=0):
    """
    Compute saliency maps for a batch of inputs.

    Args:
        model: The trained model.
        inputs: Input images of shape (batch_size, 6, C, H, W).
        labels: True labels corresponding to the images.
        view_index: Index of the view for which saliency maps are computed (default: 0).

    Returns:
        saliency_maps: A list of saliency maps for each sample.
    """
    model.eval()
    inputs = inputs.cuda()
    labels = labels.cuda()

    # Require gradient for the inputs
    inputs.requires_grad_()

    # Forward pass
    outputs = model(inputs)
    loss = outputs.squeeze().mean()

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Get the gradients w.r.t. the inputs
    saliency = inputs.grad.abs().detach().cpu()  # Shape: (batch_size, 6, C, H, W)
    saliency_maps = saliency[:, view_index, :, :, :]  # Select the specified view

    # Convert saliency maps to numpy arrays and aggregate the channels
    saliency_maps = saliency_maps.permute(0, 2, 3, 1).numpy()  # Shape: (batch_size, H, W, C)
    saliency_maps = np.max(saliency_maps, axis=3)  # Aggregate across channels
    return saliency_maps

def denormalize_image(tensor):
    """
    Denormalize an image tensor for visualization.
    """
    mean = np.array([-0.0437, -0.0380, -0.1277])
    std = np.array([0.2313, 0.2045, 0.0707])

    # Convert from tensor to numpy, and apply denormalization
    img = tensor.permute(1, 2, 0).numpy()  # C, H, W -> H, W, C
    img = img * std + mean
    img = np.clip(img, 0, 1)  # Ensure the pixel values are in [0, 1]
    return img

def plot_saliency_maps(images, saliency_maps, true_labels, predicted_labels):
    """
    Plot saliency maps alongside input images.

    Args:
        images: Input images (batch_size, C, H, W).
        saliency_maps: Computed saliency maps (batch_size, H, W).
        true_labels: True labels corresponding to the images.
        predicted_labels: Predicted labels.
        output_file: File name to save the plot.
    """
    num_samples = len(images)
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))

    for i in range(num_samples):
        # Denormalize the original image
        img = denormalize_image(images[i])

        # Display the original image
        axs[i, 0].imshow(img)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f"True: {true_labels[i]:.2f}, Pred: {predicted_labels[i]:.2f}")

        # Saliency map
        saliency = saliency_maps[i]
        axs[i, 1].imshow(saliency, cmap='hot')
        axs[i, 1].axis('off')
        axs[i, 1].set_title("Saliency Map")

        # Overlay
        overlay = cm.get_cmap('hot')(saliency / saliency.max())[:, :, :3]
        axs[i, 2].imshow(img, alpha=0.5)
        axs[i, 2].imshow(overlay, alpha=0.6)
        axs[i, 2].axis('off')
        axs[i, 2].set_title("Overlay of Saliency")

    plt.tight_layout()
    # Save plot dynamically to the save directory
    save_path = os.path.join(Config.save_dir, "saliency_maps.png")
    os.makedirs(Config.save_dir, exist_ok=True)  # Ensure the directory exists
    plt.savefig(save_path)
    plt.show()

def visualize_saliency_maps(model, dataloader, num_samples=5, view_index=4):
    """
    Visualize predictions with saliency maps.

    Args:
        model: The trained model.
        dataloaders: Dictionary containing train, val, and test dataloaders.
        num_samples: Number of samples to visualize.
        view_index: Index of the view to compute saliency for.
    """
    model.eval()
    images_list, true_labels_list, predicted_labels_list = [], [], []
    saliency_maps = []

    sample_count = 0

    for inputs, true_labels in dataloader['test']:
        batch_size = inputs.size(0)
        outputs = model(inputs.cuda())
        predicted_labels = outputs.squeeze().detach().cpu().numpy()

        saliency = compute_saliency_maps(model, inputs, true_labels, view_index=view_index)
        input_images = inputs[:, view_index, :, :, :]

        images_list.extend(input_images)
        saliency_maps.extend(saliency)
        true_labels_list.extend(true_labels.numpy())
        predicted_labels_list.extend(predicted_labels)

        sample_count += batch_size
        if sample_count >= num_samples:
            break

    images_list = images_list[:num_samples]
    saliency_maps = saliency_maps[:num_samples]
    true_labels_list = true_labels_list[:num_samples]
    predicted_labels_list = predicted_labels_list[:num_samples]

    plot_saliency_maps(images_list, saliency_maps, true_labels_list, predicted_labels_list)