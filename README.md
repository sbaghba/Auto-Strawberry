Auto-Strawberry

Automated Attribute Estimation of Strawberry Daughter Plants Using Stereo Vision and Deep Learning

Project Overview

Auto-Strawberry is a deep learning pipeline designed for automated estimation of multiple attributes of strawberry daughter plants. It utilizes stereo vision images and deep learning techniques, including late fusion models and saliency maps, to estimate attributes such as leaf area and plant mass.

Repository Structure

datasets/: Contains the strawberry plant datasets used for training and testing.

models/: Contains PyTorch model definitions, including late fusion models for ResNet and ViT.

utils/: Utility functions for data loading, training, evaluation, and visualization.

visualizations/: Stores generated visual outputs including saliency maps and model architecture diagrams.

main.py: Main script for training, evaluating, and visualizing the models.

config.py: Centralized configuration file for hyperparameters and paths.

requirements.txt: List of required Python packages.

LICENSE: Licensing information.

README.md: This file.

Getting Started

Prerequisites

Ensure you have Python 3.11.9 installed and all dependencies listed in requirements.txt.

Installation

# Clone the repository
git clone https://github.com/yourusername/auto-strawberry.git
cd auto-strawberry

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate

# Install required packages
pip install -r requirements.txt

Dataset

Ensure the dataset is available in the datasets folder with the correct file structure defined in the config.py file.

Training a Model

To train a model, run:

python main.py --mode train

Evaluating a Model

python main.py --mode eval --weights <path_to_model_weights>

Visualizing Saliency Maps

python main.py --mode visualize --weights <path_to_model_weights>

Configuration

Key parameters and paths are defined in the config.py file, including:

data_dir: Path to the dataset folder.

save_dir: Path for saving visualizations.

batch_size, num_epochs, and other training parameters.

Results and Visualizations

All generated visualizations such as saliency maps, leaf area distributions, and residual histograms are stored in the visualizations/ directory by default.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For any questions, feel free to reach out at your.email@example.com.

## Citing This Work
If you use this code in your research, please cite:

@inproceedings{YourPaper2024, title={Your Paper Title}, author={Your Name and Co-Authors}, booktitle={Proceedings of CVPR}, year={2024} }
