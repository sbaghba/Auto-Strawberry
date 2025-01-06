# Auto-Strawberry

Repository for Automated Attribute Estimation of Strawberry Daughter Plants Using Stereo Vision and Deep Learning.

## Project Overview
This project leverages stereo vision and deep learning techniques to estimate various growth attributes of strawberry daughter plants from multi-view images. The system combines ResNet and Vision Transformer (ViT) models to predict features like:
- Total Leaf Area
- Fresh Mass
- Dry Mass
- Largest Petiole Length
- Average Crown Diameter

The dataset includes images captured using a stereo camera setup, with six images per sample. The models employ a late fusion technique for attribute estimation.

---

## Project Structure
```plaintext
├── datasets/        # Contains data preprocessing and storage
├── models/          # PyTorch models including ResNet, ViT, and Late Fusion models
├── utils/           # Utility scripts for saliency maps, visualization, and evaluation
├── visualizations/  # Stores generated plots and model outputs
├── config.py        # Configuration file with hyperparameters and paths
├── main.py          # Entry point for training, evaluation, and visualization
├── requirements.txt # Python dependencies
├── LICENSE          # License information
└── README.md        # Project documentation
```

---

## Getting Started

### Prerequisites
Ensure you have Python 3.11.9 installed and the required packages:
```bash
pip install -r requirements.txt
```

### Dataset
The dataset used for this project consists of multi-view images captured from two cameras (`cam0`, `cam1`). The annotations are stored in a JSON file specified in `config.py`.

---

## Usage

### Training a Model
```bash
python main.py --mode train
```

### Evaluating a Model
```bash
python main.py --mode eval --weights path/to/weights.pth
```

### Visualizing Predictions and Saliency Maps
```bash
python main.py --mode visualize --weights path/to/weights.pth
```

---

## Configuration
Modify `config.py` to set parameters like:
- `batch_size`
- `learning_rate`
- `num_epochs`
- `model_name` (`resnet34`, `vit_b_16`, `efficientnet_b0`)
- `save_dir`

---

## Results and Visualization
The `visualizations/` folder stores:
- Saliency Maps
- True vs. Predicted Scatter Plots
- Residual Histograms
- Bland-Altman Plots

---

## Models Available
- **ResNet34**
- **Vision Transformer (ViT-B-16)**
- **EfficientNet B0**
- **Late Fusion Model** (for combining multi-view images)

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing
Contributions are welcome! Please create an issue or submit a pull request if you'd like to improve the project.

---

## Contact
For inquiries, reach out to Sina Baghban via [email](mailto:your_email@example.com).



## Citing This Work
If you use this code in your research, please cite:

@inproceedings{YourPaper2024, title={Your Paper Title}, author={Your Name and Co-Authors}, booktitle={Proceedings of CVPR}, year={2024} }
