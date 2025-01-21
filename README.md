# Auto-Strawberry

Repository for Automated Attribute Estimation of Strawberry Daughter Plants Using Stereo Vision and Deep Learning.

## Project Overview
This project leverages stereo vision and deep learning techniques to estimate various growth attributes of strawberry daughter plants from multi-view images. The system uses modified ResNet, EfficientNet, and Vision Transformer (ViT) backbones to predict features like:
- Total Leaf Area
- Fresh Mass
- Largest Petiole Length
- Average Crown Diameter

The dataset includes images captured using a stereo camera setup, with six images per sample. The models employ a late fusion technique for attribute estimation.

---

## Project Structure
```plaintext
├── datasets/        # Contains dataset and annotations
├── models/          # PyTorch models including Late Fusion
├── utils/           # Utility code for training, visualization, and evaluation.
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

### Dataset and Pre-trained Weights
Please download the dataset and pre-trained weights from the official [Hugging Face repository](https://huggingface.co/sinabjam/Auto-Strawberry). Update the paths in `config.py` accordingly.
The dataset used for this project consists of multi-view images captured from two cameras (`cam0`, `cam1`). The annotations are stored in a JSON file specified in `config.py`.

---

## Usage

### Capturing Images on Raspberry PI 5
If interested in capturing images from PI 5, run `PI5_LeafDetection.py` in `utils/` folder. The code captures three images from each of the two cameras as they pass through the enclosure. Leaves are detected based on their color.

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

---

## Backbone Models Available
- **ResNet34**
- **Vision Transformer (ViT-B-16)**
- **EfficientNet B0**

---

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

---

## Contributing
Contributions are welcome! Please create an issue or submit a pull request if you'd like to improve the project.

---

## Contact
For inquiries, reach out to Sina Baghbanijam via [email](mailto:sbaghba@ncsu.edu).

## Citation
The paper related to the repository is under review and will be linked after acceptance.
