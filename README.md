# Industrial Anomaly Detection Framework

A comprehensive deep learning framework for industrial anomaly detection and defect segmentation, implementing multiple state-of-the-art approaches including PatchCore, Vision Transformers, and Deep Feature-based autoencoders.

## 🚀 Overview

This framework provides end-to-end solutions for detecting anomalies in industrial images using various deep learning architectures. It supports both image-level anomaly classification and pixel-level defect segmentation, making it suitable for quality control applications in manufacturing.

## 🎯 Key Features

- **Multiple Model Architectures**: 
  - **PatchCore**: Memory bank approach using pre-trained ResNet50 features
  - **Vision Transformer (ViT)**: Transformer-based autoencoder with attention mechanisms
  - **Deep Feature Autoencoder**: Multi-layer feature extraction with ResNet50 backbone
  - **Base Autoencoder**: Simple CNN-based reconstruction model
  - **Transformer Autoencoder**: Hybrid CNN-Transformer architecture

- **Comprehensive Evaluation**:
  - ROC AUC scoring for anomaly classification
  - PRO (Per-Region-Overlap) scoring for segmentation quality
  - Precision-Recall curves and F1-score optimization
  - Adaptive threshold computation (standard/aggressive/conservative modes)

- **Advanced Training Features**:
  - Early stopping with validation monitoring
  - Learning rate scheduling (Cosine Annealing)
  - Multi-class and foundational model training
  - TensorBoard integration for training visualization

- **Production-Ready Components**:
  - Segmentation map generation for defect localization
  - Configurable threshold strategies
  - Model checkpointing and weight management
  - Comprehensive testing and evaluation pipelines

## 📊 Supported Datasets

Designed for the **MVTec Anomaly Detection Dataset** with support for:

### Product Categories
- `bottle`, `cable`, `capsule`, `carpet`, `fabric`, `grid`, `hazelnut`
- `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`
- `transistor`, `wood`, `zipper` and more

### Foundational Classes
Special support for foundational object classes: `carpet`, `hazelnut`, `leather`, `wood`

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.9+

### Dependencies
```bash
# Core ML libraries
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
scikit-learn>=1.0.0

# Data processing and visualization
PIL (Pillow)>=8.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.62.0

# Configuration and utilities
PyYAML>=5.4.0
einops>=0.4.0
tensorboard>=2.7.0
torchinfo>=1.6.0

# Dataset handling
kagglehub  # For MVTec dataset download
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Singh8899/Anomaly_Detection.git
cd Anomaly_Detection
```

2. Install dependencies:
```bash
pip install torch torchvision numpy scikit-learn pillow matplotlib seaborn tqdm pyyaml einops tensorboard torchinfo kagglehub
```

3. Configure dataset path in `config.yaml`:
```yaml
DATASET_PATH: "/path/to/mvtec-ad-dataset"
```

## 🚦 Quick Start

### Training a Model

#### Basic Autoencoder
```bash
python train_test.py --model_name base --product_class hazelnut --mode train --train_path trains/base_hazelnut
```

#### Vision Transformer
```bash
python train_test.py --model_name vit --product_class carpet --mode train --train_path trains/vit_carpet
```

#### Deep Feature Autoencoder
```bash
python deep_feature_ad/deep_feature_ad_trainer.py --product_class leather
```

#### PatchCore
```bash
python train_test.py --model_name patchcore --product_class wood --mode train --train_path trains/patchcore_wood
```

### Testing a Model

```bash
python train_test.py --model_name vit --product_class carpet --mode test --train_path trains/vit_carpet --test_path tests/vit_carpet
```

### Foundational Model Training
Train across multiple object classes:
```bash
python deep_feature_ad/deep_feature_ad_trainer.py --product_class foundational
```

## 📁 Project Structure

```
├── config.yaml                    # Main configuration file
├── train_test.py                  # Unified training/testing interface
├── dataset_preprocesser.py        # MVTec dataset handling
│
├── base_model/                    # Simple CNN autoencoder
│   └── base_autoencoder.py
│
├── vit_model/                     # Vision Transformer implementation
│   ├── ViT.py                    # Main ViT manager and architecture
│   ├── student_transformer.py    # Transformer components
│   ├── model_res18.py            # Decoder components
│   ├── mdn1.py                   # Mixture Density Network
│   ├── spatial.py                # Spatial processing utilities
│   └── utility_fun.py            # Helper functions
│
├── deep_feature_ad/              # Deep feature anomaly detection
│   ├── deep_feature_ad_trainer.py      # Training pipeline
│   ├── deep_feature_ad_tester.py       # Testing pipeline
│   ├── deep_feature_ad_manager.py      # Threshold computation
│   ├── deep_feature_anomaly_detector.py # Core detector
│   └── deep_feature_autoencoder_model.py # Model architecture
│
├── patchcore/                    # PatchCore implementation
│   ├── patchcore_class.py       # Main PatchCore manager
│   └── patchcore.py             # Legacy implementation
│
├── trafo_model/                  # Transformer autoencoder
│   ├── trafo_autoencoder.py     # Hybrid CNN-Transformer model
│   └── transformers_custom.py   # Custom transformer layers
│
├── trains/                       # Training outputs and checkpoints
└── masked_anomaly/               # Additional model variants
```

## ⚙️ Configuration

The framework uses `config.yaml` for centralized configuration:

```yaml
DATASET_PATH: "/path/to/mvtec-ad"

MODELS_CONFIG:
  base_autoencoder:
    batch_size: 16
    num_epochs: 30
    learning_rate: 1e-3
    
  vit_autoencoder:
    batch_size: 8
    num_epochs: 300
    learning_rate: 1e-4
    
  DeepFeatureAE:
    batch_size: 64
    layer_hooks: ['layer2', 'layer3']
    latent_dim: 100
    threshold_computation_mode: 'all'
    
  patchcore:
    memory_bank_subsample_ratio: 0.1
    threshold_multiplier: 2.0
    num_neighbors: 9
```

## 🎯 Model Architectures

### 1. PatchCore
- **Approach**: Memory bank of normal feature patches
- **Backbone**: Pre-trained ResNet50 (layers 2-3)
- **Detection**: k-NN distance in feature space
- **Strengths**: High accuracy, interpretable results

### 2. Vision Transformer (ViT-AE)
- **Approach**: Transformer-based autoencoder with patch attention
- **Components**: ViT encoder + CNN decoder + Mixture Density Network
- **Loss**: MSE + SSIM + Gaussian likelihood
- **Strengths**: Captures long-range dependencies

### 3. Deep Feature Autoencoder
- **Approach**: Multi-scale feature reconstruction
- **Backbone**: ResNet50 with multiple layer hooks
- **Detection**: Top-k reconstruction error averaging
- **Strengths**: Multi-resolution anomaly detection

### 4. Base Autoencoder
- **Approach**: Simple CNN reconstruction
- **Architecture**: Encoder-decoder with skip connections
- **Detection**: Pixel-wise MSE thresholding
- **Strengths**: Fast training, baseline performance

## 📈 Performance Metrics

The framework evaluates models using:

- **ROC AUC**: Image-level anomaly classification
- **PRO Score**: Pixel-level segmentation quality
- **PR AUC**: Precision-Recall area under curve
- **Accuracy**: Binary classification with optimized thresholds

### Threshold Strategies
- **Standard** (μ + 3σ): Balanced performance
- **Aggressive** (μ + 1σ): High sensitivity
- **Conservative** (μ + 5σ): Low false positive rate

## 🔧 Advanced Usage

### Custom Model Training
```python
from deep_feature_ad.deep_feature_ad_trainer import DeepFeatureADTrainer

trainer = DeepFeatureADTrainer(
    config_path="config.yaml",
    train_path="output/train",
    test_path="output/test",
    product_class="hazelnut"
)
trainer.train_single_class()
trainer.compute_threshold()
```

### Segmentation Map Generation
```python
trainer.generate_segmentation_maps(num_examples=10)
trainer.plot_anomalies_thresholds()
```

### Multi-Class Evaluation
```python
# Test across all foundational classes
python deep_feature_ad/deep_feature_ad_tester.py --product_class foundational
```

## 📊 Results and Outputs

### Training Outputs
- Model weights (`.pth` files)
- Training statistics (`training_statistics.yaml`)
- TensorBoard logs
- Training error histograms

### Testing Outputs
- ROC curves and score distributions
- Segmentation maps overlaid on original images
- Performance metrics summaries
- Per-class evaluation results

## 🤝 Contributing

This project was developed by **Jaspinder Singh** and contributors. The framework is designed to be extensible for additional anomaly detection approaches.

### Adding New Models
1. Create a new model directory under the project root
2. Implement manager class with `train()` and `test()` methods
3. Add model configuration to `config.yaml`
4. Update `train_test.py` with new model option

## 📜 License

This project is available under standard academic/research use terms.

## 🔗 References

- **MVTec AD Dataset**: [Industrial Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **PatchCore**: "Towards Total Recall in Industrial Anomaly Detection"
- **Vision Transformers**: "An Image is Worth 16x16 Words"

---

**Note**: This framework requires the MVTec Anomaly Detection dataset. Please ensure you have the proper licensing and dataset access before use.