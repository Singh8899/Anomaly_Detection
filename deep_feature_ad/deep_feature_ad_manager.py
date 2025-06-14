"""
@author: Carlo Merola
"""
import os
import sys
import yaml
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, classification_report

from deep_feature_anomaly_detector import DeepFeatureAnomalyDetector

# changing parent directory to import MVTecAD2 dataset
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
original_cwd = os.getcwd()
os.chdir(parent_dir)  # Change to parent directory to import MVTecAD2
from dataset_preprocesser import MVTecAD2
os.chdir(original_cwd)  # Change back to original working directory

class DeepFeatureADManager:
    """
    Manager for Deep Feature-based Anomaly Detection.
    Handles computation of threshold based on data std, training, validation, and evaluation of the model.
    """
    def __init__(self, product_class, config_path, train_path, test_path, threshold_computation_mode='all'):
        """
        threshold_computation_mode: options: standard 3sigma, aggressive 1sigma, conservative 5sigma, all
        all: computes all three thresholds and saves them in the model config
        """
        self.config_path = config_path
        self.train_path = train_path
        self.test_path = test_path
        self.product_class = product_class
        self.threshold_computation_mode = threshold_computation_mode # options: standard 3sigma, aggressive 1sigma, conservative 5sigma
        # Load configuration
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        # Use base_autoencoder config as template (can be customized)
        self.model_config = self.config['MODELS_CONFIG']['DeepFeatureAE']
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        print(f"USING DEVICE: {self.device}")
        
        # Initialize model and move to device
        self.model = DeepFeatureAnomalyDetector(layer_hooks=['layer2', 'layer3'], latent_dim=100, smooth=True, is_bn=True)
        
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.autoencoder.parameters(), 
            lr=float(self.model_config['learning_rate'])
        )
        
        # transform for input images
        self.transform = transforms.Compose([
            transforms.Resize((self.model_config['input_size'], self.model_config['input_size'])),
            transforms.ToTensor(),
            #  Normalize the input images to match pre-trained model expectations on ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize dataset (only normal samples for training)
        self.train_dataset = MVTecAD2(
            self.product_class, 
            "train", 
            self.train_path, 
            transform=self.transform
        )
        self.test_dataset = MVTecAD2(
            self.product_class, 
            "test", 
            self.test_path, 
            transform=self.transform
        )
            
        # Threshold for anomaly detection
        self.thresholds = None
        
    def compute_threshold(self):
        """
        Compute the anomaly detection threshold based on the training data and reconstruction error/anomaly scores.
        Need to train the model first to compute the thresholds. 
        In an untrained model, we will have higher reconstruction errors -> higher anomaly scores and thus higher thresholds.
        """
        mode = self.threshold_computation_mode
        if mode == "standard":
            sigma_multiplier = 3.0
        elif mode == "aggressive":
            sigma_multiplier = 1.0
        elif mode == "conservative":
            sigma_multiplier = 5.0
        elif mode == "all":
            sigma_multiplier = [3.0, 1.0, 5.0]
        else:
            raise ValueError(f"Invalid threshold computation mode: {mode}. Must be 'standard', 'aggressive', 'conservative', or 'all'")
        
        
        self.model.eval()
        train_loader = DataLoader(self.train_dataset, batch_size=self.model_config['batch_size'], shuffle=False)

        anomaly_scores = []
        maps = []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Computing thresholds"):
                images = batch["sample"].to(self.device)

                # Forward pass through the autoencoder from deep_feature_autoencoder_model imported into DeepFeatureAnomalyDetector
                features, reconstructed = self.model(images)

                # the threshold is computed based on the reconstruction error and anomaly scores
                # Compute reconstruction error and error map
                error_map = self.model.compute_reconstruction_error(features, reconstructed)    # error_map saved as a class attribute
                scores = self.model.compute_anomaly_score(error_map, k=10)

                maps.append(error_map.cpu().numpy())  # Collect error maps and move to CPU
                anomaly_scores.extend(scores.cpu().numpy()) # Collect anomaly scores and move to CPU

        # compute statistics for anomaly scores and set the threshold
        anomaly_scores = np.array(anomaly_scores)
        mean_error = np.mean(anomaly_scores)
        std_error = np.std(anomaly_scores)
        print(f"Mean Anomaly Score: {mean_error}, Std Anomaly Score: {std_error}")
        
        # if sigma_multiplier is a list, compute thresholds for each multiplier
        if isinstance(sigma_multiplier, list):
            # If multiple sigma multipliers are provided, compute thresholds for each
            thresholds = [mean_error + m * std_error for m in sigma_multiplier]
            print(f"Computed thresholds: {thresholds}")
        else:
            # Single sigma multiplier
            thresholds = mean_error + sigma_multiplier * std_error
            print(f"Computed threshold: {mean_error + sigma_multiplier * std_error}")
            
        self.thresholds = thresholds
    
    def save_thresholds_for_class(self):
        """
        Save computed thresholds to a YAML file for the specific product class.
        """
        threshold_file = os.path.join(self.train_path, f"{self.product_class}_thresholds.yaml")
        
        # Convert NumPy types to Python native types for YAML serialization
        if isinstance(self.thresholds, list):
            # Convert each np.float32 to Python float
            thresholds = [float(threshold) for threshold in self.thresholds]
        else:
            # Single threshold case
            thresholds = float(self.thresholds)
        
        with open(threshold_file, 'w') as file:
            yaml.safe_dump({'thresholds': thresholds}, file)
        
        print(f"Thresholds saved to: {threshold_file}")
            
            
            
if __name__ == "__main__":
    product_class = "hazelnut"  # Example product class

    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(parent, "config.yaml")

    experiment_name = f"deep_feature_{product_class}"
    experiment_dir = os.path.join("output", experiment_name)
    
    train_path = os.path.join(experiment_dir, "train")
    test_path = os.path.join(experiment_dir, "test")
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Initialize the manager
    manager = DeepFeatureADManager(
                        product_class=product_class,
                        config_path=config_path,
                        train_path=train_path,
                        test_path=test_path,
                        threshold_computation_mode='all' 
                    )
    # Ensure the model is in training mode
    # Compute and save thresholds
    manager.compute_threshold()
    manager.save_thresholds_for_class()