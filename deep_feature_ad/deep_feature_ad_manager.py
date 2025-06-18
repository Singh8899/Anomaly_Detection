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
    Handles computation of threshold based on data (product) std, computation of statistics based on anomaly scores, and generation of segmentation maps. 
    """
    def __init__(self, product_class, config_path, train_path, test_path, threshold_computation_mode='all', model=None):
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
        if model is not None:
            # If a pre-trained model is provided, use it
            self.model = model
            print("Using provided pre-trained model.")
            
        else:
            self.model = DeepFeatureAnomalyDetector(layer_hooks=self.model_config['layer_hooks'], 
                                                    latent_dim=self.model_config['latent_dim'], 
                                                    smooth=self.model_config['smooth'], 
                                                    is_bn=self.model_config['is_bn']
                                                )

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
        self.sigma_multiplier = None
        self.mean_error = None
        self.std_error = None
        self.error_map = None
        self.anomaly_scores = None
        
    def load_model_weights(self, weight_path):
        """
        Load model weights from the specified path.
        """
        if weight_path:
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            print(f"Model weights loaded from {weight_path}")
        else:
            print("No model weights provided. Using untrained model.")
        
    def compute_threshold(self):
        """
        Compute the anomaly detection threshold based on the training data and reconstruction error/anomaly scores.
        Need to train the model first to compute the thresholds. 
        In an untrained model, we will have higher reconstruction errors -> higher anomaly scores and thus higher thresholds.
        """
        mode = self.threshold_computation_mode
        if mode == "standard":
            self.sigma_multiplier = 3.0
        elif mode == "aggressive":
            self.sigma_multiplier = 1.0
        elif mode == "conservative":
            self.sigma_multiplier = 5.0
        elif mode == "all":
            self.sigma_multiplier = [3.0, 1.0, 5.0]
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
        self.anomaly_scores = np.array(anomaly_scores)
        self.mean_error = np.mean(self.anomaly_scores)
        self.std_error = np.std(self.anomaly_scores)
        print(f"Mean Anomaly Score: {self.mean_error}, Std Anomaly Score: {self.std_error}")

        # if sigma_multiplier is a list, compute thresholds for each multiplier
        if isinstance(self.sigma_multiplier, list):
            # If multiple sigma multipliers are provided, compute thresholds for each
            thresholds = [self.mean_error + m * self.std_error for m in self.sigma_multiplier]
            print(f"Computed thresholds: {thresholds}")
        else:
            # Single sigma multiplier
            thresholds = self.mean_error + self.sigma_multiplier * self.std_error
            print(f"Computed threshold: {self.mean_error + self.sigma_multiplier * self.std_error}")
            
        self.thresholds = thresholds

        # return self.thresholds

    def save_thresholds_for_class(self, foundational=False):
        """
        Save computed thresholds to a YAML file for the specific product class.
        """
        if foundational:
            # If foundational, save in a different path
            threshold_file = os.path.join(self.train_path, f"{self.product_class}_foundational_thresholds.yaml")
        else:
            # Save in the regular train path
            threshold_file = os.path.join(self.train_path, f"{self.product_class}_thresholds.yaml")
        
        # Save threshold info
        threshold_info = {
            'mode': self.threshold_computation_mode,
            'sigma_multiplier': self.sigma_multiplier,
            'mean_error': float(self.mean_error),
            'std_error': float(self.std_error),
            'thresholds': None,
            'num_samples': len(self.anomaly_scores)
        }
        
        # Convert NumPy types to Python native types for YAML serialization
        if isinstance(self.thresholds, list):
            # Convert each np.float32 to Python float
            thresholds = [float(threshold) for threshold in self.thresholds]
        else:
            # Single threshold case
            thresholds = float(self.thresholds)
        
        # Update the threshold info with the computed thresholds converted to native types
        threshold_info['thresholds'] = thresholds
        
        with open(threshold_file, 'w') as file:
            yaml.safe_dump({'thresholds': threshold_info}, file)
        
        print(f"Thresholds saved to: {threshold_file}")
        
    def load_thresholds_for_class(self, threshold_file=None):
        """
        Load thresholds from a YAML file for the specific product class.
        """
        with open(threshold_file, 'r') as file:
            threshold_info = yaml.safe_load(file)['thresholds']
        
        self.threshold_computation_mode = threshold_info['mode']
        self.sigma_multiplier = threshold_info['sigma_multiplier']
        self.mean_error = threshold_info['mean_error']
        self.std_error = threshold_info['std_error']
        self.thresholds = threshold_info['thresholds']
        
        print(f"Loaded thresholds: {self.thresholds}")

    def plot_anomalies_thresholds(self):
        """
        Plot the computed thresholds.
        """
        if not isinstance(self.thresholds, list):
            self.thresholds = [self.thresholds]
        if not isinstance(self.sigma_multiplier, list):
            self.sigma_multiplier = [self.sigma_multiplier]
        
        # getting subplots for each threshold. Number of subplots is equal to the number of thresholds
        fig, axes = plt.subplots(nrows=len(self.thresholds), ncols=1, figsize=(10, 6 * len(self.thresholds)))
        
        for i, threshold in enumerate(self.thresholds):
            ax = axes[i]
            ax.hist(self.anomaly_scores, bins=50, density=True, alpha=0.7, color='blue', label='Training Errors')
            ax.axvline(self.mean_error, color='green', linestyle='--', label=f'Mean (μ): {self.mean_error:.4f}')
            ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold (μ + {self.sigma_multiplier[i]}σ): {threshold:.4f}')
            ax.set_title(f'Histogram of Training Reconstruction Errors ({self.threshold_computation_mode.upper()} Mode)')
            ax.set_xlabel('Reconstruction Error')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = os.path.join(self.train_path, f"training_errors_histogram_{self.threshold_computation_mode}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Histogram saved to: {save_path}")
        
    
    def generate_segmentation_maps(self, num_examples=5, foundational=False):
        """
        Generate a segmentation map for the given image using the trained model.
        """
        self.model.eval()
        seg_output_dir = os.path.join(self.test_path, "segmentation_maps")
        if foundational:
            seg_output_dir = os.path.join(seg_output_dir, self.product_class)
        os.makedirs(seg_output_dir, exist_ok=True)
        
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_examples:
                    break
                
                images = batch["sample"].to(self.device)
                #image_name = batch["name"][0]
                image_path = batch["image_path"][0]
                
                # only anomalous images are used for segmentation maps
                if 'good' in image_path:
                    continue
        
                # Forward pass through the autoencoder
                features, reconstructed = self.model(images)
                
                # Compute error map
                error_map = self.model.compute_reconstruction_error(features, reconstructed)
                seg_map = self.model.get_segmentation_map(error_map=error_map, target_size=(self.model_config['input_size'], self.model_config['input_size']))
                anomaly_scores = self.model.compute_anomaly_score(error_map, k=10)

                thresholds = self.thresholds if isinstance(self.thresholds, list) else [self.thresholds]
                
                original_image = images[0].cpu().numpy().transpose(1, 2, 0)
                
                # denormalize the image
                original_image = (original_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                original_image = np.clip(original_image, 0, 1)  # Ensure pixel values are in [0, 1]
                
                seg_map = seg_map[0].cpu().numpy()  
                error_map = error_map[0].cpu().numpy()
                
                # save maps
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                
                axes[0, 0].imshow(original_image)
                axes[0, 0].set_title("Original Image")
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(seg_map, cmap='jet')
                axes[0, 1].set_title("Segmentation Map")
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(original_image)
                axes[0, 2].imshow(seg_map, cmap='jet', alpha=0.5)
                axes[0, 2].set_title("Overlay Segmentation Map")
                axes[0, 2].axis('off')
                
                # save mask for each threshold
                for j, threshold in enumerate(thresholds):
                    axes[1, j].imshow(original_image)
                    axes[1, j].imshow(seg_map>threshold, cmap='gray')
                    axes[1, j].set_title(f"Mask - Threshold: {threshold:.4f}")
                    axes[1, j].axis('off')

                fig.tight_layout()

                # save figures
                fig.savefig(os.path.join(seg_output_dir, f"segmentation_map_{i}.png"))
                plt.close(fig)
                
                print(f"Segmentation map saved for image {i+1} at {os.path.join(seg_output_dir, f'segmentation_map_{i}.png')}")
                
                

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
    manager.plot_anomalies_thresholds()
    manager.generate_segmentation_maps(num_examples=5)