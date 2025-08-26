"""
@author: Carlo Merola
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from deep_feature_ad_manager import DeepFeatureADManager
from deep_feature_anomaly_detector import DeepFeatureAnomalyDetector
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# changing parent directory to import MVTecAD2 dataset
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
original_cwd = os.getcwd()
os.chdir(parent_dir)  # Change to parent directory to import MVTecAD2
from dataset_preprocesser import MVTecAD2

os.chdir(original_cwd)  # Change back to original working directory


class DeepFeatureADTester:
    def __init__(self, config_path, product_class):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.product_class = product_class
        
        # Initialize detector
        self.detector = DeepFeatureAnomalyDetector(
            layer_hooks=self.config['MODELS_CONFIG']['DeepFeatureAE']['layer_hooks'],
            latent_dim=self.config['MODELS_CONFIG']['DeepFeatureAE']['latent_dim'],
            smooth=self.config['MODELS_CONFIG']['DeepFeatureAE']['smooth'],
            is_bn=self.config['MODELS_CONFIG']['DeepFeatureAE']['is_bn']
        ).to(self.device)
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((self.config['MODELS_CONFIG']['DeepFeatureAE']['input_size'], 
                             self.config['MODELS_CONFIG']['DeepFeatureAE']['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_dataset = MVTecAD2(
            self.product_class,
            "test",
            self.config['DATASET_PATH'],
            transform=self.transform
        )

    def load_config(self, path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def test(self, weights_path, threshold_path=None, save_plots=True):
        """
        Test the model on the test dataset and compute ROC AUC.
        """
        # Load model weights
        if not os.path.exists(weights_path):
            print(f"Model weights not found at {weights_path}")
            return
        
        # get the model weights from the training path
        self.detector.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.detector.eval()
        
        # Load threshold if provided
        thresholds = None
        if threshold_path and os.path.exists(threshold_path):
            with open(threshold_path, "r") as file:
                threshold_info = yaml.safe_load(file)
                if 'thresholds' in threshold_info:
                    threshold_data = threshold_info['thresholds']
                    thresholds = threshold_data['thresholds']
                    # Ensure thresholds is always a list for consistent handling
                    if not isinstance(thresholds, list):
                        thresholds = [thresholds]
                    print(f"Using thresholds: {thresholds}")
        
        print("=== Testing Deep Feature Autoencoder ===")
        
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.config['MODELS_CONFIG']['DeepFeatureAE']['batch_size'], 
            shuffle=False
        )
        
        test_scores = []
        test_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                images = batch['sample'].to(self.device)
                image_paths = batch['image_path']
                
                # Forward pass
                features, reconstructed = self.detector(images)
                
                # Compute error map and anomaly scores
                error_map = self.detector.compute_reconstruction_error(features, reconstructed)
                scores = self.detector.compute_anomaly_score(error_map, k=10)
                
                # Generate labels from image paths (good=0, bad=1)
                labels = [0 if 'good' in path else 1 for path in image_paths]
                
                test_scores.extend(scores.cpu().numpy())
                test_labels.extend(labels)
        
        # Compute ROC AUC
        test_scores = np.array(test_scores)
        test_labels = np.array(test_labels)
        
        roc_auc = roc_auc_score(test_labels, test_scores)
        print(f"ROC AUC on test set: {roc_auc:.4f}")
        
        # Compute accuracy if thresholds are available
        accuracies = []
        if thresholds is not None:
            for i, threshold in enumerate(thresholds):
                predictions = (test_scores > threshold).astype(int)
                accuracy = np.mean(predictions == test_labels)
                accuracies.append(accuracy)
                print(f"Accuracy on test set (threshold {i+1}): {accuracy:.4f}")
        
        if save_plots:
            self._save_plots(test_labels, test_scores, roc_auc, thresholds)
        
        # Save test results
        test_results = {
            'roc_auc': float(roc_auc),
            'accuracies': [float(acc) for acc in accuracies] if thresholds is not None else None,
            'thresholds': [float(t) for t in thresholds] if thresholds is not None else None,
            'num_test_samples': len(test_scores),
            'num_normal': int(np.sum(test_labels == 0)),
            'num_anomalous': int(np.sum(test_labels == 1))
        }
        
        return roc_auc, test_results
    
    def _save_plots(self, test_labels, test_scores, roc_auc, thresholds):
        """
        Save ROC curve and score distribution plots.
        """
        # Create output directory for test plots
        test_plots_dir = f"output/deep_feature_{self.product_class}/test"
        os.makedirs(test_plots_dir, exist_ok=True)
        
        # Create and save ROC curve plot
        from sklearn.metrics import roc_curve
        fpr, tpr, roc_thresholds = roc_curve(test_labels, test_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.product_class.capitalize()} (AUC = {roc_auc:.4f})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        roc_save_path = os.path.join(test_plots_dir, "roc_curve.png")
        plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to: {roc_save_path}")
        
        # Create and save score distribution plot with all thresholds
        plt.figure(figsize=(12, 6))
        plt.hist(test_scores[test_labels == 0], bins=30, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(test_scores[test_labels == 1], bins=30, alpha=0.7, label='Anomaly', color='red', density=True)
        
        if thresholds is not None:
            colors = ['black', 'purple', 'orange']
            threshold_names = ['Standard (3σ)', 'Aggressive (1σ)', 'Conservative (5σ)']
            for i, threshold in enumerate(thresholds):
                color = colors[i] if i < len(colors) else 'gray'
                name = threshold_names[i] if i < len(threshold_names) else f'Threshold {i+1}'
                plt.axvline(threshold, color=color, linestyle='--', 
                           label=f'{name}: {threshold:.4f}')
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title(f'Anomaly Score Distribution - {self.product_class.capitalize()} (ROC AUC = {roc_auc:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        dist_save_path = os.path.join(test_plots_dir, "score_distribution.png")
        plt.savefig(dist_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Score distribution saved to: {dist_save_path}")


if __name__ == "__main__":
    # Configuration
    product_class = "hazelnut"
    
    # Get paths
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(parent, "config.yaml")
    
    #experiment_name = f"deep_feature_{product_class}"
    experiment_name = f"deep_feature_foundational"
    experiment_dir = os.path.join("output", experiment_name)
    
    train_path = os.path.join(experiment_dir, "train")
    test_path = os.path.join(experiment_dir, "test")
    
    # Initialize tester
    print(f"Initializing tester for {product_class}...")
    tester = DeepFeatureADTester(config_path, product_class)
    
    # Set up paths for weights and thresholds
    #weights_path = os.path.join(train_path, "checkpoints", f"{product_class}_dfad_weights.pth")
    weights_path = os.path.join(train_path, "checkpoints", f"wood_dfad_weights.pth")
    threshold_path = os.path.join(train_path, f"{product_class}_thresholds.yaml")
    
    # Check if files exist
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        print("Please run the trainer first to generate model weights.")
        sys.exit(1)
        
    if not os.path.exists(threshold_path):
        print(f"Warning: Threshold file not found at {threshold_path}")
        print("Testing without thresholds (only ROC AUC will be computed)")
        threshold_path = None
    
    # Run testing
    print(f"Testing model for {product_class}...")
    roc_auc, test_results = tester.test(
        weights_path=weights_path,
        threshold_path=threshold_path,
        save_plots=True
    )
    
    # Save test results
    results_save_path = os.path.join(test_path, "test_results.yaml")
    os.makedirs(test_path, exist_ok=True)
    with open(results_save_path, "w") as file:
        yaml.dump(test_results, file)
    
    print(f"\n=== Testing Complete ===")
    print(f"ROC AUC: {roc_auc:.4f}")
    if test_results['accuracies'] is not None:
        for i, acc in enumerate(test_results['accuracies']):
            print(f"Accuracy (threshold {i+1}): {acc:.4f}")
    print(f"Test results saved to: {results_save_path}")
    print(f"Plots saved to: {test_path}")