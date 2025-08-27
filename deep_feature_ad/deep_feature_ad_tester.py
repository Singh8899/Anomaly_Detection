"""
@author: Carlo Merola
"""
import os
import sys
import yaml
import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score

from deep_feature_anomaly_detector import DeepFeatureAnomalyDetector    # load detector with initial weights

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
        
        # Dataset is instantiated on-demand per class to support foundational multi-class testing
        self.test_dataset = None

    def load_config(self, path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def test(self, weights_path, threshold_path=None, save_plots=True, product_class_override=None, experiment_tag=None, subfolder=None):
        """
        Test the model on the test dataset and compute ROC AUC.
        Args:
            product_class_override: optional class name used for dataset loading (foundational mode)
            experiment_tag: name segment for output folder, defaults to current product_class
            subfolder: optional subfolder under test/ for saving plots/results
        """
        # Resolve which class to run
        run_class = product_class_override or self.product_class
        exp_tag = experiment_tag or self.product_class

        # Load model weights
        if not os.path.exists(weights_path):
            print(f"Model weights not found at {weights_path}")
            return

        self.detector.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.detector.eval()
        
        # Load threshold if provided
        thresholds = None
        if threshold_path and os.path.exists(threshold_path):
            with open(threshold_path, "r") as file:
                threshold_info = yaml.safe_load(file)

            # Be robust to a few possible shapes
            if isinstance(threshold_info, dict):
                if 'thresholds' in threshold_info and isinstance(threshold_info['thresholds'], dict) and 'thresholds' in threshold_info['thresholds']:
                    thresholds = threshold_info['thresholds']['thresholds']
                elif 'thresholds' in threshold_info:
                    thresholds = threshold_info['thresholds']
                elif 'threshold' in threshold_info:
                    thresholds = threshold_info['threshold']
            elif isinstance(threshold_info, list):
                thresholds = threshold_info

            # Ensure thresholds is always a list for consistent handling
            if thresholds is not None and not isinstance(thresholds, list):
                thresholds = [thresholds]
            if thresholds is not None:
                print(f"Using thresholds: {thresholds}")

        # Build dataset/loader for the selected class
        self.test_dataset = MVTecAD2(
            run_class,
            "test",
            self.config['DATASET_PATH'],
            transform=self.transform
        )
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
        print(f"ROC AUC on test set ({run_class}): {roc_auc:.4f}")
        
        # Compute accuracy if thresholds are available
        accuracies = []
        if thresholds is not None:
            for i, threshold in enumerate(thresholds):
                predictions = (test_scores > threshold).astype(int)
                accuracy = np.mean(predictions == test_labels)
                accuracies.append(accuracy)
                print(f"Accuracy on test set (threshold {i+1}): {accuracy:.4f}")
        
        if save_plots:
            self._save_plots(test_labels, test_scores, roc_auc, thresholds, experiment_tag=exp_tag, subfolder=subfolder)
        
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
    
    def _save_plots(self, test_labels, test_scores, roc_auc, thresholds, experiment_tag=None, subfolder=None):
        """
        Save ROC curve and score distribution plots.
        """
        # Create output directory for test plots
        exp = experiment_tag or self.product_class
        test_plots_dir = f"output/deep_feature_{exp}/test"
        if subfolder:
            test_plots_dir = os.path.join(test_plots_dir, subfolder)
        os.makedirs(test_plots_dir, exist_ok=True)

        # Create and save ROC curve plot
        fpr, tpr, roc_thresholds = roc_curve(test_labels, test_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        title_name = subfolder or exp
        plt.title(f'ROC Curve - {title_name.capitalize()} (AUC = {roc_auc:.4f})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        roc_save_path = os.path.join(test_plots_dir, "roc_curve.png")
        plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to: {roc_save_path}")

        # Create and save score distribution plot with all thresholds__________________________________________
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
        plt.title(f'Anomaly Score Distribution - {title_name.capitalize()} (ROC AUC = {roc_auc:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        dist_save_path = os.path.join(test_plots_dir, "score_distribution.png")
        plt.savefig(dist_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Score distribution saved to: {dist_save_path}")


if __name__ == "__main__":
    # Configuration
    parser = argparse.ArgumentParser(description='Deep Feature Anomaly Detection Tester')
    parser.add_argument('--product_class', type=str, default='carpet', help='Product class to test on (use "foundational" or "foundational_7" to test the foundational model across classes)')
    args = parser.parse_args()
    product_class = args.product_class
    
    #product_class = "foundational"  # Change to 'foundational' to test on all classes for foundational model evaluation
    
    # Get paths
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(parent, "config.yaml")
    
    experiment_name = f"deep_feature_{product_class}"
    experiment_dir = os.path.join("output", experiment_name)
    train_path = os.path.join(experiment_dir, "train")
    test_path = os.path.join(experiment_dir, "test")
    
    # Initialize tester
    print(f"Initializing tester for {product_class}...")
    tester = DeepFeatureADTester(config_path, product_class)
    
    if not product_class.lower().startswith('foundational'):
        # Standard single-class testing
        weights_path = os.path.join(train_path, "checkpoints", f"{product_class}_dfad_weights.pth")
        threshold_path = os.path.join(train_path, f"{product_class}_thresholds.yaml")

        if not os.path.exists(weights_path):
            print(f"Error: Model weights not found at {weights_path}")
            print("Please run the trainer first to generate model weights.")
            sys.exit(1)

        if not os.path.exists(threshold_path):
            print(f"Warning: Threshold file not found at {threshold_path}")
            print("Testing without thresholds (only ROC AUC will be computed)")
            threshold_path = None

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
    else:
        # Foundational model testing across multiple classes
        print(f"{product_class} mode: evaluating across available classes...")
        checkpoints_dir = os.path.join(train_path, "checkpoints")
        if not os.path.isdir(checkpoints_dir):
            print(f"Error: checkpoints directory not found at {checkpoints_dir}")
            sys.exit(1)

        # Use the single shared foundational checkpoint
        weights_path = os.path.join(checkpoints_dir, "wood_dfad_weights.pth")
        if not os.path.exists(weights_path):
            print(f"Error: Foundational weights not found at {weights_path}")
            sys.exit(1)

        # Discover classes from available *_foundational_thresholds.yaml files
        available_thresholds = [f for f in os.listdir(train_path) if f.endswith('_foundational_thresholds.yaml')]
        classes = [fn.split('_foundational_thresholds.yaml')[0] for fn in available_thresholds]
        # Fallback to common classes if none found
        if not classes:
            if product_class.lower() == 'foundational_7':
                classes = ['bottle', 'carpet', 'hazelnut', 'leather', 'transistor', 'walnuts', 'wood']
            else:
                classes = ['carpet', 'hazelnut', 'leather', 'wood']

        summary = {}
        for cls in classes:
            print(f"\n--- Testing foundational model on class: {cls} ---")
            threshold_path = os.path.join(train_path, f"{cls}_foundational_thresholds.yaml")
            if not os.path.exists(threshold_path):
                print(f"Note: Thresholds not found for {cls} at {threshold_path}. Proceeding without thresholds.")
                threshold_path = None

            roc_auc, test_results = tester.test(
                weights_path=weights_path,
                threshold_path=threshold_path,
                save_plots=True,
                product_class_override=cls,
                experiment_tag=product_class,
                subfolder=cls
            )

            # Save per-class results under deep_feature_foundational/test/<class>/
            class_test_dir = os.path.join(test_path, cls)
            os.makedirs(class_test_dir, exist_ok=True)
            results_save_path = os.path.join(class_test_dir, "test_results.yaml")
            with open(results_save_path, "w") as file:
                yaml.dump(test_results, file)

            print(f"Completed {cls}: ROC AUC = {roc_auc:.4f}")
            print(f"Results saved to: {results_save_path}")
            summary[cls] = test_results

        # Save an aggregated summary
        aggregated_path = os.path.join(test_path, f"{product_class}_summary.yaml")
        with open(aggregated_path, "w") as file:
            yaml.dump(summary, file)
        print(f"\n=== Foundational Testing Complete ===")
        print(f"Aggregated summary saved to: {aggregated_path}")