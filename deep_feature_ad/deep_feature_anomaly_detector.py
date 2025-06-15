"""@author: Carlo Merola"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_feature_autoencoder_model import DeepFeatureAutoEncoder


class DeepFeatureAnomalyDetector(nn.Module):
    """
    Pipeline for deep feature-based anomaly detection.
    Computes reconstruction error from deep features and generates anomaly scores.
    Computes segmentation maps for visualizing anomalies.
    """

    def __init__(self, layer_hooks=['layer2', 'layer3'], latent_dim=100, smooth=True, is_bn=True):
        """
        Initialize the anomaly detector with specified layer hooks and other parameters.
        """
        super(DeepFeatureAnomalyDetector, self).__init__()
        
        # Initialize the deep feature autoencoder
        self.autoencoder = DeepFeatureAutoEncoder(
            layer_hooks=layer_hooks,
            latent_dim=latent_dim,
            smooth=smooth,
            is_bn=is_bn
        )
        self.threshold = None  # Threshold for anomaly detection, to be set later
        self.error_map = None  # Placeholder for the error map
        
    def forward(self, x):
        """
        Forward pass through the autoencoder.
        Returns the features and reconstructed output.
        """
        features, reconstructed = self.autoencoder(x)
        return features, reconstructed
        
    def set_threshold(self, threshold):
        self.threshold = threshold

    def compute_reconstruction_error(self, features, reconstructed):
        """
        Compute the reconstruction error for the input features.
        Outputs an error map that indicates the per-pixel reconstruction error.
        """
        error_map = torch.norm(features - reconstructed, p=2, dim=1)
        self.error_map = error_map  # Store the error map for later use
        return error_map

    def get_reconstruction_error(self):
        """
        Get the stored reconstruction error map.
        """
        if self.error_map is None:
            raise ValueError("Reconstruction error has not been computed yet.")
        return self.error_map
    
    def compute_anomaly_score(self, error_map, k=10):
        """
        Compute anomaly score as average of top-k errors.
        """
        if error_map is None:
            if self.error_map is not None:
                error_map = self.error_map
            else:
                raise ValueError("Reconstruction error has not been computed yet.")
        
        batch_size, height, width = error_map.shape
        anomaly_scores = []
        for i in range(batch_size):
            # Flatten the error map for the current sample
            flat_error = error_map[i].flatten()
            
            # Get top-k errors
            # anomalies manifest as localized regions with high reconstruction error
            # taking the mean of all pixels will dilute the anomaly signal and focus on the most anomalous regions
            top_k_errors, _ = torch.topk(flat_error, k)
            
            # Compute average of top-k errors
            anomaly_score = torch.mean(top_k_errors)
            anomaly_scores.append(anomaly_score)
            
        # get one tensor for all anomaly scores
        return torch.stack(anomaly_scores)
    
    def get_segmentation_map(self, error_map, target_size=(224, 224)):
        """
        Generate segmentation map by upsampling error map to input image size.
        """
        if error_map is None:
            if self.error_map is not None:
                error_map = self.error_map
            else:
                raise ValueError("Reconstruction error has not been computed yet.")
            
        # Upsample error map to target size
        upsampled = F.interpolate(error_map.unsqueeze(1), size=target_size, mode='bilinear')

        # Create a 3-channel segmentation map
        seg_map = upsampled.squeeze(1)
        return seg_map

    def predict_anomaly(self, k=10, threshold=None):
        """
        Binary prediction if image is anomalous based on the error map and a threshold.
        """        
        scores = self.compute_anomaly_score(k)
        if threshold is not None:
            self.set_threshold(threshold)
            predictions = (scores > self.threshold).float()
            return predictions, scores
        else:
            return None, scores  # If no threshold is set, return only scores
        
    
    def get_stats(self):
        """
        Get statistics of the model, including layer hooks and latent dimension.
        """
        return self.autoencoder.get_stats()