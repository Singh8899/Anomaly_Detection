"""@autor: Carlo Merola"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class FeatureExtractor(nn.Module):
    """
    Extracts intermediate features from a pretrained ResNet50 model and returns a concatenated feature representation.
    Captures activations from specified layers using forward hooks, aligns them, and applies optional smoothing.
    """
    
    def __init__(self, layer_hooks=['layer2', 'layer3'], smooth=True):
        super(FeatureExtractor, self).__init__()
        
        # Load pretrained model
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.eval() # set inference mode
        # Disable gradient computations for backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
            
        # list to store and concatenate features
        # features will extracted from different layers and concatenated
        # to exploit both semantic richer information in later layers
        # and spatial information of the previous layers
        self.collected_features = []
        
        # Hook function to store layer outputs
        def hook_capture(_, __, output):
            self.collected_features.append(output)
            
        # Register hooks on specific layers passed as argument
        # layer1[-1] outputs [256, 56, 56]
        # layer2[-1] outputs [512, 28, 28]
        # layer3[-1] outputs [1024, 14, 14]
        # layer4[-1] outputs [2048, 7, 7]
        for layer in layer_hooks:
            getattr(self.backbone, layer)[-1].register_forward_hook(hook_capture)
        
        # arg to control whether to apply smoothing
        # smoothing allows for less noisy features
        # and robustness to small perturbations or individual pixel changes
        # (for egz. isolated activations of neurons)
        self.smooth = smooth
        self.smoothing_layer = nn.AvgPool2d(kernel_size=3, stride=1, padding=1) if self.smooth else None
        
        def forward(self, x):
            """
            Extract and align features from ResNet50 backbone.
            
            Args:
                x: Input tensor [B, 3, 224, 224]
            Returns:
                merged_features
            """
            
            self.collected_features = []    # to reset at each forward pass
            
            # activate hooks
            with torch.no_grad():
                _ = self.backbone(x)
            
            # exrtract features from the collected list
            first_layer_features = self.collected_features[0]
            second_layer_features = self.collected_features[1]
            
            # define target size of concatenation with the bigger spatial resolution
            target_size = first_layer_features.shape[-2:] # [28, 28] for layer2
            
            # TODO: try with transposed convolutions
            second_layer_resized = F.interpolate(
                second_layer_features, 
                size=target_size, 
                mode='bilinear', 
            )
        
            if self.smooth:
                # apply smoothing to both feature maps (using interpolated features for second layer)
                first_layer_smooth = self.smoothing_layer(first_layer_features)
                second_layer_smooth = self.smoothing_layer(second_layer_resized)
                
            # concatenate features along channel dimension
            merged_features = torch.cat([first_layer_smooth, second_layer_smooth], dim=1)
            # [B, 1536, 28, 28] = [512, 28, 28] + [1024, 28, 28] if layer2 and layer3 are used
            
            return merged_features
        


class EncoderLayer(nn.Module):
    """
    Encoder layer for the Autoencoder.
    It consists of a series of convolutional layers followed by ReLU activations.
    
    Why kernel_size=1: Pointwise Feature Transformation - working with already processed features from ResNet50.
    The features already contain spatial and semantic information.
    Using 1x1 convolutions allows for efficient channel-wise transformations without altering spatial dimensions.
    Linear combination of feature channles at each spatial location.
    ResNet50 for spatial patterns, AE focus on compression/reconstruction.
    """
    
    def __init__(self, in_channels, output_dim, kernel_size=1, stride=1, padding=0, is_bn=True):
        super(EncoderLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, output_dim, kernel_size, stride, padding)
        if is_bn:
            self.bn = nn.BatchNorm2d(output_dim)
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = x if self.bn is None else self.bn(x)    # identity if no batch normalization
        x = self.relu(x)
        return x
    
class Encoder(nn.Module):
    """
    Encoder for the Autoencoder.
    It consists of a composition of EncoderLayers to compress the input features.
    """
    
    def __init__(self, in_channels=1536, latent_dim=100, is_bn=True):
        super(Encoder, self).__init__()
        # if using layer2 and layer3 features, in_channels = 512 + 1024 = 1536
        # Define encoder layers with gradual compression strategy:
        # 1536 → 818 → 200 → 100 (smooth dimensionality reduction)
        # Using 2*latent_dim as intermediate step prevents information bottleneck
        
        # First compression: 1536 → 818 channels
        self.layer1 = EncoderLayer(in_channels, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0, is_bn=is_bn)
        # Second compression: 818 → 200 channels (2 * latent_dim)
        self.layer2 = EncoderLayer((in_channels + 2 * latent_dim) // 2, 2 * latent_dim, kernel_size=1, stride=1, padding=0, is_bn=is_bn)
        # Final compression: 200 → 100 channels (latent_dim)
        self.layer3 = EncoderLayer(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0, is_bn=is_bn)
        
    def forward(self, x):
        x = self.layer1(x)  # [B, 818, 28, 28] - initial feature representation if using layer2 and layer3
        x = self.layer2(x)  # [B, 200, 28, 28] - intermediate feature representation
        x = self.layer3(x)  # [B, 100, 28, 28] - compressed feature representation
        return x
    
class DecoderLayer(nn.Module):
    """
    Decoder layer for the Autoencoder.
    It consists of a series of transposed convolutional layers followed by ReLU activations.
    
    Why kernel_size=1: Pointwise Feature Transformation - working with already processed features from ResNet50.
    The features already contain spatial and semantic information.
    Using 1x1 convolutions allows for efficient channel-wise transformations without altering spatial dimensions.
    Linear combination of feature channles at each spatial location.
    ResNet50 for spatial patterns, AE focus on compression/reconstruction.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, is_bn=True):
        super(DecoderLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if is_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x):
            x = self.conv(x)
            x = x if self.bn is None else self.bn(x)
            x = self.relu(x)
            return x
    
class Decoder(nn.Module):
    """
    Decoder for the Autoencoder.
    It consists of a composition of DecoderLayers to reconstruct the input features.
    """
    
    def __init__(self, in_channels=1536, latent_dim=100, is_bn=True):
        super(Decoder, self).__init__()
        # Define decoder layers with gradual expansion strategy:
        # 100 → 200 → 818 → 1536 (smooth dimensionality expansion)
        
        # First expansion: 100 → 200 channels (2 * latent_dim)
        self.layer1 = DecoderLayer(in_channels, 2 * latent_dim, kernel_size=1, stride=1, padding=0, is_bn=is_bn)
        # Second expansion: 200 → 818 channels
        self.layer2 = DecoderLayer(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0, is_bn=is_bn)
        # Final expansion: 818 → 1536 channels (reconstruct original feature size)
        self.layer3 = DecoderLayer((in_channels + 2 * latent_dim) // 2, in_channels, kernel_size=1, stride=1, padding=0, is_bn=is_bn)
        
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x

class AE(nn.Module):
    """
    Autoencoder for ```feature``` reconstruction.
    It operates at the feature level, not pixel level.
    Advantages:
        - More robust to noise and small perturbations
        - Leverages Transfer Learning: Resnet50 already good at detecting shapes and patterns
        - AE only focuses on understanding if anomaly in shape or texture or pattern, doens't waste energy on learning how to understand shapes, edges and other things
    """
    
    def __init__(self, in_channels=1536, latent_dim=100, is_bn=True):
        super(AE, self).__init__()
        
        # Encoder and Decoder
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim, is_bn=is_bn)
        self.decoder = Decoder(in_channels=in_channels, latent_dim=latent_dim, is_bn=is_bn)
        
    def forward(self, x):
        """
        Forward pass through the Autoencoder.
        
        Args:
            x: Input tensor [B, 1536, 28, 28] if using layer2 and layer3 features of ResNet50
        Returns:
            reconstructed tensor [B, 1536, 28, 28] if using layer2 and layer3 features of ResNet50
        """
        encoded = self.encoder(x)  # [B, 100, 28, 28]
        reconstructed = self.decoder(encoded)  # [B, 1536, 28, 28]
        return reconstructed



class DeepFeatureAutoEncoder(nn.Module):
    """
    Deep Feature Autoencoder for Anomaly Detection.
    Combines FeatureExtractor with Autoencoder to reconstruct features.
    """
    
    def __init__(self, layer_hooks=['layer2', 'layer3'], latent_dim=100, is_bn=True, smooth=True):
        super(DeepFeatureAutoEncoder, self).__init__()
        
        # Feature extractor to get intermediate features
        self.feature_extractor = FeatureExtractor(layer_hooks=layer_hooks, smooth=smooth)
        
        # Calculate the in_channels for autoencoder based on layer_hooks used
        layer_channels = {
            'layer1': 256,
            'layer2': 512, 
            'layer3': 1024,
            'layer4': 2048
        }
        in_channels = sum(layer_channels[layer] for layer in layer_hooks)
        # in_channels = 512 + 1024 = 1536 if using layer2 and layer3 features
        
        # Autoencoder for feature reconstruction
        self.autoencoder = AE(in_channels=in_channels, latent_dim=latent_dim, is_bn=is_bn)
        
    def forward(self, x):
        """
        Forward pass through the Deep Feature Autoencoder.
        
        Args:
            x: Input tensor [B, 3, 224, 224]
        Returns:
            reconstructed features [B, 1536, 28, 28] if using layer2 and layer3 features of ResNet50
        """
        features = self.feature_extractor(x)  # Extract features
        reconstructed = self.autoencoder(features)  # Reconstruct features
        return features, reconstructed