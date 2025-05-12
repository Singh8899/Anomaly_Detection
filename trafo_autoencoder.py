import os
import yaml
import numpy as np
from PIL                        import Image
from torchinfo import summary
import torch
import torch.nn as nn
from torch.utils.data           import DataLoader
from torch.utils.tensorboard    import SummaryWriter
from torchvision.models         import efficientnet_b4, EfficientNet_B4_Weights
from torch.optim.lr_scheduler   import CosineAnnealingLR

from torchvision                import transforms

from matplotlib                 import pyplot as plt

from tqdm                       import tqdm

from dataset_preprocesser       import MVTecAD2

from sklearn.metrics import roc_auc_score
import os




class TransAEManager():
    def __init__(self, product_class, config_path, train_path, test_path):
        self.config_path = config_path
        self.train_path = train_path
        self.test_path = test_path
        self.product_class = product_class

        # Load configuration from config.yaml
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.model_config = self.config['MODELS_CONFIG']
        self.model_config = self.model_config['trafo_autoencoder']

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model, loss function, and optimizer
        self.model = TransformerAE()
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        self.criterion = nn.MSELoss()

        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.model_config['learning_rate']))

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        # Initialize cosine annealing learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.model_config.get("num_epochs")),
            eta_min=float(self.model_config.get("min_lr"))
        )
        patience = int(self.model_config.get("patience"))
        delta = float(self.model_config.get("delta"))
        self.early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=False)

    def train(self):
        """
        Train the autoencoder model with a training and validation phase.
        This method splits the training dataset into training and validation subsets,
        and then iteratively updates the model weights using the training data while
        monitoring the performance using the validation data. Metrics are logged to TensorBoard.
        """
        log_dir = os.path.join(self.train_path)
        writer = SummaryWriter(log_dir=log_dir)
        
        # Retrieve hyperparameters from configuration
        batch_size          = int(self.model_config.get("batch_size"))
        num_epochs          = int(self.model_config.get("num_epochs"))
        validation_split    = float(self.model_config.get("validation_split"))
        num_workers         = int(self.model_config.get("num_workers"))
        print(f"Batch size: {batch_size}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Validation split: {validation_split}")
        print(f"Training path: {self.train_path}")
        # Move the model to the appropriate device (GPU or CPU)
        self.model.to(self.device)

        # Initialize the training dataset
        self.train_dataset = MVTecAD2(self.product_class, "train", self.train_path, transform=self.transform)
        
        # Split the dataset into training and validation subsets
        val_size    = int(validation_split * len(self.train_dataset))
        train_size  = len(self.train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(self.train_dataset, [train_size, val_size])
        print(f"Training on {len(train_subset)} samples, validating on {len(val_subset)} samples.")
        
        # Create DataLoaders for training and validation
        self.train_loader = DataLoader(train_subset, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  num_workers=num_workers)

        self.val_loader = DataLoader(val_subset, 
                                batch_size=batch_size, 
                                shuffle=False,
                                num_workers=num_workers)
        
        # Training and validation loop
        print("Starting training...")
        print(summary(self.model, input_size=(batch_size, 3, 256, 256)))
        best_val = float('inf')
        best_epoch = 0
        for epoch in range(num_epochs):
            # Training phase: set model to training mode
            self.model.train()
            
            epoch_loss = 0.0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", leave=False):
                # Transfer input images to the device
                inputs = batch["sample"].to(self.device)

                # Forward pass: compute reconstructed images
                outputs, features = self.model(inputs)
                
                # Compute the training loss (mean squared error)
                loss = self.criterion(outputs, features)

                # Backward pass: compute gradients and update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate loss for the epoch
                epoch_loss += loss.item()

            # Update learning rate scheduler after each epoch
            self.scheduler.step()

            avg_train_loss = epoch_loss / len(self.train_loader)
            writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
                
            # Validation phase: set model to evaluation mode
            self.model.eval()
            val_loss = 0.0
            reconstruction_errors = []
            with torch.inference_mode():
                for batch in tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", leave=False):
                    # Transfer validation images to the device
                    inputs = batch["sample"].to(self.device)

                    # Forward pass on validation data
                    outputs, features = self.model(inputs)
                    
                    # Compute validation loss
                    loss = self.criterion(outputs, features)
                    
                    # Compute the difference vector at each spatial location
                    diff = features - outputs  # shape (B, C, H, W)
                    
                    # Calculate the pixel-level anomaly map by computing the L2 norm across channels (for each pixel)
                    anomaly_map = torch.linalg.norm(diff, dim=1)  # shape (B, C, H, W) -> (B, H, W)
                    
                    # Pool the anomaly map of shape (B, 16, 16) to a single value per image using adaptive max pooling.
                    img_anomaly_score = torch.nn.functional.adaptive_max_pool2d(anomaly_map, (1, 1)).reshape(anomaly_map.shape[0])
                    
                    reconstruction_errors.extend(img_anomaly_score.cpu().numpy())

                    # Accumulate validation loss for the epoch
                    val_loss += loss.item()
            self.early_stopping.check_early_stop(val_loss)
            avg_val_loss = val_loss / len(self.val_loader)
            mean_rec_error = torch.tensor(reconstruction_errors).mean().item()
            std_rec_error = torch.tensor(reconstruction_errors).std().item()
            print("==========================================")
            print(f"Epoch: {epoch + 1}/{num_epochs} || Train | Loss: {avg_train_loss:>.6f} || Val | Loss: {avg_val_loss:>.6f} | MSE-Mean: {mean_rec_error:>.6f} | MSE-Std: {std_rec_error:>.6f}")
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
            writer.add_scalar("Reconstruction/Mean", mean_rec_error, epoch + 1)
            writer.add_scalar("Reconstruction/Std", std_rec_error, epoch + 1)
            
            # Save the best epoch based on the lowest validation reconstruction mean error
            if mean_rec_error < best_val:
                best_val = mean_rec_error
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), os.path.join(self.train_path, "autoencoder_weights.pth"))
                print(f"Best model updated at Epoch {best_epoch} with MSE-Mean: {mean_rec_error:>.6f}")

            self.early_stopping.check_early_stop(mean_rec_error)
            if self.early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch}")
                break

        writer.close()
        print("Training completed.")

    def test(self):
        weights_path = os.path.join(self.train_path, "autoencoder_weights.pth")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        test_scores = []
        test_labels = []

        self.test_dataset = MVTecAD2(self.product_class, "test", self.test_path, transform=self.transform)
        batch_size = int(self.model_config.get("batch_size"))
        num_workers = int(self.model_config.get("num_workers"))
        test_loader = DataLoader(self.test_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=num_workers)
        
        for el in test_loader:
            # Get the input image and move to device. Add a batch dimension.
            sample = el["sample"].to(self.device)
            gt_anomaly = np.array(["bad" in path for path in el["image_path"]],dtype=int)

            with torch.inference_mode():
                # Forward pass: unsqueeze to add batch dimension
                reconstructed, features = self.model(sample)
            # Compute reconstruction error (MSE)
            stats_path = os.path.join(self.train_path, "training_statistics.yaml")
            with open(stats_path, "r") as file:
                stats = yaml.safe_load(file)
            threshold = float(stats["threshold"])

    
            # Compute the difference vector at each spatial location
            diff = features - reconstructed  # shape (B, C, H, W)
            
            # Calculate the pixel-level anomaly map by computing the L2 norm across channels (for each pixel)
            anomaly_map = torch.linalg.norm(diff, dim=1)  # shape (B, C, H, W) -> (B, H, W)
            
            # Pool the anomaly map of shape (B, 16, 16) to a single value per image using adaptive max pooling.
            img_anomaly_score = torch.nn.functional.adaptive_max_pool2d(anomaly_map, (1, 1)).reshape(anomaly_map.shape[0]).cpu().numpy()
            
            error_mask = (img_anomaly_score > threshold).astype(int)

            test_scores.extend(error_mask)
            
            # Get ground truth label: 0 for normal, 1 for defective.
            test_labels.extend(gt_anomaly)

        roc_auc = roc_auc_score(test_labels, test_scores)
        print(f"ROC AUC on test set: {roc_auc}")

        if roc_auc < 0.5:
            print("ROC AUC is less than 0.5. The model might be worse than random. Consider redesigning the Autoencoder.")

    def compute_thresh(self):
        # Set the autoencoder to evaluation mode
        self.model.eval()
        anomaly_scores = []

        # Perform inference on the test dataset
        for el in tqdm(self.train_loader, desc="Processing train dataset"):
            # Get the input image and its path
            sample      = el["sample"].to(self.device)
            # Perform forward pass to get the reconstructed image
            with torch.inference_mode():
                reconstructed, features = self.model(sample)

            # Compute the difference vector at each spatial location
            diff = features - reconstructed  # shape (B, C, H, W)
            
            # Calculate the pixel-level anomaly map by computing the L2 norm across channels (for each pixel)
            anomaly_map = torch.linalg.norm(diff, dim=1)  # shape (B, C, H, W) -> (B, H, W)
            
            # Pool the anomaly map of shape (B, 16, 16) to a single value per image using adaptive max pooling.
            img_anomaly_score = torch.nn.functional.adaptive_max_pool2d(anomaly_map, (1, 1)).reshape(anomaly_map.shape[0])

  
            anomaly_scores.extend(img_anomaly_score.cpu().numpy())

        # Print the mean anomaly score
        print(f"Mean Anomaly Score: {np.mean(anomaly_scores)}")
        # Compute mean (μ) and standard deviation (σ) of anomaly scores
        mean_error = np.mean(anomaly_scores)
        std_error  = np.std(anomaly_scores)

        # Set threshold = μ + 3σ
        threshold = mean_error + 3 * std_error
        print(f"Mean Error (μ): {mean_error}")
        print(f"Standard Deviation (σ): {std_error}")
        print(f"Threshold: {threshold}")

        # Plot histogram of training errors
        plt.hist(anomaly_scores, bins=30, density=True, alpha=0.7, color='blue', label='Training Errors')
        plt.axvline(mean_error, color='green', linestyle='--', label='Mean (μ)')
        plt.axvline(threshold, color='red', linestyle='--', label='Threshold (μ + 3σ)')
        plt.title('Histogram of Training Errors')
        plt.xlabel('Error')
        plt.ylabel('Density')
        plt.legend()
        save_path = os.path.join(self.train_path, "training_errors_histogram.png")
        plt.savefig(save_path)
        plt.close()
        return mean_error, std_error, threshold

    def save_model(self, args, mean_error, std_error, threshold):
        # Save the model weights
        model_save_path = os.path.join(self.train_path, "autoencoder_weights.pth")
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model weights saved to {model_save_path}")
        # Save the model configuration
        config_save_path = os.path.join(self.train_path, "config.yaml")
        with open(config_save_path, "w") as file:
            yaml.dump(self.model_config, file)
        print(f"Model configuration saved to {config_save_path}")
        # Save the arguments
        args_save_path = os.path.join(self.train_path, "args.yaml")
        with open(args_save_path, "w") as file:
            yaml.dump(vars(args), file)
        print(f"Arguments saved to {args_save_path}")
        # Save the training statistics
        stats_save_path = os.path.join(self.train_path, "training_statistics.yaml")
        with open(stats_save_path, "w") as file:
            stats = {
            "mean_error": float(mean_error),
            "std_error": float(std_error),
            "threshold": float(threshold)
            }
            yaml.dump(stats, file)
        print(f"Training statistics saved to {stats_save_path}")

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")


class TransformerAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone: EfficientNet-B4 feature extractor with pretrained weights
        weights = EfficientNet_B4_Weights.DEFAULT
        self.backbone = efficientnet_b4(weights=weights)
        
        # Transformer encoder-decoder
        self.transformer = torch.nn.Transformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            batch_first=True
        )
        # Learned auxiliary query embedding for the decoder (sequence length = 16*16 = 256)
        self.query_embed = nn.Parameter(torch.randn(16 * 16, 256))
        
        # Tokenization: 1x1 convolution to reduce the unified 720 channels to 256
        self.tokenizer = nn.Conv2d(720, 256, kernel_size=1)
        
        # Define the layer indices from which to extract features
        self.extract_layers = [1, 2, 3, 5, 7]

        # To map 256-dim transformer outputs back up to 720 channels
        self.proj = nn.Conv2d(256, 720, kernel_size=1)
        
    def forward(self, x):
        # Pass input through the backbone features sequentially and extract features from specified layers.
        feature_maps = []
        out = x
        for i, layer in enumerate(self.backbone.features):
            out = layer(out)
            if i in self.extract_layers:
                # Resize feature maps if necessary to have H, W equal to 16
                if out.shape[-2:] != (16, 16):
                    out = torch.nn.functional.interpolate(out, size=(16, 16), mode='bilinear', align_corners=False)
                feature_maps.append(out)
        
        # Concatenate the extracted feature maps along the channel dimension.
        # The resulting unified feature map has shape [B, 720, 16, 16]
        unified = torch.cat(feature_maps, dim=1)
        
        # Reduce the channel dimension from 720 to 256 with a 1x1 convolution.
        tokenized = self.tokenizer(unified)
        B, C, H, W = tokenized.shape
        
        # Reshape to (batch, sequence_length, embed_dim) where sequence_length = H * W
        tokens = tokenized.reshape(B, C, H * W).permute(0, 2, 1)
        
        # Prepare decoder queries: expand learned embedding for the current batch (B, sequence_length, 256)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        
        # Transformer encoder-decoder forward pass
        transformed = self.transformer(tokens, queries)
        
        # Reshape transformer output back to spatial feature map (B, 256, H, W)
        transformed = transformed.permute(0, 2, 1).reshape(B, 256, H, W)
        
        transformed = self.proj(transformed)  # -> (B, 720, 16, 16)
        
        return transformed, unified