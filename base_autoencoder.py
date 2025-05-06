import os
import yaml
import numpy as np
from PIL                        import Image

import torch
import torch.nn as nn
from torch.utils.data           import DataLoader
from torch.utils.tensorboard    import SummaryWriter

from torchvision                import transforms

from matplotlib                 import pyplot as plt

from tqdm                       import tqdm

from dataset_preprocesser       import MVTecAD2

from sklearn.metrics import roc_auc_score
import os


class BaseAutoencoder():
    def __init__(self, product_class, config_path, train_path, test_path):
        self.config_path = config_path
        self.train_path = train_path
        self.test_path = test_path
        self.product_class = product_class

        # Load configuration from config.yaml
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.model_config = self.config['MODELS_CONFIG']
        self.model_config = self.model_config['base_autoencoder']

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model, loss function, and optimizer
        self.model = Autoencoder()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.model_config['learning_rate']))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def train(self):
        """
        Train the autoencoder model with a training and validation phase.
        This method splits the training dataset into training and validation subsets,
        and then iteratively updates the model weights using the training data while
        monitoring the performance using the validation data. Metrics are logged to TensorBoard.
        """
        log_dir = os.path.join(self.train_path)
        writer = SummaryWriter(log_dir=log_dir)
        
        print(self.model)
        
        # Calculate and display the total number of parameters in the autoencoder
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters in the autoencoder: {total_params}")
        
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
        train_loader = DataLoader(train_subset, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  num_workers=num_workers)

        val_loader = DataLoader(val_subset, 
                                batch_size=batch_size, 
                                shuffle=False,
                                num_workers=num_workers)
        
        # Training and validation loop
        print("Starting training...")
        for epoch in range(num_epochs):
            # Training phase: set model to training mode
            self.model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", leave=False):
                # Transfer input images to the device
                inputs = batch["sample"].to(self.device)

                # Forward pass: compute reconstructed images
                outputs = self.model(inputs)
                
                # Compute the training loss (mean squared error)
                loss = self.criterion(outputs, inputs)

                # Backward pass: compute gradients and update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate loss for the epoch
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
            
            # Validation phase: set model to evaluation mode
            self.model.eval()
            val_loss = 0.0
            reconstruction_errors = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", leave=False):
                    # Transfer validation images to the device
                    inputs = batch["sample"].to(self.device)

                    # Forward pass on validation data
                    outputs = self.model(inputs)
                    
                    # Compute validation loss
                    loss = self.criterion(outputs, inputs)
                    
                    # Calculate per-image reconstruction error by averaging squared error per image
                    per_image_error = torch.mean((inputs - outputs) ** 2, dim=(1, 2, 3))
                    reconstruction_errors.extend(per_image_error.cpu().numpy())

                    # Accumulate validation loss for the epoch
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            mean_rec_error = torch.tensor(reconstruction_errors).mean().item()
            std_rec_error = torch.tensor(reconstruction_errors).std().item()
            print("==========================================")
            print(f"Epoch: {epoch + 1}/{num_epochs} || Train | Loss: {avg_train_loss:>.6f} || Val | Loss: {avg_val_loss:>.6f} | MSE-Mean: {mean_rec_error:>.6f} | MSE-Std: {std_rec_error:>.6f}")
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
            writer.add_scalar("Reconstruction/Mean", mean_rec_error, epoch + 1)
            writer.add_scalar("Reconstruction/Std", std_rec_error, epoch + 1)
        
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
        train_loader = DataLoader(self.test_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=num_workers)
        
        for el in train_loader:
            # Get the input image and move to device. Add a batch dimension.
            sample = el["sample"].to(self.device)
            gt_anomaly = np.array(["bad" in path for path in el["image_path"]],dtype=int)

            with torch.no_grad():
                # Forward pass: unsqueeze to add batch dimension
                reconstructed = self.model(sample)
            # Compute reconstruction error (MSE)
            stats_path = os.path.join(self.train_path, "training_statistics.yaml")
            with open(stats_path, "r") as file:
                stats = yaml.safe_load(file)
            threshold = float(stats["threshold"])

            # Compute per-image MSE and binarize based on the loaded threshold
            # Compute pixel-wise squared error and average across channels to obtain a 2D error map per image
            error = torch.mean((sample - reconstructed) ** 2, dim=(1,2,3)).cpu().numpy()  # shape: (batch, )
            # Binarize the error mask based on the threshold for each pixel
            error_mask = (error > threshold).astype(int)

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
        for el in tqdm(self.train_dataset, desc="Processing train dataset"):
            # Get the input image and its path
            sample      = el["sample"].to(self.device)
            # output_path = el["rel_out_path_thresh"]
            # Perform forward pass to get the reconstructed image
            with torch.no_grad():
                reconstructed = self.model(sample)

            # # Convert the reconstructed image to a format suitable for saving
            # reconstructed_image = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # reconstructed_image = (reconstructed_image * 255).astype(np.uint8)

            # # Save the reconstructed image
            # os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Image.fromarray(reconstructed_image).save(output_path)

            # Compute the squared difference between the input and reconstructed image
            squared_difference = (sample - reconstructed) ** 2

            # Compute the mean along the channels
            difference_image = torch.mean(squared_difference, dim=0).squeeze(0).cpu().numpy()

            # Compute the anomaly score
            anomaly_score = np.mean(difference_image)
            anomaly_scores.append(anomaly_score)

            # # Normalize the difference image to the range [0, 255]
            # difference_image = (difference_image * 255).astype(np.uint8)

            # # Save the difference image (mask)
            # mask_output_path = output_path.replace(".png", "_mask.png")
            # Image.fromarray(difference_image).save(mask_output_path)
        # print(f"Reconstructed images saved to {output_path}")
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


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, 112, 112)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, 56, 56)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (B, 256, 28, 28)
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 128, 56, 56)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 64, 112, 112)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 3, 224, 224)
            nn.Sigmoid(),  # Normalize output to [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded