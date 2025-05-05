from dataset_preprocesser import MVTecAD2
from torchvision                        import transforms
from matplotlib import pyplot as plt
import yaml
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt


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
        batch_size          = self.model_config.get("batch_size")
        num_epochs          = self.model_config.get("num_epochs")
        validation_split    = self.model_config.get("validation_split")
        
        # Move the model to the appropriate device (GPU or CPU)
        self.model.to(self.device)

        # Initialize the training dataset
        train_dataset = MVTecAD2(self.product_class, "train", self.test_path, transform=self.transform)
        
        # Split the dataset into training and validation subsets
        val_size    = int(validation_split * len(train_dataset))
        train_size  = len(train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        # Create DataLoaders for training and validation
        train_loader = DataLoader(train_subset, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  num_workers=8)

        val_loader = DataLoader(val_subset, 
                                batch_size=batch_size, 
                                shuffle=False,
                                num_workers=8)
        
        # Training and validation loop
        for epoch in range(num_epochs):
            # Training phase: set model to training mode
            self.model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
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
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
            writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
            
            # Validation phase: set model to evaluation mode
            self.model.eval()
            val_loss = 0.0
            reconstruction_errors = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
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
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
            print(f"Reconstruction Error - Mean: {mean_rec_error:.4f}, Std: {std_rec_error:.4f}")
            
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
            writer.add_scalar("Reconstruction/Mean", mean_rec_error, epoch + 1)
            writer.add_scalar("Reconstruction/Std", std_rec_error, epoch + 1)
        
        writer.close()

    def test(self):
        pass

    def save_model(self, args):
        # Save the model weights
        model_save_path = os.path.join(self.test_path, "autoencoder_weights.pth")
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model weights saved to {model_save_path}")
        # Save the model configuration
        config_save_path = os.path.join(self.test_path, "config.yaml")
        with open(config_save_path, "w") as file:
            yaml.dump(self.model_config, file)
        print(f"Model configuration saved to {config_save_path}")
        # Save the arguments
        args_save_path = os.path.join(self.test_path, "args.yaml")
        with open(args_save_path, "w") as file:
            yaml.dump(vars(args), file)
        print(f"Arguments saved to {args_save_path}")


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