"""
@author: Carlo Merola
"""
from email import parser
import os
import sys
import yaml
import torch
import argparse

from torch import nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from deep_feature_ad_manager import DeepFeatureADManager
from deep_feature_anomaly_detector import DeepFeatureAnomalyDetector

# changing parent directory to import MVTecAD2 dataset
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
original_cwd = os.getcwd()
os.chdir(parent_dir)  # Change to parent directory to import MVTecAD2
from dataset_preprocesser import MVTecAD2
os.chdir(original_cwd)  # Change back to original working directory


class DeepFeatureADTrainer:
    """
    Trainer class for deep feature-based anomaly detection.
    Handles training, validation, and testing of the DeepFeatureAnomalyDetector.
    Supports single-class training and all-class training for foundation model evaluation purposes.
    In order to run test, the threshold must be set by calling the `compute_threshold` method (calls DeepFeatureAEManager).
    """

    def __init__(self, config_path, train_path, test_path, product_class):
        """
        Initialize the trainer with configuration parameters.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.config_path = config_path
        self.train_path = train_path
        self.test_path = test_path
        self.product_class = product_class
        self.model_path = None  # Path to save the model weights
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"USING DEVICE: {self.device}")
        
        self.model_config = self.config['MODELS_CONFIG']['DeepFeatureAE']
        
        self.model = DeepFeatureAnomalyDetector(
            layer_hooks=self.model_config['layer_hooks'],
            latent_dim=self.model_config['latent_dim'],
            smooth=self.model_config['smooth'],
            is_bn=self.model_config['is_bn']
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.model_config['learning_rate']))
        
        # every 10 epochs, reduce the learning rate by a factor of 0.1
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.transform = transforms.Compose([
            transforms.Resize((self.model_config['input_size'], self.model_config['input_size'])),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5], std=[0.5])
            # normalize the input images to match pre-trained model expectations on ImageNet (pretrained models dataset statistics)
            # pretrained models were trained on data normalized this way, so inputs must match this format for meaningful results.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        
        # if product_class is selected, load the dataset for that class
        if product_class != 'foundational':
            self.ad_manager = DeepFeatureADManager(
            product_class=self.product_class,
            config_path=self.config_path,
            train_path=self.train_path,
            test_path=self.test_path,
            threshold_computation_mode=self.model_config['threshold_computation_mode'],
            model=self.model
            )
            
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
            
            # Split into train and validation
            val_size = int(self.model_config['validation_split'] * len(self.train_dataset))
            train_size = len(self.train_dataset) - val_size
            self.train_subset, self.val_subset = torch.utils.data.random_split(
                self.train_dataset, [train_size, val_size]
            )

            self.train_loader = DataLoader(self.train_subset, 
                                batch_size=self.model_config['batch_size'], 
                                shuffle=True
                                )

            self.val_loader = DataLoader(self.val_subset,
                                batch_size=self.model_config['batch_size'], 
                                shuffle=False
                                )

            self.test_loader = DataLoader(self.test_dataset, 
                                batch_size=self.model_config['batch_size'], 
                                shuffle=False
                                )
            
            # Use OneCycleLR scheduler for learning rate scheduling
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                                 steps_per_epoch=len(self.train_loader), 
                                                                 epochs=self.model_config['num_epochs'], 
                                                                 max_lr=float(self.model_config['learning_rate']), 
                                                                 pct_start=0.3, 
                                                                 anneal_strategy='linear')

        # if product_class is 'foundational', initialize loader in train_all_classes
        elif product_class == 'foundational':
            self.train_dataset = None
            self.test_dataset = None
            self.scheduler = None
            self.ad_manager = None
        

    def train_single_class(self, save_model=True):
        """ 
        Train the model on the training dataset.
        """       
        self.model.train()
        
        # Calculate total parameters (getting info from deep_feature_autoencoder_model module)
        stats = self.model.get_stats()

        # Layers stats of the model
        print(f"Number of layers: {stats['num_layers']}")
        
        # Parameters stats of the model
        print(f"Backbone parameters (frozen): {stats['backbone_params']:,}")
        print(f"Autoencoder parameters: {stats['autoencoder_params']:,}")
        print(f"Trainable parameters: {stats['trainable_params']:,}")

        # Print dataset sizes
        print(f"Training on {len(self.train_dataset)} samples, validating on {len(self.val_subset)} samples.")

        best_val_loss = float('inf')
        for epoch in range(self.model_config['num_epochs']):
            train_loss = 0.0
            train_batches = 0

            for indx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.model_config['num_epochs']} - Training")):
                images = batch['sample'].to(self.device)
                
                # Forward pass
                features, reconstructed = self.model(images)
                
                # Compute loss
                loss = self.criterion(reconstructed, features)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()          # reset gradients from previous iteration
                loss.backward()                     # compute gradients   
                self.optimizer.step()               # update weights according to gradients and optimization algorithm

                # update learning rate scheduler only if setted
                if self.scheduler:
                    self.scheduler.step()               # update learning rate

                train_loss += loss.item()
                train_batches += 1
                
            avg_train_loss = train_loss / train_batches

            avg_val_loss, best_val_loss = self._validate(best_val_loss, epoch, save_model=save_model)

            print(f"Epoch {epoch+1}/{self.model_config['num_epochs']} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")


    def _validate(self, best_val_loss, epoch, save_model=True):
        """ Validate the model on the validation dataset. """
        self.model.eval()
        
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.model_config['num_epochs']} - Validation"):
                images = batch['sample'].to(self.device)
                features, reconstructed = self.model(images)
                loss = self.criterion(reconstructed, features)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the best model checkpoint
            if save_model:
                self._save_model()

        return avg_val_loss, best_val_loss
    
    def _save_model(self):
        """ Save the model checkpoint. """
        model_dir = os.path.join(self.train_path, 'checkpoints')
        os.makedirs(model_dir, exist_ok=True)
        
        self.model_path = os.path.join(model_dir, f"{self.product_class}_dfad_weights.pth")
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

        # save current configuration
        config_save_path = os.path.join(model_dir, f"{self.product_class}_dfad_config.yaml")
        with open(config_save_path, 'w') as file:
            yaml.dump(self.config, file)
            print(f"Config saved to {config_save_path}")
            
    def load_model_weights(self, weight_path=None):
        """
        Load the model weights from the specified path.
        If no path is provided, it will use the default model path.
        """
        if weight_path is None:
            weight_path = self.model_path
        
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Model weights not found at {weight_path}")
        
        self.ad_manager.load_model_weights(weight_path)
    
    def load_computed_thresholds(self, threshold_file=None):
        """
        Load the computed thresholds for anomaly detection.
        This method should be called before testing the model.
        """
        self.ad_manager.load_thresholds_for_class(threshold_file=threshold_file)
        print(f"Thresholds loaded for {self.product_class} class.")

    def compute_threshold(self):
        """
        Compute the threshold for anomaly detection using the DeepFeatureADManager.
        This method should be called before testing the model.
        """        
        self.ad_manager.compute_threshold()
        self.ad_manager.save_thresholds_for_class()
        print(f"Thresholds computed and saved for {self.product_class}.")
        
    def generate_segmentation_maps(self, num_examples=5, foundational=False):
        """
        Generate segmentation maps for the test dataset.
        This method will visualize the anomalies detected by the model.
        """
        self.ad_manager.generate_segmentation_maps(num_examples=num_examples, foundational=foundational)
        print(f"Segmentation maps generated for {num_examples} examples in {self.product_class} class.")
    
    def plot_anomalies_thresholds(self):
        """
        Plot the anomalies and thresholds for the specified product class.
        This method will visualize the anomaly scores and thresholds.
        """
        self.ad_manager.plot_anomalies_thresholds()
        print(f"Anomalies and thresholds plotted for {self.product_class} class.")
    
    
    def train_foundational(self, scheduler=True):
        """
        Train the model on all classes for foundation model evaluation purposes.
        This method will load the dataset for chosen foundational classes and call the train_single_class method for each class to update parameters.
        """
        for product_class in self.config['FOUNDATIONAL_OBJECTS']:
            print(f"Training on {product_class} class...")
            self.product_class = product_class
            
            # Load the dataset for the current class
            self.train_dataset = MVTecAD2(
                self.product_class, 
                "train", 
                self.train_path, 
                transform=self.transform
            )
            
            # Split into train and validation
            val_size = int(self.model_config['validation_split'] * len(self.train_dataset))
            train_size = len(self.train_dataset) - val_size
            self.train_subset, self.val_subset = torch.utils.data.random_split(
                self.train_dataset, [train_size, val_size]
            )

            self.train_loader = DataLoader(self.train_subset, 
                                batch_size=self.model_config['batch_size'], 
                                shuffle=True
                                )

            self.val_loader = DataLoader(self.val_subset,
                                batch_size=self.model_config['batch_size'], 
                                shuffle=False
                                )
            
            # Use OneCycleLR scheduler for learning rate scheduling
            if scheduler:
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                                 steps_per_epoch=len(self.train_loader),    # need to initialize the scheduler after the train_loader is created
                                                                 epochs=self.model_config['num_epochs'], 
                                                                 max_lr=float(self.model_config['learning_rate']), 
                                                                 pct_start=0.3, 
                                                                 anneal_strategy='linear')
            # Train the model on the current class
            self.train_single_class(save_model=False) # do not save the model weights, as we are training on multiple classes
    
    def save_foundational_model(self):
        """
        Save the model weights after training on all foundational classes.
        This method will save the model weights for the last class trained.
        """
        self.product_class == 'foundational'
        self._save_model()
            
    def compute_foundational_thresholds(self):
        """
        Compute the thresholds for all foundational classes.
        This method will compute and save the thresholds for each class in the foundational classes.
        """
        for product_class in self.config['FOUNDATIONAL_OBJECTS']:
            print(f"Computing thresholds for {product_class} class...")
            self.product_class = product_class
            
            # Load the dataset for the current class
            self.train_dataset = MVTecAD2(
                self.product_class, 
                "train", 
                self.train_path, 
                transform=self.transform
            )

            # Compute the thresholds for the current class, loading manager with current class information and distribution
            self.ad_manager = DeepFeatureADManager(
                product_class=self.product_class,
                config_path=self.config_path,
                train_path=self.train_path,
                test_path=self.test_path,
                threshold_computation_mode=self.model_config['threshold_computation_mode'],
                model=self.model
            )
            self.ad_manager.compute_threshold()
            self.ad_manager.save_thresholds_for_class(foundational=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Feature Anomaly Detection Trainer')
    parser.add_argument('--product_class', type=str, default='hazelnut', help='Product class to train on')
    parser.add_argument('--lr_scheduler', type=bool, default=True, help='Use learning rate scheduler')
    args = parser.parse_args()
    
    product_class = args.product_class
    lr_scheduler = args.lr_scheduler

    #product_class = "foundational"  # Change to 'foundational' to train on all classes for foundational model evaluation
    
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(parent, "config.yaml")

    experiment_name = f"deep_feature_{product_class}"
    experiment_dir = os.path.join("output", experiment_name)
    
    train_path = os.path.join(experiment_dir, "train")
    test_path = os.path.join(experiment_dir, "test")
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Initialize the trainer
    trainer = DeepFeatureADTrainer(
        config_path=config_path,
        train_path=train_path,
        test_path=test_path,
        product_class=product_class
    )
    
    
    # Train the model on the training dataset
    if product_class == 'foundational':
        trainer.train_foundational(scheduler=lr_scheduler)  # Train on all foundational classes if product_class is 'foundational'
        
        # Save the model weights after training on all foundational classes
        trainer.save_foundational_model()   # already called if not 'foundational', not called if 'foundational'
        # Compute the thresholds for anomaly detection
        # If product_class is 'foundational', compute thresholds for all classes
        trainer.compute_foundational_thresholds()
    """else:
        trainer.train_single_class()
        # Compute thresholds for the single class
        trainer.compute_threshold()"""
    
    # load model weights if available
    model_dir = os.path.join(train_path, 'checkpoints')
    # os.makedirs(model_dir, exist_ok=True)

    if product_class == 'foundational':
        weight_path = os.path.join(model_dir, f"wood_dfad_weights.pth")
        config = yaml.safe_load(open(config_path, 'r'))
        for product_class in config['FOUNDATIONAL_OBJECTS']:
            # Initialize the trainer with product_class
            trainer = DeepFeatureADTrainer(
                config_path=config_path,
                train_path=train_path,
                test_path=test_path,
                product_class=product_class
            )
            trainer.load_model_weights(weight_path=weight_path)
            threshold_file =f"{product_class}_foundational_thresholds.yaml" 
            threshold_file = os.path.join(train_path, threshold_file)
            trainer.load_computed_thresholds(threshold_file=threshold_file)
            trainer.generate_segmentation_maps(foundational=True)
            
    else:
        weight_path = os.path.join(model_dir, f"{product_class}_dfad_weights.pth")
        trainer = DeepFeatureADTrainer(
            config_path=config_path,
            train_path=train_path,
            test_path=test_path,
            product_class=product_class
        )
        trainer.train_single_class()
        trainer.load_model_weights(weight_path=weight_path)
        trainer.compute_threshold()  # Compute the threshold for anomaly detection
        threshold_file = os.path.join(train_path, f"{product_class}_thresholds.yaml")
        trainer.load_computed_thresholds(threshold_file=threshold_file)  # Load the computed thresholds
        trainer.generate_segmentation_maps()  # Generate segmentation maps for the test dataset
        trainer.plot_anomalies_thresholds()  # Plot the anomalies and thresholds

        
        
    # trainer.compute_threshold()
    # trainer.plot_anomalies_thresholds()
    
    #threshold_file = 'hazelnut_thresholds_after_training.yaml'
    #threshold_file = os.path.join(train_path, threshold_file)
    
    #trainer.load_computed_thresholds(threshold_file=threshold_file)
    
    #trainer.generate_segmentation_maps()

    """# Initialize the manager
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
    manager.generate_segmentation_maps(num_examples=5)"""