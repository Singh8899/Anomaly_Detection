import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm

from dataset_preprocesser import MVTecAD2


class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor, self).__init__()
        self.model = resnet50(weights=("DEFAULT"))                         
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        def hook(model, input, output):
            self.features.append(output)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def forward(self, x):
        self.features = []
        
        with torch.no_grad():
            _ = self.model(x)
        self.avg = nn.AvgPool2d(3, stride=1)
        self.shape = self.features[0].shape[-2]
        self.resize = nn.AdaptiveAvgPool2d(self.shape)

        resized_patches = [self.resize(self.avg(f)) for f in self.features]
        resized_patches = torch.cat(resized_patches, dim=1)
        patches = resized_patches.reshape(resized_patches.shape[1],  -1).T

        return patches

# # Optional: suppress warnings if you want cleaner logs
# import warnings
# warnings.filterwarnings("ignore")

class PatchCoreManager():
    def __init__(self, product_class, config_path, train_path, test_path):
        self.product_class = product_class
        self.config_path = config_path
        self.train_path = train_path
        self.test_path = test_path

        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Get PatchCore specific config
        self.model_config = self.config["MODELS_CONFIG"].get("patchcore", {})
        self.subsample_ratio = self.model_config.get("memory_bank_subsample_ratio", 0.1)
        self.threshold_multiplier = self.model_config.get("threshold_multiplier", 2.0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.feature_extractor = Feature_extractor().to(self.device)
        
        # Initialize datasets
        self.train_dataset = MVTecAD2(self.product_class, "train", self.config["DATASET_PATH"], self.transform)
        self.test_dataset = MVTecAD2(self.product_class, "test", self.config["DATASET_PATH"], transform=self.transform)
        
        # Initialize variables for training results
        self.memory_bank = None
        self.sub_memory_bank = None
        self.threshold = None
        self.train_scores = None
        
        # Check if directories exist
        self._check_directories()

    def train(self):
        """Extract features from training data and build memory bank"""
        print(f"\n=== Training PatchCore for class: {self.product_class} ===")
        
        # Create training output directory
        train_output_dir = os.path.join(self.train_path, f"train_patchcore_{self.product_class}")
        os.makedirs(train_output_dir, exist_ok=True)
        self.train_output_dir = train_output_dir
        
        # Extract features from training data
        memory_bank = []
        for x in tqdm(self.train_dataset, desc=f"[{self.product_class}] Feature Extraction (Train)", total=len(self.train_dataset)):
            with torch.no_grad():
                image = x["sample"].to(self.device)
                patches = self.feature_extractor(image.unsqueeze(0))
                memory_bank.append(patches.detach())

        self.memory_bank = torch.cat(memory_bank, dim=0)
        
        # Subsample memory bank for efficiency
        subsample_size = int(self.memory_bank.shape[0] * self.subsample_ratio)
        selected_patches = np.random.choice(self.memory_bank.shape[0], size=subsample_size, replace=False)
        self.sub_memory_bank = self.memory_bank[selected_patches]
        
        print(f"Memory bank created with {self.memory_bank.shape[0]} patches")
        print(f"Subsampled to {self.sub_memory_bank.shape[0]} patches")

    def compute_thresh(self):
        """Compute threshold based on training data scores"""
        if self.sub_memory_bank is None:
            raise ValueError("Must call train() before compute_thresh()")
            
        print("Computing threshold from training data...")
        
        y_score_max = []
        for x in tqdm(self.train_dataset, desc=f"[{self.product_class}] Scoring (Train)", total=len(self.train_dataset)):
            with torch.no_grad():
                image = x["sample"].to(self.device)
                patches = self.feature_extractor(image.unsqueeze(0))
                distances = torch.cdist(patches, self.sub_memory_bank)
                dist_score, _ = torch.min(distances, dim=1)
                y_score_max.append(dist_score.max().item())

        self.train_scores = y_score_max
        mean = np.mean(y_score_max)
        std = np.std(y_score_max)
        self.threshold = mean + self.threshold_multiplier * std

        print(f"Mean training score: {mean:.4f}")
        print(f"Std training score: {std:.4f}")
        print(f"Threshold: {self.threshold:.4f}")

        # Save histogram
        plt.figure()
        plt.hist(y_score_max, bins=30, alpha=0.7, color="blue", label="Training Scores")
        plt.axvline(mean, color='green', linestyle='dashed', linewidth=1, label='Mean')
        plt.axvline(self.threshold, color='r', linestyle='dashed', linewidth=1, label='Threshold')
        plt.title(f"Training Scores Histogram - {self.product_class}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(os.path.join(self.train_output_dir, f"training_errors_histogram.png"))
        plt.close()

        return mean, std, self.threshold

    def test(self):
        """Test the model on test dataset"""
        if self.sub_memory_bank is None or self.threshold is None:
            # Load from saved model if not trained in this session
            self.load_model()
        
        print(f"\n=== Testing PatchCore for class: {self.product_class} ===")
        
        # Create test output directory
        test_output_dir = os.path.join(self.test_path, f"test_patchcore_{self.product_class}")
        os.makedirs(test_output_dir, exist_ok=True)
        
        y_test_score = []
        y_test_true = []
        seg_maps = []

        for idx, x in enumerate(tqdm(self.test_dataset, desc=f"[{self.product_class}] Inference (Test)", total=len(self.test_dataset))):
            with torch.no_grad():
                image = x["sample"].to(self.device)
                patches = self.feature_extractor(image.unsqueeze(0))

                distances = torch.cdist(patches, self.sub_memory_bank)
                dist_score, _ = torch.min(distances, dim=1)
                seg_map = dist_score.view(1, 1, 28, 28)
                seg_maps.append(seg_map)
                y_test_score.append(dist_score.max().item())
                label = Path(x["image_path"]).parent.name
           
                y_test_true.append(0 if label == "good" else 1)

                # Save sample segmentation maps (e.g. first 5)
                if idx < 5:
                    # Save per-sample comparison visualization
                    interpolated_map = nn.functional.interpolate(seg_map, size=(224, 224), mode='bilinear')
                    binary_map = (interpolated_map > self.threshold * 1.25).float()

                    original = x["sample"].permute(1, 2, 0).cpu().numpy()
                    gt_map = x["ht"].squeeze().cpu().numpy()
                    pred_map = binary_map.squeeze().cpu().numpy()

                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                    axs[0].imshow(original)
                    axs[0].set_title("Original")
                    axs[0].axis("off")

                    axs[1].imshow(gt_map, cmap="gray")
                    axs[1].set_title("Ground Truth")
                    axs[1].axis("off")

                    axs[2].imshow(pred_map, cmap="gray")
                    axs[2].set_title("Prediction")
                    axs[2].axis("off")

                    plt.tight_layout()
                    plt.savefig(os.path.join(test_output_dir, f"sample_{idx}_comparison.png"))
                    plt.close()

        auc_roc_score = roc_auc_score(y_test_true, y_test_score)
        print(f"[{self.product_class}] AUC ROC Score: {auc_roc_score:.4f}")

        with open(os.path.join(test_output_dir, f"{self.product_class}_metrics.txt"), "w") as f:
            f.write(f"AUC ROC Score: {auc_roc_score:.4f}\n")
            f.write(f"Threshold used: {self.threshold:.4f}\n")

        return auc_roc_score

    def save_model(self, args, mean_error, std_error, threshold):
        """Save model weights and statistics"""
        # Save the memory bank
        memory_bank_path = os.path.join(self.train_output_dir, "memory_bank.pth")
        torch.save({
            'memory_bank': self.memory_bank,
            'sub_memory_bank': self.sub_memory_bank,
            'threshold': threshold,
            'train_scores': self.train_scores
        }, memory_bank_path)
        print(f"Memory bank saved to {memory_bank_path}")

        # Save the model configuration  
        config_save_path = os.path.join(self.train_output_dir, "config.yaml")
        with open(config_save_path, "w") as file:
            yaml.dump(self.config, file)
        print(f"Configuration saved to {config_save_path}")

        # Save the arguments
        args_save_path = os.path.join(self.train_output_dir, "args.yaml")
        with open(args_save_path, "w") as file:
            yaml.dump(vars(args), file)
        print(f"Arguments saved to {args_save_path}")

        # Save the training statistics
        stats_save_path = os.path.join(self.train_output_dir, "training_statistics.yaml")
        with open(stats_save_path, "w") as file:
            stats = {
                "mean_error": float(mean_error),
                "std_error": float(std_error),
                "threshold": float(threshold),
            }
            yaml.dump(stats, file)
        print(f"Training statistics saved to {stats_save_path}")

    def load_model(self):
        """Load saved model for testing"""
        train_output_dir = os.path.join(self.train_path, f"train_patchcore_{self.product_class}")
        memory_bank_path = os.path.join(train_output_dir, "memory_bank.pth")
        stats_path = os.path.join(train_output_dir, "training_statistics.yaml")
        
        if os.path.exists(memory_bank_path):
            checkpoint = torch.load(memory_bank_path, map_location=self.device)
            self.memory_bank = checkpoint['memory_bank']
            self.sub_memory_bank = checkpoint['sub_memory_bank']
            self.threshold = checkpoint['threshold']
            self.train_scores = checkpoint.get('train_scores', None)
            print(f"Model loaded from {memory_bank_path}")
        else:
            raise FileNotFoundError(f"No saved model found at {memory_bank_path}")

    def _check_directories(self):
        """Check if required directories exist"""
        dataset_path = self.config.get("DATASET_PATH", "")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        product_train_path = os.path.join(dataset_path, self.product_class, "train")
        product_test_path = os.path.join(dataset_path, self.product_class, "test")
        
        if not os.path.exists(product_train_path):
            raise FileNotFoundError(f"Training data path not found: {product_train_path}")
        
        if not os.path.exists(product_test_path):
            print(f"Warning: Test data path not found: {product_test_path}")

    def train_test(self):
        """Legacy method for backward compatibility - runs both training and testing"""
        classes = [self.product_class] if self.product_class != "all" else sorted(
            [d for d in os.listdir(self.train_path) if os.path.isdir(os.path.join(self.train_path, d))])

        for cls in classes:
            # Create temporary manager for each class if processing all
            if cls != self.product_class:
                temp_manager = PatchCoreManager(cls, self.config_path, self.train_path, self.test_path)
                temp_manager.train()
                temp_manager.compute_thresh()
                temp_manager.test()
            else:
                self.train()
                self.compute_thresh()
                self.test()


# Remove the standalone execution code at the bottom
# c = PatchCoreManager(
#     product_class="hazelnut",
#     config_path="config.yaml", 
#     train_path="train",
#     test_path="test"
# )
# c.train_test()