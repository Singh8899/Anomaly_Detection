import os
import yaml
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path

from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

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

# Optional: suppress warnings if you want cleaner logs
import warnings
warnings.filterwarnings("ignore")
class PatchCoreManager():
    def __init__(self, product_class, config_path, train_path, test_path):
        self.product_class = product_class
        self.config_path = config_path
        self.train_path = train_path
        self.test_path = test_path

        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.feature_extractor = Feature_extractor().to(self.device)

    def train_test(self):
        classes = [self.product_class] if self.product_class != "all" else sorted(
            [d for d in os.listdir(self.train_path) if os.path.isdir(os.path.join(self.train_path, d))])

        for cls in classes:
            print(f"\n=== Processing class: {cls} ===")
            save_dir = os.path.join(self.test_path, f"{cls}_results")
            os.makedirs(save_dir, exist_ok=True)

            train_dataset = MVTecAD2(cls, "train", self.train_path, self.transform)
            test_dataset = MVTecAD2(cls, "test", self.test_path, transform=self.transform)
            
            MEMORY_BANK = []
            for x in tqdm(train_dataset, desc=f"[{cls}] Feature Extraction (Train)", total=len(train_dataset)):
                with torch.no_grad():
                    image = x["sample"].to(self.device)
                    patches = self.feature_extractor(image.unsqueeze(0))
                    MEMORY_BANK.append(patches.detach())

            MEMORY_BANK = torch.cat(MEMORY_BANK, dim=0)
            selected_patches = np.random.choice(MEMORY_BANK.shape[0], size=MEMORY_BANK.shape[0] // 10, replace=False)
            sub_MEMORY_BANK = MEMORY_BANK[selected_patches]

            y_score_max = []
            for x in tqdm(train_dataset, desc=f"[{cls}] Scoring (Train)", total=len(train_dataset)):
                with torch.no_grad():
                    image = x["sample"].to(self.device)
                    patches = self.feature_extractor(image.unsqueeze(0))
                    distances = torch.cdist(patches, sub_MEMORY_BANK)
                    dist_score, _ = torch.min(distances, dim=1)
                    y_score_max.append(dist_score.max().item())

            mean = np.mean(y_score_max)
            std = np.std(y_score_max)
            threshold = mean + 2 * std

            # Save histogram
            plt.figure()
            plt.hist(y_score_max, bins=10)
            plt.axvline(threshold, color='r', linestyle='dashed', linewidth=1)
            plt.title(f"Training Scores Histogram - {cls}")
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(save_dir, f"{cls}_histogram.png"))
            plt.close()

            y_test_score = []
            y_test_true = []
            seg_maps = []

            for idx, x in enumerate(tqdm(test_dataset, desc=f"[{cls}] Inference (Test)", total=len(test_dataset))):
                with torch.no_grad():
                    image = x["sample"].to(self.device)
                    patches = self.feature_extractor(image.unsqueeze(0))

                    distances = torch.cdist(patches, sub_MEMORY_BANK)
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
                        binary_map = (interpolated_map > threshold * 1.25).float()

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
                        plt.savefig(os.path.join(save_dir, f"sample_{idx}_comparison.png"))
                        plt.close()

            auc_roc_score = roc_auc_score(y_test_true, y_test_score)
            print(f"[{cls}] AUC ROC Score: {auc_roc_score:.4f}")

            with open(os.path.join(save_dir, f"{cls}_metrics.txt"), "w") as f:
                f.write(f"AUC ROC Score: {auc_roc_score:.4f}\n")
                f.write(f"Threshold used: {threshold:.4f}\n") 


c = PatchCoreManager(
    product_class="hazelnut",
    config_path="config.yaml",
    train_path="train",
    test_path="test"
)
c.train_test()