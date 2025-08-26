import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
import yaml
from einops import rearrange
from PIL import Image
from sklearn.metrics import (auc, precision_recall_curve, roc_auc_score,
                             roc_curve)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm

import vit_model.mdn1 as mdn1
import vit_model.model_res18 as M
import vit_model.pytorch_ssim as pytorch_ssim
import vit_model.spatial as S
from dataset_preprocesser import MVTecAD2
from vit_model.mdn1 import add_noise
from vit_model.student_transformer import ViT
from vit_model.utility_fun import *


class ViTManager:
    def __init__(self, product_class, config_path, train_path, test_path):
        self.config_path = config_path
        self.train_path = train_path
        self.test_path = test_path
        self.product_class = product_class

        self.ssim_loss = pytorch_ssim.SSIM()  # SSIM Loss
        # Load configuration from config.yaml
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.model_config = self.config["MODELS_CONFIG"]
        self.model_config = self.model_config["vit_autoencoder"]

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model, loss function, and optimizer
        self.model = VT_AE().cuda()
        self.G_estimate = mdn1.MDN().cuda()
        # for param in self.model.backbone.parameters():
        #     param.requires_grad = False

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.model_config['learning_rate']))
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.G_estimate.parameters()),
            lr=float(self.model_config["learning_rate"]),
            weight_decay=0.0001,
        )
        self.transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor()]
        )
        # Initialize cosine annealing learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.model_config.get("num_epochs")),
            eta_min=float(self.model_config.get("min_lr")),
        )
        patience = int(self.model_config.get("patience"))
        delta = float(self.model_config.get("delta"))
        self.early_stopping = EarlyStopping(
            patience=patience, delta=delta, verbose=False
        )
        # Initialize the datasets
        self.train_dataset = MVTecAD2(
            self.product_class, "train", self.train_path, transform=self.transform
        )
        self.test_dataset = MVTecAD2(
            self.product_class, "test", self.test_path, transform=self.transform
        )
        print("learning_rate", float(self.model_config["learning_rate"]))

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
        batch_size = int(self.model_config.get("batch_size"))
        num_epochs = int(self.model_config.get("num_epochs"))
        validation_split = float(self.model_config.get("validation_split"))
        num_workers = int(self.model_config.get("num_workers"))
        print(f"Batch size: {batch_size}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Validation split: {validation_split}")
        print(f"Training path: {self.train_path}")

        # Split the dataset into training and validation subsets
        val_size = int(validation_split * len(self.train_dataset))
        train_size = len(self.train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size]
        )
        print(
            f"Training on {len(train_subset)} samples, validating on {len(val_subset)} samples."
        )

        # Create DataLoaders for training and validation
        self.train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, num_workers=num_workers
        )

        # Training and validation loop
        print("Starting training...")
        print(summary(self.model, input_size=(batch_size, 3, 512, 512)))
        print(summary(self.model, input_size=(batch_size, 3, 512, 512)))
        best_val = float("inf")
        best_epoch = 0
        for epoch in range(num_epochs):
            # Training phase: set model to training mode
            self.model.train()
            self.G_estimate.train()
            epoch_loss = 0.0
            for batch in tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} - Training",
                leave=False,
            ):

                self.model.zero_grad()
                # Transfer input images to the device
                inputs = batch["sample"].to(self.device)
                # Forward pass: compute reconstructed images
                vector, reconstructions = self.model(inputs)
                pi, mu, sigma = self.G_estimate(vector)

                # Loss calculations
                loss1 = F.mse_loss(
                    reconstructions, inputs, reduction="mean"
                )  # Rec Loss
                loss2 = -self.ssim_loss(
                    inputs, reconstructions
                )  # SSIM loss for structural similarity
                loss3 = mdn1.mdn_loss_function(
                    vector, mu, sigma, pi
                )  # MDN loss for gaussian approximation

                loss = 5 * loss1 + 0.5 * loss2 + loss3  # Total loss
                # loss = 5*loss1 + 0.5*loss2 + loss3/(vector.shape[0]*vector.shape[1]*vector.shape[2])       #Total loss

                # Tensorboard definitions
                writer.add_scalar("TRAIN/recon-loss", loss1.item(), epoch)
                writer.add_scalar("TRAIN/ssim loss", loss2.item(), epoch)
                writer.add_scalar("TRAIN/Gaussian loss", loss3.item(), epoch)
                writer.add_histogram("TRAIN/Vectors", vector, epoch)

                # Backward pass: compute gradients and update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate loss for the epoch
                epoch_loss += loss.item()

            # Update learning rate scheduler after each epoch
            self.scheduler.step()
            # Tensorboard definitions for the mean epoch values
            avg_train_loss = epoch_loss / len(self.train_loader)
            writer.add_scalar("TRAIN/Loss", avg_train_loss, epoch + 1)

            # Validation phase: set model to evaluation mode
            self.model.eval()
            self.G_estimate.eval()
            val_loss = 0.0
            reconstruction_errors = []
            with torch.inference_mode():
                for batch in tqdm(
                    self.val_loader,
                    desc=f"Epoch {epoch + 1}/{num_epochs} - Validation",
                    leave=False,
                ):
                    # Transfer validation images to the device
                    inputs = batch["sample"].to(self.device)

                    # Forward pass on validation data
                    vector, reconstructions = self.model(inputs)
                    pi, mu, sigma = self.G_estimate(vector)

                    # Loss calculations
                    loss1 = F.mse_loss(
                        reconstructions, inputs, reduction="mean"
                    )  # Rec Loss
                    loss2 = -self.ssim_loss(
                        inputs, reconstructions
                    )  # SSIM loss for structural similarity
                    loss3 = mdn1.mdn_loss_function(
                        vector, mu, sigma, pi, test=False
                    )  # MDN loss for gaussian approximation
                    loss = 5 * loss1 + 0.5 * loss2 + loss3  # Total loss
                    # loss = 5*loss1 + 0.5*loss2 + loss3/(vector.shape[0]*vector.shape[1]*vector.shape[2])      #Total loss

                    # Compute the difference vector at each spatial location
                    diff = inputs - reconstructions  # shape (B, C, H, W)

                    # Calculate the pixel-level anomaly map by computing the L2 norm across channels (for each pixel)
                    anomaly_map = torch.linalg.norm(
                        diff, dim=1
                    )  # shape (B, C, H, W) -> (B, H, W)

                    # Pool the anomaly map of shape (B, 16, 16) to a single value per image using adaptive max pooling.
                    img_anomaly_score = torch.nn.functional.adaptive_max_pool2d(
                        anomaly_map, (1, 1)
                    ).reshape(anomaly_map.shape[0])

                    reconstruction_errors.extend(img_anomaly_score.cpu().numpy())

                    writer.add_scalar("VAL/recon-loss", loss1.item(), epoch)
                    writer.add_scalar("VAL/ssim loss", loss2.item(), epoch)
                    writer.add_scalar("VAL/Gaussian loss", loss3.item(), epoch)
                    writer.add_histogram("VAL/Vectors", vector, epoch)

                    # Accumulate validation loss for the epoch
                    val_loss += loss.item()
            self.early_stopping.check_early_stop(val_loss)
            avg_val_loss = val_loss / len(self.val_loader)
            mean_rec_error = torch.tensor(reconstruction_errors).mean().item()
            std_rec_error = torch.tensor(reconstruction_errors).std().item()
            print("==========================================")
            print(
                f"Epoch: {epoch + 1}/{num_epochs} || Train | Loss: {avg_train_loss:>.6f} || Val | Loss: {avg_val_loss:>.6f} | MSE-Mean: {mean_rec_error:>.6f} | MSE-Std: {std_rec_error:>.6f}"
            )
            writer.add_scalar("VAL/Loss", avg_val_loss, epoch + 1)
            writer.add_scalar("Reconstruction/Mean", mean_rec_error, epoch + 1)
            writer.add_scalar("Reconstruction/Std", std_rec_error, epoch + 1)
            writer.add_image(
                "Reconstructed Image",
                utils.make_grid(reconstructions),
                epoch,
                dataformats="CHW",
            )
            writer.add_image(
                "Original Image", utils.make_grid(inputs), epoch, dataformats="CHW"
            )

            # # Save the best epoch based on the lowest validation reconstruction mean error
            # if mean_rec_error < best_val:
            #     best_val = mean_rec_error
            #     best_epoch = epoch + 1
            #     torch.save(self.model.state_dict(), os.path.join(self.train_path, "vit_weights.pth"))
            #     torch.save(self.G_estimate.state_dict(), os.path.join(self.train_path, "g_weights.pth"))
            #     print(f"Best model updated at Epoch {best_epoch} with MSE-Mean: {mean_rec_error:>.6f}")

            # Save the best epoch based on the lowest Train loss
            if avg_train_loss < best_val:
                best_val = avg_train_loss
                best_epoch = epoch + 1
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.train_path, "vit_weights.pth"),
                )
                torch.save(
                    self.G_estimate.state_dict(),
                    os.path.join(self.train_path, "g_weights.pth"),
                )
                print(
                    f"Best model updated at Epoch {best_epoch} with Train-Loss: {avg_train_loss:>.6f}"
                )
            # self.early_stopping.check_early_stop(mean_rec_error)
            # if self.early_stopping.stop_training:
            #     print(f"Early stopping at epoch {epoch}")
            #     break

        writer.close()
        print("Training completed.")

    def test(self, upsample=1):

        norm_loss_t = []
        normalised_score_t = []
        mask_score_t = []
        loss1_tn = []
        loss2_tn = []
        loss3_tn = []
        loss1_ta = []
        loss2_ta = []
        loss3_ta = []

        score_tn = []
        score_ta = []
        patch_size = 64
        batch_size = int(self.model_config.get("batch_size"))
        num_workers = int(self.model_config.get("num_workers"))
        test_loader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, num_workers=num_workers
        )

        vit_weights_path = os.path.join(self.train_path, "vit_weights.pth")
        g_weights_path = os.path.join(self.train_path, "g_weights.pth")

        self.model_test = VT_AE(train=False).cuda()
        self.model_test.load_state_dict(
            torch.load(vit_weights_path, map_location=self.device)
        )
        self.model_test.eval()

        self.G_estimate_test = mdn1.MDN().cuda()
        self.G_estimate.load_state_dict(
            torch.load(g_weights_path, map_location=self.device)
        )
        self.G_estimate.eval()
        stats_path = os.path.join(self.train_path, "training_statistics.yaml")
        with open(stats_path, "r") as file:
            stats = yaml.safe_load(file)
        threshold = float(stats["threshold"])

        t_loss_all_normal = []
        t_loss_all_anomaly = []
        for c, el in enumerate(test_loader):
            # Get the input image and move to device. Add a batch dimension.
            sample = el["sample"].to(self.device)
            mask = el["ht"].to(self.device)
            n = np.array(["bad" in path for path in el["image_path"]], dtype=int)[0]

            vector, reconstructions = self.model_test(sample)
            pi, mu, sigma = self.G_estimate_test(vector)

            # Loss calculations
            loss1 = F.mse_loss(reconstructions, sample, reduction="mean")  # Rec Loss
            loss2 = -self.ssim_loss(
                sample, reconstructions
            )  # SSIM loss for structural similarity
            loss3 = mdn1.mdn_loss_function(
                vector, mu, sigma, pi, test=True
            )  # MDN loss for gaussian approximation
            loss = loss1 - loss2 + loss3.max()  # Total loss
            norm_loss_t.append(loss3.detach().cpu().numpy())

            if n == 0:
                loss1_tn.append(loss1.detach().cpu().numpy())
                loss2_tn.append(loss2.detach().cpu().numpy())
                loss3_tn.append(loss3.sum().detach().cpu().numpy())
                t_loss_all_normal.append(loss.detach().cpu().numpy())
            if n == 1:
                loss1_ta.append(loss1.detach().cpu().numpy())
                loss2_ta.append(loss2.detach().cpu().numpy())
                loss3_ta.append(loss3.sum().detach().cpu().numpy())
                t_loss_all_anomaly.append(loss.detach().cpu().numpy())

            if upsample == 0:
                # Mask patch
                mask_patch = rearrange(
                    mask.squeeze(0).squeeze(0),
                    "(h p1) (w p2) -> (h w) p1 p2",
                    p1=patch_size,
                    p2=patch_size,
                )
                mask_patch_score = Binarization(mask_patch.sum(1).sum(1), 0.0)
                mask_score_t.append(mask_patch_score)  # Storing all masks
                norm_score = Binarization(norm_loss_t[-1], threshold)
                m = torch.nn.UpsamplingNearest2d((512, 512))
                score_map = m(
                    torch.tensor(
                        norm_score.reshape(-1, 1, 512 // patch_size, 512 // patch_size)
                    )
                )

                normalised_score_t.append(norm_score)  # Storing all patch scores
            elif upsample == 1:
                mask_score_t.append(
                    mask.squeeze(0).squeeze(0).cpu().numpy()
                )  # Storing all masks

                m = torch.nn.UpsamplingBilinear2d((512, 512))
                norm_score = norm_loss_t[-1].reshape(
                    -1, 1, 512 // patch_size, 512 // patch_size
                )
                score_map = m(torch.tensor(norm_score))
                score_map = Filter(score_map, type=0)

                normalised_score_t.append(score_map)  # Storing all score maps

            # ## Plotting
            # if c%5 == 0:
            #     plot(sample.cpu(), mask.cpu() ,score_map[0][0])
            # if n == 0:
            #     score_tn.append(score_map.max())
            # if n ==1:
            #     score_ta.append(score_map.max())

        ## PRO Score
        scores = np.asarray(normalised_score_t).flatten()
        masks = np.asarray(mask_score_t).flatten()
        masks = (masks > 0.5).astype(int)
        PRO_score = roc_auc_score(masks, scores)

        ## Image Anomaly Classification Score (AUC)
        roc_data = np.concatenate((t_loss_all_normal, t_loss_all_anomaly))
        roc_targets = np.concatenate(
            (np.zeros(len(t_loss_all_normal)), np.ones(len(t_loss_all_anomaly)))
        )
        AUC_Score_total = roc_auc_score(roc_targets, roc_data)

        # AUC Precision Recall Curve
        precision, recall, thres = precision_recall_curve(roc_targets, roc_data)
        AUC_PR = auc(recall, precision)

        print(f"PRO Score: {PRO_score}")
        print(f"AUC Score: {AUC_Score_total}")
        print(f"AUC PR Score: {AUC_PR}")
        return PRO_score, AUC_Score_total, AUC_PR

    # def compute_thresh(self):
    #     # Set the autoencoder to evaluation mode
    #     self.model.eval()
    #     anomaly_scores = []

    #     # Perform inference on the test dataset
    #     for el in tqdm(self.train_loader, desc="Processing train dataset"):
    #         # Get the input image and its path
    #         sample      = el["sample"].to(self.device)
    #         # Perform forward pass to get the reconstructed image
    #         with torch.inference_mode():
    #             reconstructed, features = self.model(sample)

    #         # Compute the difference vector at each spatial location
    #         diff = features - reconstructed  # shape (B, C, H, W)

    #         # Calculate the pixel-level anomaly map by computing the L2 norm across channels (for each pixel)
    #         anomaly_map = torch.linalg.norm(diff, dim=1)  # shape (B, C, H, W) -> (B, H, W)

    #         # Pool the anomaly map of shape (B, 16, 16) to a single value per image using adaptive max pooling.
    #         img_anomaly_score = torch.nn.functional.adaptive_max_pool2d(anomaly_map, (1, 1)).reshape(anomaly_map.shape[0])

    #         anomaly_scores.extend(img_anomaly_score.cpu().numpy())

    #     # Print the mean anomaly score
    #     print(f"Mean Anomaly Score: {np.mean(anomaly_scores)}")
    #     # Compute mean (μ) and standard deviation (σ) of anomaly scores
    #     mean_error = np.mean(anomaly_scores)
    #     std_error  = np.std(anomaly_scores)

    #     # Set threshold = μ + 3σ
    #     threshold = mean_error + 3 * std_error
    #     print(f"Mean Error (μ): {mean_error}")
    #     print(f"Standard Deviation (σ): {std_error}")
    #     print(f"Threshold: {threshold}")

    #     # Plot histogram of training errors
    #     plt.hist(anomaly_scores, bins=30, density=True, alpha=0.7, color='blue', label='Training Errors')
    #     plt.axvline(mean_error, color='green', linestyle='--', label='Mean (μ)')
    #     plt.axvline(threshold, color='red', linestyle='--', label='Threshold (μ + 3σ)')
    #     plt.title('Histogram of Training Errors')
    #     plt.xlabel('Error')
    #     plt.ylabel('Density')
    #     plt.legend()
    #     save_path = os.path.join(self.train_path, "training_errors_histogram.png")
    #     plt.savefig(save_path)
    #     plt.close()
    #     return mean_error, std_error, threshold

    def thresholding(self, upsample=1, thres_type=0, fpr_thres=0.3):
        """
        Parameters
        ----------
        data : TYPE, optional
            DESCRIPTION. The default is data.train_loader.
        upsample : INT, optional
            DESCRIPTION. 0 - NearestUpsample2d; 1- BilinearUpsampling.
        thres_type : INT, optional
            DESCRIPTION. 0 - 30% of fpr reached; 1 - thresholding using best F1 score
        fpr_thres : FLOAT, Optional
            DESCRIPTION. False Positive Rate threshold value. Default is 0.3

        Returns
        -------
        Threshold: Threshold value

        """
        norm_loss_t = []
        normalised_score_t = []
        mask_score_t = []
        patch_size = 64
        self.model.eval()
        self.G_estimate.eval()

        for el in tqdm(self.test_loader, desc="Processing test dataset"):

            images = el["sample"].to(self.device)
            mask = el["ht"].to(self.device)
            vector, reconstructions = self.model(images)
            pi, mu, sigma = self.G_estimate(vector)

            # Loss calculations
            loss1 = F.mse_loss(reconstructions, images, reduction="mean")  # Rec Loss
            loss2 = -self.ssim_loss(
                images, reconstructions
            )  # SSIM loss for structural similarity
            loss3 = mdn1.mdn_loss_function(
                vector, mu, sigma, pi, test=True
            )  # MDN loss for gaussian approximation
            loss = loss1 + loss2 + loss3.sum()  # Total loss
            norm_loss_t.append(loss3.detach().cpu().numpy())

            if upsample == 0:
                # Mask patch
                mask_patch = rearrange(
                    mask.squeeze(0).squeeze(0),
                    "(h p1) (w p2) -> (h w) p1 p2",
                    p1=patch_size,
                    p2=patch_size,
                )
                mask_patch_score = Binarization(mask_patch.sum(1).sum(1), 0.0)
                mask_score_t.append(mask_patch_score)  # Storing all masks
                norm_score = norm_loss_t[-1]
                normalised_score_t.append(norm_score)  # Storing all patch scores
            elif upsample == 1:
                mask_score_t.append(
                    mask.squeeze(0).squeeze(0).cpu().numpy()
                )  # Storing all masks
                m = torch.nn.UpsamplingBilinear2d((512, 512))
                norm_score = norm_loss_t[-1].reshape(
                    -1, 1, 512 // patch_size, 512 // patch_size
                )
                score_map = m(torch.tensor(norm_score))
                score_map = Filter(
                    score_map, type=0
                )  # add normalization here for the testing
                normalised_score_t.append(score_map)  # Storing all score maps

        scores = np.asarray(normalised_score_t).flatten()
        masks = np.asarray(mask_score_t).flatten()
        masks = (masks > 0.5).astype(int)
        if thres_type == 0:
            fpr, tpr, _ = roc_curve(masks, scores)
            fp3 = np.where(fpr <= fpr_thres)
            threshold = _[fp3[-1][-1]]
        elif thres_type == 1:
            precision, recall, thresholds = precision_recall_curve(masks, scores)
            a = 2 * precision * recall
            b = precision + recall
            f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            threshold = thresholds[np.argmax(f1)]
        return threshold

    def save_model(self, args, threshold):
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
            stats = {"threshold": float(threshold)}
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


class VT_AE(nn.Module):
    def __init__(
        self,
        image_size=512,
        patch_size=64,
        num_classes=1,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        train=True,
    ):

        super(VT_AE, self).__init__()
        self.vt = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
        )

        self.decoder = M.decoder2(8)
        self.Digcap = S.DigitCaps(
            in_num_caps=((image_size // patch_size) ** 2) * 8 * 8, in_dim_caps=8
        )
        self.mask = (
            torch.ones(1, image_size // patch_size, image_size // patch_size)
            .bool()
            .cuda()
        )
        self.Train = train

        if self.Train:
            print("\nInitializing network weights.........")
            initialize_weights(self.vt, self.decoder)

    def forward(self, x):
        b = x.size(0)
        encoded = self.vt(x, self.mask)
        if self.Train:
            encoded = add_noise(encoded)
        encoded1, vectors = self.Digcap(encoded.view(b, encoded.size(1) * 8 * 8, -1))
        recons = self.decoder(encoded1.view(b, -1, 8, 8))

        return encoded, recons


# Initialize weight function
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

