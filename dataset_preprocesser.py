import os
import glob
import torch
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor


# Load configuration from config.yaml
config_path = "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

DATASET_PATH = config.get("DATASET_PATH", "")
DATASET_OBJECTS = config.get("DATASET_OBJECTS", [])


class MVTecAD2(Dataset):
    """Dataset class for MVTec AD 2 objects.
    Args:
        mad2_object (str): can, fabric,ecc. or 'all' for all objects
        split (str): train, test
        transform (function, optional): transform applied to samples, defaults to 'to_tensor'
    """

    def __init__(
        self,
        mad2_object,
        split,
        output_dir,
        transform=to_tensor,
    ):

        assert split in {"train", "test"}, f"unknown split: {split}"

        assert mad2_object in (
            DATASET_OBJECTS + ["all"]
        ), f"unknown MVTec AD 2 object: {mad2_object}"

        self.output_dir = output_dir
        self.object = mad2_object
        self.split = split
        self.transform = transform

        self._image_base_dir = DATASET_PATH
        # get all images from the split
        if self.object == "all":
            self._image_paths = []
            for obj in DATASET_OBJECTS:
                object_dir = os.path.join(self._image_base_dir, obj)
                self._image_paths.extend(sorted(self._get_pattern(object_dir)))
            temp_split = self.split
            self.split = "validation"
            for obj in DATASET_OBJECTS:
                object_dir = os.path.join(self._image_base_dir, obj)
                self._image_paths.extend(sorted(self._get_pattern(object_dir)))
            self.split = temp_split
        # get images for a particular object
        else:
            object_dir = os.path.join(self._image_base_dir, mad2_object)
            # get all images from the split
            self._image_paths = sorted(self._get_pattern(object_dir))

    def _get_pattern(self, object_dir) -> list[str]:
        # Build two patterns: one for 'good', one for 'bad'
        patterns = [
            os.path.join(object_dir, self.split, "good", "**", "*.png"),
            os.path.join(object_dir, self.split, "bad", "**", "*.png"),
        ]
        all_matches = []
        for pattern in patterns:
            matches = glob.glob(pattern, recursive=True)
            if not matches:
                # Fallback to one level deep (no '**')
                fallback_pattern = pattern.replace("**" + os.sep, "")
                matches = glob.glob(fallback_pattern, recursive=True)
            all_matches.extend(matches)
        return all_matches

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Get dataset item for the index ``idx``.
        Args:
            idx (int): Index to get the item.
        Returns:
            dict[str,  str | torch.Tensor]: Dict containing the sample image,
            image path, and the relative anomaly image output path for both
            image types continuous and thresholded.
        """

        image_path = self._image_paths[idx]
        sample = default_loader(image_path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.split == "test":
            return {
                "sample": sample,
                "ht": self.get_gt_image(idx),
                "image_path": image_path,
                "rel_out_path_cont": self.get_relative_anomaly_image_out_path(idx),
                "rel_out_path_thresh": self.get_relative_anomaly_image_out_path(
                    idx, True
                ),
            }
        else:
            return {
                "sample": sample,
                "image_path": image_path,
                "rel_out_path_cont": self.get_relative_anomaly_image_out_path(idx),
                "rel_out_path_thresh": self.get_relative_anomaly_image_out_path(
                    idx, True
                ),
            }

    @property
    def image_paths(self):
        return self._image_paths

    @property
    def has_segmentation_gt(self) -> bool:
        return self.split == "test"

    def get_relative_anomaly_image_out_path(self, idx, thresholded=False):
        """Returns a path relative to the experiment directory
        for storing the (thresholded) anomaly image in the required structure.
        Args:
            idx (int): sample index
            thresholded (bool): return output path for thresholded image,
            defaults to 'False'
        Returns:
            str: relative output path to write anomaly image
        """

        image_path = Path(self._image_paths[idx])
        relpath = image_path.relative_to(self._image_base_dir)

        if not thresholded:
            base_dir = "anomaly_images"
            suffix = ".tiff"
        else:
            base_dir = "anomaly_images_thresholded"
            suffix = ".png"

        return os.path.join(self.output_dir, base_dir, relpath.with_suffix(suffix))

    def get_gt_image(self, idx):
        """Returns the ground truth image where values of 255 denote
        anomalous pixels and values of 0 anomaly-free ones. For good images,
        an all-zero pixel image is returned.
        In case no segmentation ground truth is available
        (test_private/test_private_mixed), an all-zero pixel image is returned as well.
        Args:
            idx (int): sample index
        Returns:
            torch.Tensor: ground truth image if available, otherwise an all-zero pixel image
        """
        image_path = self.image_paths[idx]
        if (
            self.has_segmentation_gt
            and "good" not in self.get_relative_anomaly_image_out_path(idx)
        ):
            base_path, file_name = image_path.split("/bad/")
            gt_image_path = os.path.join(
                base_path, "ground_truth/bad", file_name
            ).replace(".png", "_mask.png")

            gt_image_pil = Image.open(gt_image_path)
            gt_image = np.array(gt_image_pil)
        else:
            # Create an all-zero pixel image with the same dimensions as the input image
            sample_image = Image.open(image_path)
            gt_image = np.zeros(sample_image.size[::-1], dtype=np.uint8)

        # Convert the ground truth image to a tensor
        gt_tensor = torch.from_numpy(gt_image).unsqueeze(0).float() / 255.0

        # Apply the transform if it exists
        if self.transform is not None:
            # Check if the transform contains Resize or CenterCrop
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize) or isinstance(
                    t, transforms.CenterCrop
                ):
                    gt_tensor = t(gt_tensor)

        return gt_tensor
