import os.path as osp
import random

import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


from src.utils import get_task


class FundusDataset(torch.utils.data.Dataset):
    """Dataset for fundus dataset.

    Attributes:
        home_dir: home directory of image
        data: dataframe of fundus dataset
        transform: transform for fundus image
        tasks: list of tasks
        pair_sampling: whether to use pair sampling

    Args:
        home_dir: home directory for the image data
        csv_files: list of csv files
        img_size: image size
        use_kornia: whether to use kornia
        task_names: list of labels for use
        drop_na_labels: whether to drop na labels
        self_training: dict which have csv_files, mode, threshold, temperature as keys
        pair_sampling: whether to use pair sampling
        train: whether to use training mode
        tta: whether to use test time augmentation

    Methods:
        __init__: initialize dataset
        _read_image: read image from gcs
        __getitem__: return image and labels
        __len__: return length of dataset
    """

    def __init__(
        self,
        home_dir: str,
        csv_files: list[str],
        img_size: int,
        use_kornia: bool,
        task_names: list[str],
        drop_na_labels: bool,
        resolution_dir: str,
        wide_resolution_dir: str | None = None,
        self_training: dict | None = None,
        pair_sampling: bool = False,
        train: bool = False,
        tta: bool = False,
    ) -> None:
        super().__init__()

        self.pair_sampling = pair_sampling
        self.home_dir = home_dir
        self.data = pd.concat([pd.read_csv(csv_file, encoding="cp949") for csv_file in csv_files])
        self.data["img_path"] = self.data[resolution_dir]

        if wide_resolution_dir:
            # Use .loc to filter rows where camera_type is "Wide" and then assign new values to img_path
            self.data.loc[self.data["camera_type"] == "Wide", "img_path"] = self.data.loc[
                self.data["camera_type"] == "Wide", "img_path"
            ].str.replace(resolution_dir, wide_resolution_dir)
        self.data.dropna(subset=["img_path"], inplace=True)

        task_names = task_names.copy()

        if self_training:
            if self_training.get("csv_files") is None or len(self_training["csv_files"]) == 0:
                raise ValueError("csv_files must be provided for self-training")
            mode = self_training.get("mode", "soft")
            threshold = self_training.get("threshold")
            temperature = self_training.get("temperature")

            self.tasks = [
                get_task(
                    task_name=label,
                    self_training_mode=mode,
                    self_training_threshold=threshold,
                    self_training_temperature=temperature,
                )
                for label in task_names
            ]

        else:
            self.tasks = [get_task(task_name=label) for label in task_names]

        task_labels = sum([task.task_labels for task in self.tasks], [])
        for label in task_labels:
            if label not in self.data:
                self.data[label] = np.nan
        self.data = self.data[
            task_labels + ["img_path", "image_id", "camera_type", "exam_id", "patient_id"]
        ]

        if self_training:
            csv_files = self_training.get("csv_files")
            self_training_data = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files])
            self.data = pd.merge(self.data, self_training_data, how="inner", on="image_id")

        if drop_na_labels and self_training is None:
            self.data.dropna(subset=task_labels, how="all", inplace=True)

        self.data["patient_code"], _ = pd.factorize(self.data["patient_id"])

        self.data = self.data.to_dict("records")

        self.transform = A.Compose(
            [
                A.Resize(img_size, img_size),  # Resize images to the specified size
                A.HorizontalFlip(p=0.5),  # Random horizontal flip
                A.VerticalFlip(p=0.5),  # Random vertical flip
                A.RandomBrightnessContrast(p=0.2),  # Adjust brightness and contrast
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),  # Color augmentations
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),  # Random blur
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize for ImageNet weights
                ToTensorV2(),  # Convert to PyTorch tensor
            ]
        )

    def _read_image(self, gcs_img_path: str) -> np.ndarray:
        img_path = osp.join(self.home_dir, osp.relpath(gcs_img_path, "gs://"))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        gcs_img_path = self.data[index]["img_path"]
        img = self._read_image(gcs_img_path)
        img = self.transform(img)

        img_dict = {
            "img": img,
            "img_path": gcs_img_path,
            "image_id": self.data[index]["image_id"],
            "patient_code": torch.tensor([self.data[index]["patient_code"]], dtype=torch.long),
        }

        for task in self.tasks:
            img_dict.update(task.categorize_labels(self.data[index]))


        return img_dict

    def __len__(self):
        """Return length of dataset."""
        return len(self.data)