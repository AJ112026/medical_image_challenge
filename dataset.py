import io
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms

def read_metadata(path, training=True):
    """Read metadata and return a pandas DataFrame.

    Required columns:
        Training: isic_id, patient_id, target
        Test: isic_id
    """
    df = pd.read_csv(path, low_memory=False)  # CSV inlezen

    required = ["isic_id", "patient_id", "target"] if training else ["isic_id"]
    missing = [col for col in required if col not in df.columns]  # welke kolommen ontbreken?
    if missing:
        raise ValueError(f"Ontbrekende kolommen: {missing}")

    if df["isic_id"].isna().any():  # zijn er lege isic_id's?
        raise ValueError("Er zijn lege isic_id's")

    if training and not set(df["target"].unique()) <= {0, 1}:
        raise ValueError("Target bevat andere waarden dan 0 en 1")

    return df


def split_metadata(metadata, seed=42):
    """Return training and validation DataFrames.

    Requirements:
        - No patient appears in both splits.
        - Both splits contain both target classes.
        - Every input row belongs to exactly one split.
        - The split is reproducible.
    """
    # TODO: Split the data using patient_id as the grouping variable.
    # Hint: Consider sklearn.model_selection.StratifiedGroupKFold.
    # TODO: Check patient overlap and class distributions.
    raise NotImplementedError("TODO: implement a patient-level split")

def build_transforms(image_size=224, training=True):
    """Return a transform mapping an RGB PIL image to a tensor [3, H, W]."""
    steps = [transforms.Resize((image_size, image_size))]  # uitsneden zijn ~128 px, het model verwacht ~224 px

    if training:  # augmentatie ALLEEN tijdens het trainen
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # willekeurige draaiing in stappen van 90 graden (geen interpolatie, geen zwarte hoeken)
            transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
            # lichte verandering van helderheid en contrast (belichting verschilt tussen foto's);
            # bewust GEEN tint/verzadiging: de kleur van de vlek is een belangrijk signaal
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]

    steps += [
        transforms.ToTensor(),  # PIL [H, W, 3] in 0-255  ->  float32 [3, H, W] in 0-1
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet-statistieken
    ]
    return transforms.Compose(steps)


class ISICDataset(Dataset):
    """Load images from a competition HDF5 file.

    Each item must contain:
        image: float32 tensor with shape [3, H, W]
        isic_id: string

    Training items must also contain:
        target: scalar float32 tensor

    After batching, images have shape [B, 3, H, W] and targets have
    shape [B].
"""


    def __init__(self, metadata, image_path, transform, training=True):
        self.metadata = metadata.reset_index(drop=True).copy()
        self.image_path = Path(image_path)
        self.transform = transform
        self.training = training


    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        isic_id = row["isic_id"]

        with h5py.File(self.image_path, "r") as f:
            if isic_id not in f:
                raise KeyError(f"{isic_id} staat niet in {self.image_path}")
            raw = f[isic_id][()]  # gecodeerde JPEG-bytes

        image = Image.open(io.BytesIO(np.asarray(raw).tobytes())).convert("RGB")
        image = self.transform(image)  # -> float32 tensor [3, H, W]

        item = {"image": image, "isic_id": isic_id}
        if self.training:
            item["target"] = torch.tensor(row["target"], dtype=torch.float32)
        return item
