from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


def read_metadata(path, training=True):
    """Read metadata and return a pandas DataFrame.

    Required columns:
        Training: isic_id, patient_id, target
        Test: isic_id
    """
    # TODO: Read the CSV file.
    # TODO: Check required columns and missing identifiers.
    # TODO: Check that training targets contain only 0 and 1.
    raise NotImplementedError("TODO: read and validate metadata")


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
    # TODO: Resize the image.
    # TODO: Apply augmentation only during training.
    # TODO: Convert to a tensor and normalize.
    raise NotImplementedError("TODO: define image transforms")


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

    # TODO: Open the HDF5 file using h5py.File.
    # TODO: Read the encoded image bytes using isic_id as the key.
    # Hint: Decode the bytes using io.BytesIO and PIL.Image.
    # TODO: Convert the image to RGB and apply self.transform.
    # TODO: Return a dictionary following the contract above.
    # Include target only when self.training is True.
    #
    # Use a context manager for the HDF5 file. Do not share an
    # open file handle between DataLoader workers.

    raise NotImplementedError(f"TODO: load image {isic_id}")
