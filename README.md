# Skin Cancer Detection (ISIC 2024)

[ISIC 2024 Kaggle competition](https://www.kaggle.com/competitions/isic-2024-challenge).

The task is to predict a probability of malignancy for each skin lesion image. This template provides interfaces and code structure. Complete the `TODO` sections to build your solution.

## Setup

Use Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the competition files from the [Data Page](https://www.kaggle.com/competitions/isic-2024-challenge/data).

## Tasks

Complete the files in this order:

1. `dataset.py`: Read metadata, split by patient, decode images, and define transforms.
2. `model.py`: Define a binary image classifier.
3. `main.py`: Connect the components, save/load a checkpoint, and export predictions.

Function docstrings specify the expected inputs and outputs.

Keep all images from each patient in one split. Inspect class imbalance, use augmentation only during training, and select your checkpoint using validation results.

The competition evaluates partial ROC AUC above a true positive rate of 0.80. Refer to the [official metric implementation](https://github.com/ISIC-Research/Challenge-2024-Metrics). Ordinary ROC AUC is
not the competition score.
