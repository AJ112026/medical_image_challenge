import argparse
from pathlib import Path
from dataset import (
    ISICDataset,
    build_transforms,
    read_metadata,
    split_metadata,
)
from model import SkinLesionClassifier

def run_training(args):
    """Build the training pipeline and save the best checkpoint."""
    # TODO:

def run_prediction(args):
  """Load a checkpoint and export submission.csv."""
  # TODO:



if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="ISIC 2024 student lab")

  parser.add_argument(
      "--mode", choices=("train", "predict"), default="train"
  )
  parser.add_argument("--data-dir", type=Path, default=Path("data"))
  parser.add_argument(
      "--checkpoint", type=Path, default=Path("outputs/model.pt")
  )
  parser.add_argument(
      "--output-dir", type=Path, default=Path("outputs")
  )
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--image-size", type=int, default=224)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--num-workers", type=int, default=0)

  args = parser.parse_args()

  if args.mode == "train":
      run_training(args)
  else:
      run_prediction(args)
