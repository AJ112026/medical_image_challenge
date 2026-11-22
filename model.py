from torch import nn


class SkinLesionClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: Define a small CNN or adapt a torchvision backbone.
        # TODO: Define a classification head with one output per image.

        raise NotImplementedError("TODO: define model layers")

    def forward(self, images):
        """Return raw logits.

        Args:
            images: float tensor with shape [B, 3, H, W].

        Returns:
            Float tensor with shape [B], including when B equals 1.
        """
        # TODO: Extract image features.
        # TODO: Apply the classification head.
        # TODO: Return logits with the required shape.
        #
        # Do not apply sigmoid here when using BCEWithLogitsLoss.

        raise NotImplementedError("TODO: implement the forward pass")


def predict(model, loader, device):
  """Generate predictions.

  Returns:
      A tuple (isic_ids, probabilities) containing aligned lists.
      Each probability must be a finite number in [0, 1].
  """
  # TODO: Enable evaluation mode and disable gradient computation.
  # TODO: Compute logits and convert them to probabilities.
  # TODO: Preserve the association between image IDs and predictions.

  raise NotImplementedError("TODO: implement prediction")