import timm
import torch
from torch import nn


class SkinLesionClassifier(nn.Module):
    def __init__(self, model_name="edgenext_small", pretrained=False, dropout=0.2):
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.backbone.num_features, 1),)

        
    def forward(self, images):
        """Return raw logits.

        Args:
            images: float tensor with shape [B, 3, H, W].

        Returns:
            Float tensor with shape [B], including when B equals 1.
        """

        features = self.backbone(images)
        logits = self.head(features)
        return logits.squeeze(-1)


def predict(model, loader, device, use_tta=False):
    """Generate predictions.

    Returns:
        A tuple (isic_ids, probabilities) containing aligned lists.
        Each probability must be a finite number in [0, 1].
    """
    model.to(device)  # model en afbeelding op zelfde rekenapparaat
    model.eval()  # zet model in testmodus

    isic_ids = []
    probabilities = []

    with torch.no_grad():  # zet bijhouden gradienten uit (sneller en minder geheugen)
        for batch in loader:
            images = batch["image"].to(device)

            if use_tta:
                views = [
                    images,
                    torch.flip(images, dims=[3]),     # horizontaal gespiegeld
                    torch.flip(images, dims=[2]),     # verticaal gespiegeld
                    torch.flip(images, dims=[2, 3]),  # beide richtingen gespiegeld
                ]
                probs = sum(torch.sigmoid(model(v)) for v in views) / len(views)
            else:
                logits = model(images)
                probs = torch.sigmoid(logits)  # zet logits naar kans tss 0 en 1

            isic_ids.extend(batch["isic_id"])  # zet de isic_ids van de batch in de lijst
            probabilities.extend(probs.cpu().tolist())

    return isic_ids, probabilities