"""Evaluatiemetric voor ISIC 2024: partiele AUC boven een minimale TPR.

De competitiescore is NIET de gewone ROC AUC, maar de oppervlakte van de
ROC-curve in het gebied met TPR >= min_tpr (standaard 0.80). De maximale
score is dus 1 - min_tpr = 0.20 en een willekeurig model scoort ongeveer 0.02.

Referentie: officiele implementatie van de organisatoren,
https://github.com/ISIC-Research/Challenge-2024-Metrics
(deze versie is een eigen herschrijving die numeriek met de officiele
code is vergeleken).
"""

import numpy as np
from sklearn.metrics import auc, roc_curve


def partial_auc(y_true, y_score, min_tpr=0.80):
    """Bereken de partiele AUC boven `min_tpr`.

    Args:
        y_true: array met 0 (goedaardig) en 1 (kwaadaardig).
        y_score: array met scores of kansen; hoger = kwaadaardiger.
            Alleen de volgorde telt, dus logits of kansen geven dezelfde score.
        min_tpr: minimale gevoeligheid (true positive rate), in [0, 1).

    Returns:
        Float tussen 0 en 1 - min_tpr.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if len(np.unique(y_true)) != 2:
        raise ValueError("y_true moet beide klassen (0 en 1) bevatten.")
    if not 0 <= min_tpr < 1:
        raise ValueError("min_tpr moet in [0, 1) liggen.")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("y_score bevat NaN of oneindig.")

    # Draai de klassen om: de kwaadaardige gevallen worden de 'negatieven'.
    # Dan komt "TPR >= min_tpr" overeen met "FPR <= max_fpr" in de omgedraaide curve.
    flipped_true = 1 - y_true
    flipped_score = -y_score
    max_fpr = 1 - min_tpr

    fpr, tpr, _ = roc_curve(flipped_true, flipped_score)

    # Knip de curve af bij max_fpr, met lineaire interpolatie op de rand.
    keep = fpr < max_fpr
    x = np.append(fpr[keep], max_fpr)
    y = np.append(tpr[keep], np.interp(max_fpr, fpr, tpr))
    return float(auc(x, y))