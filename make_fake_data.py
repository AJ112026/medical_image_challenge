"""Maak een kleine NEP-dataset in dezelfde structuur als de ISIC 2024-data.

Alleen bedoeld om je code lokaal snel te testen (CPU, enkele seconden), zonder
de echte dataset van enkele GB te downloaden. De resultaten zeggen niets over
echte prestaties.

Aangemaakte bestanden in de uitvoermap (standaard `data/`):
    train-metadata.csv   kolommen: isic_id, patient_id, target (+ age_approx, sex)
    train-image.hdf5     sleutel = isic_id, waarde = JPEG-bytes
    test-metadata.csv    kolom: isic_id
    test-image.hdf5      zoals train-image.hdf5
    sample_submission.csv  kolommen: isic_id, target

Gebruik:
    python make_fake_data.py
    python make_fake_data.py --n-train 1000 --n-patients 60 --out data
"""

import argparse
import io
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

IMAGE_SIZE = 128  # de echte uitsneden zijn ongeveer 128 x 128 pixels


def make_image(rng, malignant):
    """Teken een huidkleurige uitsnede met een vlek; kwaadaardig = donkerder en onregelmatiger."""
    skin = np.array([205, 160, 140]) + rng.integers(-25, 25, size=3)
    img = np.empty((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    img[:] = skin
    img += rng.normal(0, 6, size=img.shape)  # ruis

    canvas = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    cx, cy = IMAGE_SIZE // 2 + rng.integers(-10, 10, size=2)
    if malignant:
        radius = int(rng.integers(22, 36))
        colour = tuple(int(c) for c in rng.integers(25, 80, size=3))
        # onregelmatige rand: meerdere overlappende ellipsen
        for _ in range(4):
            dx, dy = rng.integers(-8, 8, size=2)
            rx, ry = radius + rng.integers(-8, 8, size=2)
            draw.ellipse([cx + dx - rx, cy + dy - ry, cx + dx + rx, cy + dy + ry], fill=colour)
    else:
        radius = int(rng.integers(8, 20))
        colour = tuple(int(c) for c in np.array([150, 105, 90]) + rng.integers(-15, 15, size=3))
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=colour)
    return canvas.filter(ImageFilter.GaussianBlur(1.2))


def jpeg_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def write_split(path_hdf5, ids, labels, rng):
    with h5py.File(path_hdf5, "w") as f:
        for isic_id, target in zip(ids, labels):
            data = np.frombuffer(jpeg_bytes(make_image(rng, bool(target))), dtype=np.uint8)
            f.create_dataset(isic_id, data=data)


def main():
    parser = argparse.ArgumentParser(description="Maak nepdata in ISIC 2024-formaat")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--n-patients", type=int, default=40)
    parser.add_argument("--pos-frac", type=float, default=0.10,
                        help="aandeel kwaadaardig (echt: 0.001, hier hoger zodat elke split beide klassen heeft)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # --- train ---
    train_ids = [f"ISIC_{i:07d}" for i in range(1, args.n_train + 1)]
    patient_weights = rng.dirichlet(np.ones(args.n_patients))  # sommige patienten hebben veel vlekken
    patient_idx = rng.choice(args.n_patients, size=args.n_train, p=patient_weights)
    targets = np.zeros(args.n_train, dtype=int)
    n_pos = max(2, int(round(args.pos_frac * args.n_train)))
    targets[rng.choice(args.n_train, size=n_pos, replace=False)] = 1

    train = pd.DataFrame({
        "isic_id": train_ids,
        "target": targets,
        "patient_id": [f"IP_{p:07d}" for p in patient_idx],
        "age_approx": rng.integers(20, 85, size=args.n_train),
        "sex": rng.choice(["male", "female"], size=args.n_train),
    })
    train.to_csv(args.out / "train-metadata.csv", index=False)
    write_split(args.out / "train-image.hdf5", train_ids, targets, rng)

    # --- test (zonder labels, zoals in de competitie) ---
    test_ids = [f"ISIC_{i:07d}" for i in range(args.n_train + 1, args.n_train + args.n_test + 1)]
    test_targets = (rng.random(args.n_test) < args.pos_frac).astype(int)  # alleen om afbeeldingen te tekenen
    pd.DataFrame({"isic_id": test_ids}).to_csv(args.out / "test-metadata.csv", index=False)
    write_split(args.out / "test-image.hdf5", test_ids, test_targets, rng)
    pd.DataFrame({"isic_id": test_ids, "target": 0.3}).to_csv(args.out / "sample_submission.csv", index=False)

    print(f"Klaar in '{args.out}': {args.n_train} train-afbeeldingen "
          f"({int(targets.sum())} kwaadaardig, {train['patient_id'].nunique()} patienten), "
          f"{args.n_test} test-afbeeldingen.")


if __name__ == "__main__":
    main()