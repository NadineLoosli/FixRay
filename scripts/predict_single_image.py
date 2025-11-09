#!/usr/bin/env python3
"""
Minimal-Beispiel: predict_single_image.py
Dieses Skript ist ein Platzhalter — hier kannst du das Laden deines Modells und Vorhersagen implementieren.
"""
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

from fixray.predict import load_model, predict


def main():
    parser = argparse.ArgumentParser(description="Predict a single image with FixRay model")
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--model", type=Path, default=Path("models/latest.pt"), help="Path to model file")
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    image = Image.open(args.image).convert("RGB")

    try:
        model = load_model(args.model)
    except NotImplementedError:
        model = None

    result = predict(image, model)
    print("Prediction result:", result)


if __name__ == "__main__":
    main()
