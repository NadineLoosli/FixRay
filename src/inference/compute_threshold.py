import os
import sys
from pathlib import Path
from PIL import Image
import torch
import numpy as np

from torchvision import transforms

# Stelle sicher, dass inference Modul importierbar ist
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.predict_single_image import load_model, CONFIG

def collect_scores(model, img_path, device, transform):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    if not outputs:
        return 0.0
    out = outputs[0]
    scores = out.get("scores", None)
    if scores is None or not hasattr(scores, "cpu"):
        return 0.0
    arr = scores.detach().cpu().numpy()
    return float(arr.max()) if arr.size > 0 else 0.0

def compute_threshold(fracatlas_root, model_path=None, device=None):
    if model_path is None:
        model_path = CONFIG.get("model_path")
    if device is None:
        device = CONFIG.get("device", torch.device("cpu"))

    # build model
    model = load_model(model_path, device=device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    fractured_dir = Path(fracatlas_root) / "FracAtlas" / "images" / "Fractured"
    nonfractured_dir = Path(fracatlas_root) / "FracAtlas" / "images" / "Non_fractured"

    if not fractured_dir.exists() or not nonfractured_dir.exists():
        raise FileNotFoundError(f"Erwarte Ordner: {fractured_dir} und {nonfractured_dir}")

    y_true = []
    y_scores = []

    for p in sorted(fractured_dir.glob("*.jpg")):
        try:
            s = collect_scores(model, p, device, transform)
        except Exception as e:
            print("Warnung, skip:", p, e)
            s = 0.0
        y_true.append(1)
        y_scores.append(s)

    for p in sorted(nonfractured_dir.glob("*.jpg")):
        try:
            s = collect_scores(model, p, device, transform)
        except Exception as e:
            print("Warnung, skip:", p, e)
            s = 0.0
        y_true.append(0)
        y_scores.append(s)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y_true, y_scores)
        j = tpr - fpr
        idx = int(np.nanargmax(j))
        best_thr = float(thr[idx])
        print(f"Optimal threshold (Youden's J): {best_thr:.4f}")
        print(f"TPR={tpr[idx]:.3f}, FPR={fpr[idx]:.3f}, images evaluated={len(y_true)}")
        return best_thr
    except Exception:
        positives = y_scores[y_true == 1]
        if positives.size == 0:
            print("Keine positiven Beispiele gefunden — benutze Default 0.5")
            return 0.5
        thr = float(max(0.0, positives.mean() - positives.std()))
        print(f"Fallback threshold (mean-std): {thr:.4f}")
        return thr

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools\\compute_threshold.py C:\\Pfad\\zu\\fracture-segmentation")
        sys.exit(1)
    root = sys.argv[1]
    thr = compute_threshold(root)
    print("Empfohlene Confidence-Schwelle:", thr)