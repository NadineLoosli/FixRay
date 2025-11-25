import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import models, transforms


# Basisverzeichnis: .../FixRay
BASE_DIR = Path(__file__).resolve().parents[3]

CONFIG: Dict[str, Any] = {
    "model_name": "resnet18",
    "device": "cpu",  # wird in load_model ggf. überschrieben
    "confidence_threshold": 0.5,
    "output_dir": str(BASE_DIR / "results" / "inference_outputs"),
}


def load_model(device: str = "cpu") -> Dict[str, Any]:
    """
    Lädt ein vortrainiertes ResNet18 (ImageNet) aus torchvision.
    Es ist ein echtes Deep-Learning-Modell, aber NICHT auf Frakturbildern
    trainiert – dient als KI-Demonstrator.
    """
    model = models.resnet18(pretrained=True)
    model.eval()

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model.to(device)

    # Output-Verzeichnis anlegen
    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CONFIG.copy()
    cfg["device"] = device
    cfg["net"] = model
    return cfg


# Bildvorverarbeitung für ResNet (ImageNet-Standard)
_PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def _compute_fracture_score(img: Image.Image, model_cfg: Dict[str, Any]) -> float:
    """
    Führt das Bild durch das vortrainierte ResNet und berechnet
    einen pseudo-„Fraktur-Score“ aus den Logits.

    Technisch:
      - Wir nehmen den maximalen Logit aus den 1000 ImageNet-Klassen
      - wenden eine Sigmoid-Funktion an → Wert in [0, 1]
    """
    device = model_cfg["device"]
    net = model_cfg["net"]

    img_t = _PREPROCESS(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = net(img_t)  # Shape: [1, 1000]
        max_logit = logits.max().item()

    # Sigmoid → 0..1
    score = float(torch.sigmoid(torch.tensor(max_logit)).item())
    return score


def _annotate_image(
    img: Image.Image,
    fracture: bool,
    score: float,
) -> Image.Image:
    """
    Zeichnet einen Balken mit Text auf das Bild:
      - Rot: Fraktur-Verdacht
      - Grün: kein Fraktur-Verdacht
    """
    annotated = img.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    width, height = annotated.size
    bar_height = int(height * 0.08)

    if fracture:
        bar_color = (180, 40, 40)
        text = f"Fraktur-Verdacht (Score: {score:.2f})"
    else:
        bar_color = (40, 160, 80)
        text = f"Keine Fraktur detektiert (Score: {score:.2f})"

    # Balken am oberen Bildrand
    draw.rectangle(
        [(0, 0), (width, bar_height)],
        fill=bar_color,
    )

    # Text
    text_x = int(width * 0.02)
    text_y = int(bar_height * 0.25)
    draw.text((text_x, text_y), text, fill=(255, 255, 255))

    return annotated


def analyze_image(
    image_path: str,
    model: Dict[str, Any],
    output_dir: str | None = None,
    confidence_threshold: float | None = None,
) -> Dict[str, Any]:
    """
    Analysiert ein Bild mit dem vortrainierten ResNet und gibt
    ein Ergebnis-Dict zurück:

      {
        "status": "SUCCESS",
        "fracture": bool,
        "score": float,
        "output_image": str (Pfad)
      }
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "error_message": f"Bild kann nicht geladen werden: {exc}",
        }

    if confidence_threshold is None:
        confidence_threshold = float(model.get("confidence_threshold", 0.5))

    score = _compute_fracture_score(img, model_cfg=model)
    fracture = score >= confidence_threshold

    # Ausgabe-Verzeichnis
    if output_dir is None:
        output_dir = model.get(
            "output_dir",
            str(BASE_DIR / "results" / "inference_outputs"),
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dateiname für annotiertes Bild
    base_name = os.path.basename(image_path)
    name_no_ext, _ = os.path.splitext(base_name)
    out_path = out_dir / f"{name_no_ext}_annotated.png"

    annotated = _annotate_image(img, fracture=fracture, score=score)
    annotated.save(out_path)

    return {
        "status": "SUCCESS",
        "fracture": fracture,
        "score": score,
        "output_image": str(out_path),
    }
