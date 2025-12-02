# src/app/inference/predict_single_image.py

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ----------------------------------------------------
# "CONFIG" – nur noch für Heuristik, kein echtes KI-Modell
# ----------------------------------------------------

CONFIG = {
    # wird von main.py gelesen, aber wir nutzen kein echtes Modell
    "device": "cpu",
    # Standard-Schwelle für die Helligkeits-Heuristik
    "confidence_threshold": 0.15,
    # wohin annotierte Bilder geschrieben werden
    "output_dir": "results",
}


# ----------------------------------------------------
# Dummy-Model-Loader (wird von main.py aufgerufen)
# ----------------------------------------------------

def load_model(model_path: str | None = None, device=None):
    """
    FixRay erwartet eine load_model-Funktion.
    Für die einfache Heuristik brauchen wir aber kein echtes Modell.
    Diese Funktion gibt deshalb einfach None zurück, wir ignorieren 'model'
    später in analyze_image.
    """
    return None


# ----------------------------------------------------
# Heuristische Fraktur-Erkennung
# ----------------------------------------------------

def analyze_image(image_path, model, output_dir=None, confidence_threshold=0.15):
    """
    Sehr einfache, Helligkeits-basierte Heuristik:

    Idee:
    - Röntgenbild ist hell, Knochen sind hell, Spalten / Fraktur-Linien
      sind oft dunkler.
    - Wir betrachten die zentrale Vertikal- und Horizontallinie im Bild
      und messen, ob es dort einen starken Helligkeitsabfall gibt
      (hell -> deutlich dunkler).

    Vorgehen:
    - Bild nach 512x512 skalieren
    - Grauwertbild (0..1)
    - zentrale Vertikal-Linie und Horizontal-Linie extrahieren
    - für jede Linie:
        gap = mean(line) - min(line)
      (d.h. wie viel dunkler ist der dunkelste Punkt im Vergleich zum Rest)
    - score = max(vert_gap, horiz_gap)
    - Wenn score >= confidence_threshold -> "Fraktur: JA", sonst "NEIN".

    WICHTIG:
    Das ist KEINE medizinisch valide Erkennung, sondern nur eine
    anschauliche Demo-Heuristik für den Prototyp.
    """

    if output_dir is None:
        output_dir = CONFIG.get("output_dir", "results")

    if confidence_threshold is None:
        confidence_threshold = CONFIG.get("confidence_threshold", 0.15)

    # Bild laden
    try:
        orig_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            "status": "ERROR",
            "error_message": f"Bild konnte nicht geladen werden: {e}",
        }

    # Arbeitsbild vereinheitlichen (512x512, Graustufen)
    work_img = orig_img.resize((512, 512))
    gray = work_img.convert("L")
    arr = np.asarray(gray).astype("float32") / 255.0  # Werte 0..1

    h, w = arr.shape

    # zentrale Vertikal- und Horizontallinie
    vert_line = arr[:, w // 2]
    horiz_line = arr[h // 2, :]

    def gap_score(line: np.ndarray) -> float:
        line_mean = float(line.mean())
        line_min = float(line.min())
        # Je größer der Unterschied, desto "dunklerer Spalt"
        return max(0.0, line_mean - line_min)

    vert_gap = gap_score(vert_line)
    horiz_gap = gap_score(horiz_line)

    # Score = maximaler Helligkeitssprung
    score = max(vert_gap, horiz_gap)

    # Schwellenentscheidung: JA/NEIN
    fracture = score >= float(confidence_threshold)

    # Für Visualisierung: dominante Linie und dunkelster Punkt bestimmen
    if vert_gap >= horiz_gap:
        dominant = "vertical"
        line = vert_line
        min_idx = int(np.argmin(line))
    else:
        dominant = "horizontal"
        line = horiz_line
        min_idx = int(np.argmin(line))

    # Annotiertes Bild erzeugen
    annotated = orig_img.copy()
    draw = ImageDraw.Draw(annotated)
    img_w, img_h = annotated.size

    if fracture:
        # Linie ungefähr dort einzeichnen, wo der "dunkelste" Bereich war
        if dominant == "vertical":
            # x ~ Bildmitte, y ~ Position des Minimums (umgerechnet)
            x = img_w // 2
            y = int(min_idx * (img_h / 512))
            draw.line([(x, 0), (x, img_h)], fill=(255, 0, 0), width=3)
        else:
            y = img_h // 2
            x = int(min_idx * (img_w / 512))
            draw.line([(0, y), (img_w, y)], fill=(255, 0, 0), width=3)

    # Datei speichern
    out_path = None
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "last_result_heuristic.png"
        annotated.save(out_path)

    return {
        "status": "SUCCESS",
        "fracture": bool(fracture),
        "score": float(score),  # "Stärke" des Helligkeitssprungs
        "method": "brightness_gap_heuristic",
        "output_image": str(out_path) if out_path else None,
    }
