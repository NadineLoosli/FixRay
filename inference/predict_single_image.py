import os
import logging
from datetime import datetime

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Minimal CONFIG — passe Pfade bei Bedarf an
CONFIG = {
    "num_classes": 2,
    "confidence_threshold": 0.5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_path": r"C:\Users\nadin\FixRay\fracture-segmentation\fracture_detection_model_final.pth",
    "output_dir": r"C:\Users\nadin\FixRay\results",
}

def get_model(num_classes):
    logger.info("Initialisiere Mask R-CNN Modell...")
    try:
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    except Exception:
        model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    model.to(CONFIG["device"])
    model.eval()
    logger.info("Modell initialisiert")
    return model

def load_model(model_path, device=None):
    if device is None:
        device = CONFIG["device"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")

    logger.info(f"Lade Modell von: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
        model.eval()
        logger.info("Checkpoint ist ein nn.Module - direkt verwendet")
        return model

    model = get_model(CONFIG["num_classes"])

    state_dict = None
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            try:
                first_val = next(iter(checkpoint.values()))
                if isinstance(first_val, torch.Tensor):
                    state_dict = checkpoint
            except StopIteration:
                state_dict = None

    if state_dict is None:
        raise RuntimeError("Unbekanntes Checkpoint-Format. Prüfe torch.load(...) Inhalt")

    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Modell geladen und eval gesetzt")
    return model

def analyze_image(image_path, model, output_dir=None, confidence_threshold=None):
    if output_dir is None:
        output_dir = CONFIG["output_dir"]
    if confidence_threshold is None:
        confidence_threshold = CONFIG["confidence_threshold"]

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    logger.info(f"Analysiere Bild: {image_path}")
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(CONFIG["device"])

    with torch.no_grad():
        outputs = model(input_tensor)

    if not outputs:
        return {"status": "ERROR", "error_message": "Leere Ausgabe vom Modell", "image_path": image_path}

    out = outputs[0]
    boxes = out.get("boxes", torch.empty((0, 4))).cpu().numpy()
    scores = out.get("scores", torch.empty((0,))).cpu().numpy()
    masks = out.get("masks", None)
    if masks is not None:
        masks = masks.cpu().numpy()

    keep = scores >= confidence_threshold
    boxes_f = boxes[keep]
    scores_f = scores[keep]
    masks_f = masks[keep] if masks is not None else None

    fractures_detected = int(len(boxes_f))

    draw_img = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", draw_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes_f):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)
        label_txt = f"Fraktur {scores_f[i]:.2f}"
        text_w = (len(label_txt) * 6) + 6
        draw.rectangle([x1, max(0, y1-18), x1 + text_w, y1], fill=(255, 0, 0, 200))
        draw.text((x1+3, max(0, y1-16)), label_txt, fill=(255, 255, 255, 255), font=font)

        if masks_f is not None:
            mask = masks_f[i, 0]
            mask_img = Image.fromarray((mask * 255).astype("uint8")).resize((w, h)).convert("L")
            red = Image.new("RGBA", (w, h), (255, 0, 0, 100))
            draw_img = Image.composite(red, draw_img, mask_img)

    result_img = Image.alpha_composite(draw_img, overlay).convert("RGB")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"fracture_analysis_{timestamp}.png")
    result_img.save(out_path)

    summary = {
        "status": "SUCCESS",
        "image_path": image_path,
        "fractures_detected": fractures_detected,
        "confidence_scores": scores_f.tolist(),
        "output_image": out_path
    }
    logger.info(f"Analyse fertig: {fractures_detected} Treffer. Ergebnis: {out_path}")
    return summary
import os
import tempfile
import streamlit as st
from PIL import Image

# Importiere Funktionen aus deinem Inferenz-Skript
from inference.predict_single_image import load_model, analyze_image, CONFIG

st.set_page_config(page_title="FixRay - Frakturerkennung", layout="centered")

st.title("FixRay — Frakturerkennung (Drag & Drop JPG)")

st.markdown("Droppe ein JPG/JPEG/PNG oder wähle eine Datei aus. Das Modell analysiert das Bild und liefert ein annotiertes Ergebnisbild.")

# Confidence-Slider
confidence = st.slider("Confidence-Schwelle", min_value=0.0, max_value=1.0, value=float(CONFIG.get('confidence_threshold', 0.5)), step=0.01)

# Lade/Cache das Modell einmal
@st.cache(allow_output_mutation=True)
def get_model():
    try:
        model = load_model(CONFIG['model_path'], CONFIG['device'])
        return model
    except Exception as e:
        return e

uploaded = st.file_uploader("Bild hochladen", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    # Temporär speichern
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.image(Image.open(tmp_path), caption="Eingabebild", use_column_width=True)

    model_or_err = get_model()
    if isinstance(model_or_err, Exception):
        st.error(f"Fehler beim Laden des Modells: {model_or_err}")
    else:
        model = model_or_err
        with st.spinner("Analysiere Bild..."):
            result = analyze_image(tmp_path, model, CONFIG['output_dir'], confidence_threshold=confidence)

        if result.get('status') == 'SUCCESS':
            st.success(f"Analyse abgeschlossen — {result.get('fractures_detected', 0)} Fraktur(en) erkannt")
            out_img = result.get('output_image')
            if out_img and os.path.exists(out_img):
                st.image(out_img, caption="Ergebnis", use_column_width=True)
            st.json({
                "image_path": result.get('image_path'),
                "fractures_detected": result.get('fractures_detected'),
                "confidence_scores": result.get('confidence_scores'),
                "output_image": result.get('output_image')
            })
        else:
            st.error(f"Analyse fehlgeschlagen: {result.get('error_message')}")

    # Option: temporäre Datei entfernen
    try:
        os.remove(tmp_path)
    except Exception:
        pass
else:
    st.info("Lade ein Bild hoch, um die Analyse zu starten.")