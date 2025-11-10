import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import tempfile
import streamlit as st
from PIL import Image

# sichere torch-Importprüfung
try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    torch = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(e)

from inference.predict_single_image import load_model, analyze_image, CONFIG

st.set_page_config(page_title="FixRay - Frakturerkennung", layout="centered")

# Dark theme CSS
st.markdown(
    """
    <style>
      .stApp, .main, .block-container {
        background-color: #000 !important;
        color: #fff !important;
      }
      .stMarkdown, .stText, .stSlider, .stJson, .stNumberInput, .stSelectbox {
        color: #fff !important;
      }
      .stImage img { filter: brightness(0.95) !important; }
      a { color: #9ad4ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("FixRay — Frakturerkennung (Drag & Drop JPG)")
st.markdown("Droppe ein JPG/JPEG/PNG oder wähle eine Datei aus. Das Modell liefert ein annotiertes Ergebnisbild.")

if not TORCH_AVAILABLE:
    st.error(
        "PyTorch ist nicht installiert oder kann nicht importiert:\n"
        f"{TORCH_IMPORT_ERROR}\n\n"
        "Installiere mit:\n"
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    )
    st.stop()

MODEL_PATH = CONFIG.get("model_path")
OUTPUT_DIR = CONFIG.get("output_dir", os.path.join(os.getcwd(), "results"))
DEFAULT_CONF = CONFIG.get("confidence_threshold", 0.5)
DEVICE = CONFIG.get("device", torch.device("cpu"))

confidence = st.slider("Confidence-Schwelle", min_value=0.0, max_value=1.0,
                       value=float(DEFAULT_CONF), step=0.01)

@st.cache_resource
def get_model():
    try:
        return load_model(MODEL_PATH, DEVICE)
    except Exception as e:
        return e

uploaded = st.file_uploader("Bild hochladen", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.image(Image.open(tmp_path), caption="Eingabebild", use_container_width=True)

    model_or_err = get_model()
    if isinstance(model_or_err, Exception):
        st.error(f"Fehler beim Laden des Modells: {model_or_err}")
    else:
        model = model_or_err
        with st.spinner("Analysiere Bild..."):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            result = analyze_image(tmp_path, model, OUTPUT_DIR, confidence_threshold=confidence)

        if result.get('status') == 'SUCCESS':
            st.success(f"Analyse abgeschlossen — {result.get('fractures_detected', 0)} Fraktur(en) erkannt")
            out_img = result.get('output_image')
            if out_img and os.path.exists(out_img):
                st.image(out_img, caption="Ergebnis", use_container_width=True)
            st.json({
                "image_path": result.get('image_path'),
                "fractures_detected": result.get('fractures_detected'),
                "confidence_scores": result.get('confidence_scores'),
                "output_image": result.get('output_image')
            })
        else:
            st.error(f"Analyse fehlgeschlagen: {result.get('error_message')}")

    try:
        os.remove(tmp_path)
    except Exception:
        pass
else:
    st.info("Lade ein Bild hoch, um die Analyse zu starten.")