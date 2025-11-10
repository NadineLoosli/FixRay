import os
import tempfile
import streamlit as st
from PIL import Image
import torch

# Importiere nur die Funktionen (keine CONFIG)
from inference.predict_single_image import load_model, analyze_image

st.set_page_config(page_title="FixRay - Frakturerkennung", layout="centered")

# Dark theme CSS: schwarzer Hintergrund, weißer Text
st.markdown(
    """
    <style>
      html, body, .stApp, .main, .block-container, .stFileUpload, .stButton>button {
        background-color: #000000 !important;
        color: #ffffff !important;
      }
      .stMarkdown, .stText, .stSlider, .stJson, .stNumberInput, .stSelectbox {
        color: #ffffff !important;
      }
      /* Bildhelligkeit leicht anpassen, damit Annotationen sichtbar bleiben */
      .stImage img { filter: brightness(0.95) !important; }
      /* Links / kleine UI-Anpassungen */
      a { color: #9ad4ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("FixRay — Frakturerkennung (Drag & Drop JPG)")

st.markdown("Droppe ein JPG/JPEG/PNG oder wähle eine Datei aus. Das Modell analysiert das Bild und liefert ein annotiertes Ergebnisbild.")

# Lokale Konfiguration (falls predict_single_image.py kein CONFIG exportiert)
MODEL_PATH = r'C:\Users\nadin\FixRay\fracture-segmentation\fracture_detection_model_final.pth'
OUTPUT_DIR = r'C:/Users/nadin/FixRay/results'
DEFAULT_CONF = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.image(Image.open(tmp_path), caption="Eingabebild", use_container_width=True)

    model_or_err = get_model()
    if isinstance(model_or_err, Exception):
        st.error(f"Fehler beim Laden des Modells: {model_or_err}")
    else:
        model = model_or_err
        with st.spinner("Analysiere Bild..."):
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