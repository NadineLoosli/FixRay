import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import tempfile, os
from PIL import Image

from src.inference.predict_single_image import load_model, analyze_image, CONFIG

st.set_page_config(page_title="FixRay — Frakturerkennung", layout="centered")
st.title("FixRay — Frakturerkennung (Drag & Drop JPG)")

confidence = st.slider("Confidence-Schwelle", min_value=0.0, max_value=1.0,
                       value=float(CONFIG.get("confidence_threshold", 0.5)),
                       step=0.01, key="confidence_slider_v1")

st.write("Droppe ein JPG/JPEG/PNG oder wähle eine Datei aus.")
uploaded = st.file_uploader("Bild hochladen", type=['jpg', 'jpeg', 'png'], key="uploader_v1")

@st.cache_resource
def _load_model_cached():
    try:
        model = load_model(CONFIG.get("model_path"), CONFIG.get("device"))
        return model
    except Exception as e:
        return e

model_or_exc = _load_model_cached()
if isinstance(model_or_exc, Exception):
    st.error(f"Fehler beim Laden des Modell: {model_or_exc}")
    st.stop()
model = model_or_exc

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    st.image(tmp_path, caption="Eingabebild", use_container_width=True)
    st.write("Analysiere...")
    try:
        with st.spinner("Running model..."):
            res = analyze_image(tmp_path, model, output_dir=CONFIG.get("output_dir"), confidence_threshold=confidence)
    except Exception as e:
        st.error(f"Fehler bei Inferenz: {e}")
        st.stop()

    if res.get("status") != "SUCCESS":
        st.error("Inference failed: " + str(res.get("error_message", res)))
    else:
        out_img = res.get("output_image")
        st.success(f"Ergebnis: {'Fraktur' if res.get('fracture') else 'Keine Fraktur'} (score={res.get('score'):.2f})")
        if out_img and os.path.exists(out_img):
            st.image(out_img, caption="Annotiertes Ergebnis", use_container_width=True)
        else:
            st.write("Kein annotiertes Bild gefunden.")