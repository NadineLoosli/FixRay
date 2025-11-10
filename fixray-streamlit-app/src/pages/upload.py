import os
import tempfile
import streamlit as st
from PIL import Image

from inference.predict_single_image import load_model, analyze_image, CONFIG

st.set_page_config(page_title="FixRay - Frakturerkennung", layout="centered")

st.title("FixRay — Frakturerkennung (Drag & Drop JPG)")

st.markdown("Droppe ein JPG/JPEG/PNG oder wähle eine Datei aus. Das Modell analysiert das Bild und liefert ein annotiertes Ergebnisbild.")

confidence = st.slider("Confidence-Schwelle", min_value=0.0, max_value=1.0, value=float(CONFIG.get('confidence_threshold', 0.5)), step=0.01)

@st.cache(allow_output_mutation=True)
def get_model():
    try:
        model = load_model(CONFIG['model_path'], CONFIG['device'])
        return model
    except Exception as e:
        return e

uploaded = st.file_uploader("Bild hochladen", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
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

    try:
        os.remove(tmp_path)
    except Exception:
        pass
else:
    st.info("Lade ein Bild hoch, um die Analyse zu starten.")