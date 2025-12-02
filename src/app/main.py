# src/app/main.py

import sys
import os
import io
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

# --------- Pfade einrichten, damit app.inference importierbar ist ---------
THIS_DIR = Path(__file__).resolve().parent      # .../src/app
SRC_DIR = THIS_DIR.parent                       # .../src

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.inference.predict_single_image import load_model, analyze_image, CONFIG  # noqa


# --------- Streamlit Grundkonfiguration ---------
st.set_page_config(
    page_title="FixRay — Frakturerkennung (Heuristik)",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #222 0, #000 55%);
        color: #f5f5f5;
        font-family: system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }
    .title {
        text-align: left;
        font-size: 2.3rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: left;
        font-size: 0.95rem;
        color: #aaa;
        margin-bottom: 1.5rem;
    }

    .upload-wrapper {
        margin-top: 1.2rem;
        margin-bottom: 1.8rem;
    }
    .upload-card {
        border-radius: 1rem;
        padding: 1.4rem 1.6rem;
        background: linear-gradient(135deg, #151515, #101018);
        border: 1px solid #333;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 18px 35px rgba(0, 0, 0, 0.55);
    }
    .upload-icon {
        font-size: 2.2rem;
    }
    .upload-text-main {
        font-size: 1.05rem;
        font-weight: 600;
    }
    .upload-text-sub {
        font-size: 0.85rem;
        color: #b5b5b5;
        margin-top: 0.25rem;
    }

    [data-testid="stFileUploader"] > label {
        display: none;
    }
    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        border-radius: 0.85rem;
        border: 1px dashed #555;
        background: rgba(255,255,255,0.01);
    }
    [data-testid="stFileUploader"] div[role="button"] {
        background: #222;
        border-radius: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------- Header ---------
st.markdown(
    '<div class="title">FixRay — heuristische Frakturerkennung</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Prototyp mit einfacher Helligkeits-Heuristik. '
    'Kein medizinisches Produkt, ersetzt keine ärztliche Diagnose.</div>',
    unsafe_allow_html=True,
)

# --------- Sidebar ---------
st.sidebar.header("Prototyp")
st.sidebar.caption("Nur zu Demonstrations- und Lehrzwecken.")

confidence = st.sidebar.slider(
    "Helligkeitsschwelle für Frakturerkennung",
    min_value=0.0,
    max_value=0.5,
    value=float(CONFIG.get("confidence_threshold", 0.15)),
    step=0.01,
    help=(
        "Je niedriger die Schwelle, desto empfindlicher reagiert die Heuristik "
        "(mehr mögliche Frakturen, aber mehr Fehlalarme). "
        "Je höher die Schwelle, desto strenger wird entschieden "
        "(weniger Funde, aber eher nur deutliche Helligkeitssprünge)."
    ),
)

st.sidebar.markdown(f"Aktuelle Schwelle: **{confidence:.2f}**")


# --------- Modell lazy laden (hier Dummy, aber kompatibel) ---------
@st.cache_resource
def get_model():
    # Für diese Heuristik wird kein echtes Modell benötigt.
    # Die Funktion bleibt, damit der App-Code stabil ist.
    try:
        model = load_model(
            model_path=CONFIG.get("model_path", None),
            device=CONFIG.get("device", None),
        )
        return model
    except Exception as e:
        return e


# --------- Upload-Bereich ---------
st.markdown('<div class="upload-wrapper">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="upload-card">
      <div class="upload-icon">🩻</div>
      <div>
        <div class="upload-text-main">Röntgenbild hochladen</div>
        <div class="upload-text-sub">Ziehe ein Bild hierhin oder wähle eine Datei aus (JPG/PNG).</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded is None:
    st.info("Bitte ein Röntgenbild hochladen, um die Analyse zu starten.")
    st.stop()

# Datei-Inhalt EINMAL lesen
uploaded_bytes = uploaded.read()
if not uploaded_bytes:
    st.error("Die hochgeladene Datei ist leer oder konnte nicht gelesen werden.")
    st.stop()

# Bild aus Bytes erzeugen
try:
    img = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
except Exception as e:
    st.error(f"Bild kann nicht geladen werden: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Originalbild")
    st.image(img, use_container_width=True)

# Temporäre Datei für die Analyse anlegen
suffix = os.path.splitext(uploaded.name)[1] or ".png"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded_bytes)
    tmp_path = tmp.name

# Modell laden (Dummy oder Fehlerobjekt)
with st.spinner("Vorbereitung der Analyse…"):
    model_or_err = get_model()

if isinstance(model_or_err, Exception):
    st.error(f"Fehler beim Laden des Modells / der Heuristik: {model_or_err}")
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    st.stop()
else:
    model = model_or_err

st.divider()
st.subheader("Analyse")

try:
    with st.spinner("Heuristische Frakturerkennung wird ausgeführt…"):
        res = analyze_image(
            image_path=tmp_path,
            model=model,
            output_dir=CONFIG.get("output_dir"),
            confidence_threshold=confidence,
        )
except Exception as e:
    st.error(f"Fehler während der Analyse: {e}")
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    st.stop()

# Ergebnis auswerten
if res.get("status") != "SUCCESS":
    st.error("Analyse fehlgeschlagen: " + str(res.get("error_message", res)))
else:
    frac = res.get("fracture")
    score = res.get("score", 0.0)

    if frac:
        st.error(f"Ja: mögliche Fraktur erkannt (Heuristik-Score: {score:.3f}).")
    else:
        st.success(f"Nein: keine deutliche Frakturlinie erkannt (Score: {score:.3f}).")

    st.caption(
        "Hinweis: Die Entscheidung basiert auf einer einfachen Helligkeits-Heuristik "
        "entlang zentraler Bildachsen und ist nicht klinisch validiert."
    )

    out_img_path = res.get("output_image")
    if out_img_path and os.path.exists(out_img_path):
        with col2:
            st.subheader("Annotiertes Bild")
            st.image(out_img_path, use_container_width=True)
    else:
        st.caption("Kein annotiertes Bild verfügbar.")

# Temporäre Datei aufräumen
try:
    os.remove(tmp_path)
except OSError:
    pass
