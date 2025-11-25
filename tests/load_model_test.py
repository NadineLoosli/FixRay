from inference.predict_single_image import load_model, CONFIG
import traceback, sys
mp = CONFIG.get("model_path")
try:
    m = load_model(mp, device=CONFIG.get("device"))
    print("Model loaded OK:", type(m))
except Exception:
    traceback.print_exc()
    sys.exit(1)