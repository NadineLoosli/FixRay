from pathlib import Path
from PIL import Image
import numpy as np


def load_model(model_path: Path):
    """Platzhalter: Modelldatei laden.
    Implementiere das Laden für PyTorch / TensorFlow falls gewünscht.
    """
    if not model_path.exists():
        raise NotImplementedError("Model loading not implemented. Provide a model file or implement load_model().")
    # TODO: add real loading code (torch.load, tf.keras.models.load_model, ...)
    return None


def predict(image: Image.Image, model) -> dict:
    """Platzhalter-Predict: gibt Bildgröße zurück.
    Ersetze mit echter Preprocessing / Modellinferenz.
    """
    arr = np.array(image)
    return {"shape": arr.shape, "note": "Implement prediction with your model."}
