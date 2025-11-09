from pathlib import Path
from typing import Any
from PIL import Image
import numpy as np

# Try to import the model/inference API from your fracture-segmentation repo.
# Adjust import names below according to that package's public API.
try:
    # example: from fracture_segmentation.inference import load_model as fs_load_model, predict as fs_predict
    from fracture_segmentation import load_model as fs_load_model  # <- adjust if the package exposes a different symbol
    from fracture_segmentation import predict as fs_predict      # <- adjust if needed
    _HAS_FS = True
except (ImportError, ModuleNotFoundError):
    fs_load_model = None
    fs_predict = None
    _HAS_FS = False


def load_model(model_path: Path):
    """Wrapper: load model either via fracture-segmentation package or raise helpful error."""
    if _HAS_FS and fs_load_model is not None:
        return fs_load_model(model_path)
    raise NotImplementedError(
        "fracture-segmentation package not available. "
        "Install dependency (see requirements.txt) or use git submodule. "
        "If you intended to use a local copy, ensure PYTHONPATH includes it."
    )


def predict(image: Image.Image, model: Any) -> dict:
    """Wrapper: if fracture-segmentation provides predict, call it; otherwise return placeholder."""
    if _HAS_FS and fs_predict is not None:
        # pass through to the fracture-segmentation prediction function (adjust signature if needed)
        return fs_predict(image, model)
    arr = np.array(image)
    return {"shape": arr.shape, "note": "fracture-segmentation not installed; implement prediction."}
