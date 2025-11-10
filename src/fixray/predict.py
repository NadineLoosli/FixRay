from pathlib import Path
from PIL import Image
import numpy as np
import os

# If submodule is not installed, try to add libs/fracture-segmentation to PYTHONPATH at runtime
_submodule_path = Path(__file__).resolve().parents[2] / "libs" / "fracture-segmentation"
if _submodule_path.exists():
    import sys
    if str(_submodule_path) not in sys.path:
        sys.path.insert(0, str(_submodule_path))

# Try to import the model/inference API from fracture-segmentation.
try:
    from fracture_segmentation import load_model as fs_load_model
    from fracture_segmentation import predict as fs_predict
    _HAS_FS = True
except Exception:
    fs_load_model = None
    fs_predict = None
    _HAS_FS = False


def load_model(model_path: Path):
    """Wrapper: load model either via fracture-segmentation package or raise helpful error."""
    if _HAS_FS and fs_load_model is not None:
        return fs_load_model(model_path)
    raise NotImplementedError(
        "fracture-segmentation package not available. Initialize submodules with scripts/bootstrap_submodules.sh and install the package, or add the submodule path to PYTHONPATH."
    )


def predict(image: Image.Image, model) -> dict:
    """Wrapper: if fracture-segmentation provides predict, call it; otherwise return placeholder."""
    if _HAS_FS and fs_predict is not None:
        return fs_predict(image, model)
    arr = np.array(image)
    return {"shape": arr.shape, "note": "fracture-segmentation not available; implement prediction."}
