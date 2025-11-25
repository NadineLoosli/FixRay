# shim: importiere predict_single_image aus src Package

# Minimal stub to avoid import recursion and allow the app to run.
import os
import importlib
from typing import Any, Dict
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Lazy import torch to avoid import errors at module import time in the editor
def _torch():
    import torch
    return torch

CONFIG = {
    "model_path": os.path.abspath(os.path.join(os.getcwd(), "fracture-segmentation", "fracture_detection_model_final.pth")),
    "device": "cpu",
    "confidence_threshold": 0.5,
    "output_dir": os.path.join(os.getcwd(), "results"),
}

def load_model(model_path: str = None, device: Any = None):
    import importlib
    torch = _torch()
    device = torch.device(device or CONFIG.get("device") or "cpu")
    model_path = model_path or CONFIG.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 1) try scripted model
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.to(device)
        model.eval()
        return model
    except Exception:
        pass

    # 2) try torch.load (module instance or checkpoint dict)
    try:
        obj = torch.load(model_path, map_location=device)
        # if it's an nn.Module instance
        if hasattr(obj, "eval") and hasattr(obj, "state_dict"):
            obj.to(device)
            obj.eval()
            return obj

        # if it's a dict/checkpoint: accept several common keys
        if isinstance(obj, dict):
            # common places: 'model_state_dict', 'state_dict', 'model'
            sd = None
            for k in ("model_state_dict", "state_dict", "model"):
                if k in obj:
                    sd = obj[k]
                    break
            # if sd still None but obj looks like state_dict (OrderedDict) use obj itself
            if sd is None:
                # maybe the dict is directly a state_dict-like (keys are strings mapping to tensors)
                # heuristics: first value has 'shape' attribute
                first_val = next(iter(obj.values()))
                if hasattr(first_val, "shape"):
                    sd = obj

            if sd is not None and isinstance(sd, dict):
                # try to import FracAtlas class from src.inference.model
                try:
                    mod = importlib.import_module("src.inference.model")
                    cls = getattr(mod, "FracAtlas", None)
                    if cls is None:
                        raise RuntimeError("src.inference.model.FracAtlas not found")
                    model = cls()

                    # map checkpoint keys to model keys
                    from collections import OrderedDict
                    mapped_sd = OrderedDict()
                    for k, v in sd.items():
                        new_k = k
                        # remove common dataparallel prefix
                        if new_k.startswith("module."):
                            new_k = new_k[len("module."):]
                        # map torchvision/retinanet/maskrcnn backbone naming to simple backbone.* used in FracAtlas
                        new_k = new_k.replace("backbone.body.", "backbone.")
                        # if checkpoint uses .body and model expects .backbone, also handle other variants
                        new_k = new_k.replace("backbone.body.", "backbone.")
                        # keep other keys unchanged
                        mapped_sd[new_k] = v

                    # load with strict=False to ignore unexpected keys (e.g. RPN/ROI heads) and accept missing keys
                    try:
                        load_res = model.load_state_dict(mapped_sd, strict=False)
                    except Exception as e:
                        raise RuntimeError(f"load_state_dict failed: {e}")

                    model.to(device)
                    model.eval()
                    return model
                except Exception as e:
                    raise RuntimeError(f"Failed to instantiate FracAtlas and load state_dict: {e}")

            raise RuntimeError("Checkpoint dict found but no recognizable state_dict inside. Keys: " + ", ".join(list(obj.keys())[:50]))
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    raise RuntimeError("Could not load model from path: " + str(model_path))


def _preprocess_image_pil(img: Image.Image, input_size=224):
    import torchvision.transforms as T
    transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transforms(img).unsqueeze(0)


def analyze_image(image_path: str, model, output_dir: str = None, confidence_threshold: float = None) -> Dict:
    """
    Run single-image inference and produce an annotated image.
    Returns dict with keys: status, output_image, fracture (bool), score (float)
    """
    torch = _torch()
    output_dir = output_dir or CONFIG.get("output_dir") or "."
    confidence_threshold = confidence_threshold if confidence_threshold is not None else CONFIG.get("confidence_threshold", 0.5)
    os.makedirs(output_dir, exist_ok=True)

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"status": "ERROR", "error_message": f"Cannot open image: {e}"}

    x = _preprocess_image_pil(img)  # shape [1,C,H,W]
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device(CONFIG.get("device","cpu"))
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        out = model(x)

    # normalize output to a single score in [0,1] representing fracture probability
    score = None
    try:
        if isinstance(out, dict):
            # if model returns dict with logits/scores
            if "scores" in out:
                arr = out["scores"]
                if hasattr(arr, "detach"):
                    arr = arr.detach().cpu().numpy()
                score = float(np.max(arr))
            elif "logits" in out:
                logits = out["logits"]
                if hasattr(logits, "detach"):
                    logits = logits.detach().cpu().numpy()
                logits = np.asarray(logits)
                if logits.ndim == 2 and logits.shape[1] == 2:
                    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
                    score = float(probs[0,1])
                else:
                    score = float(1/(1+np.exp(-logits.ravel()[0])))
        else:
            # assume tensor
            if hasattr(out, "detach"):
                t = out.detach().cpu().numpy()
            else:
                t = np.asarray(out)
            t = np.asarray(t)
            if t.ndim == 2 and t.shape[1] == 2:
                ex = np.exp(t)
                probs = ex / ex.sum(axis=1, keepdims=True)
                score = float(probs[0,1])
            else:
                score = float(1/(1+np.exp(-t.ravel()[0])))
    except Exception:
        score = None

    if score is None:
        return {"status": "ERROR", "error_message": "Could not interpret model output. Please adapt analyze_image to your model's output."}

    fracture = bool(score >= float(confidence_threshold))

    # annotate image: draw label and score
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except Exception:
        font = ImageFont.load_default()
    label = f"Fraktur: {'JA' if fracture else 'NEIN'}  ({score:.2f})"
    margin = 8
    x0, y0 = 10, 10

    # compute text size in a Pillow-version-robust way
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except Exception:
        try:
            text_width, text_height = font.getsize(label)
        except Exception:
            text_width, text_height = (len(label) * 7, 14)

    draw.rectangle([x0 - margin, y0 - margin, x0 + text_width + margin, y0 + text_height + margin], fill=(0, 0, 0))
    draw.text((x0, y0), label, fill=(255, 255, 255), font=font)

    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_annotated.jpg")
    img.save(out_path, quality=90)

    return {
        "status": "SUCCESS",
        "output_image": out_path,
        "fracture": fracture,
        "score": float(score),
        "confidence_threshold": float(confidence_threshold),
    }
