from bla.src_01.app.inference.predict_single_image import CONFIG
from PIL import Image
from torchvision import transforms
import torch, sys, traceback, numpy as np, inspect, importlib

def run_debug(image_path):
    model_path = CONFIG.get("model_path")
    device = CONFIG.get("device", torch.device("cpu"))
    print("Model:", model_path, "device:", device)

    # find loader in module
    mod = importlib.import_module("src.inference.predict_single_image")
    loader = None
    for name in ("load_model", "get_model", "get_model_cached"):
        fn = getattr(mod, name, None)
        if callable(fn):
            loader = fn
            break
    if loader is None:
        print("Kein Model-Loader (load_model/get_model/get_model_cached) in inference.predict_single_image gefunden.")
        return

    m = None
    # try calling with sensible signatures
    errors = []
    try:
        sig = inspect.signature(loader)
        params = sig.parameters
        # prefer (model_path, device=...)
        if ("model_path" in params or len(params) >= 1) and "device" in params:
            try:
                m = loader(model_path, device=device)
            except Exception as e:
                errors.append(("model_path,device", repr(e)))
        if m is None and ("model_path" in params or len(params) >= 1):
            try:
                m = loader(model_path)
            except Exception as e:
                errors.append(("model_path", repr(e)))
        if m is None and "device" in params:
            try:
                m = loader(device=device)
            except Exception as e:
                errors.append(("device", repr(e)))
        if m is None:
            try:
                m = loader()
            except Exception as e:
                errors.append(("no-args", repr(e)))
    except Exception as e:
        errors.append(("signature-inspect", repr(e)))

    # final brute-force attempts if still None
    if m is None:
        for attempt in (
            (model_path, device),
            (model_path,),
            (device,),
            (),
        ):
            try:
                if len(attempt) == 2:
                    m = loader(attempt[0], device=attempt[1])
                elif len(attempt) == 1:
                    try:
                        m = loader(attempt[0])
                    except TypeError:
                        m = loader(device=attempt[0])
                else:
                    m = loader()
                break
            except Exception as e:
                errors.append((attempt, repr(e)))
                m = None

    if m is None:
        print("Failed to call model loader with tried signatures. Errors:")
        for e in errors[:10]:
            print(e)
        traceback.print_stack()
        return

    # do inference
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        print("Konnte Bild nicht öffnen:", image_path)
        traceback.print_exc()
        return

    x = transform(img).unsqueeze(0).to(device)
    m.eval()
    with torch.no_grad():
        try:
            outputs = m(x)
        except Exception:
            traceback.print_exc()
            return

    if not outputs:
        print("No outputs from model.")
        return
    out = outputs[0]
    for k in ("boxes","scores","labels","masks"):
        v = out.get(k, None)
        if v is None:
            print(f"{k}: None")
            continue
        try:
            arr = v.detach().cpu().numpy()
            sample = arr[:5].tolist() if arr.size > 0 else []
            print(f"{k}: shape={arr.shape} sample={sample}")
        except Exception:
            print(f"{k}: type={type(v)} repr={v}")

    if out.get("labels", None) is not None:
        labs = out["labels"].detach().cpu().numpy()
        uniq, cnt = np.unique(labs, return_counts=True)
        print("Unique labels:", dict(zip(uniq.tolist(), cnt.tolist())))

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_inference.py path/to/image.jpg")
        sys.exit(1)
    run_debug(sys.argv[1])

cd C:\Users\nadin\FixRay
& .\.venv\Scripts\Activate.ps1
streamlit run src\app\main.py --server.address 127.0.0.1 --server.port 8505 --logger.level debug
Start-Process "http://127.0.0.1:8505"