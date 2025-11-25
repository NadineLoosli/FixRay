import sys, torch
from src.inference.model import FracAtlas

path = sys.argv[1]
obj = torch.load(path, map_location="cpu")
sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
print("Loaded checkpoint type:", type(obj))
try:
    model = FracAtlas()
    model.load_state_dict(sd)
    print("state_dict loaded into FracAtlas OK")
except Exception as e:
    print("load_state_dict failed:", e)
    # print first 120 keys for diagnosis
    if isinstance(sd, dict):
        print("state_dict keys (first 120):")
        for k in list(sd.keys())[:120]:
            v = sd[k]
            print(k, getattr(v, "shape", type(v)))