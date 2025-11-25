import sys
import torch

def inspect(path):
    obj = torch.load(path, map_location="cpu")
    print("Loaded type:", type(obj))
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print("Top-level keys (first 50):", keys[:50])
        if 'state_dict' in obj:
            sd = obj['state_dict']
            print("\nstate_dict keys (first 80):")
            for k in list(sd.keys())[:80]:
                v = sd[k]
                print(k, getattr(v, "shape", type(v)))
        else:
            print("\nstate_dict-like keys and shapes:")
            for k in keys[:200]:
                v = obj[k]
                print(k, getattr(v, "shape", type(v)))
    else:
        print("Not a dict. Object repr:", repr(obj)[:500])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <path_to_checkpoint>")
        sys.exit(1)
    inspect(sys.argv[1])