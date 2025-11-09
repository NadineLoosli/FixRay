# Modelle & Integration

Dieses Dokument beschreibt, wie FixRay das Modell aus dem separaten Repository `NadineLoosli/fracture-segmentation` nutzt.

Empfohlene Vorgehensweise (Submodule):
- Das Repository `fracture-segmentation` ist als Git-Submodule unter `libs/fracture-segmentation` eingebunden.
- Nach dem Clone des FixRay-Repos führe aus:

```
# initial clone
git clone --recurse-submodules https://github.com/NadineLoosli/FixRay.git
# oder wenn bereits geklont:
./scripts/bootstrap_submodules.sh
```

- Falls das Submodule ein paketierbares Python-Paket (setup.py/pyproject.toml) enthält, installiert das Bootstrapping-Skript es editable mittels pip und macht die API importierbar.

Wichtige Hinweise:
- Achte auf Lizenzkompatibilität zwischen beiden Repositories.
- Modellgewichte sollten nicht direkt im Repo liegen; verwende Releases/S3 oder ein Download-Skript und den Ordner `/models/`.
