# Modelle & Integration

Dieses Dokument beschreibt, wie FixRay das Modell aus dem separaten Repository `NadineLoosli/fracture-segmentation` nutzt.

Empfohlene Wege:
- Kurzfristig: pip-Installation als Git-Dependency (requirements.txt enthält: git+https://github.com/NadineLoosli/fracture-segmentation.git@main#egg=fracture_segmentation)
- Langfristig / Reproduzierbarkeit: Nutze Releases für Modellgewichte (GitHub Releases, S3 etc.) und lade Gewichte via Script in /models/.

Wichtige Punkte:
- API: Stelle sicher, dass `fracture-segmentation` eine saubere Inference-API exportiert (z.B. `load_model(path)` und `predict(image, model)`), damit FixRay-Wrapper sie ohne Änderungen aufrufen kann.
- Modellgewichte: Nicht im Repo speichern; stattdessen Releases oder Download-Skript nutzen.
- Lizenz: Prüfe Kompatibilität der Lizenz beider Repos.
