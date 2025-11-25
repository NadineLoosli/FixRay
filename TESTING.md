# Teststrategie für FixRay (minimal)

Ziel
- Sicherstellen, dass die Kernlogik korrekt bleibt.
- Automatisierte Tests bei jedem Push / Pull Request ausführen.
- Deployment der statischen Seite (public/) auf GitHub Pages nach erfolgreichem Merge in main.

Umfang
- Unit-Tests: Einzelne Funktionen/Module isoliert testen (pytest).
- Optional: Integrationstests / Smoke-Tests für wichtige Abläufe.
- Optional: Statische Prüfungen (Linter, Typprüfung).

Konventionen
- Tests liegen im Verzeichnis 	ests/.
- Testdateien: 	est_*.py oder *_test.py.
- Testfunktionen: def test_*():.

Lokales Ausführen
- python -m venv .venv
- .venv\Scripts\Activate (Windows) oder source .venv/bin/activate (Unix)
- pip install -r requirements.txt (falls vorhanden)
- pip install pytest
- pytest -q
