# FixRay

Kurzbeschreibung
- Das bestehende Problem liegt in der zeitintensiven und hohen Fachkompetenz erfordernden Diagnostik von Frakturen. Dies führt zu langen Wartezeiten, einem erhöhten Fehlerrisiko bei der Diagnose und kann im schlimmsten Fall Komplikationen für die Patientinnen und Patienten zur Folge haben.
- Die Frakturdiagnostik soll beschleunigt und die diagnostische Genauigkeit verbessert werden – kürzere Wartezeiten, weniger Fehler, bessere Versorgung der Patientinnen und Patienten.
- FixRay unterstützt medizinisches Fachpersonal durch eine intelligente Kombination aus KI-basierter Bild- und Textanalyse. Durch den Einsatz eines Large Language Models (LLM) werden Anamnese und klinische Befunde erfasst, interpretiert und in den diagnostischen Prozess integriert. Eine KI-basierte Bildanalyse erkennt und klassifiziert potenzielle Frakturen auf Röntgenbildern und schlägt basierend darauf die wahrscheinlichsten Frakturtypen und deren AO-Klassifikation vor.
- Im Gegensatz zum manuellen Vorgehen legt FixRay den Fokus auf Effizienzsteigerung: Das Gesundheitspersonal gewinnt durch den Gebrauch unserer Software an Zeit, profitiert von einer niedrigeren Fehlerquote und kann sich stärker auf Fortbildung und komplexe Fälle konzentrieren.


Status
- Projekt steht in der Anfangsphase (Initial-Setup).
- Noch kein fertiger Code.

Schnellstart (statische Seite)
1. Repository klonen:
```bash
git clone https://github.com/NadineLoosli/FixRay.git
cd FixRay
```
2. Dateien erstellen (falls noch nicht vorhanden) und pushen:
```bash
git add .
git commit -m "Initial project files"
git push origin main
```
3. Lokale Vorschau (einfach):
```bash
# mit Python
python3 -m http.server 8000
# dann im Browser öffnen: http://localhost:8000
```

GitHub Pages
- Wenn du eine Projektseite willst: in den Repository Settings → Pages den Branch `main` und Ordner `/ (root)` auswählen (oder `docs/`), speichern. `index.html` muss im Root oder in `docs/` liegen.

Mitwirken
- Issues: benutze die Issues, um Aufgaben oder Bugs zu erfassen.
- Pull Requests: Fork → Branch → PR erstellen.

Wichtige Links
- Repo: https://github.com/NadineLoosli/FixRay

Kontakt
- Autor: Nadine Loosli
- Coautor: Anna Horvath
- Coautor: Sura Karaburun

### Markdown
[Cheat Sheet](https://www.markdownguide.org/cheat-sheet/)

### Git Cheat Sheet
[Command and workflow summary](https://krishnaiitd.github.io/gitcommands/git-workflow/)
- `git status`: Show status of the local repository
- `git log`: Show commit history (press `q` to exit)
- `git pull`: Download current status from GitHub, update local files (may lead to conflicts...)
- `git push`: Upload all local commits to GitHub (may lead to conflicts...)
- `git add`: Mark files as "to be committed" (either `git add FILE` or `git add .` to add all files)
- `git commit -m "MESSAGE"`: Create a new commit with the given message
  - If you forgot the `-m "MESSAGE"`, git might open a new text file in the "vim" text editor. Exit it with ":q!".
