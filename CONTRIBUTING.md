Danke, dass du mithelfen möchtest!

Kurze Anleitung:
1. Issue anlegen oder ein vorhandenes Issue übernehmen.
2. Fork → branch erstellen (z. B. fix/kurze-beschreibung) → Änderungen committen.
3. Pull Request öffnen und im PR-Text beschreiben, was du gemacht hast.

Beispiel-Workflow (für Anfänger):
- Forke das Repo (oben rechts auf GitHub auf "Fork" klicken).
- Lokales Klonen:
  git clone https://github.com/<dein-benutzername>/FixRay.git
  cd FixRay
- Neuen Branch erstellen:
  git checkout -b fix/mein-fix
- Änderungen machen, committen und pushen:
  git add .
  git commit -m "Kurzbeschreibung"
  git push origin fix/mein-fix
- Erstelle einen Pull Request auf GitHub.

Style & Tests (optional)
- Falls später Code hinzukommt, hier beschreiben: Linter, Code-Style, Tests, Node/Python-Versionen.