<#
  create-ci-and-tests.ps1
  Legt TESTING.md, .github\workflows\ci-pages.yml, tests\test_example.py, public\index.html an,
  legt Branch add-github-actions-ci an (oder wechselt zu ihm), committet und pusht zum Remote.
  Ausführen im Repo-Root (z. B. C:\Users\nadin\FixRay).
#>

$branch = "add-github-actions-ci"

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Prüfe, ob wir in einem Verzeichnis mit .git sind
if (-not (Test-Path .git)) {
  Write-Err "Kein Git-Repository hier. Wechsle in den geklonten Repo-Ordner oder clone das Repo zuerst."
  Write-Err "Beende."
  exit 1
}

# Prüfe git installiert
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  Write-Err "git ist nicht installiert oder nicht im PATH. Bitte installiere Git und versuche es erneut."
  exit 1
}

# Remote prüfen
$originUrl = ""
try { $originUrl = git remote get-url origin 2>$null } catch {}
if ($originUrl) {
  Write-Info "Remote 'origin' -> $originUrl"
  if ($originUrl -notmatch "github.com") {
    Write-Warn "Dein Remote 'origin' zeigt nicht auf GitHub. Wenn du GitHub Actions nutzen willst, stelle sicher, dass origin korrekt ist."
  }
} else {
  Write-Warn "Kein Remote 'origin' gefunden. Du kannst später 'git remote add origin <url>' ausführen."
}

# fetch
Write-Info "Hole remote refs (falls vorhanden)..."
git fetch origin --prune

# Branch-Logik: anlegen oder wechseln
$localExists = git show-ref --verify --quiet "refs/heads/$branch"; $localExists = $?
$remoteExists = $false
try {
  $ls = git ls-remote --heads origin $branch 2>$null
  if ($ls) { $remoteExists = $true }
} catch {}

if ($localExists) {
  Write-Info "Wechsle zu lokalem Branch '$branch'..."
  git checkout $branch
} elseif ($remoteExists) {
  Write-Info "Erstelle lokalen Tracking-Branch von origin/$branch..."
  git fetch origin $branch
  git checkout -b $branch origin/$branch
} else {
  Write-Info "Erstelle neuen lokalen Branch '$branch'..."
  git checkout -b $branch
}

# Verzeichnisse anlegen
Write-Info "Erstelle Verzeichnisse..."
New-Item -ItemType Directory -Force -Path ".github\workflows" | Out-Null
New-Item -ItemType Directory -Force -Path "tests" | Out-Null
New-Item -ItemType Directory -Force -Path "public" | Out-Null

# Dateien schreiben (überschreibt vorhandene)
Write-Info "Schreibe .github\workflows\ci-pages.yml..."
@"
name: CI & Deploy to GitHub Pages

on:
  push:
    branches:
      - '**'
  pull_request:
    branches:
      - '**'

jobs:
  test:
    name: Run tests
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          pip install pytest

      - name: Run tests
        run: pytest -q

  deploy:
    name: Deploy to GitHub Pages
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      pages: write
      id-token: write
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Build (no-op by default)
        run: |
          echo "No build step configured; ensure public/ contains index.html"

      - name: Upload artifact for GitHub Pages
        uses: actions/upload-pages-artifact@v1
        with:
          path: ./public

      - name: Deploy to GitHub Pages
        uses: actions/deploy-pages@v1
"@ | Set-Content -Path ".github\workflows\ci-pages.yml" -Encoding UTF8

Write-Info "Schreibe TESTING.md..."
@"
# Teststrategie für FixRay (minimal)

Ziel
- Sicherstellen, dass die Kernlogik korrekt bleibt.
- Automatisierte Tests bei jedem Push / Pull Request ausführen.
- Deployment der statischen Seite (`public/`) auf GitHub Pages nach erfolgreichem Merge in `main`.

Umfang
- Unit-Tests: Einzelne Funktionen/Module isoliert testen (pytest).
- Optional: Integrationstests / Smoke-Tests für wichtige Abläufe.
- Optional: Statische Prüfungen (Linter, Typprüfung).

Konventionen
- Tests liegen im Verzeichnis `tests/`.
- Testdateien: `test_*.py` oder `*_test.py`.
- Testfunktionen: `def test_*():`.

Lokales Ausführen
- python -m venv .venv
- .venv\Scripts\Activate (Windows)
- pip install -r requirements.txt (falls vorhanden)
- pip install pytest
- pytest -q
"@ | Set-Content -Path "TESTING.md" -Encoding UTF8

Write-Info "Schreibe tests\test_example.py..."
@"
def test_basic_sanity():
    assert 1 + 1 == 2
"@ | Set-Content -Path "tests\test_example.py" -Encoding UTF8

Write-Info "Schreibe public\index.html..."
@"
<!doctype html>
<html lang=""de"">
<head>
  <meta charset=""utf-8"" />
  <title>FixRay — GitHub Pages (minimal)</title>
</head>
<body>
  <h1>FixRay — minimal static site</h1>
  <p>Dies ist eine minimal vorhandene Seite, damit GitHub Pages etwas veröffentlichen kann.</p>
</body>
</html>
"@ | Set-Content -Path "public\index.html" -Encoding UTF8

# Gitignore-Erweiterung (results/ / logs)
if (-not (Test-Path .gitignore)) { "" | Out-File .gitignore -Encoding UTF8 }
if (-not (Select-String -Path .gitignore -Pattern '^results/' -SimpleMatch -Quiet)) {
  Add-Content -Path .gitignore -Value "results/"
}
if (-not (Select-String -Path .gitignore -Pattern '^streamlit.log' -SimpleMatch -Quiet)) {
  Add-Content -Path .gitignore -Value "streamlit.log"
}

# Stage/commit nur relevanter Dateien
Write-Info "Stage relevante Dateien..."
git add .github\workflows\ci-pages.yml TESTING.md tests\test_example.py public\index.html .gitignore

$hasChanges = git status --porcelain
if (-not [string]::IsNullOrWhiteSpace($hasChanges)) {
  Write-Info "Committe Änderungen..."
  git commit -m "Add test strategy, example pytest test and GitHub Actions CI + Pages deploy"
} else {
  Write-Info "Keine Änderungen zum commiten."
}

# Push
try {
  Write-Info "Pushe Branch zu origin/$branch ..."
  git push -u origin $branch
  Write-Info "Push erfolgreich."
} catch {
  Write-Err "Push fehlgeschlagen: $($_.Exception.Message)"
  Write-Warn "Wenn origin nicht gesetzt ist: git remote add origin https://github.com/<owner>/<repo>.git"
  exit 1
}

Write-Info "Fertig. Erstelle nun auf GitHub einen Pull Request add-github-actions-ci -> main."