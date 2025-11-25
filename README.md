# FixRay — KI-unterstützte Frakturerkennung

FixRay ist eine leichtgewichtige, prototypische Anwendung zur Erkennung von Knochenfrakturen auf Röntgenbildern.  
Die App nutzt ein **vortrainiertes ResNet18-Modell** (ImageNet) als Feature-Extractor und führt darauf basierende Heuristiken aus, um mögliche Frakturen zu identifizieren.

Die Anwendung ist als **Prototyp** zu verstehen und dient ausschliesslich zu Forschungs-, Lehr- und Demonstrationszwecken.  
**FixRay ist kein medizinisches Produkt und ersetzt keine ärztliche Diagnose.**

---
## Entstehungskontext 
Diesesd Projekt wurde im Rahmen des Moduls Software Engineering an der Fachhochschule Nordwestschweiz(FHNW) entwickelt. 
Ziel war es, einen lauffähigen Software-Prototypen zu konzipieren und umzusetzen,  
inklusive grundlegender **KI-Integration**, **Architektur**, **Dokumentation** 
und **User Interface**.


## Features

- Upload von **JPG/JPEG/PNG** über Drag & Drop
- Automatische Analyse mit ResNet18-Feature-Extraktion
- Visuelle Ausgabe des annotierten Ergebnisses
- Einfache Bedienung über **Streamlit Web-UI**
- Kompletter Offline-Prototyp (keine Cloud nötig)

---

## Modell & Analyseverfahren

- Backbone: **ResNet18 (ImageNet pretrained)**
- Verarbeitung über PyTorch
- Extraktion von Feature-Maps
- Heuristische Erkennung auffälliger Regionen
- Ausgabe:
  **„Fraktur vorhanden“** oder  
  **„Keine Fraktur erkannt“**


## Projektstruktur 

FixRay/
│
├── src/
│ ├── app/ # Streamlit App
│ │ └── main.py
│ ├── inference/ # Modell-Handling & Predictions
│ │ ├── model_loader.py
│ │ └── predict_single_image.py
│ └── utils/ # Hilfsfunktionen (Images, IO)
│
├── models/ # (Optional) Modell-Weights
├── results/ # Ausgabe-Bilder
├── requirements.txt
└── README.md

## Technologien 
-Python 3.10 
-Streamlist (UI)
-PyTorch
-Pillow
-NumPy


## Anwendung 
Repo klonen 
```bash
git clone https://github.com/NadineLoosli/FixRay.git
cd FixRay

Virtuelle Umgebung
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# oder
.venv\Scripts\activate     # Windows

pip install -r requirements.txt

START:
streamlit run src/app/main.py

Danach: 
http://localhost:8501
 im Browser öffnen.


