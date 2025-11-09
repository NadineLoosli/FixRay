# ============================================================================
# FixRay - Frakturerkennungssystem
# Bildanalyse mit Mask R-CNN für medizinisches Fachpersonal
# ============================================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from datetime import datetime
import logging

# ============================================================================
# 1. LOGGING-KONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fixray_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 2. KONFIGURATION
# ============================================================================
CONFIG = {
    'num_classes': 2,  # Klasse 0: Hintergrund, Klasse 1: Fraktur
    'confidence_threshold': 0.5,
    'image_size': 800,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'model_path': r'../fracture-segmentation/fracture_detection_model_final.pth',
    'input_dir': r'C:/Users/nadin/FixRay/data/raw',
    'output_dir': r'C:/Users/nadin/FixRay/results',
}

logger.info(f"Device: {CONFIG['device']}")
logger.info(f"Model Path: {CONFIG['model_path']}")

# ============================================================================
# 3. MODELLAUFBAU
# ============================================================================
def get_model(num_classes):
    """
    Erstellt ein Mask R-CNN Modell mit ResNet-50 Backbone.
    
    Args:
        num_classes: Anzahl der Klassen (z.B. 2 für Fraktur/Nicht-Fraktur)
    
    Returns:
        Initialisiertes Mask R-CNN Modell
    """
    logger.info("Initialisiere Modell...")
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Box Predictor anpassen
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Mask Predictor anpassen
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 
        hidden_layer, 
        num_classes
    )
    
    logger.info("Modell erfolgreich initialisiert")
    return model

# ============================================================================
# 4. MODELL LADEN
# ============================================================================
def load_model(model_path, device):
    """
    Lädt trainiertes Modell von Disk.
    
    Args:
        model_path: Pfad zum gespeicherten Modell
        device: CPU oder CUDA
    
    Returns:
        Geladenes Modell im Evaluationsmodus
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")
    
    logger.info(f"Lade Modell von: {model_path}")
    
    model = get_model(CONFIG['num_classes'])
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    model.to(device)
    
    logger.info("Modell erfolgreich geladen")
    return model

# ============================================================================
# 5. BILDVORVERARBEITUNG
# ============================================================================
def prepare_image(image_path, image_size=800):
    """
    Lädt und verarbeitet ein Bild für das Modell.
    
    Args:
        image_path: Pfad zum Bild
        image_size: Größe zum Skalieren
    
    Returns:
        (torch.Tensor, PIL.Image): Vorverarbeiteter Tensor und Original-PIL-Bild
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    
    logger.info(f"Lade Bild: {image_path}")
    
    # Bild laden und zu RGB konvertieren
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    # Größe ändern
    image = image.resize((image_size, image_size))
    
    # Normalisierung (ImageNet-Standard)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(CONFIG['device'])
    
    logger.info(f"Bild verarbeitet: Original {original_size} → {image_size}x{image_size}")
    
    return input_tensor, image

# ============================================================================
# 6. INFERENZ (VORHERSAGE)
# ============================================================================
def inference(model, input_tensor):
    """
    Führt Inference durch.
    
    Args:
        model: Trainiertes Modell
        input_tensor: Eingabebild als Tensor
    
    Returns:
        dict: Modell-Output (boxes, labels, scores, masks)
    """
    logger.info("Starte Inferenz...")
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Ergebnisse von GPU zu CPU verschieben
    results = {
        'masks': output[0]['masks'].cpu().numpy(),
        'boxes': output[0]['boxes'].cpu().numpy(),
        'labels': output[0]['labels'].cpu().numpy(),
        'scores': output[0]['scores'].cpu().numpy(),
    }
    
    logger.info(f"Inferenz abgeschlossen. Erkannte Objekte: {len(results['scores'])}")
    
    return results

# ============================================================================
# 7. ERGEBNISAUSWERTUNG
# ============================================================================
def filter_results(results, confidence_threshold=0.5):
    """
    Filtert Ergebnisse nach Confidence-Schwellenwert.
    
    Args:
        results: Modell-Output
        confidence_threshold: Minimaler Confidence-Score
    
    Returns:
        dict: Gefilterte Ergebnisse
    """
    valid = results['scores'] > confidence_threshold
    
    filtered = {
        'masks': results['masks'][valid],
        'boxes': results['boxes'][valid],
        'labels': results['labels'][valid],
        'scores': results['scores'][valid],
    }
    
    logger.info(f"Nach Filterung ({confidence_threshold}): {len(filtered['scores'])} Objekte")
    
    return filtered

# ============================================================================
# 8. VISUALISIERUNG
# ============================================================================
def visualize_results(image, filtered_results, output_path, confidence_threshold=0.5):
    """
    Visualisiert Erkennungsergebnisse mit Bounding Boxes und Masken.
    
    Args:
        image: PIL-Bild
        filtered_results: Gefilterte Ergebnisse
        output_path: Speicherpfad für das Ergebnis
        confidence_threshold: Schwellenwert
    """
    logger.info(f"Visualisiere Ergebnisse...")
    
    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(image)
    ax.axis('off')
    
    num_fractures = len(filtered_results['scores'])
    
    # Zeichne Bounding Boxes und Masken
    for i in range(num_fractures):
        score = filtered_results['scores'][i]
        box = filtered_results['boxes'][i]
        mask = filtered_results['masks'][i, 0] > 0.5
        
        # Bounding Box zeichnen (rot)
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=3,
            edgecolor='red',
            facecolor='none',
            label='Fraktur erkannt' if i == 0 else ''
        )
        ax.add_patch(rect)
        
        # Maske als Overlay (rotes Overlay mit Transparenz)
        masked_overlay = np.zeros((*mask.shape, 4))
        masked_overlay[mask] = [1, 0, 0, 0.4]  # Rot mit 40% Transparenz
        ax.imshow(masked_overlay, interpolation='nearest')
        
        # Confidence-Score anzeigen
        ax.text(
            box[0],
            box[1] - 10,
            f'Fraktur: {score:.2%}',
            color='white',
            fontsize=11,
            weight='bold',
            bbox=dict(facecolor='red', alpha=0.8, edgecolor='white', linewidth=1)
        )
    
    # Titel
    if num_fractures > 0:
        title = f"✓ Fraktur erkannt! ({num_fractures} Treffer)"
        color = 'green'
    else:
        title = "✗ Keine Fraktur erkannt"
        color = 'gray'
    
    ax.set_title(title, fontsize=16, weight='bold', color=color, pad=20)
    
    # Speichern
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    logger.info(f"Ergebnis gespeichert: {output_path}")
    plt.close()

# ============================================================================
# 9. HAUPT-ANALYSESFUNKTION
# ============================================================================
def analyze_image(image_path, model, output_dir, confidence_threshold=0.5):
    """
    Komplette Bildanalyse-Pipeline.
    
    Args:
        image_path: Pfad zum Eingabebild
        model: Trainiertes Modell
        output_dir: Ausgabeverzeichnis
        confidence_threshold: Confidence-Schwellenwert
    
    Returns:
        dict: Analyseergebnisse
    """
    logger.info("=" * 60)
    logger.info(f"Starte Bildanalyse: {os.path.basename(image_path)}")
    logger.info("=" * 60)
    
    try:
        # Bild vorbereiten
        input_tensor, image = prepare_image(image_path)
        
        # Inferenz durchführen
        results = inference(model, input_tensor)
        
        # Ergebnisse filtern
        filtered_results = filter_results(results, confidence_threshold)
        
        # Visualisieren
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"fracture_analysis_{timestamp}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        visualize_results(image, filtered_results, output_path, confidence_threshold)
        
        # Zusammenfassung
        summary = {
            'image_path': image_path,
            'timestamp': timestamp,
            'fractures_detected': len(filtered_results['scores']),
            'confidence_scores': filtered_results['scores'].tolist(),
            'output_image': output_path,
            'status': 'SUCCESS'
        }
        
        logger.info(f"Analyse abgeschlossen: {summary['fractures_detected']} Fraktur(en) erkannt")
        logger.info("=" * 60)
        
        return summary
        
    except Exception as e:
        logger.error(f"Fehler bei Bildanalyse: {str(e)}", exc_info=True)
        return {
            'image_path': image_path,
            'status': 'ERROR',
            'error_message': str(e)
        }

# ============================================================================
# 10. HAUPTPROGRAMM
# ============================================================================
if __name__ == "__main__":
    logger.info("FixRay - Frakturerkennungssystem startet...")
    
    # Verzeichnisse erstellen
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Modell laden
    model = load_model(CONFIG['model_path'], CONFIG['device'])
    
    # Beispiel: Einzelnes Bild analysieren
    image_path = os.path.join(CONFIG['input_dir'], 'deinbild.jpg')
    
    if os.path.exists(image_path):
        result = analyze_image(
            image_path,
            model,
            CONFIG['output_dir'],
            confidence_threshold=CONFIG['confidence_threshold']
        )
        print("\n" + "=" * 60)
        print("ANALYSEERGEBNIS:")
        print("=" * 60)
        print(f"Status: {result['status']}")
        print(f"Frakturen erkannt: {result.get('fractures_detected', 'N/A')}")
        if result['status'] == 'SUCCESS':
            print(f"Konfidenz-Scores: {[f'{s:.2%}' for s in result['confidence_scores']]}")
            print(f"Ergebnis-Bild: {result['output_image']}")
        print("=" * 60)
    else:
        logger.error(f"Eingabebild nicht gefunden: {image_path}")
    
    # Optional: Alle Bilder in Verzeichnis analysieren
    # for filename in os.listdir(CONFIG['input_dir']):
    #     if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
    #         full_path = os.path.join(CONFIG['input_dir'], filename)
    #         analyze_image(full_path, model, CONFIG['output_dir'])
