import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Modell laden (angepasst an deine Speicherstruktur)
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

# Schritt 1: Modell initialisieren und gewichte laden
num_classes = 2
model = get_model(num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state = torch.load('fracture_detection_model_final.pth', map_location=device)
model.load_state_dict(state['model_state_dict'])
model.eval()
model.to(device)

# Schritt 2: Bild laden und vorbereiten
img_path = 'C:/Users/nadin/FixRay/data/raw/deinbild.jpg'   # Pfad anpassen!
image = Image.open(img_path).convert("RGB")
image = image.resize((800, 800))
image_np = np.array(image)/255.0
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0).to(device)

# Schritt 3: Vorhersage
with torch.no_grad():
    output = model(input_tensor)[0]

# Schritt 4: Ergebnisse interpretieren
# Zeige Maske für das erste/vorhandene Objekt
masks = output['masks'].cpu().numpy()
boxes = output['boxes'].cpu().numpy()
labels = output['labels'].cpu().numpy()
scores = output['scores'].cpu().numpy()

# Beispiel: Anzeigen der ersten erkannten Maske (falls vorhanden)
if len(masks) > 0 and scores[0] > 0.5:
    plt.imshow(image)
    plt.imshow(masks[0,0], alpha=0.5, cmap='jet')
    plt.title(f'Score: {scores[0]:.2f}, Label: {labels[0]}')
    plt.axis('off')
    plt.show()
else:
    print('Keine Fraktur mit ausreichend hohem Score erkannt.')
print(f"Anzahl gefundene Objekte: {len(masks)}")
print(f"Scores: {scores}")