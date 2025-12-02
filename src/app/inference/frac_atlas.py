# src/inference/frac_atlas.py

import torch.nn as nn
import torchvision.models as models

class FracAtlas(nn.Module):
    """
    ResNet50 backbone, Klassifikations-Head mit num_classes Outputs.
    Für Frakturerkennung: num_classes=2 (0 = keine Fraktur, 1 = Fraktur).
    """
    def __init__(self, num_classes: int = 2, pretrained_backbone: bool = False):
        super().__init__()
        backbone = models.resnet50(pretrained=pretrained_backbone)
        in_features = backbone.fc.in_features  # 2048 für ResNet50
        backbone.fc = nn.Identity()            # Original-FC entfernen
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)  # [B, 2048]
        out = self.head(feat)    # [B, num_classes]
        return out
