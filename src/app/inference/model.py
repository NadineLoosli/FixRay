import torch
import torch.nn as nn
import torchvision.models as models

class FracAtlas(nn.Module):
    """
    ResNet50 backbone compatible with detection checkpoints that use 'backbone.body.*'.
    Head is a small classifier (num_classes=2).
    """
    def __init__(self, num_classes: int = 2, pretrained_backbone: bool = False):
        super().__init__()
        # use ResNet50 as backbone to match checkpoint bottleneck shapes
        backbone = models.resnet50(pretrained=pretrained_backbone)
        in_features = backbone.fc.in_features  # 2048 for resnet50
        backbone.fc = nn.Identity()  # remove original fc
        self.backbone = backbone
        # simple classification head
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)  # [B, 2048]
        out = self.head(feat)
        return out