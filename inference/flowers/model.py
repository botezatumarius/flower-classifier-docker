import json
import os
from pathlib import Path

import timm
import torch
from PIL import Image
from torch import nn

from flowers.classes import CLASSES

BACKBONE = os.getenv("BACKBONE", "vit_small_patch14_dinov2.lvd142m")
IMAGE_SIZE = 224
ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
HEAD_PATH = ARTIFACTS / "head.pt"
METRICS_PATH = ARTIFACTS / "metrics.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_backbone() -> nn.Module:
    return timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=IMAGE_SIZE).to(DEVICE).eval()


def build_transform(backbone: nn.Module):
    config = timm.data.resolve_data_config({"input_size": (3, IMAGE_SIZE, IMAGE_SIZE)}, model=backbone)
    return timm.data.create_transform(**config)


def build_head(backbone: nn.Module) -> nn.Linear:
    return nn.Linear(backbone.num_features, len(CLASSES)).to(DEVICE)


class FlowerClassifier:
    def __init__(self):
        self.backbone = load_backbone()
        self.transform = build_transform(self.backbone)
        self.head = build_head(self.backbone).eval()
        self.head.load_state_dict(torch.load(HEAD_PATH, map_location=DEVICE))
        self.info = {**json.loads(METRICS_PATH.read_text()), "device": str(DEVICE)}

    @torch.inference_mode()
    def predict(self, image: Image.Image, k: int = 5) -> list[dict]:
        batch = self.transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        scores, indices = self.head(self.backbone(batch)).softmax(dim=-1)[0].topk(k)
        return [{"label": CLASSES[i], "score": round(s, 4)} for s, i in zip(scores.tolist(), indices.tolist())]
