import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import Flowers102

from flowers.model import ARTIFACTS, BACKBONE, DEVICE, HEAD_PATH, METRICS_PATH, build_head, build_transform, load_backbone

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))


@torch.no_grad()
def embed(backbone: nn.Module, split: str) -> TensorDataset:
    dataset = Flowers102(DATA_DIR, split=split, transform=build_transform(backbone), download=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=min(4, os.cpu_count()))
    features, labels = zip(*((backbone(images.to(DEVICE)), targets) for images, targets in loader))
    print(f"embedded {split}: {len(dataset)} images", flush=True)
    return TensorDataset(torch.cat(features), torch.cat(labels).to(DEVICE))


@torch.no_grad()
def accuracy(head: nn.Module, data: TensorDataset) -> float:
    features, labels = data.tensors
    return (head(features).argmax(dim=-1) == labels).float().mean().item()


def main():
    torch.manual_seed(0)
    backbone = load_backbone()
    train, val = embed(backbone, "train"), embed(backbone, "val")
    head = build_head(backbone)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, EPOCHS + 1):
        head.train()
        for features, labels in DataLoader(train, batch_size=BATCH_SIZE, shuffle=True):
            loss = loss_fn(head(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        head.eval()
        print(f"epoch {epoch:>2}/{EPOCHS}  loss {loss.item():.3f}  "
              f"train acc {accuracy(head, train):.3f}  val acc {accuracy(head, val):.3f}", flush=True)

    ARTIFACTS.mkdir(exist_ok=True)
    torch.save(head.state_dict(), HEAD_PATH)
    metrics = {"backbone": BACKBONE, "epochs": EPOCHS, "train_images": len(train), "val_accuracy": round(accuracy(head, val), 4)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"saved {HEAD_PATH} {metrics}")


if __name__ == "__main__":
    main()
