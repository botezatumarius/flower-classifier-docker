---
title: Flower Classifier API
emoji: 🌸
colorFrom: pink
colorTo: green
sdk: docker
app_port: 8000
pinned: false
short_description: DINOv2 ViT-S/14 + linear head on Oxford Flowers-102
---

# Flower Classifier API

FastAPI inference service for 102 flower species: a frozen DINOv2 ViT-S/14 backbone with a linear head trained on Oxford Flowers-102.

- `GET /health` liveness probe
- `GET /info` backbone, training metrics and device
- `POST /predict?k=5` multipart `file` upload, returns the top-k species with probabilities

```bash
curl -F file=@rose.jpg "https://<user>-<space>.hf.space/predict?k=3"
```
