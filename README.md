# Flower classifier in Docker

Deep learning inference split across containers: a FastAPI service runs a frozen **DINOv2 ViT-S/14** backbone with a linear head trained on **Oxford Flowers-102**, and a Gradio container provides the GUI.

```
flower-classifier/
├── compose.yaml          # inference + gui (+ trainer profile)
├── compose.gpu.yaml      # NVIDIA override: CUDA wheels + GPU reservation
├── inference/            # one image, two targets: train and serve
│   ├── Dockerfile
│   ├── flowers/          # model, training, CLI, API
│   └── artifacts/        # trained head + metrics, baked into the serve image
├── gui/                  # Gradio client, no ML dependencies
└── scripts/              # docker run inference, Hugging Face Space deploy
```

## Run

```bash
docker compose --profile train run --rm trainer    # optional: retrain the head
docker compose up --build -d --wait                # http://localhost:7860
scripts/predict.sh gui/examples                    # terminal inference
```

On an NVIDIA host with the Container Toolkit:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml --profile train run --rm trainer
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

## Hugging Face Space

The serving image is Space-ready (`inference/README.md` card, UID 1000, `serve` as default target). Hosting Docker Spaces requires a Hugging Face PRO plan.

```bash
hf auth login
scripts/deploy_space.sh <user>/flower-classifier-api
SPACE_URL=https://<user>-flower-classifier-api.hf.space docker compose up -d gui
```

The GUI then offers both backends: the local container and the online Space API.
