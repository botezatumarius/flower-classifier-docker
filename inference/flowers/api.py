from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

from flowers.classes import CLASSES
from flowers.model import FlowerClassifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = FlowerClassifier()
    yield


app = FastAPI(title="Flower classifier", description="DINOv2 ViT-S/14 + linear head on Oxford Flowers-102", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    return app.state.classifier.info


@app.post("/predict")
def predict(file: Annotated[UploadFile, File()], k: Annotated[int, Query(ge=1, le=len(CLASSES))] = 5) -> dict:
    try:
        image = Image.open(file.file)
        image.load()
    except OSError:
        raise HTTPException(status_code=415, detail="File is not a valid image") from None
    start = perf_counter()
    predictions = app.state.classifier.predict(image, k)
    return {"predictions": predictions, "latency_ms": round((perf_counter() - start) * 1000, 1)}
