# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS runtime
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG BACKBONE=vit_small_patch14_dinov2.lvd142m
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    BACKBONE=${BACKBONE}
RUN useradd --uid 1000 --create-home app
WORKDIR /app

# Rarely changing layers first: dependencies, then pretrained weights
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url ${TORCH_INDEX} -r requirements.txt
RUN python -c "import timm; timm.create_model('${BACKBONE}', pretrained=True)"
ENV HF_HUB_OFFLINE=1
COPY flowers ./flowers

FROM runtime AS train
COPY requirements-train.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-train.txt
RUN mkdir /data && chown app /data
USER app
CMD ["python", "-m", "flowers.train"]

FROM runtime AS serve
COPY artifacts ./artifacts
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
CMD ["uvicorn", "flowers.api:app", "--host", "0.0.0.0", "--port", "8000"]
