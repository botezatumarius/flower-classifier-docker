import os
from pathlib import Path

import gradio as gr
import requests

BACKENDS = {
    name: url
    for name, url in (("Local container", os.getenv("API_URL")), ("Hugging Face Space", os.getenv("SPACE_URL")))
    if url
}
EXAMPLES = sorted(str(path) for path in Path("examples").glob("*.jpg"))


def classify(image: str | None, backend: str) -> tuple[dict[str, float], str]:
    if image is None:
        raise gr.Error("Upload a flower photo first.")
    try:
        with open(image, "rb") as file:
            response = requests.post(f"{BACKENDS[backend]}/predict", files={"file": file}, timeout=120)
        response.raise_for_status()
    except requests.RequestException as error:
        raise gr.Error(f"{backend} is unavailable: {error}") from error
    body = response.json()
    return {p["label"]: p["score"] for p in body["predictions"]}, f"{body['latency_ms']} ms · {backend}"


with gr.Blocks(title="Flower species classifier") as demo:
    gr.Markdown(
        "# Flower species classifier\n"
        "DINOv2 ViT-S/14 with a linear head trained on Oxford Flowers-102. "
        "This container only renders the UI; predictions come from the inference API."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", label="Flower photo", height=320)
            backend = gr.Radio(list(BACKENDS), value=next(iter(BACKENDS)), label="Inference backend", visible=len(BACKENDS) > 1)
            submit = gr.Button("Classify", variant="primary")
        with gr.Column():
            species = gr.Label(num_top_classes=5, label="Species")
            latency = gr.Textbox(label="Model latency", interactive=False)
    gr.Examples(EXAMPLES, inputs=image, examples_per_page=len(EXAMPLES))
    submit.click(classify, inputs=[image, backend], outputs=[species, latency])

if __name__ == "__main__":
    demo.launch()
