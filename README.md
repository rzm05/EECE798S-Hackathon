# Legal Knowledge Graph Extractor (LLM-only, Ontology-Aligned)
# Please note that the notebook should be downloaded to be accessed and tested (even if showing invalid) through collab

## Team

Omar Ramadan - Rasha Malaeb - Zaynab El Haj – AUB EECE 798S

Supervised by: *Dr. Ammar Mohanna*

---

Extracts a knowledge graph from legal documents (TXT / pasted text / selectable-text PDFs) using an LLM and renders it with edge labels in a simple Gradio UI. Output edges are ontology-constrained (e.g., `Party`, `Clause`, `Date`, `Amount`, …) and downloadable as JSON.

> **Note on hackathon implementation:**
> The main prototyping was done on **Google Colab** because we could not use the provided API tokens in our environment due to token limit. We therefore ran a lightweight, **locally loaded** Hugging Face model inside Colab. We couldn’t reproduce this on our limited local GPUs, hence the Colab path for the working demo.

> We have also considered data samples from the following dataset (which are attached as a zip file in this repository): https://huggingface.co/datasets/theatticusproject/cuad/tree/main/CUAD_v1

---

## Features

* **Three inputs:** TXT upload, pasted text, or PDF (text-selectable).
* **Ontology-aligned extraction:** Labels limited to a fixed set; relationships normalized.
* **Graph rendering:** NetworkX + Matplotlib; PNG preview in-app.
* **Self-contained:** Single `app.py` with a simple Gradio UI.
* **Dockerized:** CPU or GPU (via NVIDIA Container Toolkit).

---

## Quick Start (Docker)

### 1) Repository Structure

```
.
├─ app.py              # single app file
├─ requirements.txt    # provided in this repo
└─ Dockerfile          # provided in this repo
```

### 2) Build

```bash
docker build -t legal-kg:latest .
```

### 3) Run (choose one)

**GPU (recommended if available):**

```bash
docker run --rm -p 7860:7860 --gpus all \
  -e HF_TOKEN=hf_xxx_your_token_here \
  -e MODEL_NAME=tiiuae/Falcon3-1B-Instruct \
  --name legal-kg legal-kg:latest
```

**CPU (works anywhere, slower):**

```bash
docker run --rm -p 7860:7860 \
  -e HF_TOKEN=hf_xxx_your_token_here \
  -e MODEL_NAME=tiiuae/Falcon3-1B-Instruct \
  --name legal-kg legal-kg:latest
```

Open your browser at **[http://localhost:7860](http://localhost:7860)**.

---

## Running on Google Colab (reference)

Because of API token constraints during the hackathon, we validated the full pipeline on **Colab**:

1. Upload `app.py` or run the inline cells version.
2. Set your Hugging Face token in Colab:

   ```python
   from google.colab import userdata
   userdata.set("hack-ai","<your_hf_token>")
   ```
3. Install Python deps with `pip` (already included in our Colab cells).
4. Launch Gradio; Colab will show a public/share URL.

This was necessary because we couldn’t use the provided API tokens in our local environment and didn’t have enough local GPU memory to comfortably host the model.

---

## Environment Variables

* `HF_TOKEN` – (optional, but recommended) Hugging Face token to pull model weights if required.
* `MODEL_NAME` – override the model (default: `tiiuae/Falcon3-1B-Instruct`).
* `PORT` – Gradio port inside the container (default: `7860`).

The app sets `HF_HOME=/app/.cache/huggingface` inside the container so models cache to a writable path.

---

## Requirements

* **Docker** (Desktop on macOS/Windows or Engine on Linux).
* **Optional GPU:** NVIDIA drivers + **NVIDIA Container Toolkit** to use `--gpus all`.

*(If running without Docker, use Python 3.10+ and install from `requirements.txt`. For CPU-only hosts, expect slower inference.)*

---

## How It Works (high level)

1. **Chunking:** Input text is chunked with overlap and sentence-aware boundaries.
2. **LLM extraction:** Each chunk prompts the model to emit edges constrained to:

   * **Labels:** `Party`, `Obligation`, `Right`, `Clause`, `Date`, `Amount`, `Service`, `Condition`, `ConfidentialInfo`, `Miscellaneous`.
   * **Relationships:**
     `Party has Obligation`, `Party holds Right`, `Clause references Clause`,
     `Condition triggers Obligation/Right`, `Party responsible for Service`,
     `Party pays Amount`, `Party protected by ConfidentialInfo`.
3. **Sanitization:** Keys/quotes fixed; labels/relations normalized; duplicates removed.
4. **Graph build & render:** Directed graph with labeled edges (PNG preview).
5. **Export:** Downloadable `edges.json`.

---

## Usage (inside the app)

1. Choose a tab: **TXT Upload**, **Paste Text**, or **PDF Upload**.
2. Click the corresponding **Extract** button.
3. View the graph preview, scan the edge list, and download `edges.json`.

> **PDF note:** Only text-selectable PDFs work with `pypdf`. Scanned/image PDFs won’t extract text.

---

## Troubleshooting

* **Model won’t download / auth error**
  Ensure `-e HF_TOKEN=...` is provided (if the model requires auth).
* **No GPU detected in container**
  Check `nvidia-smi` works on host; run with `--gpus all` and the NVIDIA runtime installed.
* **Slow / OOM on CPU**
  Use a smaller model via `MODEL_NAME` or switch to a GPU host.
* **Matplotlib backend errors**
  We set `Agg` in code for headless rendering; font warnings are harmless.
* **PDF text empty**
  Your PDF is likely scanned; convert to OCR or provide TXT.

---

## Security & Privacy

* All processing is **local to your machine/container**.
* Uploaded text/PDF content is only used in memory for extraction and rendering.
* If you mount volumes to persist caches or outputs, ensure proper file permissions.

---

## Development Notes

* The app is a single `app.py` with a Gradio UI.
* Matplotlib is set to `Agg` for headless containers.
* The UI binds to `0.0.0.0` and uses `PORT` (default `7860`).
* First run downloads model weights to `/app/.cache/huggingface`.

---

## License

MSFEA, AUB © 2025

---

## Acknowledgements

* Hugging Face Transformers / Accelerate
* NetworkX, Matplotlib, Gradio

---
