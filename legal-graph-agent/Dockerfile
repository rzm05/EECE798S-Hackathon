# GPU-ready base with PyTorch + CUDA 12.1 already installed.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# System hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Optional: git if you ever pull extras at runtime
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Your single-file app
COPY app.py ./app.py

# Gradio default port
EXPOSE 7860

# Hint to transformers to use a writable cache inside the container
ENV HF_HOME=/app/.cache/huggingface

# Hugging Face token can be injected at runtime
ENV HF_TOKEN=""

# Run
CMD ["python", "app.py"]
