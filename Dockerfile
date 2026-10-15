# Container for the live demo (Hugging Face Spaces / Render). Serves API + dashboard on :7860.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
ENV HOME=/tmp HF_HOME=/tmp/hf YOLO_CONFIG_DIR=/tmp/yolo PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-base-patch32', allow_patterns=['*.json','*.txt','model.safetensors'])"

COPY . .
RUN mkdir -p backend/uploads && chmod -R 777 /app /tmp

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "7860"]
