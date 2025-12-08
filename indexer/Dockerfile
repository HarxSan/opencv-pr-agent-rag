FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY indexer.py .

RUN mkdir -p /app/logs /app/cache/embeddings

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/huggingface

CMD ["python", "-u", "indexer.py"]