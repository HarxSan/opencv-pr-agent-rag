FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir pr-agent

RUN mkdir -p /app/logs /app/cache/embeddings /app/secrets

COPY config.py rag_retriever.py pr_agent_runner.py server.py ./

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/huggingface

HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "600", "--keep-alive", "5", "server:app"]