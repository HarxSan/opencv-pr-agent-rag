#!/bin/bash
set -e

echo "========================================"
echo "OpenCV PR-Agent RAG - Starting Services"
echo "========================================"

python llm_proxy.py &
LLM_PROXY_PID=$!
echo "LLM Proxy started (PID: $LLM_PROXY_PID)"

sleep 2

echo "Starting webhook server..."
exec gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 2 \
    --threads 4 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    server:app
