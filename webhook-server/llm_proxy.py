import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from flask import Flask, request, jsonify, Response
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

LIGHTNING_ENDPOINT = os.getenv(
    "LIGHTNING_AI_ENDPOINT",
    "https://3000-01kbs5g1xhxxrzv5rh00t479a4.cloudspaces.litng.ai"
)
MODEL_NAME = os.getenv("LIGHTNING_AI_MODEL_NAME", "nareshmlx/code-reviewer-opencv-harxsan-v2")
MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "32000"))
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.3"))


def get_upstream_url() -> str:
    base = LIGHTNING_ENDPOINT.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "lightning-ai"
            }
        ]
    })


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json()
        
        messages = data.get('messages', [])
        model = data.get('model', MODEL_NAME)
        temperature = data.get('temperature', TEMPERATURE)
        max_tokens = data.get('max_tokens', 4096)
        stream = data.get('stream', False)
        
        upstream_url = f"{get_upstream_url()}/chat/completions"
        
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": min(max_tokens, MAX_TOKENS),
            "stream": stream
        }
        
        logger.info(f"Proxying request to {upstream_url}")
        logger.debug(f"Messages count: {len(messages)}")
        
        if stream:
            def generate():
                with requests.post(
                    upstream_url,
                    json=payload,
                    stream=True,
                    timeout=300
                ) as response:
                    for line in response.iter_lines():
                        if line:
                            yield line + b'\n'
            
            return Response(
                generate(),
                content_type='text/event-stream'
            )
        else:
            response = requests.post(
                upstream_url,
                json=payload,
                timeout=300
            )
            
            if response.status_code != 200:
                logger.error(f"Upstream error: {response.status_code} - {response.text}")
                return jsonify({
                    "error": {
                        "message": f"Upstream model error: {response.text}",
                        "type": "upstream_error",
                        "code": response.status_code
                    }
                }), response.status_code
            
            result = response.json()
            logger.info("Request completed successfully")
            
            return jsonify(result)
    
    except requests.exceptions.Timeout:
        logger.error("Request to upstream timed out")
        return jsonify({
            "error": {
                "message": "Request timed out",
                "type": "timeout_error"
            }
        }), 504
    
    except Exception as e:
        logger.error(f"Proxy error: {e}", exc_info=True)
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }), 500


@app.route('/v1/completions', methods=['POST'])
def completions():
    try:
        data = request.get_json()
        
        prompt = data.get('prompt', '')
        
        messages = [{"role": "user", "content": prompt}]
        
        data['messages'] = messages
        data['model'] = MODEL_NAME
        
        upstream_url = f"{get_upstream_url()}/chat/completions"
        
        response = requests.post(
            upstream_url,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": data.get('temperature', TEMPERATURE),
                "max_tokens": min(data.get('max_tokens', 4096), MAX_TOKENS)
            },
            timeout=300
        )
        
        if response.status_code != 200:
            return jsonify({"error": response.text}), response.status_code
        
        result = response.json()
        
        completion_result = {
            "id": result.get("id", ""),
            "object": "text_completion",
            "created": result.get("created", int(datetime.now().timestamp())),
            "model": MODEL_NAME,
            "choices": [
                {
                    "text": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": result.get("choices", [{}])[0].get("finish_reason", "stop")
                }
            ],
            "usage": result.get("usage", {})
        }
        
        return jsonify(completion_result)
    
    except Exception as e:
        logger.error(f"Completions error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    try:
        response = requests.get(
            f"{LIGHTNING_ENDPOINT}/health",
            timeout=10
        )
        upstream_ok = response.status_code == 200
    except:
        upstream_ok = False
    
    return jsonify({
        "status": "healthy" if upstream_ok else "degraded",
        "upstream_endpoint": LIGHTNING_ENDPOINT,
        "upstream_reachable": upstream_ok,
        "model": MODEL_NAME
    }), 200 if upstream_ok else 503


if __name__ == '__main__':
    port = int(os.getenv("LLM_PROXY_PORT", "5001"))
    logger.info(f"Starting LLM proxy on port {port}")
    logger.info(f"Upstream endpoint: {LIGHTNING_ENDPOINT}")
    app.run(host='0.0.0.0', port=port, debug=False)
