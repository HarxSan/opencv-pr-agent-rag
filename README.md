# OpenCV PR-Agent RAG v2.0

Production-grade PR review system for OpenCV using Qodo PR-Agent with custom RAG (Retrieval-Augmented Generation) pipeline and a fine-tuned Qwen 2.5 Coder model.

## Features

- **Comment-triggered only**: No automatic PR creation triggers - all commands invoked via comments
- **RAG-enhanced reviews**: Injects relevant OpenCV codebase context for better reviews
- **Custom model support**: Uses your fine-tuned Qwen 2.5 Coder model via any vLLM-compatible endpoint
- **Large PR support**: Handles PRs up to 10MB diff size and 500 files
- **All PR-Agent commands**: /review, /improve, /ask, /describe, /update_changelog, /add_docs, /test

## Architecture

```
GitHub PR Comment (/review)
        ↓
   ngrok tunnel
        ↓
 Webhook Server (Flask)
        ↓
    ┌───┴───┐
    ↓       ↓
 Qdrant   PR-Agent CLI
  RAG         ↓
    ↓    vLLM Model Endpoint
    └───────┬───────┘
            ↓
    Review posted to PR
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- GitHub Personal Access Token (or GitHub App)
- OpenCV repository cloned locally
- ngrok for webhook forwarding

### Setup

1. **Clone and configure**:
   ```bash
   git clone <this-repo>
   cd opencv-pr-agent-rag
   ./setup.sh
   # Edit .env with your configuration
   ```

2. **Start Qdrant**:
   ```bash
   docker-compose up -d qdrant
   ```

3. **Index OpenCV codebase** (first time only, ~30-60 minutes):
   ```bash
   docker-compose --profile indexer run indexer
   ```

4. **Start webhook server**:
   ```bash
   docker-compose up -d pr-agent-rag
   ```

5. **Setup ngrok**:
   ```bash
   ngrok http 5000
   # Copy the HTTPS URL
   ```

6. **Configure GitHub webhook**:
   - Go to your repository Settings → Webhooks
   - Add webhook with URL: `https://YOUR_NGROK_URL/webhook`
   - Content type: `application/json`
   - Secret: (from your .env GITHUB_WEBHOOK_SECRET)
   - Events: Issue comments, Pull requests

7. **Test**:
   ```bash
   ./test.sh
   # Or comment /review on any PR
   ```

## Configuration

### Required Settings

| Variable | Description |
|----------|-------------|
| `GITHUB_USER_TOKEN` | GitHub PAT with repo access |
| `OPENCV_REPO_PATH` | Local path to OpenCV clone |
| `MODEL_ENDPOINT` | Your vLLM-compatible model endpoint URL |
| `MODEL_NAME` | Model name (e.g., nareshmlx/code-reviewer-opencv-harxsan-v2) |

### Optional Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_DIFF_SIZE` | 10485760 (10MB) | Maximum PR diff size |
| `MAX_FILES_PER_PR` | 500 | Maximum files per PR |
| `REQUEST_TIMEOUT` | 600 | Command timeout (seconds) |
| `RAG_TOP_K` | 15 | Context chunks to retrieve |
| `RAG_MIN_SCORE` | 0.4 | Minimum similarity score |
| `RAG_ENABLED_COMMANDS` | review,improve,ask | Commands with RAG |

## Commands

| Command | Description | RAG Context |
|---------|-------------|-------------|
| `/review` | Full PR review | ✓ |
| `/improve` | Code improvement suggestions | ✓ |
| `/ask <question>` | Ask about the PR | ✓ |
| `/describe` | Generate PR description | ✗ |
| `/update_changelog` | Update changelog | ✗ |
| `/add_docs` | Add documentation | ✗ |
| `/test` | Generate tests | ✗ |
| `/help` | Show help | ✗ |

## API Endpoints

- `GET /health` - Health check and component status
- `POST /webhook` - GitHub webhook receiver
- `POST /context` - Test RAG retrieval
- `POST /test` - Manual command testing

## Troubleshooting

### Qdrant connection failed
```bash
# Check Qdrant is running
docker-compose logs qdrant

# Verify collection exists
curl http://localhost:6333/collections/opencv_codebase
```

### RAG returns no results
```bash
# Check indexing completed
curl http://localhost:6333/collections/opencv_codebase | jq .result.points_count

# Re-index if needed
docker-compose --profile indexer run indexer
```

### Model endpoint errors
```bash
# Test endpoint directly
curl -X POST https://YOUR_ENDPOINT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-coder-7b-instruct","messages":[{"role":"user","content":"test"}],"max_tokens":10}'
```

### Webhook not receiving events
1. Check ngrok is running and URL is correct
2. Verify webhook secret matches
3. Check GitHub webhook delivery history for errors

## Development

### Local testing without Docker
```bash
cd webhook-server
pip install -r requirements.txt
pip install pr-agent
python server.py
```

### View logs
```bash
docker-compose logs -f pr-agent-rag
```

## License

MIT
