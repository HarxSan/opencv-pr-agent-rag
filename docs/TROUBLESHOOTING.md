# Troubleshooting Guide

## Common Issues and Solutions

### 1. Qdrant Connection Issues

**Symptom**: Server health shows RAG component failed

**Diagnosis**:
```bash
# Check if Qdrant is running
docker-compose ps qdrant

# Check Qdrant logs
docker-compose logs qdrant

# Test Qdrant directly
curl http://localhost:6333/readyz
```

**Solutions**:
- Ensure Qdrant container is running: `docker-compose up -d qdrant`
- Check if port 6333 is available
- Verify QDRANT_API_KEY matches in both .env and Qdrant config

### 2. Collection Not Found

**Symptom**: Error "Collection 'opencv_codebase' not found"

**Solution**:
```bash
# Run the indexer
docker-compose --profile indexer run indexer

# Verify collection was created
curl http://localhost:6333/collections/opencv_codebase
```

### 3. Model Endpoint Errors

**Symptom**: PR-Agent commands fail with model errors

**Diagnosis**:
```bash
# Test endpoint health
curl https://YOUR_ENDPOINT/health

# Test chat completions
curl -X POST https://YOUR_ENDPOINT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-7b-instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 10
  }'
```

**Solutions**:
- Verify LIGHTNING_AI_ENDPOINT is correct
- Check if the model is running in Lightning AI Studio
- Ensure the endpoint accepts OpenAI-compatible requests

### 4. Webhook Signature Mismatch

**Symptom**: 401 "Invalid signature" errors

**Solutions**:
- Verify GITHUB_WEBHOOK_SECRET in .env matches GitHub webhook settings
- Ensure no trailing whitespace in the secret
- Regenerate secret if needed:
  ```bash
  openssl rand -hex 32
  ```

### 5. GitHub Authentication Failures

**Symptom**: "Failed to fetch PR information"

**Diagnosis**:
```bash
# Test GitHub API access
curl -H "Authorization: token YOUR_TOKEN" \
  https://api.github.com/repos/HarxSan/opencv/pulls/1
```

**Solutions**:
- Verify GITHUB_USER_TOKEN has correct scopes
- For GitHub App, check app is installed on the repository
- Ensure private key file exists and has correct permissions

### 6. RAG Returns No Results

**Symptom**: Reviews don't include codebase context

**Diagnosis**:
```bash
# Check collection has points
curl http://localhost:6333/collections/opencv_codebase | jq .result.points_count

# Test RAG retrieval
curl -X POST http://localhost:5000/context \
  -H "Content-Type: application/json" \
  -d '{"title":"Fix Mat issue","changed_files":["modules/core/src/matrix.cpp"],"diff":""}'
```

**Solutions**:
- Re-run indexer if points_count is 0
- Lower RAG_MIN_SCORE (default 0.4)
- Increase RAG_TOP_K for more results
- Check embedding model matches between indexer and server

### 7. Command Timeout

**Symptom**: Commands fail with timeout errors

**Solutions**:
- Increase REQUEST_TIMEOUT in .env (default 600s)
- For large PRs, model inference may be slow
- Check model endpoint performance

### 8. Large PR Handling Issues

**Symptom**: PRs with many files or large diffs fail

**Solutions**:
- MAX_DIFF_SIZE default is 10MB (10485760 bytes)
- Increase if needed: `MAX_DIFF_SIZE=20971520` (20MB)
- MAX_FILES_PER_PR default is 500

### 9. Docker Build Failures

**Symptom**: Container won't build

**Solutions**:
```bash
# Clean rebuild
docker-compose build --no-cache pr-agent-rag

# Check Docker disk space
docker system df
docker system prune -a  # Clean unused images
```

### 10. Memory Issues

**Symptom**: Container OOM killed

**Solutions**:
- Embedding model requires ~2GB RAM
- Add memory limits in docker-compose.yml:
  ```yaml
  pr-agent-rag:
    deploy:
      resources:
        limits:
          memory: 4G
  ```

## Diagnostic Commands

### View Server Logs
```bash
docker-compose logs -f pr-agent-rag
```

### Check All Components
```bash
./test.sh
```

### Interactive Container Access
```bash
docker-compose exec pr-agent-rag bash
```

### Health Check Details
```bash
curl -s http://localhost:5000/health | jq .
```

### Test Specific Command
```bash
curl -X POST http://localhost:5000/test \
  -H "Content-Type: application/json" \
  -d '{
    "pr_url": "https://github.com/HarxSan/opencv/pull/1",
    "command": "review"
  }' | jq .
```

## Performance Tuning

### For Large Codebases
- Increase BATCH_SIZE in indexer for faster indexing
- Use SSD storage for Qdrant data

### For Faster Reviews
- Reduce RAG_TOP_K if context is too large
- Lower RAG_MAX_CONTEXT_TOKENS
- Use a faster model endpoint

### For Better Context Quality
- Increase RAG_TOP_K for more candidates
- Lower RAG_MIN_SCORE for broader search
- Re-index with different chunking parameters

## Getting Help

1. Check server logs: `docker-compose logs pr-agent-rag`
2. Review GitHub webhook delivery history
3. Test components individually with curl commands above
4. Open an issue with:
   - Error message
   - Relevant logs
   - Configuration (sanitized)