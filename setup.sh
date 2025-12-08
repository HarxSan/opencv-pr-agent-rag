#!/bin/bash
set -e

echo "========================================================================"
echo "OpenCV PR-Agent RAG v2.0 - Setup"
echo "========================================================================"
echo ""

# Check dependencies
command -v docker >/dev/null 2>&1 || { echo "Error: Docker not installed"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1 || { echo "Error: Docker Compose not installed"; exit 1; }

echo "✓ Docker and Docker Compose found"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data/{qdrant,logs,cache}
mkdir -p secrets
chmod 700 secrets

echo "✓ Directories created"

# Check for .env
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env from template..."
    cp .env.example .env
    echo "✓ Created .env file"
    echo ""
    echo "⚠️  Please edit .env and configure:"
    echo "   - GITHUB_USER_TOKEN (or GitHub App credentials)"
    echo "   - OPENCV_REPO_PATH (path to your local OpenCV clone)"
    echo "   - LIGHTNING_AI_ENDPOINT (your model endpoint)"
else
    echo "✓ .env file exists"
fi

# Generate Qdrant API key if not set
if [ -f .env ] && ! grep -q "^QDRANT_API_KEY=." .env; then
    QDRANT_KEY=$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | xxd -p)
    sed -i.bak "s/^QDRANT_API_KEY=$/QDRANT_API_KEY=$QDRANT_KEY/" .env 2>/dev/null || \
    sed -i '' "s/^QDRANT_API_KEY=$/QDRANT_API_KEY=$QDRANT_KEY/" .env
    echo "✓ Generated Qdrant API key"
fi

# Generate webhook secret if not set
if [ -f .env ] && ! grep -q "^GITHUB_WEBHOOK_SECRET=." .env; then
    WEBHOOK_SECRET=$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | xxd -p)
    sed -i.bak "s/^GITHUB_WEBHOOK_SECRET=$/GITHUB_WEBHOOK_SECRET=$WEBHOOK_SECRET/" .env 2>/dev/null || \
    sed -i '' "s/^GITHUB_WEBHOOK_SECRET=$/GITHUB_WEBHOOK_SECRET=$WEBHOOK_SECRET/" .env
    echo "✓ Generated webhook secret"
fi

# Clean up backup files
rm -f .env.bak 2>/dev/null

echo ""
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Edit .env with your configuration"
echo ""
echo "2. Start Qdrant:"
echo "   docker-compose up -d qdrant"
echo ""
echo "3. Index OpenCV codebase (first time only, ~30-60 min):"
echo "   docker-compose --profile indexer run indexer"
echo ""
echo "4. Start the webhook server:"
echo "   docker-compose up -d pr-agent-rag"
echo ""
echo "5. Set up ngrok for webhook forwarding:"
echo "   ngrok http 5000"
echo ""
echo "6. Configure GitHub webhook with the ngrok URL"
echo "   URL: https://YOUR_NGROK_URL/webhook"
echo ""
echo "7. Test with /review comment on any PR"
echo ""
echo "========================================================================"
