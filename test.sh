#!/bin/bash
set -e

echo "========================================================================"
echo "OpenCV PR-Agent RAG v2.0 - System Tests"
echo "========================================================================"
echo ""

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

PASS=0
FAIL=0

test_check() {
    local name="$1"
    local result="$2"
    
    echo -n "Testing $name... "
    if [ "$result" = "0" ]; then
        echo "✓ PASS"
        ((PASS++))
    else
        echo "✗ FAIL"
        ((FAIL++))
    fi
}

# Test 1: Qdrant health
echo "1. Infrastructure Tests"
echo "------------------------"

QDRANT_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:6333/readyz 2>/dev/null || echo "000")
test_check "Qdrant health" "$([[ "$QDRANT_HEALTH" == "200" ]] && echo 0 || echo 1)"

# Test 2: Qdrant collection
COLLECTION_INFO=$(curl -s "http://localhost:6333/collections/${QDRANT_COLLECTION_NAME:-opencv_codebase}" 2>/dev/null)
POINTS_COUNT=$(echo "$COLLECTION_INFO" | grep -o '"points_count":[0-9]*' | cut -d: -f2 || echo "0")
test_check "Qdrant collection (${POINTS_COUNT:-0} points)" "$([[ "${POINTS_COUNT:-0}" -gt "0" ]] && echo 0 || echo 1)"

# Test 3: Webhook server health
echo ""
echo "2. Webhook Server Tests"
echo "-----------------------"

SERVER_HEALTH=$(curl -s http://localhost:5000/health 2>/dev/null)
SERVER_STATUS=$(echo "$SERVER_HEALTH" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
test_check "Server health ($SERVER_STATUS)" "$([[ "$SERVER_STATUS" == "healthy" ]] && echo 0 || echo 1)"

# Test 4: RAG component
RAG_OK=$(echo "$SERVER_HEALTH" | grep -o '"rag":{[^}]*"ok":true' || echo "")
test_check "RAG component" "$([[ -n "$RAG_OK" ]] && echo 0 || echo 1)"

# Test 5: Model endpoint configured
MODEL_ENDPOINT=$(echo "$SERVER_HEALTH" | grep -o '"endpoint":"[^"]*"' | head -1 | cut -d'"' -f4 || echo "")
test_check "Model endpoint configured" "$([[ -n "$MODEL_ENDPOINT" && "$MODEL_ENDPOINT" != "not configured" ]] && echo 0 || echo 1)"

# Test 6: GitHub auth
echo ""
echo "3. GitHub Auth Tests"
echo "--------------------"

GITHUB_AUTH=$(echo "$SERVER_HEALTH" | grep -o '"auth":"[^"]*"' | cut -d'"' -f4 || echo "none")
test_check "GitHub authentication ($GITHUB_AUTH)" "$([[ "$GITHUB_AUTH" != "none" ]] && echo 0 || echo 1)"

# Test 7: RAG context retrieval
echo ""
echo "4. RAG Pipeline Tests"
echo "---------------------"

RAG_TEST=$(curl -s -X POST http://localhost:5000/context \
    -H "Content-Type: application/json" \
    -d '{
        "title": "Fix memory leak in cv::Mat",
        "description": "Fixes issue with Mat memory management",
        "changed_files": ["modules/core/src/matrix.cpp"],
        "diff": "- old line\n+ new line"
    }' 2>/dev/null)

RAG_SUCCESS=$(echo "$RAG_TEST" | grep -o '"success":true' || echo "")
RAG_COUNT=$(echo "$RAG_TEST" | grep -o '"count":[0-9]*' | cut -d: -f2 || echo "0")
test_check "RAG context retrieval (${RAG_COUNT:-0} chunks)" "$([[ -n "$RAG_SUCCESS" ]] && echo 0 || echo 1)"

# Test 8: Model endpoint connectivity
echo ""
echo "5. Model Endpoint Tests"
echo "-----------------------"

if [ -n "$LIGHTNING_AI_ENDPOINT" ]; then
    MODEL_PING=$(curl -s -o /dev/null -w "%{http_code}" "${LIGHTNING_AI_ENDPOINT}/health" 2>/dev/null || \
                 curl -s -o /dev/null -w "%{http_code}" "${LIGHTNING_AI_ENDPOINT}/v1/models" 2>/dev/null || echo "000")
    test_check "Model endpoint reachable" "$([[ "$MODEL_PING" == "200" ]] && echo 0 || echo 1)"
else
    echo "⚠ Skipping model endpoint test (LIGHTNING_AI_ENDPOINT not set)"
fi

# Summary
echo ""
echo "========================================================================"
echo "TEST SUMMARY"
echo "========================================================================"
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo "========================================================================"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "✓ All tests passed! System is ready."
    echo ""
    echo "To test with a real PR:"
    echo "  1. Start ngrok: ngrok http 5000"
    echo "  2. Configure webhook URL in GitHub"
    echo "  3. Comment /review on any PR"
    exit 0
else
    echo "✗ Some tests failed. Check the logs:"
    echo "  docker-compose logs pr-agent-rag"
    exit 1
fi
