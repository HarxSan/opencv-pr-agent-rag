import sys
import time
import hmac
import hashlib
import json
import requests
from datetime import datetime

WEBHOOK_URL = "http://localhost:5000/webhook"
WEBHOOK_SECRET = "69b98e719fa77884c1cb8d1b5c2c418408e36911c1bdfc25769c5437ef67719b"

def generate_signature(payload_bytes):
    signature = hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()
    return f'sha256={signature}'

def test_ping():
    print("\n=== Test 1: Ping Event ===")
    payload = {
        "zen": "Keep it simple.",
        "hook_id": 123456,
        "hook": {
            "id": 123456,
            "type": "App",
            "active": True
        }
    }
    payload_bytes = json.dumps(payload).encode('utf-8')
    signature = generate_signature(payload_bytes)
    
    response = requests.post(
        WEBHOOK_URL,
        data=payload_bytes,
        headers={
            'Content-Type': 'application/json',
            'X-GitHub-Event': 'ping',
            'X-Hub-Signature-256': signature
        }
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200, "Ping test failed"
    print("✅ Ping test passed")

def test_signature_verification():
    print("\n=== Test 2: Signature Verification ===")
    payload = {"test": "data"}
    payload_bytes = json.dumps(payload).encode('utf-8')
    wrong_signature = 'sha256=wrongsignature'
    
    response = requests.post(
        WEBHOOK_URL,
        data=payload_bytes,
        headers={
            'Content-Type': 'application/json',
            'X-GitHub-Event': 'push',
            'X-Hub-Signature-256': wrong_signature
        }
    )
    print(f"Status: {response.status_code}")
    assert response.status_code == 401, "Should reject invalid signature"
    print("✅ Signature verification test passed")

def test_issue_comment_review():
    print("\n=== Test 3: Issue Comment /review Command ===")
    payload = {
        "action": "created",
        "installation": {
            "id": 12345678
        },
        "repository": {
            "full_name": "test-user/opencv",
            "name": "opencv"
        },
        "issue": {
            "number": 9,
            "pull_request": {
                "url": "https://api.github.com/repos/test-user/opencv/pulls/9"
            }
        },
        "comment": {
            "body": "/review",
            "user": {
                "login": "test-user"
            }
        }
    }
    payload_bytes = json.dumps(payload).encode('utf-8')
    signature = generate_signature(payload_bytes)
    
    print("Note: This will fail if installation token can't be generated")
    print("Payload:", json.dumps(payload, indent=2)[:200] + "...")
    
    response = requests.post(
        WEBHOOK_URL,
        data=payload_bytes,
        headers={
            'Content-Type': 'application/json',
            'X-GitHub-Event': 'issue_comment',
            'X-Hub-Signature-256': signature
        }
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("⚠️  Issue comment test completed (may fail without valid installation)")

def test_health_endpoint():
    print("\n=== Test 4: Health Endpoint ===")
    response = requests.get('http://localhost:5000/health')
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200, "Health check failed"
    print("✅ Health check passed")

def test_token_caching():
    print("\n=== Test 5: Token Caching ===")
    print("Checking cache stats...")
    response = requests.get('http://localhost:5000/health')
    data = response.json()
    print(f"Cached installations: {data.get('cached_installations', 0)}")
    print("✅ Token caching check completed")

if __name__ == '__main__':
    print("=" * 60)
    print("Multi-Tenant GitHub App Server Tests")
    print("=" * 60)
    print(f"Target: {WEBHOOK_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    
    try:
        test_health_endpoint()
        time.sleep(1)
        
        test_ping()
        time.sleep(1)
        
        test_signature_verification()
        time.sleep(1)
        
        test_token_caching()
        time.sleep(1)
        
        test_issue_comment_review()
        
        print("\n" + "=" * 60)
        print("✅ Core tests completed successfully")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)