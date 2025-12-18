import hmac
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class WebhookVerifier:
    
    def __init__(self, webhook_secret: str):
        if not webhook_secret:
            raise ValueError("Webhook secret is required")
        self.secret = webhook_secret.encode('utf-8')
        logger.info("Webhook verifier initialized")
    
    def verify_signature(self, payload: bytes, signature_header: Optional[str]) -> bool:
        if not signature_header:
            logger.warning("No signature header provided")
            return False
        
        if not signature_header.startswith('sha256='):
            logger.warning(f"Invalid signature format: {signature_header[:20]}")
            return False
        
        expected_signature = signature_header[7:]
        
        computed_signature = hmac.new(
            self.secret,
            payload,
            hashlib.sha256
        ).hexdigest()
        
        is_valid = hmac.compare_digest(computed_signature, expected_signature)
        
        if not is_valid:
            logger.warning("Webhook signature verification failed")
        else:
            logger.debug("Webhook signature verified")
        
        return is_valid
    
    def verify_request(self, payload: bytes, headers: dict) -> bool:
        signature = headers.get('X-Hub-Signature-256')
        return self.verify_signature(payload, signature)