import os
import time
import jwt
import requests
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class GitHubAppAuth:

    def __init__(self, app_id: str, private_key_path: str):
        self.app_id = int(app_id)
        self.private_key = self._load_private_key(private_key_path)
        self._token_cache: Dict[int, Dict] = {}
        self._jwt_cache: Optional[Dict] = None
        self._cache_lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28'
        })
        logger.info(f"GitHub App Auth initialized for App ID: {app_id}")
    
    def _load_private_key(self, path: str) -> bytes:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Private key not found: {path}")
        with open(path, 'rb') as f:
            return f.read()
    
    def _generate_jwt(self) -> str:
        with self._cache_lock:
            # Check if we have a cached JWT that's still valid
            if self._jwt_cache:
                # JWT is valid if it expires more than 60 seconds from now
                if self._jwt_cache['exp'] > int(time.time()) + 60:
                    logger.debug("Using cached JWT")
                    return self._jwt_cache['token']

            # Generate new JWT
            now = int(time.time())
            payload = {
                'iat': now - 60,
                'exp': now + 600,
                'iss': self.app_id
            }
            token = jwt.encode(payload, self.private_key, algorithm='RS256')

            # Cache the JWT with its expiration time
            self._jwt_cache = {
                'token': token,
                'exp': now + 600
            }
            logger.debug(f"Generated new JWT, expires at {now + 600}")
            return token
    
    def _is_token_valid(self, token_data: Dict) -> bool:
        if not token_data:
            return False
        expires_at = datetime.fromisoformat(token_data['expires_at'].replace('Z', '+00:00'))
        buffer = timedelta(minutes=5)
        return datetime.now(expires_at.tzinfo) < (expires_at - buffer)
    
    def get_installation_token(self, installation_id: int) -> str:
        with self._cache_lock:
            cached = self._token_cache.get(installation_id)
            if cached and self._is_token_valid(cached):
                logger.debug(f"Using cached token for installation {installation_id}")
                return cached['token']
        
        logger.info(f"Generating new token for installation {installation_id}")
        jwt_token = self._generate_jwt()
        
        try:
            response = self._session.post(
                f'https://api.github.com/app/installations/{installation_id}/access_tokens',
                headers={'Authorization': f'Bearer {jwt_token}'},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            token_data = {
                'token': data['token'],
                'expires_at': data['expires_at']
            }
            
            with self._cache_lock:
                self._token_cache[installation_id] = token_data
            
            logger.info(f"Token generated for installation {installation_id}, expires at {data['expires_at']}")
            return token_data['token']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get installation token: {e}")
            raise
    
    def revoke_installation_token(self, installation_id: int, token: str) -> bool:
        try:
            response = self._session.delete(
                f'https://api.github.com/installation/token',
                headers={'Authorization': f'Bearer {token}'},
                timeout=10
            )
            
            with self._cache_lock:
                self._token_cache.pop(installation_id, None)
            
            if response.status_code == 204:
                logger.info(f"Token revoked for installation {installation_id}")
                return True
            return False
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to revoke token: {e}")
            return False
    
    def clear_cache(self, installation_id: Optional[int] = None):
        with self._cache_lock:
            if installation_id:
                self._token_cache.pop(installation_id, None)
                logger.info(f"Cleared cache for installation {installation_id}")
            else:
                self._token_cache.clear()
                logger.info("Cleared all token cache")
    
    def get_cache_stats(self) -> Dict:
        with self._cache_lock:
            return {
                'cached_installations': len(self._token_cache),
                'installations': list(self._token_cache.keys())
            }