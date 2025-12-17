import os
import sys
import json
import hmac
import hashlib
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import asdict

from flask import Flask, request, jsonify, Response
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import load_config, Config, ModelConfig
from rag_retriever import RAGRetriever
from pr_agent_runner import PRAgentRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

config: Optional[Config] = None
rag_retriever: Optional[RAGRetriever] = None
pr_agent_runner: Optional[PRAgentRunner] = None


def reload_model_config():
    import os
    from config import ModelConfig
    
    new_model_config = ModelConfig()
    
    global config
    if config:
        old_endpoint = config.model.endpoint
        old_model = config.model.model_name
        
        config.model = new_model_config
        
        if old_endpoint != new_model_config.endpoint or old_model != new_model_config.model_name:
            logger.info(
                f"Model config reloaded: "
                f"{old_model} -> {new_model_config.model_name}, "
                f"{old_endpoint} -> {new_model_config.endpoint}"
            )


class GitHubClient:
    def __init__(self, cfg: Config):
        self.config = cfg
        self.session = self._create_session()
        self._setup_auth()
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
    
    def _setup_auth(self):
        if self.config.github.user_token:
            self.session.headers['Authorization'] = f'token {self.config.github.user_token}'
        self.session.headers['Accept'] = 'application/vnd.github.v3+json'
        self.session.headers['User-Agent'] = 'OpenCV-PR-Agent-RAG/1.0'
    
    def _get_paginated(self, url: str, max_items: int = 500) -> List[Dict]:
        items = []
        page = 1
        per_page = 100
        
        while len(items) < max_items:
            params = {'page': page, 'per_page': per_page}
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            page_items = response.json()
            if not page_items:
                break
            
            items.extend(page_items)
            
            if len(page_items) < per_page:
                break
            
            page += 1
            
            if page > 10:
                logger.warning(f"Reached pagination limit for {url}")
                break
        
        return items[:max_items]
    
    def get_pr_info(self, owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> List[Dict]:
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files'
        return self._get_paginated(url, max_items=self.config.server.max_files_per_pr)
    
    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'
        headers = {'Accept': 'application/vnd.github.v3.diff'}
        
        response = self.session.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        
        diff = response.text
        
        if len(diff) > self.config.server.max_diff_size:
            logger.warning(f"Diff truncated from {len(diff)} to {self.config.server.max_diff_size} bytes")
            
            lines = diff[:self.config.server.max_diff_size].split('\n')
            last_complete = len(lines) - 1
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].startswith('diff --git'):
                    last_complete = i - 1
                    break
            
            diff = '\n'.join(lines[:last_complete + 1])
            diff += f"\n\n... (diff truncated, total size exceeded {self.config.server.max_diff_size // 1024 // 1024}MB limit)"
        
        return diff
    
    def post_comment(self, owner: str, repo: str, issue_number: int, body: str) -> Dict[str, Any]:
        url = f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments'
        response = self.session.post(url, json={'body': body}, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def add_reaction(self, owner: str, repo: str, comment_id: int, reaction: str) -> bool:
        url = f'https://api.github.com/repos/{owner}/{repo}/issues/comments/{comment_id}/reactions'
        headers = {'Accept': 'application/vnd.github+json'}
        
        try:
            response = self.session.post(url, json={'content': reaction}, headers=headers, timeout=10)
            return response.status_code in [200, 201]
        except Exception as e:
            logger.warning(f"Failed to add reaction: {e}")
            return False


def verify_github_signature(payload: bytes, signature: str, secret: str) -> bool:
    if not signature:
        return False
    
    if not signature.startswith('sha256='):
        if signature.startswith('sha1='):
            expected = 'sha1=' + hmac.new(
                secret.encode('utf-8'),
                payload,
                hashlib.sha1
            ).hexdigest()
            return hmac.compare_digest(expected, signature)
        return False
    
    expected = 'sha256=' + hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected, signature)


def extract_pr_context(gh_client: GitHubClient, owner: str, repo: str,
                       pr_number: int) -> Dict[str, Any]:
    try:
        logger.info(f"Fetching PR context for {owner}/{repo}#{pr_number}")
        
        pr_info = gh_client.get_pr_info(owner, repo, pr_number)
        
        files = gh_client.get_pr_files(owner, repo, pr_number)
        changed_files = [f['filename'] for f in files]
        
        diff = gh_client.get_pr_diff(owner, repo, pr_number)
        
        context = {
            'title': pr_info.get('title', ''),
            'description': pr_info.get('body', '') or '',
            'changed_files': changed_files,
            'diff': diff,
            'base_branch': pr_info.get('base', {}).get('ref', ''),
            'head_branch': pr_info.get('head', {}).get('ref', ''),
            'author': pr_info.get('user', {}).get('login', ''),
            'additions': pr_info.get('additions', 0),
            'deletions': pr_info.get('deletions', 0),
            'changed_files_count': pr_info.get('changed_files', len(changed_files))
        }
        
        logger.info(
            f"PR Context: {len(changed_files)} files, "
            f"{context['additions']}+ {context['deletions']}- lines, "
            f"{len(diff)} bytes diff"
        )
        
        return context
        
    except Exception as e:
        logger.error(f"Failed to extract PR context: {e}")
        return {}


def is_valid_command_comment(comment_body: str) -> bool:
    if not comment_body:
        return False
    
    valid_commands = [
        '/review', '/improve', '/ask', '/describe',
        '/update_changelog', '/add_docs', '/test', '/help', '/config'
    ]
    
    stripped = comment_body.strip()
    return any(stripped.startswith(cmd) for cmd in valid_commands)


def process_webhook_async(event_type: str, payload: Dict[str, Any]):
    global config, pr_agent_runner
    
    start_time = time.time()
    
    try:
        if event_type == 'issue_comment':
            action = payload.get('action')
            if action not in ['created']:
                return
            
            comment = payload.get('comment', {})
            issue = payload.get('issue', {})
            repo = payload.get('repository', {})
            
            if not issue.get('pull_request'):
                logger.debug("Comment is not on a PR")
                return
            
            comment_body = comment.get('body', '')
            if not is_valid_command_comment(comment_body):
                logger.debug(f"Not a command comment: {comment_body[:30]}...")
                return
            
            owner = repo.get('owner', {}).get('login')
            repo_name = repo.get('name')
            pr_number = issue.get('number')
            comment_id = comment.get('id')
            
            if not all([owner, repo_name, pr_number]):
                logger.error("Missing required fields in payload")
                return
            
            pr_url = f"https://github.com/{owner}/{repo_name}/pull/{pr_number}"
            
            logger.info(f"Processing command on {pr_url}: {comment_body[:50]}...")
            
            reload_model_config()
            
            gh_client = GitHubClient(config)
            
            gh_client.add_reaction(owner, repo_name, comment_id, 'eyes')
            
            pr_context = extract_pr_context(gh_client, owner, repo_name, pr_number)
            
            if not pr_context:
                gh_client.add_reaction(owner, repo_name, comment_id, 'confused')
                gh_client.post_comment(
                    owner, repo_name, pr_number,
                    "⚠️ Failed to fetch PR information. Please try again."
                )
                return
            
            result = pr_agent_runner.process_comment(comment_body, pr_url, pr_context)
            
            if result:
                elapsed = time.time() - start_time
                
                if result.get('success'):
                    gh_client.add_reaction(owner, repo_name, comment_id, 'rocket')
                    logger.info(
                        f"Command /{result.get('command')} completed in {elapsed:.1f}s "
                        f"(RAG: {result.get('rag_chunks', 0)} chunks)"
                    )
                else:
                    gh_client.add_reaction(owner, repo_name, comment_id, 'confused')
                    error_msg = result.get('error', result.get('stderr', 'Unknown error'))[:500]
                    logger.error(f"Command failed: {error_msg}")
                    
                    gh_client.post_comment(
                        owner, repo_name, pr_number,
                        f"⚠️ PR-Agent command failed:\n```\n{error_msg}\n```"
                    )
        
        elif event_type == 'pull_request':
            action = payload.get('action')
            logger.info(f"Received pull_request event: {action} (automatic triggers disabled)")
    
    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)


@app.route('/health', methods=['GET'])
def health_check():
    health = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',
        'components': {}
    }
    
    if rag_retriever:
        try:
            rag_ok, rag_msg = rag_retriever.health_check()
            health['components']['rag'] = {'ok': rag_ok, 'message': rag_msg}
        except Exception as e:
            health['components']['rag'] = {'ok': False, 'message': str(e)}
    else:
        health['components']['rag'] = {'ok': False, 'message': 'Not initialized'}
    
    health['components']['model'] = {
        'endpoint': config.model.endpoint if config else 'not configured',
        'model': config.model.model_name if config else 'not configured'
    }
    
    health['components']['github'] = {
        'auth': 'app' if (config and config.github.is_app_configured()) else 'token' if (config and config.github.is_token_configured()) else 'none'
    }
    
    health['config'] = {
        'max_diff_size_mb': config.server.max_diff_size / 1024 / 1024 if config else 0,
        'max_files': config.server.max_files_per_pr if config else 0,
        'timeout_s': config.server.request_timeout if config else 0,
        'rag_enabled_commands': config.rag.enabled_commands if config else []
    }
    
    all_ok = all(
        c.get('ok', True) for c in health['components'].values()
        if isinstance(c, dict) and 'ok' in c
    )
    health['status'] = 'healthy' if all_ok else 'degraded'
    
    return jsonify(health), 200 if all_ok else 503


@app.route('/webhook', methods=['POST'])
def webhook():
    signature = request.headers.get('X-Hub-Signature-256', '') or request.headers.get('X-Hub-Signature', '')
    
    if config.github.webhook_secret:
        if not verify_github_signature(request.data, signature, config.github.webhook_secret):
            logger.warning("Invalid webhook signature")
            return jsonify({'error': 'Invalid signature'}), 401
    
    event_type = request.headers.get('X-GitHub-Event', '')
    delivery_id = request.headers.get('X-GitHub-Delivery', '')
    
    logger.info(f"Webhook received: event={event_type}, delivery={delivery_id}")
    
    if event_type == 'ping':
        return jsonify({'message': 'pong', 'delivery_id': delivery_id}), 200
    
    if event_type not in ['issue_comment', 'pull_request']:
        return jsonify({'message': f'Event {event_type} not handled'}), 200
    
    try:
        payload = request.get_json()
    except Exception as e:
        logger.error(f"Failed to parse payload: {e}")
        return jsonify({'error': 'Invalid JSON'}), 400
    
    thread = threading.Thread(
        target=process_webhook_async,
        args=(event_type, payload),
        daemon=True
    )
    thread.start()
    
    return jsonify({'message': 'Processing', 'delivery_id': delivery_id}), 202


@app.route('/test', methods=['POST'])
def test_endpoint():
    try:
        data = request.get_json()
        pr_url = data.get('pr_url')
        command = data.get('command', 'review')
        
        if not pr_url:
            return jsonify({'error': 'pr_url required'}), 400
        
        parts = pr_url.rstrip('/').split('/')
        owner = parts[-4]
        repo = parts[-3]
        pr_number = int(parts[-1])
        
        reload_model_config()
        
        gh_client = GitHubClient(config)
        pr_context = extract_pr_context(gh_client, owner, repo, pr_number)
        
        if not pr_context:
            return jsonify({'error': 'Failed to fetch PR context'}), 500
        
        result = pr_agent_runner.run_command(
            pr_url=pr_url,
            command=command,
            pr_context=pr_context
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/context', methods=['POST'])
def get_context():
    try:
        data = request.get_json()
        
        if not rag_retriever:
            return jsonify({'error': 'RAG not initialized'}), 503
        
        chunks = rag_retriever.retrieve(
            pr_title=data.get('title', ''),
            pr_description=data.get('description', ''),
            changed_files=data.get('changed_files', []),
            diff=data.get('diff', '')
        )
        
        context = rag_retriever.format_context(chunks)
        
        return jsonify({
            'success': True,
            'count': len(chunks),
            'context_size': len(context),
            'chunks': [
                {
                    'file_path': c.file_path,
                    'module': c.module,
                    'chunk_type': c.chunk_type,
                    'relevance_score': round(c.relevance_score, 3),
                    'lines': f"{c.start_line}-{c.end_line}",
                    'code_preview': c.code[:200] + '...' if len(c.code) > 200 else c.code
                }
                for c in chunks
            ],
            'formatted_context': context[:5000] + '...' if len(context) > 5000 else context
        })
        
    except Exception as e:
        logger.error(f"Context error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def init_app():
    global config, rag_retriever, pr_agent_runner
    
    logger.info("=" * 70)
    logger.info("OpenCV PR-Agent RAG v2.0 - Starting")
    logger.info("=" * 70)
    
    config = load_config()
    config.log_config()
    
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        logger.warning("Starting with incomplete configuration")
    
    try:
        rag_retriever = RAGRetriever(config.qdrant, config.rag)
        logger.info("RAG retriever initialized")
    except Exception as e:
        logger.error(f"RAG init failed: {e}")
        rag_retriever = None
    
    pr_agent_runner = PRAgentRunner(config, rag_retriever)
    logger.info("PR-Agent runner initialized")
    
    logger.info("=" * 70)
    logger.info("Server ready - waiting for webhooks")
    logger.info("=" * 70)


init_app()


if __name__ == '__main__':
    app.run(
        host=config.server.host,
        port=config.server.port,
        debug=False,
        threaded=True
    )