import os
import sys
import logging
import threading
from pathlib import Path
from flask import Flask, request, jsonify

from config import load_config
from github_app_auth import GitHubAppAuth
from webhook_verifier import WebhookVerifier
from multi_tenant_github import MultiTenantGitHubClient
from pat_github import PATGitHubClient
from rag_retriever import RAGRetriever
from pr_agent_runner import PRAgentRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/server.log')
    ]
)

logger = logging.getLogger(__name__)

config = load_config()
errors = config.validate()
if errors:
    logger.error("Configuration errors:")
    for error in errors:
        logger.error(f"  - {error}")
    sys.exit(1)

config.log_config()

app = Flask(__name__)

auth_manager = None
webhook_verifier = None
github_app_client = None
pat_client = None
rag_retriever = None
pr_agent_runner = None

def initialize_components():
    global auth_manager, webhook_verifier, github_app_client, pat_client, rag_retriever, pr_agent_runner

    try:
        if config.github.is_app_configured():
            auth_manager = GitHubAppAuth(
                config.github.app_id,
                config.github.private_key_path
            )
            if config.github.webhook_secret:
                webhook_verifier = WebhookVerifier(config.github.webhook_secret)
            github_app_client = MultiTenantGitHubClient(auth_manager)
            logger.info("GitHub App auth initialized")

        if config.github.is_token_configured():
            pat_client = PATGitHubClient(config.github.user_token)
            if not config.github.is_app_configured() and config.github.webhook_secret:
                webhook_verifier = WebhookVerifier(config.github.webhook_secret)
                logger.info("Using Personal Access Token auth with webhook verification")
            elif not config.github.is_app_configured():
                logger.info("Using Personal Access Token auth without webhook verification")

        try:
            rag_retriever = RAGRetriever(config.qdrant, config.rag)
            rag_ok, rag_msg = rag_retriever.health_check()
            if rag_ok:
                logger.info(f"RAG initialized: {rag_msg}")
            else:
                logger.warning(f"RAG health check failed: {rag_msg}")
                rag_retriever = None
        except Exception as e:
            logger.warning(f"RAG initialization failed: {e}")
            rag_retriever = None

        pr_agent_runner = PRAgentRunner(config, rag_retriever)
        logger.info("PR-Agent runner initialized")

        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        sys.exit(1)

initialize_components()

@app.route('/health', methods=['GET'])
def health():
    health_data = {
        'status': 'healthy',
        'timestamp': str(Path('/app/logs/server.log').stat().st_mtime if Path('/app/logs/server.log').exists() else 0),
        'config': {
            'auth': 'GitHub App' if config.github.is_app_configured() else 'PAT' if config.github.is_token_configured() else 'none',
            'endpoint': config.model.endpoint,
            'model': config.model.model_name
        }
    }
    
    if auth_manager:
        cache_stats = auth_manager.get_cache_stats()
        health_data['github_app'] = {
            'cached_installations': cache_stats['cached_installations']
        }

    if pat_client:
        health_data['pat_client'] = 'configured'
    
    if rag_retriever:
        rag_ok, rag_msg = rag_retriever.health_check()
        health_data['rag'] = {
            'ok': rag_ok,
            'message': rag_msg
        }
    else:
        health_data['rag'] = {
            'ok': False,
            'message': 'Not initialized'
        }
    
    return jsonify(health_data)

@app.route('/webhook', methods=['POST'])
def webhook():
    payload_bytes = request.get_data()
    
    if webhook_verifier and not webhook_verifier.verify_request(payload_bytes, request.headers):
        logger.warning(f"Invalid webhook signature from {request.remote_addr}")
        return jsonify({'error': 'Invalid signature'}), 401
    
    event_type = request.headers.get('X-GitHub-Event')
    payload = request.json
    
    if event_type == 'ping':
        logger.info("Received ping event")
        return jsonify({'message': 'Pong'}), 200
    
    if event_type == 'issue_comment':
        thread = threading.Thread(
            target=handle_issue_comment_async,
            args=(payload,),
            daemon=True
        )
        thread.start()
        return jsonify({'message': 'Processing comment'}), 202
    
    if event_type == 'pull_request':
        action = payload.get('action')
        if action in ['opened', 'synchronize', 'reopened']:
            logger.info(f"PR {action}: {payload['pull_request']['title']}")
        return jsonify({'message': 'PR event received'}), 200
    
    logger.debug(f"Unhandled event type: {event_type}")
    return jsonify({'message': 'Event received'}), 200

def handle_issue_comment_async(payload: dict):
    try:
        action = payload.get('action')
        if action != 'created':
            return

        if 'pull_request' not in payload.get('issue', {}):
            return

        comment_body = payload['comment']['body'].strip()
        if not comment_body.startswith('/'):
            return

        repo_full_name = payload['repository']['full_name']
        pr_number = payload['issue']['number']
        commenter = payload['comment']['user']['login']
        comment_id = payload['comment']['id']

        logger.info(f"Command '{comment_body}' from {commenter} on PR #{pr_number} in {repo_full_name}")

        pr_url = f"https://github.com/{repo_full_name}/pull/{pr_number}"

        installation_id = payload.get('installation', {}).get('id') if github_app_client else None

        if github_app_client and installation_id:
            add_reaction_app(installation_id, repo_full_name, comment_id, 'eyes')
        elif pat_client:
            add_reaction_pat(repo_full_name, comment_id, 'eyes')

        pr_context = None
        if github_app_client and installation_id:
            pr_context = fetch_pr_context_app(installation_id, repo_full_name, pr_number)
        elif pat_client:
            pr_context = fetch_pr_context_pat(repo_full_name, pr_number)

        result = pr_agent_runner.process_comment(comment_body, pr_url, pr_context)

        if result:
            if result['success']:
                logger.info(f"Successfully processed /{result['command']} for PR #{pr_number}")
                if github_app_client and installation_id:
                    add_reaction_app(installation_id, repo_full_name, comment_id, '+1')
                elif pat_client:
                    add_reaction_pat(repo_full_name, comment_id, '+1')
            else:
                logger.error(f"Failed to process command: {result.get('error', 'Unknown error')}")
                error_msg = result.get('error', 'Command failed')
                if github_app_client and installation_id:
                    post_error_comment_app(installation_id, repo_full_name, pr_number, error_msg)
                elif pat_client:
                    post_error_comment_pat(repo_full_name, pr_number, error_msg)

    except Exception as e:
        logger.error(f"Error in async handler: {e}", exc_info=True)

def fetch_pr_context_app(installation_id: int, repo_full_name: str, pr_number: int) -> dict:
    try:
        pr_details = github_app_client.get_pr_details(installation_id, repo_full_name, pr_number)
        diff = github_app_client.get_pr_diff(installation_id, pr_details['diff_url'])

        return {
            'title': pr_details['title'],
            'description': pr_details['body'],
            'changed_files': [f['filename'] for f in pr_details['changed_files']],
            'diff': diff or ''
        }
    except Exception as e:
        logger.error(f"Failed to fetch PR context via App: {e}")
        return {'title': '', 'description': '', 'changed_files': [], 'diff': ''}

def fetch_pr_context_pat(repo_full_name: str, pr_number: int) -> dict:
    try:
        pr_details = pat_client.get_pr_details(repo_full_name, pr_number)
        diff = pat_client.get_pr_diff(pr_details['diff_url'])

        return {
            'title': pr_details['title'],
            'description': pr_details['body'],
            'changed_files': [f['filename'] for f in pr_details['changed_files']],
            'diff': diff or ''
        }
    except Exception as e:
        logger.error(f"Failed to fetch PR context via PAT: {e}")
        return {'title': '', 'description': '', 'changed_files': [], 'diff': ''}

def add_reaction_app(installation_id: int, repo_full_name: str, comment_id: int, reaction: str):
    try:
        import requests
        token = auth_manager.get_installation_token(installation_id)
        requests.post(
            f'https://api.github.com/repos/{repo_full_name}/issues/comments/{comment_id}/reactions',
            headers={
                'Authorization': f'Bearer {token}',
                'Accept': 'application/vnd.github+json'
            },
            json={'content': reaction},
            timeout=10
        )
    except:
        pass

def add_reaction_pat(repo_full_name: str, comment_id: int, reaction: str):
    try:
        pat_client.add_reaction(repo_full_name, comment_id, reaction)
    except:
        pass

def post_error_comment_app(installation_id: int, repo_full_name: str, pr_number: int, error_msg: str):
    try:
        comment = f"❌ **PR-Agent Error**\n\n{error_msg}\n\nPlease check the command syntax or contact support."
        github_app_client.post_pr_comment(installation_id, repo_full_name, pr_number, comment)
    except:
        pass

def post_error_comment_pat(repo_full_name: str, pr_number: int, error_msg: str):
    try:
        comment = f"❌ **PR-Agent Error**\n\n{error_msg}\n\nPlease check the command syntax or contact support."
        pat_client.post_pr_comment(repo_full_name, pr_number, comment)
    except:
        pass

@app.route('/context', methods=['POST'])
def test_context():
    if not rag_retriever:
        return jsonify({'error': 'RAG not available'}), 503
    
    data = request.json
    try:
        chunks = rag_retriever.retrieve(
            pr_title=data.get('title', ''),
            pr_description=data.get('description', ''),
            changed_files=data.get('changed_files', []),
            diff=data.get('diff', '')
        )
        
        return jsonify({
            'success': True,
            'count': len(chunks),
            'chunks': [c.to_dict() for c in chunks[:5]]
        })
    except Exception as e:
        logger.error(f"Context test failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['POST'])
def test_command():
    data = request.json
    pr_url = data.get('pr_url')
    command = data.get('command', 'review')
    
    if not pr_url:
        return jsonify({'error': 'pr_url required'}), 400
    
    try:
        result = pr_agent_runner.run_command(pr_url, command)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Test command failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = config.server.port
    app.run(host=config.server.host, port=port, debug=False, threaded=True)