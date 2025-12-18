import os
import sys
import logging
import requests
import hmac
import hashlib
import threading
from datetime import datetime
from flask import Flask, request, jsonify
from github_app_auth import GitHubAppAuth
from webhook_verifier import WebhookVerifier
from multi_tenant_github import MultiTenantGitHubClient
from rag_retriever import RAGRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/server.log')
    ]
)

logger = logging.getLogger(__name__)

GITHUB_APP_ID = os.getenv('GITHUB_APP_ID')
GITHUB_APP_PRIVATE_KEY_PATH = os.getenv('GITHUB_APP_PRIVATE_KEY_PATH')
GITHUB_WEBHOOK_SECRET = os.getenv('GITHUB_WEBHOOK_SECRET')
LIGHTNING_AI_ENDPOINT = os.getenv('LIGHTNING_AI_ENDPOINT')
LIGHTNING_AI_MODEL_NAME = os.getenv('LIGHTNING_AI_MODEL_NAME')
MODEL_MAX_TOKENS = int(os.getenv('MODEL_MAX_TOKENS', '4096'))
MODEL_TEMPERATURE = float(os.getenv('MODEL_TEMPERATURE', '0.1'))

if not all([GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY_PATH, GITHUB_WEBHOOK_SECRET]):
    logger.error("Missing required environment variables")
    sys.exit(1)

app = Flask(__name__)

try:
    auth_manager = GitHubAppAuth(GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY_PATH)
    webhook_verifier = WebhookVerifier(GITHUB_WEBHOOK_SECRET)
    github_client = MultiTenantGitHubClient(auth_manager)
    
    try:
        rag_retriever = RAGRetriever()
        logger.info("RAG retriever initialized successfully")
    except Exception as e:
        rag_retriever = None
        logger.warning(f"RAG retriever initialization failed: {e}")
    
    logger.info("Server initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize server: {e}")
    sys.exit(1)

@app.route('/health', methods=['GET'])
def health():
    cache_stats = auth_manager.get_cache_stats()
    
    rag_status = {'available': False, 'message': 'Not initialized'}
    if rag_retriever:
        try:
            rag_ok, rag_msg = rag_retriever.health_check()
            rag_status = {'available': rag_ok, 'message': rag_msg}
        except:
            rag_status = {'available': False, 'message': 'Health check failed'}
    
    return jsonify({
        'status': 'healthy',
        'app_id': GITHUB_APP_ID,
        'cached_installations': cache_stats['cached_installations'],
        'model_endpoint': LIGHTNING_AI_ENDPOINT,
        'rag': rag_status
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    payload_bytes = request.get_data()
    
    if not webhook_verifier.verify_request(payload_bytes, request.headers):
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
        return jsonify({'message': 'Processing'}), 202
    
    if event_type == 'pull_request':
        return handle_pull_request(payload)
    
    logger.info(f"Unhandled event type: {event_type}")
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
        
        installation_id = payload['installation']['id']
        repo_full_name = payload['repository']['full_name']
        pr_number = payload['issue']['number']
        comment_id = payload['comment']['id']
        commenter = payload['comment']['user']['login']
        
        logger.info(f"Processing '{comment_body}' from {commenter} on PR #{pr_number} in {repo_full_name}")
        
        owner, repo = repo_full_name.split('/')
        
        try:
            github_client.get_client(installation_id).get_user().login
            react_result = requests.post(
                f'https://api.github.com/repos/{repo_full_name}/issues/comments/{comment_id}/reactions',
                headers={
                    'Authorization': f'Bearer {auth_manager.get_installation_token(installation_id)}',
                    'Accept': 'application/vnd.github+json'
                },
                json={'content': 'eyes'},
                timeout=10
            )
        except:
            pass
        
        if comment_body == '/review':
            process_review_command(installation_id, repo_full_name, pr_number)
        
    except Exception as e:
        logger.error(f"Error in async handler: {e}", exc_info=True)

def handle_pull_request(payload: dict):
    try:
        action = payload.get('action')
        if action not in ['opened', 'synchronize', 'reopened']:
            return jsonify({'message': 'Ignored'}), 200
        
        logger.info(f"PR {action}: {payload['pull_request']['title']}")
        return jsonify({'message': 'PR event received'}), 200
        
    except Exception as e:
        logger.error(f"Error handling PR event: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def process_review_command(installation_id: int, repo_full_name: str, pr_number: int):
    try:
        logger.info(f"Fetching PR details for #{pr_number} in {repo_full_name}")
        pr_details = github_client.get_pr_details(installation_id, repo_full_name, pr_number)
        
        diff = github_client.get_pr_diff(installation_id, pr_details['diff_url'])
        if not diff:
            logger.warning(f"Could not fetch diff for PR #{pr_number}")
            diff = ""
        
        rag_chunks = []
        if rag_retriever:
            logger.info(f"Fetching RAG context for PR #{pr_number}")
            try:
                rag_chunks = rag_retriever.retrieve(
                    pr_title=pr_details['title'],
                    pr_description=pr_details['body'],
                    changed_files=[f['filename'] for f in pr_details['changed_files']],
                    diff=diff
                )
                logger.info(f"Retrieved {len(rag_chunks)} RAG chunks")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        else:
            logger.info("RAG not available, proceeding without context")
        
        logger.info(f"Generating review for PR #{pr_number} with {len(rag_chunks)} RAG chunks")
        review = generate_review(pr_details, diff, rag_chunks)
        
        formatted_review = f"""## 🤖 AI-Powered Code Review

{review}

---
*Review generated by OpenCV PR Agent with RAG-enhanced context*  
*Found {len(rag_chunks)} relevant code chunks from the codebase*
"""
        
        success = github_client.post_pr_comment(
            installation_id, 
            repo_full_name, 
            pr_number, 
            formatted_review
        )
        
        if success:
            logger.info(f"Successfully posted review to PR #{pr_number}")
        else:
            logger.error(f"Failed to post review to PR #{pr_number}")
            
    except Exception as e:
        logger.error(f"Error processing review: {e}", exc_info=True)

def generate_review(pr_details: dict, diff: str, rag_chunks: list) -> str:
    try:
        if rag_chunks:
            context_text = "\n\n".join([
                f"File: {chunk.file_path}\nModule: {chunk.module}\nLines {chunk.start_line}-{chunk.end_line}:\n{chunk.code[:400]}"
                for chunk in rag_chunks[:10]
            ])
        else:
            context_text = "No relevant context found in codebase"
        
        changed_files_summary = ', '.join([f['filename'] for f in pr_details['changed_files'][:10]])
        if len(pr_details['changed_files']) > 10:
            changed_files_summary += f" (+{len(pr_details['changed_files']) - 10} more)"
        
        max_diff_length = 6000
        if len(diff) > max_diff_length:
            diff = diff[:max_diff_length] + f"\n... (diff truncated, {len(diff) - max_diff_length} chars omitted)"
        
        max_context_length = 4000
        if len(context_text) > max_context_length:
            context_text = context_text[:max_context_length] + "\n... (context truncated)"
        
        prompt = f"""You are an expert OpenCV code reviewer. Review this pull request.

PR Title: {pr_details['title']}
Author: {pr_details['author']}
Changed Files: {changed_files_summary}

Relevant Code Context from Codebase:
{context_text}

PR Diff:
{diff}

Provide a concise code review focusing on:
1. Correctness and potential bugs
2. OpenCV API usage and best practices
3. Memory management and performance
4. Code quality and maintainability

Keep the review focused and actionable. Be specific with line numbers and code snippets where relevant."""

        response = requests.post(
            f'{LIGHTNING_AI_ENDPOINT}/v1/chat/completions',
            json={
                'model': LIGHTNING_AI_MODEL_NAME,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': MODEL_MAX_TOKENS,
                'temperature': MODEL_TEMPERATURE
            },
            timeout=120
        )
        response.raise_for_status()
        
        return response.json()['choices'][0]['message']['content']
        
    except Exception as e:
        logger.error(f"Review generation failed: {e}")
        return f"Failed to generate review: {str(e)}"

if __name__ == '__main__':
    port = int(os.getenv('SERVER_PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)