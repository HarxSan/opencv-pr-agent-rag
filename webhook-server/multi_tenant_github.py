import logging
from typing import Optional
from github import Github, GithubException
from github_app_auth import GitHubAppAuth

logger = logging.getLogger(__name__)

class MultiTenantGitHubClient:
    
    def __init__(self, auth_manager: GitHubAppAuth):
        self.auth_manager = auth_manager
        self._clients = {}
        logger.info("Multi-tenant GitHub client initialized")
    
    def get_client(self, installation_id: int) -> Github:
        try:
            token = self.auth_manager.get_installation_token(installation_id)
            
            if installation_id in self._clients:
                client = self._clients[installation_id]
                try:
                    client.get_user().login
                    logger.debug(f"Reusing client for installation {installation_id}")
                    return client
                except GithubException:
                    logger.info(f"Client token expired for installation {installation_id}, refreshing")
                    del self._clients[installation_id]
            
            client = Github(token, per_page=100)
            self._clients[installation_id] = client
            logger.info(f"Created new client for installation {installation_id}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to create client for installation {installation_id}: {e}")
            raise
    
    def get_pr_details(self, installation_id: int, repo_full_name: str, pr_number: int) -> dict:
        try:
            client = self.get_client(installation_id)
            repo = client.get_repo(repo_full_name)
            pr = repo.get_pull(pr_number)
            
            changed_files = []
            try:
                for file in pr.get_files():
                    changed_files.append({
                        'filename': file.filename,
                        'additions': file.additions,
                        'deletions': file.deletions,
                        'changes': file.changes,
                        'patch': file.patch
                    })
            except GithubException as e:
                logger.warning(f"Could not fetch files for PR #{pr_number}: {e}")
            
            return {
                'number': pr.number,
                'title': pr.title,
                'body': pr.body or '',
                'state': pr.state,
                'author': pr.user.login,
                'base_ref': pr.base.ref,
                'head_ref': pr.head.ref,
                'diff_url': pr.diff_url,
                'changed_files': changed_files,
                'additions': pr.additions,
                'deletions': pr.deletions,
                'changed_files_count': pr.changed_files
            }
            
        except GithubException as e:
            logger.error(f"GitHub API error fetching PR: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching PR: {e}")
            raise
    
    def post_pr_comment(self, installation_id: int, repo_full_name: str, 
                        pr_number: int, comment: str) -> bool:
        try:
            client = self.get_client(installation_id)
            repo = client.get_repo(repo_full_name)
            pr = repo.get_pull(pr_number)
            
            pr.create_issue_comment(comment)
            logger.info(f"Posted comment to PR #{pr_number} in {repo_full_name}")
            return True
            
        except GithubException as e:
            logger.error(f"Failed to post comment: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error posting comment: {e}")
            return False
    
    def get_pr_diff(self, installation_id: int, diff_url: str) -> Optional[str]:
        try:
            import requests
            token = self.auth_manager.get_installation_token(installation_id)
            
            response = requests.get(
                diff_url,
                headers={
                    'Authorization': f'token {token}',
                    'Accept': 'application/vnd.github.v3.diff'
                },
                timeout=30
            )
            response.raise_for_status()
            
            return response.text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch diff: {e}")
            return None
    
    def cleanup_client(self, installation_id: int):
        if installation_id in self._clients:
            del self._clients[installation_id]
            logger.info(f"Cleaned up client for installation {installation_id}")