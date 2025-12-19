import logging
import requests
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class PATGitHubClient:

    def __init__(self, token: str):
        self.token = token
        self._session = requests.Session()
        self._session.headers.update({
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28'
        })
        logger.info("PAT GitHub client initialized")

    def get_pr_details(self, repo_full_name: str, pr_number: int) -> dict:
        try:
            response = self._session.get(
                f'https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}',
                timeout=30
            )
            response.raise_for_status()
            pr = response.json()

            changed_files = []
            try:
                files_response = self._session.get(
                    f'https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files',
                    timeout=30
                )
                files_response.raise_for_status()
                files = files_response.json()

                for file in files:
                    changed_files.append({
                        'filename': file['filename'],
                        'additions': file.get('additions', 0),
                        'deletions': file.get('deletions', 0),
                        'changes': file.get('changes', 0),
                        'patch': file.get('patch')
                    })
            except Exception as e:
                logger.warning(f"Could not fetch files for PR #{pr_number}: {e}")

            return {
                'number': pr['number'],
                'title': pr['title'],
                'body': pr.get('body') or '',
                'state': pr['state'],
                'author': pr['user']['login'],
                'base_ref': pr['base']['ref'],
                'head_ref': pr['head']['ref'],
                'diff_url': pr['diff_url'],
                'changed_files': changed_files,
                'additions': pr.get('additions', 0),
                'deletions': pr.get('deletions', 0),
                'changed_files_count': pr.get('changed_files', 0)
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"GitHub API error fetching PR: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching PR: {e}")
            raise

    def post_pr_comment(self, repo_full_name: str, pr_number: int, comment: str) -> bool:
        try:
            response = self._session.post(
                f'https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments',
                json={'body': comment},
                timeout=30
            )
            response.raise_for_status()
            logger.info(f"Posted comment to PR #{pr_number} in {repo_full_name}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to post comment: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error posting comment: {e}")
            return False

    def get_pr_diff(self, diff_url: str) -> Optional[str]:
        try:
            response = self._session.get(
                diff_url,
                headers={'Accept': 'application/vnd.github.v3.diff'},
                timeout=30
            )
            response.raise_for_status()
            return response.text

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch diff: {e}")
            return None

    def add_reaction(self, repo_full_name: str, comment_id: int, reaction: str) -> bool:
        try:
            response = self._session.post(
                f'https://api.github.com/repos/{repo_full_name}/issues/comments/{comment_id}/reactions',
                json={'content': reaction},
                timeout=10
            )
            return response.status_code in [200, 201]
        except Exception as e:
            logger.warning(f"Failed to add reaction: {e}")
            return False
