#!/usr/bin/env python3

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from config import load_config
from rag_retriever import RAGRetriever
from pr_agent_runner import PRAgentRunner
from pat_github import PATGitHubClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

VALID_COMMANDS = [
    'review', 'improve', 'ask', 'describe',
    'update_changelog', 'add_docs', 'test', 'help'
]

class PRReviewCLI:

    def __init__(self):
        self.config = load_config()
        self.rag_retriever = None
        self.pat_client = None
        self.pr_agent_runner = None

        errors = self.config.validate()
        if errors:
            logger.error("Configuration errors:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)

        self._initialize()

    def _initialize(self):
        try:
            self.rag_retriever = RAGRetriever(self.config.qdrant, self.config.rag)
            rag_ok, rag_msg = self.rag_retriever.health_check()
            if rag_ok:
                logger.info(f"RAG initialized: {rag_msg}")
            else:
                logger.warning(f"RAG health check failed: {rag_msg}")
                self.rag_retriever = None
        except Exception as e:
            logger.warning(f"RAG initialization failed: {e}")
            self.rag_retriever = None

        if self.config.github.is_token_configured():
            self.pat_client = PATGitHubClient(self.config.github.user_token)
            logger.info("PAT client initialized")
        else:
            logger.warning("No GitHub token configured - cannot fetch PR context or post reviews")

        self.pr_agent_runner = PRAgentRunner(self.config, self.rag_retriever)
        logger.info("PR-Agent runner initialized")

    def parse_pr_url(self, pr_url: str) -> Optional[tuple]:
        parts = pr_url.strip().rstrip('/').split('/')

        if len(parts) < 4:
            return None

        if 'github.com' not in pr_url:
            return None

        try:
            idx = parts.index('pull')
            if idx + 1 < len(parts):
                pr_number = int(parts[idx + 1])
                repo_parts = parts[idx - 2:idx]
                if len(repo_parts) == 2:
                    owner, repo = repo_parts
                    return owner, repo, pr_number
        except (ValueError, IndexError):
            pass

        return None

    def fetch_pr_context(self, owner: str, repo: str, pr_number: int) -> Optional[dict]:
        if not self.pat_client:
            logger.error("No GitHub client available to fetch PR context")
            return None

        try:
            repo_full_name = f"{owner}/{repo}"
            pr_details = self.pat_client.get_pr_details(repo_full_name, pr_number)
            diff = self.pat_client.get_pr_diff(pr_details['diff_url'])

            return {
                'title': pr_details['title'],
                'description': pr_details['body'],
                'changed_files': [f['filename'] for f in pr_details['changed_files']],
                'diff': diff or ''
            }
        except Exception as e:
            logger.error(f"Failed to fetch PR context: {e}")
            return None

    def run_review(self, pr_url: str, command: str, args: Optional[str] = None, post_to_github: bool = False):
        logger.info(f"Processing: {pr_url} with command /{command}")

        parsed = self.parse_pr_url(pr_url)
        if not parsed:
            logger.error(f"Invalid PR URL: {pr_url}")
            logger.error("Expected format: https://github.com/owner/repo/pull/123")
            return False

        owner, repo, pr_number = parsed
        logger.info(f"Parsed: {owner}/{repo}#{pr_number}")

        pr_context = self.fetch_pr_context(owner, repo, pr_number)
        if not pr_context:
            logger.warning("Could not fetch PR context - proceeding without RAG")

        if not post_to_github:
            logger.info("Running in read-only mode (will NOT post to GitHub)")
            result = self.pr_agent_runner.run_command(pr_url, command, args, pr_context, skip_github_auth=True)
        else:
            result = self.pr_agent_runner.run_command(pr_url, command, args, pr_context, skip_github_auth=False)

        if result:
            if result.get('success'):
                logger.info(f"✓ Command /{command} completed successfully")
                if result.get('stdout'):
                    print("\n" + "="*70)
                    print("PR-AGENT OUTPUT:")
                    print("="*70)
                    print(result['stdout'])
                    print("="*70 + "\n")
                return True
            else:
                logger.error(f"✗ Command /{command} failed")
                if result.get('error'):
                    logger.error(f"Error: {result['error']}")
                if result.get('stderr'):
                    print("\n" + "="*70)
                    print("ERROR OUTPUT:")
                    print("="*70)
                    print(result['stderr'])
                    print("="*70 + "\n")
                return False
        else:
            logger.error("No result from PR-Agent runner")
            return False

def main():
    parser = argparse.ArgumentParser(
        description='OpenCV PR-Agent RAG - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --pr-url https://github.com/opencv/opencv/pull/12345 review
  %(prog)s --pr-url https://github.com/opencv/opencv/pull/12345 improve --post
  %(prog)s --pr-url https://github.com/opencv/opencv/pull/12345 ask "What does this PR change?" --post
  %(prog)s --pr-url https://github.com/opencv/opencv/pull/12345 describe
        '''
    )

    parser.add_argument(
        '--pr-url',
        required=True,
        help='GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)'
    )

    parser.add_argument(
        'command',
        choices=VALID_COMMANDS,
        help='PR-Agent command to execute'
    )

    parser.add_argument(
        'args',
        nargs='?',
        help='Additional arguments for the command (e.g., question for /ask)'
    )

    parser.add_argument(
        '--post',
        action='store_true',
        help='Post review to GitHub (default: show in terminal only)'
    )

    parser.add_argument(
        '--yes',
        '-y',
        action='store_true',
        help='Skip confirmation prompt when --post is used'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cli = PRReviewCLI()

    post_to_github = args.post

    if post_to_github and not args.yes:
        print("\n" + "="*70)
        print("⚠ WARNING: Review will be POSTED to GitHub PR")
        print("="*70)
        response = input("Continue? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Cancelled.")
            sys.exit(0)
        print()

    success = cli.run_review(args.pr_url, args.command, args.args, post_to_github)

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
