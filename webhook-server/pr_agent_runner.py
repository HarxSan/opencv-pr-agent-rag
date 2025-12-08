import logging
import subprocess
import os
import shutil
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path
import re

from config import Config
from rag_retriever import RAGRetriever

logger = logging.getLogger(__name__)


VALID_COMMANDS = {
    'review', 'improve', 'ask', 'describe',
    'update_changelog', 'add_docs', 'test', 'help', 'config'
}

RAG_CONTEXT_COMMANDS = {'review', 'improve', 'ask'}


class PRAgentRunner:
    def __init__(self, config: Config, rag_retriever: Optional[RAGRetriever]):
        self.config = config
        self.rag_retriever = rag_retriever
        self._config_dir: Optional[Path] = None
    
    def _get_base_instructions(self, command: str) -> str:
        instructions = {
            'review': """You are reviewing code for the OpenCV computer vision library.

Focus your review on:
1. Memory Management: cv::Mat lifecycle, reference counting, buffer allocations
2. Algorithm Correctness: numerical stability, edge cases, proper handling of empty inputs
3. Performance: SIMD optimization opportunities, cache-friendly access patterns, avoid unnecessary copies
4. Thread Safety: proper use of parallel_for_, atomic operations, mutex usage
5. API Consistency: follow OpenCV naming conventions, proper use of InputArray/OutputArray
6. Error Handling: CV_Assert usage, proper error messages, exception safety

When referencing the codebase context below, cite specific functions and patterns.""",

            'improve': """Suggest improvements following OpenCV coding standards.

Focus on:
1. Performance optimizations using SIMD intrinsics or parallel_for_
2. Memory efficiency (avoid unnecessary Mat copies, use move semantics)
3. Code clarity and maintainability
4. Proper error handling with CV_Assert/CV_Error
5. Documentation improvements (@brief, @param, @returns)

Reference the codebase context to suggest patterns already used in OpenCV.""",

            'ask': """Answer questions about this OpenCV pull request.

Use the codebase context provided to give accurate, specific answers.
Reference actual file paths, function names, and line numbers when relevant.
If the context doesn't contain relevant information, say so clearly."""
        }
        return instructions.get(command, "")
    
    def _build_extra_instructions(self, command: str, rag_context: str,
                                   user_question: Optional[str] = None) -> str:
        parts = []
        
        base = self._get_base_instructions(command)
        if base:
            parts.append(base)
        
        if rag_context:
            parts.append("\n\n" + rag_context)
        
        if user_question and command == 'ask':
            parts.append(f"\n\nUser Question: {user_question}")
        
        return "\n".join(parts)
    
    def _setup_config_directory(self, extra_instructions: str = "") -> Path:
        if self._config_dir and self._config_dir.exists():
            shutil.rmtree(self._config_dir)
        
        config_dir = Path(tempfile.mkdtemp(prefix="pr_agent_"))
        settings_dir = config_dir / "pr_agent" / "settings"
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        escaped_instructions = extra_instructions.replace('"""', '\\"\\"\\"').replace("'''", "\\'\\'\\'")
        
        config_toml = f'''[config]
git_provider = "github"
publish_output = true
publish_output_progress = true
verbosity_level = 2
use_repo_settings_file = false
use_wiki_settings_file = false
use_global_settings_file = false

model = "openai/{self.config.model.model_name}"
fallback_models = ["openai/{self.config.model.model_name}"]
custom_model_max_tokens = {self.config.model.max_tokens}
max_model_tokens = {self.config.model.max_tokens}
duplicate_examples = true

ai_timeout = {self.config.server.request_timeout}
large_patch_policy = "clip"

patch_extra_lines_before = 5
patch_extra_lines_after = 3
allow_dynamic_context = true
max_extra_lines_before_dynamic_context = 10

[pr_reviewer]
require_score_review = false
require_tests_review = true
require_security_review = true
require_estimate_effort_to_review = true
require_can_be_split_review = false
num_code_suggestions = 4
inline_code_comments = false
ask_and_reflect = false
automatic_review = false
persistent_comment = true
enable_help_text = true
extra_instructions = """{escaped_instructions}"""

[pr_code_suggestions]
num_code_suggestions = 6
rank_suggestions = true
commitable_code_suggestions = false
focus_only_on_problems = true
persistent_comment = true
suggestions_score_threshold = 0
extra_instructions = """{escaped_instructions}"""

[pr_description]
publish_labels = true
use_bullet_points = true
add_original_user_description = true
keep_original_user_title = false
extra_instructions = ""

[pr_questions]
extra_instructions = """{escaped_instructions}"""

[pr_add_docs]
docs_style = "Google"
extra_instructions = ""

[pr_update_changelog]
push_changelog_changes = false
extra_instructions = ""

[github_app]
pr_commands = []
handle_push_trigger = false
push_commands = []

[github_action_config]
auto_review = false
auto_describe = false
auto_improve = false
'''
        
        (settings_dir / "configuration.toml").write_text(config_toml)
        
        secrets_toml = f'''[openai]
key = "{self.config.model.api_key}"
api_base = "{self.config.model.api_base}"

[github]
user_token = "{self.config.github.user_token or ''}"
'''
        
        if self.config.github.is_app_configured():
            secrets_toml += f'''
app_id = "{self.config.github.app_id}"
webhook_secret = "{self.config.github.webhook_secret or ''}"
'''
            private_key = self.config.github.private_key
            if private_key:
                escaped_key = private_key.replace('"', '\\"')
                secrets_toml += f'private_key = """{escaped_key}"""\n'
        
        secrets_path = settings_dir / ".secrets.toml"
        secrets_path.write_text(secrets_toml)
        os.chmod(secrets_path, 0o600)
        
        self._config_dir = config_dir
        logger.debug(f"PR-Agent config created at {config_dir}")
        
        return config_dir
    
    def _build_environment(self, config_dir: Path) -> Dict[str, str]:
        env = os.environ.copy()
        
        env['OPENAI__KEY'] = self.config.model.api_key
        env['OPENAI__API_BASE'] = self.config.model.api_base
        
        env['CONFIG__GIT_PROVIDER'] = 'github'
        env['CONFIG__MODEL'] = f'openai/{self.config.model.model_name}'
        env['CONFIG__CUSTOM_MODEL_MAX_TOKENS'] = str(self.config.model.max_tokens)
        env['CONFIG__DUPLICATE_EXAMPLES'] = 'true'
        
        if self.config.github.user_token:
            env['GITHUB__USER_TOKEN'] = self.config.github.user_token
        
        if self.config.github.is_app_configured():
            env['GITHUB__APP_ID'] = self.config.github.app_id
            if self.config.github.webhook_secret:
                env['GITHUB__WEBHOOK_SECRET'] = self.config.github.webhook_secret
        
        settings_path = config_dir / "pr_agent" / "settings"
        env['PR_AGENT_SETTINGS_PATH'] = str(settings_path)
        
        return env
    
    def _parse_command(self, comment_body: str) -> tuple[Optional[str], Optional[str]]:
        comment_body = comment_body.strip()
        
        if not comment_body.startswith('/'):
            return None, None
        
        pattern = r'^/(\w+)(?:\s+(.*))?$'
        match = re.match(pattern, comment_body, re.DOTALL)
        
        if not match:
            return None, None
        
        command = match.group(1).lower()
        args = match.group(2).strip() if match.group(2) else None
        
        if command not in VALID_COMMANDS:
            logger.debug(f"Unknown command: {command}")
            return None, None
        
        return command, args
    
    def _retrieve_rag_context(self, command: str, pr_context: Dict[str, Any]) -> str:
        if not self.rag_retriever:
            logger.debug("RAG retriever not available")
            return ""
        
        if command not in RAG_CONTEXT_COMMANDS:
            logger.debug(f"RAG not enabled for command: {command}")
            return ""
        
        if not self.config.rag.is_rag_enabled_for(command):
            logger.debug(f"RAG disabled for command: {command}")
            return ""
        
        try:
            chunks = self.rag_retriever.retrieve(
                pr_title=pr_context.get('title', ''),
                pr_description=pr_context.get('description', ''),
                changed_files=pr_context.get('changed_files', []),
                diff=pr_context.get('diff', '')
            )
            
            if not chunks:
                logger.info("No RAG context chunks found")
                return ""
            
            context = self.rag_retriever.format_context(chunks)
            logger.info(f"Retrieved {len(chunks)} RAG chunks, {len(context)} chars")
            return context
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return ""
    
    def run_command(self, pr_url: str, command: str, args: Optional[str] = None,
                    pr_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        logger.info(f"Running /{command} on {pr_url}")
        
        rag_context = ""
        if pr_context:
            rag_context = self._retrieve_rag_context(command, pr_context)
        
        extra_instructions = self._build_extra_instructions(command, rag_context, args if command == 'ask' else None)
        
        config_dir = self._setup_config_directory(extra_instructions)
        env = self._build_environment(config_dir)
        
        cli_args = [
            'python', '-m', 'pr_agent.cli',
            f'--pr_url={pr_url}',
            command
        ]
        
        if command == 'ask' and args:
            cli_args.append(args)
        
        logger.info(f"Executing: pr_agent.cli {command}")
        logger.debug(f"Model: {self.config.model.model_name}")
        logger.debug(f"API Base: {self.config.model.api_base}")
        
        try:
            result = subprocess.run(
                cli_args,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.server.request_timeout,
                cwd=str(config_dir)
            )
            
            success = result.returncode == 0
            
            if not success:
                logger.error(f"PR-Agent failed with return code {result.returncode}")
                logger.error(f"stderr: {result.stderr[:1000] if result.stderr else 'empty'}")
            else:
                logger.info(f"PR-Agent completed successfully")
            
            return {
                'success': success,
                'command': command,
                'pr_url': pr_url,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'rag_context_size': len(rag_context),
                'rag_chunks': rag_context.count('---') // 2 if rag_context else 0
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {self.config.server.request_timeout}s")
            return {
                'success': False,
                'command': command,
                'pr_url': pr_url,
                'error': f'Command timed out after {self.config.server.request_timeout}s'
            }
        except Exception as e:
            logger.error(f"Error executing PR-Agent: {e}")
            return {
                'success': False,
                'command': command,
                'pr_url': pr_url,
                'error': str(e)
            }
        finally:
            if self._config_dir and self._config_dir.exists():
                try:
                    shutil.rmtree(self._config_dir)
                except Exception:
                    pass
                self._config_dir = None
    
    def process_comment(self, comment_body: str, pr_url: str,
                        pr_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        command, args = self._parse_command(comment_body)
        
        if not command:
            logger.debug(f"Not a valid command: {comment_body[:50]}...")
            return None
        
        logger.info(f"Processing /{command}" + (f" with args: {args[:50]}..." if args else ""))
        
        return self.run_command(pr_url, command, args, pr_context)