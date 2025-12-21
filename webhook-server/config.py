import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class GitHubConfig:
    app_id: Optional[str] = None
    private_key_path: Optional[str] = None
    webhook_secret: Optional[str] = None
    user_token: Optional[str] = None
    repo: str = ""
    
    def __post_init__(self):
        self.app_id = os.getenv("GITHUB_APP_ID")
        self.private_key_path = os.getenv("GITHUB_APP_PRIVATE_KEY_PATH")
        self.webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET")
        self.user_token = os.getenv("GITHUB_USER_TOKEN")
        self.repo = os.getenv("GITHUB_REPO", "")
    
    @property
    def private_key(self) -> Optional[str]:
        if self.private_key_path:
            path = Path(self.private_key_path)
            if path.exists():
                return path.read_text()
        env_key = os.getenv("GITHUB_APP_PRIVATE_KEY")
        if env_key:
            return env_key.replace("\\n", "\n")
        return None
    
    def is_app_configured(self) -> bool:
        return bool(self.app_id and self.private_key and self.webhook_secret)
    
    def is_token_configured(self) -> bool:
        return bool(self.user_token)


@dataclass
class ModelConfig:
    endpoint: str = ""
    model_name: str = ""
    max_tokens: int = 32000
    temperature: float = 0.3
    api_key: str = "not-used"

    def __post_init__(self):
        self.endpoint = os.getenv("MODEL_ENDPOINT") or os.getenv(
            "LIGHTNING_AI_ENDPOINT",
            "https://midi-medieval-court-hosting.trycloudflare.com"
        )
        self.endpoint = self.endpoint.rstrip("/")

        self.model_name = os.getenv("MODEL_NAME") or os.getenv(
            "LIGHTNING_AI_MODEL_NAME",
            "nareshmlx/code-reviewer-opencv-harxsan-v2"
        )

        self.max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "32000"))
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", "0.3"))

        self.api_key = (
            os.getenv("VLLM_API_KEY") or
            os.getenv("MODEL_API_KEY") or
            os.getenv("LIGHTNING_AI_API_KEY") or
            "not-used"
        )
    
    @property
    def api_base(self) -> str:
        base = self.endpoint.rstrip("/")
        if not base.endswith("/v1"):
            return f"{base}/v1"
        return base
    
    @property
    def litellm_model_name(self) -> str:
        return f"openai/{self.model_name}"


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    collection_name: str = "opencv_codebase"
    
    def __post_init__(self):
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", "6333"))
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "opencv_codebase")


@dataclass
class RAGConfig:
    top_k: int = 15
    min_score: float = 0.4
    max_context_tokens: int = 12000
    max_context_chars: int = 48000
    enabled_commands: List[str] = field(default_factory=list)
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    cache_dir: str = "/app/cache/embeddings"
    
    def __post_init__(self):
        self.top_k = int(os.getenv("RAG_TOP_K", "15"))
        self.min_score = float(os.getenv("RAG_MIN_SCORE", "0.4"))
        self.max_context_tokens = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "12000"))
        self.max_context_chars = self.max_context_tokens * 4
        
        commands_str = os.getenv("RAG_ENABLED_COMMANDS", "review,improve,ask")
        self.enabled_commands = [cmd.strip().lower() for cmd in commands_str.split(",") if cmd.strip()]
        
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self.cache_dir = os.getenv("EMBEDDING_CACHE_DIR", "/app/cache/embeddings")
    
    def is_rag_enabled_for(self, command: str) -> bool:
        cmd = command.lower().lstrip("/")
        return cmd in self.enabled_commands


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    log_level: str = "INFO"
    max_diff_size: int = 10485760
    max_files_per_pr: int = 500
    request_timeout: int = 600
    
    def __post_init__(self):
        self.host = os.getenv("SERVER_HOST", "0.0.0.0")
        self.port = int(os.getenv("SERVER_PORT", "5000"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.max_diff_size = int(os.getenv("MAX_DIFF_SIZE", "10485760"))
        self.max_files_per_pr = int(os.getenv("MAX_FILES_PER_PR", "500"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "600"))


@dataclass
class Config:
    github: GitHubConfig = field(default_factory=GitHubConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    
    def validate(self) -> List[str]:
        errors = []
        
        if not self.github.is_app_configured() and not self.github.is_token_configured():
            errors.append("Either GitHub App or Personal Access Token must be configured")
        
        if not self.model.endpoint:
            errors.append("LIGHTNING_AI_ENDPOINT is required")
        
        return errors
    
    def log_config(self):
        logger.info("=" * 60)
        logger.info("Configuration Summary")
        logger.info("=" * 60)
        logger.info(f"GitHub Auth: {'App' if self.github.is_app_configured() else 'Token' if self.github.is_token_configured() else 'NONE'}")
        logger.info(f"Model Endpoint: {self.model.endpoint}")
        logger.info(f"Model Name: {self.model.model_name}")
        logger.info(f"Max Tokens: {self.model.max_tokens}")
        logger.info(f"Qdrant: {self.qdrant.host}:{self.qdrant.port}")
        logger.info(f"Collection: {self.qdrant.collection_name}")
        logger.info(f"RAG Enabled Commands: {self.rag.enabled_commands}")
        logger.info(f"RAG Top-K: {self.rag.top_k}")
        logger.info(f"Max Diff Size: {self.server.max_diff_size / 1024 / 1024:.1f} MB")
        logger.info(f"Request Timeout: {self.server.request_timeout}s")
        logger.info("=" * 60)


def load_config() -> Config:
    config = Config(
        github=GitHubConfig(),
        model=ModelConfig(),
        qdrant=QdrantConfig(),
        rag=RAGConfig(),
        server=ServerConfig()
    )
    return config
