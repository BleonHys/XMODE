from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import json
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency, enforced via requirements
    load_dotenv = None  # type: ignore


_BASE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_ENV_PATH = _BASE_DIR / ".env"
_DEFAULT_CONFIG_PATH = _BASE_DIR / "config" / "defaults.json"


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _ensure_env_loaded(env_path: Path) -> None:
    if load_dotenv is None:
        raise RuntimeError(
            "python-dotenv is not installed. Add it to requirements.txt or install it to load env files."
        )
    # Do not override already exported values by default.
    load_dotenv(dotenv_path=env_path, override=False)


@dataclass
class LangChainConfig:
    project: str = "XMODE"
    tracing_v2: bool = False


@dataclass
class ModelConfig:
    provider: str = "openai"
    default_chat_model: str = "gpt-4o"
    default_temperature: float = 0.0
    anthropic_model: str = "claude-3-5-sonnet-20241022"


@dataclass
class ArtworkConfig:
    base_dir: Path
    language: str
    db_path: Path
    questions_file: Path
    output_dir_template: str
    log_dir: Path
    thread_id: str
    langchain_project: str
    vqa_provider: str = "ollama"
    ollama_endpoint: str = "http://localhost:11434/api/generate"
    ollama_model: str = "qwen3-vl:4b-instruct"
    openrouter_endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_model: str = "qwen/qwen3-vl-8b-instruct"
    openrouter_timeout: int = 120
    openrouter_max_tokens: int = 512

    def __post_init__(self) -> None:
        self.db_path = _resolve_path(self.base_dir, str(self.db_path))
        self.questions_file = _resolve_path(self.base_dir, str(self.questions_file))
        self.log_dir = _resolve_path(self.base_dir, str(self.log_dir))

    @property
    def output_dir(self) -> Path:
        return _resolve_path(self.base_dir, self.output_dir_template.format(language=self.language))


@dataclass
class EHRXQAConfig:
    base_dir: Path
    language: str
    db_path: Path
    dataset_file: Path
    output_dir_template: str
    thread_id: str
    m3ae_endpoint: str
    langchain_project: str

    def __post_init__(self) -> None:
        self.db_path = _resolve_path(self.base_dir, str(self.db_path))
        self.dataset_file = _resolve_path(self.base_dir, str(self.dataset_file))

    @property
    def output_dir(self) -> Path:
        return _resolve_path(self.base_dir, self.output_dir_template.format(language=self.language))

    @property
    def dataset_dir(self) -> Path:
        return self.dataset_file.parent


@dataclass
class Settings:
    base_dir: Path
    langchain: LangChainConfig
    models: ModelConfig
    artwork: ArtworkConfig
    ehrxqa: EHRXQAConfig
    config_path: Path
    env_path: Path
    downloads: Dict[str, Any]

    def require_env(self, *variables: str) -> None:
        missing = [var for var in variables if not os.environ.get(var)]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                f"Missing required environment variables: {joined}. Add them to {self.env_path} or export them."
            )

    @property
    def openai_api_key(self) -> str:
        value = os.environ.get("OPENAI_API_KEY", "")
        if not value:
            raise RuntimeError("OPENAI_API_KEY not set. Add it to .env or export it before running XMODE.")
        return value

    @property
    def anthropic_api_key(self) -> str:
        value = os.environ.get("ANTHROPIC_API_KEY", "")
        if not value:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to .env or export it before running XMODE.")
        return value

    @property
    def openrouter_api_key(self) -> str:
        value = os.environ.get("OPENROUTER_API_KEY", "")
        if not value:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Add it to .env or export it before running the OpenRouter VQA backend."
            )
        return value

    @property
    def openrouter_site_url(self) -> Optional[str]:
        return os.environ.get("OPENROUTER_SITE_URL")

    @property
    def openrouter_site_name(self) -> Optional[str]:
        return os.environ.get("OPENROUTER_SITE_NAME")

    @property
    def langchain_api_key(self) -> Optional[str]:
        return os.environ.get("LANGCHAIN_API_KEY")

    @property
    def files_dir(self) -> Path:
        return self.base_dir / "files"

    @property
    def langsmith_project(self) -> Optional[str]:
        # Support both env vars; prefer explicit config, then env overrides
        return os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT")


def _load_raw_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    config_path = Path(os.environ.get("XMODE_CONFIG_PATH", _DEFAULT_CONFIG_PATH))
    env_path = Path(os.environ.get("XMODE_ENV_PATH", _DEFAULT_ENV_PATH))

    _ensure_env_loaded(env_path)

    raw = _load_raw_config(config_path)

    langchain_raw = raw.get("langchain", {})
    models_raw = raw.get("models", {})
    artwork_raw = raw.get("artwork", {})
    ehrxqa_raw = raw.get("ehrxqa", {})

    langchain_cfg = LangChainConfig(**langchain_raw)
    models_cfg = ModelConfig(**models_raw)
    artwork_cfg = ArtworkConfig(base_dir=_BASE_DIR, **artwork_raw)
    ehrxqa_cfg = EHRXQAConfig(base_dir=_BASE_DIR, **ehrxqa_raw)

    return Settings(
        base_dir=_BASE_DIR,
        langchain=langchain_cfg,
        models=models_cfg,
        artwork=artwork_cfg,
        ehrxqa=ehrxqa_cfg,
        config_path=config_path,
        env_path=env_path,
        downloads=raw.get("downloads", {}),
    )


__all__ = ["Settings", "get_settings", "LangChainConfig", "ModelConfig", "ArtworkConfig", "EHRXQAConfig"]
