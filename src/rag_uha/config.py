"""Chargement de la configuration.

- `configs/config.yaml` : tous les réglages (modèles, chemins, paramètres).
- `.env` : les clés API uniquement, lues via `os.environ`.
- Variable d'environnement `RAG_UHA_PIPELINE` : surcharge `pipeline` sans toucher au YAML.

Usage :
    from rag_uha.config import load_settings
    settings = load_settings()                    # configs/config.yaml
    settings = load_settings("configs/autre.yaml")
    settings.llm_query.model                      # "openai/gpt-oss-120b"
    settings.llm_query.api_key                    # lu dans .env
    settings.paths.chroma_dir                     # Path absolu
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# Racine du projet = dossier qui contient pyproject.toml (deux niveaux au-dessus de ce fichier)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


class _Strict(BaseModel):
    """Toute clé inconnue dans le YAML est une erreur : évite les fautes de frappe silencieuses."""

    model_config = ConfigDict(extra="forbid")


class AppSettings(_Strict):
    name: str = "Assistant UHA"
    logo_path: Path = Path("app/assets/logo_uha.png")
    default_mode: Literal["rag", "graphrag", "both"] = "both"


class PathsSettings(_Strict):
    raw_dir: Path
    documents_csv: Path
    chunks_file: Path
    chroma_dir: Path
    lightrag_dir: Path
    runs_dir: Path
    results_dir: Path
    questions_file: Path
    demo_cache_file: Path

    def resolve(self, root: Path) -> PathsSettings:
        """Rend tous les chemins absolus par rapport à la racine du projet."""
        return PathsSettings(
            **{k: (root / v if not Path(v).is_absolute() else Path(v)) for k, v in self}
        )


class LLMSettings(_Strict):
    """Un rôle de LLM (indexing, query ou judge)."""

    provider: Literal["groq", "openrouter", "ollama", "openai"] = "groq"
    base_url: str
    model: str
    api_key_env: str | None = "GROQ_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1000
    max_concurrency: int = 1
    extra: dict[str, Any] = Field(default_factory=dict)  # ex. reasoning_effort

    @property
    def api_key(self) -> str:
        """Clé lue dans l'environnement. Ollama n'en a pas besoin : on renvoie un jeton factice."""
        if not self.api_key_env:
            return "not-needed"
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Variable {self.api_key_env} absente : copie .env.example en .env et remplis-la."
            )
        return key


class RetrySettings(_Strict):
    max_attempts: int = 8
    min_wait_s: float = 5
    max_wait_s: float = 120


class EmbeddingsSettings(_Strict):
    provider: Literal["cloudflare", "local"] = "cloudflare"
    model: str = "@cf/baai/bge-m3"
    account_id_env: str = "CF_ACCOUNT_ID"
    api_token_env: str = "CF_API_TOKEN"
    batch_size: int = 50
    dim: int = 1024

    def _env(self, name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Variable {name} absente : voir .env.example.")
        return value

    @property
    def account_id(self) -> str:
        return self._env(self.account_id_env)

    @property
    def api_token(self) -> str:
        return self._env(self.api_token_env)


class ChunkingSettings(_Strict):
    chunk_tokens: int = 1200
    overlap_tokens: int = 100
    tokenizer: str = "cl100k_base"


class RAGSettings(_Strict):
    collection: str = "uha_chunks"
    top_k: int = 4


class GraphRAGSettings(_Strict):
    mode: Literal["local", "global", "hybrid", "mix", "naive"] = "hybrid"
    top_k: int = 20
    max_total_tokens: int = 5000
    entity_extract_max_gleaning: int = 0
    force_llm_summary_on_merge: int = 6
    summary_language: str = "English"
    entity_types: list[str] = Field(default_factory=list)
    enable_llm_cache: bool = True


class GenerationSettings(_Strict):
    answer_in_question_language: bool = True
    cite_sources: bool = True
    refuse_if_no_context: bool = True


class EvalSettings(_Strict):
    k: int = 4
    ragas_metrics: list[str] = Field(default_factory=list)
    ragas_optional: list[str] = Field(default_factory=list)
    price_per_m_tokens: dict[str, tuple[float, float]] = Field(default_factory=dict)


class Settings(_Strict):
    app: AppSettings = Field(default_factory=AppSettings)
    pipeline: Literal["rag", "graphrag"] = "rag"
    paths: PathsSettings
    llm_indexing: LLMSettings
    llm_query: LLMSettings
    llm_judge: LLMSettings
    retry: RetrySettings = Field(default_factory=RetrySettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    graphrag: GraphRAGSettings = Field(default_factory=GraphRAGSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)

    config_path: Path | None = None  # d'où vient cette config (journal, metadata.json)

    def llm(self, role: Literal["indexing", "query", "judge"]) -> LLMSettings:
        return {"indexing": self.llm_indexing, "query": self.llm_query, "judge": self.llm_judge}[
            role
        ]


def load_settings(config_path: str | Path = DEFAULT_CONFIG) -> Settings:
    """Lit le YAML, charge .env, applique les surcharges d'environnement, résout les chemins."""
    load_dotenv(PROJECT_ROOT / ".env")  # sans effet si le fichier n'existe pas

    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    if override := os.environ.get("RAG_UHA_PIPELINE"):
        data["pipeline"] = override

    settings = Settings(**data, config_path=path)
    settings.paths = settings.paths.resolve(PROJECT_ROOT)
    if not settings.app.logo_path.is_absolute():
        settings.app.logo_path = PROJECT_ROOT / settings.app.logo_path
    return settings
