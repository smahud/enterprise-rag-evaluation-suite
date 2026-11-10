# Configuration management for enterprise-rag-evaluation-suite

from functools import lru_cache
from typing import Optional, List, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_RETRIEVAL_")

    embedder: str = Field(default="BAAI/bge-large-en-v1.5", description="Embedding model")
    top_k: int = Field(default=50, ge=1, le=1000, description="Top-k documents to retrieve")
    index_type: str = Field(default="hnsw", description="Index type (hnsw, flat, ivf)")
    ef_search: int = Field(default=128, description="HNSW ef_search parameter")
    batch_size: int = Field(default=32, description="Batch size for embedding")


class RerankingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_RERANKING_")

    model: str = Field(default="BAAI/bge-reranker-large", description="Reranking model")
    top_k: int = Field(default=10, ge=1, le=100, description="Top-k after reranking")
    batch_size: int = Field(default=16, description="Batch size for reranking")
    use_fp16: bool = Field(default=True, description="Use FP16 for inference")


class GenerationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_GENERATION_")

    model: str = Field(default="gpt-4o-mini", description="Generation model")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=512, ge=1, le=8192, description="Max tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    api_base: Optional[str] = Field(default=None, description="API base URL (for local models)")
    api_key: Optional[str] = Field(default=None, description="API key")


class EvaluationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_EVALUATION_")

    metrics: List[str] = Field(
        default=["recall", "ndcg", "mrr", "faithfulness", "hallucination_rate", "answer_relevance"],
        description="Metrics to compute"
    )
    sample_size: int = Field(default=1000, ge=1, description="Number of samples to evaluate")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    num_generations: int = Field(default=3, description="Generations for self-consistency")


class DatasetConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_DATASET_")

    name: str = Field(default="msmarco", description="Dataset name")
    split: str = Field(default="dev", description="Dataset split")
    cache_dir: str = Field(default="./cache/datasets", description="Cache directory")
    max_samples: Optional[int] = Field(default=None, description="Max samples to load")


class OutputConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_OUTPUT_")

    directory: str = Field(default="./results", description="Output directory")
    formats: List[str] = Field(default=["json", "html", "markdown"], description="Output formats")
    include_raw: bool = Field(default=False, description="Include raw predictions")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json, console)")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_dir: str = Field(default="./cache", description="Cache directory")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()


@lru_cache
def get_settings() -> Settings:
    return Settings()