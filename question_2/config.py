import os
from typing import Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration and environment settings for Question Generation
    and Duplicate Detection.
    """
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Vector Embedding Settings (Local Sentence-Transformers)
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Similarity Thresholds
    # Scores >= SEMANTIC_DUP_THRESHOLD are flagged as semantic duplicates
    SEMANTIC_DUP_THRESHOLD: float = 0.88
    # Scores < UNIQUE_THRESHOLD are immediately deemed unique
    UNIQUE_THRESHOLD: float = 0.65
    # Scores in [UNIQUE_THRESHOLD, SEMANTIC_DUP_THRESHOLD) escalate to Tier 4 Judge

    # Lexical threshold (Token overlap / Jaccard)
    LEXICAL_DUP_THRESHOLD: float = 0.85

    # Generation & Retrieval Parameters
    TOP_K_RETRIEVAL: int = 10
    GENERATION_TEMPERATURE: float = 0.7
    MAX_GENERATION_RETRIES: int = 3

    # LLM Provider Configuration
    # Supports "auto", "groq", "gemini", "openai", or "mock" (offline fallback)
    LLM_PROVIDER: str = "auto"
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    # LLM Model Choices
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    OPENAI_MODEL: str = "gpt-4o-mini"

    def get_active_llm_provider(self) -> str:
        """
        Determines the active LLM provider based on configuration and available API keys.
        """
        if self.LLM_PROVIDER != "auto":
            return self.LLM_PROVIDER.lower()
        if self.GROQ_API_KEY:
            return "groq"
        if self.GEMINI_API_KEY:
            return "gemini"
        if self.OPENAI_API_KEY:
            return "openai"
        return "mock"


settings = Settings()
