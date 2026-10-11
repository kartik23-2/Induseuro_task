import os
from pydantic import BaseModel, Field
from typing import Optional


class Settings(BaseModel):
    # Vector Embedding Settings (Local Sentence-Transformers)
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Similarity Thresholds
    # Scores >= EXACT_DUP_SIM_THRESHOLD are flagged as semantic duplicates
    SEMANTIC_DUP_THRESHOLD: float = 0.85
    # Scores < UNIQUE_THRESHOLD are immediately deemed unique
    UNIQUE_THRESHOLD: float = 0.65
    # Scores in [UNIQUE_THRESHOLD, SEMANTIC_DUP_THRESHOLD) escalate to Tier 4 Judge
    
    # Lexical threshold (Token overlap / Jaccard)
    LEXICAL_DUP_THRESHOLD: float = 0.85
    
    # LLM Provider Configuration
    # Supports "groq", "gemini", "openai", or "mock" (offline fallback)
    LLM_PROVIDER: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    GROQ_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    GEMINI_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    OPENAI_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    
    # LLM Model Choices
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    OPENAI_MODEL: str = "gpt-4o-mini"


settings = Settings()
