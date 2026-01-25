"""Application configuration."""
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Hugging Face API
    huggingface_api_key: str = Field(
        default="",
        description="Hugging Face API key (get from https://huggingface.co/settings/tokens)"
    )
    
    # Data directories
    data_dir: Path = Path("./data")
    repos_dir: Path = Path("./data/repos")
    chroma_db_dir: Path = Path("./data/chroma_db")
    metadata_db_path: Path = Path("./data/metadata.db")
    
    # Embedding model
    embedding_model: str = "all-mpnet-base-v2"
    
    # LLM models (Hugging Face model IDs)
    # Note: Router API may require specific model formats
    # Try: meta-llama/Llama-3.2-3B-Instruct, google/gemma-2-2b-it, or mistralai/Mistral-7B-Instruct-v0.1
    planner_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    synthesizer_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    verifier_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Agent settings
    max_retrieval_iterations: int = 3
    max_chunks_per_query: int = 12
    chunk_size: int = 1200
    chunk_overlap: int = 200
    
    # Context window optimization
    context_window_size: int = 8192  # Maximum context window size in tokens
    reserve_prompt_tokens: int = 2000  # Tokens reserved for prompt template
    reserve_response_tokens: int = 1000  # Tokens reserved for LLM response
    
    # Multi-query retrieval
    query_variations: int = 3  # Number of query variations per base query (3-5 recommended)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create data directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate required settings
        if not self.huggingface_api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY is required. "
                "Please create a .env file with your Hugging Face API key.\n"
                "Get your API key from: https://huggingface.co/settings/tokens\n"
                "Then add to .env: HUGGINGFACE_API_KEY=hf_your_token_here"
            )


# Global settings instance
try:
    settings = Settings()
except ValueError as e:
    print(f"\n❌ Configuration Error: {e}\n")
    raise
