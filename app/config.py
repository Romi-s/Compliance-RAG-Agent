from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    chroma_host: Optional[str] = None
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_token: Optional[str] = None
    chroma_persist_dir: str = "./chroma_data"
    collection_name: str = "compliance_docs"

    chunk_size: int = 512
    chunk_overlap: int = 64
    max_pdf_pages: int = 100

    retrieval_top_k: int = 10
    final_top_k: int = 5

    max_file_size_mb: int = 50

    api_key: Optional[str] = None

    # Public-demo cost controls
    free_queries_per_day: int = 5       # free questions per visitor (per IP), per day
    global_daily_cap: int = 300         # hard ceiling across all visitors, per day

    model_config = {"env_file": ".env", "extra": "ignore"}

    def require_api_key(self) -> str:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        return self.openai_api_key


settings = Settings()
