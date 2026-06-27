from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env into os.environ *before* anything reads it. pydantic-settings below
# fills the Settings model from .env, but it does NOT export to os.environ -- and
# the LangSmith SDK reads os.environ["LANGSMITH_*"] directly. Without this,
# tracing is silently off locally (it still works on Cloud Run, where the vars
# are real environment variables). override=False so real env vars always win.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
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
    free_uploads_per_day: int = 3       # free document uploads per visitor (per IP), per day
    global_daily_cap: int = 300         # hard ceiling across all visitors, per day
    max_upload_mb: int = 5              # size cap for public demo uploads (owner uses max_file_size_mb)
    max_demo_pdf_pages: int = 30        # page cap for public demo PDF uploads

    model_config = {"env_file": ".env", "extra": "ignore"}

    def require_api_key(self) -> str:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        return self.openai_api_key


settings = Settings()
