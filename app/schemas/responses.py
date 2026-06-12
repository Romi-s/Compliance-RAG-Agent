from typing import List, Optional

from pydantic import BaseModel


class CitationResponse(BaseModel):
    source: str
    page: int
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationResponse]
    model: str
    chunks_used: int
    free_remaining: Optional[int] = None
    used_own_key: bool = False


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    message: str


class CollectionStatsResponse(BaseModel):
    total_chunks: int
    sources: List[str]
