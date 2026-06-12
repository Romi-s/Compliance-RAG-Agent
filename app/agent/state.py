from typing import List, Optional, TypedDict


class RetrievedChunk(TypedDict):
    text: str
    source: str
    page: int
    chunk_index: int
    score: float


class Citation(TypedDict):
    source: str
    page: int
    chunk_index: int
    text: str


class QAState(TypedDict):
    question: str
    retrieved_chunks: List[RetrievedChunk]
    answer: str
    citations: List[Citation]
    error: Optional[str]
    # Per-request OpenAI key (bring-your-own-key). None -> use the server's key.
    api_key: Optional[str]
