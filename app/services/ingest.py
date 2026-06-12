import hashlib
from typing import List, Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from app.config import settings
from app.services.text_extractor import extract_text_from_pdf


def get_chroma_client() -> chromadb.ClientAPI:
    if settings.chroma_host:
        kwargs = {
            "host": settings.chroma_host,
            "port": settings.chroma_port,
            "ssl": settings.chroma_ssl,
        }
        if settings.chroma_token:
            kwargs["headers"] = {"Authorization": f"Bearer {settings.chroma_token}"}
        return chromadb.HttpClient(**kwargs)
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection() -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: List[str], api_key: Optional[str] = None) -> List[List[float]]:
    client = OpenAI(api_key=api_key or settings.require_api_key())
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    api_key: Optional[str] = None,
    max_pages: Optional[int] = None,
) -> dict:
    pages = extract_text_from_pdf(pdf_bytes, max_pages=max_pages or settings.max_pdf_pages)
    return _ingest_pages(pages, filename, api_key=api_key)


def ingest_text(text: str, filename: str, api_key: Optional[str] = None) -> dict:
    return _ingest_pages([(1, text)], filename, api_key=api_key)


def _ingest_pages(pages: list, filename: str, api_key: Optional[str] = None) -> dict:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for page_num, text in pages:
        page_chunks = splitter.split_text(text)
        for i, chunk in enumerate(page_chunks):
            chunk_id = hashlib.sha256(
                f"{filename}:{page_num}:{i}:{chunk[:50]}".encode()
            ).hexdigest()[:16]
            chunks.append(chunk)
            metadatas.append({
                "source": filename,
                "page": page_num,
                "chunk_index": i,
            })
            ids.append(chunk_id)

    if not chunks:
        return {"filename": filename, "chunks_added": 0}

    embeddings = embed_texts(chunks, api_key=api_key)
    collection = get_collection()

    batch_size = 5000
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=chunks[start:end],
            metadatas=metadatas[start:end],
        )

    return {"filename": filename, "chunks_added": len(chunks)}
