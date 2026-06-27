import re
import time

from app.agent.state import Citation, QAState
from app.config import settings
from app.services.retriever import hybrid_retrieve


def validate_input(state: QAState) -> dict:
    if not state["question"].strip():
        return {"error": "Question must not be empty"}
    if len(state["question"]) > 2000:
        return {"error": "Question exceeds 2000 character limit"}
    return {}


def retrieve(state: QAState) -> dict:
    if state.get("error"):
        return {}
    try:
        t0 = time.perf_counter()
        chunks = hybrid_retrieve(state["question"], api_key=state.get("api_key"))
        retrieval_ms = (time.perf_counter() - t0) * 1000
        if not chunks:
            return {"error": "No relevant documents found. Please ingest documents first."}
        return {"retrieved_chunks": chunks, "retrieval_ms": retrieval_ms}
    except Exception as exc:
        return {"error": f"Retrieval failed: {exc}"}


def generate(state: QAState) -> dict:
    if state.get("error"):
        return {}
    try:
        from langsmith.wrappers import wrap_openai
        from openai import OpenAI

        # wrap_openai adds a child LLM span (model, token usage, cost, latency) to
        # the active trace. It's a no-op when LANGSMITH_TRACING isn't enabled.
        client = wrap_openai(
            OpenAI(api_key=state.get("api_key") or settings.require_api_key())
        )

        context_parts = []
        for i, chunk in enumerate(state["retrieved_chunks"], 1):
            context_parts.append(
                f"[{i}] (Source: {chunk['source']}, Page {chunk['page']})\n"
                f"{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

        system_prompt = (
            "You are a compliance expert assistant. Answer questions based ONLY on the "
            "provided document excerpts. Cite your sources using [1], [2], etc. to "
            "reference the excerpt numbers. If the provided excerpts don't contain "
            "enough information to answer the question, say so explicitly."
        )
        user_prompt = f"Document excerpts:\n\n{context}\n\nQuestion: {state['question']}"

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
        )
        generation_ms = (time.perf_counter() - t0) * 1000
        return {"answer": response.choices[0].message.content, "generation_ms": generation_ms}
    except Exception as exc:
        return {"error": f"Generation failed: {exc}"}


def format_response(state: QAState) -> dict:
    if state.get("error"):
        return {}

    cited_indices = {
        int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", state["answer"])
    }

    citations = []
    for i, chunk in enumerate(state["retrieved_chunks"], 1):
        if i in cited_indices:
            citations.append(
                Citation(
                    source=chunk["source"],
                    page=chunk["page"],
                    chunk_index=chunk["chunk_index"],
                    text=chunk["text"][:200],
                )
            )

    return {"answer": state["answer"].strip(), "citations": citations}
