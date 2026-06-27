"""Step 1 of the eval: run the REAL RAG pipeline over the held-out QA set and
dump predictions to eval/predictions.json.

    # from the project root, with the serving venv active:
    python -m eval.generate_predictions

Each prediction captures everything ragas needs (question, generated answer,
retrieved contexts, reference) plus a refusal flag for out-of-scope questions.
These runs are tagged "eval" in LangSmith so they don't mix with live traffic.
"""

import json
from pathlib import Path

from app.agent.graph import qa_graph
from app.services.seed import ensure_seeded

_HERE = Path(__file__).parent
QA_SET = json.loads((_HERE / "qa_set.json").read_text("utf-8"))
OUT_PATH = _HERE / "predictions.json"

# Phrases that signal the model correctly declined to answer (used only to grade
# the out-of-scope question -- a confident answer there would be a hallucination).
REFUSAL_MARKERS = [
    "cannot", "can't", "not enough", "no information", "insufficient",
    "couldn't find", "could not find", "don't have", "do not have",
    "not available", "no relevant", "not contain", "doesn't contain",
    "do not contain", "unable", "not provided", "outside",
]


def _looks_like_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def main() -> None:
    ensure_seeded()  # make sure the demo corpus is in Chroma before we query

    predictions = []
    for q in QA_SET:
        result = qa_graph.invoke(
            {
                "question": q["question"],
                "retrieved_chunks": [],
                "answer": "",
                "citations": [],
                "error": None,
                "api_key": None,  # use the server's OPENAI_API_KEY
            },
            config={
                "run_name": "eval_qa",
                "tags": ["eval", q["category"]],
                "metadata": {"eval": True, "qa_id": q["id"]},
            },
        )

        if result.get("error"):
            raise RuntimeError(f"Pipeline error on {q['id']!r}: {result['error']}")

        answer = result["answer"]
        contexts = [c["text"] for c in result["retrieved_chunks"]]

        predictions.append(
            {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "answer": answer,
                "contexts": contexts,
                "reference": q["reference"],
                "out_of_scope": q.get("out_of_scope", False),
                "refused": _looks_like_refusal(answer),
                "retrieval_ms": result.get("retrieval_ms"),
                "generation_ms": result.get("generation_ms"),
            }
        )
        tag = "OOS" if q.get("out_of_scope") else "   "
        print(f"[{tag}] {q['id']:<26} {len(contexts)} chunks, "
              f"{result.get('generation_ms', 0):.0f} ms gen")

    OUT_PATH.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nWrote {len(predictions)} predictions -> {OUT_PATH}")

    # Quick safety check on the out-of-scope question (no LLM judge needed).
    for p in predictions:
        if p["out_of_scope"]:
            verdict = "PASS (refused)" if p["refused"] else "FAIL (answered anyway!)"
            print(f"Out-of-scope grounding check [{p['id']}]: {verdict}")


if __name__ == "__main__":
    main()
