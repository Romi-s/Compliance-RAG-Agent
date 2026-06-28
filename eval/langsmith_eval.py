"""run the RAG pipeline over the LangSmith dataset and grade it with
custom LLM-as-judge evaluators, recording an experiment on the LangSmith dashboard.

    # from the project root, with the SERVING venv active:
    python -m eval.langsmith_eval                  # full dataset
    python -m eval.langsmith_eval --smoke          # curated smoke subset (fast PR gate)
    python -m eval.langsmith_eval --judge-model gpt-4o

These evaluators are plain OpenAI calls, so they share the app's modern LangChain stack with no
conflict.

Evaluators (LLM-as-judge, score 0.0-1.0):
  faithfulness        - every claim in the answer is supported by retrieved context
  answer_relevance    - the answer actually addresses the question
  context_precision   - retrieved context is relevant/sufficient (answerable only)
  correct_refusal     - out-of-scope questions are declined, not fabricated (OOS only)

Needs OPENAI_API_KEY + LANGSMITH_API_KEY (+ LANGSMITH_ENDPOINT for the EU account),
all read from ../.env via app.config.
"""

import argparse
import json
from pathlib import Path

from app.config import settings  # imports load .env (OPENAI_/LANGSMITH_ env vars)

from langsmith import Client, evaluate
from openai import OpenAI

from app.agent.graph import qa_graph
from app.services.seed import ensure_seeded

_HERE = Path(__file__).parent
DEFAULT_DATASET = "compliance-rag-eval"
SUMMARY_PATH = _HERE / "langsmith_summary.json"

_judge_model = "gpt-4o-mini"
_judge = OpenAI(api_key=settings.openai_api_key)

# Returned by an evaluator that doesn't apply to a given example. langsmith
# rejects None / [] / {} (any falsy result), but a non-empty dict carrying an
# empty results list is accepted and records no feedback.
_SKIP = {"results": []}


# --------------------------------------------------------------------------- #
# Target: the system under test                                               #
# --------------------------------------------------------------------------- #
def run_qa(inputs: dict) -> dict:
    """Invoke the real RAG graph for one dataset example."""
    result = qa_graph.invoke(
        {
            "question": inputs["question"],
            "retrieved_chunks": [],
            "answer": "",
            "citations": [],
            "error": None,
            "api_key": None,  # use the server's OPENAI_API_KEY
        },
        config={
            "run_name": "eval_qa",
            "tags": ["eval", "langsmith-eval"],
            "metadata": {"eval": True},
        },
    )
    return {
        "answer": result.get("answer", ""),
        "contexts": [c["text"] for c in result.get("retrieved_chunks", [])],
        "error": result.get("error"),
        "retrieval_ms": result.get("retrieval_ms"),
        "generation_ms": result.get("generation_ms"),
    }


# --------------------------------------------------------------------------- #
# LLM-as-judge plumbing                                                        #
# --------------------------------------------------------------------------- #
def _judge_score(system: str, user: str) -> dict:
    """Ask the judge model for a JSON {score, reasoning} and parse it defensively."""
    resp = _judge.chat.completions.create(
        model=_judge_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        score = float(data.get("score"))
    except (json.JSONDecodeError, TypeError, ValueError):
        score = 0.0
        data = {"reasoning": f"unparseable judge output: {raw[:200]}"}
    score = max(0.0, min(1.0, score))  # clamp to [0, 1]
    return {"score": score, "comment": str(data.get("reasoning", ""))[:500]}


def _is_oos(example) -> bool:
    return bool((example.metadata or {}).get("out_of_scope"))


# --------------------------------------------------------------------------- #
# Evaluators  (stable (run, example) signature)                               #
# --------------------------------------------------------------------------- #
def faithfulness(run, example) -> dict:
    out = run.outputs or {}
    context = "\n\n".join(out.get("contexts", [])) or "(no context retrieved)"
    res = _judge_score(
        "You grade FAITHFULNESS: is every factual claim in the ANSWER supported by "
        "the CONTEXT? Respond as JSON {\"score\": <0.0-1.0>, \"reasoning\": <short>}. "
        "1.0 = fully grounded; 0.0 = central claims unsupported or contradicted. "
        "If the answer declines because the context lacks the information, that is "
        "faithful (high score).",
        f"CONTEXT:\n{context}\n\nANSWER:\n{out.get('answer', '')}",
    )
    return {"key": "faithfulness", **res}


def answer_relevance(run, example) -> dict:
    out = run.outputs or {}
    res = _judge_score(
        "You grade ANSWER RELEVANCE: does the ANSWER address the QUESTION? Respond as "
        "JSON {\"score\": <0.0-1.0>, \"reasoning\": <short>}. Judge relevance, not "
        "factual correctness. A clear, correct refusal to an unanswerable question is "
        "relevant (high score); an evasive or off-topic answer is low.",
        f"QUESTION:\n{example.inputs.get('question', '')}\n\nANSWER:\n{out.get('answer', '')}",
    )
    return {"key": "answer_relevance", **res}


def context_precision(run, example):
    # Not meaningful for out-of-scope questions (the answer is correctly absent
    # from the corpus, so retrieving "irrelevant" context is expected). Returning
    # an empty results set records no feedback (langsmith rejects None / []).
    if _is_oos(example):
        return _SKIP
    out = run.outputs or {}
    context = "\n\n".join(out.get("contexts", [])) or "(no context retrieved)"
    res = _judge_score(
        "You grade CONTEXT PRECISION: is the retrieved CONTEXT relevant and sufficient "
        "to answer the QUESTION, using the REFERENCE answer as the gold target of what "
        "information is needed? Respond as JSON {\"score\": <0.0-1.0>, \"reasoning\": "
        "<short>}. 1.0 = context clearly contains the needed facts; 0.0 = irrelevant or "
        "missing the key information.",
        f"QUESTION:\n{example.inputs.get('question', '')}\n\n"
        f"REFERENCE:\n{example.outputs.get('reference', '')}\n\nCONTEXT:\n{context}",
    )
    return {"key": "context_precision", **res}


# Phrases that signal a correct decline on an out-of-scope question.
_REFUSAL_MARKERS = (
    "cannot", "can't", "not enough", "no information", "insufficient",
    "couldn't find", "could not find", "don't have", "do not have",
    "not available", "no relevant", "not contain", "doesn't contain",
    "do not contain", "unable", "not provided", "outside", "only cover",
    "only covers", "not mention",
)


def correct_refusal(run, example):
    # Only applies to out-of-scope questions: a confident answer there is a
    # hallucination. Cheap deterministic check, no judge call needed. Records no
    # feedback for answerable examples.
    if not _is_oos(example):
        return _SKIP
    answer = (run.outputs or {}).get("answer", "").lower()
    refused = any(m in answer for m in _REFUSAL_MARKERS)
    return {
        "key": "correct_refusal",
        "score": 1.0 if refused else 0.0,
        "comment": "declined as expected" if refused else "answered an out-of-scope question",
    }


EVALUATORS = [faithfulness, answer_relevance, context_precision, correct_refusal]


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def _write_summary(results, dataset_name: str, smoke: bool) -> dict:
    """Aggregate per-metric means into eval/langsmith_summary.json for the CI gate.

    Iterates the results directly so we don't pull in pandas (keeps the serving
    venv and CI image lean).
    """
    from collections import defaultdict

    by_key: dict[str, list[float]] = defaultdict(list)
    n_examples = 0
    for row in results:
        n_examples += 1
        for r in (row.get("evaluation_results") or {}).get("results", []):
            if r.score is not None:
                by_key[r.key].append(float(r.score))

    means = {k: round(sum(v) / len(v), 4) for k, v in by_key.items() if v}
    counts = {k: len(v) for k, v in by_key.items() if v}

    summary = {
        "dataset": dataset_name,
        "experiment": getattr(results, "experiment_name", None),
        "n_examples": n_examples,
        "smoke": smoke,
        "metrics": means,
        "metric_counts": counts,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), "utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="langsmith_eval")
    ap.add_argument("--name", default=DEFAULT_DATASET, help="LangSmith dataset name")
    ap.add_argument("--smoke", action="store_true",
                    help="grade only the curated smoke subset (fast PR gate)")
    ap.add_argument("--judge-model", default="gpt-4o-mini",
                    help="OpenAI model used as the judge (default: gpt-4o-mini)")
    ap.add_argument("--max-concurrency", type=int, default=4)
    args = ap.parse_args()

    global _judge_model
    _judge_model = args.judge_model

    ensure_seeded()  # corpus must be in Chroma before we query
    client = Client()

    if not client.has_dataset(dataset_name=args.name):
        raise SystemExit(
            f"Dataset '{args.name}' not found. Run `python -m eval.sync_dataset` first."
        )

    # Full dataset by name, or the curated smoke subset (selected by metadata, so
    # it's deterministic and balanced -- list order from the API is neither).
    if args.smoke:
        data = [ex for ex in client.list_examples(dataset_name=args.name)
                if (ex.metadata or {}).get("smoke")]
        if not data:
            raise SystemExit(
                "No examples marked smoke=true. Run `python -m eval.sync_dataset` "
                "to (re)write the smoke metadata."
            )
        prefix = "compliance-rag-smoke"
    else:
        data = args.name
        prefix = "compliance-rag-full"

    results = evaluate(
        run_qa,
        data=data,
        evaluators=EVALUATORS,
        experiment_prefix=prefix,
        max_concurrency=args.max_concurrency,
        metadata={"judge_model": args.judge_model, "smoke": bool(args.smoke)},
    )

    summary = _write_summary(results, args.name, args.smoke)
    print("\nAggregate scores")
    print("-" * 48)
    for key, val in summary["metrics"].items():
        print(f"  {key:<20} {val:.3f}")
    print(f"\n{summary['n_examples']} examples graded -> {SUMMARY_PATH.name}")
    if summary["experiment"]:
        print(f"Experiment: {summary['experiment']} (open it in LangSmith)")


if __name__ == "__main__":
    main()
