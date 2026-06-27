"""Step 2 of the eval: grade eval/predictions.json with ragas.

Runs in the isolated eval venv (eval/.venv) -- NOT the serving venv -- because
ragas needs an older LangChain stack. See eval/README.md.

    eval/.venv/Scripts/python.exe eval/score_ragas.py [--judge-model gpt-4o-mini]

Needs OPENAI_API_KEY (read from ../.env): ragas uses an LLM as the judge.

Metrics (answerable questions only):
  faithfulness        - answer grounded in retrieved context (anti-hallucination)
  answer_relevancy    - answer addresses the question
  context_precision   - retrieved chunks are relevant (signal vs noise)
  context_recall      - retrieval covered what the reference answer needs

The out-of-scope question is reported via the refusal flag already computed in
step 1 (a confident answer there would be a hallucination).
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).parent
load_dotenv(_HERE.parent / ".env")  # OPENAI_API_KEY for the judge

from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402

# Metric symbol names drift across ragas releases; try the long-stable lowercase
# singletons first, fall back to the 0.2 class names.
try:
    from ragas.metrics import (  # type: ignore
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
except ImportError:  # pragma: no cover - depends on installed ragas version
    from ragas.metrics import (  # type: ignore
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    METRICS = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(prog="score_ragas")
    ap.add_argument("--judge-model", default="gpt-4o-mini",
                    help="OpenAI model ragas uses to grade (default: gpt-4o-mini)")
    args = ap.parse_args()

    preds = json.loads((_HERE / "predictions.json").read_text("utf-8"))
    answerable = [p for p in preds if not p.get("out_of_scope")]
    oos = [p for p in preds if p.get("out_of_scope")]

    samples = [
        {
            "user_input": p["question"],
            "response": p["answer"],
            "retrieved_contexts": p["contexts"],
            "reference": p["reference"],
        }
        for p in answerable
    ]
    dataset = EvaluationDataset.from_list(samples)

    judge = LangchainLLMWrapper(ChatOpenAI(model=args.judge_model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    print(f"Scoring {len(samples)} answerable questions with ragas "
          f"(judge={args.judge_model})...\n")
    result = evaluate(dataset=dataset, metrics=METRICS, llm=judge, embeddings=embeddings)

    df = result.to_pandas()
    metric_cols = [c for c in df.columns
                   if c not in ("user_input", "response", "retrieved_contexts", "reference")]

    # Per-question scores, keyed back to the QA ids.
    print("Per-question scores")
    print("-" * 72)
    for p, (_, row) in zip(answerable, df.iterrows()):
        scores = "  ".join(f"{c}={row[c]:.2f}" for c in metric_cols)
        print(f"{p['id']:<26} {scores}")

    print("\nAggregate (mean across questions)")
    print("-" * 72)
    for c in metric_cols:
        print(f"{c:<22} {df[c].mean():.3f}")

    # Safety / grounding result for the out-of-scope question.
    if oos:
        print("\nOut-of-scope refusal check (no LLM judge)")
        print("-" * 72)
        for p in oos:
            verdict = "PASS (refused)" if p["refused"] else "FAIL (answered anyway!)"
            print(f"{p['id']:<26} {verdict}")

    out_csv = _HERE / "ragas_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote per-question results -> {out_csv}")


if __name__ == "__main__":
    main()
