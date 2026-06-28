"""
    python -m eval.check_thresholds

Floors are deliberately a bit below observed scores: they catch real regressions
without flapping on normal LLM-judge noise. correct_refusal is held at 1.0 --
answering an out-of-scope question is a hallucination and should always block.
"""

import json
import sys
from pathlib import Path

SUMMARY_PATH = Path(__file__).parent / "langsmith_summary.json"

FLOORS = {
    "faithfulness": 0.80,
    "answer_relevance": 0.80,
    "context_precision": 0.70,
    "correct_refusal": 1.00,
}


def main() -> None:
    if not SUMMARY_PATH.exists():
        sys.exit(f"{SUMMARY_PATH.name} not found -- run `python -m eval.langsmith_eval` first.")

    data = json.loads(SUMMARY_PATH.read_text("utf-8"))
    metrics = data.get("metrics", {})
    counts = data.get("metric_counts", {})
    scope = "smoke" if data.get("smoke") else "full"
    print(f"Threshold gate ({scope}, {data.get('n_examples', '?')} examples, "
          f"experiment {data.get('experiment')})")
    print("-" * 60)

    failures = []
    for metric, floor in FLOORS.items():
        value = metrics.get(metric)
        if value is None:
            print(f"  {metric:<18} (not measured -- skipped)")
            continue
        ok = value >= floor
        n = counts.get(metric, "?")
        print(f"  {metric:<18} {value:.3f} >= {floor:.2f}  n={n}  {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{metric}={value:.3f} < {floor:.2f}")

    print("-" * 60)
    if failures:
        sys.exit("GATE FAILED: " + "; ".join(failures))
    print("GATE PASSED")


if __name__ == "__main__":
    main()
