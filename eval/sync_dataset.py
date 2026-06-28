"""push eval/qa_set.json up to a LangSmith Dataset.

    # from the project root, with the SERVING venv active:
    python -m eval.sync_dataset                 # default name "compliance-rag-eval"
    python -m eval.sync_dataset --name my-set   # custom dataset name

Idempotent. Each LangSmith example is keyed by our own ``qa_id`` (stored in the
example metadata), so re-running this:
  - creates pairs that are new,
  - updates pairs whose question / reference / flags changed,
  - deletes LangSmith examples whose qa_id is no longer in qa_set.json.

"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).parent
load_dotenv(_HERE.parent / ".env")  # LANGSMITH_API_KEY / LANGSMITH_ENDPOINT

from langsmith import Client  # noqa: E402

DEFAULT_DATASET = "compliance-rag-eval"

# A small, balanced subset (one per category + a hard-recall pair + a refusal)
# used as the fast PR smoke gate. Marked in each example's metadata so the eval
# harness can select it deterministically -- list_examples(limit=N) does NOT
# preserve qa_set.json order, so "first N" is neither stable nor representative.
SMOKE_IDS = {
    "def_personal_data",        # definitions
    "right_of_access",          # data-subject-rights
    "erasure_exceptions",       # right-to-erasure
    "dpo_required",             # governance
    "fine_tiers",               # penalties
    "breach_notification_window",  # breach-notification
    "fine_higher_tier_scope",   # hard-recall (specific article list)
    "ccpa_out_of_scope",        # out-of-scope / refusal
}


def _payload(q: dict) -> tuple[dict, dict, dict]:
    """Split one qa_set entry into (inputs, outputs, metadata) for LangSmith."""
    inputs = {"question": q["question"]}
    outputs = {"reference": q["reference"]}
    metadata = {
        "qa_id": q["id"],
        "category": q["category"],
        "out_of_scope": bool(q.get("out_of_scope", False)),
        "smoke": q["id"] in SMOKE_IDS,
    }
    return inputs, outputs, metadata


def _differs(example, inputs: dict, outputs: dict, metadata: dict) -> bool:
    """True if the live LangSmith example no longer matches the local pair."""
    ex_meta = example.metadata or {}
    return (
        (example.inputs or {}) != inputs
        or (example.outputs or {}) != outputs
        or ex_meta.get("category") != metadata["category"]
        or bool(ex_meta.get("out_of_scope")) != metadata["out_of_scope"]
        or bool(ex_meta.get("smoke")) != metadata["smoke"]
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="sync_dataset")
    ap.add_argument("--name", default=DEFAULT_DATASET, help="LangSmith dataset name")
    args = ap.parse_args()

    qa_set = json.loads((_HERE / "qa_set.json").read_text("utf-8"))
    client = Client()

    if client.has_dataset(dataset_name=args.name):
        dataset = client.read_dataset(dataset_name=args.name)
        print(f"Using existing dataset: {args.name}")
    else:
        dataset = client.create_dataset(
            args.name,
            description="Held-out GDPR Q&A for the compliance-rag-agent eval suite.",
        )
        print(f"Created dataset: {args.name}")

    # Map LangSmith examples by our stable qa_id so we can update in place.
    existing = {}
    for ex in client.list_examples(dataset_id=dataset.id):
        qa_id = (ex.metadata or {}).get("qa_id")
        if qa_id:
            existing[qa_id] = ex

    created = updated = unchanged = 0
    seen = set()

    for q in qa_set:
        qa_id = q["id"]
        seen.add(qa_id)
        inputs, outputs, metadata = _payload(q)
        ex = existing.get(qa_id)

        if ex is None:
            client.create_example(
                inputs=inputs, outputs=outputs, metadata=metadata, dataset_id=dataset.id
            )
            created += 1
        elif _differs(ex, inputs, outputs, metadata):
            client.update_example(
                ex.id, inputs=inputs, outputs=outputs, metadata=metadata
            )
            updated += 1
        else:
            unchanged += 1

    # Drop LangSmith examples that no longer exist locally.
    deleted = 0
    for qa_id, ex in existing.items():
        if qa_id not in seen:
            client.delete_example(ex.id)
            deleted += 1

    print(
        f"\nSynced '{args.name}': "
        f"{created} created, {updated} updated, {unchanged} unchanged, {deleted} deleted "
        f"({len(qa_set)} pairs total)."
    )
    url = getattr(dataset, "url", None)
    if url:
        print(f"Dashboard: {url}")


if __name__ == "__main__":
    main()
