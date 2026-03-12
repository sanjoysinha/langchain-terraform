"""
RAG Evaluation using RAGAS.

Run manually:   python evaluation/evaluate.py
Runs nightly:   via GitHub Actions scheduled workflow

Metrics measured:
  - faithfulness:       Is the answer grounded in the retrieved context?
  - answer_relevancy:   Is the answer actually relevant to the question?
  - context_precision:  Are the retrieved chunks precise and not noisy?
  - context_recall:     Does retrieved context cover what was needed?
"""
import os
import json
from datetime import datetime

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# ── Sample evaluation dataset ─────────────────────────────────────────────────
# In production: load from a curated JSONL file or Google Sheet.
# Each entry needs: question, ground_truth answer, and contexts (retrieved chunks).
EVAL_DATASET = [
    {
        "question": "What is the main topic of the document?",
        "ground_truth": "The document covers retrieval augmented generation.",
        "contexts": [
            "Retrieval Augmented Generation (RAG) is a technique that combines "
            "information retrieval with text generation to produce grounded answers.",
        ],
        "answer": "The main topic is Retrieval Augmented Generation (RAG).",
    },
    {
        "question": "How does chunking affect retrieval quality?",
        "ground_truth": "Smaller chunks improve precision but may lose context; larger chunks retain context but reduce precision.",
        "contexts": [
            "Document chunking splits text into smaller segments for embedding. "
            "Chunk size of 1000 tokens with 200 token overlap balances context retention and retrieval precision.",
        ],
        "answer": "Chunking affects retrieval by controlling how much context is embedded per segment.",
    },
]


def run_evaluation(output_file: str = "evaluation/results.json") -> dict:
    dataset = Dataset.from_list(EVAL_DATASET)

    print("Running RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    scores = {
        "timestamp": datetime.utcnow().isoformat(),
        "faithfulness": round(results["faithfulness"], 4),
        "answer_relevancy": round(results["answer_relevancy"], 4),
        "context_precision": round(results["context_precision"], 4),
        "context_recall": round(results["context_recall"], 4),
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(scores, f, indent=2)

    print("\n── Evaluation Results ──────────────────")
    for metric, value in scores.items():
        if metric != "timestamp":
            status = "✓" if value >= 0.7 else "✗ BELOW THRESHOLD"
            print(f"  {metric:<22} {value:.4f}  {status}")
    print("────────────────────────────────────────")
    print(f"Results saved to {output_file}")

    # Fail CI if any metric drops below threshold
    threshold = 0.6
    failing = [k for k, v in scores.items() if k != "timestamp" and v < threshold]
    if failing:
        raise ValueError(f"Metrics below threshold ({threshold}): {failing}")

    return scores


if __name__ == "__main__":
    run_evaluation()
