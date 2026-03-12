"""
eval_rag.py — RAGAS-based evaluation of the RAG chain (chain.py).

Metrics evaluated:
  - Faithfulness         : Answer is grounded in retrieved context (no hallucination)
  - Response Relevancy   : Answer addresses the question asked
  - Context Precision    : Retrieved chunks ranked by relevance (reference-based)
  - Context Recall       : Retrieved chunks contain enough info to answer

Usage:
  python eval_rag.py                         # full eval (requires OPENAI_API_KEY)
  python eval_rag.py --dry-run               # validate dataset structure, no LLM calls
  python eval_rag.py --output results/r.json # custom output path
  python eval_rag.py --model gpt-4o          # model for RAG chain and evaluation

Results are saved to results/rag_eval.json by default.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_api_key() -> bool:
    """Return True if OPENAI_API_KEY is set and non-empty."""
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key and key.strip())


def build_eval_vectorstore():
    """
    Build a fresh in-memory FAISS vectorstore from synthetic Acme documents.
    Never touches the production vectorstore/ directory.
    """
    from eval_dataset import build_synthetic_vectorstore
    print("  Embedding synthetic documents with text-embedding-3-small ...")
    return build_synthetic_vectorstore(persist_dir=None)


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------

def collect_rag_sample(
    vectorstore,
    question: str,
    ground_truth: str,
    model_name: str = "gpt-4o",
) -> dict:
    """
    Run the RAG chain for one question and return a RAGAS-compatible sample dict.

    The chain is built fresh per call so the vectorstore is always the synthetic one.
    chain.py's build_rag_chain() returns a create_retrieval_chain whose output
    dict contains both 'answer' and 'source_documents'.

    Returns:
        {
            "user_input": str,
            "retrieved_contexts": list[str],   # page_content of retrieved chunks
            "response": str,                   # LLM answer
            "reference": str,                  # ground truth for metric scoring
        }
    """
    from chain import build_rag_chain

    rag_chain = build_rag_chain(vectorstore, model_name=model_name)
    result = rag_chain.invoke({"input": question, "chat_history": []})

    answer = result.get("answer", "")
    source_docs = result.get("source_documents", [])
    contexts = [doc.page_content for doc in source_docs]

    return {
        "user_input": question,
        "retrieved_contexts": contexts,
        "response": answer,
        "reference": ground_truth,
    }


def build_evaluation_dataset(
    vectorstore,
    qa_pairs: list[dict],
    model_name: str = "gpt-4o",
    verbose: bool = True,
) -> list[dict]:
    """
    Collect RAG outputs for all QA pairs.
    Returns a list of RAGAS sample dicts.
    """
    samples = []
    n = len(qa_pairs)
    for i, qa in enumerate(qa_pairs):
        if verbose:
            short_q = qa["question"][:65] + ("..." if len(qa["question"]) > 65 else "")
            print(f"  [{i+1}/{n}] {short_q}")
        sample = collect_rag_sample(
            vectorstore,
            question=qa["question"],
            ground_truth=qa["ground_truth"],
            model_name=model_name,
        )
        samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    samples: list[dict],
    eval_model_name: str = "gpt-4o",
    show_progress: bool = True,
) -> dict:
    """
    Run RAGAS evaluate() on collected RAG samples.

    Uses ragas>=0.2.0 API:
      - EvaluationDataset.from_list() with fields:
          user_input, retrieved_contexts, response, reference
      - LangchainLLMWrapper to wrap ChatOpenAI
      - Metrics: Faithfulness, ResponseRelevancy,
                 LLMContextPrecisionWithReference, LLMContextRecall

    Returns:
        {
            "scores": {metric_name: float},
            "per_sample": list[dict],
            "metadata": {timestamp, model, num_samples},
        }
    """
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
    except ImportError as e:
        print(f"ERROR: RAGAS not installed. Run: pip install ragas>=0.2.0\n  {e}")
        sys.exit(1)

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=eval_model_name, temperature=0)
    evaluator_llm = LangchainLLMWrapper(llm)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]

    evaluation_dataset = EvaluationDataset.from_list(samples)

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        raise_exceptions=False,
        show_progress=show_progress,
    )

    # Extract aggregate scores — result behaves like a dict
    metric_keys = {
        "faithfulness": "faithfulness",
        "response_relevancy": "answer_relevancy",    # RAGAS key alias
        "context_precision": "context_precision",
        "context_recall": "context_recall",
    }

    scores = {}
    for our_key, ragas_key in metric_keys.items():
        try:
            val = result[ragas_key]
            scores[our_key] = round(float(val), 4) if val is not None else None
        except (KeyError, TypeError):
            # Try our key as fallback
            try:
                val = result[our_key]
                scores[our_key] = round(float(val), 4) if val is not None else None
            except (KeyError, TypeError):
                scores[our_key] = None

    # Per-sample breakdown
    try:
        per_sample = result.to_pandas().to_dict(orient="records")
    except Exception:
        per_sample = []

    return {
        "scores": scores,
        "per_sample": per_sample,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "eval_model": eval_model_name,
            "num_samples": len(samples),
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

METRIC_THRESHOLDS = {
    "faithfulness": 0.7,
    "response_relevancy": 0.7,
    "context_precision": 0.7,
    "context_recall": 0.7,
}

METRIC_LABELS = {
    "faithfulness": "Faithfulness        ",
    "response_relevancy": "Response Relevancy  ",
    "context_precision": "Context Precision   ",
    "context_recall": "Context Recall      ",
}


def print_results(results: dict) -> None:
    """Pretty-print RAGAS evaluation results to stdout."""
    print("\n" + "=" * 62)
    print("  RAGAS RAG EVALUATION RESULTS")
    print("=" * 62)

    scores = results["scores"]
    valid_scores = [v for v in scores.values() if v is not None]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    for key, label in METRIC_LABELS.items():
        val = scores.get(key)
        threshold = METRIC_THRESHOLDS[key]
        if val is None:
            status = "  N/A  "
            display = "  N/A"
        else:
            status = "  PASS" if val >= threshold else "  FAIL"
            display = f" {val:.4f}"
        print(f"  {label}: {display}{status}  (threshold >= {threshold})")

    print(f"\n  Average Score       :  {avg:.4f}")
    print(f"  Samples evaluated   :  {results['metadata']['num_samples']}")
    print(f"  Eval model          :  {results['metadata']['eval_model']}")
    print(f"  Timestamp           :  {results['metadata']['timestamp']}")

    n_pass = sum(
        1 for k, v in scores.items()
        if v is not None and v >= METRIC_THRESHOLDS[k]
    )
    n_total = sum(1 for v in scores.values() if v is not None)
    print(f"\n  Metrics passing     :  {n_pass}/{n_total}")
    print("=" * 62)


def save_results(results: dict, output_path: str) -> None:
    """Save evaluation results as JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Dry-run validation (no LLM calls)
# ---------------------------------------------------------------------------

def dry_run_validation(qa_pairs: list[dict]) -> bool:
    """
    Validate dataset structure without making any LLM or API calls.
    Checks that each QA pair has the required keys and correct value types.
    Returns True if all pairs are valid.
    """
    required_keys = {"question", "ground_truth", "expected_tools", "question_type"}
    all_valid = True

    print(f"\n  Validating {len(qa_pairs)} RAG evaluation QA pairs ...")
    for i, qa in enumerate(qa_pairs):
        missing = required_keys - set(qa.keys())
        if missing:
            print(f"  ERROR: QA pair {i} missing keys: {missing}")
            all_valid = False
            continue
        if not isinstance(qa["question"], str) or not qa["question"].strip():
            print(f"  ERROR: QA pair {i} has empty/invalid 'question'")
            all_valid = False
        if not isinstance(qa["ground_truth"], str) or not qa["ground_truth"].strip():
            print(f"  ERROR: QA pair {i} has empty/invalid 'ground_truth'")
            all_valid = False
        if not isinstance(qa["expected_tools"], list):
            print(f"  ERROR: QA pair {i} 'expected_tools' is not a list")
            all_valid = False

    if all_valid:
        print(f"  OK: All {len(qa_pairs)} QA pairs are structurally valid.")
    return all_valid


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="RAGAS RAG Evaluation for the LLM_LANGCHAIN_APP"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate dataset structure without making LLM calls"
    )
    parser.add_argument(
        "--output", default="results/rag_eval.json",
        help="Output JSON file path (default: results/rag_eval.json)"
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="OpenAI model for RAG chain and evaluation (default: gpt-4o)"
    )
    args = parser.parse_args()

    from eval_dataset import get_rag_eval_pairs
    qa_pairs = get_rag_eval_pairs()

    if args.dry_run:
        print("\n[DRY RUN] Validating RAG evaluation dataset structure ...")
        valid = dry_run_validation(qa_pairs)
        return 0 if valid else 1

    if not check_api_key():
        print("ERROR: OPENAI_API_KEY is not set. Add it to .env or environment.")
        return 1

    print(f"\n[eval_rag] Building synthetic vectorstore ...")
    vectorstore = build_eval_vectorstore()

    print(f"\n[eval_rag] Collecting RAG outputs for {len(qa_pairs)} questions ...")
    samples = build_evaluation_dataset(vectorstore, qa_pairs, model_name=args.model)

    print(f"\n[eval_rag] Running RAGAS evaluation (model={args.model}) ...")
    results = run_ragas_evaluation(samples, eval_model_name=args.model)

    print_results(results)
    save_results(results, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
