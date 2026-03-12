"""
eval_agent.py — Agent behavior evaluation for the LLM_LANGCHAIN_APP.

Evaluates 5 metrics per question:
  1. tool_selection_accuracy  : Did the agent use the expected tool(s)?
  2. iteration_efficiency     : Did the agent answer without wasting steps?
  3. goal_completion          : Did the agent produce a useful final answer?
  4. correctness_llm_judge    : Is the answer factually correct? (LLM-as-judge, 0-5 → 0.0-1.0)
  5. tool_call_count          : Total tool calls made (informational)

The agent loop runs fully automated (no human-in-the-loop approval).
All tool calls are auto-approved.

Usage:
  python eval_agent.py                     # full eval for 'user' role
  python eval_agent.py --role admin        # include web_search cases
  python eval_agent.py --no-judge          # skip LLM-as-judge (faster, cheaper)
  python eval_agent.py --dry-run           # validate structure, no LLM calls
  python eval_agent.py --output path.json  # custom output path

Results are saved to results/agent_eval.json by default.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from config import MAX_AGENT_ITERATIONS
from guardrails import AgentTracer
from agent import invoke_agent, execute_tool_calls, build_initial_messages


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentRunResult:
    """Result from a single automated agent evaluation run."""
    final_answer: str
    tool_calls: list = field(default_factory=list)   # [{"name", "args", "iteration"}]
    iterations: int = 0
    hit_max_iterations: bool = False
    tracer_events: list = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def tools_used(self) -> list[str]:
        return [tc["name"] for tc in self.tool_calls]

    @property
    def unique_tools_used(self) -> set:
        return set(self.tools_used)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)


# ---------------------------------------------------------------------------
# Automated agent loop (no Streamlit, no human approval)
# ---------------------------------------------------------------------------

def run_agent_loop(
    llm_with_tools,
    tool_map: dict,
    question: str,
    max_iterations: int = MAX_AGENT_ITERATIONS,
) -> AgentRunResult:
    """
    Drive the agent loop fully automatically for evaluation.

    Mirrors the logic in app.py but:
      - No Streamlit session state
      - Auto-approves ALL tool calls immediately
      - Returns structured AgentRunResult instead of updating UI

    Args:
        llm_with_tools : LLM with bound tools (from build_agent())
        tool_map       : {tool_name: tool_fn} dict (from build_agent())
        question       : The evaluation question to answer
        max_iterations : Cap on LLM invocations (defaults to config value)

    Returns:
        AgentRunResult with final answer, tool call history, and tracer events
    """
    tracer = AgentTracer()
    tracer.log_user_input(question)

    messages = build_initial_messages(question, chat_history=[])
    iteration = 0
    all_tool_calls: list[dict] = []
    final_answer = ""
    hit_max = False

    start_time = time.perf_counter()

    while True:
        iteration += 1
        tracer.log_llm_invoked(iteration)

        # Enforce max iterations: inject override message then get final answer
        if iteration > max_iterations:
            hit_max = True
            tracer.log_max_iterations_reached(iteration - 1)
            messages.append(HumanMessage(content=(
                "You have reached the maximum number of tool calls. "
                "Please provide your best final answer now based on the "
                "information gathered so far. Do not request any more tools."
            )))

        ai_msg: AIMessage = invoke_agent(llm_with_tools, messages)
        messages.append(ai_msg)

        # Stop if no tool calls (or forced final answer after max iterations)
        if hit_max or not ai_msg.tool_calls:
            final_answer = ai_msg.content or ""
            tracer.log_final_answer(len(final_answer))
            break

        # Auto-approve and execute all proposed tool calls
        for tc in ai_msg.tool_calls:
            tracer.log_tool_proposed(tc["name"], tc["args"])
            tracer.log_tool_approved(tc["name"])
            all_tool_calls.append({
                "name": tc["name"],
                "args": tc["args"],
                "iteration": iteration,
            })

        tool_messages: list[ToolMessage] = execute_tool_calls(
            tool_map, ai_msg.tool_calls, tracer=tracer
        )
        messages.extend(tool_messages)

    duration_ms = (time.perf_counter() - start_time) * 1000

    return AgentRunResult(
        final_answer=final_answer,
        tool_calls=all_tool_calls,
        iterations=iteration,
        hit_max_iterations=hit_max,
        tracer_events=tracer.get_events(),
        duration_ms=round(duration_ms, 1),
    )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def score_tool_selection(result: AgentRunResult, expected_tools: list[str]) -> float:
    """
    Fraction of expected tools that were actually used by the agent.
    Score = |used ∩ expected| / |expected|

    Returns 1.0 if no tools were expected (question answerable without tools).
    This metric is intentionally lenient: using extra tools does not penalize.
    """
    if not expected_tools:
        return 1.0
    expected = set(expected_tools)
    used = result.unique_tools_used
    overlap = len(used & expected)
    return round(overlap / len(expected), 4)


def score_iteration_efficiency(
    result: AgentRunResult,
    max_iterations: int = MAX_AGENT_ITERATIONS,
) -> float:
    """
    How efficiently did the agent reach its answer?

    Formula: 1.0 - (iterations - 1) / max_iterations
      - 1 iteration  → 1.0  (perfect)
      - max iterations → 0.0
      - Hitting the cap adds a -0.2 penalty (agent couldn't self-terminate)

    Returns value clamped to [0.0, 1.0].
    """
    base = 1.0 - (result.iterations - 1) / max_iterations
    if result.hit_max_iterations:
        base = max(0.0, base - 0.2)
    return round(max(0.0, min(1.0, base)), 4)


def score_goal_completion(result: AgentRunResult) -> float:
    """
    Did the agent produce a substantive final answer?

    1.0 → non-empty answer with no error/failure markers
    0.5 → answer exists but signals graceful failure (partial credit)
    0.0 → empty answer
    """
    answer = result.final_answer.strip()
    if not answer:
        return 0.0

    graceful_failure_markers = [
        "i was unable to generate",
        "i cannot answer",
        "i don't have enough information",
        "i do not have enough information",
        "i'm unable to",
        "i am unable to",
        "error:",
    ]
    lower = answer.lower()
    if any(marker in lower for marker in graceful_failure_markers):
        return 0.5

    return 1.0


def score_correctness_llm_judge(
    question: str,
    answer: str,
    ground_truth: str,
    judge_model: str = "gpt-4o",
) -> dict:
    """
    LLM-as-judge: score the factual correctness of the agent's answer
    against the ground truth reference.

    Scoring rubric (0–5):
      5 = Fully correct, all key facts match the reference
      4 = Mostly correct with minor omissions
      3 = Partially correct, some key facts right
      2 = Mostly incorrect, only one or two facts correct
      1 = Largely incorrect
      0 = Completely wrong or no answer provided

    Returns:
        {
            "score": float,       # normalized to 0.0–1.0
            "raw_score": int,     # 0–5 integer
            "reasoning": str,     # one-sentence explanation
            "error": str | None,  # if parsing failed
        }
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage as HM

    judge_llm = ChatOpenAI(model=judge_model, temperature=0)

    system_prompt = (
        "You are an objective evaluator assessing the factual correctness of an AI answer "
        "against a reference answer. Score the answer from 0 to 5:\n"
        "  5 = Fully correct, all key facts present and accurate\n"
        "  4 = Mostly correct, minor omissions only\n"
        "  3 = Partially correct, some key facts right\n"
        "  2 = Mostly incorrect, one or two facts correct\n"
        "  1 = Largely incorrect\n"
        "  0 = Completely wrong or no answer provided\n\n"
        'Respond ONLY with valid JSON in this exact format: '
        '{"score": <integer 0-5>, "reasoning": "<one sentence explanation>"}'
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Reference answer: {ground_truth}\n\n"
        f"AI answer to evaluate: {answer}"
    )

    try:
        response = judge_llm.invoke([
            SystemMessage(content=system_prompt),
            HM(content=user_prompt),
        ])
        raw_text = response.content.strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        raw_score = int(parsed.get("score", 0))
        raw_score = max(0, min(5, raw_score))
        return {
            "score": round(raw_score / 5.0, 4),
            "raw_score": raw_score,
            "reasoning": parsed.get("reasoning", ""),
            "error": None,
        }
    except Exception as e:
        return {
            "score": 0.0,
            "raw_score": 0,
            "reasoning": "",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

def evaluate_case(
    qa: dict,
    llm_with_tools,
    tool_map: dict,
    use_judge: bool = True,
    judge_model: str = "gpt-4o",
) -> dict:
    """
    Run the agent for one QA case and compute all metrics.

    Returns a flat dict with the question, metrics, and agent run details.
    """
    question = qa["question"]
    ground_truth = qa["ground_truth"]
    expected_tools = qa.get("expected_tools", [])
    question_type = qa.get("question_type", "unknown")

    result = run_agent_loop(llm_with_tools, tool_map, question)

    tool_accuracy = score_tool_selection(result, expected_tools)
    iter_efficiency = score_iteration_efficiency(result)
    goal_comp = score_goal_completion(result)

    if use_judge:
        judge_result = score_correctness_llm_judge(
            question, result.final_answer, ground_truth, judge_model=judge_model
        )
        correctness = judge_result["score"]
        correctness_reasoning = judge_result["reasoning"]
        correctness_error = judge_result["error"]
    else:
        correctness = None
        correctness_reasoning = "skipped (--no-judge)"
        correctness_error = None

    return {
        "question": question,
        "question_type": question_type,
        "expected_tools": expected_tools,
        "tools_used": sorted(result.tools_used),
        "tool_call_count": result.tool_call_count,
        "iterations": result.iterations,
        "hit_max_iterations": result.hit_max_iterations,
        "duration_ms": result.duration_ms,
        "final_answer_length": len(result.final_answer),
        # Metrics
        "tool_selection_accuracy": tool_accuracy,
        "iteration_efficiency": iter_efficiency,
        "goal_completion": goal_comp,
        "correctness_llm_judge": correctness,
        "correctness_reasoning": correctness_reasoning,
        "correctness_error": correctness_error,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics and reporting
# ---------------------------------------------------------------------------

METRIC_THRESHOLDS = {
    "tool_selection_accuracy": 0.7,
    "iteration_efficiency": 0.6,
    "goal_completion": 0.8,
    "correctness_llm_judge": 0.6,
}


def compute_aggregate(case_results: list[dict]) -> dict:
    """Compute mean of each metric across all cases (ignoring None values)."""
    metrics = list(METRIC_THRESHOLDS.keys())
    aggregates = {}
    for metric in metrics:
        values = [c[metric] for c in case_results if c.get(metric) is not None]
        if values:
            aggregates[metric] = round(sum(values) / len(values), 4)
        else:
            aggregates[metric] = None
    aggregates["avg_iterations"] = round(
        sum(c["iterations"] for c in case_results) / len(case_results), 2
    )
    aggregates["avg_duration_ms"] = round(
        sum(c["duration_ms"] for c in case_results) / len(case_results), 1
    )
    aggregates["avg_tool_call_count"] = round(
        sum(c["tool_call_count"] for c in case_results) / len(case_results), 2
    )
    return aggregates


def print_case_table(case_results: list[dict]) -> None:
    """Print a per-case result table."""
    print("\n" + "-" * 100)
    print(f"  {'#':<3} {'Question':<45} {'Tools':<24} {'It':<4} {'Goal':<6} {'Correct':<9} {'ms':<7}")
    print("-" * 100)
    for i, c in enumerate(case_results):
        short_q = c["question"][:44]
        tools_str = ",".join(c["tools_used"]) if c["tools_used"] else "—"
        tools_str = tools_str[:22]
        correct_str = f"{c['correctness_llm_judge']:.2f}" if c["correctness_llm_judge"] is not None else "skip"
        print(
            f"  {i+1:<3} {short_q:<45} {tools_str:<24} "
            f"{c['iterations']:<4} {c['goal_completion']:<6.1f} {correct_str:<9} {c['duration_ms']:<7.0f}"
        )
    print("-" * 100)


def print_aggregate_results(aggregates: dict, n_cases: int, role: str, use_judge: bool) -> None:
    """Print aggregate metrics with pass/fail thresholds."""
    print("\n" + "=" * 62)
    print("  AGENT EVALUATION AGGREGATE RESULTS")
    print("=" * 62)
    print(f"  Role            : {role}")
    print(f"  Cases evaluated : {n_cases}")
    print(f"  LLM-as-judge    : {'enabled' if use_judge else 'disabled'}")
    print()

    metric_labels = {
        "tool_selection_accuracy": "Tool Selection Accuracy ",
        "iteration_efficiency":    "Iteration Efficiency    ",
        "goal_completion":         "Goal Completion         ",
        "correctness_llm_judge":   "Correctness (LLM Judge) ",
    }

    n_pass = 0
    n_total = 0
    for key, label in metric_labels.items():
        val = aggregates.get(key)
        threshold = METRIC_THRESHOLDS[key]
        if val is None:
            print(f"  {label}:  N/A    (skipped)")
        else:
            status = "PASS" if val >= threshold else "FAIL"
            print(f"  {label}:  {val:.4f}  {status}  (threshold >= {threshold})")
            n_total += 1
            if val >= threshold:
                n_pass += 1

    print()
    print(f"  Avg iterations      :  {aggregates['avg_iterations']}")
    print(f"  Avg tool calls      :  {aggregates['avg_tool_call_count']}")
    print(f"  Avg duration (ms)   :  {aggregates['avg_duration_ms']}")
    print(f"\n  Metrics passing     :  {n_pass}/{n_total}")
    print("=" * 62)


def save_results(
    case_results: list[dict],
    aggregates: dict,
    output_path: str,
    metadata: dict,
) -> None:
    """Save full evaluation results as JSON."""
    payload = {
        "metadata": metadata,
        "aggregate": aggregates,
        "cases": case_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Dry-run validation (no LLM calls)
# ---------------------------------------------------------------------------

def dry_run_validation(qa_pairs: list[dict]) -> bool:
    """
    Validate dataset and scoring function correctness without LLM calls.
    Returns True if all checks pass.
    """
    print("\n[DRY RUN] Validating agent evaluation setup ...")

    # 1. Dataset structure
    required_keys = {"question", "ground_truth", "expected_tools", "question_type"}
    all_valid = True
    for i, qa in enumerate(qa_pairs):
        missing = required_keys - set(qa.keys())
        if missing:
            print(f"  ERROR: QA pair {i} missing keys: {missing}")
            all_valid = False

    if all_valid:
        print(f"  OK: {len(qa_pairs)} QA pairs are structurally valid.")
    else:
        return False

    # 2. Tool selection scoring
    mock_result_hit = AgentRunResult(
        final_answer="answer", tool_calls=[{"name": "document_search", "args": {}, "iteration": 1}],
        iterations=1, hit_max_iterations=False, tracer_events=[], duration_ms=100.0,
    )
    score = score_tool_selection(mock_result_hit, ["document_search"])
    assert score == 1.0, f"Expected 1.0, got {score}"
    score_miss = score_tool_selection(mock_result_hit, ["web_search"])
    assert score_miss == 0.0, f"Expected 0.0, got {score_miss}"
    print("  OK: tool_selection_accuracy scoring correct.")

    # 3. Iteration efficiency scoring
    result_1iter = AgentRunResult(
        final_answer="x", tool_calls=[], iterations=1,
        hit_max_iterations=False, tracer_events=[], duration_ms=0.0,
    )
    eff = score_iteration_efficiency(result_1iter)
    assert eff == 1.0, f"Expected 1.0, got {eff}"

    result_max = AgentRunResult(
        final_answer="x", tool_calls=[], iterations=MAX_AGENT_ITERATIONS,
        hit_max_iterations=True, tracer_events=[], duration_ms=0.0,
    )
    eff_max = score_iteration_efficiency(result_max)
    assert eff_max == 0.0, f"Expected 0.0 for hit_max, got {eff_max}"
    print("  OK: iteration_efficiency scoring correct.")

    # 4. Goal completion scoring
    assert score_goal_completion(AgentRunResult(
        final_answer="The answer is 42.", tool_calls=[], iterations=1,
        hit_max_iterations=False, tracer_events=[], duration_ms=0.0,
    )) == 1.0
    assert score_goal_completion(AgentRunResult(
        final_answer="", tool_calls=[], iterations=1,
        hit_max_iterations=False, tracer_events=[], duration_ms=0.0,
    )) == 0.0
    assert score_goal_completion(AgentRunResult(
        final_answer="I was unable to generate an answer.", tool_calls=[], iterations=1,
        hit_max_iterations=False, tracer_events=[], duration_ms=0.0,
    )) == 0.5
    print("  OK: goal_completion scoring correct.")

    print(f"\n  All dry-run checks passed.")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Agent Evaluation for the LLM_LANGCHAIN_APP"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate dataset structure and metric functions without LLM calls"
    )
    parser.add_argument(
        "--role", default="user", choices=["user", "admin", "viewer"],
        help="Agent role to evaluate (default: user)"
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="OpenAI model for the agent (default: gpt-4o)"
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM-as-judge correctness scoring (faster, cheaper)"
    )
    parser.add_argument(
        "--output", default="results/agent_eval.json",
        help="Output JSON file path (default: results/agent_eval.json)"
    )
    args = parser.parse_args()

    from eval_dataset import get_agent_eval_cases
    qa_pairs = get_agent_eval_cases(role=args.role)

    if args.dry_run:
        ok = dry_run_validation(qa_pairs)
        return 0 if ok else 1

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key.strip():
        print("ERROR: OPENAI_API_KEY is not set. Add it to .env or environment.")
        return 1

    # Build synthetic vectorstore (for document tools)
    print(f"\n[eval_agent] Building synthetic vectorstore ...")
    from eval_dataset import build_synthetic_vectorstore
    from guardrails import ReadOnlyVectorstore

    raw_vs = build_synthetic_vectorstore(persist_dir=None)
    vectorstore = ReadOnlyVectorstore(raw_vs)

    # Build agent components
    from agent import build_agent
    print(f"[eval_agent] Building agent (role={args.role}, model={args.model}) ...")
    llm_with_tools, _, tool_map = build_agent(
        vectorstore, model_name=args.model, role=args.role
    )

    use_judge = not args.no_judge

    # Run evaluation for each case
    print(f"[eval_agent] Evaluating {len(qa_pairs)} cases ...\n")
    case_results = []
    for i, qa in enumerate(qa_pairs):
        short_q = qa["question"][:55] + ("..." if len(qa["question"]) > 55 else "")
        print(f"  [{i+1}/{len(qa_pairs)}] {short_q}")
        case_result = evaluate_case(
            qa, llm_with_tools, tool_map,
            use_judge=use_judge, judge_model=args.model,
        )
        case_results.append(case_result)

    # Aggregate and report
    aggregates = compute_aggregate(case_results)
    print_case_table(case_results)
    print_aggregate_results(aggregates, len(case_results), args.role, use_judge)

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "model": args.model,
        "use_judge": use_judge,
        "num_cases": len(case_results),
    }
    save_results(case_results, aggregates, args.output, metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
