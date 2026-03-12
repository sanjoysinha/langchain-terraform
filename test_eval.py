"""
test_eval.py — Structural tests for the evaluation suite.

Tests all evaluation components WITHOUT making any LLM or API calls.
Uses mocks and known-good inputs to verify:
  1. eval_dataset.py – Dataset structure and vectorstore factory API
  2. eval_rag.py     – Dry-run validation and sample format
  3. eval_agent.py   – AgentRunResult, all scoring functions, auto-loop logic

Run with: python test_eval.py
"""

from dotenv import load_dotenv
load_dotenv()

PASS = 0
FAIL = 0


def check(label: str, condition: bool) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {label}")
    else:
        FAIL += 1
        print(f"  FAIL: {label}")


# ══════════════════════════════════════════════════════════════════
print("\n1. EVAL_DATASET — Synthetic Corpus Structure")
print("=" * 55)

from eval_dataset import (
    SYNTHETIC_DOCS, GROUND_TRUTH_QA,
    get_rag_eval_pairs, get_agent_eval_cases,
)

# Corpus checks
check("5 synthetic documents defined", len(SYNTHETIC_DOCS) == 5)
check("All docs are non-empty strings",
      all(isinstance(v, str) and len(v) > 50 for v in SYNTHETIC_DOCS.values()))
check("company_history doc present", "company_history" in SYNTHETIC_DOCS)
check("product_catalog doc present", "product_catalog" in SYNTHETIC_DOCS)
check("support_policy doc present", "support_policy" in SYNTHETIC_DOCS)
check("security_compliance doc present", "security_compliance" in SYNTHETIC_DOCS)
check("engineering_practices doc present", "engineering_practices" in SYNTHETIC_DOCS)

# Spot-check known facts are in the corpus
check("Elena Vasquez in company_history",
      "Elena Vasquez" in SYNTHETIC_DOCS["company_history"])
check("AES-256 in security_compliance",
      "AES-256" in SYNTHETIC_DOCS["security_compliance"])
check("80 percent in engineering_practices",
      "80 percent" in SYNTHETIC_DOCS["engineering_practices"])
check("99 dollars in product_catalog",
      "99 dollars" in SYNTHETIC_DOCS["product_catalog"])

# QA pair structure
check("12 QA pairs defined", len(GROUND_TRUTH_QA) == 12)

required_qa_keys = {"question", "ground_truth", "expected_tools", "question_type"}
all_valid = all(required_qa_keys.issubset(qa.keys()) for qa in GROUND_TRUTH_QA)
check("All QA pairs have required keys", all_valid)

check("All questions are non-empty strings",
      all(isinstance(qa["question"], str) and qa["question"].strip()
          for qa in GROUND_TRUTH_QA))
check("All ground_truths are non-empty strings",
      all(isinstance(qa["ground_truth"], str) and qa["ground_truth"].strip()
          for qa in GROUND_TRUTH_QA))
check("All expected_tools are lists",
      all(isinstance(qa["expected_tools"], list) for qa in GROUND_TRUTH_QA))

# get_rag_eval_pairs — excludes out_of_scope and web_search_required
rag_pairs = get_rag_eval_pairs()
check("get_rag_eval_pairs returns 9+ pairs", len(rag_pairs) >= 9)
check("get_rag_eval_pairs excludes out_of_scope",
      all(qa["question_type"] != "out_of_scope" for qa in rag_pairs))
check("get_rag_eval_pairs excludes web_search_required",
      all(qa["question_type"] != "web_search_required" for qa in rag_pairs))

# get_agent_eval_cases — role filtering
user_cases = get_agent_eval_cases(role="user")
admin_cases = get_agent_eval_cases(role="admin")
check("user role excludes web_search_required cases",
      all(qa.get("requires_role") != "admin" for qa in user_cases))
check("admin role includes all 12 cases", len(admin_cases) == 12)
check("user role has fewer cases than admin", len(user_cases) < len(admin_cases))


# ══════════════════════════════════════════════════════════════════
print("\n2. EVAL_DATASET — Vectorstore Factory (structure only, no API calls)")
print("=" * 55)

import inspect
from eval_dataset import build_synthetic_vectorstore

sig = inspect.signature(build_synthetic_vectorstore)
check("build_synthetic_vectorstore has persist_dir param",
      "persist_dir" in sig.parameters)
check("persist_dir defaults to None",
      sig.parameters["persist_dir"].default is None)


# ══════════════════════════════════════════════════════════════════
print("\n3. EVAL_RAG — Dry-Run Validation")
print("=" * 55)

from eval_rag import dry_run_validation, collect_rag_sample, run_ragas_evaluation

# Valid dataset passes
result = dry_run_validation(rag_pairs)
check("dry_run_validation passes on valid pairs", result is True)

# Detect missing keys
bad_pairs = [{"question": "test"}]  # missing ground_truth, expected_tools, question_type
result_bad = dry_run_validation(bad_pairs)
check("dry_run_validation detects missing keys", result_bad is False)

# collect_rag_sample function signature
sig_collect = inspect.signature(collect_rag_sample)
check("collect_rag_sample has vectorstore param", "vectorstore" in sig_collect.parameters)
check("collect_rag_sample has question param", "question" in sig_collect.parameters)
check("collect_rag_sample has ground_truth param", "ground_truth" in sig_collect.parameters)
check("collect_rag_sample has model_name param", "model_name" in sig_collect.parameters)

# RAGAS required field names (validate sample format)
ragas_fields = {"user_input", "retrieved_contexts", "response", "reference"}
sample_template = {
    "user_input": "test question",
    "retrieved_contexts": ["context chunk 1"],
    "response": "test answer",
    "reference": "ground truth answer",
}
check("RAGAS sample has all required fields",
      ragas_fields.issubset(sample_template.keys()))
check("retrieved_contexts is a list",
      isinstance(sample_template["retrieved_contexts"], list))


# ══════════════════════════════════════════════════════════════════
print("\n4. EVAL_AGENT — AgentRunResult Dataclass")
print("=" * 55)

from eval_agent import AgentRunResult

# Basic construction
r = AgentRunResult(
    final_answer="The answer is 42.",
    tool_calls=[
        {"name": "document_search", "args": {"query": "test"}, "iteration": 1},
        {"name": "document_search", "args": {"query": "test2"}, "iteration": 2},
        {"name": "web_search", "args": {"query": "test3"}, "iteration": 2},
    ],
    iterations=2,
    hit_max_iterations=False,
    tracer_events=[],
    duration_ms=1234.5,
)

check("tools_used returns list of tool names",
      r.tools_used == ["document_search", "document_search", "web_search"])
check("unique_tools_used returns set",
      r.unique_tools_used == {"document_search", "web_search"})
check("tool_call_count returns 3", r.tool_call_count == 3)
check("final_answer stored correctly", r.final_answer == "The answer is 42.")
check("duration_ms stored correctly", r.duration_ms == 1234.5)

# Empty run
empty_r = AgentRunResult(
    final_answer="", tool_calls=[], iterations=1,
    hit_max_iterations=False, tracer_events=[], duration_ms=0.0,
)
check("Empty run: tools_used is empty list", empty_r.tools_used == [])
check("Empty run: unique_tools_used is empty set", empty_r.unique_tools_used == set())
check("Empty run: tool_call_count is 0", empty_r.tool_call_count == 0)


# ══════════════════════════════════════════════════════════════════
print("\n5. EVAL_AGENT — score_tool_selection")
print("=" * 55)

from eval_agent import score_tool_selection

def make_result(tool_names):
    return AgentRunResult(
        final_answer="ans",
        tool_calls=[{"name": n, "args": {}, "iteration": 1} for n in tool_names],
        iterations=1, hit_max_iterations=False, tracer_events=[], duration_ms=0.0,
    )

check("Exact match → 1.0",
      score_tool_selection(make_result(["document_search"]), ["document_search"]) == 1.0)
check("Complete miss → 0.0",
      score_tool_selection(make_result(["web_search"]), ["document_search"]) == 0.0)
check("No expected tools → 1.0",
      score_tool_selection(make_result([]), []) == 1.0)
check("Partial match (1 of 2) → 0.5",
      score_tool_selection(
          make_result(["document_search"]),
          ["document_search", "web_search"]
      ) == 0.5)
check("Extra tool used (still matches expected) → 1.0",
      score_tool_selection(
          make_result(["document_search", "web_search"]),
          ["document_search"]
      ) == 1.0)
check("Score is a float in [0.0, 1.0]",
      0.0 <= score_tool_selection(make_result(["document_search"]), ["document_search"]) <= 1.0)


# ══════════════════════════════════════════════════════════════════
print("\n6. EVAL_AGENT — score_iteration_efficiency")
print("=" * 55)

from eval_agent import score_iteration_efficiency
from config import MAX_AGENT_ITERATIONS

def make_iter_result(iters, hit_max=False):
    return AgentRunResult(
        final_answer="ans", tool_calls=[], iterations=iters,
        hit_max_iterations=hit_max, tracer_events=[], duration_ms=0.0,
    )

check("1 iteration → 1.0 (perfect)",
      score_iteration_efficiency(make_iter_result(1)) == 1.0)
check("2 iterations → 0.8",
      score_iteration_efficiency(make_iter_result(2)) == round(1 - 1/MAX_AGENT_ITERATIONS, 4))
expected_at_max = round(1 - (MAX_AGENT_ITERATIONS - 1) / MAX_AGENT_ITERATIONS, 4)
check(f"Max iterations without hit_max → {expected_at_max}",
      score_iteration_efficiency(make_iter_result(MAX_AGENT_ITERATIONS)) == expected_at_max)
check("hit_max_iterations = True → score <= 0.0 (penalty applied)",
      score_iteration_efficiency(make_iter_result(MAX_AGENT_ITERATIONS, hit_max=True)) <= 0.0)
check("Score always in [0.0, 1.0]",
      all(
          0.0 <= score_iteration_efficiency(make_iter_result(i)) <= 1.0
          for i in range(1, MAX_AGENT_ITERATIONS + 1)
      ))


# ══════════════════════════════════════════════════════════════════
print("\n7. EVAL_AGENT — score_goal_completion")
print("=" * 55)

from eval_agent import score_goal_completion

def make_answer_result(answer):
    return AgentRunResult(
        final_answer=answer, tool_calls=[], iterations=1,
        hit_max_iterations=False, tracer_events=[], duration_ms=0.0,
    )

check("Good answer → 1.0",
      score_goal_completion(make_answer_result("AcmeCloud Pro costs $99/user/month.")) == 1.0)
check("Empty answer → 0.0",
      score_goal_completion(make_answer_result("")) == 0.0)
check("Whitespace-only answer → 0.0",
      score_goal_completion(make_answer_result("   ")) == 0.0)
check("Graceful failure marker → 0.5",
      score_goal_completion(make_answer_result(
          "I was unable to generate a complete answer with the available context."
      )) == 0.5)
check("'I cannot answer' → 0.5",
      score_goal_completion(make_answer_result("I cannot answer this question.")) == 0.5)
check("Verbose good answer → 1.0",
      score_goal_completion(make_answer_result(
          "The SLA guarantees 99.9% uptime for all paid tiers."
      )) == 1.0)


# ══════════════════════════════════════════════════════════════════
print("\n8. EVAL_AGENT — score_correctness_llm_judge (function signature)")
print("=" * 55)

from eval_agent import score_correctness_llm_judge

sig_judge = inspect.signature(score_correctness_llm_judge)
check("judge has 'question' param", "question" in sig_judge.parameters)
check("judge has 'answer' param", "answer" in sig_judge.parameters)
check("judge has 'ground_truth' param", "ground_truth" in sig_judge.parameters)
check("judge has 'judge_model' param", "judge_model" in sig_judge.parameters)
check("judge_model defaults to gpt-4o",
      sig_judge.parameters["judge_model"].default == "gpt-4o")


# ══════════════════════════════════════════════════════════════════
print("\n9. EVAL_AGENT — run_agent_loop with Mock LLM")
print("=" * 55)

# Mock LLM: returns a tool call on iter 1, final answer on iter 2
from langchain_core.messages import AIMessage, ToolMessage

class MockLLM:
    """Simulates a tool-calling LLM. No OpenAI calls made."""
    def __init__(self, tool_call_first=True):
        self.call_count = 0
        self.tool_call_first = tool_call_first

    def invoke(self, messages):
        self.call_count += 1
        if self.tool_call_first and self.call_count == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "document_search",
                    "args": {"query": "test query"},
                    "id": "tc_mock_001",
                }],
            )
        return AIMessage(content="The answer is 42 according to the documents.")

    def bind_tools(self, tools):
        return self


class MockTool:
    """Simulates a tool execution. Returns instantly."""
    def __init__(self, name):
        self.name = name

    def invoke(self, args):
        return f"Mock result for {self.name} with args {args}"


mock_llm = MockLLM(tool_call_first=True)
mock_tool = MockTool("document_search")
mock_tool_map = {"document_search": mock_tool}

from eval_agent import run_agent_loop

# Patch execute_tool_calls to use our mock without timeout infrastructure
import eval_agent as _ea
_orig_execute = _ea.execute_tool_calls

def _mock_execute_tool_calls(tool_map, tool_calls, tracer=None):
    messages = []
    for tc in tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        if tracer:
            tracer.log_tool_executed(tc["name"], 10.0, True)
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return messages

_ea.execute_tool_calls = _mock_execute_tool_calls

result_mock = run_agent_loop(mock_llm, mock_tool_map, "Who founded Acme?")

_ea.execute_tool_calls = _orig_execute  # restore

check("Mock agent loop completes without error", result_mock is not None)
check("Mock agent loop has non-empty final answer",
      bool(result_mock.final_answer.strip()))
check("Mock agent loop used document_search",
      "document_search" in result_mock.unique_tools_used)
check("Mock agent loop took 2 iterations", result_mock.iterations == 2)
check("hit_max_iterations is False", result_mock.hit_max_iterations is False)
check("tool_calls recorded correctly", result_mock.tool_call_count == 1)
check("tracer_events populated", len(result_mock.tracer_events) > 0)
check("duration_ms is positive", result_mock.duration_ms > 0)

# Test direct-answer scenario (no tool call)
mock_llm_direct = MockLLM(tool_call_first=False)
result_direct = run_agent_loop(mock_llm_direct, mock_tool_map, "What is 2+2?")
check("Direct answer: 1 iteration", result_direct.iterations == 1)
check("Direct answer: no tool calls", result_direct.tool_call_count == 0)
check("Direct answer: non-empty answer", bool(result_direct.final_answer.strip()))


# ══════════════════════════════════════════════════════════════════
print("\n10. EVAL_AGENT — compute_aggregate and dry_run_validation")
print("=" * 55)

from eval_agent import compute_aggregate, dry_run_validation as agent_dry_run

sample_cases = [
    {
        "question": "Q1", "question_type": "factual", "expected_tools": ["document_search"],
        "tools_used": ["document_search"], "tool_call_count": 1, "iterations": 1,
        "hit_max_iterations": False, "duration_ms": 500.0, "final_answer_length": 50,
        "tool_selection_accuracy": 1.0, "iteration_efficiency": 1.0,
        "goal_completion": 1.0, "correctness_llm_judge": 0.8,
        "correctness_reasoning": "correct", "correctness_error": None,
    },
    {
        "question": "Q2", "question_type": "factual", "expected_tools": ["document_search"],
        "tools_used": [], "tool_call_count": 0, "iterations": 3,
        "hit_max_iterations": False, "duration_ms": 1200.0, "final_answer_length": 30,
        "tool_selection_accuracy": 0.0, "iteration_efficiency": 0.6,
        "goal_completion": 1.0, "correctness_llm_judge": 0.4,
        "correctness_reasoning": "partially wrong", "correctness_error": None,
    },
]

agg = compute_aggregate(sample_cases)
check("aggregate tool_selection_accuracy is mean", agg["tool_selection_accuracy"] == 0.5)
check("aggregate goal_completion is mean", agg["goal_completion"] == 1.0)
check("aggregate avg_iterations computed", agg["avg_iterations"] == 2.0)
check("aggregate avg_duration_ms computed", agg["avg_duration_ms"] == 850.0)
check("aggregate correctness_llm_judge is mean", agg["correctness_llm_judge"] == 0.6)

# agent dry_run with user role cases
ok = agent_dry_run(get_agent_eval_cases(role="user"))
check("agent dry_run passes for user role cases", ok is True)

bad_agent_cases = [{"question": "Q?"}]  # missing required keys
ok_bad = agent_dry_run(bad_agent_cases)
check("agent dry_run fails for malformed cases", ok_bad is False)


# ══════════════════════════════════════════════════════════════════
print("\n11. EVAL_RAG — check_api_key and save_results signature")
print("=" * 55)

from eval_rag import check_api_key, save_results as rag_save_results
import os

# With no key set
original_key = os.environ.pop("OPENAI_API_KEY", None)
check("check_api_key returns False when key missing", check_api_key() is False)

# Restore
if original_key:
    os.environ["OPENAI_API_KEY"] = original_key
    check("check_api_key returns True when key present", check_api_key() is True)
else:
    print("  NOTE: OPENAI_API_KEY not set in environment (expected in CI)")

sig_save = inspect.signature(rag_save_results)
check("save_results has 'results' param", "results" in sig_save.parameters)
check("save_results has 'output_path' param", "output_path" in sig_save.parameters)


# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 55)

if FAIL == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"WARNING: {FAIL} test(s) failed!")
    import sys
    sys.exit(1)
