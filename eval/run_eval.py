"""Evaluation Suite Runner: Executes visible and original cases with deterministic assertions."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config
from src.main import initialize_agent
from src.session import SessionManager
from eval.assertions import check_case_expectations


def load_cases(visible_path: Path, original_path: Path, paraphrase_path: Path = None) -> list[dict]:
    """Load visible, original, and (optionally) paraphrase evaluation cases."""
    all_cases = []

    if visible_path.exists():
        with open(visible_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for c in data.get("cases", []):
                c["_source"] = "visible"
                all_cases.append(c)

    if original_path.exists():
        with open(original_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for c in data.get("cases", []):
                c["_source"] = "original"
                all_cases.append(c)

    if paraphrase_path and paraphrase_path.exists():
        with open(paraphrase_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for c in data.get("cases", []):
                c["_source"] = "paraphrase"
                all_cases.append(c)

    return all_cases


def run_evaluation(
    cases: list[dict],
    agent=None,
    verbose: bool = False,
) -> tuple[dict[str, dict], list[dict]]:
    """Run evaluation cases and collect results."""
    if agent is None:
        agent = initialize_agent()

    category_stats: dict[str, dict] = {}
    case_results: list[dict] = []

    print(f"\nRunning {len(cases)} evaluation test cases...\n")
    print(f"{'Status':<8} | {'ID':<40} | {'Category':<22} | {'Suite'}")
    print("-" * 80)

    for case in cases:
        case_id = case["id"]
        category = case.get("category", "general")
        source = case.get("_source", "visible")
        messages = case.get("messages", [])
        expect = case.get("expect", {})

        # Each test case runs in its own isolated session
        session = agent.session_manager.create_session()
        last_response = None

        # Execute all turns in order
        for msg in messages:
            content = msg.get("content", "")
            last_response = agent.process_message(content, session_id=session.session_id)

        # Run deterministic assertions on the final turn response
        passed, failures = check_case_expectations(last_response, expect)

        # Update category stats
        stats = category_stats.setdefault(category, {"total": 0, "passed": 0, "failed": 0})
        stats["total"] += 1
        if passed:
            stats["passed"] += 1
            status_str = "\033[92mPASS\033[0m"
        else:
            stats["failed"] += 1
            status_str = "\033[91mFAIL\033[0m"

        print(f"{status_str:<17} | {case_id:<40} | {category:<22} | {source}", flush=True)

        if not passed and verbose:
            print(f"   \033[93mFailures for {case_id}:\033[0m", flush=True)
            for f in failures:
                print(f"     - {f}", flush=True)
            print(f"   Response was:\n     {last_response.message[:250]}...\n", flush=True)

        case_results.append({
            "id": case_id,
            "category": category,
            "source": source,
            "passed": passed,
            "failures": failures,
            "response": last_response.message if last_response else "",
        })

        import time
        time.sleep(1.0)

    return category_stats, case_results


def print_summary_table(category_stats: dict[str, dict], total_cases: int):
    """Print clean ASCII summary table of results."""
    print("\n" + "=" * 65)
    print(f"{'Category':<28} | {'Passed':<8} | {'Total':<6} | {'Accuracy'}")
    print("=" * 65)

    total_passed = 0
    for cat, stats in sorted(category_stats.items()):
        total = stats["total"]
        passed = stats["passed"]
        total_passed += passed
        pct = (passed / total * 100) if total > 0 else 0
        print(f"{cat:<28} | {passed:<8} | {total:<6} | {pct:>5.1f}%")

    print("-" * 65)
    overall_pct = (total_passed / total_cases * 100) if total_cases > 0 else 0
    print(f"{'OVERALL':<28} | {total_passed:<8} | {total_cases:<6} | {overall_pct:>5.1f}%")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run AI support agent evaluation suite")
    parser.add_argument("--category", help="Run only cases in a specific category")
    parser.add_argument("--case", help="Run a specific test case ID")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print failure details")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate the naive baseline agent (full-corpus prompt, no RAG, no tools) "
        "instead of the full system. Used to produce the README baseline numbers.",
    )
    parser.add_argument(
        "--no-paraphrase",
        action="store_true",
        help="Skip the paraphrase/combination robustness cases.",
    )
    parser.add_argument(
        "--save-results",
        metavar="PATH",
        help="Write per-case and category results to a JSON file.",
    )
    args = parser.parse_args()

    visible_path = config.EVAL_CASES_PATH
    original_path = _PROJECT_ROOT / "eval" / "original-cases.json"
    paraphrase_path = None if args.no_paraphrase else _PROJECT_ROOT / "eval" / "paraphrase-cases.json"

    cases = load_cases(visible_path, original_path, paraphrase_path)

    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]
    if args.case:
        cases = [c for c in cases if c.get("id") == args.case]

    if not cases:
        print("No evaluation cases matched criteria.")
        sys.exit(0)

    if args.baseline:
        from eval.baseline_agent import BaselineAgent

        agent = BaselineAgent()
        print("Mode: BASELINE (naive full-corpus prototype, no RAG / no tools / no guardrails)")
        category_stats, case_results = run_evaluation(cases, agent=agent, verbose=args.verbose)
    else:
        category_stats, case_results = run_evaluation(cases, verbose=args.verbose)
    print_summary_table(category_stats, len(cases))

    if args.save_results:
        out = {
            "mode": "baseline" if args.baseline else "final",
            "total_cases": len(cases),
            "overall_passed": sum(1 for cr in case_results if cr["passed"]),
            "categories": category_stats,
            "cases": case_results,
        }
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Results written to {args.save_results}\n")

    all_passed = all(cr["passed"] for cr in case_results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
