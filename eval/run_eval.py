"""
Runs every scenario in eval/scenarios.py against the agent and prints a
pass-rate breakdown by category, then writes results.json.

Run with:  python -m eval.run_eval

These are the real numbers that should go into a resume bullet — not
estimates, not placeholders. Re-run after every meaningful change to
agent/graph.py or agent/tools.py.
"""

from __future__ import annotations

import json
from collections import defaultdict

from agent.graph import run_agent
from eval.scenarios import SCENARIOS


def run_all():
    results = []
    by_category = defaultdict(lambda: {"pass": 0, "total": 0})

    for scenario in SCENARIOS:
        state = run_agent(user_text=scenario.user_text, is_driving=scenario.is_driving)
        passed = bool(scenario.expected(state))

        by_category[scenario.category]["total"] += 1
        if passed:
            by_category[scenario.category]["pass"] += 1

        results.append(
            {
                "id": scenario.id,
                "category": scenario.category,
                "user_text": scenario.user_text,
                "is_driving": scenario.is_driving,
                "expected_description": scenario.expected_description,
                "passed": passed,
                "intent": state.get("intent"),
                "safety_allowed": state.get("safety_allowed"),
                "needs_confirmation": state.get("needs_confirmation"),
                "response": state.get("response"),
            }
        )

    return results, by_category


def print_report(results, by_category):
    print("\n=== Scenario results ===")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']:10s} ({r['category']:10s}) \"{r['user_text']}\"")
        if not r["passed"]:
            print(f"          expected: {r['expected_description']}")
            print(f"          got: intent={r['intent']} safety_allowed={r['safety_allowed']}")

    print("\n=== Pass rate by category ===")
    total_pass, total_count = 0, 0
    for category, counts in by_category.items():
        p, t = counts["pass"], counts["total"]
        total_pass += p
        total_count += t
        print(f"{category:12s}: {p}/{t} ({100 * p / t:.1f}%)")

    print(f"\nOverall: {total_pass}/{total_count} ({100 * total_pass / total_count:.1f}%)")


if __name__ == "__main__":
    results, by_category = run_all()
    print_report(results, by_category)

    with open("eval/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results written to eval/results.json")
