#!/usr/bin/env python3
"""
generate_template.py — Generate the blank comparison_template.csv for head-to-head scoring.

Writes one row per BenchmarkClaim with claim metadata pre-filled and all tool verdict
columns left blank. Run this once to create the template; then fill in tool columns
manually after querying Elicit, Consensus, and Semantic Scholar.

Usage:
    python benchmarks/comparison/generate_template.py
"""

import csv
import pathlib
import sys

# Resolve project root so the script works from any cwd.
_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

from benchmarks.benchmark_claims import BENCHMARK_CLAIMS

COLUMNS = [
    "claim_id",
    "tier",
    "claim",
    "expected_verdict",
    "failure_mode",
    "scoring_notes",
    # Elicit
    "elicit_response",
    "elicit_verdict",
    "elicit_correct",
    # Consensus
    "consensus_response",
    "consensus_verdict",
    "consensus_correct",
    # Semantic Scholar
    "semantic_scholar_response",
    "semantic_scholar_verdict",
    "semantic_scholar_correct",
]

OUTPUT_PATH = pathlib.Path(__file__).parent / "comparison_template.csv"


def main() -> None:
    rows = []
    for bc in BENCHMARK_CLAIMS:
        rows.append({
            "claim_id":         bc.claim_id,
            "tier":             bc.tier.value,
            "claim":            bc.claim,
            "expected_verdict": bc.expected_verdict.value,
            "failure_mode":     bc.failure_mode,
            "scoring_notes":    bc.scoring_notes,
            # Tool columns — left blank for manual entry
            "elicit_response":           "",
            "elicit_verdict":            "",
            "elicit_correct":            "",
            "consensus_response":        "",
            "consensus_verdict":         "",
            "consensus_correct":         "",
            "semantic_scholar_response": "",
            "semantic_scholar_verdict":  "",
            "semantic_scholar_correct":  "",
        })

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
