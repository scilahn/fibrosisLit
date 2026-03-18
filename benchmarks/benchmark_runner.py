#!/usr/bin/env python3
"""
benchmark_runner.py — Run all benchmark claims through claim_evaluator and report results.

Iterates over BENCHMARK_CLAIMS, calls evaluate_claim() on each, compares the
evaluator's tier against the benchmark's expected verdict, writes structured
results to benchmarks/results/, and prints a formatted summary table with
per-tier accuracy.

This runner evaluates prior-based claim support only — it does not retrieve
papers. The verdict reflects how well each claim aligns with IPF domain priors
encoded in fibrosis_priors.py. Paper-level evidence evaluation (evidence_quality.py)
is a separate pipeline (scripts/search_eval.py).

Output files:
    results_{timestamp}.json — full audit: all scores, rationale, contested positions
    results_{timestamp}.csv  — flat summary table for analysis

Known accuracy ceiling: ~76% (19/25) due to four claim types that require
clinical trial outcome knowledge or model-validity semantics beyond what
pattern-based priors can detect. See claim_evaluator.py docstring.
"""

import csv
import json
import logging
import pathlib
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from benchmarks.benchmark_claims import BENCHMARK_CLAIMS, BenchmarkClaim, ExpectedVerdict
from evaluators.claim_evaluator import evaluate_claim, ClaimEvaluationResult
from evaluators.contradiction_detector import ContestedFlag

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# Map claim_evaluator tier → benchmark verdict vocabulary.
# OVERCLAIMED maps to UNSUPPORTED; INSUFFICIENT_EVIDENCE is handled in _is_correct().
TIER_TO_VERDICT: dict[str, str] = {
    "WELL_SUPPORTED": "SUPPORTED",
    "CONTESTED":      "CONTESTED",
    "OVERCLAIMED":    "UNSUPPORTED",
}

CSV_COLUMNS: list[str] = [
    "claim_id", "tier", "claim", "expected_verdict", "actual_verdict", "correct",
    "pathway_support_score", "model_penalty", "prior_support_score",
    "contested_flags", "model_warnings",
    "llm_verdict", "verdict", "verdict_confidence",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_correct(actual_tier: str, expected_verdict: ExpectedVerdict) -> bool:
    """
    Return True when the evaluator's tier matches the benchmark's expected verdict.

    INSUFFICIENT_EVIDENCE is treated as equivalent to UNSUPPORTED — both indicate
    the claim lacks sufficient support, and both are produced by the OVERCLAIMED tier.
    This avoids penalizing correct OVERCLAIMED classifications on the two OC claims
    whose benchmark verdict is INSUFFICIENT_EVIDENCE rather than UNSUPPORTED.
    """
    actual_verdict = TIER_TO_VERDICT.get(actual_tier, "UNKNOWN")
    if expected_verdict in (ExpectedVerdict.UNSUPPORTED, ExpectedVerdict.INSUFFICIENT_EVIDENCE):
        return actual_verdict == "UNSUPPORTED"
    return actual_verdict == expected_verdict.value


def _flag_to_dict(flag: ContestedFlag) -> dict:
    """Serialize a ContestedFlag to a JSON-safe dict, preserving all positions."""
    return {
        "debate_name":      flag.debate_name,
        "matched_snippet":  flag.matched_snippet,
        "debate":           flag.debate,
        "positions":        flag.positions,
    }


def _result_to_json_row(
    bc: BenchmarkClaim,
    result: ClaimEvaluationResult,
    correct: bool,
) -> dict:
    """Build the full-detail JSON row for a single claim."""
    return {
        "claim_id":              bc.claim_id,
        "tier":                  bc.tier.value,
        "claim":                 result.claim,
        "expected_verdict":      bc.expected_verdict.value,
        "actual_verdict":        TIER_TO_VERDICT.get(result.tier, "UNKNOWN"),
        "correct":               correct,
        "pathway_support_score": result.pathway_support_score,
        "model_penalty":         result.model_penalty,
        "prior_support_score":   result.prior_support_score,
        "detected_pathways":     result.detected_pathways,
        "detected_model_mentions": result.detected_model_mentions,
        "contested_flags":       [_flag_to_dict(f) for f in result.contested_flags],
        "warnings":              result.warnings,
        "rationale":             result.rationale,
        # Two-vote fields (present when LLM vote ran)
        "llm_verdict":           result.llm_verdict,
        "llm_confidence":        result.llm_confidence,
        "llm_reasoning":         result.llm_reasoning,
        "verdict":               result.verdict,
        "verdict_confidence":    result.verdict_confidence,
        "verdict_rationale":     result.verdict_rationale,
        "references":            result.references,
    }


def _result_to_csv_row(
    bc: BenchmarkClaim,
    result: ClaimEvaluationResult,
    correct: bool,
) -> dict:
    """Build the flat CSV row for a single claim."""
    return {
        "claim_id":              bc.claim_id,
        "tier":                  bc.tier.value,
        "claim":                 result.claim,
        "expected_verdict":      bc.expected_verdict.value,
        "actual_verdict":        TIER_TO_VERDICT.get(result.tier, "UNKNOWN"),
        "correct":               correct,
        "pathway_support_score": f"{result.pathway_support_score:.3f}",
        "model_penalty":         f"{result.model_penalty:.3f}",
        "prior_support_score":   f"{result.prior_support_score:.3f}",
        "contested_flags":       ";".join(f.debate_name for f in result.contested_flags),
        "model_warnings":        ";".join(result.warnings),
        "llm_verdict":           result.llm_verdict or "N/A",
        "verdict":               result.verdict,
        "verdict_confidence":    result.verdict_confidence,
    }


def _print_summary_table(rows: list[dict]) -> None:
    """Print a formatted results table to stdout."""
    W_ID, W_TIER, W_EXP, W_ACT, W_OK, W_SCORE, W_CLAIM = 6, 14, 22, 14, 3, 7, 55

    header = (
        f"{'ID':<{W_ID}} {'Tier':<{W_TIER}} {'Expected':<{W_EXP}} "
        f"{'Actual':<{W_ACT}} {'OK':>{W_OK}} {'Score':>{W_SCORE}}  Claim"
    )
    sep = "-" * (W_ID + 1 + W_TIER + 1 + W_EXP + 1 + W_ACT + 1 + W_OK + 1 + W_SCORE + 2 + W_CLAIM)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in rows:
        claim_str = r["claim"]
        claim_trunc = claim_str[:W_CLAIM - 1] + "…" if len(claim_str) > W_CLAIM else claim_str
        ok_mark = "✓" if r["correct"] else "✗"
        score = float(r["prior_support_score"])
        print(
            f"{r['claim_id']:<{W_ID}} {r['tier']:<{W_TIER}} {r['expected_verdict']:<{W_EXP}} "
            f"{r['actual_verdict']:<{W_ACT}} {ok_mark:>{W_OK}} "
            f"{score:>{W_SCORE}.3f}  {claim_trunc}"
        )
    print(sep)


def _print_accuracy_summary(rows: list[dict]) -> None:
    """Print overall and per-tier accuracy."""
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])

    tier_counts: dict[str, dict[str, int]] = {}
    for r in rows:
        t = r["tier"]
        if t not in tier_counts:
            tier_counts[t] = {"correct": 0, "total": 0}
        tier_counts[t]["total"] += 1
        if r["correct"]:
            tier_counts[t]["correct"] += 1

    pct = 100 * correct / total if total else 0.0
    print(f"\n=== Accuracy Summary ===")
    print(f"Overall: {correct}/{total} ({pct:.1f}%)")
    for tier, counts in sorted(tier_counts.items()):
        c, n = counts["correct"], counts["total"]
        tier_pct = 100 * c / n if n else 0.0
        print(f"  {tier:<14}: {c}/{n} ({tier_pct:.1f}%)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(
    disease: str = "ipf",
    output_dir: pathlib.Path = RESULTS_DIR,
) -> list[dict]:
    """
    Run all benchmark claims through evaluate_claim and write results.

    Iterates BENCHMARK_CLAIMS in order, calls evaluate_claim() on each claim
    text, compares the returned tier against the benchmark expected_verdict,
    and accumulates rows for JSON and CSV output.

    Exceptions per claim are logged and a placeholder error row is written so
    the output files always contain exactly len(BENCHMARK_CLAIMS) entries.

    Args:
        disease:    Disease context passed through to evaluate_claim().
        output_dir: Directory for output files (created if absent).

    Returns:
        List of CSV-format row dicts (one per claim), suitable for display.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"results_{timestamp}.json"
    csv_path  = output_dir / f"results_{timestamp}.csv"

    total = len(BENCHMARK_CLAIMS)
    json_rows: list[dict] = []
    csv_rows:  list[dict] = []

    for i, bc in enumerate(BENCHMARK_CLAIMS, start=1):
        print(f"Running {bc.claim_id} ({i}/{total})…", flush=True)
        try:
            result = evaluate_claim(bc.claim, disease=disease)
            correct = _is_correct(result.tier, bc.expected_verdict)
            json_rows.append(_result_to_json_row(bc, result, correct))
            csv_rows.append(_result_to_csv_row(bc, result, correct))
        except Exception as exc:
            logger.error("Error evaluating %s: %s", bc.claim_id, exc, exc_info=True)
            error_row = {col: "ERROR" for col in CSV_COLUMNS}
            error_row["claim_id"] = bc.claim_id
            error_row["tier"]     = bc.tier.value
            error_row["claim"]    = bc.claim
            error_row["correct"]  = False
            json_rows.append({"claim_id": bc.claim_id, "error": str(exc)})
            csv_rows.append(error_row)

    # Write JSON
    payload = {
        "run_at":  timestamp,
        "disease": disease,
        "total":   total,
        "results": json_rows,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nResults written to {json_path}")
    print(f"CSV written to     {csv_path}")

    return csv_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    rows = run()
    _print_summary_table(rows)
    _print_accuracy_summary(rows)


if __name__ == "__main__":
    main()
