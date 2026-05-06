"""Regenerate the canonical all_sessions.csv from session_data/*.json.

Each contributor writes per-session JSONs to session_data/ and per-session
CSVs to sessions_csv/. The per-session files are tracked in git and never
collide (filenames are 4-char unique session IDs). This script walks every
JSON in session_data/, filters to finalized sessions, and emits one merged
CSV at data/all_sessions.csv ready for the NumPyro analysis pipeline.

The JSONs are the source of truth — sessions_csv/ is a redundant view for
quick inspection. If sessions_csv/<sid>.csv ever disagrees with the JSON,
trust the JSON; this script reads only JSONs.

Usage:
    python scripts/merge_data.py            # writes data/all_sessions.csv
    python scripts/merge_data.py --dry-run  # prints what would happen, no write
    python scripts/merge_data.py --include-unfinalized
                                            # also include sessions whose
                                            # finalized_at is null (sets
                                            # session_completed=False); off by
                                            # default since unfinalized
                                            # sessions are usually abandoned
                                            # and their final_cash is null
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Resolve project root the same way persistence.py does (one level up).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = _PROJECT_ROOT / "session_data"
OUT_PATH = _PROJECT_ROOT / "data" / "all_sessions.csv"

CSV_COLUMNS = [
    "session_id", "condition", "b_value", "true_prob",
    "round", "action", "shares",
    "price_before", "price_after", "cost",
    "cash_before", "cash_after",
    "q_yes_after", "q_no_after",
    "client_ms_on_round", "timestamp_utc",
    "draw_outcome", "final_cash", "session_completed",
    "comprehension_attempts", "initial_belief_pct", "final_belief_pct",
    "risk_tolerance", "trading_experience_months",
    "prediction_market_familiarity", "field_of_study",
]


def _row_for_trade(snap, trade):
    survey = snap.get("survey") or {}
    return {
        "session_id": snap["session_id"],
        "condition": snap["condition"],
        "b_value": snap["b_value"],
        "true_prob": snap["true_prob"],
        "round": trade["round"],
        "action": trade["action"],
        "shares": trade["shares"],
        "price_before": trade["price_before"],
        "price_after": trade["price_after"],
        "cost": trade["cost"],
        "cash_before": trade["cash_before"],
        "cash_after": trade["cash_after"],
        "q_yes_after": trade["q_yes_after"],
        "q_no_after": trade["q_no_after"],
        "client_ms_on_round": trade["client_ms_on_round"],
        "timestamp_utc": trade.get("server_ts"),
        "draw_outcome": snap.get("draw_outcome"),
        "final_cash": snap.get("final_cash"),
        "session_completed": snap.get("finalized_at") is not None,
        "comprehension_attempts": snap.get("comprehension_attempts"),
        "initial_belief_pct": snap.get("initial_belief_pct"),
        "final_belief_pct": survey.get("final_belief_pct"),
        "risk_tolerance": survey.get("risk_tolerance"),
        "trading_experience_months": survey.get("trading_experience_months"),
        "prediction_market_familiarity": survey.get("prediction_market_familiarity"),
        "field_of_study": survey.get("field_of_study"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="don't write, just report")
    parser.add_argument("--include-unfinalized", action="store_true",
                        help="include sessions with finalized_at == null")
    args = parser.parse_args()

    if not SESSIONS_DIR.exists():
        print(f"error: {SESSIONS_DIR} does not exist", file=sys.stderr)
        return 1

    json_paths = sorted(SESSIONS_DIR.glob("*.json"))
    if not json_paths:
        print(f"warning: no JSON files in {SESSIONS_DIR}", file=sys.stderr)

    snapshots = []
    skipped_unfinalized = []
    skipped_unsurveyed = []
    for p in json_paths:
        try:
            snap = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"skip (bad JSON): {p.name}: {e}", file=sys.stderr)
            continue
        if snap.get("finalized_at") is None and not args.include_unfinalized:
            skipped_unfinalized.append(snap.get("session_id", p.stem))
            continue
        if snap.get("survey") is None and not args.include_unfinalized:
            skipped_unsurveyed.append(snap.get("session_id", p.stem))
            continue
        snapshots.append(snap)

    # Deterministic order: by finalized_at, falling back to created_at, then sid.
    snapshots.sort(key=lambda s: (
        s.get("finalized_at") or s.get("created_at") or "",
        s["session_id"],
    ))

    rows = []
    for snap in snapshots:
        for trade in snap.get("trades", []):
            rows.append(_row_for_trade(snap, trade))

    print(f"sessions found:        {len(json_paths)}")
    print(f"sessions included:     {len(snapshots)}")
    print(f"sessions skipped (no finalize): {len(skipped_unfinalized)} {skipped_unfinalized}")
    print(f"sessions skipped (no survey):   {len(skipped_unsurveyed)} {skipped_unsurveyed}")
    print(f"total rows:            {len(rows)}")
    print(f"output:                {OUT_PATH}")

    if args.dry_run:
        print("(dry run — no file written)")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {len(rows)} rows to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
