"""Atomic JSON-per-session writes and CSV append/rewrite for all_sessions.csv.

A single lock file guards CSV reads/writes so concurrent finalize/survey calls
do not interleave; per-session JSON files have unique names so they don't need a
shared lock.
"""

import csv
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .lmsr_backend import TradingSession

# Resolve data dir relative to the project root (one level above server/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
CSV_PATH = DATA_DIR / "all_sessions.csv"
LOCK_PATH = DATA_DIR / ".lock"

CSV_COLUMNS = [
    "session_id",
    "condition",
    "b_value",
    "true_prob",
    "round",
    "action",
    "shares",
    "price_before",
    "price_after",
    "cost",
    "cash_before",
    "cash_after",
    "q_yes_after",
    "q_no_after",
    "client_ms_on_round",
    "timestamp_utc",                # CHANGES #8 — server-side ISO 8601 per row
    "draw_outcome",
    "final_cash",
    "session_completed",            # CHANGES #12 — True iff finalized_at is non-null
    "comprehension_attempts",       # CHANGES #3 — how many tries before passing the comprehension check
    "initial_belief_pct",           # CHANGES #4 — pre-trial belief P(red) in percent
    "final_belief_pct",             # CHANGES #15 — post-session belief P(red), pairs with initial_belief_pct
    "risk_tolerance",
    "trading_experience_months",
    "prediction_market_familiarity",
    "field_of_study",
]


def _ensure_dirs() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def _csv_lock():
    """Coarse cross-process file lock around CSV writes via O_EXCL on a lockfile."""
    _ensure_dirs()
    while True:
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            # Tiny busy-wait; in practice contention is essentially zero (one
            # serial participant at a time).
            import time
            time.sleep(0.005)
    try:
        yield
    finally:
        try:
            os.unlink(LOCK_PATH)
        except FileNotFoundError:
            pass


def _atomic_write(path: Path, data: str) -> None:
    """Write `data` to `path` atomically: write to .tmp, fsync, rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def session_to_dict(
    session: TradingSession,
    *,
    condition: str,
    true_prob: float,
    created_at: str,
    finalized_at: Optional[str] = None,
    draw_outcome: Optional[str] = None,
    final_cash: Optional[float] = None,
    survey: Optional[Dict[str, Any]] = None,
    comprehension_attempts: Optional[int] = None,
    initial_belief_pct: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "session_id": session.session_id,
        "condition": condition,
        "b_value": int(session.b),
        "true_prob": true_prob,
        "starting_cash": session.starting_cash,
        "max_rounds": session.max_rounds,
        "created_at": created_at,
        "finalized_at": finalized_at,
        "draw_outcome": draw_outcome,
        "final_cash": final_cash,
        "comprehension_attempts": comprehension_attempts,
        "initial_belief_pct": initial_belief_pct,
        "trades": [asdict(t) for t in session.trades],
        "survey": survey,
    }


def write_session_json(session_dict: Dict[str, Any]) -> Path:
    _ensure_dirs()
    path = SESSIONS_DIR / f"{session_dict['session_id']}.json"
    _atomic_write(path, json.dumps(session_dict, indent=2))
    return path


def read_session_json(session_id: str) -> Optional[Dict[str, Any]]:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _row_for_trade(session: Dict[str, Any], trade: Dict[str, Any]) -> Dict[str, Any]:
    survey = session.get("survey") or {}
    return {
        "session_id": session["session_id"],
        "condition": session["condition"],
        "b_value": session["b_value"],
        "true_prob": session["true_prob"],
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
        "draw_outcome": session.get("draw_outcome"),
        "final_cash": session.get("final_cash"),
        "session_completed": session.get("finalized_at") is not None,
        "comprehension_attempts": session.get("comprehension_attempts"),
        "initial_belief_pct": session.get("initial_belief_pct"),
        "final_belief_pct": survey.get("final_belief_pct"),
        "risk_tolerance": survey.get("risk_tolerance"),
        "trading_experience_months": survey.get("trading_experience_months"),
        "prediction_market_familiarity": survey.get("prediction_market_familiarity"),
        "field_of_study": survey.get("field_of_study"),
    }


def append_session_to_csv(session_dict: Dict[str, Any]) -> None:
    """Append all rows of a freshly-finalized session to all_sessions.csv."""
    _ensure_dirs()
    rows = [_row_for_trade(session_dict, t) for t in session_dict["trades"]]
    with _csv_lock():
        new_file = not CSV_PATH.exists()
        with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if new_file:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)


def rewrite_session_rows_in_csv(session_dict: Dict[str, Any]) -> None:
    """Replace all rows for `session_id` in all_sessions.csv (used after survey)."""
    _ensure_dirs()
    sid = session_dict["session_id"]
    with _csv_lock():
        if not CSV_PATH.exists():
            # Nothing to rewrite; just append fresh rows.
            new_rows = [_row_for_trade(session_dict, t) for t in session_dict["trades"]]
            with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                for row in new_rows:
                    writer.writerow(row)
            return

        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            kept: List[Dict[str, str]] = [r for r in reader if r["session_id"] != sid]

        new_rows = [_row_for_trade(session_dict, t) for t in session_dict["trades"]]
        # Cast values to str for the kept rows (DictReader already gives us str);
        # DictWriter writes both fine.
        tmp_path = CSV_PATH.with_suffix(".csv.tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for r in kept:
                writer.writerow(r)
            for r in new_rows:
                writer.writerow(r)
        os.replace(tmp_path, CSV_PATH)
