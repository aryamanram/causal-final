"""HTTP route handlers for the prediction-market app."""

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Response

from . import randomization
from .lmsr_backend import TradingSession
from .models import (
    BeliefRequest,
    ComprehensionRequest,
    CreateSessionResponse,
    FinalizeRequest,
    FinalizeResponse,
    OkResponse,
    PreviewRequest,
    PreviewResponse,
    SurveyRequest,
    TradeRequest,
    TradeResponse,
)
from .persistence import (
    CSV_PATH,
    append_session_to_csv,
    read_session_json,
    rewrite_session_rows_in_csv,
    session_to_dict,
    write_session_json,
)
from .session_store import store

router = APIRouter()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Per-session metadata that lives outside TradingSession (condition string,
# created_at, finalized_at, urn outcome, survey). Keyed by session_id.
_session_meta: Dict[str, Dict[str, Any]] = {}


def _persist(session: TradingSession) -> None:
    meta = _session_meta[session.session_id]
    snapshot = session_to_dict(
        session,
        condition=meta["condition"],
        true_prob=meta["true_prob"],
        created_at=meta["created_at"],
        finalized_at=meta.get("finalized_at"),
        draw_outcome=meta.get("draw_outcome"),
        final_cash=meta.get("final_cash"),
        survey=meta.get("survey"),
        comprehension_attempts=meta.get("comprehension_attempts"),
        initial_belief_pct=meta.get("initial_belief_pct"),
    )
    write_session_json(snapshot)


def _require_session(session_id: str) -> TradingSession:
    if store.is_finalized(session_id):
        raise HTTPException(status_code=409, detail="session already finalized")
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session


@router.post("/api/session", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    session_id = randomization.new_session_id()
    # Tiny chance of collision; loop until unique.
    while store.get(session_id) is not None or _session_meta.get(session_id):
        session_id = randomization.new_session_id()

    condition = randomization.assign_condition()
    b_value = randomization.CONDITION_TO_B[condition]
    created_at = _now_iso()

    session = TradingSession(
        session_id=session_id,
        b=float(b_value),
        starting_cash=randomization.STARTING_CASH,
        max_rounds=randomization.MAX_ROUNDS,
    )
    store.add(session)
    _session_meta[session_id] = {
        "condition": condition,
        "true_prob": randomization.TRUE_PROB,
        "created_at": created_at,
    }
    _persist(session)

    return CreateSessionResponse(
        session_id=session_id,
        condition=condition,  # included for replay/debug; the frontend MUST NOT display it
        b_value=b_value,
        true_prob=randomization.TRUE_PROB,
        starting_cash=randomization.STARTING_CASH,
        max_rounds=randomization.MAX_ROUNDS,
        created_at=created_at,
    )


@router.post("/api/comprehension", response_model=OkResponse)
def comprehension(req: ComprehensionRequest) -> OkResponse:
    """Records how many attempts the participant made on the comprehension check
    (CHANGES.md #3). Called once, after the participant answers correctly."""
    session = _require_session(req.session_id)
    _session_meta[req.session_id]["comprehension_attempts"] = req.attempts
    _persist(session)
    return OkResponse()


@router.post("/api/belief", response_model=OkResponse)
def belief(req: BeliefRequest) -> OkResponse:
    """Records the participant's pre-trial belief P(red) in percent (CHANGES.md #4).
    Called once, immediately before round 1."""
    session = _require_session(req.session_id)
    _session_meta[req.session_id]["initial_belief_pct"] = req.initial_belief_pct
    _persist(session)
    return OkResponse()


@router.post("/api/preview", response_model=PreviewResponse)
def preview(req: PreviewRequest) -> PreviewResponse:
    session = _require_session(req.session_id)
    cost, price_after, cash_after, ok = session.preview(req.action, req.shares)
    return PreviewResponse(
        cost=cost,
        price_after_preview=price_after,
        cash_after_preview=cash_after,
        would_succeed=ok,
    )


def _do_trade(req: TradeRequest) -> TradeResponse:
    session = _require_session(req.session_id)

    # Idempotency: if we've already processed this request_id, return the result
    # of that trade. Checked BEFORE is_complete() so a retry of round 25 still
    # short-circuits cleanly instead of raising "rounds exhausted".
    if req.request_id in session.processed_request_ids:
        last = session.trades[-1]
        return TradeResponse(
            round_just_completed=last.round,
            next_round=None if session.is_complete() else session.current_round,
            current_price=session.current_price_cents(),
            current_cash=round(session.cash, 2),
            q_yes=session.q_yes,
            q_no=session.q_no,
            session_complete=session.is_complete(),
        )

    if session.is_complete():
        raise HTTPException(status_code=409, detail="session rounds exhausted")

    try:
        record = session.execute_trade(
            action=req.action,
            shares=req.shares,
            client_ms_on_round=req.client_ms_on_round,
            server_ts=_now_iso(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session.processed_request_ids.add(req.request_id)
    _persist(session)

    return TradeResponse(
        round_just_completed=record.round,
        next_round=None if session.is_complete() else session.current_round,
        current_price=session.current_price_cents(),
        current_cash=round(session.cash, 2),
        q_yes=session.q_yes,
        q_no=session.q_no,
        session_complete=session.is_complete(),
    )


@router.post("/api/trade", response_model=TradeResponse)
def trade(req: TradeRequest) -> TradeResponse:
    if req.action == "pass":
        raise HTTPException(status_code=400, detail="use /api/pass for pass actions")
    if req.shares < 1:
        raise HTTPException(status_code=400, detail="shares must be >= 1 for a buy")
    return _do_trade(req)


@router.post("/api/pass", response_model=TradeResponse)
def pass_round(req: TradeRequest) -> TradeResponse:
    # Force action/shares to canonical pass values regardless of what the client
    # sent — this endpoint only ever records passes.
    forced = TradeRequest(
        session_id=req.session_id,
        action="pass",
        shares=0,
        client_ms_on_round=req.client_ms_on_round,
        request_id=req.request_id,
    )
    return _do_trade(forced)


@router.post("/api/finalize", response_model=FinalizeResponse)
def finalize(req: FinalizeRequest) -> FinalizeResponse:
    session = _require_session(req.session_id)
    if not session.is_complete():
        raise HTTPException(
            status_code=409,
            detail=f"session has {len(session.trades)}/{session.max_rounds} rounds played",
        )

    meta = _session_meta[session.session_id]
    draw = randomization.draw_urn(meta["true_prob"])
    yes_payout = round(session.q_yes * 1.00, 2) if draw == "red" else 0.00
    no_payout = round(session.q_no * 1.00, 2) if draw == "blue" else 0.00
    cash_remaining = round(session.cash, 2)
    final_cash = round(cash_remaining + yes_payout + no_payout, 2)
    final_price = session.current_price_cents()
    total_pnl = round(final_cash - session.starting_cash, 2)

    meta["draw_outcome"] = draw
    meta["final_cash"] = final_cash
    meta["finalized_at"] = _now_iso()

    # Persist the JSON with finalize fields, then append flattened rows.
    _persist(session)
    snapshot = read_session_json(session.session_id)
    if snapshot is not None:
        append_session_to_csv(snapshot)

    store.mark_finalized(session.session_id)

    return FinalizeResponse(
        draw_outcome=draw,
        final_price=final_price,
        yes_shares=session.q_yes,
        no_shares=session.q_no,
        yes_payout=yes_payout,
        no_payout=no_payout,
        cash_remaining=cash_remaining,
        final_cash=final_cash,
        total_pnl=total_pnl,
    )


@router.post("/api/survey", response_model=OkResponse)
def survey(req: SurveyRequest) -> OkResponse:
    snapshot = read_session_json(req.session_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="session not found")
    if snapshot.get("finalized_at") is None:
        raise HTTPException(status_code=409, detail="session not finalized yet")

    snapshot["survey"] = {
        "risk_tolerance": req.risk_tolerance,
        "trading_experience_months": req.trading_experience_months,
        "prediction_market_familiarity": req.prediction_market_familiarity,
        "field_of_study": req.field_of_study,
        "final_belief_pct": req.final_belief_pct,   # CHANGES #15
        "submitted_at": _now_iso(),
    }
    write_session_json(snapshot)
    rewrite_session_rows_in_csv(snapshot)

    # Also update the in-memory meta in case the session is still around.
    if req.session_id in _session_meta:
        _session_meta[req.session_id]["survey"] = snapshot["survey"]

    return OkResponse()


@router.get("/api/export")
def export_csv() -> Response:
    if not CSV_PATH.exists():
        return Response(content="", media_type="text/csv")
    with CSV_PATH.open("rb") as f:
        body = f.read()
    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="all_sessions.csv"'},
    )
