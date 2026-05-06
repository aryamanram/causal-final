"""Pydantic request/response schemas. Field names mirror the on-disk JSON keys."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class CreateSessionResponse(BaseModel):
    session_id: str
    condition: Literal["low", "high"]
    b_value: int
    true_prob: float
    starting_cash: float
    max_rounds: int
    created_at: str


class ComprehensionRequest(BaseModel):
    session_id: str
    attempts: int = Field(ge=1)


class BeliefRequest(BaseModel):
    session_id: str
    initial_belief_pct: int = Field(ge=0, le=100)


class PreviewRequest(BaseModel):
    session_id: str
    action: Literal["buy_yes", "buy_no"]
    shares: int = Field(ge=1, le=10)


class PreviewResponse(BaseModel):
    cost: float
    price_after_preview: float
    cash_after_preview: float
    would_succeed: bool


class TradeRequest(BaseModel):
    session_id: str
    action: Literal["buy_yes", "buy_no", "pass"]
    shares: int = Field(ge=0, le=10)
    client_ms_on_round: int = Field(ge=0)
    request_id: str  # client UUID for idempotency


class TradeResponse(BaseModel):
    round_just_completed: int
    next_round: Optional[int]
    current_price: float
    current_cash: float
    q_yes: int
    q_no: int
    session_complete: bool


class FinalizeRequest(BaseModel):
    session_id: str


class FinalizeResponse(BaseModel):
    draw_outcome: Literal["red", "blue"]
    final_price: float
    yes_shares: int
    no_shares: int
    yes_payout: float
    no_payout: float
    cash_remaining: float
    final_cash: float
    total_pnl: float


class SurveyRequest(BaseModel):
    session_id: str
    risk_tolerance: int = Field(ge=1, le=5)
    trading_experience_months: int = Field(ge=0)
    prediction_market_familiarity: bool
    field_of_study: str = Field(max_length=100)
    final_belief_pct: int = Field(ge=0, le=100)  # CHANGES #15 — post-session belief, pairs with initial_belief_pct


class OkResponse(BaseModel):
    ok: bool = True
