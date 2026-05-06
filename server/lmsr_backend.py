"""LMSR (Logarithmic Market Scoring Rule) market maker for a binary outcome.

Cost function:  C(q_yes, q_no) = b * log(exp(q_yes/b) + exp(q_no/b))
Yes price:      p_yes = exp(q_yes/b) / (exp(q_yes/b) + exp(q_no/b))

Trade cost (dollars) for buying delta_yes Yes shares and delta_no No shares
is C(q_yes + delta_yes, q_no + delta_no) - C(q_yes, q_no), where prices are
expressed as dollars in [0, 1] (we display as cents in [0, 100]).
"""

import math
from dataclasses import dataclass, field
from typing import List, Literal, Tuple


def _logsumexp(a: float, b: float) -> float:
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class LMSR:
    def __init__(self, b: float):
        if b <= 0:
            raise ValueError("b must be positive")
        self.b = float(b)

    def cost(self, q_yes: float, q_no: float) -> float:
        return self.b * _logsumexp(q_yes / self.b, q_no / self.b)

    def price_yes(self, q_yes: float, q_no: float) -> float:
        # Numerically-stable softmax for two terms.
        m = max(q_yes, q_no) / self.b
        e_yes = math.exp(q_yes / self.b - m)
        e_no = math.exp(q_no / self.b - m)
        return e_yes / (e_yes + e_no)

    def trade_cost(
        self,
        q_yes: float,
        q_no: float,
        delta_yes: float,
        delta_no: float,
    ) -> float:
        return self.cost(q_yes + delta_yes, q_no + delta_no) - self.cost(q_yes, q_no)


@dataclass
class TradeRecord:
    round: int
    action: Literal["buy_yes", "buy_no", "pass"]
    shares: int
    price_before: float        # cents (0..100), 1 decimal
    price_after: float
    cost: float                # dollars
    cash_before: float
    cash_after: float
    q_yes_after: int
    q_no_after: int
    client_ms_on_round: int
    server_ts: str


@dataclass
class TradingSession:
    """In-memory state for a single participant's trading session."""

    session_id: str
    b: float
    starting_cash: float = 15.00
    max_rounds: int = 25

    q_yes: int = 0
    q_no: int = 0
    cash: float = 15.00
    current_round: int = 1
    trades: List[TradeRecord] = field(default_factory=list)

    # Tracks request_ids that have been processed for idempotency.
    processed_request_ids: set = field(default_factory=set)

    def __post_init__(self):
        self._lmsr = LMSR(self.b)
        self.cash = self.starting_cash

    @property
    def lmsr(self) -> LMSR:
        return self._lmsr

    def current_price_cents(self) -> float:
        """Yes-share price in cents (0..100), rounded to 1 decimal."""
        return round(self._lmsr.price_yes(self.q_yes, self.q_no) * 100.0, 1)

    def is_complete(self) -> bool:
        return len(self.trades) >= self.max_rounds

    def preview(
        self, action: Literal["buy_yes", "buy_no"], shares: int
    ) -> Tuple[float, float, float, bool]:
        """Returns (cost_dollars, price_after_cents, cash_after, would_succeed)."""
        if shares <= 0:
            return 0.0, self.current_price_cents(), self.cash, False
        delta_yes = shares if action == "buy_yes" else 0
        delta_no = shares if action == "buy_no" else 0
        cost = self._lmsr.trade_cost(self.q_yes, self.q_no, delta_yes, delta_no)
        new_q_yes = self.q_yes + delta_yes
        new_q_no = self.q_no + delta_no
        new_price = round(self._lmsr.price_yes(new_q_yes, new_q_no) * 100.0, 1)
        cost_r = round(cost, 2)
        new_cash = round(self.cash - cost_r, 2)
        would_succeed = cost_r <= self.cash + 1e-9
        return cost_r, new_price, new_cash, would_succeed

    def execute_trade(
        self,
        action: Literal["buy_yes", "buy_no", "pass"],
        shares: int,
        client_ms_on_round: int,
        server_ts: str,
    ) -> TradeRecord:
        if self.is_complete():
            raise ValueError("session already complete")

        round_idx = self.current_round
        price_before = self.current_price_cents()
        cash_before = round(self.cash, 2)

        if action == "pass":
            cost = 0.0
            price_after = price_before
        else:
            if shares < 1 or shares > 10:
                raise ValueError("shares must be in 1..10 for a buy")
            delta_yes = shares if action == "buy_yes" else 0
            delta_no = shares if action == "buy_no" else 0
            cost = self._lmsr.trade_cost(self.q_yes, self.q_no, delta_yes, delta_no)
            cost = round(cost, 2)
            if cost > self.cash + 1e-9:
                raise ValueError("insufficient cash")
            self.q_yes += delta_yes
            self.q_no += delta_no
            self.cash = round(self.cash - cost, 2)
            price_after = self.current_price_cents()

        record = TradeRecord(
            round=round_idx,
            action=action,
            shares=shares if action != "pass" else 0,
            price_before=price_before,
            price_after=price_after,
            cost=cost,
            cash_before=cash_before,
            cash_after=round(self.cash, 2),
            q_yes_after=self.q_yes,
            q_no_after=self.q_no,
            client_ms_on_round=client_ms_on_round,
            server_ts=server_ts,
        )
        self.trades.append(record)
        self.current_round += 1
        return record
