"""In-memory store of active TradingSession objects, keyed by session_id."""

from typing import Dict, Optional

from .lmsr_backend import TradingSession


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, TradingSession] = {}
        # Tracks which session_ids have been finalized so we can reject mutations
        # to them with a 409 even after we drop the in-memory state.
        self._finalized: set = set()

    def add(self, session: TradingSession) -> None:
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> Optional[TradingSession]:
        return self._sessions.get(session_id)

    def mark_finalized(self, session_id: str) -> None:
        self._finalized.add(session_id)

    def is_finalized(self, session_id: str) -> bool:
        return session_id in self._finalized


store = SessionStore()
