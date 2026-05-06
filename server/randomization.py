"""Per-session condition assignment, urn draw, and session_id generation.

The module-level `random.Random()` is seeded once at import via os.urandom; it is
never reseeded, so condition assignments stay independent across the session.
"""

import os
import random

# A dedicated Random instance so a caller mutating the global random module
# (e.g. in a test) does not affect us, and vice versa.
_rng = random.Random()
_rng.seed(int.from_bytes(os.urandom(16), "big"))

# Excludes I, L, O, V plus the vowels per the spec ("uppercase consonants" with
# I, L, O, V removed).
_CONSONANTS = "BCDFGHJKMNPQRSTWXYZ"

CONDITION_TO_B = {"low": 10, "high": 100}
TRUE_PROB = 0.70
STARTING_CASH = 15.00
MAX_ROUNDS = 25


def new_session_id() -> str:
    return "".join(_rng.choice(_CONSONANTS) for _ in range(4))


def assign_condition() -> str:
    return _rng.choice(["low", "high"])


def draw_urn(true_prob: float = TRUE_PROB) -> str:
    return "red" if _rng.random() < true_prob else "blue"
