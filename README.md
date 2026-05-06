# Prediction Market Experimental Web App

Operationalizes the V2 prediction-market prototype (CS 598 Causal Methods for
HCI final report project) as an interactive web application. One participant
per browser session. 25 rounds of LMSR-priced trading against a server-side
market maker, then a brief covariate survey. All data persisted as JSON +
flattened CSV ready for the NumPyro analysis pipeline.

## Quick start (local)

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

Open <http://localhost:8000/> in a modern browser at viewport ≥ 1024px.

For in-person sessions on the same WiFi, other devices can reach
`http://<host-LAN-IP>:8000/`.

## Layout

```
causal-final/
├── server/
│   ├── main.py              ← FastAPI app, static mount
│   ├── routes.py            ← /api/* handlers
│   ├── lmsr_backend.py      ← LMSR market maker + TradingSession
│   ├── models.py            ← Pydantic request/response schemas
│   ├── session_store.py     ← in-memory session dict
│   ├── persistence.py       ← atomic JSON & CSV writers
│   └── randomization.py     ← per-session b assignment, urn draw
├── static/
│   ├── index.html           ← all 5 screens
│   ├── styles.css           ← design tokens, layout
│   ├── app.js               ← state machine, API calls
│   └── chart.umd.js         ← vendored Chart.js 4.4.1
├── session_data/
│   └── <sid>.json           ← per-session JSON, tracked in git
├── sessions_csv/
│   └── <sid>.csv            ← per-session 25-row CSV, tracked in git
├── data/
│   └── all_sessions.csv     ← merged CSV, NumPyro-ready (gitignored)
└── scripts/
    └── merge_data.py        ← rebuild data/all_sessions.csv from session_data/
```

## Data sharing across contributors (manual merge workflow)

Each contributor runs the app on their own machine. Every finalized session
writes:

- `session_data/<sid>.json` — the canonical snapshot.
- `sessions_csv/<sid>.csv` — a 25-row standalone CSV mirror of the JSON.

Both are tracked in git. Because session IDs are unique 4-character codes,
two contributors **never write to the same file**, so commits/pushes don't
conflict. Anyone who pulls the repo sees everyone's sessions.

The merged `data/all_sessions.csv` is **gitignored** — it's a derived artifact
that is regenerated locally from `session_data/` whenever you want a single
file for analysis:

```sh
python scripts/merge_data.py            # writes data/all_sessions.csv
python scripts/merge_data.py --dry-run  # just report, no write
```

The script considers only sessions with `finalized_at` set and a populated
`survey` block (i.e. complete sessions). Pass `--include-unfinalized` if you
need to inspect partial sessions.

**Suggested workflow:**

1. `git pull` before running sessions.
2. Run as many sessions as you like.
3. `git add session_data/ sessions_csv/ && git commit -m "data: <n> sessions" && git push`.
4. At analysis time, `python scripts/merge_data.py` to rebuild
   `data/all_sessions.csv` from the union of everyone's contributions.

## Data export

The full per-round CSV is at `data/all_sessions.csv`. It can also be downloaded
via `GET /api/export`. **The export endpoint is not authenticated** — do not
expose it on a public deployment without adding env-gated auth (see spec §4.7).

## Notes on correctness (spec §8)

- The frontend never displays `b_value` or `condition`. Treatment is invisible
  to the participant; the only legal manifestation is the realized price path.
- All LMSR pricing is server-side. The frontend renders whatever the server
  returns.
- `/api/trade` and `/api/pass` are idempotent: each request carries a
  client-generated `request_id` (UUID); duplicates are short-circuited.
- Per-session JSON is written atomically (write to `.tmp`, fsync, rename) on
  every state change. Already-completed sessions survive server crashes.
- No IP, User-Agent, cookies, or `localStorage` are persisted. The 4-character
  `session_id` is the only identifier.

## Acceptance check (spec §10)

After running a complete session, verify:

```sh
ls data/sessions/                 # one JSON per session
cat data/sessions/<id>.json       # 25 trades, survey object, final_cash
head -1 data/all_sessions.csv     # 21-column header, see spec App. A
```

A session's row count in `all_sessions.csv` is exactly 25 (one per round).

## CHANGES.md additions (May 2026)

The flow now runs **welcome → comprehension check → belief elicitation →
25 rounds → session complete → survey → thank-you**. The CSV schema gained
five columns:

- `comprehension_attempts` (int) — number of tries before the participant
  passed the 2-question payout-structure check.
- `initial_belief_pct` (int 0–100) — pre-trial belief P(red), elicited before
  the participant sees any market price.
- `final_belief_pct` (int 0–100) — post-session belief P(red), asked on the
  survey screen. Pairs with `initial_belief_pct` for a clean before/after
  measurement.
- `timestamp_utc` (ISO 8601) — server-side absolute time per row.
- `session_completed` (bool) — `True` iff the session was finalized (urn drawn,
  payouts computed). For now every CSV row necessarily has this `True` because
  rows are only appended at `/api/finalize`; if you ever back-fill partial
  sessions from `data/sessions/*.json`, those rows will appear as `False`.

The Session Complete screen now shows an explicit wallet decomposition
(`cash_held + share_payouts = final_wallet → −starting_cash = P&L`) so the
arithmetic can be verified by eye, and a persistent payout reminder is on every
round screen.

### Methodology decisions captured in this app (CHANGES P2/P3)

- **#9a dwell-time filter (decision: soft):** `client_ms_on_round` is logged
  per row but no minimum-dwell gate is enforced. In analysis, pre-register a
  filter (e.g., flag rounds where `client_ms_on_round < 1500` as "rapid") and
  apply it before treating the trade as deliberate. No app change.
- **#10 capital × liquidity asymmetry (decision: a — document & accept):**
  starting cash is $15.00 across both conditions. Under b=100 the maximum
  achievable price by spending the full $15 on Yes is ≈57¢, below the 70¢
  ground truth — a real confound that mixes liquidity and capital constraint.
  Acknowledge in the limitations section of the report; the treatment effect
  is still valid, just interpretable as "liquidity + capital constraint".
- **#11 selling shares (decision: deferred):** participants can only buy or
  pass; sells are not implemented. A defensible simplification for a 25-round,
  $15-cap, single-event experiment.
- **#12 mid-session refresh (decision: deferred + flag):** sessions are
  in-memory server-side and not resumable on browser refresh. Per-trade JSON
  snapshots are still persisted, so already-played rounds aren't lost — but
  the participant cannot resume, so the experimenter should restart the
  session if a refresh occurs. The CSV now carries a `session_completed`
  boolean (`True` iff the session reached `/api/finalize`); a populated
  `survey` block in the JSON is the secondary completeness signal.
- **#13 trade-size cap:** capped at 10 shares per trade.
- **#14 running P&L during session:** intentionally not displayed.
- **#15 post-session belief survey (done):** the survey now collects a 5th
  field, `final_belief_pct` (0–100), so analysis has a paired before/after
  measurement (`initial_belief_pct` → `final_belief_pct`) on the same scale.
