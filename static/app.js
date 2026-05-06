/* Prediction-Market Experimental Web App — frontend state machine
 *
 * Critical correctness notes (spec §8):
 *   - The frontend NEVER displays b_value or condition.
 *   - The frontend NEVER computes LMSR prices itself; it only renders what
 *     /api/preview and /api/trade return.
 *   - Each /api/trade call carries a client-generated request_id (UUID-ish);
 *     on retry we reuse the same id so the server short-circuits.
 *   - session_id lives only in JS memory — never localStorage / cookies.
 *   - client_ms_on_round is measured purely on the client.
 */

'use strict';

const STARTING_CASH  = 15.00;
const MAX_ROUNDS     = 25;
const PREVIEW_DEBOUNCE_MS = 100;

// CHANGES #3 — comprehension-check ground truth. Scenario-based:
//   CQ1: 3 Yes shares + red drawn  → $3.00 payout (Yes pays $1 each on red).
//   CQ2: 5 No shares + red drawn   → $0.00 payout (No pays only on blue).
// Values match the `value` attributes on the radio inputs in index.html.
const CQ1_ANSWER = "3";
const CQ2_ANSWER = "0";

// ─── Session state (volatile; lost on reload, by design) ─────────
const state = {
  session_id: null,
  current_round: 1,
  cash: STARTING_CASH,
  q_yes: 0,
  q_no: 0,
  current_price: 50.0,             // cents (1 decimal)
  trades_made: 0,
  passes: 0,
  price_history: [50.0],           // y-values, length = current_round
  last_trade_recap: null,          // string for the recap strip

  // Trade-panel staging
  selected_side: null,             // "buy_yes" | "buy_no" | null
  shares: 1,

  // Round timing
  round_start_ms: null,

  // Idempotency: the request_id for the in-flight (or about-to-be-sent) trade
  pending_request_id: null,
};

// ─── Tiny helpers ────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const fmtMoney = (n) => `$${Number(n).toFixed(2)}`;
// CHANGES #7 — display ALL prices as integer cents in the UI for consistency.
// (CSV continues to log full precision; rounding is display-only.)
const fmtCents = (cents) => `¢${Math.round(Number(cents))}`;
const fmtCentsDec = fmtCents;  // legacy alias retained — same integer rendering

function uuid() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return 'r-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

async function api(path, body) {
  const init = { method: 'POST', headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) init.body = JSON.stringify(body);
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail;
    try { detail = (await res.json()).detail; } catch { detail = res.statusText; }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

function showScreen(name) {
  document.querySelectorAll('.screen').forEach((el) => {
    el.hidden = el.dataset.screen !== name;
  });
}

// ─── Chart (Chart.js) ────────────────────────────────────────────
let chart = null;

function initChart() {
  const ctx = $('price-chart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [1],
      datasets: [{
        data: [50],
        borderColor: '#335FAE',
        backgroundColor: '#335FAE',
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5,
        tension: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => `${Number(item.parsed.y).toFixed(1)}¢`,
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          min: 1,
          max: MAX_ROUNDS,
          ticks: {
            stepSize: 4,
            color: '#74747C',
            font: { size: 9 },
          },
          grid: { color: '#E6E6EA' },
        },
        y: {
          min: 0,
          max: 100,
          ticks: {
            stepSize: 25,
            color: '#74747C',
            font: { size: 9 },
            callback: (v) => v,
          },
          grid: {
            color: (ctx) => (ctx.tick.value === 50 ? 'transparent' : '#E6E6EA'),
          },
        },
      },
    },
    plugins: [{
      // Custom dashed reference line at ¢50.
      id: 'refLine50',
      afterDraw(chart) {
        const { ctx, chartArea, scales } = chart;
        const y = scales.y.getPixelForValue(50);
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = '#A6A6AE';
        ctx.lineWidth = 1;
        ctx.moveTo(chartArea.left, y);
        ctx.lineTo(chartArea.right, y);
        ctx.stroke();
        ctx.restore();
      },
    }],
  });
}

function updateChart() {
  const labels = state.price_history.map((_, i) => i + 1);
  chart.data.labels = labels;
  chart.data.datasets[0].data = state.price_history.map((y, i) => ({ x: i + 1, y }));
  chart.update();
}

// ─── Welcome screen wiring ───────────────────────────────────────
function initWelcome() {
  $('consent-checkbox').addEventListener('change', (e) => {
    $('start-btn').disabled = !e.target.checked;
  });

  $('start-btn').addEventListener('click', async () => {
    $('start-btn').disabled = true;
    try {
      const data = await api('/api/session');
      state.session_id = data.session_id;
      // INTENTIONAL: we ignore data.condition and data.b_value here. They are
      // never rendered to the participant. (Treatment invisibility, spec §8.1.)

      $('session-id-display').textContent = data.session_id;
      $('session-id-flash').hidden = false;
      $('survey-session-id').textContent = data.session_id;

      setTimeout(() => {
        // CHANGES #3 — funnel through comprehension first; #4 then belief; then trade.
        showScreen('comprehension');
      }, 3000);
    } catch (err) {
      alert('Could not start session: ' + err.message);
      $('start-btn').disabled = false;
    }
  });
}

// ─── Round screen wiring ─────────────────────────────────────────
function startRound1() {
  state.current_round = 1;
  state.cash = STARTING_CASH;
  state.q_yes = 0;
  state.q_no = 0;
  state.current_price = 50.0;
  state.trades_made = 0;
  state.passes = 0;
  state.price_history = [50.0];
  state.last_trade_recap = null;
  resetTradePanel();
  renderRound();
  initChart();
}

function renderRound() {
  $('round-counter').textContent = `Round ${state.current_round} of ${MAX_ROUNDS}`;
  $('price-value').textContent = fmtCents(state.current_price);

  const pctRed = Math.round(state.current_price);
  $('prob-pill').textContent = `Market estimate: ${pctRed}% chance of red`;

  $('cell-cash').textContent     = fmtMoney(state.cash);
  $('cell-yes').textContent      = state.q_yes;
  $('cell-no').textContent       = state.q_no;
  $('cell-round').textContent    = state.current_round;
  $('cell-trades').textContent   = state.trades_made;
  $('cell-passes').textContent   = state.passes;

  // Recap strip on rounds 2..25 only (spec §5.3.2).
  const strip = $('recap-strip');
  if (state.current_round > 1 && state.last_trade_recap) {
    strip.hidden = false;
    strip.innerHTML = state.last_trade_recap;
  } else {
    strip.hidden = true;
  }

  state.round_start_ms = performance.now();
}

function resetTradePanel() {
  state.selected_side = null;
  state.shares = 1;
  state.pending_request_id = null;
  $('btn-buy-yes').classList.remove('selected');
  $('btn-buy-no').classList.remove('selected');
  $('quantity-block').hidden = true;
  $('qty-input').value = 1;
  $('confirm-btn').disabled = true;
  $('confirm-btn').textContent = 'Confirm trade';
  $('prev-cost').textContent   = '$0.00';
  $('prev-payout').textContent = '$0.00';
  $('prev-price').textContent  = fmtCents(state.current_price);
}

function selectSide(side) {
  state.selected_side = side;
  state.shares = 1;
  $('btn-buy-yes').classList.toggle('selected', side === 'buy_yes');
  $('btn-buy-no').classList.toggle('selected', side === 'buy_no');
  $('quantity-block').hidden = false;
  $('qty-input').value = 1;
  refreshPreview();
}

let _previewTimer = null;
function refreshPreview() {
  // Clamp the qty input to [1, 10].
  let n = parseInt($('qty-input').value, 10);
  if (!Number.isFinite(n) || n < 1) n = 1;
  if (n > 10) n = 10;
  $('qty-input').value = n;
  state.shares = n;

  // Payout if win is purely shares × $1, no preview round-trip needed.
  $('prev-payout').textContent = fmtMoney(n * 1.00);

  if (_previewTimer) clearTimeout(_previewTimer);
  _previewTimer = setTimeout(async () => {
    if (!state.selected_side) return;
    try {
      const res = await api('/api/preview', {
        session_id: state.session_id,
        action: state.selected_side,
        shares: state.shares,
      });
      $('prev-cost').textContent = fmtMoney(res.cost);
      $('prev-price').textContent = fmtCentsDec(res.price_after_preview);
      const ok = res.would_succeed;
      $('confirm-btn').disabled = !ok;
      $('confirm-btn').textContent = ok ? 'Confirm trade' : 'Insufficient cash';
    } catch (err) {
      console.warn('preview failed:', err.message);
    }
  }, PREVIEW_DEBOUNCE_MS);
}

async function confirmTrade() {
  if (!state.selected_side) return;
  if (!state.pending_request_id) state.pending_request_id = uuid();

  const ms = Math.round(performance.now() - state.round_start_ms);
  const sharesThisRound = state.shares;
  const sideThisRound = state.selected_side;
  const priceBefore = state.current_price;

  $('confirm-btn').disabled = true;

  try {
    const res = await api('/api/trade', {
      session_id: state.session_id,
      action: sideThisRound,
      shares: sharesThisRound,
      client_ms_on_round: ms,
      request_id: state.pending_request_id,
    });
    onRoundResolved({
      action: sideThisRound,
      shares: sharesThisRound,
      priceBefore,
      res,
    });
  } catch (err) {
    alert('Trade failed: ' + err.message + '\nYou can try again.');
    $('confirm-btn').disabled = false;
  }
}

async function passRound() {
  if (!state.pending_request_id) state.pending_request_id = uuid();
  const ms = Math.round(performance.now() - state.round_start_ms);
  const priceBefore = state.current_price;

  try {
    const res = await api('/api/pass', {
      session_id: state.session_id,
      action: 'pass',
      shares: 0,
      client_ms_on_round: ms,
      request_id: state.pending_request_id,
    });
    onRoundResolved({ action: 'pass', shares: 0, priceBefore, res });
  } catch (err) {
    alert('Pass failed: ' + err.message);
  }
}

function onRoundResolved({ action, shares, priceBefore, res }) {
  state.current_price = res.current_price;
  state.cash = res.current_cash;
  state.q_yes = res.q_yes;
  state.q_no = res.q_no;
  state.price_history.push(res.current_price);

  if (action === 'pass') {
    state.passes += 1;
    state.last_trade_recap =
      `<strong>Last round:</strong> passed (no trade) · price unchanged at ` +
      `${fmtCentsDec(priceBefore)}.`;
  } else {
    state.trades_made += 1;
    const sideLabel = action === 'buy_yes' ? 'Yes' : 'No';
    const plural = shares === 1 ? 'share' : 'shares';
    state.last_trade_recap =
      `<strong>Last trade:</strong> bought ${shares} ${sideLabel} ${plural} ` +
      `at ${fmtCentsDec(priceBefore)} · price moved ${fmtCentsDec(priceBefore)} ` +
      `→ ${fmtCentsDec(res.current_price)}.`;
  }

  state.current_round = (res.next_round !== null && res.next_round !== undefined)
    ? res.next_round
    : state.current_round + 1;

  resetTradePanel();
  updateChart();

  if (res.session_complete) {
    finalizeAndShowComplete();
  } else {
    renderRound();
  }
}

async function finalizeAndShowComplete() {
  try {
    const res = await api('/api/finalize', { session_id: state.session_id });
    renderComplete(res);
    showScreen('complete');
  } catch (err) {
    alert('Could not finalize session: ' + err.message);
  }
}

// ─── Session Complete screen ─────────────────────────────────────
function renderComplete(r) {
  // Outcome
  $('result-line-1').textContent = `A ${r.draw_outcome} ball was drawn`;
  if (r.draw_outcome === 'red') {
    $('result-line-2').textContent =
      'Yes shares paid out ¢100 each. No shares expired worthless.';
  } else {
    $('result-line-2').textContent =
      'No shares paid out ¢100 each. Yes shares expired worthless.';
  }

  // CHANGES #1 — explicit wallet decomposition.
  const isRed = r.draw_outcome === 'red';

  // Cash held at end of round 25 = final cash MINUS the share payouts that
  // were just credited. The server gives us cash_remaining directly.
  $('bk-cash-end').textContent = fmtMoney(r.cash_remaining);

  // Yes line (shares × per-share payout, sign reflects whether it paid out).
  $('bk-yes-label').textContent = isRed
    ? `+ ${r.yes_shares} Yes shares × $1.00`
    : `+ ${r.yes_shares} Yes shares × $0.00`;
  $('bk-yes-amount').textContent = isRed && r.yes_payout > 0
    ? `+${fmtMoney(r.yes_payout)}`
    : '$0.00';
  $('bk-yes-amount').classList.toggle('positive', isRed && r.yes_payout > 0);
  $('bk-yes-amount').classList.toggle('muted', !(isRed && r.yes_payout > 0));

  // No line.
  $('bk-no-label').textContent = !isRed
    ? `+ ${r.no_shares} No shares × $1.00`
    : `+ ${r.no_shares} No shares × $0.00`;
  $('bk-no-amount').textContent = !isRed && r.no_payout > 0
    ? `+${fmtMoney(r.no_payout)}`
    : '$0.00';
  $('bk-no-amount').classList.toggle('positive', !isRed && r.no_payout > 0);
  $('bk-no-amount').classList.toggle('muted', !( !isRed && r.no_payout > 0));

  // Final wallet value = cash_remaining + payouts (matches server final_cash).
  $('bk-final-wallet').textContent = fmtMoney(r.final_cash);

  // P&L line — colored positive/negative.
  const pnl = r.total_pnl;
  const pnlEl = $('bk-pnl-line');
  pnlEl.textContent = (pnl >= 0 ? '+' : '−') + fmtMoney(Math.abs(pnl));
  pnlEl.classList.toggle('positive', pnl >= 0);
  pnlEl.classList.toggle('negative', pnl < 0);

  // Performance grid (slimmed: trades, passes, final price only — money lives
  // in the breakdown above to avoid duplicate-headline confusion).
  $('pg-trades').textContent      = state.trades_made;
  $('pg-passes').textContent      = state.passes;
  $('pg-final-price').textContent = fmtCents(r.final_price);
}

function initComplete() {
  $('take-survey-btn').addEventListener('click', () => showScreen('survey'));
}

// ─── Comprehension screen wiring (CHANGES #3) ────────────────────
function initComprehension() {
  let attempts = 0;
  const form = $('comprehension-form');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    attempts += 1;

    const fd = new FormData(form);
    const a1 = fd.get('cq1');  // string radio value, or null if unanswered
    const a2 = fd.get('cq2');

    if (a1 === null || a2 === null) {
      return alert('Please answer both questions before continuing.');
    }

    const ok1 = a1 === CQ1_ANSWER;
    const ok2 = a2 === CQ2_ANSWER;

    $('cq1-error').hidden = ok1;
    $('cq2-error').hidden = ok2;

    if (!(ok1 && ok2)) {
      $('comp-attempts-note').hidden = false;
      $('comp-attempt-num').textContent = attempts + 1;
      return;
    }

    $('comp-submit').disabled = true;
    try {
      await api('/api/comprehension', {
        session_id: state.session_id,
        attempts,
      });
      showScreen('belief');
    } catch (err) {
      alert('Could not record comprehension result: ' + err.message);
      $('comp-submit').disabled = false;
    }
  });
}

// ─── Belief-elicitation wiring (CHANGES #4) ──────────────────────
function initBelief() {
  $('belief-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const v = parseInt($('belief-input').value, 10);
    if (!Number.isFinite(v) || v < 0 || v > 100) {
      return alert('Please enter a whole number between 0 and 100.');
    }
    $('belief-submit').disabled = true;
    try {
      await api('/api/belief', {
        session_id: state.session_id,
        initial_belief_pct: v,
      });
      showScreen('round');
      startRound1();
    } catch (err) {
      alert('Could not record belief: ' + err.message);
      $('belief-submit').disabled = false;
    }
  });
}

// ─── Survey screen wiring ────────────────────────────────────────
function initSurvey() {
  $('survey-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const risk = parseInt(fd.get('risk_tolerance'), 10);
    const exp  = parseInt(fd.get('trading_experience_months'), 10);
    const fam  = fd.get('prediction_market_familiarity');
    const fos  = (fd.get('field_of_study') || '').toString().trim();
    const fb   = parseInt(fd.get('final_belief_pct'), 10);

    if (!risk) return alert('Please select a risk tolerance.');
    if (!Number.isFinite(exp) || exp < 0) return alert('Please enter trading experience as 0 or more.');
    if (!fam) return alert('Please answer the prediction-market question.');
    if (!fos) return alert('Please enter your field of study.');
    if (!Number.isFinite(fb) || fb < 0 || fb > 100) {
      return alert('Please enter your final belief as a whole number between 0 and 100.');
    }

    $('survey-submit').disabled = true;
    try {
      await api('/api/survey', {
        session_id: state.session_id,
        risk_tolerance: risk,
        trading_experience_months: exp,
        prediction_market_familiarity: fam === 'yes',
        field_of_study: fos.slice(0, 100),
        final_belief_pct: fb,
      });
      showScreen('thanks');
    } catch (err) {
      alert('Survey submission failed: ' + err.message);
      $('survey-submit').disabled = false;
    }
  });
}

// ─── Round-screen event hookup ───────────────────────────────────
function initRoundScreen() {
  $('btn-buy-yes').addEventListener('click', () => selectSide('buy_yes'));
  $('btn-buy-no').addEventListener('click',  () => selectSide('buy_no'));

  $('qty-minus').addEventListener('click', () => {
    $('qty-input').value = Math.max(1, (parseInt($('qty-input').value, 10) || 1) - 1);
    refreshPreview();
  });
  $('qty-plus').addEventListener('click', () => {
    $('qty-input').value = Math.min(10, (parseInt($('qty-input').value, 10) || 1) + 1);
    refreshPreview();
  });
  $('qty-input').addEventListener('input', refreshPreview);

  $('confirm-btn').addEventListener('click', confirmTrade);
  $('cancel-btn').addEventListener('click', resetTradePanel);

  $('pass-btn').addEventListener('click', () => {
    $('pass-modal-text').textContent =
      `Pass round ${state.current_round}? You won't trade this round.`;
    $('pass-modal').hidden = false;
  });
  $('pass-cancel').addEventListener('click',  () => { $('pass-modal').hidden = true; });
  $('pass-confirm').addEventListener('click', () => {
    $('pass-modal').hidden = true;
    passRound();
  });
}

// ─── Boot ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initWelcome();
  initComprehension();
  initBelief();
  initRoundScreen();
  initComplete();
  initSurvey();
});
