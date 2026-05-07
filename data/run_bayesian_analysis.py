#!/usr/bin/env python3
"""
CS598 Final Project — NumPyro Bayesian analysis + remaining figures.

Generates everything still missing from the report:
  • Posterior estimates (mean, 95% HPDI, P(τ>0)) for τ on every outcome
  • Discrete-time hazard model for convergence (log HR low vs. high)
  • Mediation decomposition (NDE, NIE, NDE/total) with HPDIs
  • Figure 6 — empirical convergence curves by condition
  • Figure 8 — observed vs. ZI vs. BR baselines, both b conditions

Usage
-----
  pip install numpyro jax pandas matplotlib numpy
  python run_bayesian_analysis.py

Input files (place in working directory):
  all_sessions.csv         per-trade rows
  sessions_summary.csv     per-session aggregated

Output files:
  bayesian_results.txt     tau posteriors, hazard, mediation
  fig_convergence.png      Figure 6
  fig_baselines.png        Figure 8

The script tolerates missing covariate columns; it skips them with a warning
and proceeds without those adjustment terms.
"""

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

warnings.filterwarnings("ignore")
numpyro.set_host_device_count(4)

TRUTH_CENTS = 70.0
TRUTH_PROB = 0.70
BAND = 5.0
N_ROUNDS = 25
STARTING_CASH = 15.0


# ============================================================
# LMSR backend (for ZI / BR baseline simulation)
# ============================================================
def lmsr_price(q_yes, q_no, b):
    m = max(q_yes / b, q_no / b)
    e_y = np.exp(q_yes / b - m)
    e_n = np.exp(q_no / b - m)
    return e_y / (e_y + e_n)


def lmsr_cost(q_yes, q_no, b):
    m = max(q_yes / b, q_no / b)
    return b * (m + np.log(np.exp(q_yes / b - m) + np.exp(q_no / b - m)))


def trade_cost(q_yes, q_no, dy, dn, b):
    return lmsr_cost(q_yes + dy, q_no + dn, b) - lmsr_cost(q_yes, q_no, b)


# ============================================================
# Helpers
# ============================================================
def hpdi(samples, prob=0.95):
    s = np.sort(np.asarray(samples).ravel())
    n = len(s)
    w = int(np.floor(prob * n))
    diffs = s[w:] - s[: n - w]
    i = int(np.argmin(diffs))
    return float(s[i]), float(s[i + w])


def summarize(name, samples, lines):
    m = float(np.mean(samples))
    lo, hi = hpdi(samples, 0.95)
    p_pos = float((np.asarray(samples) > 0).mean())
    lines.append(f"{name}: mean = {m:+.4f}   95% HPDI = [{lo:+.4f}, {hi:+.4f}]   P(τ>0) = {p_pos:.3f}")
    return m, lo, hi, p_pos


def run_mcmc(model, *args, num_warmup=1500, num_samples=2000, num_chains=4, seed=0):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(seed), *args)
    return mcmc.get_samples()


def standardize(v):
    v = np.asarray(v, dtype=float)
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)


# ============================================================
# Load data
# ============================================================
sessions = pd.read_csv("sessions_summary.csv")
trades = pd.read_csv("all_sessions.csv")

# Canonicalize condition column → T (1 = low liquidity / b=10, 0 = high / b=100)
if "condition" in sessions.columns:
    sessions["T"] = (sessions["condition"].str.lower() == "low").astype(int)
elif "b" in sessions.columns:
    sessions["T"] = (sessions["b"] == 10).astype(int)
else:
    raise SystemExit("sessions_summary.csv needs a 'condition' or 'b' column.")

T = sessions["T"].values.astype(int)
N = len(sessions)

# Build covariate matrix from whatever standard columns are available
COV_CANDIDATES = [
    "risk_tolerance", "risk_tol",
    "trading_exp_months", "trading_exp", "trading_experience",
    "pm_familiar", "pm_familiarity",
    "quant_field",
]
seen = set()
cov_cols = []
for c in COV_CANDIDATES:
    if c in sessions.columns and c not in seen:
        sessions[c + "_z"] = standardize(sessions[c].fillna(sessions[c].median()))
        cov_cols.append(c + "_z")
        seen.add(c)

X = sessions[cov_cols].values if cov_cols else np.zeros((N, 0))
print(f"N = {N}  (low={T.sum()}, high={N - T.sum()})  covariates: {cov_cols or '(none)'}")

# ============================================================
# Bayesian models
# ============================================================
def normal_model(T, X, y=None):
    a = numpyro.sample("alpha", dist.Normal(0.0, 5.0))
    tau = numpyro.sample("tau", dist.Normal(0.0, 10.0))
    if X.shape[1] > 0:
        b = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([X.shape[1]]))
        mu = a + tau * T + jnp.dot(X, b)
    else:
        mu = a + tau * T
    s = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("y", dist.Normal(mu, s), obs=y)


def negbin_model(T, X, y=None):
    a = numpyro.sample("alpha", dist.Normal(np.log(10.0), 1.0))
    tau = numpyro.sample("tau", dist.Normal(0.0, 0.5))
    if X.shape[1] > 0:
        b = numpyro.sample("beta", dist.Normal(0.0, 0.5).expand([X.shape[1]]))
        log_mu = a + tau * T + jnp.dot(X, b)
    else:
        log_mu = a + tau * T
    log_phi = numpyro.sample("log_phi", dist.Normal(0.0, 1.0))
    numpyro.sample("y", dist.NegativeBinomial2(jnp.exp(log_mu), jnp.exp(log_phi)), obs=y)


def betabinom_model(T, X, n_trials, y=None):
    a = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
    tau = numpyro.sample("tau", dist.Normal(0.0, 0.5))
    if X.shape[1] > 0:
        b = numpyro.sample("beta", dist.Normal(0.0, 0.5).expand([X.shape[1]]))
        logit_p = a + tau * T + jnp.dot(X, b)
    else:
        logit_p = a + tau * T
    p = jax.nn.sigmoid(logit_p)
    log_kappa = numpyro.sample("log_kappa", dist.Normal(2.0, 1.0))
    kappa = jnp.exp(log_kappa)
    numpyro.sample("y", dist.BetaBinomial(p * kappa, (1.0 - p) * kappa, n_trials), obs=y)


def hazard_model(T, round_idx, n_rounds, y=None):
    # Random-walk smoothed baseline log-hazard across rounds (more stable than separate intercepts)
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
    alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0, 1).expand([n_rounds]))
    alpha = numpyro.deterministic("alpha", -3.0 + sigma_alpha * jnp.cumsum(alpha_raw))
    tau = numpyro.sample("tau", dist.Normal(0.0, 1.0))
    logit_h = alpha[round_idx] + tau * T
    numpyro.sample("y", dist.Bernoulli(logits=logit_h), obs=y)


# ============================================================
# Outcome → column resolution (handle naming variants)
# ============================================================
def col(*candidates):
    for c in candidates:
        if c in sessions.columns:
            return c
    return None


outcomes_continuous = {
    "MSE":              col("mse", "mse_truth"),
    "Final pricing |p25 − 0.70|": col("final_err", "final_pricing_error", "final_abs_err"),
    "Mean trade size (shares)":   col("mean_trade_size", "avg_trade_size"),
    "Final price (cents)":        col("final_price", "final_price_cents"),
}

n_trades_col = col("n_trades", "trades_made", "num_trades")
n_passes_col = col("n_passes", "passes", "num_passes")

# ============================================================
# Run continuous-outcome regressions
# ============================================================
out = []
out.append("=" * 70)
out.append("BAYESIAN POSTERIOR ESTIMATES — τ is (low − high) effect")
out.append("=" * 70)

for label, c in outcomes_continuous.items():
    if c is None:
        out.append(f"[skipped] {label}: no matching column")
        continue
    y = sessions[c].values.astype(float)
    samples = run_mcmc(normal_model, T, X, y, seed=hash(c) & 0xFFFF)
    summarize(f"{label:40s}", np.array(samples["tau"]), out)

# Trade frequency (Negative Binomial)
if n_trades_col:
    y = sessions[n_trades_col].values.astype(int)
    samples = run_mcmc(negbin_model, T, X, y)
    tau = np.array(samples["tau"])
    summarize(f"{'Trade frequency (log RR)':40s}", tau, out)
    rr = np.exp(tau)
    out.append(f"  → Rate ratio (low / high): mean = {rr.mean():.3f}   95% HPDI = [{hpdi(rr)[0]:.3f}, {hpdi(rr)[1]:.3f}]")
else:
    out.append("[skipped] Trade frequency: no matching column")

# Pass rate (Beta-Binomial)
if n_passes_col:
    y = sessions[n_passes_col].values.astype(int)
    n_trials = np.full(N, N_ROUNDS)
    samples = run_mcmc(betabinom_model, T, X, n_trials, y)
    tau = np.array(samples["tau"])
    summarize(f"{'Pass rate (log OR)':40s}", tau, out)
    # Marginal effect on probability scale (low − high), averaged over the sample
    a_post = np.array(samples["alpha"])
    if X.shape[1] > 0:
        b_post = np.array(samples["beta"])
        x_bar = X.mean(0)
        logit_high = a_post + b_post @ x_bar
    else:
        logit_high = a_post
    p_high = 1 / (1 + np.exp(-logit_high))
    p_low = 1 / (1 + np.exp(-(logit_high + tau)))
    diff_pp = (p_low - p_high) * 100
    out.append(f"  → Pass rate diff (pp, low − high): mean = {diff_pp.mean():+.2f}   95% HPDI = [{hpdi(diff_pp)[0]:+.2f}, {hpdi(diff_pp)[1]:+.2f}]")
else:
    out.append("[skipped] Pass rate: no matching column")

# ============================================================
# Discrete-time hazard model for convergence
# ============================================================
def first_sustained_round(prices, truth=TRUTH_CENTS, band=BAND):
    """Earliest round t after which all prices stay within ±band of truth."""
    p = np.asarray(prices, dtype=float)
    for t in range(len(p)):
        if np.all(np.abs(p[t:] - truth) <= band):
            return t + 1  # 1-indexed
    return len(p) + 1  # right-censored


# Build long-format hazard dataset
hz_rows = []
for sid, sub in trades.groupby("session_id"):
    sub = sub.sort_values("round")
    prices = sub["price_after"].values
    cv = first_sustained_round(prices)
    Ti = int((sessions.loc[sessions["session_id"] == sid, "T"]).iloc[0])
    n_obs = min(cv, N_ROUNDS)
    for r in range(1, n_obs + 1):
        event = int(r == cv)
        hz_rows.append((r - 1, Ti, event))  # 0-indexed round

hz = np.array(hz_rows, dtype=int)
samples = run_mcmc(
    hazard_model,
    hz[:, 1],         # T
    hz[:, 0],         # round_idx
    N_ROUNDS,
    hz[:, 2],         # event
    seed=99,
)
tau_hz = np.array(samples["tau"])
out.append("\n" + "=" * 70)
out.append("CONVERGENCE HAZARD MODEL")
out.append("=" * 70)
summarize(f"{'Log hazard ratio (low vs. high)':40s}", tau_hz, out)
hr = np.exp(tau_hz)
lo, hi = hpdi(hr)
out.append(f"  → Hazard ratio: mean = {hr.mean():.2f}   95% HPDI = [{lo:.2f}, {hi:.2f}]")

# ============================================================
# Mediation: T → mean_trade_size → final pricing error
# Also reports T → trade_freq → final_err for completeness
# ============================================================
def mediation(T, X, M, Y, seed=0):
    def model_M(T, X, M=None):
        a0 = numpyro.sample("a0", dist.Normal(0.0, 5.0))
        a_T = numpyro.sample("a_T", dist.Normal(0.0, 1.0))
        if X.shape[1] > 0:
            a_X = numpyro.sample("a_X", dist.Normal(0.0, 1.0).expand([X.shape[1]]))
            mu = a0 + a_T * T + jnp.dot(X, a_X)
        else:
            mu = a0 + a_T * T
        s = numpyro.sample("s_M", dist.HalfNormal(1.0))
        numpyro.sample("M", dist.Normal(mu, s), obs=M)

    def model_Y(T, M, X, Y=None):
        c0 = numpyro.sample("c0", dist.Normal(0.0, 5.0))
        c_T = numpyro.sample("c_T", dist.Normal(0.0, 1.0))
        c_M = numpyro.sample("c_M", dist.Normal(0.0, 1.0))
        if X.shape[1] > 0:
            c_X = numpyro.sample("c_X", dist.Normal(0.0, 1.0).expand([X.shape[1]]))
            mu = c0 + c_T * T + c_M * M + jnp.dot(X, c_X)
        else:
            mu = c0 + c_T * T + c_M * M
        s = numpyro.sample("s_Y", dist.HalfNormal(1.0))
        numpyro.sample("Y", dist.Normal(mu, s), obs=Y)

    sM = run_mcmc(model_M, T, X, M, seed=seed)
    sY = run_mcmc(model_Y, T, M, X, Y, seed=seed + 1)
    a_T = np.array(sM["a_T"])
    c_T = np.array(sY["c_T"])
    c_M = np.array(sY["c_M"])
    n = min(len(a_T), len(c_T), len(c_M))
    NDE = c_T[:n]
    NIE = (a_T * c_M)[:n]
    Total = NDE + NIE
    return NDE, NIE, Total


def report_mediation(label, NDE, NIE, Total, lines):
    lines.append(f"\n--- {label} ---")
    lines.append(f"NDE   mean = {NDE.mean():+.4f}   95% HPDI = [{hpdi(NDE)[0]:+.4f}, {hpdi(NDE)[1]:+.4f}]")
    lines.append(f"NIE   mean = {NIE.mean():+.4f}   95% HPDI = [{hpdi(NIE)[0]:+.4f}, {hpdi(NIE)[1]:+.4f}]")
    lines.append(f"Total mean = {Total.mean():+.4f}   95% HPDI = [{hpdi(Total)[0]:+.4f}, {hpdi(Total)[1]:+.4f}]")
    # Proportions only meaningful when Total is consistently signed
    sign_consistent = (np.sign(Total) == np.sign(Total.mean())).mean() > 0.95
    if sign_consistent:
        prop_NDE = NDE / Total
        prop_NIE = NIE / Total
        lines.append(f"NDE/Total mean = {prop_NDE.mean():.3f}   95% HPDI = [{hpdi(prop_NDE)[0]:.3f}, {hpdi(prop_NDE)[1]:.3f}]")
        lines.append(f"NIE/Total mean = {prop_NIE.mean():.3f}   95% HPDI = [{hpdi(prop_NIE)[0]:.3f}, {hpdi(prop_NIE)[1]:.3f}]")
    else:
        lines.append("(Total effect changes sign across posterior — proportions not reported.)")


out.append("\n" + "=" * 70)
out.append("MEDIATION DECOMPOSITION (Imai/Keele/Yamamoto 2010, two-model)")
out.append("=" * 70)

err_col = outcomes_continuous["Final pricing |p25 − 0.70|"]
ts_col = outcomes_continuous["Mean trade size (shares)"]
if err_col and ts_col:
    NDE, NIE, Total = mediation(T, X, sessions[ts_col].values, sessions[err_col].values, seed=2026)
    report_mediation("T → mean_trade_size → final pricing error", NDE, NIE, Total, out)

if err_col and n_trades_col:
    NDE, NIE, Total = mediation(T, X, sessions[n_trades_col].values.astype(float), sessions[err_col].values, seed=4040)
    report_mediation("T → trade_frequency → final pricing error", NDE, NIE, Total, out)

# ============================================================
# Sensitivity to ρ (Imai et al. parametric sensitivity)
# Approximation: bias of NIE ≈ ρ · σ_eM · σ_eY · (T variance term)
# We report the threshold ρ at which NIE crosses zero.
# ============================================================
if err_col and ts_col:
    M_resid = sessions[ts_col].values - (sessions[ts_col].values.mean())  # crude
    Y_resid = sessions[err_col].values - sessions[err_col].values.mean()
    sigma_M = M_resid.std()
    sigma_Y = Y_resid.std()
    NIE_pt = NIE.mean()
    rho_crit = NIE_pt / (sigma_M * sigma_Y) if sigma_M * sigma_Y > 0 else float("nan")
    out.append(f"\nSensitivity (rough): ρ that overturns NIE ≈ {rho_crit:+.2f}")

Path("bayesian_results.txt").write_text("\n".join(out))
print("\n".join(out))
print("\nWrote bayesian_results.txt")

# ============================================================
# Figure 6 — empirical convergence curves
# ============================================================
def cumul_converged(sids):
    n = len(sids)
    counts = np.zeros(N_ROUNDS)
    for sid in sids:
        sub = trades[trades["session_id"] == sid].sort_values("round")
        cv = first_sustained_round(sub["price_after"].values)
        if cv <= N_ROUNDS:
            counts[cv - 1:] += 1
    return counts / max(n, 1)


low_sids = sessions.loc[T == 1, "session_id"].values
high_sids = sessions.loc[T == 0, "session_id"].values
p_low = cumul_converged(low_sids)
p_high = cumul_converged(high_sids)
rounds = np.arange(1, N_ROUNDS + 1)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.step(rounds, p_low, where="post", color="#1f77b4", linewidth=2.5,
        label=f"Low Liquidity (b=10), n={len(low_sids)}")
ax.step(rounds, p_high, where="post", color="#d62728", linewidth=2.5,
        label=f"High Liquidity (b=100), n={len(high_sids)}")
ax.set_xlim(1, N_ROUNDS)
ax.set_ylim(0, 1)
ax.set_xlabel("Round")
ax.set_ylabel("Cumulative proportion converged")
ax.set_title(f"Convergence to within ±{int(BAND)}¢ of {int(TRUTH_CENTS)}¢ (sustained through round {N_ROUNDS})")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("fig_convergence.png", dpi=160, bbox_inches="tight")
plt.close()
print("Wrote fig_convergence.png")

# ============================================================
# Figure 8 — observed vs. ZI vs. BR baselines
# ============================================================
def simulate_zi(b_val, n_sessions, seed=0):
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(n_sessions):
        q_y = q_n = 0.0
        cash = STARTING_CASH
        ps = []
        for _r in range(N_ROUNDS):
            action = rng.choice(["yes", "no", "pass"])
            if action != "pass":
                qty = int(rng.integers(1, 11))
                dy, dn = (qty, 0) if action == "yes" else (0, qty)
                c = trade_cost(q_y, q_n, dy, dn, b_val)
                if c <= cash:
                    q_y += dy
                    q_n += dn
                    cash -= c
            ps.append(lmsr_price(q_y, q_n, b_val) * 100)
        paths.append(ps)
    return np.asarray(paths)


def simulate_br(b_val, n_sessions):
    """BR: each round, buy 1 Yes share if price < truth and cash allows."""
    paths = []
    for _ in range(n_sessions):
        q_y = q_n = 0.0
        cash = STARTING_CASH
        ps = []
        for _r in range(N_ROUNDS):
            if lmsr_price(q_y, q_n, b_val) < TRUTH_PROB:
                c = trade_cost(q_y, q_n, 1, 0, b_val)
                if c <= cash:
                    q_y += 1
                    cash -= c
            ps.append(lmsr_price(q_y, q_n, b_val) * 100)
        paths.append(ps)
    return np.asarray(paths)


def observed_paths(sids):
    paths = []
    for sid in sids:
        sub = trades[trades["session_id"] == sid].sort_values("round")
        # Prepend round-0 starting price 50
        paths.append(np.concatenate([[50.0], sub["price_after"].values]))
    return np.asarray(paths)


obs_low = observed_paths(low_sids)
obs_high = observed_paths(high_sids)
zi_low = simulate_zi(10, len(low_sids), seed=11)
zi_high = simulate_zi(100, len(high_sids), seed=22)
br_low = simulate_br(10, len(low_sids))
br_high = simulate_br(100, len(high_sids))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
panels = [
    (axes[0], "Low Liquidity (b=10)", obs_low, zi_low, br_low, "#1f77b4", (0, 100)),
    (axes[1], "High Liquidity (b=100)", obs_high, zi_high, br_high, "#d62728", (40, 80)),
]
r_obs = np.arange(0, N_ROUNDS + 1)
r_sim = np.arange(1, N_ROUNDS + 1)
for ax, title, obs, zi, br, color, ylim in panels:
    obs_q1 = np.percentile(obs, 25, axis=0)
    obs_q3 = np.percentile(obs, 75, axis=0)
    ax.fill_between(r_obs, obs_q1, obs_q3, color=color, alpha=0.18, label="Observed IQR")
    ax.plot(r_obs, obs.mean(0), color=color, linewidth=2.5, label="Observed mean")
    ax.plot(r_sim, zi.mean(0), color="gray", linestyle="--", linewidth=1.8, label="Zero-Intelligence")
    ax.plot(r_sim, br.mean(0), color="black", linestyle="-.", linewidth=1.8, label="Bayesian-Rational")
    ax.axhline(TRUTH_CENTS, color="green", linestyle=":", linewidth=1.4, alpha=0.85,
               label=f"True P(red) = {int(TRUTH_CENTS)}¢")
    ax.set_xlim(0, N_ROUNDS)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean price (¢)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)

plt.suptitle("Observed price paths vs. Zero-Intelligence and Bayesian-Rational baselines",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("fig_baselines.png", dpi=160, bbox_inches="tight")
plt.close()
print("Wrote fig_baselines.png")

print("\nDone. Plug results into:")
print("  • Section 8.2 — τ on MSE / final price / final pricing error")
print("  • Section 8.3 — log hazard ratio (replaces 'substantially above zero')")
print("  • Section 8.4 — table 'Effect (95% HPDI)' column entries")
print("  • Section 8.5 — NDE / NIE / Total + NDE/Total proportion")
print("  • Figure 6  — fig_convergence.png")
print("  • Figure 8  — fig_baselines.png")
