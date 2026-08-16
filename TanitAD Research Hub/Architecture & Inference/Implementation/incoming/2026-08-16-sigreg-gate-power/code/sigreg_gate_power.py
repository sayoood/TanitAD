#!/usr/bin/env python3
"""How much collapse can the O6 spectrum monitor actually SEE at n=48?

⛔ WHAT THIS EXISTS TO SETTLE. The S-W gate's O6 criterion is
``"O6_rank_retention": ">= 0.8x effective rank across phases"``
(``STAGE_GATE_SPEC`` in ``stack/scripts/train_v6_staged.py``). The quantity it
compares is ``spectrum_report(...)["effective_rank"]`` computed from ONE
training batch: ``z_op_win`` reshaped to ``[B*W, d_op]``.

On the live v6F S-W run that is **n = 48 rows, d = 2048** — 8 windows x 6
CONSECUTIVE frames (``--batch 8 --window 6``), drawn from only **4 episodes**
(``--eps-per-batch 4``), verified against the run's own argv in
``…/2026-08-15-v6-thor-resume/code/RESTART_v6F_SW.sh``.

Two things follow that no amount of reading the number can fix:

1. **A centred covariance from n rows has rank <= n-1.** At n=48 the estimator
   CANNOT report more than 47 no matter what the representation does, so
   "15 of 2048" is not a statement about 2048 dimensions. It is 15 of 47.
2. **The 48 rows are not 48 independent draws.** They are ~4 episodes' worth of
   near-duplicate frames. The estimator's variance is set by the number of
   independent CLUSTERS, not by the row count.

This script measures, rather than asserts, what that costs the gate:

* ``null``    — the sampling distribution of ``effective_rank`` when NOTHING
                changed between two phases, and the resulting FALSE-POSITIVE
                rate of the ``>= 0.8x`` criterion.
* ``power``   — the true collapse ratio needed before the criterion fires at
                80 % power, i.e. the effect size it can actually resolve.
* ``satur``   — how ``effective_rank`` responds to the TRUE rank, to show where
                the estimator saturates and stops carrying information.
* ``calib``   — which sampling regime reproduces the live run's banked spread
                (mean 15.13, range 3.37 -> 30.06 over 38 records), which decides
                whether the low reading is a REPRESENTATION fact or a SAMPLER
                fact.
* ``fix``     — the same measurements for the proposed pooled/clustered
                estimator, so the fix is chosen on evidence.

Evidence class: **MEASURED (ours)** — simulation under a stated generative
model, CPU-only, seeded. The live run's 38 spectrum records are NOT in the repo
(they are on Thor); their summary statistics enter here only as the ``calib``
TARGET, and are labelled INHERITED wherever quoted.

Usage:
    python sigreg_gate_power.py --out raw/  [--trials 2000] [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

# The instrument under test, imported (never reimplemented) so this script
# cannot drift from what the trainer actually emits.
_STACK = Path(__file__).resolve()
for _p in _STACK.parents:
    if (_p / "stack" / "tanitad").is_dir():
        sys.path.insert(0, str(_p / "stack"))
        break
from tanitad.eval.spectral import effective_rank        # noqa: E402

# --------------------------------------------------------------------------- #
# The live run's geometry, read off its argv — not assumed.
# --------------------------------------------------------------------------- #
LIVE_B, LIVE_W, LIVE_EPS = 8, 6, 4          # --batch / --window / --eps-per-batch
LIVE_N = LIVE_B * LIVE_W                    # 48 rows per spectrum call
LIVE_D = 2048                               # readout-grid 4 x 4 x readout-dim 128

#: INHERITED (brief, from the live run's spectrum records on Thor; the 38 raw
#: records are NOT in this repo). Used ONLY as the calibration target.
BANKED = {"n": 48, "d": 2048, "top_k": 8, "records": 38,
          "effective_rank_mean": 15.13,
          "effective_rank_min": 3.37, "effective_rank_max": 30.06,
          "top_k_share_min": 0.8615, "top_k_share_max": 0.9998,
          "step_200": 16.75, "step_4000": 12.10, "step_7600": 17.59}


# --------------------------------------------------------------------------- #
# Generative model of the LATENT and of the SAMPLER
# --------------------------------------------------------------------------- #
def powerlaw_eigs(d: int, alpha: float) -> torch.Tensor:
    """Population covariance eigenvalues lambda_i ~ i^-alpha, normalised.

    A power law is the honest default: real representation spectra decay
    smoothly, and a hard rank-k spike would make the estimator look better than
    it is (a spike is the easiest possible thing to detect)."""
    i = torch.arange(1, d + 1, dtype=torch.float64)
    e = i ** (-float(alpha))
    return e / e.sum()


def true_effective_rank(eigs: torch.Tensor) -> float:
    """The ESTIMAND: the same functional the instrument computes, applied to the
    POPULATION spectrum. ``effective_rank`` consumes SINGULAR values, and the
    population singular values are sqrt(lambda)."""
    return effective_rank(eigs.sqrt())


def draw_batch(eigs: torch.Tensor, gen: torch.Generator, *,
               n_eps: int = LIVE_EPS, n_win: int = LIVE_B, w: int = LIVE_W,
               rho_ep: float = 0.0, rho_win: float = 0.0) -> torch.Tensor:
    """One spectrum call's ``[n_win*w, d]`` tensor under the NESTED correlation
    the real sampler has: frames inside a window are near-duplicates, windows
    inside an episode share a scene.

    ``rho_ep`` / ``rho_win`` are the fraction of each row's variance carried by
    the episode- and window-level factor. ``rho_ep = rho_win = 0`` is the
    (false, but standard) iid-rows assumption the current reading implies.
    """
    d = eigs.numel()
    n = n_win * w
    a_ep = math.sqrt(max(rho_ep, 0.0))
    a_win = math.sqrt(max(rho_win, 0.0))
    a_row = math.sqrt(max(1.0 - rho_ep - rho_win, 0.0))
    if a_row == 0.0 and (a_ep + a_win) == 0.0:
        raise ValueError("degenerate correlation split")
    ep_of_win = torch.arange(n_win) % n_eps            # windows -> episodes
    g_ep = torch.randn(n_eps, d, generator=gen, dtype=torch.float64)
    g_win = torch.randn(n_win, d, generator=gen, dtype=torch.float64)
    g_row = torch.randn(n, d, generator=gen, dtype=torch.float64)
    g = (a_ep * g_ep[ep_of_win].repeat_interleave(w, dim=0)
         + a_win * g_win.repeat_interleave(w, dim=0)
         + a_row * g_row)
    # Work directly in the eigenbasis: the spectrum is rotation-invariant, so a
    # random orthogonal mixing would cost d^2 memory and change nothing.
    return g * eigs.sqrt()


def _svals(zc: torch.Tensor) -> torch.Tensor:
    """Singular values of an already-centred ``[n, d]`` via the SMALLER Gram.

    Identical mathematics to ``torch.linalg.svdvals``; the agreement is
    MEASURED once per run by :func:`verify_fast_path` and the max discrepancy
    is written into the artifact, so the speedup is never taken on faith.
    A 1536x2048 ``svdvals`` costs ~0.5 s and the pooled arms need thousands.
    """
    n, d = zc.shape
    g = zc @ zc.T if n <= d else zc.T @ zc
    ev = torch.linalg.eigvalsh(g).clamp_min(0)
    return ev.flip(0).sqrt()


def er_of(z: torch.Tensor) -> float:
    """``spectrum_report``'s effective_rank, reproduced (centre, singular values
    of the centred rows, entropy rank on sigma)."""
    zc = z.double() - z.double().mean(dim=0, keepdim=True)
    return effective_rank(_svals(zc))


def verify_fast_path(gen: torch.Generator) -> dict:
    """⛔ The fast path is a claim until it is checked. Compare it against the
    instrument's own ``svdvals`` route on the exact shapes this script uses."""
    rows = []
    for n in (LIVE_N, 8 * LIVE_N, 32 * LIVE_N):
        z = torch.randn(n, LIVE_D, generator=gen, dtype=torch.float64)
        zc = z - z.mean(dim=0, keepdim=True)
        ref = effective_rank(torch.linalg.svdvals(zc))
        fast = effective_rank(_svals(zc))
        rows.append({"n": n, "d": LIVE_D, "svdvals": round(ref, 9),
                     "gram": round(fast, 9),
                     "abs_diff": abs(ref - fast),
                     "rel_diff": abs(ref - fast) / ref})
    return {"checks": rows,
            "max_rel_diff": max(r["rel_diff"] for r in rows)}


def sample_ers(eigs, trials, gen, **kw) -> torch.Tensor:
    return torch.tensor([er_of(draw_batch(eigs, gen, **kw))
                         for _ in range(trials)], dtype=torch.float64)


# --------------------------------------------------------------------------- #
# 1. THE NULL — what the criterion does when nothing happened
# --------------------------------------------------------------------------- #
def q(x: torch.Tensor, p: float) -> float:
    return float(torch.quantile(x, p))


def summarise(x: torch.Tensor) -> dict:
    return {"n_trials": int(x.numel()), "mean": round(float(x.mean()), 4),
            "sd": round(float(x.std()), 4),
            "cv": round(float(x.std() / x.mean()), 4),
            "min": round(float(x.min()), 4), "p05": round(q(x, 0.05), 4),
            "p50": round(q(x, 0.50), 4), "p95": round(q(x, 0.95), 4),
            "max": round(float(x.max()), 4),
            "ratio_max_min": round(float(x.max() / x.min()), 3)}


def pairwise_fire_rate(a: torch.Tensor, b: torch.Tensor, thr: float) -> float:
    """P(b/a < thr) over independent pairings — the criterion applied to two
    single readings, which is how a gate comparing phases would use it."""
    m = min(a.numel(), b.numel())
    return float(((b[:m] / a[:m]) < thr).double().mean())


def run_null(eigs, trials, gen, rho_ep, rho_win, thr=0.8) -> dict:
    a = sample_ers(eigs, trials, gen, rho_ep=rho_ep, rho_win=rho_win)
    b = sample_ers(eigs, trials, gen, rho_ep=rho_ep, rho_win=rho_win)
    r = b / a
    return {"true_effective_rank": round(true_effective_rank(eigs), 4),
            "rank_ceiling": LIVE_N - 1,
            "reading": summarise(a),
            "ratio_null": summarise(r),
            "false_positive_rate_at_0.8": round(
                pairwise_fire_rate(a, b, thr), 4),
            # the threshold that WOULD give a 5 % false-positive rate, i.e. what
            # the criterion should have said if it were calibrated at all
            "threshold_for_5pct_FP": round(q(r, 0.05), 4)}


# --------------------------------------------------------------------------- #
# 2. POWER — the effect size the criterion can actually resolve
# --------------------------------------------------------------------------- #
def collapse_eigs(eigs: torch.Tensor, keep: int, floor: float) -> torch.Tensor:
    """A COLLAPSED population spectrum: the top ``keep`` directions untouched,
    everything below scaled by ``floor``. This is what representation collapse
    looks like — not a hard truncation (that is easier to detect), a squeeze."""
    e = eigs.clone()
    e[keep:] *= floor
    return e / e.sum()


def run_power(base_eigs, trials, gen, rho_ep, rho_win, thr=0.8) -> list[dict]:
    a = sample_ers(base_eigs, trials, gen, rho_ep=rho_ep, rho_win=rho_win)
    base_true = true_effective_rank(base_eigs)
    rows = []
    for keep, floor in [(47, 0.5), (32, 0.1), (16, 0.01), (8, 0.01),
                        (8, 1e-3), (4, 1e-4), (2, 1e-6), (1, 1e-8)]:
        ce = collapse_eigs(base_eigs, keep, floor)
        b = sample_ers(ce, trials, gen, rho_ep=rho_ep, rho_win=rho_win)
        rows.append({
            "keep": keep, "floor": floor,
            "true_effective_rank": round(true_effective_rank(ce), 4),
            "true_ratio": round(true_effective_rank(ce) / base_true, 4),
            "observed_reading": summarise(b),
            "observed_ratio_mean": round(float((b[:len(a)] / a).mean()), 4),
            "power_at_0.8": round(pairwise_fire_rate(a, b, thr), 4)})
    return rows


# --------------------------------------------------------------------------- #
# 3. SATURATION — does the reading track the truth at all?
# --------------------------------------------------------------------------- #
def run_saturation(d, trials, gen, rho_ep, rho_win) -> list[dict]:
    out = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
        e = powerlaw_eigs(d, alpha)
        x = sample_ers(e, trials, gen, rho_ep=rho_ep, rho_win=rho_win)
        out.append({"alpha": alpha,
                    "true_effective_rank": round(true_effective_rank(e), 3),
                    "observed": summarise(x)})
    return out


# --------------------------------------------------------------------------- #
# 4. CALIBRATION — is the live 15.13 a representation fact or a sampler fact?
# --------------------------------------------------------------------------- #
def run_calibration(d, trials, gen) -> list[dict]:
    grid = []
    for rho_ep, rho_win in [(0.0, 0.0), (0.0, 0.5), (0.0, 0.8),
                            (0.2, 0.6), (0.3, 0.6), (0.4, 0.5), (0.5, 0.4)]:
        for alpha in [0.0, 0.5, 1.0, 2.0]:
            e = powerlaw_eigs(d, alpha)
            x = sample_ers(e, trials, gen, rho_ep=rho_ep, rho_win=rho_win)
            s = summarise(x)
            grid.append({"rho_ep": rho_ep, "rho_win": rho_win, "alpha": alpha,
                         "true_effective_rank": round(
                             true_effective_rank(e), 3),
                         "observed": s,
                         "abs_err_vs_banked_mean": round(
                             abs(s["mean"] - BANKED["effective_rank_mean"]), 3),
                         "covers_banked_range": bool(
                             s["min"] <= BANKED["effective_rank_min"]
                             and s["max"] >= BANKED["effective_rank_max"])})
    grid.sort(key=lambda r: r["abs_err_vs_banked_mean"])
    return grid


# --------------------------------------------------------------------------- #
# 5. THE FIX — pooling calls, and a CLUSTER bootstrap on the pool
# --------------------------------------------------------------------------- #
def pooled_er(eigs, gen, n_calls, rho_ep, rho_win) -> tuple[float, torch.Tensor]:
    """``n_calls`` independent spectrum calls pooled into ONE estimate. Each call
    is a fresh batch, so pooling raises BOTH the row count and the number of
    independent clusters — which is the quantity the variance actually depends
    on."""
    zs = [draw_batch(eigs, gen, rho_ep=rho_ep, rho_win=rho_win)
          for _ in range(n_calls)]
    z = torch.cat(zs, dim=0)
    return er_of(z), z


def cluster_bootstrap_er(z: torch.Tensor, block: int, gen: torch.Generator,
                         reps: int = 128) -> dict:
    """Bootstrap CI resampling WINDOW BLOCKS (``block`` consecutive rows), not
    rows. Resampling rows would treat 6 near-duplicate frames as 6 independent
    facts and produce an interval that is too narrow — the same error class as
    ``overlapping_holdout_se`` in the eval harness.

    Runs on the PRECOMPUTED Gram: a resample is a submatrix gather plus a
    double-centring, so ``d`` is paid once instead of ``reps`` times."""
    zd = z.double()
    g_full = zd @ zd.T
    nb = zd.shape[0] // block
    off = torch.arange(block)
    vals = []
    for _ in range(reps):
        idx = torch.randint(0, nb, (nb,), generator=gen)
        rows = (idx[:, None] * block + off).reshape(-1)
        g = g_full[rows][:, rows]
        m = g.mean(0, keepdim=True)
        gc = g - m - m.T + g.mean()            # centred Gram
        ev = torch.linalg.eigvalsh(gc).clamp_min(0).flip(0)
        vals.append(effective_rank(ev.sqrt()))
    v = torch.tensor(vals, dtype=torch.float64)
    return {"lo": round(q(v, 0.025), 4), "hi": round(q(v, 0.975), 4),
            "sd": round(float(v.std()), 4), "reps": reps,
            "block_rows": block}


def run_banked_implied(gen: torch.Generator, trials: int = 20000,
                       thr: float = 0.8) -> dict:
    """The criterion's false-positive rate implied by the LIVE RUN'S OWN spread.

    The simulation above is a model; this is the run. The 38 banked records give
    mean 15.13 and range 3.37 -> 30.06 with NO trend (200 -> 16.75,
    4000 -> 12.10, 7600 -> 17.59). Fit a lognormal by matching that mean and by
    converting the observed log-range through the expected range of 38 normal
    draws (simulated here, not taken from a table), then ask what
    ``ratio < 0.8`` does under it.

    ⚠️ ASSUMPTION, stated because it moves the answer: this treats the WHOLE
    banked spread as sampling noise. Any genuine drift inside it makes this an
    UPPER bound on the false-positive rate, exactly as the model-based number is
    a LOWER bound (its simulated spread is narrower than the banked one). The
    two bracket the truth; neither is quoted alone.
    """
    n_rec = BANKED["records"]
    # expected range of n_rec iid N(0,1) — simulated, so no table is trusted
    draws = torch.randn(4000, n_rec, generator=gen, dtype=torch.float64)
    e_range = float((draws.max(dim=1).values - draws.min(dim=1).values).mean())
    log_range = math.log(BANKED["effective_rank_max"]) - math.log(
        BANKED["effective_rank_min"])
    sigma = log_range / e_range
    mu = math.log(BANKED["effective_rank_mean"]) - 0.5 * sigma ** 2
    # ratio of two independent readings: log-ratio ~ N(0, 2 sigma^2)
    sd_logratio = sigma * math.sqrt(2.0)
    fp = 0.5 * (1.0 + math.erf(math.log(thr) / (sd_logratio * math.sqrt(2.0))))
    z95 = 1.6448536269514722
    return {
        "source": "BANKED summary statistics (INHERITED — the 38 raw records "
                  "live on Thor, not in this repo)",
        "expected_range_of_38_normals_SIMULATED": round(e_range, 4),
        "sigma_log": round(sigma, 4), "mu_log": round(mu, 4),
        "implied_CV_of_one_reading": round(
            math.sqrt(math.exp(sigma ** 2) - 1.0), 4),
        "sd_of_log_ratio": round(sd_logratio, 4),
        "false_positive_rate_at_0.8": round(fp, 4),
        "threshold_for_5pct_FP": round(math.exp(-z95 * sd_logratio), 4),
        "reading_ratio_needed_at_5pct_FP": round(
            1.0 / math.exp(-z95 * sd_logratio), 3),
        "note": "an UPPER bound on FP: it charges the entire banked spread to "
                "noise. The model-based null is the LOWER bound."}


# --------------------------------------------------------------------------- #
# 6. DOES THE INTERVAL COVER? — a CI nobody has checked is a decoration
# --------------------------------------------------------------------------- #
def _er_from_gram(g: torch.Tensor) -> float:
    m = g.mean(0, keepdim=True)
    gc = g - m - m.T + g.mean()
    return effective_rank(torch.linalg.eigvalsh(gc).clamp_min(0).flip(0).sqrt())


def block_cis(z: torch.Tensor, block: int, gen: torch.Generator,
              reps: int = 128) -> dict:
    """Percentile, PIVOTAL (basic) and leave-one-block-out jackknife intervals
    from the same resamples, so the three can be compared on coverage.

    ⛔ WHY THE PERCENTILE ONE IS SUSPECT A PRIORI: bootstrap-with-replacement
    DUPLICATES blocks, and duplicated rows are exactly rank-deficient. For a
    RANK functional that is a systematic downward bias, not noise — so the
    percentile interval can sit entirely below the point estimate. Measured,
    not assumed, by :func:`run_coverage`.
    """
    zd = z.double()
    g_full = zd @ zd.T
    nb = zd.shape[0] // block
    off = torch.arange(block)
    theta = _er_from_gram(g_full)
    vals = []
    for _ in range(reps):
        idx = torch.randint(0, nb, (nb,), generator=gen)
        rows = (idx[:, None] * block + off).reshape(-1)
        vals.append(_er_from_gram(g_full[rows][:, rows]))
    v = torch.tensor(vals, dtype=torch.float64)
    lo, hi = q(v, 0.025), q(v, 0.975)
    # leave-one-block-out jackknife — no duplicates, so no rank deficiency
    jk = []
    for b in range(nb):
        keep = torch.cat([torch.arange(b * block),
                          torch.arange((b + 1) * block, nb * block)])
        jk.append(_er_from_gram(g_full[keep][:, keep]))
    j = torch.tensor(jk, dtype=torch.float64)
    se_jk = float(((nb - 1) / nb * ((j - j.mean()) ** 2).sum()).sqrt())
    return {"theta": theta,
            "percentile": (lo, hi),
            "pivotal": (2 * theta - hi, 2 * theta - lo),
            "jackknife": (theta - 1.96 * se_jk, theta + 1.96 * se_jk),
            "se_jackknife": se_jk}


def run_coverage(eigs, gen, rho_ep, rho_win, n_calls: int,
                 datasets: int = 60, reps: int = 96, d: int | None = None
                 ) -> dict:
    """Coverage of each interval against the ESTIMAND THAT IS IDENTIFIED at this
    n: the finite-n expectation E[ER_hat | n]. The POPULATION effective rank is
    NOT identified at n=48 and an interval claiming to cover it would be a lie
    — which is the whole point of stamping the ceiling in the record."""
    ref = torch.tensor([pooled_er(eigs, gen, n_calls, rho_ep, rho_win)[0]
                        for _ in range(max(datasets, 60))], dtype=torch.float64)
    target = float(ref.mean())
    hit = {"percentile": 0, "pivotal": 0, "jackknife": 0}
    width = {"percentile": [], "pivotal": [], "jackknife": []}
    for _ in range(datasets):
        _, z = pooled_er(eigs, gen, n_calls, rho_ep, rho_win)
        ci = block_cis(z, LIVE_W, gen, reps=reps)
        for k in hit:
            lo, hi = ci[k]
            hit[k] += int(lo <= target <= hi)
            width[k].append(hi - lo)
    return {"n_calls": n_calls, "n_rows": n_calls * LIVE_N,
            "estimand_E[ER|n]": round(target, 4),
            "true_population_effective_rank": round(
                true_effective_rank(eigs), 3),
            "datasets": datasets,
            "coverage": {k: round(v / datasets, 3) for k, v in hit.items()},
            "mean_width": {k: round(float(sum(w) / len(w)), 4)
                           for k, w in width.items()}}


#: Pool sizes simulated. ``32`` is the shipped default: 32 CONSECUTIVE trainer
#: steps x 48 rows = 1536 rows and ~128 distinct episodes, spanning ~14 min of
#: Thor wall-clock — long enough to buy clusters, short enough that the
#: representation has not drifted underneath the estimate.
POOLS = (1, 8, 32)


def run_fix(base_eigs, gen, rho_ep, rho_win, trials, thr=0.8) -> dict:
    out = {"note": "pooling N spectrum calls; rank ceiling = min(N*48-1, 2048)"}
    for n_calls in POOLS:
        ceiling = min(n_calls * LIVE_N - 1, LIVE_D)
        t = max(trials // (2 * n_calls), 25)
        a = torch.tensor([pooled_er(base_eigs, gen, n_calls, rho_ep, rho_win)[0]
                          for _ in range(t)], dtype=torch.float64)
        b = torch.tensor([pooled_er(base_eigs, gen, n_calls, rho_ep, rho_win)[0]
                          for _ in range(t)], dtype=torch.float64)
        r = b / a
        # a real collapse to compare against, at fixed severity
        ce = collapse_eigs(base_eigs, 16, 0.01)
        c = torch.tensor([pooled_er(ce, gen, n_calls, rho_ep, rho_win)[0]
                          for _ in range(t)], dtype=torch.float64)
        out[f"pool_{n_calls}"] = {
            "n_rows": n_calls * LIVE_N, "rank_ceiling": ceiling,
            "reading": summarise(a),
            "false_positive_rate_at_0.8": round(pairwise_fire_rate(a, b, thr), 4),
            "threshold_for_5pct_FP": round(q(r, 0.05), 4),
            "power_vs_keep16_floor0.01_at_0.8": round(
                pairwise_fire_rate(a, c, thr), 4),
            "collapsed_reading": summarise(c)}
    # one worked bootstrap CI at the pooled size we would ship
    k = POOLS[-1]
    erk, zk = pooled_er(base_eigs, gen, k, rho_ep, rho_win)
    out[f"bootstrap_demo_pool_{k}"] = {
        "effective_rank": round(erk, 4),
        "cluster_bootstrap_95CI": cluster_bootstrap_er(zk, LIVE_W, gen),
        "row_bootstrap_95CI_WRONG_UNIT": cluster_bootstrap_er(zk, 1, gen)}
    # and the same CI on ONE call, to show what a single reading is worth
    er1, z1 = pooled_er(base_eigs, gen, 1, rho_ep, rho_win)
    out["bootstrap_demo_pool_1"] = {
        "effective_rank": round(er1, 4),
        "cluster_bootstrap_95CI": cluster_bootstrap_er(z1, LIVE_W, gen)}
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    trials = 120 if a.quick else a.trials
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()

    # The healthy baseline: a mildly decaying power law over d=2048. alpha is
    # chosen in `calib`; alpha=1.0 is the reporting default and its true
    # effective rank is stated in every block, so nothing is implicit.
    base = powerlaw_eigs(LIVE_D, 1.0)

    res = {
        "meta": {"live_geometry": {"batch": LIVE_B, "window": LIVE_W,
                                   "eps_per_batch": LIVE_EPS, "n_rows": LIVE_N,
                                   "d_op": LIVE_D,
                                   "rank_ceiling": LIVE_N - 1,
                                   "source": "RESTART_v6F_SW.sh (argv of the "
                                             "RUNNING pid 25477)"},
                 "banked_records_INHERITED": BANKED,
                 "trials": trials, "seed": a.seed,
                 "torch": torch.__version__,
                 "fast_path_verification": verify_fast_path(gen)},
        "saturation": run_saturation(LIVE_D, trials, gen, 0.0, 0.0),
        "calibration": run_calibration(LIVE_D, max(trials // 2, 60), gen),
    }
    # the correlation regime that best reproduces the banked spread drives the
    # null/power numbers, so the answer is not conditioned on a guess
    best = res["calibration"][0]
    rho_ep, rho_win, alpha = best["rho_ep"], best["rho_win"], best["alpha"]
    res["meta"]["calibrated_regime"] = {
        "rho_ep": rho_ep, "rho_win": rho_win, "alpha": alpha,
        "abs_err_vs_banked_mean": best["abs_err_vs_banked_mean"],
        "covers_banked_range": best["covers_banked_range"]}
    cal = powerlaw_eigs(LIVE_D, alpha)

    res["null_iid_rows"] = run_null(base, trials, gen, 0.0, 0.0)
    res["null_calibrated"] = run_null(cal, trials, gen, rho_ep, rho_win)
    res["power_iid_rows"] = run_power(base, trials, gen, 0.0, 0.0)
    res["power_calibrated"] = run_power(cal, trials, gen, rho_ep, rho_win)
    res["fix_pooling_calibrated"] = run_fix(cal, gen, rho_ep, rho_win, trials)
    res["banked_implied_false_positive_rate"] = run_banked_implied(gen)
    # ⚠️ Coverage is measured at pools 1 and 8, NOT 32. Each bootstrap replicate
    # is an eigendecomposition of an n x n Gram, so a pool-32 coverage study is
    # O(1536^3) x reps x datasets = tens of TFLOP on a CPU-only box; pool 8 is
    # 64x cheaper and answers the same question, because WHICH interval covers
    # is a property of the resampling scheme (duplicate blocks are exactly
    # rank-deficient), not of the pool size. Stated because the omission would
    # otherwise look like a choice of the flattering configuration.
    res["interval_coverage"] = [
        run_coverage(cal, gen, rho_ep, rho_win, k,
                     datasets=(30 if a.quick else 60),
                     reps=(48 if a.quick else 96))
        for k in (1, 8)]
    res["meta"]["seconds"] = round(time.time() - t0, 1)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "sigreg_gate_power.json"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()
                      if k in ("meta", "null_iid_rows", "null_calibrated")},
                     indent=1))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
