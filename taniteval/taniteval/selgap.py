"""TanitEval — SEL_GAP: the oracle-vs-selected gap for ANY fan+selector surface.

WHY THIS MODULE EXISTS (V18 backlog E8.1)
-----------------------------------------
Every level of the hierarchy is a *fan + selector*: the operative head proposes
an anchor fan and argmaxes a confidence, the tactical level proposes goal
candidates and picks one, the strategic level proposes routes/options and picks
one. In each case two numbers decide where the error lives:

  ``selected``  the error of the candidate the selector actually PICKED
  ``oracle``    the error of the best candidate that was IN the fan

and their difference, ``sel_gap = selected - oracle``, separates
*"the fan cannot propose it"* (oracle high) from *"the selector cannot find
it"* (oracle low, gap high). v5f's banked T0 eval carries exactly this triple
(ade 0.4011 / oracle_ade 0.1975 / sel_gap 0.2036 over 881 windows) — but
computed ad hoc inside ``stack/scripts/eval_flagship_v4.py``
(``collect_planner``: ``lg["sel_gap"]`` from ``v15_losses``, and the 4wp
``fan_err4.min(dim=1)`` cross-check) and inside each trainer separately
(``refc_train.py:479``, ``train_flagship_v16.py:364``). This module makes it a
first-class PER-LEVEL metric with a decision-grade interval.

⛔ BINDING FRAMING — read before quoting a number out of this module
-------------------------------------------------------------------
* **sel_gap is a PER-LEVEL metric and is NEVER pooled across levels.** An
  operative sel_gap, a tactical-goal sel_gap and a strategic-goal sel_gap are
  three different rows answering three different questions (which level's
  selector loses the coverage its fan bought); a composite over them hides
  exactly the trade-off the hierarchy programme exists to see. Same rule as
  the four-metric-families rule in ``CLAUDE.md`` — per family, never pooled.
* **The interval on the gap is the episode-cluster bootstrap, NEVER
  ``overlapping_holdout_se``.** Windows inside one episode (stride 8 over a
  ~199-frame clip) are strongly dependent, so the episode is the independent
  unit. The old estimator is not merely too narrow — its mean-of-split-means
  centre also *biases the point estimate* (measured 2026-07-25: −6.67 % to
  +11.69 % on headline ADE, sign flips on paired deltas). This module
  therefore delegates its CI to :func:`taniteval.ci.episode_cluster_bootstrap`
  **verbatim** — the resampling it uses is that function's own
  (``taniteval/ci.py``, ``episode_cluster_bootstrap`` lines 225–258, which
  draws episodes with replacement via ``_draws`` lines 158–164). Calling it,
  rather than re-implementing it, is what "mirror exactly" means here: the
  selgap CI can never drift from the program's decision-grade estimator.

The metric is agnostic to WHAT the per-candidate error is — dense ADE, 4wp
ADE, goal FDE, a route-choice cost — as long as it is a scalar per (window,
candidate). That is what generalises the instrument across the three levels.

Inputs are numpy arrays; torch tensors are accepted and converted (detached,
moved to CPU) so eval loops can pass their batched outputs directly.
"""
from __future__ import annotations

import numpy as np

from . import ci as _ci

__all__ = ["TOPK", "selgap", "selgap_report"]

BLOCK = "taniteval.selgap"
VERSION = "1.0.0"

#: the top-k oracle sizes reported by default. k >= C clips to C (== full oracle).
TOPK = (4, 8, 16)


def _to_numpy(x, dtype):
    """numpy view of ``x``; torch tensors are detached + moved to CPU first."""
    if hasattr(x, "detach"):                       # torch.Tensor, no torch import
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _validate(fan_err, sel_idx, eid):
    fan = _to_numpy(fan_err, np.float64)
    sel = _to_numpy(sel_idx, np.int64)
    if fan.ndim != 2:
        raise ValueError(f"fan_err must be [N, C] per-candidate scalar error, "
                         f"got shape {fan.shape}")
    n, c = fan.shape
    if c < 1:
        raise ValueError("fan_err needs at least one candidate (C >= 1)")
    if sel.shape != (n,):
        raise ValueError(f"sel_idx must be [N]={n}, got shape {sel.shape}")
    if len(eid) != n:
        raise ValueError(f"eid/fan_err length mismatch: {len(eid)} vs {n}")
    if sel.size and (sel.min() < 0 or sel.max() >= c):
        raise ValueError(f"sel_idx out of range [0, {c}): "
                         f"min {sel.min()}, max {sel.max()}")
    if n == 0:
        raise ValueError("selgap needs at least one window")
    return fan, sel, n, c


def _oracle_topk(fan, scores, ks):
    """``{k: mean over windows of min fan_err among the top-k candidates}``.

    Ranking, in order of preference:
    * ``scores`` given ([N, C], higher = preferred by the selector): per-window
      top-k by the SELECTOR's own scores — "what would an oracle re-ranker
      recover if the selector shortlisted k instead of committing to 1".
    * ``scores`` None: candidates ranked by their CORPUS-MEAN error (a static
      ranking of the fan slots) — "how much of the oracle is concentrated in
      the k statically-best candidates", a fan-compression diagnostic. This is
      the only ranking computable from ``fan_err`` alone; a per-window sort of
      the errors themselves would trivially return the full oracle at any k.

    Both rankings are monotone in k by construction (min over a superset).
    """
    n, c = fan.shape
    out, tag = {}, None
    if scores is not None:
        s = _to_numpy(scores, np.float64)
        if s.shape != (n, c):
            raise ValueError(f"scores must match fan_err shape {(n, c)}, "
                             f"got {s.shape}")
        order = np.argsort(-s, axis=1, kind="stable")      # per-window, best first
        ranked = np.take_along_axis(fan, order, axis=1)
        tag = "selector_scores"
    else:
        order = np.argsort(fan.mean(axis=0), kind="stable")  # static, best first
        ranked = fan[:, order]
        tag = "corpus_mean_error"
    for k in ks:
        kk = min(int(k), c)
        out[int(k)] = float(ranked[:, :kk].min(axis=1).mean())
    return out, tag


def selgap(fan_err, sel_idx, eid, n_boot=_ci.DEFAULT_N_BOOT, seed=0, *,
           scores=None, level=None, alpha=0.05, topk=TOPK) -> dict:
    """Oracle-vs-selected gap over a candidate fan, with an episode-cluster CI.

    Parameters
    ----------
    fan_err : [N, C] array/tensor
        Per-candidate scalar error for each of N windows and C candidates —
        ANY metric (dense ADE, 4wp ADE, goal FDE, route cost). Lower = better.
    sel_idx : [N] int array/tensor
        The selector's pick per window (index into the C candidates).
    eid : [N]
        Episode id per window — the RESAMPLING UNIT of the CI.
    n_boot, seed, alpha
        Passed straight to :func:`taniteval.ci.episode_cluster_bootstrap`.
    scores : optional [N, C]
        The selector's per-candidate scores (higher = preferred). When given,
        the top-k oracles use the selector's own ranking; otherwise they use
        the static corpus-mean-error ranking (see :func:`_oracle_topk`).
    level : optional str
        Hierarchy level tag stamped into the result (e.g. ``"operative"``,
        ``"tactical_goal"``, ``"strategic_goal"``). Purely provenance — it does
        NOT change the arithmetic — but it is what makes the ⛔ never-pool-
        across-levels rule auditable in a results file.

    Returns
    -------
    dict with (all headline floats at the program's 4 dp, per ``ci.DISPLAY_DP``):
      ``selected``       mean error of the picked candidate
      ``oracle``         mean of the per-window min over all C candidates
      ``gap``            selected − oracle (the full-set point estimate; the
                         bootstrap supplies the interval, it does not move it)
      ``gap_frac``       gap / selected (NaN when selected == 0)
      ``oracle_top{k}``  top-k oracles for k in ``topk`` (default 4/8/16)
      ``topk_ranking``   which ranking the top-k used (provenance)
      ``sel_rank_mean``  mean 0-based rank of the pick among its fan's errors
                         (rank = #candidates STRICTLY better; 0 = oracle pick)
      ``sel_rank_pct_mean`` / ``sel_rank_pct_median``
                         that rank as a percentile of C−1, mean and median
                         (0.0 = always the oracle pick, 1.0 = always the worst)
      ``gap_ci``         the FULL :func:`taniteval.ci.episode_cluster_bootstrap`
                         dict on the per-window gap — estimator, lo/hi, ci95,
                         se, n_episodes, n_boot all travel with the number.
      ``n_windows`` / ``n_candidates`` / ``n_episodes`` / ``level`` / ``block``
                         provenance.

    ⛔ sel_gap is a PER-LEVEL metric — never pool it across hierarchy levels.
    ⛔ The interval is the episode-cluster bootstrap (``taniteval/ci.py``,
    ``episode_cluster_bootstrap`` lines 225–258 + ``_draws`` 158–164, CALLED
    here, not copied), never ``overlapping_holdout_se``: that estimator both
    under-covers and biases the centre (CLAUDE.md, measured 2026-07-25).
    """
    fan, sel, n, c = _validate(fan_err, sel_idx, eid)

    sel_err = fan[np.arange(n), sel]                     # [N] error of the pick
    oracle_err = fan.min(axis=1)                          # [N] best-in-fan
    gap_pw = sel_err - oracle_err                         # [N] >= 0 by constr.

    selected = float(sel_err.mean())
    oracle = float(oracle_err.mean())
    gap = float(gap_pw.mean())
    gap_frac = float(gap / selected) if selected != 0.0 else float("nan")

    # rank of the pick among its own fan's errors (strictly-better count, so
    # ties with the oracle candidate still count as rank 0 — a selector that
    # picks an exact co-minimum is not penalised for the tie).
    rank = (fan < sel_err[:, None]).sum(axis=1).astype(np.float64)   # [N]
    pct = rank / (c - 1) if c > 1 else np.zeros_like(rank)

    topk_vals, topk_ranking = _oracle_topk(fan, scores, topk)

    # THE decision-grade interval, delegated verbatim — see module docstring.
    gap_ci = _ci.episode_cluster_bootstrap(gap_pw, eid, reduce="mean",
                                           n_boot=n_boot, seed=seed,
                                           alpha=alpha)

    out = {
        "block": BLOCK, "version": VERSION, "level": level,
        "selected": round(selected, _ci.DISPLAY_DP),
        "oracle": round(oracle, _ci.DISPLAY_DP),
        "gap": round(gap, _ci.DISPLAY_DP),
        "gap_frac": (round(gap_frac, _ci.DISPLAY_DP)
                     if np.isfinite(gap_frac) else float("nan")),
        "topk_ranking": topk_ranking,
        "sel_rank_mean": round(float(rank.mean()), _ci.DISPLAY_DP),
        "sel_rank_pct_mean": round(float(pct.mean()), _ci.DISPLAY_DP),
        "sel_rank_pct_median": round(float(np.median(pct)), _ci.DISPLAY_DP),
        "gap_ci": gap_ci,
        "n_windows": int(n),
        "n_candidates": int(c),
        "n_episodes": int(gap_ci["n_episodes"]),
        "_read": ("PER-LEVEL metric — never pool across hierarchy levels; "
                  "gap_ci is the episode-cluster bootstrap, never "
                  "overlapping_holdout_se"),
    }
    for k, v in topk_vals.items():
        out[f"oracle_top{k}"] = round(v, _ci.DISPLAY_DP)
    return out


def selgap_report(fan_err, sel_idx, eid, n_boot=_ci.DEFAULT_N_BOOT, seed=0, *,
                  scores=None, level=None, alpha=0.05, topk=TOPK) -> str:
    """:func:`selgap`, rendered as a compact printable text block.

    Same arguments, same arithmetic (it calls :func:`selgap`); the block quotes
    the gap WITH its interval and estimator on one line, because a gap printed
    without them is exactly the kind of bare number the estimator rule forbids.
    """
    r = selgap(fan_err, sel_idx, eid, n_boot=n_boot, seed=seed, scores=scores,
               level=level, alpha=alpha, topk=topk)
    b = r["gap_ci"]
    lvl = f" [{r['level']}]" if r["level"] else ""
    tk = "  ".join(f"top{k} {r[f'oracle_top{k}']:.4f}" for k in topk)
    gf = (f"{100.0 * r['gap_frac']:.1f}% of selected"
          if np.isfinite(r["gap_frac"]) else "n/a (selected == 0)")
    lines = [
        f"sel_gap{lvl}  N={r['n_windows']} windows / "
        f"{r['n_episodes']} episodes / C={r['n_candidates']} candidates",
        f"  selected   {r['selected']:.4f}",
        f"  oracle     {r['oracle']:.4f}   ({tk}; ranking={r['topk_ranking']})",
        f"  gap        {r['gap']:.4f}   [{b['lo']:.4f}, {b['hi']:.4f}] "
        f"95% {b['estimator']} (n_boot={b['n_boot']})",
        f"  gap_frac   {gf}",
        f"  sel rank   mean {r['sel_rank_mean']:.2f} "
        f"(pct mean {r['sel_rank_pct_mean']:.3f} / "
        f"median {r['sel_rank_pct_median']:.3f}; 0 = oracle pick)",
        "  ⛔ per-level metric — never pool across levels; interval is the "
        "episode-cluster bootstrap, never overlapping_holdout_se",
    ]
    return "\n".join(lines)
