"""EP-ADMISSION N1: the frozen A2 reliability statistic restricted to the CHANGED picks (EPADM-S0BANK DQ3, arm N1; design: ../SIM_DESIGN_N1.md, hashed BEFORE the first run).  CPU only.

WHAT IT IS.  The frozen gate's A2 reads, for each head h in (NC, DAC, TTC) and each candidate bank (= each search seed), the ratio of OBSERVED to PREDICTED unsafe mass at the candidate's picks,
    A2(h) = sum_t (1 - y_h[t, pick_c(t)]) / sum_t (1 - sigma_h[t, pick_c(t)])        (`epreliab_lib.reliability_stats`, "composite_picks", deficit form; a log-cluster bootstrap percentile CI),
and refuses a seed iff the LOWER CI of any head exceeds the margin (1.05 at the registration: `ep_admission.a2_head_ok_r1(ci, 1.05)`); the candidate is refused iff ANY seed is refused.
N1 changes ONE thing: the sum runs only over the CHANGED picks, i.e. over the tokens where this seed's pick differs from P2's pick,
    changed(t) = [ rule_pick(candidate rows of t) != rule_pick(P2 rows of t) ]        (the gate's own `pair_terms(...)["changed"]`),
    N1(h)      = sum_t changed(t) * (1 - y_h[t, pick_c(t)])  /  sum_t changed(t) * (1 - sigma_h[t, pick_c(t)]),
computed by the gate's own `Stats.ratio` (so the bootstrap, the draws and the percentile rule are the frozen ones), the margin test by the frozen `a2_head_ok_r1`, the "any seed" quantifier unchanged.
`changed` is per (token, seed): the three search seeds move different picks, so each seed has its OWN changed set.

UNDEFINED IS NOT A PASS.  A seed with ZERO changed picks (or zero predicted unsafe mass at them) has no N1 ratio: `n1_reliability` RAISES `N1Undefined`, `n1_verdict` returns "UNDEFINED-N1" (never "ADMIT-N1").
A definite refusal (FAIL-N1) outranks an undefined seed in the label; both are NON-admissions.

Used by `epadm_sim_n1.py` (the simulation) and, unchanged, on a real adapted bank once one exists (`n1_from_bank`).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C1_DIR = os.environ.get("EPADM_C1_DIR") or os.path.join(HERE, "..", "..", "2026-10-19-epadm-searchbank", "code")
sys.path.insert(0, C1_DIR)
import epadm_searchbank as SB                                                              # noqa: E402

A = SB.gate()                                                                              # the frozen, sha-checked ep_admission
HEADS = tuple(A.A2_HEADS)                                                                  # ("NC", "DAC", "TTC")
N1_MARGIN = 1.05                                                                           # MM_RULING_EPADM_R1.md section 2 (the registered A2 margin)


class N1Undefined(RuntimeError):
    """N1 has no value: no changed pick on a seed, or no predicted unsafe mass at the changed picks.  Never a pass."""


def changed_mask(pk_p, pk_c) -> np.ndarray:
    """The tokens whose pick on this candidate bank differs from P2's pick: the gate's `pair_terms(...)['changed']` (a slot-index difference)."""
    return np.asarray(pk_c) != np.asarray(pk_p)


def n1_ci_logvec(obs_ch_log: np.ndarray, pred_ch_log: np.ndarray, draws: np.ndarray) -> dict:
    """N1 from per-LOG sums over the changed picks (the simulation's shortcut; equal to the `Stats.ratio` path, tested): the registered ratio-of-sums bootstrap CI.  Refuses when the predicted mass is 0."""
    _, _, F = A.libs()
    if not float(np.sum(pred_ch_log)) > 0.0:
        raise N1Undefined("predicted unsafe mass at the changed picks is 0 (no changed pick in these logs): N1 is undefined, not a pass")
    return F.ratio_ci(obs_ch_log, pred_ch_log, draws)


def n1_reliability(S, lab: np.ndarray, zc: np.ndarray, pk_p: np.ndarray, pk_c: np.ndarray, heads: tuple = HEADS) -> dict:
    """N1 of ONE candidate bank (one search seed) through the frozen `Stats` object `S` (same draws, same log assignment as `analyze_candidate`).
    -> {"n_changed", "n_sets", "ci": {head: {point, lo, hi, ...}}}.  Raises N1Undefined on zero changed picks or zero predicted mass."""
    _, P, _ = A.libs()
    ch = changed_mask(pk_p, pk_c)
    n_ch = int(ch.sum())
    if n_ch == 0:
        raise N1Undefined("zero changed picks on this candidate bank: N1 is undefined, not a pass")
    chf = ch.astype(np.float64)
    out = {}
    for h in heads:
        col = A.HEAD_COL[h]
        y = lab[..., col].astype(np.float64)
        sg = P.sigmoid64(zc[..., col])
        ye, se = P.at(y, pk_c), P.at(sg, pk_c)
        pred = (1.0 - se) * chf
        if not float(pred.sum()) > 0.0:
            raise N1Undefined(f"{h}: predicted unsafe mass at the {n_ch} changed picks is 0: N1 is undefined, not a pass")
        out[h] = S.ratio((1.0 - ye) * chf, pred)
    return {"n_changed": n_ch, "n_sets": int(len(ch)), "ci": out}


def n1_seed_ok(cis: dict, margin: float = N1_MARGIN, heads: tuple = HEADS) -> bool:
    """One seed: every head's lower CI <= margin (the frozen `a2_head_ok_r1`; a NaN bound is not ok)."""
    return bool(all((h in cis) and A.a2_head_ok_r1(cis[h], margin) for h in heads))


def n1_verdict(per_seed: list, margin: float = N1_MARGIN, heads: tuple = HEADS) -> str:
    """The N1 verdict over the search seeds.  `per_seed` = one {head: CI} per seed, or None for a seed whose N1 is undefined.
    FAIL-N1 iff ANY defined seed is refused (the gate's 'any seed' quantifier); else UNDEFINED-N1 if ANY seed is undefined (or there is no seed); else ADMIT-N1."""
    if len(per_seed) == 0:
        return "UNDEFINED-N1"
    if any((c is not None) and not n1_seed_ok(c, margin, heads) for c in per_seed):
        return "FAIL-N1"
    if any(c is None for c in per_seed):
        return "UNDEFINED-N1"
    return "ADMIT-N1"


def n1_from_bank(bank_dir: str, margin: float = N1_MARGIN) -> dict:
    """N1 on an adapted bank (the adapter's layout: one P2 bank + one candidate bank per search seed), read THROUGH THE GATE'S LOADERS.  For a real bank this computes the candidate's A2-N1 CIs: do not call it
    on a real bank before the amendment that licenses the read is sealed.  Returns the per-seed results, the verdict at `margin` and the number of changed picks per seed."""
    p2, cands, _ = SB.load_with_gate(bank_dir)
    S = A.make_stats(p2["log_idx"], p2["n_logs"])
    _, P, _ = A.libs()
    pk_p = P.rule_pick(p2["z6"])
    per_seed, n_changed = [], {}
    for s, cand in enumerate(cands):                                                         # one candidate bank per search seed, in the adapter's seed order
        pk_c = P.rule_pick(cand["z6"])
        n_changed[s] = int(changed_mask(pk_p, pk_c).sum())
        try:
            per_seed.append(n1_reliability(S, p2["lab"], cand["z6"], pk_p, pk_c))
        except N1Undefined:
            per_seed.append(None)
    cis = [None if r is None else r["ci"] for r in per_seed]
    return {"verdict": n1_verdict(cis, margin), "margin": margin, "n_changed_by_seed": n_changed, "per_seed": per_seed}
