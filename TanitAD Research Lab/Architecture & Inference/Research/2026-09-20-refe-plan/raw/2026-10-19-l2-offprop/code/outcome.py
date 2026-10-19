#!/usr/bin/env python3
"""The L2-OFFPROP outcome mapping: DRAFT_PREREG_L2.md section 4 as amended by the Master Mind's rulings (MM_RULING_L2_BUILD.md, 2026-10-19), clause by clause.  Pure python, no numpy: it is the one function that turns numbers into a verdict, so it is small enough to be
mutated (`test_outcome.py` applies a textual RED mutation to EVERY branch and requires the scenario suite to fail on each).

    VOID          C-FEAT fails, OR >= 2 of 5 GLOBAL-null replicates exceed P2's A1-OOF by > 0.05 (the machinery manufactures concordance)             -> fix the instrument; no call
    L2-FLAT       A1-OOF lower CI <= H-null replicates' max point on EITHER seed                                                                    -> L2-BRANCH (then the trunk, P6)
    L2-ADMIT      on BOTH seeds: A1-OOF point >= 0.80 AND lower CI > H-null max AND A4 <= 0.10 AND A5 <= 0.05 AND in-fan(H) >= in-fan(H0) - 0.02    -> L1 Step B with the L2 head, fresh carve
    L2-PARTIAL    lower CI > H-null max on BOTH seeds, but the point < 0.80 or a non-degradation term fails                                         -> L2-BRANCH, own amendment

A1 is the POOLED lon + lat quantity (MM ruling E2: the one Step A's gate used, 0.6851); lon-only and lat-only are reported beside it and gate nothing.
H0 = the SAME recipe with the pairwise weight 0 (in-fan BCE only; same init, folds, seeds), compared at the SAME training seed.  P2's in-fan value is REPORTED (`infan_p2`) and NEVER gated: P2 is near in-sample
on its own in-fan labels, so an out-of-fold head refit on 400 tokens cannot be held to it (MM ruling: the clause "as drafted is unreachable by construction").

VOID is evaluated first (an instrument fault licenses no call); FLAT second; then ADMIT, else PARTIAL.  The four rows are exhaustive and exclusive given that order.

AMBIGUITIES the draft leaves and this module makes EXPLICIT PARAMETERS rather than silent choices (listed in READY.md for the Master Mind to settle when sealing):
  null_pool  "all"          : 'the H-null replicates' max point' = the max over ALL H-null cells (5 replicates x 2 training seeds = 10), the strictest reading.  "matched_seed": the 5 replicates of the same training seed.
  void_scope "any_seed"     : the VOID count is taken per training seed (5 replicates each); VOID if EITHER seed has >= 2 exceeding.  "pooled": count over all 10.
"""
from __future__ import annotations

A1_GATE = 0.80
A4_GATE = 0.10
A5_GATE = 0.05
INFAN_MARGIN = 0.02
VOID_EXCEED = 0.05
VOID_COUNT = 2


def void_reason(cfeat_pass, p2_a1, null_points, void_scope="any_seed"):
    if not cfeat_pass:
        return "C-FEAT failed"
    if void_scope == "pooled":
        groups = [[p for s in null_points for p in null_points[s]]]
    else:
        groups = [list(null_points[s]) for s in sorted(null_points)]
    for g in groups:
        n_exceed = sum(1 for p in g if p > p2_a1 + VOID_EXCEED)
        if n_exceed >= VOID_COUNT:
            return f"{n_exceed} of {len(g)} H-null replicates exceed P2's A1-OOF by > {VOID_EXCEED}"
    return None


def l2_outcome(cfeat_pass, p2_a1, reads, null_points, null_pool="all", void_scope="any_seed"):
    """reads: {seed: {"a1_point", "a1_lo", "a4_point", "a5_point", "infan", "infan_h0", ("infan_p2": reported, never read here)}};  null_points: {seed: [A1-OOF point of each GLOBAL-null replicate trained at that seed]}.
    Returns {"outcome", "why", "per_seed": {...}}."""
    why = void_reason(cfeat_pass, p2_a1, null_points, void_scope)
    if why is not None:
        return {"outcome": "VOID", "why": why, "per_seed": {}}
    seeds = sorted(reads)
    pooled_max = max(p for s in null_points for p in null_points[s])
    per = {}
    for s in seeds:
        r = reads[s]
        null_max = pooled_max if null_pool == "all" else max(null_points[s])
        per[s] = {"null_max": null_max,
                  "lower_ci_above_null": r["a1_lo"] > null_max,
                  "point_ge_gate": r["a1_point"] >= A1_GATE,
                  "a4_ok": r["a4_point"] <= A4_GATE,
                  "a5_ok": r["a5_point"] <= A5_GATE,
                  "infan_ok": r["infan"] >= r["infan_h0"] - INFAN_MARGIN,
                  "infan_vs_h0": r["infan"] - r["infan_h0"], "infan_p2_reported_not_gated": r.get("infan_p2")}
    lower_all = all(per[s]["lower_ci_above_null"] for s in seeds)
    if not lower_all:
        return {"outcome": "L2-FLAT", "why": "A1-OOF lower CI <= the H-null max on at least one seed", "per_seed": per}
    admit = all(per[s]["point_ge_gate"] and per[s]["a4_ok"] and per[s]["a5_ok"] and per[s]["infan_ok"] for s in seeds)
    if admit:
        return {"outcome": "L2-ADMIT", "why": "both seeds clear every term", "per_seed": per}
    return {"outcome": "L2-PARTIAL", "why": "lower CI above the null on both seeds, but a point gate or a non-degradation term fails", "per_seed": per}


NEXT_ARM = {"L2-ADMIT": "L1 Step B with the L2 EP head on a FRESH carve of navtrain held-out logs (not these 24), Step A re-run there first; then the EP-ADMISSION gate (--rules r1), then the shared navtest look",
            "L2-PARTIAL": "L2-BRANCH: the same pairs, the last scorer-branch block trainable as well (cached ctx; dev-box GPU; SH5's lockstep trainer); own amendment before any run",
            "L2-FLAT": "the EP head token does not carry within-slot EP ordering => L2-BRANCH, and if that is also flat, the trunk (P6)",
            "VOID": "fix the instrument; no call"}
