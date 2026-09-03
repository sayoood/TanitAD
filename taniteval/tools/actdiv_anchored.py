#!/usr/bin/env python3
"""GS-8 — ANCHORED action-divergence: Delta-JEPA's Figure-6 read on the v7 predictor.

⭐ THE QUESTION (H-GS8-1). The banked ``actdiv`` instrument (MM-E10, ``actdiv_thor.py``
/ ``actdiv_local.py``) reads ONE number: the std of the h=1 prediction across rolled
action variants over the std across scenes — 0.004–0.006 on every 30k arm. That is a
MAGNITUDE read. It cannot see STRUCTURE: whether a *given* action moves the prediction
in a *consistent* direction, and whether a *larger* action moves it *further*. Delta-JEPA
(arXiv 2606.31232, §"Action-Sensitive Latent Dynamics", Figure 6) reads exactly that:

    keep the history fixed, replace the FINAL action with each candidate action ``a``,
    and measure the displacement against the predictor's OWN ZERO-ACTION prediction:

        d_i(a) = z_hat_{t+1}^{(i)}(a) − z_hat_{t+1}^{(i)}(0)

    then look at the ACTION-WISE MEANS  m(a) = mean_i d_i(a):  "well-separated action-wise
    mean responses, with larger action magnitudes generally inducing larger predicted
    shifts" = action-sensitive;  "means remain concentrated near the origin and
    substantially overlap" = insensitive (their LeWorldModel result).

This tool implements that read as a diagnostic with controls that MUST read known values.
No training change. It runs on the SAME diagnostic windows the banked ``actdiv`` used
(24 sorted clips of ``physicalai-val-…-w120-256x640cyl``, first 60 frames, 3-frame
stacks, W=6 windows at ``range(0, n, n // 5)`` — 144 windows/arm) so the two instruments
are read on one population.

WHAT IS REPORTED (per checkpoint, per predictor horizon)
--------------------------------------------------------
GRID (the Figure-6 read, primary):  candidates on two axes — channel 0 (``kappa``: the
  corpus' road-wheel-angle proxy ``atan(L·κ)``) and channel 1 (``accel``: a_long) — at
  ±{0.25, 0.5, 1, 2, 3}·σ of the diagnostic windows' own final actions, the other channel
  held at 0, plus the zero anchor. For every candidate: the mean displacement vector, its
  norm, and the norm relative to the scene spread (std of the zero-action prediction across
  windows — the banked C1 denominator, so the magnitude is on the old instrument's scale).
  Across candidates: the BETWEEN/WITHIN ratio ``F_sep`` (one-way pseudo-F of the
  displacement vectors by candidate), MONOTONICITY per axis (Spearman ρ of ‖m‖ against
  |level|), SIGN CONSISTENCY per level (cos(m(+L), m(−L)); a linear-ish response reads < 0),
  and the |a| binning (candidates pooled by |level| across both axes).
REALISED (the task's literal read, secondary): every window under its OWN final action
  vs the zero anchor, binned by |a| (σ-normalised), by κ and by a (quartiles); the same
  between/within ratio; its shuffled control FEEDS rolled actions to the windows and
  labels by the ORIGINAL bin.
LEGACY: the banked ``ratio_action_over_scene`` recomputed VERBATIM (8 rolled variants of
  the whole window's actions), so the new read sits beside the old number on the same
  windows — and under ``--actdiv-compat`` it must REPRODUCE the banked value (a known-value
  reproduction of the instrument, see below).

CONTROLS — each must read a KNOWN value or the panel is VOID
-------------------------------------------------------------
  C0 IDENTITY      the same inputs twice → max|Δ| EXACTLY 0.0 (deterministic forward).
  C1 SCENE REF     std of the zero-action prediction across windows must be ≫ 0, or the
                   ratio is 0/0 (the MM-E10 "degenerate in both directions" case).
  SHUFFLED LABELS  permute the candidate labels WITHIN each window → ``F_sep`` must fall
                   into its permutation band (reported: median, p95 of 200 permutations).
  SHUFFLED ACTIONS (realised mode) roll the actions across windows and label by the
                   original bin → the between/within ratio must fall to the floor.
  ZERO MODEL       a predictor that ignores its actions (this tool wraps the real
                   predictor so every candidate is fed the zero action) → every
                   displacement is EXACTLY 0, every norm 0, ``F_sep`` degenerate.
  SCALE (output)   multiplying every prediction by c leaves ``F_sep``, ρ, cos and the
                   relative magnitudes EXACTLY unchanged (the statistics are homogeneous);
                   asserted, not assumed.
  SCALE (input)    ``z ← c·z`` for c ∈ {0.5, 2}: a real re-forward. The verdict CLASS
                   should survive; if it does not, the reading is stamped scale-fragile.

⛔ HORIZONS. Only head ``1`` is ever trained (``tanitad/models/predictor.py``: heads 2/4
are allocated, computed and consumed by no loss; MM-E14 retracted the h2/h4 readings).
h=1 is the verdict horizon; other horizons are reported under ``untrained_heads`` as
INIT NOISE and must not be quoted.

⛔ TWO INSTRUMENT FACTS THIS TOOL MAKES EXPLICIT (both in the banked ``actdiv``):
  1. the banked scripts hardcode ``SPEED_SCALE = 30.0`` for the third action channel,
     while the trainer's ``_lift3`` (``scripts/train_v6_staged.py``) divides by
     ``tanitad.models.flagship_v15.SPEED_SCALE`` = 10.0 — the contract the checkpoints
     were trained with;
  2. the banked scripts feed the speed at the window's FIRST position, while the trainer
     feeds ``pose_last[:, 3]`` — the LAST position.
  The DEFAULT here is the TRAINER's contract (``_lift3`` imported from the trainer, the
  checkpoint's own ``cond_param``, v0 = last). ``--actdiv-compat`` reproduces the banked
  construction (v/30, v0 = first) so the banked number can be re-read on this box as a
  known-value check of the window selection and encoding.

TIER. ``T0-DIAGNOSTIC`` — a world-model probe. Nothing here is a driving-performance claim
(EVAL_DOCTRINE.md); the four metric families are REFUSED with the reason stamped.

PROVENANCE (MM-C12). ``load_trunk_auto`` REBUILDS the model from the checkpoint's config,
so the code tree is load-bearing: the tool preflights the ``tanitad`` import from the
requested stack, refuses a wrong tree, and stamps ``tanitad.__file__`` and every
checkpoint's md5 into the JSON.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import pathlib
import sys
import time

import numpy as np

# ---- path bootstrap, the taniteval-tools convention (seam_probe.py) -----------------
_HERE = os.path.dirname(os.path.abspath(__file__))     # <repo>/taniteval/tools
_TE_PARENT = os.path.dirname(_HERE)                    # <repo>/taniteval
_REPO = os.path.dirname(_TE_PARENT)                    # <repo>
for _pth in (os.path.join(_REPO, "stack"),
             os.path.join(_REPO, "stack", "scripts"), _TE_PARENT):
    if os.path.isdir(_pth) and _pth not in sys.path:
        sys.path.insert(0, _pth)

EVAL_TIER = "T0-DIAGNOSTIC"
HYPOTHESIS = "H-GS8-1"
PAPER = "2606.31232 (Delta-JEPA), Figure 6 / §Action-Sensitive Latent Dynamics"

#: banked ``actdiv`` window construction — VERBATIM constants
N_STACK = 3
F_MAX = 60
N_CLIPS_BANKED = 24
WINDOWS_PER_CLIP = 5           # range(0, n, max(1, n // 5))
LEGACY_N_VAR = 8
LEGACY_SPEED_SCALE = 30.0      # the banked scripts' hardcoded divisor (NOT the trainer's)

LEVELS_DEFAULT = (0.25, 0.5, 1.0, 2.0, 3.0)
AXIS_NAMES = ("kappa", "accel")           # action channel 0, channel 1
N_PERM_DEFAULT = 200
REALISED_BINS = 4                          # quartiles

#: PRE-REGISTERED thresholds (SPEC.md §4). Changing these after a run is a retraction.
THRESHOLDS = {
    "structured_f_over_null_p95": 5.0,     # F_sep >= 5 x p95 of the label-permutation null
    "monotone_rho_min": 0.8,               # Spearman(|level|, ||m||) on EACH axis
    "sign_cos_max": 0.0,                   # cos(m(+L), m(-L)) < 0 at EVERY level, each axis
    "material_rel_mag_at_2sigma": 0.0595,  # ||m(2σ)|| / scene_spread >= the MM-E19 bar
    "material_level": 2.0,
    "scene_spread_min": 1e-6,              # below: C1 degenerate -> VOID
}
VERDICTS = ("VOID", "INSENSITIVE", "SEPARATED-NONMONOTONE", "STRUCTURED-WEAK",
            "SENSITIVE")


def _p(*a):
    print(*a, flush=True)


def md5_of(path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# =====================================================================================
# statistics — numpy only, torch-free, unit-testable
# =====================================================================================
def spearman(x, y) -> float:
    """Spearman rank correlation with average ranks for ties (NaN if degenerate)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 3:
        return float("nan")

    def _rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=np.float64)
        sv = v[order]
        i = 0
        while i < v.size:
            j = i
            while j + 1 < v.size and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r

    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def candidate_grid(sigma, levels=LEVELS_DEFAULT, channels=(0, 1)) -> list[dict]:
    """The Figure-6 candidate set for a 2-channel action: one axis at a time, both
    signs, σ-multiples of the diagnostic windows' own final-action spread, plus the
    zero anchor (id ``zero``, level 0). ``channels[ax]`` is the action-channel index of
    axis ``AXIS_NAMES[ax]`` (v7: (steer≈κ, accel) = (0, 1); refav1: (a, κ) ⇒ (1, 0))."""
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    if sigma.size != 2:
        raise ValueError(f"need σ for exactly 2 action channels, got {sigma.size}")
    cands = [{"id": "zero", "axis": -1, "axis_name": "zero", "level": 0.0,
              "abs_level": 0.0, "a2": [0.0, 0.0]}]
    for ax, name in enumerate(AXIS_NAMES):
        ch = int(channels[ax])
        for L in levels:
            for sgn in (+1.0, -1.0):
                a2 = [0.0, 0.0]
                a2[ch] = float(sgn * L * sigma[ch])
                cands.append({"id": f"{name}{'+' if sgn > 0 else '-'}{L:g}",
                              "axis": ax, "axis_name": name, "channel": ch,
                              "level": float(sgn * L), "abs_level": float(L),
                              "a2": a2})
    return cands


def candidate_grid_abs(kappa_levels, accel_levels, channels=(1, 0)) -> list[dict]:
    """Candidates at ABSOLUTE physical levels (rad/m on the κ axis, m/s² on the a
    axis), one axis at a time, both signs, plus the zero anchor. Same record layout as
    :func:`candidate_grid` so every statistic downstream is shared."""
    cands = [{"id": "zero", "axis": -1, "axis_name": "zero", "level": 0.0,
              "abs_level": 0.0, "a2": [0.0, 0.0]}]
    per_axis = {"kappa": kappa_levels, "accel": accel_levels}
    for ax, name in enumerate(AXIS_NAMES):
        ch = int(channels[ax])
        for L in per_axis[name]:
            for sgn in (+1.0, -1.0):
                a2 = [0.0, 0.0]
                a2[ch] = float(sgn * L)
                cands.append({"id": f"{name}{'+' if sgn > 0 else '-'}{L:g}",
                              "axis": ax, "axis_name": name, "channel": ch,
                              "level": float(sgn * L), "abs_level": float(L),
                              "a2": a2})
    return cands


def axis_separation(D, cands, n_perm=N_PERM_DEFAULT, seed=0) -> dict:
    """Per-axis between/within ratio with its OWN label-permutation null, over that
    axis's candidates only (``D`` [N, C, S] aligned with ``cands``, zero excluded)."""
    out = {}
    for ax, name in enumerate(AXIS_NAMES):
        idx = [k for k, c in enumerate(cands) if c["axis"] == ax]
        if len(idx) < 2:
            continue
        bw = between_within(D[:, idx])
        null = shuffled_label_null(D[:, idx], n_perm=n_perm, seed=seed + ax + 1)
        out[name] = {"n_candidates": len(idx), "F_sep": bw["F_sep"],
                     "null": null,
                     "F_over_null_p95": (bw["F_sep"] / null["p95"] if null["p95"] > 0
                                         else float("nan")),
                     "mean_norms": {cands[k]["id"]: bw["norms"][j] for j, k in enumerate(idx)}}
    return out


def between_within(D, labels=None) -> dict:
    """One-way between/within ratio of displacement VECTORS.

    ``D``      [N, C, S]  (window i, candidate c)  — equal group sizes, or
    ``D``      [N, S] with ``labels`` [N]         — unequal group sizes (realised mode).

    F_sep = (SS_between / df_between) / (SS_within / df_within) with
      SS_between = Σ_g n_g ‖m_g − m̄‖²,  df_between = G − 1,
      SS_within  = Σ_g Σ_{i∈g} ‖d_i − m_g‖²,  df_within = N_total − G,
    where m̄ is the grand mean of ALL displacement vectors. Under label exchangeability
    E[F_sep] ≈ 1; the permutation band is MEASURED (``shuffled_label_null``), never
    assumed. Degenerate (all-zero) input reports ``degenerate=True`` and F_sep = 0.
    """
    D = np.asarray(D, dtype=np.float64)
    if labels is None:
        if D.ndim != 3:
            raise ValueError(f"D must be [N, C, S] without labels, got {D.shape}")
        N, C, S = D.shape
        flat = D.reshape(N * C, S)
        labels = np.tile(np.arange(C), N)
    else:
        if D.ndim != 2:
            raise ValueError(f"D must be [N, S] with labels, got {D.shape}")
        flat = D
        labels = np.asarray(labels)
    groups = np.unique(labels)
    G = len(groups)
    Ntot = flat.shape[0]
    grand = flat.mean(0)
    means, norms, sizes = [], [], []
    ss_b = 0.0
    ss_w = 0.0
    for g in groups:
        sel = flat[labels == g]
        m = sel.mean(0)
        means.append(m)
        norms.append(float(np.linalg.norm(m)))
        sizes.append(int(sel.shape[0]))
        ss_b += sel.shape[0] * float(((m - grand) ** 2).sum())
        ss_w += float(((sel - m) ** 2).sum())
    df_b = max(G - 1, 1)
    df_w = max(Ntot - G, 1)
    degenerate = bool(ss_w == 0.0 and ss_b == 0.0)
    if ss_w == 0.0:
        f_sep = 0.0 if ss_b == 0.0 else float("inf")
    else:
        f_sep = (ss_b / df_b) / (ss_w / df_w)
    return {"groups": [g.item() if hasattr(g, "item") else g for g in groups],
            "means": np.stack(means), "norms": norms, "sizes": sizes,
            "ss_between": ss_b, "ss_within": ss_w, "df_between": df_b,
            "df_within": df_w, "F_sep": float(f_sep), "degenerate": degenerate,
            "max_abs": float(np.abs(flat).max()) if flat.size else 0.0}


def shuffled_label_null(D, n_perm=N_PERM_DEFAULT, seed=0) -> dict:
    """Permute the candidate labels WITHIN each window and recompute F_sep."""
    D = np.asarray(D, dtype=np.float64)
    N, C, S = D.shape
    rng = np.random.default_rng(seed)
    vals = np.empty(n_perm, dtype=np.float64)
    for p in range(n_perm):
        perm = np.stack([rng.permutation(C) for _ in range(N)])          # [N, C]
        Dp = np.take_along_axis(D, perm[:, :, None], axis=1)
        vals[p] = between_within(Dp)["F_sep"]
    return {"n_perm": int(n_perm), "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)), "max": float(vals.max()),
            "mean": float(vals.mean())}


def axis_reads(bw, cands) -> dict:
    """Monotonicity + sign consistency per axis from the candidate means."""
    means = bw["means"]
    norms = np.asarray(bw["norms"])
    out = {}
    by_id = {c["id"]: k for k, c in enumerate(cands)}
    for ax, name in enumerate(AXIS_NAMES):
        idx = [k for k, c in enumerate(cands) if c["axis"] == ax]
        lv = np.array([cands[k]["abs_level"] for k in idx])
        nm = norms[idx]
        rho = spearman(lv, nm)
        levels = sorted(set(lv.tolist()))
        sign_cos = {}
        per_level_norm = {}
        for L in levels:
            kp = by_id[f"{name}+{L:g}"]
            km = by_id[f"{name}-{L:g}"]
            a, b = means[kp], means[km]
            den = float(np.linalg.norm(a) * np.linalg.norm(b))
            sign_cos[f"{L:g}"] = float((a * b).sum() / den) if den > 0 else float("nan")
            per_level_norm[f"{L:g}"] = float(0.5 * (norms[kp] + norms[km]))
        out[name] = {"spearman_rho_abslevel_vs_norm": rho,
                     "norm_by_level": per_level_norm,
                     "sign_cos_by_level": sign_cos,
                     "monotone": bool(np.isfinite(rho) and rho >= THRESHOLDS["monotone_rho_min"]),
                     "sign_consistent": bool(all(np.isfinite(v) and v < THRESHOLDS["sign_cos_max"]
                                                for v in sign_cos.values()))}
    # pooled |a| bins: candidates grouped by |level| across both axes
    pooled = {}
    for L in sorted(set(c["abs_level"] for c in cands if c["axis"] >= 0)):
        ks = [k for k, c in enumerate(cands) if c["axis"] >= 0 and c["abs_level"] == L]
        pooled[f"{L:g}"] = float(np.mean(norms[ks]))
    lv = np.array([float(k) for k in pooled])
    out["abs_a_pooled"] = {"norm_by_abs_level": pooled,
                           "spearman_rho": spearman(lv, np.array(list(pooled.values())))}
    return out


def summarize_grid(D, cands, scene_spread, n_perm=N_PERM_DEFAULT, seed=0) -> dict:
    """The full grid read from anchored displacements ``D`` [N, C, S] (C == len(cands),
    candidate 0 being the zero anchor whose displacement is identically 0 and which is
    EXCLUDED from the between/within statistic)."""
    D = np.asarray(D, dtype=np.float64)
    if D.shape[1] != len(cands):
        raise ValueError(f"D has {D.shape[1]} candidates, grid has {len(cands)}")
    zero_k = [k for k, c in enumerate(cands) if c["axis"] < 0]
    keep = [k for k in range(len(cands)) if k not in zero_k]
    zero_max = float(np.abs(D[:, zero_k]).max()) if zero_k else 0.0
    Dk = D[:, keep]
    ck = [cands[k] for k in keep]
    bw = between_within(Dk)
    null = shuffled_label_null(Dk, n_perm=n_perm, seed=seed)
    ax = axis_reads(bw, ck)
    ss = float(scene_spread)
    # ⛔ UNITS. ``scene_spread`` is the banked C1 denominator: the MEAN OVER DIMS of the
    # per-dim std of the zero-action prediction. The commensurate numerator is therefore
    # the PER-DIM RMS of the mean displacement, ‖m‖/√S — not the L2 norm ‖m‖, which is
    # √S (~45× at S = 2048) larger and would read a 0.006-ratio arm as "material".
    # (Caught on the first real read, 2026-09-03; the raw norms are stored unscaled.)
    root_s = float(np.sqrt(D.shape[2]))
    rel = {c["id"]: (bw["norms"][j] / root_s / ss if ss > 0 else float("nan"))
           for j, c in enumerate(ck)}
    L2 = f"{THRESHOLDS['material_level']:g}"
    at_l2 = [rel[c["id"]] for c in ck if f"{c['abs_level']:g}" == L2]
    if not at_l2:                      # an absolute-level grid (refav1): use each axis's top level
        top = {}
        for c in ck:
            top.setdefault(c["axis_name"], []).append(c)
        at_l2 = [rel[c["id"]] for cs in top.values()
                 for c in cs if c["abs_level"] == max(x["abs_level"] for x in cs)]
        L2 = "top-level-per-axis"
    rel_at_2s = float(np.mean(at_l2))
    per_axis = axis_separation(Dk, ck, n_perm=n_perm, seed=seed)
    return {
        "material_level_used": L2,
        "per_axis": per_axis,
        "n_windows": int(D.shape[0]), "n_candidates": int(len(ck)),
        "state_dim": int(D.shape[2]),
        "zero_anchor_max_abs": zero_max,
        "F_sep": bw["F_sep"], "degenerate": bw["degenerate"],
        "ss_between": bw["ss_between"], "ss_within": bw["ss_within"],
        "null": null,
        "F_over_null_p95": (bw["F_sep"] / null["p95"] if null["p95"] > 0 else float("nan")),
        "scene_spread": ss,
        "rel_units": "per-dim RMS of the mean displacement (‖m‖/√S) over the per-dim scene std",
        "per_candidate": {c["id"]: {"axis": c["axis_name"], "level": c["level"],
                                    "a2": c["a2"], "mean_norm": bw["norms"][j],
                                    "rel_to_scene": rel[c["id"]]}
                          for j, c in enumerate(ck)},
        "axes": ax,
        "rel_mag_at_material_level": rel_at_2s,
        "max_abs_displacement": bw["max_abs"],
    }


def verdict(summary, thresholds=THRESHOLDS, c0_max_abs=0.0, zero_model_max_abs=0.0) -> dict:
    """The pre-registered reading (SPEC.md §4). Order matters: VOID first."""
    reasons = []
    if c0_max_abs != 0.0:
        reasons.append(f"C0 identity max|Δ| = {c0_max_abs:.3e} ≠ 0 (non-deterministic forward)")
    if zero_model_max_abs != 0.0:
        reasons.append(f"zero-model control max|d| = {zero_model_max_abs:.3e} ≠ 0")
    if not (summary["scene_spread"] > thresholds["scene_spread_min"]):
        reasons.append(f"C1 scene spread {summary['scene_spread']:.3e} ~ 0 (degenerate)")
    if summary.get("degenerate"):
        reasons.append("all displacements identically zero")
    if reasons:
        return {"verdict": "VOID", "reasons": reasons}
    f_ok = np.isfinite(summary["F_over_null_p95"]) and \
        summary["F_over_null_p95"] >= thresholds["structured_f_over_null_p95"]
    ax = summary["axes"]
    mono = all(ax[n]["monotone"] for n in AXIS_NAMES)
    sign = all(ax[n]["sign_consistent"] for n in AXIS_NAMES)
    material = np.isfinite(summary["rel_mag_at_material_level"]) and \
        summary["rel_mag_at_material_level"] >= thresholds["material_rel_mag_at_2sigma"]
    flags = {"separated": bool(f_ok), "monotone_both_axes": bool(mono),
             "sign_consistent_both_axes": bool(sign), "material": bool(material)}
    if not f_ok:
        v = "INSENSITIVE"
    elif not (mono and sign):
        v = "SEPARATED-NONMONOTONE"
    elif not material:
        v = "STRUCTURED-WEAK"
    else:
        v = "SENSITIVE"
    return {"verdict": v, "flags": flags, "thresholds": dict(thresholds)}


def output_scale_invariance(D, cands, scene_spread, c=10.0, n_perm=50, seed=0) -> dict:
    """The homogeneity assertion: scaling every prediction by ``c`` scales D AND the
    scene spread by ``c``; F_sep, ρ, cos and the relative magnitudes must be EXACTLY
    unchanged (to float rounding)."""
    a = summarize_grid(D, cands, scene_spread, n_perm=n_perm, seed=seed)
    b = summarize_grid(np.asarray(D) * c, cands, scene_spread * c, n_perm=n_perm, seed=seed)

    def _rel(x, y):
        if not (np.isfinite(x) and np.isfinite(y)):
            return 0.0 if (np.isnan(x) and np.isnan(y)) else float("inf")
        return abs(x - y) / max(abs(x), abs(y), 1e-300)

    checks = {"F_sep": _rel(a["F_sep"], b["F_sep"]),
              "rel_mag_at_material_level": _rel(a["rel_mag_at_material_level"],
                                                b["rel_mag_at_material_level"])}
    for n in AXIS_NAMES:
        checks[f"rho_{n}"] = _rel(a["axes"][n]["spearman_rho_abslevel_vs_norm"],
                                  b["axes"][n]["spearman_rho_abslevel_vs_norm"])
        for L, v in a["axes"][n]["sign_cos_by_level"].items():
            checks[f"cos_{n}_{L}"] = _rel(v, b["axes"][n]["sign_cos_by_level"][L])
    worst = max(checks.values())
    # 1e-6, not machine epsilon: the predictions arrive as float32 and a near-degenerate
    # within-scatter (F_sep ~ 1e4-1e5 on a linear synthetic) cancels to ~1e-8 relative.
    return {"c": c, "max_rel_dev": float(worst), "invariant": bool(worst < 1e-6),
            "checks": checks}


# =====================================================================================
# realised mode (the task's literal read) — numpy statistics
# =====================================================================================
def quantile_bins(x, n_bins=REALISED_BINS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(qs, x, side="right")


def realised_reads(d_true, d_rolled, a2_last, sigma) -> dict:
    """``d_true`` [N, S]: each window under its OWN final action vs the zero anchor;
    ``d_rolled`` list of [N, S]: the same windows fed ROLLED actions (window i gets
    window i+j's final action) — the shuffled-ACTION control, labelled by window i's
    ORIGINAL bin. Bins: |a| (σ-normalised norm), κ (channel 0), a (channel 1)."""
    a2 = np.asarray(a2_last, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    keys = {"abs_a": np.linalg.norm(a2 / np.where(sig > 0, sig, 1.0), axis=1),
            "kappa": a2[:, 0], "accel": a2[:, 1]}
    out = {}
    for name, key in keys.items():
        lab = quantile_bins(key)
        bw = between_within(np.asarray(d_true), lab)
        rolled = [between_within(np.asarray(dr), lab)["F_sep"] for dr in d_rolled]
        # bin means keyed by the bin's mean |key| so a monotone read is possible
        centers = [float(np.mean(np.abs(key[lab == g]))) for g in bw["groups"]]
        out[name] = {"bins": int(len(bw["groups"])), "sizes": bw["sizes"],
                     "bin_centers_abs": centers, "bin_mean_norms": bw["norms"],
                     "F_sep": bw["F_sep"],
                     "F_sep_shuffled_actions": rolled,
                     "F_sep_shuffled_actions_max": float(max(rolled)) if rolled else float("nan"),
                     "spearman_rho_center_vs_norm": spearman(centers, bw["norms"])}
    return out


# =====================================================================================
# torch glue — encoding, window selection (banked actdiv VERBATIM), lifts
# =====================================================================================
def frames_of(path):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    lens = d["jpeg_len"].numpy().tolist()
    off = [0]
    for L in lens:
        off.append(off[-1] + int(L))
    return d, raw, off, len(lens)


def encode_clip(world, path, dev, max_frames=F_MAX):
    """STRIDE-1 latents through THIS arm's encoder — the banked ``actdiv`` encoder
    VERBATIM (3-frame stacks oldest-first, the newest frame's index pairs with its
    pose/action row — the same convention ``v2_dataset._decode_stacked`` uses).
    Returns ``(z [n, S], actions [n, 2], speed [n])``."""
    import torch
    from PIL import Image
    d, raw, off, n = frames_of(path)
    n = min(n, max_frames)
    imgs = []
    for i in range(n):
        im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")
        imgs.append(torch.from_numpy(np.asarray(im).copy())
                    .permute(2, 0, 1).float() / 255.0)
    if not imgs or float(imgs[0].abs().mean()) == 0.0:
        raise SystemExit(f"[FATAL] {path} decoded to all-zero frames")
    Z, B = [], 16
    with torch.no_grad():
        for s in range(0, n, B):
            chunk = []
            for i in range(s, min(s + B, n)):
                idx = [max(i - j, 0) for j in range(N_STACK - 1, -1, -1)]
                chunk.append(torch.cat([imgs[k] for k in idx], 0))
            x = torch.stack(chunk)[:, None].to(dev)
            Z.append(world.encode_window(x)[:, 0].float().cpu())
    return torch.cat(Z), d["actions"].float()[:n], d["poses"].float()[:n, 3]


def window_starts(n_usable: int) -> list[int]:
    """The banked ``actdiv`` window starts: ``range(0, n, max(1, n // 5))``."""
    if n_usable < 4:
        return []
    return list(range(0, n_usable, max(1, n_usable // WINDOWS_PER_CLIP)))


def select_windows(world, clips, dev, max_frames=F_MAX):
    """Banked ``actdiv`` window selection VERBATIM, but keeping BOTH the first and the
    last speed of each window so either lift convention can be applied."""
    import torch
    W = int(world.window)
    Z, A, V0, V1, CID = [], [], [], [], []
    for c in clips:
        z, act, spd = encode_clip(world, c, dev, max_frames)
        n = min(len(z) - W, len(act) - W, len(spd) - W)
        for i in window_starts(n):
            Z.append(z[i:i + W]); A.append(act[i:i + W])
            V0.append(spd[i]); V1.append(spd[i + W - 1]); CID.append(os.path.basename(c))
    if len(Z) < 16:
        raise SystemExit(f"[REFUSED] too few windows ({len(Z)})")
    return (torch.stack(Z), torch.stack(A), torch.stack(V0).reshape(-1),
            torch.stack(V1).reshape(-1), CID)


def lift_trainer(a2, v0, cond_param):
    """The TRAINER's own lift — ``_lift3`` imported from ``train_v6_staged``."""
    from train_v6_staged import _lift3  # noqa: PLC0415
    return _lift3(a2, v0, cond_param)


def lift_legacy(a2, v_first):
    """The banked ``actdiv`` lift VERBATIM: [a2, v_first / 30] over the window."""
    import torch
    W = a2.shape[1]
    return torch.cat([a2, (v_first / LEGACY_SPEED_SCALE)[:, None, None].expand(-1, W, -1)], -1)


def with_last_action(a3, a2_new, replace="last"):
    """Return a copy of ``a3`` [N, W, 3] with channels 0:2 of the LAST (or ALL)
    positions replaced by ``a2_new`` [N, 2] (or a broadcastable [2])."""
    import torch
    out = a3.clone()
    a2_new = torch.as_tensor(a2_new, dtype=a3.dtype, device=a3.device)
    if a2_new.dim() == 1:
        a2_new = a2_new[None].expand(a3.shape[0], -1)
    if replace == "last":
        out[:, -1, :2] = a2_new
    elif replace == "all":
        out[:, :, :2] = a2_new[:, None, :].expand(-1, a3.shape[1], -1)
    else:
        raise ValueError(f"replace must be 'last' or 'all', got {replace!r}")
    return out


class ZeroActionModel:
    """The ZERO-MODEL control: the real predictor fed the zero action whatever it is
    given, so every anchored displacement is EXACTLY 0."""

    def __init__(self, predictor, replace="last"):
        self.predictor = predictor
        self.replace = replace

    def __call__(self, zs, a3):
        return self.predictor(zs, with_last_action(a3, [0.0, 0.0], self.replace))


def anchored_displacements(predictor, zs, a3, cands, horizons, replace="last"):
    """d_i(a) = z_hat^{(i)}(a) − z_hat^{(i)}(0) for every candidate, every horizon.
    Returns ``{h: D [N, C, S]}``, ``{h: zero prediction [N, S]}``, and the C0 read."""
    import torch
    with torch.no_grad():
        a0 = with_last_action(a3, [0.0, 0.0], replace)
        base = predictor(zs, a0)
        again = predictor(zs, a0)
        c0 = max(float((again[h].float() - base[h].float()).abs().max()) for h in horizons)
        D = {h: [] for h in horizons}
        for c in cands:
            out = predictor(zs, with_last_action(a3, c["a2"], replace))
            for h in horizons:
                D[h].append((out[h].float() - base[h].float()).cpu().numpy())
    D = {h: np.stack(D[h], axis=1) for h in horizons}            # [N, C, S]
    zero = {h: base[h].float().cpu().numpy() for h in horizons}
    return D, zero, c0


def realised_displacements(predictor, zs, a3, horizons, n_var=LEGACY_N_VAR, replace="last"):
    """Each window under its OWN final action vs the zero anchor, plus ``n_var-1``
    ROLLED variants (window i fed window (i+j)'s final action) for the shuffled-ACTION
    control. Returns ``{h: d_true [N, S]}``, ``{h: [d_rolled_j [N, S], ...]}``."""
    import torch
    with torch.no_grad():
        base = predictor(zs, with_last_action(a3, [0.0, 0.0], replace))
        own = predictor(zs, a3 if replace == "last" else with_last_action(a3, a3[:, -1, :2], replace))
        d_true = {h: (own[h].float() - base[h].float()).cpu().numpy() for h in horizons}
        d_roll = {h: [] for h in horizons}
        for j in range(1, n_var):
            aj = with_last_action(a3, torch.roll(a3[:, -1, :2], shifts=j, dims=0), replace)
            out = predictor(zs, aj)
            for h in horizons:
                d_roll[h].append((out[h].float() - base[h].float()).cpu().numpy())
    return d_true, d_roll


def legacy_ratio(predictor, zs, a3, horizons, n_var=LEGACY_N_VAR) -> dict:
    """The banked ``actdiv`` compute section VERBATIM: roll the WHOLE window's 3-channel
    actions by j = 1..n_var-1, std across variants over std across windows."""
    import torch
    out = {}
    with torch.no_grad():
        base = predictor(zs, a3)
        for h in horizons:
            b_h = base[h].float()
            ident = float((predictor(zs, a3)[h].float() - b_h).abs().max())
            preds = [b_h]
            for j in range(1, n_var):
                preds.append(predictor(zs, torch.roll(a3, shifts=j, dims=0))[h].float())
            P = torch.stack(preds, 0)
            act_sp = float(P.std(dim=0).mean())
            scene_sp = float(b_h.std(dim=0).mean())
            out[f"h{h}"] = {"action_spread": act_sp, "scene_spread": scene_sp,
                            "ratio_action_over_scene": (act_sp / scene_sp if scene_sp > 1e-9 else None),
                            "C0_identity_max_abs_diff": ident, "C0_passes": ident == 0.0}
    return out


# =====================================================================================
# the per-arm read
# =====================================================================================
def read_arm(predictor, zs, a3, horizons, trained_horizons, *, levels=LEVELS_DEFAULT,
             n_perm=N_PERM_DEFAULT, seed=0, replace="last", scale_control=True,
             sigma=None, a2_last=None) -> dict:
    """Everything the read needs from a predictor and a window batch. ``predictor`` is
    any callable ``(states [N, W, S], actions [N, W, 3]) -> {h: [N, S]}``."""
    import torch
    a2_last = a3[:, -1, :2].float().cpu().numpy() if a2_last is None else np.asarray(a2_last)
    sigma = a2_last.std(0) if sigma is None else np.asarray(sigma)
    cands = candidate_grid(sigma, levels)
    D, zero, c0 = anchored_displacements(predictor, zs, a3, cands, horizons, replace)
    zm = ZeroActionModel(predictor, replace)
    Dz, _, _ = anchored_displacements(zm, zs, a3, cands, horizons, replace)
    d_true, d_roll = realised_displacements(predictor, zs, a3, horizons, replace=replace)
    per_h = {}
    for h in horizons:
        scene_spread = float(zero[h].std(0).mean())
        summ = summarize_grid(D[h], cands, scene_spread, n_perm=n_perm, seed=seed)
        zero_model_max = float(np.abs(Dz[h]).max())
        entry = {"trained_head": bool(h in trained_horizons),
                 "grid": summ,
                 "controls": {"C0_identity_max_abs_diff": c0, "C0_passes": c0 == 0.0,
                              "C1_scene_spread": scene_spread,
                              "zero_model_max_abs_displacement": zero_model_max,
                              "zero_model_passes": zero_model_max == 0.0,
                              "shuffled_labels_null": summ["null"]},
                 "realised": realised_reads(d_true[h], d_roll[h], a2_last, sigma)}
        if scale_control:
            entry["controls"]["output_scale"] = output_scale_invariance(
                D[h], cands, scene_spread, c=10.0, n_perm=min(50, n_perm), seed=seed)
        entry["verdict"] = verdict(summ, c0_max_abs=c0, zero_model_max_abs=zero_model_max)
        per_h[f"h{h}"] = entry
    out = {"sigma_final_action": sigma.tolist(), "candidates": cands,
           "replace": replace, "horizons": per_h}
    if scale_control:
        inp = {}
        for c in (0.5, 2.0):
            Dc, zc, _ = anchored_displacements(predictor, zs * c, a3, cands, horizons, replace)
            sc = {}
            for h in horizons:
                ss = float(zc[h].std(0).mean())
                summ_c = summarize_grid(Dc[h], cands, ss, n_perm=min(50, n_perm), seed=seed)
                vc = verdict(summ_c, c0_max_abs=c0)
                sc[f"h{h}"] = {"F_sep": summ_c["F_sep"], "F_over_null_p95": summ_c["F_over_null_p95"],
                               "rel_mag_at_material_level": summ_c["rel_mag_at_material_level"],
                               "verdict": vc["verdict"],
                               "verdict_invariant": vc["verdict"] == per_h[f"h{h}"]["verdict"]["verdict"]}
            inp[f"x{c:g}"] = sc
        out["input_scale_control"] = inp
    return out


def _compact_table(arm, res, horizons, trained):
    for h in horizons:
        e = res["horizons"][f"h{h}"]
        g = e["grid"]
        ax = g["axes"]
        tag = "" if h in trained else "   [UNTRAINED HEAD — init noise, not quotable]"
        _p(f"  [{arm}] h={h}{tag}")
        _p(f"     F_sep {g['F_sep']:.3f}  null(median {g['null']['median']:.3f}, p95 "
           f"{g['null']['p95']:.3f})  F/p95 {g['F_over_null_p95']:.2f}  scene_spread "
           f"{g['scene_spread']:.5f}")
        for n in AXIS_NAMES:
            a = ax[n]
            _p(f"     {n:<6} rho {a['spearman_rho_abslevel_vs_norm']:+.3f}  norms "
               + " ".join(f"{k}σ:{v:.2e}" for k, v in a["norm_by_level"].items())
               + "  cos(+,-) " + " ".join(f"{v:+.2f}" for v in a["sign_cos_by_level"].values()))
        _p(f"     rel-mag@{THRESHOLDS['material_level']:g}σ {g['rel_mag_at_material_level']:.5f}"
           f"  (bar {THRESHOLDS['material_rel_mag_at_2sigma']})  C0 {e['controls']['C0_identity_max_abs_diff']:.1e}"
           f"  zero-model {e['controls']['zero_model_max_abs_displacement']:.1e}"
           f"  -> {e['verdict']['verdict']}")


# =====================================================================================
# CLI
# =====================================================================================
def resolve_stack(requested: str | None) -> pathlib.Path:
    """MM-C12: an explicit stack that does not hold ``tanitad/__init__.py`` is a
    REFUSAL; otherwise <repo>/stack (this file's own repo)."""
    cand = pathlib.Path(requested) if requested else pathlib.Path(_REPO) / "stack"
    if not (cand / "tanitad" / "__init__.py").is_file():
        _p(f"[REFUSED] stack {cand} does not hold tanitad/__init__.py")
        raise SystemExit(2)
    sp = str(cand.resolve())
    for p in (sp, os.path.join(sp, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return cand.resolve()


def preflight(stack: pathlib.Path) -> str:
    import tanitad
    got = pathlib.Path(tanitad.__file__).resolve()
    if stack not in got.parents:
        _p(f"[REFUSED] tanitad imported from the WRONG TREE: {got} (requested {stack}) — MM-C12")
        raise SystemExit(2)
    import tanitad.eval.v6_probe_trunk  # noqa: F401  — fail in 2 s, not after the encode
    import train_v6_staged  # noqa: F401
    _p(f"  [stack] tanitad imported from: {got}")
    return str(got)


def parse_arms(a) -> list[tuple[str, pathlib.Path]]:
    arms = []
    for spec in (a.arm or []):
        if "=" not in spec:
            raise SystemExit(f"--arm expects NAME=PATH, got {spec!r}")
        n, p = spec.split("=", 1)
        arms.append((n, pathlib.Path(p)))
    if a.assets and a.arms:
        for n in str(a.arms).split(","):
            if n:
                arms.append((n, pathlib.Path(a.assets) / f"v7tiny_{n}" / "ckpt.pt"))
    if not arms:
        raise SystemExit("no arms: pass --arm NAME=PATH (repeatable) or --assets DIR --arms a,b")
    return arms


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="GS-8 anchored action-divergence (Delta-JEPA Fig. 6)")
    ap.add_argument("--arm", action="append", help="NAME=PATH to ckpt.pt (repeatable)")
    ap.add_argument("--assets", default="", help="dir holding v7tiny_<arm>/ckpt.pt")
    ap.add_argument("--arms", default="", help="comma-separated arm names under --assets")
    ap.add_argument("--corpus", required=False, default="",
                    help="dir of *.v2ep.pt clips — the banked actdiv corpus "
                         "(physicalai-val-…-w120-256x640cyl, first 24 sorted clips)")
    ap.add_argument("--nclips", type=int, default=N_CLIPS_BANKED)
    ap.add_argument("--max-frames", type=int, default=F_MAX)
    ap.add_argument("--stack", default="")
    ap.add_argument("--device", default="", choices=("", "cuda", "cpu"))
    ap.add_argument("--out", required=False, default="")
    ap.add_argument("--levels", default=",".join(f"{x:g}" for x in LEVELS_DEFAULT))
    ap.add_argument("--replace", default="last", choices=("last", "all"),
                    help="which window positions the candidate replaces (paper: last)")
    ap.add_argument("--also-replace-all", action="store_true",
                    help="additionally run the replace=all variant")
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-scale-control", action="store_true")
    ap.add_argument("--actdiv-compat", action="store_true",
                    help="reproduce the banked actdiv lift (v_first / 30) instead of the "
                         "trainer's _lift3 (cond_param, v_last / SPEED_SCALE)")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic predictors only: assert every control reads its known value")
    _add_refav1_args(ap)
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if a.family == "refav1":
        if not a.out:
            raise SystemExit("--out is required")
        return refav1_main(a)

    import torch
    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
    stack = resolve_stack(a.stack or None)
    tanitad_file = preflight(stack)
    from tanitad.eval.v6_probe_trunk import load_trunk_auto  # noqa: PLC0415
    from tanitad.models.flagship_v15 import SPEED_SCALE  # noqa: PLC0415
    from train_v6_staged import COND_INCUMBENT  # noqa: PLC0415

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not a.corpus or not a.out:
        raise SystemExit("--corpus and --out are required (unless --selftest)")
    clips = sorted(glob.glob(os.path.join(a.corpus, "*.v2ep.pt")))[:a.nclips]
    if len(clips) < a.nclips:
        _p(f"[REFUSED] corpus {a.corpus} holds {len(clips)} clips, --nclips {a.nclips} "
           f"requested — n would differ from the banked instrument")
        return 2
    levels = tuple(float(x) for x in a.levels.split(","))
    lift_desc = ("BANKED-ACTDIV-COMPAT: [a2, v_first/30.0] (NOT the trainer's contract)"
                 if a.actdiv_compat else
                 f"TRAINER _lift3: [cond_param(a2), v_last/{SPEED_SCALE:g}] (train_v6_staged._lift3)")
    res = {"_evidence_class": f"MEASURED (ours; dev-box {dev})",
           "eval_tier": EVAL_TIER, "hypothesis": HYPOTHESIS, "paper": PAPER,
           "tanitad_imported_from": tanitad_file,
           "metric_families": {"longitudinal": "REFUSED", "lateral": "REFUSED",
                               "tactical": "REFUSED", "strategic": "REFUSED",
                               "reason": "T0-DIAGNOSTIC world-model probe — no driving "
                                         "metric exists here to family-ise (EVAL_DOCTRINE)"},
           "corpus": a.corpus, "n_clips": len(clips), "clip_files": [os.path.basename(c) for c in clips],
           "max_frames": a.max_frames, "window_rule": "range(0, n, max(1, n // 5)) — banked actdiv",
           "lift": lift_desc, "levels_sigma": list(levels), "replace": a.replace,
           "n_perm": a.n_perm, "seed": a.seed, "thresholds": THRESHOLDS,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "arms": {}}
    for arm, p in parse_arms(a):
        if not p.is_file():
            _p(f"  {arm}: NO CKPT at {p} — SKIPPED (never approximated)")
            res["arms"][arm] = {"skipped": f"no checkpoint at {p}"}
            continue
        t0 = time.time()
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        world, _g, step = load_trunk_auto(ck, dev, ckpt_path=str(p))
        world.eval()
        cfg_args = (ck.get("config") or {}).get("args") or {}
        cond_param = str(cfg_args.get("cond_param", COND_INCUMBENT))
        horizons = sorted(int(h) for h in world.stack.cfg.predictor.horizons)
        trained = tuple(int(h) for h in getattr(world.predictor, "trained_horizons", (1,)))
        zs, a2, v_first, v_last, cids = select_windows(world, clips, dev, a.max_frames)
        zs, a2 = zs.to(dev), a2.to(dev)
        if a.actdiv_compat:
            a3 = lift_legacy(a2, v_first.to(dev))
        else:
            a3 = lift_trainer(a2, v_last.to(dev), cond_param)
        _p(f"[{arm}] step {step}  {zs.shape[0]} windows  W={int(world.window)}  S={zs.shape[-1]}"
           f"  horizons={horizons} trained={trained}  cond_param={cond_param}  lift: {lift_desc}")
        entry = {"ckpt_path": str(p), "ckpt_md5": md5_of(p), "step": int(step),
                 "state_dim": int(zs.shape[-1]), "window": int(world.window),
                 "n_windows": int(zs.shape[0]), "cond_param": cond_param,
                 "horizons": horizons, "trained_horizons": list(trained),
                 "untrained_heads_note": ("heads != 1 are allocated but consumed by no loss "
                                          "(predictor.py; MM-E14): their rows are INIT NOISE"),
                 "legacy_ratio_actdiv_verbatim": legacy_ratio(world.predictor, zs, a3, horizons)}
        entry["anchored"] = read_arm(world.predictor, zs, a3, horizons, trained, levels=levels,
                                     n_perm=a.n_perm, seed=a.seed, replace=a.replace,
                                     scale_control=not a.no_scale_control)
        _compact_table(arm, entry["anchored"], horizons, trained)
        if a.also_replace_all:
            other = "all" if a.replace == "last" else "last"
            entry[f"anchored_replace_{other}"] = read_arm(
                world.predictor, zs, a3, horizons, trained, levels=levels, n_perm=a.n_perm,
                seed=a.seed, replace=other, scale_control=False)
        lr = entry["legacy_ratio_actdiv_verbatim"]["h1"]
        _p(f"     legacy ratio h1 {lr['ratio_action_over_scene']:.6f} (action {lr['action_spread']:.5f} "
           f"/ scene {lr['scene_spread']:.5f})  C0 {'PASS' if lr['C0_passes'] else 'FAIL'}")
        entry["seconds"] = round(time.time() - t0, 1)
        entry["verdict_h1"] = entry["anchored"]["horizons"]["h1"]["verdict"]["verdict"]
        res["arms"][arm] = entry
        del world, zs, a2, a3
        if dev == "cuda":
            torch.cuda.empty_cache()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=_json_default)
    _p("WROTE", a.out)
    return 0


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, pathlib.Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


# =====================================================================================
# self-test on synthetic predictors — the instrument must detect what it hunts
# =====================================================================================
class LinearResponsePredictor:
    """A synthetic action-SENSITIVE predictor: d_i(a) = (M a) ⊙ (1 + ε g_i) — same
    direction for every window, magnitude ∝ |a|, small window-specific scatter."""

    def __init__(self, S, seed=0, eps=0.1, gain=1.0):
        import torch
        g = torch.Generator().manual_seed(seed)
        self.M = torch.randn(S, 2, generator=g) * gain
        self.eps = eps
        self.K = torch.randn(S, S, generator=g) * 0.1

    def __call__(self, zs, a3):
        import torch
        z = zs[:, -1]
        gi = torch.tanh(z @ self.K)                        # window-specific, bounded
        resp = (a3[:, -1, :2] @ self.M.T) * (1.0 + self.eps * gi)
        return {1: z + resp, 2: z + 0.5 * resp}


class ActionBlindPredictor:
    def __call__(self, zs, a3):
        import torch
        z = zs[:, -1]
        return {1: z + torch.tanh(z), 2: z}


def selftest(n=96, W=6, S=32, seed=0) -> int:
    import torch
    g = torch.Generator().manual_seed(seed)
    zs = torch.randn(n, W, S, generator=g)
    a2 = torch.randn(n, W, 2, generator=g) * torch.tensor([0.02, 0.6])
    a3 = torch.cat([a2, torch.full((n, W, 1), 0.5)], -1)
    ok = True
    sens = read_arm(LinearResponsePredictor(S, seed), zs, a3, [1, 2], (1,), n_perm=100)
    h1 = sens["horizons"]["h1"]
    _p(f"  sensitive synthetic: F_sep {h1['grid']['F_sep']:.1f}  null p95 "
       f"{h1['grid']['null']['p95']:.2f}  verdict {h1['verdict']['verdict']}")
    ok &= h1["verdict"]["verdict"] == "SENSITIVE"
    ok &= h1["controls"]["C0_passes"] and h1["controls"]["zero_model_passes"]
    ok &= h1["controls"]["output_scale"]["invariant"]
    ok &= h1["grid"]["null"]["p95"] < 3.0
    blind = read_arm(ActionBlindPredictor(), zs, a3, [1, 2], (1,), n_perm=100)
    b1 = blind["horizons"]["h1"]
    _p(f"  blind synthetic: max|d| {b1['grid']['max_abs_displacement']:.1e}  verdict "
       f"{b1['verdict']['verdict']}")
    ok &= b1["verdict"]["verdict"] == "VOID" and b1["grid"]["max_abs_displacement"] == 0.0
    _p("  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# =====================================================================================
# refav1 family — H-REFAV1-LAT-INSENSITIVE (SPEC.md §10)
# =====================================================================================
REFAV1_KAPPA_LEVELS = (0.02, 0.05, 0.1)      # rad/m, at a = 0
REFAV1_ACCEL_LEVELS = (0.5, 1.5)             # m/s², at κ = 0
REFAV1_CHANNELS = (1, 0)                     # (a, κ) controls: κ is channel 1, a is channel 0
REFAV1_SUBSAMPLE = 8192                      # fixed coordinate subsample of the full field
REFAV1_VERDICTS = ("VOID", "LAT-INSENSITIVE-CONFIRMED", "LAT-INSENSITIVE-REFUTED",
                   "BOTH-INSENSITIVE")


def _load_refav1_arm():
    """Import the sibling ``refav1_arm.py`` BY FILE so its strict loader, config
    cross-check and window loader are REUSED, never copied."""
    import importlib.util
    p = os.path.join(_HERE, "refav1_arm.py")
    spec = importlib.util.spec_from_file_location("refav1_arm_for_actdiv", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verdict_refav1(per_axis, c0_max_abs, zero_model_max_abs, scene_spread,
                   thresholds=THRESHOLDS) -> dict:
    reasons = []
    if c0_max_abs != 0.0:
        reasons.append(f"C0 identity max|Δ| = {c0_max_abs:.3e} ≠ 0")
    if zero_model_max_abs != 0.0:
        reasons.append(f"zero-model control max|d| = {zero_model_max_abs:.3e} ≠ 0")
    if not (scene_spread > thresholds["scene_spread_min"]):
        reasons.append(f"scene spread {scene_spread:.3e} ~ 0")
    if reasons:
        return {"verdict": "VOID", "reasons": reasons}
    bar = thresholds["structured_f_over_null_p95"]
    fk = per_axis.get("kappa", {}).get("F_over_null_p95", float("nan"))
    fa = per_axis.get("accel", {}).get("F_over_null_p95", float("nan"))
    k_sep = bool(np.isfinite(fk) and fk >= bar)
    a_sep = bool(np.isfinite(fa) and fa >= bar)
    if k_sep:
        v = "LAT-INSENSITIVE-REFUTED"
    elif a_sep:
        v = "LAT-INSENSITIVE-CONFIRMED"
    else:
        v = "BOTH-INSENSITIVE"
    return {"verdict": v, "kappa_F_over_null_p95": fk, "accel_F_over_null_p95": fa,
            "kappa_separated": k_sep, "accel_separated": a_sep, "bar": bar}


def _full_field_stats(sum_d, sum_sq, n, cands_k):
    """Exact per-candidate statistics of the FULL field from running sums:
    m_c = Σ_i d_ic / N,  SS_within = Σ_c (Σ_i‖d_ic‖² − N‖m_c‖²),  SS_between = N Σ_c‖m_c − m̄‖²."""
    C = sum_d.shape[0]
    m = sum_d / n                                    # [C, S]
    grand = m.mean(0)
    norms = np.linalg.norm(m, axis=1)
    ss_w = float(sum(sum_sq[c] - n * float((m[c] ** 2).sum()) for c in range(C)))
    ss_b = float(n * ((m - grand) ** 2).sum())
    df_b, df_w = max(C - 1, 1), max(n * C - C, 1)
    f = (ss_b / df_b) / (ss_w / df_w) if ss_w > 0 else (0.0 if ss_b == 0 else float("inf"))
    out = {"F_sep": f, "ss_between": ss_b, "ss_within": ss_w,
           "mean_norm": {c["id"]: float(norms[j]) for j, c in enumerate(cands_k)},
           "state_dim": int(m.shape[1])}
    # sign consistency and per-axis F from the exact means (within from the sums)
    by_id = {c["id"]: j for j, c in enumerate(cands_k)}
    axes = {}
    for ax, name in enumerate(AXIS_NAMES):
        idx = [j for j, c in enumerate(cands_k) if c["axis"] == ax]
        if len(idx) < 2:
            continue
        mm = m[idx]
        g = mm.mean(0)
        ssb = float(n * ((mm - g) ** 2).sum())
        ssw = float(sum(sum_sq[j] - n * float((m[j] ** 2).sum()) for j in idx))
        fa = (ssb / max(len(idx) - 1, 1)) / (ssw / max(n * len(idx) - len(idx), 1)) if ssw > 0 else float("inf")
        levels = sorted(set(cands_k[j]["abs_level"] for j in idx))
        cos = {}
        for L in levels:
            a_, b_ = m[by_id[f"{name}+{L:g}"]], m[by_id[f"{name}-{L:g}"]]
            den = float(np.linalg.norm(a_) * np.linalg.norm(b_))
            cos[f"{L:g}"] = float((a_ * b_).sum() / den) if den > 0 else float("nan")
        nl = [float(0.5 * (norms[by_id[f"{name}+{L:g}"]] + norms[by_id[f"{name}-{L:g}"]]))
              for L in levels]
        axes[name] = {"F_sep_exact": fa, "sign_cos_by_level": cos,
                      "norm_by_level": {f"{L:g}": v for L, v in zip(levels, nl)},
                      "spearman_rho_abslevel_vs_norm": spearman(levels, nl)}
    out["axes"] = axes
    return out


def refav1_read(model, cfg, ld, sel, cands, *, n_perm=N_PERM_DEFAULT, seed=0,
                batch=4, n_roll=2, log_every=10) -> dict:
    """The anchored read on refav1 at one operative step. ``sel`` = [(wi, ei, t)]."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ck = [c for c in cands if c["axis"] >= 0]
    C = len(ck)
    n_win = len(sel)
    # spaces: tac (the planner's cost space), pooled (token mean), full (sums + subsample)
    D_tac, D_pool, D_sub = [], [], []
    sum_d = None
    sum_sq = np.zeros(C, dtype=np.float64)
    zero_tac, zero_pool, zero_sub = [], [], []
    own_tac, own_pool = [], []
    roll_tac, roll_pool = [[] for _ in range(n_roll)], [[] for _ in range(n_roll)]
    a_own = []
    c0_max, zm_max = 0.0, 0.0
    sub_idx = None
    zero_ctrl_ids = []
    for ax, name in enumerate(AXIS_NAMES):
        cs = [c for c in ck if c["axis"] == ax]
        if cs:
            zero_ctrl_ids.append(cs[0]["id"])
    t0 = time.time()
    # first actions of every selected window (for the rolled variants)
    firsts = []
    for (wi, ei, t) in sel:
        ld._order, ld._cursor = [wi], 0
        b = ld.batch(1)
        firsts.append(b["actions"][0, 0].float().clone())
    firsts = torch.stack(firsts)                                   # [N, 2]
    with torch.no_grad():
        for s in range(0, n_win, batch):
            chunk = sel[s:s + batch]
            feats, acts, v0s, navs = [], [], [], []
            for (wi, ei, t) in chunk:
                ld._order, ld._cursor = [wi], 0
                b = ld.batch(1)
                feats.append(b["feats"][0].float())
                acts.append(b["actions"][0].float())
                v0s.append(b["v0"][0].float())
                navs.append(b["nav_cmd"][0] if b.get("nav_cmd") is not None else None)
            feats = torch.stack(feats)
            acts = torch.stack(acts)                              # [B, K, 2]
            v0 = torch.stack(v0s)
            nav = None if navs[0] is None else torch.stack(navs)
            field = model.encode(feats)
            last = model._last_state(field)
            brains = model._run_brains(field.mean(dim=-2), nav)
            intent = None if brains is None else brains["intent"]
            B = last.shape[0]

            def step_with(a2_first):
                ctrl = acts.clone()
                ctrl[:, 0, :] = a2_first
                a_in = model.augment_actions(ctrl, v0)
                return model.operative.step(last, a_in[:, 0], intent=intent)

            zero_a = torch.zeros(B, 2)
            base = step_with(zero_a)
            again = step_with(zero_a)
            c0_max = max(c0_max, float((again - base).abs().max()))
            base_tac = model._tac_field(base).flatten(1)
            base_pool = base.mean(dim=-2)
            if sub_idx is None:
                S_full = base.flatten(1).shape[1]
                sub_idx = torch.as_tensor(np.sort(rng.choice(S_full, size=min(REFAV1_SUBSAMPLE, S_full),
                                                             replace=False)))
                sum_d = np.zeros((C, S_full), dtype=np.float64)
            zero_tac.append(base_tac.numpy()); zero_pool.append(base_pool.numpy())
            zero_sub.append(base.flatten(1)[:, sub_idx].numpy())
            dt_tac = np.zeros((B, C, base_tac.shape[1]), dtype=np.float32)
            dt_pool = np.zeros((B, C, base_pool.shape[1]), dtype=np.float32)
            dt_sub = np.zeros((B, C, len(sub_idx)), dtype=np.float32)
            for j, c in enumerate(ck):
                cand = torch.tensor(c["a2"], dtype=torch.float32)[None].expand(B, -1)
                out = step_with(cand)
                d = (out - base)
                df = d.flatten(1)
                sum_d[j] += df.double().sum(0).numpy()
                sum_sq[j] += float((df.double() ** 2).sum())
                dt_tac[:, j] = (model._tac_field(out).flatten(1) - base_tac).numpy()
                dt_pool[:, j] = (out.mean(dim=-2) - base_pool).numpy()
                dt_sub[:, j] = df[:, sub_idx].numpy()
                if c["id"] in zero_ctrl_ids:          # ZERO MODEL: fed zero whatever the candidate
                    zm = step_with(zero_a)
                    zm_max = max(zm_max, float((zm - base).abs().max()))
            D_tac.append(dt_tac); D_pool.append(dt_pool); D_sub.append(dt_sub)
            # realised: own first action, and rolled first actions from other windows
            own = step_with(acts[:, 0])
            own_tac.append((model._tac_field(own).flatten(1) - base_tac).numpy())
            own_pool.append((own.mean(dim=-2) - base_pool).numpy())
            a_own.append(acts[:, 0].numpy())
            for r in range(n_roll):
                rolled = torch.stack([firsts[(s + i + (r + 1) * 7) % n_win] for i in range(B)])
                o = step_with(rolled)
                roll_tac[r].append((model._tac_field(o).flatten(1) - base_tac).numpy())
                roll_pool[r].append((o.mean(dim=-2) - base_pool).numpy())
            done = s + B
            if (done // batch) % log_every == 0 or done >= n_win:
                el = time.time() - t0
                _p(f"    {done}/{n_win} windows  {el / 60:.1f} min  (~{el / done * (n_win - done) / 60:.1f} min left)")
    D_tac = np.concatenate(D_tac).astype(np.float64)
    D_pool = np.concatenate(D_pool).astype(np.float64)
    D_sub = np.concatenate(D_sub).astype(np.float64)
    zero_tac = np.concatenate(zero_tac); zero_pool = np.concatenate(zero_pool)
    zero_sub = np.concatenate(zero_sub)
    a_own = np.concatenate(a_own)
    sigma = a_own.std(0)
    spaces = {}
    for name, D, Z in (("tac", D_tac, zero_tac), ("pooled", D_pool, zero_pool),
                       ("full_subsample", D_sub, zero_sub)):
        ss = float(Z.std(0).mean())
        Dfull = np.concatenate([np.zeros((D.shape[0], 1, D.shape[2])), D], axis=1)  # zero anchor first
        summ = summarize_grid(Dfull, [cands[0]] + ck, ss, n_perm=n_perm, seed=seed)
        rd = realised_reads(
            np.concatenate(own_tac if name == "tac" else own_pool) if name != "full_subsample" else None,
            [np.concatenate(x) for x in (roll_tac if name == "tac" else roll_pool)] if name != "full_subsample" else [],
            a_own, sigma) if name != "full_subsample" else None
        spaces[name] = {"scene_spread": ss, "grid": summ, "realised": rd,
                        "verdict": verdict_refav1(summ["per_axis"], c0_max, zm_max, ss)}
    full = _full_field_stats(sum_d, sum_sq, n_win, ck)
    full["scene_spread_note"] = "exact full-field means; F_sep exact; null from full_subsample"
    return {"n_windows": n_win, "n_candidates": C, "candidates": cands,
            "controls": {"C0_identity_max_abs_diff": c0_max, "C0_passes": c0_max == 0.0,
                         "zero_model_max_abs_displacement": zm_max,
                         "zero_model_passes": zm_max == 0.0,
                         "zero_model_candidates_checked": zero_ctrl_ids},
            "own_first_action_sigma": sigma.tolist(),
            "spaces": spaces, "full_field_exact": full,
            "verdict_space": "tac", "verdict": spaces["tac"]["verdict"],
            "space_agreement": {k: v["verdict"]["verdict"] for k, v in spaces.items()},
            "seconds": round(time.time() - t0, 1)}


def refav1_main(a) -> int:
    import torch
    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
    stack = resolve_stack(a.stack or None)
    ra = _load_refav1_arm()
    import tanitad
    tanitad_file = str(pathlib.Path(tanitad.__file__).resolve())
    if pathlib.Path(stack) not in pathlib.Path(tanitad_file).parents:
        _p(f"[REFUSED] tanitad from {tanitad_file}, not {stack}")
        return 2
    _p(f"  [stack] tanitad imported from: {tanitad_file}")
    kl = tuple(float(x) for x in a.kappa_levels.split(","))
    al = tuple(float(x) for x in a.accel_levels.split(","))
    cands = candidate_grid_abs(kl, al, REFAV1_CHANNELS)
    # ⛔ D-P2-LEAK-AUDIT / H-LEAK-1: the speed channel MUST be fed at the model's OWN
    # trained scale. This tool never builds the third channel itself — it calls the
    # model's ``augment_actions``, which divides by ``refa_v1.SPEED_SCALE_MPS``. The
    # constant is READ BACK from the model file here and printed beside every reading
    # so the scale is on the artefact, not in a reader's memory.
    from tanitad.refs.refa_v1 import SPEED_SCALE_MPS as _REFAV1_SPEED_SCALE
    res = {"_evidence_class": "MEASURED (ours; dev-box cpu)", "eval_tier": EVAL_TIER,
           "hypothesis": "H-REFAV1-LAT-INSENSITIVE", "paper": PAPER, "family": "refav1",
           "tanitad_imported_from": tanitad_file,
           "speed_scale_mps": float(_REFAV1_SPEED_SCALE),
           "speed_scale_source": "tanitad.refs.refa_v1.SPEED_SCALE_MPS (the model's own "
                                 "constant, applied inside model.augment_actions; this tool "
                                 "NEVER widens the action tensor itself)",
           "metric_families": {"longitudinal": "REFUSED", "lateral": "REFUSED",
                               "tactical": "REFUSED", "strategic": "REFUSED",
                               "reason": "T0-DIAGNOSTIC world-model probe (EVAL_DOCTRINE)"},
           "cache": a.refav1_cache, "episodes": a.refav1_episodes, "labels": a.refav1_labels,
           "nav": a.refav1_nav, "episodes_n": a.episodes_n, "window_stride": a.window_stride,
           "horizon": "h=1 (one operative step, 0.2 s); the h=10 planner-horizon read is a GPU follow-up",
           "kappa_levels": list(kl), "accel_levels": list(al), "channels": REFAV1_CHANNELS,
           "verdict_space": "tac (the planner's cost space: _tac_field of the terminal field)",
           "n_perm": a.n_perm, "seed": a.seed, "thresholds": THRESHOLDS,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arms": {}}
    from argparse import Namespace
    for arm, p in parse_arms(a):
        if not p.is_file():
            _p(f"  {arm}: NO CKPT at {p} — SKIPPED")
            res["arms"][arm] = {"skipped": f"no checkpoint at {p}"}
            continue
        t0 = time.time()
        cfg_path = str(p.parent / "config.json") if (p.parent / "config.json").is_file() else None
        model, cfg, prov = ra.load_model(str(p), cfg_path, "cpu")
        k_loader = int(cfg.op_steps)
        names = ra.episode_names(a.refav1_cache)[:a.episodes_n] if a.episodes_n else ra.episode_names(a.refav1_cache)
        ld = ra.build_loader(Namespace(cache=a.refav1_cache, episodes=a.refav1_episodes, lru=a.lru,
                                       labels=a.refav1_labels, nav=a.refav1_nav), cfg, k_loader, names)
        stride = max(1, int(a.window_stride))
        sel = [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows) if (t - (ld.W - 1)) % stride == 0]
        if a.max_windows:
            sel = sel[:a.max_windows]
        # ⚠️ prov['trainable_parameters'] reads 0 BY CONSTRUCTION: refav1_arm.load_model
        # calls requires_grad_(False) on every parameter BEFORE counting. Report the
        # total instead, or the log claims an empty model.
        n_par = sum(p_.numel() for p_ in model.parameters())
        a_in = int(model.augment_actions(torch.zeros(1, 1, int(cfg.a_dim)),
                                         torch.zeros(1)).shape[-1])
        scale_note = (f"SPEED_SCALE_MPS={_REFAV1_SPEED_SCALE:g}"
                      + ("" if cfg.speed_channel else " (UNUSED: speed_channel=False, "
                                                      "predictor input is (a,kappa) only)"))
        _p(f"[{arm}] step {prov['step']}  params {n_par:,} (frozen; prov.trainable reads 0 by "
           f"construction)  a_dim {cfg.a_dim} -> a_in_dim {a_in}  speed_channel {cfg.speed_channel}  "
           f"{scale_note}  W={ld.W} k_loader={k_loader}  windows {len(sel)} of {len(ld)} "
           f"(stride {stride}, {len(ld.names)} episodes)  config<-{prov['config_source']}")
        rd = refav1_read(model, cfg, ld, sel, cands, n_perm=a.n_perm, seed=a.seed, batch=a.batch)
        for sp, v in rd["spaces"].items():
            pa = v["grid"]["per_axis"]
            _p(f"  [{arm}] space={sp:<15} F/p95 kappa {pa['kappa']['F_over_null_p95']:.2f}  accel "
               f"{pa['accel']['F_over_null_p95']:.2f}  scene_spread {v['scene_spread']:.4e}  "
               f"-> {v['verdict']['verdict']}")
        fx = rd["full_field_exact"]
        _p(f"  [{arm}] full-field exact: F_sep {fx['F_sep']:.3f}  kappa F {fx['axes']['kappa']['F_sep_exact']:.3f} "
           f"accel F {fx['axes']['accel']['F_sep_exact']:.3f}  norms kappa "
           + " ".join(f"{k}:{v:.3e}" for k, v in fx['axes']['kappa']['norm_by_level'].items())
           + "  accel " + " ".join(f"{k}:{v:.3e}" for k, v in fx['axes']['accel']['norm_by_level'].items()))
        _p(f"  [{arm}] C0 {rd['controls']['C0_identity_max_abs_diff']:.1e}  zero-model "
           f"{rd['controls']['zero_model_max_abs_displacement']:.1e}  VERDICT (tac) {rd['verdict']['verdict']}")
        res["arms"][arm] = {"ckpt_path": str(p), "ckpt_md5": md5_of(p), "provenance": prov,
                            "parameters_total": int(n_par),
                            "a_in_dim": a_in, "speed_channel": bool(cfg.speed_channel),
                            "speed_scale_mps": float(_REFAV1_SPEED_SCALE),
                            "speed_scale_effective": (float(_REFAV1_SPEED_SCALE)
                                                      if cfg.speed_channel else None),
                            "read": rd, "seconds": round(time.time() - t0, 1)}
        del model
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=_json_default)
    _p("WROTE", a.out)
    return 0


def _add_refav1_args(ap):
    ap.add_argument("--family", default="v7", choices=("v7", "refav1"))
    ap.add_argument("--refav1-cache", default=r"C:\Users\Admin\refav1_eval_slice\fp8")
    ap.add_argument("--refav1-episodes", default=r"C:\Users\Admin\refav1_eval_slice\eps")
    ap.add_argument("--refav1-labels",
                    default=r"C:\Users\Admin\tanitad-wt\_s2build\release\v72\s2_labels_v7.2_eval.jsonl.gz")
    ap.add_argument("--refav1-nav",
                    default=r"C:\Users\Admin\tanitad-wt\_s2build\release\v72\s2_labels_v7.2_eval.jsonl.gz")
    ap.add_argument("--episodes-n", type=int, default=20)
    ap.add_argument("--window-stride", type=int, default=10)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--lru", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--kappa-levels", default=",".join(f"{x:g}" for x in REFAV1_KAPPA_LEVELS))
    ap.add_argument("--accel-levels", default=",".join(f"{x:g}" for x in REFAV1_ACCEL_LEVELS))


if __name__ == "__main__":
    raise SystemExit(main())
