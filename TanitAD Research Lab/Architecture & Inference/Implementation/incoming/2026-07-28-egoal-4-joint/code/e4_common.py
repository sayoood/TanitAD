#!/usr/bin/env python3
"""E-GOAL-4 -- shared feature builder for the JOINTLY TRAINED selector.

⛔ THE ONE THING THIS FILE EXISTS TO GET RIGHT (C23).
A trained selector is FAR more exposed to future content than a fixed rule was:
a fixed projection can only use a column one way, a learner can invert it. So
every fed column is declared here WITH ITS EXPRESSION, a runtime assertion
refuses the fan dump's future fields (`gt`, `a_gt`, `head_deg`, `v_target`,
`speed`, `vt_*`) reaching any arm's X, and `future_blind()` below re-derives
every column from a CORRUPTED future and requires max |Δ| == 0.0.

⛔ AND THE ANSWER IS NOT UNIFORM ACROSS BACKGROUNDS, WHICH IS WHY THE TEST IS
RUN RATHER THAN ASSERTED. `parent_resampled`'s cross coordinate is
`true_cross + resampled_residual` -- FUTURE-DERIVED BY CONSTRUCTION (it is a
SIMULATED cross-track head, inherited from E-GOAL-2/3). `sel`'s cross is the
REF-C selector's own endpoint and is future-blind. The same instrument fires on
the first and returns exactly 0.0 on the second.

FIDELITY. `ade`, `goal_reference`, `pick_nearest_to`, `realise` are IMPORTED
from E-GOAL-1's `eg_place.py`; `clip_folds`, `ci_paired`, `ci_single`, `sep`,
`r4` from `eg_common.py`. Nothing is re-implemented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STREAM = HERE.parent
ARCH = STREAM.parent
REPO = STREAM.parents[4]
EG1 = ARCH / "2026-07-27-egoal-1-lead-vehicle"
EG2 = ARCH / "2026-07-28-egoal-2-power"
EG3 = ARCH / "2026-07-28-egoal-3-trained-head"
GI = ARCH / "2026-07-27-goal-input"

for p in (EG1 / "code", EG2 / "code", EG3 / "code", GI / "code",
          REPO / "taniteval"):
    sys.path.insert(0, str(p))

from eg_place import FRAC, ade, goal_reference, pick_nearest_to, realise  # noqa: E402,F401
from eg_common import SEED, ci_paired, ci_single, clip_folds, r4, sep  # noqa: E402,F401

# --------------------------------------------------------------------------- #
# ⛔ the fan dump's FUTURE fields. None of these may reach an arm's X except in
# the two arms explicitly registered as a bound (`S_goal_oracle`) or as the
# leak-detector power control (`S_LEAK`).                                      #
# --------------------------------------------------------------------------- #
FUTURE_FIELDS = frozenset({
    "gt", "a_gt", "head_deg", "v_target", "vt_valid", "vt_lookahead", "speed",
    "y_long", "y_lat", "v_fut_2s"})

#: ⭐ EVERY FED COLUMN, WITH ITS EXPRESSION. This table IS the C23 "by
#: definition" audit; `future_blind()` is its empirical half.
COLSPEC: list[tuple[str, str, str]] = [
    # (name, block, expression -- and what it reads)
    ("logit",        "F_ans", "logits[w,c]                       REF-C's own score (past)"),
    ("softmax",      "F_ans", "softmax_c(logits[w,:])[c]         (past)"),
    ("logit_rank",   "F_ans", "rank_c(logits[w,:])[c] / 255      (past)"),
    ("end_along",    "F_ans", "fan[w,c,3,0]                      the PROPOSAL (past-conditioned)"),
    ("end_cross",    "F_ans", "fan[w,c,3,1]"),
    ("mid_along",    "F_ans", "fan[w,c,1,0]                      the 1.0 s waypoint"),
    ("mid_cross",    "F_ans", "fan[w,c,1,1]"),
    ("pathlen",      "F_ans", "sum_k |fan[w,c,k]-fan[w,c,k-1]|,  fan[w,c,-1]:=0"),
    ("mean_speed",   "F_ans", "pathlen / 2.0 s"),
    ("abs_head_end", "F_ans", "|atan2(end_cross, end_along)|"),
    ("shape_dev",    "F_ans", "end_along - 2*mid_along           non-constant-rate along"),
    ("v0",           "F_ctx", "v0[w] == poses[L] speed           (past; gate F-2 per-row)"),
    ("ax_fd",        "F_ctx", "(s[L]-s[L-1])/0.1                 offsets {0,-1}, max 0 (past)"),
    ("cv_end_along", "F_ctx", "cv[w,3,0]                         constant-velocity rollout of v0"),
    ("sel_end_along","F_ctx", "fan[w,sel[w],3,0]                 the as-trained pick (past)"),
    ("sel_end_cross","F_ctx", "fan[w,sel[w],3,1]"),
    ("logit_max",    "F_ctx", "max_c logits[w,c]"),
    ("logit_ent",    "F_ctx", "-sum_c p log p, p = softmax(logits[w,:])"),
    ("g_along",      "F_goal","THE TREATMENT: the goal head's OOF prediction"),
    ("g_cross",      "F_goal","THE TREATMENT: the cross-track BACKGROUND (see below)"),
    ("d_along",      "F_goal","end_along - g_along"),
    ("d_cross",      "F_goal","end_cross - g_cross"),
    ("d_rule",       "F_goal","⭐ mean_k |fan[w,c,k] - goal*FRAC[k]|  == THE FIXED RULE'S "
                              "OWN STATISTIC. With this column the learner CAN express "
                              "`argmin d_rule` exactly, so `CONTROL-WEAK-BY-MODEL-CLASS` "
                              "is closed by construction."),
]
F_ANS = [n for n, b, _ in COLSPEC if b == "F_ans"]
F_CTX = [n for n, b, _ in COLSPEC if b == "F_ctx"]
F_GOAL = [n for n, b, _ in COLSPEC if b == "F_goal"]
ALL_COLS = F_ANS + F_CTX + F_GOAL

HGB_KW = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
              min_samples_leaf=200, l2_regularization=1.0,
              early_stopping=False, random_state=0)

#: E-GOAL-2/3's published n=600 deployment. Gate G-0's target (raw JSON, never
#: a doc table): raw/e2_place_n600_parent_resampled.json,
#: raw/e3_place_n600_parent_resampled.json.
DEPLOY_REF = {"a0": 0.5015, "r_goal2s": 0.1933, "oracle_in_fan": 0.1547,
              "headroom": 0.3082}
#: E-GOAL-3's fixed-rule cells this stream is testing, quoted from the RAW JSON
#: `e3_place_n600_{parent_resampled,sel}.json` -> arms[k].recovery_of_headroom,
#: NEVER from EGOAL_3.md's rounded table.
E3_FIXED = {
    "parent_resampled": {"H_v0_ax": 0.4634, "H_ego": 0.4626, "CV_head": -0.1855,
                         "P_ORACLE_TRUE": 0.7741, "N_SHUF": -29.9345},
    "sel": {"H_v0_ax": 0.6146, "H_ego": 0.6148, "CV_head": -0.0611,
            "P_ORACLE_TRUE": 0.9484, "N_SHUF": -29.9135}}


def assert_no_future(names, allow: str = ""):
    bad = sorted(set(map(str, names)) & FUTURE_FIELDS)
    if bad and not allow:
        raise RuntimeError(f"⛔ FUTURE FIELD IN ARM INPUT: {bad}")
    return bad


def load_all(fan_path, feat_path, pred_path):
    """The three inputs, and NOTHING else is read."""
    import torch
    d = torch.load(fan_path, map_location="cpu", weights_only=False)
    z = np.load(feat_path, allow_pickle=True)
    p = np.load(pred_path, allow_pickle=True)
    out = {
        "fan": d["fan"].numpy().astype(np.float64),
        "logits": d["logits"].numpy().astype(np.float64),
        "sel": d["sel"].numpy(),
        "gt": d["gt"].numpy().astype(np.float64),
        "cv": d["cv"].numpy().astype(np.float64),
        "v0": d["v0"].numpy().astype(np.float64),
        "eid": np.asarray([str(x) for x in d["eid"]]),
        "head_deg": d["head_deg"].numpy().astype(np.float64),   # ⛔ FUTURE
        "X": z["X"], "Y": z["Y"], "epi": z["epi"], "L": z["L"],
        "clamped": z["clamped"], "cols": list(map(str, z["cols"])),
        "preds": {k: p[k] for k in p.files},
        "_ckpt": str(d["ckpt"]), "_ckpt_step": int(d["ckpt_step"]),
        "_n_anchors": int(d["n_anchors"]), "_nav_mode": str(d["nav_mode"]),
    }
    return out


def cross_background(mode: str, g_true, fan, sel, seed_off: int = 0, W=None):
    """The cross-track coordinate of the goal. ⛔ NAMED, and held fixed.

    `parent_resampled` -- E-GOAL-2's registered CONSERVATIVE carrier: the parent
        head's own 881 cross residuals resampled onto the TRUE cross-track.
        ⛔ FUTURE-DERIVED BY CONSTRUCTION (it is a *simulated* cross head).
    `sel` -- the REF-C selector's own 2 s endpoint cross. Zero fit, and
        ⭐ FULLY FUTURE-BLIND.
    """
    W = len(g_true) if W is None else W
    if mode == "parent_resampled":
        P = np.load(GI / "raw" / "gi_head_preds.npz", allow_pickle=True)
        pc = P["best"][:, 1] - P["gt_end"][:, 1]
        rng = np.random.default_rng(5000 + seed_off)
        return g_true[:, 1] + pc[rng.integers(0, len(pc), W)], len(pc)
    if mode == "sel":
        return fan[np.arange(W), sel, 3, 1].copy(), 0
    raise KeyError(mode)


def build_static(D):
    """`F_ans` ⧺ `F_ctx` -- everything that does NOT depend on the goal.

    Shape [W, 256, len(F_ans)+len(F_ctx)], float32. No future field is touched:
    `gt`, `a_gt`, `head_deg`, `v_target`, `speed`, `vt_*` are not read here.
    """
    fan, logits = D["fan"], D["logits"]
    W, C = logits.shape
    m = logits.max(1, keepdims=True)
    e = np.exp(logits - m)
    sm = e / e.sum(1, keepdims=True)
    rank = np.argsort(np.argsort(logits, 1), 1) / (C - 1.0)
    end = fan[:, :, 3, :]
    mid = fan[:, :, 1, :]
    seg = np.concatenate([fan[:, :, :1, :], np.diff(fan, axis=2)], axis=2)
    pathlen = np.linalg.norm(seg, axis=-1).sum(-1)
    ent = -(sm * np.log(np.clip(sm, 1e-30, None))).sum(1)

    ans = [logits, sm, rank, end[..., 0], end[..., 1], mid[..., 0], mid[..., 1],
           pathlen, pathlen / 2.0, np.abs(np.arctan2(end[..., 1], end[..., 0])),
           end[..., 0] - 2.0 * mid[..., 0]]
    ctxv = [D["v0"], D["X"][:, 1], D["cv"][:, 3, 0],
            fan[np.arange(W), D["sel"], 3, 0], fan[np.arange(W), D["sel"], 3, 1],
            logits.max(1), ent]
    assert len(ans) == len(F_ANS) and len(ctxv) == len(F_CTX)
    out = np.empty((W, C, len(F_ANS) + len(F_CTX)), np.float32)
    for j, a in enumerate(ans):
        out[:, :, j] = a
    for j, v in enumerate(ctxv):
        out[:, :, len(F_ANS) + j] = v[:, None]
    return out


def build_goal(D, goal):
    """`F_goal` -- the 5 treatment columns, [W, 256, 5] float32.

    `d_rule` is the FIXED RULE'S OWN statistic, handed to the learner.
    """
    fan = D["fan"]
    W, C = fan.shape[0], fan.shape[1]
    ref = goal_reference(goal)                       # [W, 4, 2]
    d_rule = np.linalg.norm(fan - ref[:, None], axis=-1).mean(-1)
    out = np.empty((W, C, len(F_GOAL)), np.float32)
    out[:, :, 0] = goal[:, 0][:, None]
    out[:, :, 1] = goal[:, 1][:, None]
    out[:, :, 2] = fan[:, :, 3, 0] - goal[:, 0][:, None]
    out[:, :, 3] = fan[:, :, 3, 1] - goal[:, 1][:, None]
    out[:, :, 4] = d_rule
    return out


def labels(D):
    """⭐ THE LABEL: each candidate's own realised `ade_0_2s`. It is a FUTURE
    quantity BY DESIGN -- it is the supervision target, never a feature."""
    return np.linalg.norm(D["fan"] - D["gt"][:, None], axis=-1).mean(-1)


def shuffle_across_episodes(v, epi, rng):
    """Permute a per-window quantity ACROSS episodes: every window keeps a REAL
    value, from a DIFFERENT episode. Capacity and marginal distribution fixed,
    information destroyed."""
    uniq = np.unique(epi)
    idx = {u: np.flatnonzero(epi == u) for u in uniq}
    order = rng.permutation(len(uniq))
    out = np.empty_like(v)
    for i, u in enumerate(uniq):
        src, dst = idx[uniq[order[i]]], idx[u]
        out[dst] = v[np.resize(src, len(dst))]
    return out
