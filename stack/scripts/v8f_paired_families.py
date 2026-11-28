#!/usr/bin/env python3
"""Paired A - B intervals for the TACTICAL and STRATEGIC families of two v8F-mini eval dumps on the SAME windows.

CLAUDE.md, "EVERY EVAL REPORTS FOUR METRIC FAMILIES", rule 2: each family carries its estimator (the PAIRED episode-cluster bootstrap) and its
CI on the same windows as the ADE it accompanies. The longitudinal and lateral families got their paired blocks in `taniteval.four_families` and the
readers; TACTICAL and STRATEGIC had only each arm's own (unpaired) interval. This tool closes that, from the per-window arrays already in the eval `.npz`.

  python v8f_paired_families.py --a <X eval_*.npz> --b <B eval_*.npz> --out <json> [--md <md>]

What it computes (A - B; A is the arm under test, B the base). Every definition is the eval's own; nothing is re-invented:
  TACTICAL, plan-derived  lateral / longitudinal decision and the 5-way collapse: accuracy and Cohen's kappa of the manoeuvre read off the arm's PLAN
                          (`taniteval.four_families.maneuver_kinematics` -> `refc_tactical.factor_from_kinematics`) against the one read off the human path;
                          the goal point (goal-point error = FDE6, bearing MAE and bias, range ratio, signed long / lat bias).
  TACTICAL, heads         the lat / lon head decisions (`lat_logits`, `lon_logits` argmax) against the v9 labels: macro-F1 (the eval's `C.macro_f1`, RE-COMPUTED
                          ON EVERY RESAMPLE), accuracy, kappa; the goal head (`goal_prob > 0.5` against `goal_y` where `goal_w`): macro-F1 over the 22 tokens
                          (the eval's definition, re-computed per resample), micro token accuracy, and each token's F1.
  STRATEGIC               S's checkpoint position error (the dump's `ck_pos_err`) and heading error at 30 / 50 / 80 m. The heading error is not stored in the
                          dump: it is rebuilt from the raw tick-frame `ckpt` and the held-age poses, and used ONLY if that rebuild reproduces the stored
                          `ck_pos_err` to 1e-4 m on every window (it did, to 0.0, on the MIXO pair); otherwise those cells are UNAVAILABLE with the deviation.
Estimator: `taniteval.ci.paired_episode_cluster_bootstrap` (the SAME resampled episodes for both arms, per-resample re-computation of every non-additive
statistic). A paired interval is never built by combining two intervals in quadrature. Where a cell is restricted to a mask, the bootstrap runs over the
windows inside it and the episodes that have any (the eval's own convention: strategic 137 / 137 / 136 episodes), and n and the reason are printed.

REFUSES (rc 2) unless both dumps cover the identical windows: same length, identical `eid` sequence, `eid` equal to the grid's, and each dump's stored
per-window `ade6` equals ADE6 recomputed from its `pred` and the grid's GT (this is what catches a within-episode reordering that `eid` alone cannot), and each
dump's `ck_pos_err` NaN mask equals the grid's checkpoint-label mask. Controls recorded in the output (rc 3 if one fails; the JSON is still written):
  * identity: the pipeline run A against A reads EXACTLY 0.0 [0.0, 0.0] on every cell;
  * reproduction: each arm's point values (and the per-arm intervals where the eval JSON has one) equal the eval JSON beside the dump, to 1e-6.

Variance this answers: another draw of EPISODES, with the models held fixed. Each arm is ONE training seed, so a separated cell is a necessary and not a
sufficient condition for a lever effect (H-ESTIM-SEED-1). The tier is T1-class open loop (the arm's own held outputs, no recorded future action).
STRATEGIC is active for v8F-mini (PI-V8F-2026-10-17; R72/R73): it is reported in full.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")                      # CPU only, by design

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from taniteval import v8f_consistency as C  # noqa: E402
from taniteval.ci import episode_cluster_bootstrap, paired_episode_cluster_bootstrap  # noqa: E402

PKG = os.path.join(REPO, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini")
RAW = os.path.join(PKG, "raw")
SCHEMA = "v8f-paired-families-v1"
TIER = "T1-class open loop (own held outputs; no recorded future action)"
N_BOOT = 2000
N_GOAL = 22
GOAL_BITS = 22
LAT_K, LON_K = 5, 7
TOL_ADE6 = 1e-6                    # stored ade6 vs recomputed from pred and the grid GT (float64 on both sides)
TOL_HEADING_GATE = 1e-4            # m: the held-age rebuild must reproduce the stored ck_pos_err this closely on every window
TOL_REPRO = 1e-6                   # a per-arm value vs the eval JSON's own (the JSON rounds its intervals to 4 dp; compared after the same rounding)
STRATEGIC_STATUS = "ACTIVE for v8F-mini (PI-V8F-2026-10-17; R72/R73): reported in full"
DUMP_KEYS = ("pred", "eid", "lat_logits", "lon_logits", "goal_prob", "ckpt", "ck_pos_err", "ade6")
CTX_KEYS = ("eid", "gt", "has", "lat_cls", "lon_cls", "goal_y", "goal_w", "rc", "rc_ok", "pose_s", "pose_now")


class Refusal(Exception):
    """The inputs cannot be paired: reported on stderr, rc 2, never a traceback."""


# ------------------------------------------------------------------------------------------------ inputs
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dump(path) -> dict:
    if not os.path.isfile(path):
        raise Refusal(f"eval dump not found: {path}")
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as e:
        raise Refusal(f"{path}: not a readable npz ({type(e).__name__}: {e})")
    miss = [k for k in DUMP_KEYS if k not in z.files]
    if miss:
        raise Refusal(f"{path}: the dump lacks the arrays {miss} (an eval `run` dump carries them all)")
    d = {k: np.asarray(z[k]) for k in DUMP_KEYS}
    d["_path"] = str(path)
    d["_sha256"] = sha256_file(path)
    return d


def load_json_beside(dump_path, explicit=None):
    """The eval JSON of a dump (`eval_<tag>.npz` -> `eval_<tag>.json`), or None."""
    p = explicit or os.path.splitext(str(dump_path))[0] + ".json"
    if not os.path.isfile(p):
        return None, p
    with open(p, encoding="utf-8") as f:
        return json.load(f), p


def held_poses(grid, kin_npz, eid, t, ages="cycle", hist=4, max_age_t=2, max_age_s=6, window_start=None):
    """(pose_s, pose_now) [N, 3] float32 per window: what `V8FMiniWindows.item` hands the evaluator to re-express S's tick-frame checkpoints in NOW.
    The same clamp (`room`, `max_age_t`, `max_age_s`) and the same cycle (`a_T = i % 3`, `a_S = a_T + (i // 3) % 5`) as `cmd_run`."""
    from tanitad.data import v8f_mini_data as D
    kz = np.load(kin_npz)
    off = np.concatenate([[0], np.cumsum(kz["n"])])
    ws = D.effective_window_start(hist, window_start)
    n = len(t)
    ps, pn = np.empty((n, 3), np.float32), np.empty((n, 3), np.float32)
    grids = {}
    for i in range(n):
        ci = int(eid[i])
        if ci not in grids:
            grids[ci] = kz["poses"][off[ci]:off[ci + 1]][0::2].astype(np.float64)
        g = grids[ci]
        tt = int(t[i])
        if ages == "cycle":
            a_t = i % 3
            a_s = a_t + (i // 3) % 5
        elif ages == "fresh":
            a_t = a_s = 0
        else:
            raise Refusal(f"ages must be 'cycle' or 'fresh', got {ages!r}")
        room = max(tt - ws, 0)
        a_t = int(min(a_t, room, max_age_t))
        a_s = int(max(a_t, min(a_s, room, max_age_s)))
        pn[i] = g[tt][:3].astype(np.float32)
        ps[i] = g[tt - a_s][:3].astype(np.float32)
    return ps, pn


def build_context(grid_json, kin_npz, join_npz, kin_mode="yaw", ages="cycle", hist=4, max_age_t=2, max_age_s=6, window_start=None) -> dict:
    """The label / GT side of the eval grid (no image feature needed), exactly the eval's `ctx_from_args`, plus the held-age poses."""
    from tanitad.data import v8f_mini_data as D
    for p in (grid_json, kin_npz, join_npz):
        if not os.path.isfile(p):
            raise Refusal(f"eval context file not found: {p}")
    with open(grid_json, encoding="utf-8") as f:
        grid = json.load(f)
    J = D.load_join(join_npz)
    X = D.eval_context(grid, kin_npz, J, kin_mode=kin_mode)
    ctx = {k: X[k] for k in CTX_KEYS if k in X}
    ctx["pose_s"], ctx["pose_now"] = held_poses(grid, kin_npz, X["eid"], X["t"], ages, hist, max_age_t, max_age_s, window_start)
    ctx["_sha256"] = {os.path.basename(p): sha256_file(p) for p in (grid_json, kin_npz, join_npz)}
    ctx["_kin_mode"], ctx["_ages"] = kin_mode, ages
    return ctx


def load_ctx_npz(path) -> dict:
    """TEST INPUT: a pre-built context npz in place of grid + kin + join."""
    if not os.path.isfile(path):
        raise Refusal(f"context npz not found: {path}")
    z = np.load(path, allow_pickle=False)
    miss = [k for k in CTX_KEYS if k not in z.files]
    if miss:
        raise Refusal(f"{path}: the context lacks {miss}")
    ctx = {k: np.asarray(z[k]) for k in CTX_KEYS}
    ctx["_sha256"] = {os.path.basename(path): sha256_file(path)}
    return ctx


# ------------------------------------------------------------------------------------------------ alignment
def check_alignment(A, B, ctx) -> dict:
    """Refuse unless A and B are the same windows as each other AND as the grid's. Returns the facts it verified."""
    n = len(ctx["eid"])
    for nm, d in (("A", A), ("B", B)):
        if d["pred"].ndim != 3 or d["pred"].shape[0] != n or d["pred"].shape[2] != 2:
            raise Refusal(f"dump {nm} has pred of shape {tuple(d['pred'].shape)}; the grid has {n} windows")
        for k in ("eid", "lat_logits", "lon_logits", "goal_prob", "ckpt", "ck_pos_err", "ade6"):
            if d[k].shape[0] != n:
                raise Refusal(f"dump {nm}: `{k}` has {d[k].shape[0]} rows, the grid has {n} windows")
    ne = np.flatnonzero(A["eid"] != B["eid"])
    if ne.size:
        i = int(ne[0])
        raise Refusal(f"A and B cover different windows: the episode id differs at {ne.size} of {n} windows, first at window {i} (A {A['eid'][i]}, B {B['eid'][i]})")
    for nm, d in (("A", A), ("B", B)):
        ng = np.flatnonzero(d["eid"] != np.asarray(ctx["eid"]))
        if ng.size:
            raise Refusal(f"dump {nm} is not on this grid: its episode id differs from the grid's at {ng.size} of {n} windows, first at window {int(ng[0])}")
    gt = np.asarray(ctx["gt"], np.float64)
    facts = {"n_windows": int(n), "n_episodes": int(len(np.unique(ctx["eid"]))), "eid_identical": True, "ade6_recompute_max_dev": {},
             "ck_nan_mask_equals_grid_label_mask": {}}
    for nm, d in (("A", A), ("B", B)):
        ade6 = np.linalg.norm(d["pred"].astype(np.float64) - gt, axis=-1).mean(1)
        dev = float(np.max(np.abs(ade6 - d["ade6"].astype(np.float64))))
        facts["ade6_recompute_max_dev"][nm] = dev
        if not dev <= TOL_ADE6:
            raise Refusal(f"dump {nm}: its stored per-window ade6 does not equal ADE6 recomputed from its pred and the grid's GT (max deviation {dev:.3g} m > {TOL_ADE6}): "
                          f"the windows are not in the grid's order, or the dump is from another grid")
        nan = np.isnan(d["ck_pos_err"])
        same = bool(np.array_equal(~nan, np.asarray(ctx["rc_ok"], bool)))
        facts["ck_nan_mask_equals_grid_label_mask"][nm] = same
        if not same:
            raise Refusal(f"dump {nm}: the NaN mask of `ck_pos_err` is not the grid's checkpoint-label mask (rc_ok): the windows are not the grid's")
    return facts


# ------------------------------------------------------------------------------------------------ the estimator calls
def _cell(a, b, eid, reduce, n_boot, seed, better, unit="", reason=None, extra=None):
    """One paired cell: A - B on the SAME resampled episodes (`taniteval.ci`), beside each arm's own interval (same seed, so it is the eval's own)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    eid = np.asarray(eid)
    if a.size == 0:
        return {"status": "UNAVAILABLE", "reason": reason or "no window in this cell's mask", "n_windows": 0}
    pair = paired_episode_cluster_bootstrap(a, b, eid, n_boot=n_boot, seed=seed, reduce=reduce)
    ra = episode_cluster_bootstrap(a, eid, reduce=reduce, n_boot=n_boot, seed=seed)
    rb = episode_cluster_bootstrap(b, eid, reduce=reduce, n_boot=n_boot, seed=seed)
    from taniteval.ci import resolve_reducer
    red = resolve_reducer(reduce)
    cell = {"status": "OK", "a": ra["mean"], "b": rb["mean"], "a_point": float(red(a)), "b_point": float(red(b)),
            "delta": pair["delta"], "lo": pair["lo"], "hi": pair["hi"], "separated": bool(pair["separated"]), "p_delta_gt0": pair["p_delta_gt0"],
            "n_windows": pair["n_windows"], "n_episodes": pair["n_episodes"], "n_boot": pair["n_boot"],
            "a_ci": [ra["lo"], ra["hi"]], "b_ci": [rb["lo"], rb["hi"]], "better": better, "unit": unit, "estimator": pair["estimator"]}
    for k in ("display_dp", "display_note", "degenerate", "degenerate_note"):
        if k in pair:
            cell[k] = pair[k]
    if reason:
        cell["n_reason"] = reason
    if extra:
        cell.update(extra)
    return cell


def acc_reducer(k):
    def _acc(v):
        v = np.rint(v).astype(np.int64)
        return float((v // k == v % k).mean())
    return _acc


def kappa_reducer(k):
    from taniteval.four_families import _kappa_k

    def _kap(v):
        v = np.rint(v).astype(np.int64)
        r = _kappa_k(v // k, v % k, k)
        return float("nan") if r is None else float(r)
    return _kap


def f1_reducer(k):
    """Macro-F1 of a resample: the eval's `C.macro_f1` on the (true, pred) pairs that the resample drew. Recomputed on EVERY draw."""
    def _f1(v):
        v = np.rint(v).astype(np.int64)
        if v.size == 0:
            return float("nan")
        return float(C.macro_f1(v // (k + 1), v % (k + 1), k)["macro_f1"])
    return _f1


class GoalStats:
    """Reducers of the goal head. A window is encoded as `idx * 2**22 + predicted-token bits`, so the reducer (which sees only the drawn values) can
    recover which windows were drawn and read the labels of exactly those windows. Same definitions as `v8f_mini_eval.tactical_metrics`."""

    def __init__(self, gy, gw):
        self.gy, self.gw = np.asarray(gy, bool), np.asarray(gw, bool)

    def _dec(self, v):
        v = np.rint(v).astype(np.int64)
        idx, bits = v >> GOAL_BITS, v & ((1 << GOAL_BITS) - 1)
        p = ((bits[:, None] >> np.arange(N_GOAL)) & 1).astype(bool)
        return self.gy[idx], self.gw[idx], p

    def _f1_vec(self, v):
        gy, gw, p = self._dec(v)
        tp = (p & gy & gw).sum(0).astype(np.float64)
        fp = (p & ~gy & gw).sum(0).astype(np.float64)
        fn = (~p & gy & gw).sum(0).astype(np.float64)
        valid = (gw.sum(0) > 0) & ((gy & gw).sum(0) > 0)
        den = 2 * tp + fp + fn
        f1 = np.where(tp == 0, 0.0, 2 * tp / np.where(den > 0, den, 1.0))
        return f1, valid

    def macro_f1(self, v):
        f1, valid = self._f1_vec(v)
        return float(f1[valid].mean()) if valid.any() else float("nan")

    def token_accuracy(self, v):
        gy, gw, p = self._dec(v)
        return float(((p == gy) & gw).sum() / max(int(gw.sum()), 1))

    def token_f1(self, g):
        def _f1g(v):
            f1, valid = self._f1_vec(v)
            return float(f1[g]) if valid[g] else float("nan")
        _f1g.__name__ = f"_f1_token{g}"
        return _f1g


def goal_bits(goal_prob):
    return ((np.asarray(goal_prob) > 0.5).astype(np.int64) * (1 << np.arange(N_GOAL, dtype=np.int64))).sum(1)


# ------------------------------------------------------------------------------------------------ the plan-derived reads
def plan_decisions(path_xy, dt=C.DT):
    """lat / lon / 5-way manoeuvre classes of a [N, K, 2] path, by the programme's own labeller (what `tactical_from_trajectory` does)."""
    import torch
    from taniteval import four_families as ff
    from tanitad.refs.refc_tactical import COLLAPSE_TABLE, factor_from_kinematics
    P = torch.as_tensor(np.asarray(path_xy, np.float64), dtype=torch.float32)
    dyaw, dv, v0, v1, prov = ff.maneuver_kinematics(P, dt)
    lat, lon = factor_from_kinematics(dyaw, dv, v0, v1)
    man = torch.as_tensor(COLLAPSE_TABLE, dtype=torch.long)[lat, lon]
    return lat.numpy().astype(np.int64), lon.numpy().astype(np.int64), man.numpy().astype(np.int64), prov


def goal_setting_arrays(pred, gt):
    """The per-window goal-point arrays of `four_families.tactical_goal` (float32 tensors -> float64, exactly as the eval did)."""
    from taniteval import four_families as ff
    pe = np.asarray(pred, np.float32)[:, -1].astype(np.float64)
    ge = np.asarray(gt, np.float32)[:, -1].astype(np.float64)
    err = np.linalg.norm(pe - ge, axis=-1)
    rp, rg = np.linalg.norm(pe, axis=-1), np.linalg.norm(ge, axis=-1)
    bp, bg = np.arctan2(pe[:, 1], pe[:, 0]), np.arctan2(ge[:, 1], ge[:, 0])
    db = np.degrees((bp - bg + np.pi) % (2 * np.pi) - np.pi)
    move = rg > max(ff.MIN_DS_M, 0.5)
    ratio = np.where(rg > 1e-6, rp / np.maximum(rg, 1e-6), np.nan)
    return {"err": err, "long": pe[:, 0] - ge[:, 0], "lat": pe[:, 1] - ge[:, 1], "db": db, "ratio": ratio, "move": move}


def strategic_now_frame(d, ctx):
    """S's checkpoints re-expressed in NOW (the eval's `_ckpt_to_now`), and the position / heading errors against the grid's labels."""
    import torch
    from tanitad.models.v8f_mini import transform_ckpt
    ck = transform_ckpt(torch.as_tensor(d["ckpt"]), torch.as_tensor(ctx["pose_s"]), torch.as_tensor(ctx["pose_now"])).numpy()
    return C.checkpoint_errors(ck, ctx["rc"], ctx["rc_ok"])


# ------------------------------------------------------------------------------------------------ the families
def _fmt(n):
    return f"{int(n):,}"


def tactical_block(A, B, ctx, n_boot, seed):
    eid = np.asarray(ctx["eid"])
    gt = np.asarray(ctx["gt"], np.float64)
    N = len(eid)
    out = {"plan_derived": {}, "goal_setting": {}, "heads": {}}

    # ---- plan-derived decisions: the arm's executed manoeuvre against the human's -----------------------------------------------------------------
    lat_g, lon_g, man_g, _ = plan_decisions(gt)
    la, oa, ma, prov_a = plan_decisions(A["pred"])
    lb, ob, mb, prov_b = plan_decisions(B["pred"])
    all_n = f"all {_fmt(N)} grid windows (no mask)"
    for name, g, pa, pb, k in (("lateral_decision", lat_g, la, lb, 3), ("longitudinal_decision", lon_g, oa, ob, 3), ("maneuver_5way_collapsed", man_g, ma, mb, 5)):
        ca, cb = (g * k + pa).astype(np.float64), (g * k + pb).astype(np.float64)
        out["plan_derived"][name] = {
            "accuracy": _cell(ca, cb, eid, acc_reducer(k), n_boot, seed, "higher", "", all_n),
            "kappa": _cell(ca, cb, eid, kappa_reducer(k), n_boot, seed, "higher", "", all_n)}
    out["plan_derived"]["_labeller"] = ("trajectory-derived: four_families.maneuver_kinematics -> refc_tactical.factor_from_kinematics; the same labeller reads the "
                                        "human path (grid GT) and each arm's plan. These are EXECUTED manoeuvres on both sides.")

    # ---- goal setting (the goal point at the horizon) -----------------------------------------------------------------------------------------------
    ga, gb = goal_setting_arrays(A["pred"], gt), goal_setting_arrays(B["pred"], gt)
    move = ga["move"]
    assert np.array_equal(move, gb["move"]), "the bearing mask depends on the human path only"
    n_mv, n_ex = int(move.sum()), int((~move).sum())
    reason_mv = f"{_fmt(n_mv)} of {_fmt(N)} windows: the {_fmt(n_ex)} whose human goal is within 0.5 m of the origin carry no bearing (mask: |gt[-1]| > 0.5 m)"
    for nm, key, better, unit in (("goal_point_error_m", "err", "lower", "m"), ("goal_long_bias_m", "long", None, "m"), ("goal_lat_bias_m", "lat", None, "m")):
        out["goal_setting"][nm] = _cell(ga[key], gb[key], eid, "mean", n_boot, seed, better, unit, all_n)
    mv = eid[move]
    out["goal_setting"]["goal_bearing_mae_deg"] = _cell(np.abs(ga["db"][move]), np.abs(gb["db"][move]), mv, "mean", n_boot, seed, "lower", "deg", reason_mv)
    out["goal_setting"]["goal_bearing_bias_deg"] = _cell(ga["db"][move], gb["db"][move], mv, "mean", n_boot, seed, None, "deg", reason_mv)
    out["goal_setting"]["goal_range_ratio"] = _cell(ga["ratio"][move], gb["ratio"][move], mv, "mean", n_boot, seed, None, "ratio (1.0 = the human's reach)", reason_mv)
    out["goal_setting"]["goal_point_error_m"]["_is"] = "the final-displacement error (FDE6), reported for continuity as the eval does"

    # ---- the heads against the v9 labels -------------------------------------------------------------------------------------------------------------
    has = np.asarray(ctx["has"], bool)
    n_no_row = int((~has).sum())
    for hname, logits, cls_key, k in (("lateral", "lat_logits", "lat_cls", LAT_K), ("longitudinal", "lon_logits", "lon_cls", LON_K)):
        cls = np.asarray(ctx[cls_key]).astype(np.int64)
        m = has & (cls >= 0)
        n_ign = int((has & (cls < 0)).sum())
        reason = (f"{_fmt(int(m.sum()))} of {_fmt(N)} windows are labelled (mask: has & {cls_key} >= 0): {_fmt(n_no_row)} have no v9 label row (has False) and "
                  f"{_fmt(n_ign)} have a row with no {hname} class (IGNORE)")
        true = cls[m]
        ca = (true * (k + 1) + A[logits].argmax(-1)[m]).astype(np.float64)
        cb = (true * (k + 1) + B[logits].argmax(-1)[m]).astype(np.float64)
        accA, accB = (true * k + A[logits].argmax(-1)[m]).astype(np.float64), (true * k + B[logits].argmax(-1)[m]).astype(np.float64)
        support = np.bincount(true, minlength=k).tolist()
        out["heads"][hname] = {
            "macro_f1": _cell(ca, cb, eid[m], f1_reducer(k), n_boot, seed, "higher", "", reason, {"support": support}),
            "accuracy": _cell(accA, accB, eid[m], acc_reducer(k), n_boot, seed, "higher", "", reason),
            "kappa": _cell(accA, accB, eid[m], kappa_reducer(k), n_boot, seed, "higher", "", reason),
            "_note": ("macro-F1 is over the classes present among the labels of the resample (the eval's `C.macro_f1`); accuracy and kappa are the same definitions "
                      "as the plan-derived rows, over the labelled windows. Accuracy and kappa of the heads are not in the eval JSON.")}
    # goal head
    gy = ((np.asarray(ctx["goal_y"]).astype(np.int64)[:, None] >> np.arange(N_GOAL)) & 1).astype(bool)
    gw = ((np.asarray(ctx["goal_w"]).astype(np.int64)[:, None] >> np.arange(N_GOAL)) & 1).astype(bool) & has[:, None]
    sub = np.flatnonzero(has)
    gs = GoalStats(gy[sub], gw[sub])
    idx = np.arange(len(sub), dtype=np.int64) << GOAL_BITS
    ca, cb = (idx + goal_bits(A["goal_prob"])[sub]).astype(np.float64), (idx + goal_bits(B["goal_prob"])[sub]).astype(np.float64)
    n_cells = int(gw.sum())
    reason_g = (f"{_fmt(len(sub))} of {_fmt(N)} windows have a v9 label row ({_fmt(n_no_row)} do not: has False); the cells scored are the {_fmt(n_cells)} "
                f"(window, token) pairs with goal_w set, threshold goal_prob > 0.5")
    goal = {"macro_f1": _cell(ca, cb, eid[sub], gs.macro_f1, n_boot, seed, "higher", "", reason_g),
            "token_accuracy": _cell(ca, cb, eid[sub], gs.token_accuracy, n_boot, seed, "higher", "",
                                    reason_g + " (micro accuracy over those cells: dominated by the all-negative majority, so read it beside the macro-F1)"),
            "per_token": {}}
    from tanitad.data import v8f_mini_data as D
    for g in range(N_GOAL):
        mg = gw[:, g]
        npos = int((gy[:, g] & mg).sum())
        nlab = int(mg.sum())
        nm = D.GOAL22[g] if g < len(D.GOAL22) else f"token{g}"
        if nlab == 0 or npos == 0:
            goal["per_token"][nm] = {"status": "UNAVAILABLE", "n_windows": nlab, "n_labelled": nlab, "n_positive": npos,
                                     "reason": f"no positive label for this token among its {_fmt(nlab)} labelled windows on this grid, so its F1 is undefined (the eval reports None)"}
            continue
        cell = _cell(ca, cb, eid[sub], gs.token_f1(g), n_boot, seed, "higher", "",
                     f"{_fmt(nlab)} labelled windows, {_fmt(npos)} positive")
        if cell["status"] == "OK" and not (math.isfinite(cell["lo"]) and math.isfinite(cell["hi"])):
            cell = {"status": "POINT_ONLY", "a": cell["a"], "b": cell["b"], "a_point": cell["a_point"], "b_point": cell["b_point"],
                    "delta": float(cell["a_point"] - cell["b_point"]), "n_windows": cell["n_windows"], "n_episodes": cell["n_episodes"], "n_labelled": nlab,
                    "n_positive": npos,
                    "reason": ("the token's F1 is undefined on at least one resample (no positive label in the drawn episodes), so no percentile interval exists; "
                               "the point values are reported")}
        else:
            cell["n_positive"], cell["n_labelled"] = npos, nlab
        goal["per_token"][nm] = cell
    out["heads"]["goal"] = goal
    out["unavailable_by_design"] = {
        "anchor_selection": {"status": "UNAVAILABLE", "n": int(N), "reason": "each dump's arm commits to ONE path per window: there is no candidate fan and no selection to score (a work item, not a pass)"},
        "selected_vs_executed": {"status": "UNAVAILABLE", "n": int(N), "reason": "both label streams are EXECUTED manoeuvres; scoring a declared decision against the driven path needs a tactical head's selection, which a trajectory dump does not carry (a work item, not a pass)"}}
    out["kinematics_provenance"] = {"A": prov_a, "B": prov_b}
    return out


def strategic_block(A, B, ctx, n_boot, seed):
    eid = np.asarray(ctx["eid"])
    N = len(eid)
    has = np.asarray(ctx["has"], bool)
    rc_ok = np.asarray(ctx["rc_ok"], bool)
    arcs = [int(a) for a in C.RULES["ckpt_arcs_m"]]
    out = {"status": STRATEGIC_STATUS, "arcs_m": arcs, "position_error_m": {}, "heading_error_deg": {}}
    # heading: rebuild, and trust it only if it reproduces the stored position error
    heading = {"status": "OK"}
    try:
        eA, eB = strategic_now_frame(A, ctx), strategic_now_frame(B, ctx)
        devs = {nm: float(np.nanmax(np.abs(e["pos_err_m"].astype(np.float64) - d["ck_pos_err"].astype(np.float64)))) for nm, e, d in (("A", eA, A), ("B", eB, B))}
        heading["rebuild_vs_stored_ck_pos_err_max_dev_m"] = devs
        if not all(v <= TOL_HEADING_GATE for v in devs.values()):
            heading = {"status": "UNAVAILABLE", "rebuild_vs_stored_ck_pos_err_max_dev_m": devs,
                       "reason": (f"the held-age rebuild of S's NOW-frame checkpoints does not reproduce the stored ck_pos_err (max deviation A {devs['A']:.3g} m, B {devs['B']:.3g} m "
                                  f"> {TOL_HEADING_GATE} m), so the heading error cannot be trusted; pass --ages / --hist / --max-age-t / --max-age-s / --window-start of the checkpoint's cfg")}
    except Exception as e:                                       # a work item with its reason, never a silent drop
        heading = {"status": "UNAVAILABLE", "reason": f"the NOW-frame rebuild failed ({type(e).__name__}: {e})"}
    out["heading_rebuild"] = heading
    for m, arc in enumerate(arcs):
        ok = rc_ok[:, m]
        n_ok, n_bad = int(ok.sum()), int((~ok).sum())
        n_no_row = int((~has).sum())
        n_row_bad = int((has & ~ok).sum())
        if n_bad == 0:
            reason = f"all {_fmt(N)} grid windows carry a valid checkpoint label at {arc} m"
        else:
            reason = (f"{_fmt(n_ok)} of {_fmt(N)} windows: `ck_pos_err` is NaN in both dumps at exactly the {_fmt(n_bad)} windows without a valid {arc} m checkpoint label "
                      f"({_fmt(n_no_row)} have no v9 label row, has False; {_fmt(n_row_bad)} have a row but no valid checkpoint at {arc} m)")
        out["position_error_m"][str(arc)] = _cell(A["ck_pos_err"][ok, m], B["ck_pos_err"][ok, m], eid[ok], "mean", n_boot, seed, "lower", "m", reason)
        if heading["status"] == "OK":
            out["heading_error_deg"][str(arc)] = _cell(eA["head_err_deg"][ok, m], eB["head_err_deg"][ok, m], eid[ok], "mean", n_boot, seed, "lower", "deg", reason)
        else:
            out["heading_error_deg"][str(arc)] = {"status": "UNAVAILABLE", "n_windows": n_ok, "reason": heading["reason"]}
    return out


def compute_pairs(A, B, ctx, n_boot=N_BOOT, seed=0) -> dict:
    return {"tactical": tactical_block(A, B, ctx, n_boot, seed), "strategic": strategic_block(A, B, ctx, n_boot, seed)}


# ------------------------------------------------------------------------------------------------ walking the result
def iter_cells(node, path=""):
    """Yield (dotted path, cell) for every cell dict (a dict with a `status`) under `node`."""
    if isinstance(node, dict):
        if "status" in node and ("delta" in node or node["status"] in ("UNAVAILABLE", "POINT_ONLY")) and "arcs_m" not in node:
            yield path, node
            return
        for k, v in node.items():
            if k.startswith("_") or k in ("kinematics_provenance", "heading_rebuild"):
                continue
            yield from iter_cells(v, f"{path}.{k}" if path else k)


# ------------------------------------------------------------------------------------------------ controls
def identity_control(A, ctx, n_boot, seed) -> dict:
    """The whole pipeline on A against A: every cell must read EXACTLY 0.0 [0.0, 0.0], not separated."""
    res = compute_pairs(A, A, ctx, n_boot, seed)
    bad, n = [], 0
    for p, c in iter_cells(res):
        if c["status"] == "UNAVAILABLE":
            continue
        n += 1
        if c["status"] == "POINT_ONLY":
            ok = c["delta"] == 0.0
        else:
            ok = c["delta"] == 0.0 and c["lo"] == 0.0 and c["hi"] == 0.0 and not c["separated"]
        if not ok:
            bad.append({"cell": p, "delta": c.get("delta"), "lo": c.get("lo"), "hi": c.get("hi")})
    return {"name": "identity: the pipeline run on A against A", "condition": "exactly 0.0 [0.0, 0.0], not separated, on every cell", "n_cells": n, "n_not_zero": len(bad),
            "not_zero": bad, "ok": not bad}


def _ref(raw, *path):
    cur = raw
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def json_reproduction(arm, cells, raw) -> dict:
    """Each arm's own point values, and the interval where the eval JSON has one, against the eval JSON (`raw` block) to 1e-6."""
    rows = []

    def chk(name, mine, ref, rnd):
        if ref is None or (isinstance(ref, float) and ref != ref):
            if mine is None or (isinstance(mine, float) and mine != mine):
                rows.append({"name": name, "mine": None, "json": ref, "diff": 0.0, "ok": True})
            else:
                rows.append({"name": name, "mine": mine, "json": ref, "diff": None, "ok": False})
            return
        m = round(float(mine), rnd) if rnd is not None else float(mine)
        rows.append({"name": name, "mine": m, "json": float(ref), "diff": abs(m - float(ref)), "ok": abs(m - float(ref)) <= TOL_REPRO})

    pk = "a" if arm == "A" else "b"
    ck = "a_ci" if arm == "A" else "b_ci"
    pt = pk + "_point"

    def with_ci(name, cell, ci_ref, n_ref=None):
        if cell.get("status") != "OK":
            rows.append({"name": name, "mine": cell.get("status"), "json": None, "diff": None, "ok": False})
            return
        if ci_ref is None:
            rows.append({"name": name, "mine": None, "json": None, "diff": None, "ok": False, "why": "the eval JSON has no such block"})
            return
        chk(name + ".mean", cell[pt], ci_ref.get("mean"), 4)
        chk(name + ".lo", cell[ck][0], ci_ref.get("lo"), None)
        chk(name + ".hi", cell[ck][1], ci_ref.get("hi"), None)
        if "n_windows" in ci_ref:
            rows.append({"name": name + ".n_windows", "mine": cell["n_windows"], "json": ci_ref["n_windows"], "diff": abs(cell["n_windows"] - ci_ref["n_windows"]),
                         "ok": cell["n_windows"] == ci_ref["n_windows"]})
            rows.append({"name": name + ".n_episodes", "mine": cell["n_episodes"], "json": ci_ref["n_episodes"], "diff": abs(cell["n_episodes"] - ci_ref["n_episodes"]),
                         "ok": cell["n_episodes"] == ci_ref["n_episodes"]})

    tfam = _ref(raw, "families", "tactical_from_trajectory") or {}
    T = cells["tactical"]
    for nm in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
        for met in ("accuracy", "kappa"):
            with_ci(f"tactical.plan_derived.{nm}.{met}", T["plan_derived"][nm][met], _ref(tfam, nm, "ci", met))
            if T["plan_derived"][nm][met].get("status") == "OK":
                chk(f"tactical.plan_derived.{nm}.{met}.block", T["plan_derived"][nm][met][pt], _ref(tfam, nm, met), 4)
    gset = _ref(tfam, "goal_setting") or {}
    for nm in ("goal_point_error_m", "goal_long_bias_m", "goal_lat_bias_m", "goal_bearing_mae_deg"):
        with_ci(f"tactical.goal_setting.{nm}", T["goal_setting"][nm], _ref(gset, "ci", nm))
    for nm in ("goal_bearing_bias_deg", "goal_range_ratio"):
        c = T["goal_setting"][nm]
        if c.get("status") == "OK":
            chk(f"tactical.goal_setting.{nm}", c[pt], _ref(gset, nm), 4)
    for nm, key in (("goal_point_error_m", "goal_point_error_m"), ("goal_bearing_mae_deg", "goal_bearing_mae_deg"), ("goal_long_bias_m", "goal_long_bias_m"),
                    ("goal_lat_bias_m", "goal_lat_bias_m")):
        c = T["goal_setting"][nm]
        if c.get("status") == "OK":
            chk(f"tactical.goal_setting.{nm}.block", c[pt], _ref(gset, key), 4)
    tac = _ref(raw, "tactical") or {}
    for hname, jk in (("lateral", "lat"), ("longitudinal", "lon")):
        c = T["heads"][hname]["macro_f1"]
        if c.get("status") == "OK":
            chk(f"tactical.heads.{hname}.macro_f1", c[pt], _ref(tac, jk, "macro_f1"), None)
            rows.append({"name": f"tactical.heads.{hname}.n_windows", "mine": c["n_windows"], "json": _ref(tac, jk, "n"), "diff": abs(c["n_windows"] - (_ref(tac, jk, "n") or -1)),
                         "ok": c["n_windows"] == _ref(tac, jk, "n")})
        else:
            rows.append({"name": f"tactical.heads.{hname}.macro_f1", "mine": c.get("status"), "json": None, "diff": None, "ok": False})
    c = T["heads"]["goal"]["macro_f1"]
    if c.get("status") == "OK":
        chk("tactical.heads.goal.macro_f1", c[pt], _ref(tac, "goal_macro_f1"), None)
    for nm, tc in T["heads"]["goal"]["per_token"].items():
        ref = _ref(tac, "goal_f1_per_token", nm)
        if tc.get("status") == "UNAVAILABLE":
            rows.append({"name": f"tactical.heads.goal.per_token.{nm}", "mine": None, "json": ref, "diff": 0.0 if ref is None else None, "ok": ref is None})
        else:
            chk(f"tactical.heads.goal.per_token.{nm}", tc[pt], ref, None)
    S = cells["strategic"]
    sraw = _ref(raw, "strategic") or {}
    for arc in S["arcs_m"]:
        with_ci(f"strategic.position_error_m@{arc}", S["position_error_m"][str(arc)], _ref(sraw, f"pos_err_m@{arc}"))
        hc = S["heading_error_deg"][str(arc)]
        if hc.get("status") == "OK":
            with_ci(f"strategic.heading_error_deg@{arc}", hc, _ref(sraw, f"head_err_deg@{arc}"))
        else:
            rows.append({"name": f"strategic.heading_error_deg@{arc}", "mine": hc.get("status"), "json": None, "diff": None, "ok": False})
    bad = [r for r in rows if not r["ok"]]
    diffs = [r["diff"] for r in rows if r["diff"] is not None]
    return {"arm": arm, "n_checked": len(rows), "n_failed": len(bad), "max_abs_diff": max(diffs) if diffs else None, "failed": bad, "ok": not bad}


# ------------------------------------------------------------------------------------------------ summary + rendering
def summarize(res) -> dict:
    fam = {}
    for p, c in iter_cells(res):
        f = p.split(".")[0]
        d = fam.setdefault(f, {"n_cells": 0, "n_ok": 0, "n_unavailable": 0, "n_point_only": 0, "separated": []})
        d["n_cells"] += 1
        if c["status"] == "OK":
            d["n_ok"] += 1
            if c["separated"]:
                if c.get("better") in ("lower", "higher"):
                    worse = (c["delta"] > 0) == (c["better"] == "lower")
                    word = "A worse" if worse else "A better"
                else:
                    word = "A higher" if c["delta"] > 0 else "A lower"
                d["separated"].append({"cell": p, "delta": c["delta"], "lo": c["lo"], "hi": c["hi"], "reading": word})
        elif c["status"] == "POINT_ONLY":
            d["n_point_only"] += 1
        else:
            d["n_unavailable"] += 1
    return fam


def _num(x, dp=4):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:.{dp}f}"


def _row(path, c, a_label="A", b_label="B"):
    tok = f"{c['n_labelled']:,} labelled, {c['n_positive']} pos" if "n_labelled" in c else None
    if c["status"] == "UNAVAILABLE":
        return f"| `{path}` | UNAVAILABLE | | | {tok or c.get('n_windows', '')} | {c['reason']} |"
    if c["status"] == "POINT_ONLY":
        return f"| `{path}` | {_num(c['a'])} | {_num(c['b'])} | {c['delta']:+.4f} (no interval) | {tok} | {c['reason']} |"
    sep = "**yes**" if c["separated"] else "no"
    read = ""
    if c["separated"] and c.get("better") in ("lower", "higher"):
        worse = (c["delta"] > 0) == (c["better"] == "lower")
        read = f" ({a_label} {'worse' if worse else 'better'})"
    n = f"{tok}; {c['n_windows']:,} / {c['n_episodes']}" if tok else f"{c['n_windows']:,} / {c['n_episodes']}"
    return f"| `{path}` | {_num(c['a'])} | {_num(c['b'])} | {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | {n} | {sep}{read} |"


def render_markdown(res, a_label="A", b_label="B") -> str:
    lines = [f"Contrast: {a_label} - {b_label}. Tier: {res['tier']}. Estimator: {res['estimator']}.", ""]
    head = f"| cell | {a_label} | {b_label} | {a_label} - {b_label} [95 % CI] | n (windows / episodes) | separated |"
    sepl = "|---|---|---|---|---|---|"
    groups = (("TACTICAL, plan-derived decisions and goal point", "tactical.plan_derived", "tactical.goal_setting"), ("TACTICAL, heads (lat / lon / goal)", "tactical.heads",),
              ("STRATEGIC, S's checkpoint error", "strategic",))
    for title, *prefixes in groups:
        lines += [f"### {title}", "", head, sepl]
        for p, c in iter_cells({"tactical": res["tactical"], "strategic": res["strategic"]}):
            if any(p == pre or p.startswith(pre + ".") for pre in prefixes):
                lines.append(_row(p, c, a_label, b_label))
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ the run
def run(a) -> tuple[dict, int]:
    A, B = load_dump(a.a), load_dump(a.b)
    ja, pja = load_json_beside(a.a, a.a_json)
    jb, pjb = load_json_beside(a.b, a.b_json)
    ages = a.ages
    if ages == "auto":
        got = {j.get("ages") for j in (ja, jb) if j is not None and j.get("ages")}
        if len(got) > 1:
            raise Refusal(f"the two eval JSONs disagree on `ages`: {sorted(got)}")
        ages = got.pop() if got else "cycle"
    kin_mode = a.kin_mode
    if a.ctx_npz:
        ctx = load_ctx_npz(a.ctx_npz)
    else:
        ctx = build_context(a.eval_grid, a.eval_kin, a.eval_join, kin_mode, ages, a.hist, a.max_age_t, a.max_age_s, a.window_start)
    for nm, j in (("A", ja), ("B", jb)):
        if j is not None and not a.ctx_npz and j.get("kin_mode") not in (None, kin_mode):
            raise Refusal(f"dump {nm}'s eval JSON says kin_mode {j.get('kin_mode')!r}; this run builds the grid GT with {kin_mode!r} (pass --kin-mode)")
    facts = check_alignment(A, B, ctx)
    res = {"schema": SCHEMA, "tier": TIER, "contrast": "A - B (A is the arm under test, B the base)",
           "estimator": f"paired episode-cluster bootstrap, B={a.n_boot}, {facts['n_episodes']} episodes",
           "variance_answered": ("another draw of EPISODES with the models held fixed. Each arm is ONE training seed: a separated cell is necessary, not sufficient, for a lever "
                                 "effect (H-ESTIM-SEED-1); no replicate arm exists in this read"),
           "strategic_family": STRATEGIC_STATUS,
           "a": {"label": a.label_a, "npz": A["_path"], "sha256": A["_sha256"], "eval_json": pja if ja is not None else None},
           "b": {"label": a.label_b, "npz": B["_path"], "sha256": B["_sha256"], "eval_json": pjb if jb is not None else None},
           "inputs_sha256": ctx.get("_sha256", {}), "n_windows": facts["n_windows"], "n_episodes": facts["n_episodes"], "n_boot": a.n_boot, "seed": a.seed,
           "ages": ages, "kin_mode": kin_mode, "alignment": facts}
    res.update(compute_pairs(A, B, ctx, a.n_boot, a.seed))
    res["summary"] = summarize({"tactical": res["tactical"], "strategic": res["strategic"]})
    controls = []
    if a.identity_control:
        controls.append(identity_control(A, ctx, a.n_boot, a.seed))
    else:
        controls.append({"name": "identity", "ok": None, "skipped": "--no-identity-control"})
    cells = {"tactical": res["tactical"], "strategic": res["strategic"]}
    for arm, j, pj in (("A", ja, pja), ("B", jb, pjb)):
        if a.json_controls and j is not None and isinstance(j.get("raw"), dict):
            r = json_reproduction(arm, cells, j["raw"])
            r["name"] = f"reproduction of arm {arm}'s own point values against {os.path.basename(pj)}"
            controls.append(r)
        else:
            controls.append({"name": f"reproduction of arm {arm}'s point values against its eval JSON", "ok": None,
                             "unavailable": ("--no-json-controls" if not a.json_controls else f"no eval JSON with a `raw` block beside the dump ({pj})")})
    res["controls"] = controls
    res["controls_ok"] = all(c["ok"] for c in controls if c.get("ok") is not None)
    return res, (0 if res["controls_ok"] else 3)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="the arm under test: an eval `run` dump (eval_<tag>.npz)")
    ap.add_argument("--b", required=True, help="the base arm: an eval `run` dump on the same windows")
    ap.add_argument("--out", required=True, help="the result json")
    ap.add_argument("--md", help="also write the result tables as markdown")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--a-json", help="the eval json of A (default: beside the npz)")
    ap.add_argument("--b-json", help="the eval json of B (default: beside the npz)")
    ap.add_argument("--eval-grid", default=os.path.join(RAW, "eval_grid_windows.json"))
    ap.add_argument("--eval-kin", default=os.path.join(RAW, "eval141_kin.npz"))
    ap.add_argument("--eval-join", default=os.path.join(RAW, "join_eval141.npz"))
    ap.add_argument("--kin-mode", choices=("yaw", "steer"), default="yaw")
    ap.add_argument("--ages", choices=("auto", "cycle", "fresh"), default="auto", help="the held-age scheme of the evals (auto: from the eval JSONs, else cycle)")
    ap.add_argument("--hist", type=int, default=4)
    ap.add_argument("--max-age-t", type=int, default=2)
    ap.add_argument("--max-age-s", type=int, default=6)
    ap.add_argument("--window-start", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-identity-control", dest="identity_control", action="store_false")
    ap.add_argument("--no-json-controls", dest="json_controls", action="store_false")
    ap.add_argument("--ctx-npz", help="TEST INPUT: a pre-built context npz (eid, gt, has, lat_cls, lon_cls, goal_y, goal_w, rc, rc_ok, pose_s, pose_now)")
    a = ap.parse_args(argv)
    try:
        import torch
        torch.set_num_threads(8)
        res, rc = run(a)
    except Refusal as e:
        print(f"PAIRFAM REFUSED: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"PAIRFAM INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_clean(res), f, indent=1)
    if a.md:
        with open(a.md, "w", encoding="utf-8", newline="\n") as f:
            f.write(render_markdown(res, a.label_a, a.label_b))
    print(json.dumps({"controls_ok": res["controls_ok"], "summary": res["summary"]}, indent=1))
    if rc:
        print("PAIRFAM CONTROL FAILED: see `controls` in the output json", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
