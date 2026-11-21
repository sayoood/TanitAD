"""SPEC_WPB_LADDER sec. 3-5 + amendment A1 -- the SCORER (CPU, numpy). Reads the eval passes written by
`ladder_eval.py` and each arm's own `metrics.jsonl`, and writes the per-family tables, the paired deltas, the
training-variance floor F, R = E / F and every bar of the registered SPEC as a literal.

Layout (the chain's):  <W>/eval/<arm>/<row>_s<seed>.{npz,json}    <W>/arms/<arm>/run/metrics.jsonl
Sub-commands:
  size-k  --w <W>                          -> SPEC sec. 4.2's k from V-R8's FIRST 20 % of logged gs rows
  score   --w <W> [--v9-train npz --v9-train-md5 m]   -> <W>/LADDER_SCORE.json (+ a short print)

ESTIMATORS (named, SPEC sec. 3): every CI is the PAIRED episode-cluster bootstrap over the eval139 episodes, B 2000,
fixed seed-0 draws shared by EVERY arm / row / sampler seed (so two arms are resampled on the same episodes), ratio of
sums per episode (a masked metric's denominator is its scored-window count). It answers "another draw of EPISODES".
F (the replicate floor) answers "another TRAINING run". Sampler seeds 0 / 1 answer "another INFERENCE".
A metric is oriented by `better` (+1 higher is better, -1 lower is better); "gain" = better * (treatment - base).
Every rung is read only when ALL its arms (incl. replicate + regression) have every pass it needs (SPEC sec. 10.4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B = 2000
SEED = 0
SEEDS = (0, 1)
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
SLOT_T = np.array([h * 0.1 for h in HORIZONS])
TAU_C = 0.18063741505146028            # the run's own --nav-compliance-tau-rad (10.35 deg), = route_metrics.TAU_C
STALL_M = 0.05
TURN_DEG, STRAIGHT_DEG, HEAD_DEG, MIN_LEN_M = 30.0, 10.0, 15.0, 5.0
MIN_DS_MPS = None                      # read from taniteval.four_families at import (the yaw-rate standstill mask)
LAT_SIDE = np.array([0, 1, -1, 0, 1, -1, 1, -1])            # v7 lat ids -> side
LAT3_OF_V7 = np.array([0, 0, 0, 0, 0, 0, 1, 2])             # 8-way -> lat3 (K / TURN_L / TURN_R), RESULT_STAGE2 reading
HEADLINE_MAP = ("drivable", "lane")                         # SPEC sec. 3, both headline bands summed per class
#: the per-family HEADLINE metrics (SPEC sec. 3); `better`: +1 higher is better, -1 lower is better
METRICS = {
    # LATERAL
    "dir_correct_turn": ("LATERAL", +1), "head15_all": ("LATERAL", +1), "cross_abs_6s": ("LATERAL", -1),
    "curv_mae_0_2s": ("LATERAL", -1), "yaw_mae_0_2s": ("LATERAL", -1),
    # LONGITUDINAL
    "along_abs_6s": ("LONGITUDINAL", -1), "speed_mae_0_2s": ("LONGITUDINAL", -1),
    "speed_mae_2_6s": ("LONGITUDINAL", -1), "progress_rel_err_6s": ("LONGITUDINAL", -1),
    "ceil_violation": ("LONGITUDINAL", -1),
    # plan-level ADE (the GT-straight / LEGAL clauses)
    "ade_all": ("PLAN", -1), "ade_straight": ("PLAN", -1),
    # TACTICAL (R8-4 i)
    "lat3_acc_v9": ("TACTICAL", +1), "lat3_f1_v9": ("TACTICAL", +1), "lat3_acc_gtplan": ("TACTICAL", +1),
    "cons_mae_lat": ("TACTICAL", -1), "cons_mae_lon": ("TACTICAL", -1), "cons_mae_spd": ("TACTICAL", -1),
    # A1 L5 + perception (10 cm map, raw argmax, both headline bands)
    "off_drivable": ("SAFETY", -1),
    "map_iou_drivable": ("PERCEPTION", +1), "map_iou_lane": ("PERCEPTION", +1),
}
#: the in-run (metrics.jsonl) perception scalars: no window axis -> no bootstrap, compared against F only
INRUN = {"eval_box3d_ap2m": +1, "eval_agent_ap2m": +1}
RUNGS = {"L1": ("V0", "V-R8", "V-R8d", "V0r", "V-R8r"),
         "L2": ("V-R8", "V-TACk", "V-TACk-roll", "V-TACkr", "V0", "V0r", "V-R8r"),
         "L3": ("V-R8", "V-MAP4", "V-MAP4-roll", "V-MAP4r", "V0", "V0r", "V-R8r"),
         "L4": ("V-R8", "V-VSHUF", "V0", "V0r", "V-R8r"),
         "L4b": ("V-R8", "V-R8-E8", "V-R8-E8-roll", "V0", "V0r", "V-R8r"),
         "L5": ("V-R8", "V-R8-DRV", "V-R8-DRV-roll", "V0", "V0r", "V-R8r")}
#: eval rows each rung needs on which arm (the base row everywhere it is read)
ROW_NEEDS = {"L1": {"V-R8": ("base", "legal", "rc_off", "rc_shuf")}, "L4": {"V-R8": ("base", "vmax_off", "vmax_shuf")}}


# ------------------------------------------------------------------------------------------------- #
# geometry (route_metrics / analyze_route definitions, re-typed and pinned by test_ladder_score.py)  #
# ------------------------------------------------------------------------------------------------- #
def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def terminal_heading(P):
    P = np.asarray(P, np.float64)
    d = P[..., -1, :] - P[..., -2, :]
    th = np.arctan2(d[..., 1], d[..., 0])
    return np.where(np.hypot(d[..., 0], d[..., 1]) < STALL_M, 0.0, th)


def dir_class(theta, tau=TAU_C):
    th = np.asarray(theta, np.float64)
    return np.where(th >= tau, 1, np.where(th <= -tau, -1, 0)).astype(np.int8)


def path_length(P, upto=None):
    P = np.asarray(P, np.float64)
    if upto is not None:
        P = P[..., :upto, :]
    Q = np.concatenate([np.zeros(P.shape[:-2] + (1, 2)), P], axis=-2)
    return np.linalg.norm(np.diff(Q, axis=-2), axis=-1).sum(-1)


def gt_class(gt, gv):
    th = terminal_heading(gt)
    ok = np.asarray(gv)[..., -1] & (path_length(gt) >= MIN_LEN_M)
    deg = np.degrees(np.abs(th))
    cls = np.full(th.shape, "unclassified", dtype=object)
    cls[ok & (deg >= TURN_DEG) & (th > 0)] = "turnL"
    cls[ok & (deg >= TURN_DEG) & (th < 0)] = "turnR"
    cls[ok & (deg < STRAIGHT_DEG)] = "straight"
    cls[ok & (deg >= STRAIGHT_DEG) & (deg < TURN_DEG)] = "gentle"
    return cls, th


def seg_speeds(P):
    """[W, 8] mean speed of each slot segment (first from the origin), m/s."""
    P = np.asarray(P, np.float64)
    Q = np.concatenate([np.zeros(P.shape[:-2] + (1, 2)), P], axis=-2)
    dt = np.diff(np.concatenate([[0.0], SLOT_T]))
    return np.linalg.norm(np.diff(Q, axis=-2), axis=-1) / dt


def seq_geom(wp4, dt=0.5):
    """`taniteval.four_families._seq_geometry` on the 0-2 s slots (the launch tree's own; its `pair_valid` IS the
    yaw-rate / curvature standstill mask, RETR-2026-09-26-YAWMASK)."""
    import torch
    from taniteval import four_families as ff
    return {k: (v.numpy() if hasattr(v, "numpy") else v)
            for k, v in ff._seq_geometry(torch.as_tensor(np.asarray(wp4, np.float64)), dt=dt).items()}


def planned_vmax(P):
    """the ceiling filter's OWN statistic (`refcv6_selection.planned_max_speed`) on the emitted plan."""
    import torch
    from tanitad.refs import refcv6_selection as v6sel
    t = torch.as_tensor(np.asarray(P, np.float32))[:, None]
    return v6sel.planned_max_speed(t, horizons=HORIZONS, tick_s=0.1)[:, 0].double().numpy()


# ------------------------------------------------------------------------------------------------- #
# per-window (num, den) for every metric                                                             #
# ------------------------------------------------------------------------------------------------- #
def nd_mean(x, mask):
    x = np.asarray(x, np.float64)
    m = np.asarray(mask, bool) & np.isfinite(x)
    return np.where(m, x, 0.0), m.astype(np.float64)


def window_metrics(z, cons_median=None) -> dict:
    """-> {metric: (num [W], den [W])} plus the lat3 confusion pieces under '_lat3'. Ratio-of-sums per metric."""
    P, gt, gv = z["traj"].astype(np.float64), z["gt"].astype(np.float64), z["gt_valid"].astype(bool)
    W = len(P)
    cls, th_gt = gt_class(gt, gv)
    target = np.where(np.isin(cls, ["turnL", "turnR", "straight", "gentle"]), dir_class(th_gt), 9)
    turn = np.isin(cls, ["turnL", "turnR"])
    out = {}
    thp = terminal_heading(P)
    out["dir_correct_turn"] = nd_mean((dir_class(thp) == target).astype(float), turn & (target != 9))
    herr = np.degrees(np.abs(wrap(thp - th_gt)))
    out["head15_all"] = nd_mean((herr <= HEAD_DEG).astype(float), (target != 9) & gv[:, -1])
    d = np.linalg.norm(P - gt, axis=-1)
    n = gv.sum(1)
    ade = np.where(n > 0, (d * gv).sum(1) / np.maximum(n, 1), np.nan)
    out["ade_all"] = nd_mean(ade, target != 9)
    out["ade_straight"] = nd_mean(ade, cls == "straight")
    out["cross_abs_6s"] = nd_mean(np.abs(P[:, 7, 1] - gt[:, 7, 1]), gv[:, 7])
    out["along_abs_6s"] = nd_mean(np.abs(P[:, 7, 0] - gt[:, 7, 0]), gv[:, 7])
    gp, gg = seq_geom(P[:, :4]), seq_geom(gt[:, :4])
    v4 = gv[:, :4]
    sp = np.abs(gp["speed"] - gg["speed"])
    out["speed_mae_0_2s"] = nd_mean((sp * v4).sum(1) / np.maximum(v4.sum(1), 1), v4.any(1))
    pv = gp["pair_valid"] & gg["pair_valid"] & v4[:, 1:] & v4[:, :-1]
    out["curv_mae_0_2s"] = nd_mean((np.abs(gp["curvature"] - gg["curvature"]) * pv).sum(1)
                                   / np.maximum(pv.sum(1), 1), pv.any(1))
    out["yaw_mae_0_2s"] = nd_mean((np.abs(gp["yaw_rate"] - gg["yaw_rate"]) * pv).sum(1)
                                  / np.maximum(pv.sum(1), 1), pv.any(1))
    # 2-6 s: the four 1-s segments slot 3->4 ... 6->7, scored where GT is valid at both ends
    vs = gv[:, 3:8]
    sv = vs[:, 1:] & vs[:, :-1]
    s26 = np.abs(seg_speeds(P)[:, 4:8] - seg_speeds(gt)[:, 4:8])
    out["speed_mae_2_6s"] = nd_mean((s26 * sv).sum(1) / np.maximum(sv.sum(1), 1), sv.any(1))
    Lg, Lp = path_length(gt), path_length(P)
    out["progress_rel_err_6s"] = nd_mean(np.abs(Lp - Lg) / np.maximum(Lg, 1e-9), gv[:, 7] & (Lg >= MIN_LEN_M))
    vl = np.asarray(z["v_lim"], np.float64)
    fin = np.isfinite(vl)
    viol = np.zeros(W)
    if fin.any():
        viol[fin] = (planned_vmax(P[fin]) > vl[fin] + 1e-6).astype(float)
    out["ceil_violation"] = nd_mean(viol, fin)
    # TACTICAL: lat3 = 8-way argmax mapped (RESULT_STAGE2 reading), vs v9's label and vs the GT plan's class
    pred3 = LAT3_OF_V7[np.asarray(z["p_lat"]).argmax(1)]
    lv9 = np.asarray(z["lat_v9"]).astype(np.int64)
    ok9 = (lv9 >= 0) & (lv9 < 8)
    tgt9 = np.where(ok9, LAT3_OF_V7[np.clip(lv9, 0, 7)], -1)
    out["lat3_acc_v9"] = nd_mean((pred3 == tgt9).astype(float), ok9)
    exc_cls = gt_lat3(gt, gv)
    out["lat3_acc_gtplan"] = nd_mean((pred3 == exc_cls).astype(float), exc_cls >= 0)
    out["_lat3"] = (pred3, tgt9)
    # constraint MAE (normalised units) per family, beside the TRAIN-median constant
    if "cons_pred" in z:
        cp, ct, cm = (np.asarray(z[k], np.float64) for k in ("cons_pred", "cons_tgt", "cons_mask"))
        cm = cm.astype(bool) & np.isfinite(ct)
        for nm, sl in (("lat", slice(0, 12)), ("lon", slice(12, 22)), ("spd", slice(22, 26))):
            e = np.abs(cp[:, sl] - ct[:, sl]) * cm[:, sl]
            out[f"cons_mae_{nm}"] = (e.sum(1), cm[:, sl].sum(1).astype(np.float64))
            if cons_median is not None:
                ec = np.abs(np.asarray(cons_median, np.float64)[sl][None] - np.where(cm, ct, 0.0)[:, sl]) * cm[:, sl]
                out[f"cons_const_{nm}"] = (ec.sum(1), cm[:, sl].sum(1).astype(np.float64))
    if "off_drv" in z:
        w = np.asarray(z["drv_w"], np.float64) > 0
        out["off_drivable"] = nd_mean(np.asarray(z["off_drv"], np.float64), w)
    if "map_inter" in z:
        cells = [str(c) for c in z["map_cells"]]
        I, U = np.asarray(z["map_inter"], np.float64), np.asarray(z["map_union"], np.float64)
        for cl in HEADLINE_MAP:
            js = [i for i, c in enumerate(cells) if c.startswith(cl + "_")]
            out[f"map_iou_{cl}"] = (I[:, js].sum(1), U[:, js].sum(1))
            for i in js:
                out[f"map_iou_cell_{cells[i]}"] = (I[:, i], U[:, i])
    return out


def gt_lat3(gt, gv):
    """lat3 of the GT plan (tactical_conditioning.lat3_class: 30 deg heading EXCURSION, LANE_KEEP under 5 m); -1 when
    the 6-s GT is invalid."""
    from taniteval import tactical_conditioning as tc
    c = tc.lat3_class(gt)
    return np.where(np.asarray(gv)[:, -1] & (c >= 0), c, -1)


# ------------------------------------------------------------------------------------------------- #
# bootstrap                                                                                         #
# ------------------------------------------------------------------------------------------------- #
def episodes(sha12):
    eps, inv = np.unique(np.asarray(sha12).astype(str), return_inverse=True)
    return eps, inv


def make_draws(n_ep, b=B, seed=SEED):
    return np.random.default_rng(seed).integers(0, n_ep, (b, n_ep))


def counts_of(draws, n_ep):
    return np.stack([np.bincount(d, minlength=n_ep) for d in draws]).astype(np.float64)       # [B, E]


def boot_ratio(num, den, inv, cnt):
    E = cnt.shape[1]
    Nn = np.bincount(inv, weights=num, minlength=E)
    Dd = np.bincount(inv, weights=den, minlength=E)
    pt = Nn.sum() / Dd.sum() if Dd.sum() > 0 else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (cnt @ Nn) / (cnt @ Dd)
    return pt, bs, int(Dd.sum())


def macro_f1_from_conf(C):
    """C [..., 3, 3] (target, pred) -> macro-F1 over classes with n > 0 (tactical_conditioning.class_report rule)."""
    tp = np.diagonal(C, axis1=-2, axis2=-1)
    n = C.sum(-1)
    npred = C.sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = tp / n
        prec = np.where(npred > 0, tp / np.maximum(npred, 1), np.nan)
        f1 = np.where((n > 0) & np.isfinite(prec) & ((prec + rec) > 0), 2 * prec * rec / (prec + rec), 0.0)
    f1 = np.where(n > 0, f1, np.nan)
    return np.nanmean(f1, axis=-1)


def boot_f1(pred, tgt, inv, cnt):
    E = cnt.shape[1]
    ok = tgt >= 0
    C = np.zeros((E, 3, 3))
    np.add.at(C, (inv[ok], tgt[ok], pred[ok]), 1.0)
    pt = float(macro_f1_from_conf(C.sum(0)))
    bs = macro_f1_from_conf(np.einsum("be,eij->bij", cnt, C))
    return pt, bs, int(ok.sum())


def ci(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [None, None]


# ------------------------------------------------------------------------------------------------- #
# loading                                                                                           #
# ------------------------------------------------------------------------------------------------- #
def load_pass(p: Path):
    z = np.load(str(p), allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["_json"] = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    return d


def read_metrics(run: Path):
    rows = []
    f = run / "metrics.jsonl"
    if not f.is_file():
        return rows
    for ln in f.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return rows


def gs_median(rows, key, lo_frac, hi_frac):
    st = [r for r in rows if key in r and r.get(key) is not None and np.isfinite(float(r[key]))]
    if not st:
        return None
    steps = np.array([int(r.get("step", 0)) for r in st])
    smax = max(int(r.get("step", 0)) for r in rows)
    sel = (steps > lo_frac * smax) & (steps <= hi_frac * smax)
    v = np.array([float(r[key]) for r in st])[sel]
    if not v.size:
        return None
    return {"median": float(np.median(v)), "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
            "n": int(v.size), "window_steps": [lo_frac * smax, hi_frac * smax]}


def final_eval_row(rows):
    ev = [r for r in rows if any(k.startswith("eval_") for k in r)]
    return ev[-1] if ev else {}


# ------------------------------------------------------------------------------------------------- #
# sizing (SPEC sec. 4.2)                                                                             #
# ------------------------------------------------------------------------------------------------- #
def k_from(p, n, G):
    """the smallest root above 1 of (0.99 n^2) k^2 + (0.98 c) k - 0.01 R^2 = 0, c = pG^2 - n^2, R^2 = G^2 - 2pG^2 + n^2;
    rounded UP to 2 significant figures, capped at 100. None when no root above 1 exists."""
    c = p * G * G - n * n
    R2 = G * G - 2 * p * G * G + n * n
    a, b, cc = 0.99 * n * n, 0.98 * c, -0.01 * R2
    disc = b * b - 4 * a * cc
    if a <= 0 or disc < 0:
        return None
    roots = sorted(r for r in ((-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)) if r > 1)
    if not roots:
        return None
    k = roots[0]
    e = math.floor(math.log10(k)) - 1
    k = math.ceil(k / 10 ** e - 1e-9) * 10 ** e
    return float(min(k, 100.0))


def cmd_size_k(w: Path) -> int:
    rows = read_metrics(w / "arms" / "V-R8" / "run")
    got = {k: gs_median(rows, k, 0.0, 0.2) for k in ("gs_trunk_proj_tac_v6", "gs_trunk_norm_tac_v6",
                                                      "gs_trunk_norm_total")}
    if any(v is None for v in got.values()):
        print("SIZE_K INCONCLUSIVE: missing gs readings in V-R8's first 20 %", {k: v is None for k, v in got.items()})
        return 3
    k = k_from(got["gs_trunk_proj_tac_v6"]["median"], got["gs_trunk_norm_tac_v6"]["median"],
               got["gs_trunk_norm_total"]["median"])
    rec = {"rule": "SPEC_WPB_LADDER sec. 4.2", "readings": got, "k": k}
    (w / "SIZE_K.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("SIZE_K", k)
    if k is None:
        return 4
    return 0


# ------------------------------------------------------------------------------------------------- #
# scoring                                                                                           #
# ------------------------------------------------------------------------------------------------- #
def cons_train_median(npz, md5):
    """SPEC sec. 3: the TRAIN-median constant of every normalised constraint target (the trainer's own
    `v9_constraint_targets` over EVERY train row; median over the rows where the target is supervised)."""
    import torch
    from tanitad.data import v9_labels as V9
    from tanitad.refs import refcv8_conditioning as r8c
    rel = V9.load_v9_release(str(npz), expect_md5=str(md5))
    R = rel.rows

    def cols(names):
        return np.stack([np.asarray(R[n], np.float32) if n in R else np.full(rel.n_rows, np.nan, np.float32)
                         for n in names], 1)
    t = r8c.v9_constraint_targets(torch.as_tensor(cols(V9.LAT_CONSTRAINTS)), torch.as_tensor(cols(V9.LON_CONSTRAINTS)),
                                  torch.as_tensor(cols(V9.SPEED_GOAL)),
                                  torch.as_tensor(np.asarray(R["lat_v7id_a"]).astype(np.int64)),
                                  torch.as_tensor(np.asarray(R["lon_v7id"]).astype(np.int64)))
    T = torch.cat([t["lat"], t["lon"], t["speed"]], -1).double().numpy()
    M = torch.cat([t["lat_m"], t["lon_m"], t["speed_m"]], -1).numpy().astype(bool)
    med = np.array([float(np.median(T[M[:, j], j])) if M[:, j].any() else 0.0 for j in range(T.shape[1])])
    return med, {"n_rows": int(rel.n_rows), "n_supervised": M.sum(0).astype(int).tolist(), "md5": rel.md5}


def score_pass(z, inv, cnt, cons_median):
    wm = window_metrics(z, cons_median)
    res = {}
    for k, v in wm.items():
        if k == "_lat3":
            continue
        pt, bs, n = boot_ratio(v[0], v[1], inv, cnt)
        res[k] = {"pt": pt, "bs": bs, "n": n}
    pred3, tgt9 = wm["_lat3"]
    pt, bs, n = boot_f1(pred3, tgt9, inv, cnt)
    res["lat3_f1_v9"] = {"pt": pt, "bs": bs, "n": n}
    ok = tgt9 >= 0
    if ok.any():                                    # the majority control (same windows; a constant predictor)
        maj = np.bincount(tgt9[ok], minlength=3).argmax()
        res["lat3_majority_v9"] = {"pt": float((tgt9[ok] == maj).mean()), "bs": None, "n": int(ok.sum()),
                                   "class": int(maj)}
    return res


def paired(a, b, better):
    """treatment a vs base b on the SAME windows and draws -> delta, CI, gain, separated (better / worse)."""
    if a is None or b is None or a.get("bs") is None or b.get("bs") is None:
        return None
    dl = a["pt"] - b["pt"]
    lo, hi = ci(a["bs"] - b["bs"])
    if lo is None:
        return {"delta": dl, "ci": [None, None], "gain": better * dl, "sep_better": False, "sep_worse": False}
    g_lo, g_hi = sorted((better * lo, better * hi))
    return {"delta": float(dl), "ci": [lo, hi], "gain": float(better * dl), "gain_ci": [g_lo, g_hi],
            "sep_better": bool(g_lo > 0), "sep_worse": bool(g_hi < 0)}


def floor(P, m, s, pairs):
    """F = max |rep - prim| over the named (replicate, primary) pairs, on point estimates at sampler seed s."""
    vals = []
    for r_, p_ in pairs:
        a, b = P.get((r_, "base", s), {}).get(m), P.get((p_, "base", s), {}).get(m)
        if a is None or b is None or not (np.isfinite(a["pt"]) and np.isfinite(b["pt"])):
            return None
        vals.append(abs(a["pt"] - b["pt"]))
    return float(max(vals)) if vals else None


def R_of(gain, F):
    if gain is None or F is None:
        return None
    if F == 0:
        return math.inf if gain > 0 else (0.0 if gain == 0 else -math.inf)
    return float(gain / F)


L1_PAIRS = (("V0r", "V0"), ("V-R8r", "V-R8"))


def clause(name, ok, **kw):
    return {"clause": name, "PASS": (None if ok is None else bool(ok)), **kw}


def verdict_of(regs, bars):
    """B-REG first (a failed or UNDECIDABLE regression arm makes the rung NOT QUOTABLE); then every scored bar
    (reported-only rows carry PASS None by construction and a 'reported' tag); a bar that could not be computed is
    UNDECIDABLE, never a pass."""
    if any(c["PASS"] is None for c in regs):
        return "UNDECIDABLE (a B-REG clause could not be computed)"
    if not all(c["PASS"] for c in regs):
        return "NOT QUOTABLE (B-REG failed)"
    scored = [c for c in bars if not c.get("reported")]
    if any(c["PASS"] is None for c in scored):
        return "UNDECIDABLE (a bar could not be computed)"
    return "CLEARS" if all(c["PASS"] for c in scored) else "FAILS"


def score(w: Path, cons_median=None, cons_rec=None) -> dict:
    ev = w / "eval"
    i0 = w / "I0.json"
    i0_rec = json.loads(i0.read_text(encoding="utf-8")) if i0.is_file() else None
    if not (i0_rec and i0_rec.get("PASS")):
        return {"VERDICT": "STOP: I-0 has not PASSED (SPEC sec. 5) -- nothing is scored", "I0": i0_rec}
    passes = {}
    for f in sorted(ev.glob("*/*_s[01].npz")):
        m = re.match(r"(.+)_s([01])$", f.stem)
        passes[(f.parent.name, m.group(1), int(m.group(2)))] = f
    if not passes:
        return {"VERDICT": "NOTHING TO SCORE"}
    digests = {}
    data = {}
    for key, f in passes.items():
        z = load_pass(f)
        digests[key] = z["_json"]["window_sha12_t_sha256"]
        data[key] = z
    if len(set(digests.values())) != 1:
        raise SystemExit(f"[ladder-score] the passes are NOT on the same windows: {sorted(set(digests.values()))}")
    z0 = next(iter(data.values()))
    eps, inv = episodes(z0["win_sha12"])
    draws = make_draws(len(eps))
    cnt = counts_of(draws, len(eps))
    P = {key: score_pass(z, inv, cnt, cons_median) for key, z in data.items()}
    arms_have = {a for a, _r, _s in P}
    out = {"estimator": "paired episode-cluster bootstrap, B 2000, seed-0 draws shared by every pass; ratio of sums",
           "n_windows": int(len(inv)), "n_episodes": int(len(eps)), "window_digest": next(iter(digests.values())),
           "I0": i0_rec, "cons_train_median": cons_rec, "passes": sorted(f"{a}/{r}/s{s}" for a, r, s in P),
           "tables": {}, "rungs": {}}
    # per-pass tables (every family, never pooled)
    for (a, r, s), res in sorted(P.items()):
        out["tables"][f"{a}/{r}/s{s}"] = {m: {"pt": (None if not np.isfinite(v["pt"]) else round(float(v["pt"]), 5)),
                                              "ci": (ci(v["bs"]) if v.get("bs") is not None else None), "n": v["n"]}
                                          for m, v in res.items()}
    inrun = {a: final_eval_row(read_metrics(w / "arms" / a / "run")) for a in arms_have}
    out["inrun_perception"] = {a: {k: v for k, v in r.items() if k in INRUN or k.startswith("eval_map_hires_iou_")}
                               for a, r in inrun.items()}

    def have(arms, rung):
        need = ROW_NEEDS.get(rung, {})
        miss = []
        for a in arms:
            for s in SEEDS:
                for r in need.get(a, ("base",)):
                    if (a, r, s) not in P:
                        miss.append(f"{a}/{r}/s{s}")
        return miss

    def cmp(t, b, m, s, row_t="base", row_b="base"):
        return paired(P.get((t, row_t, s), {}).get(m), P.get((b, row_b, s), {}).get(m), METRICS[m][1])

    def no_family_worse(t, b, s, pairs):
        bad = []
        for m in METRICS:
            c = cmp(t, b, m, s)
            F = floor(P, m, s, pairs)
            if c is None or F is None:
                continue
            if -c["gain"] > F:
                bad.append({"metric": m, "family": METRICS[m][0], "worse_by": round(-c["gain"], 5), "F": round(F, 5)})
        return bad

    def inrun_drop(t, b, pairs, cap=0.02):
        rows = []
        for k, better in INRUN.items():
            vt, vb = inrun.get(t, {}).get(k), inrun.get(b, {}).get(k)
            fs = [abs(inrun.get(r_, {}).get(k, np.nan) - inrun.get(p_, {}).get(k, np.nan)) for r_, p_ in pairs]
            F = max(fs) if fs and all(np.isfinite(fs)) else None
            if vt is None or vb is None or F is None:
                rows.append({"metric": k, "PASS": None, "why": "UNAVAILABLE"})
                continue
            drop = better * (vb - vt)
            rows.append({"metric": k, "drop": drop, "F": F, "PASS": bool(drop <= F and drop <= cap)})
        return rows

    def lever(t, b, m, pairs, r_min, strict_gt=False):
        per = {}
        for s in SEEDS:
            c = cmp(t, b, m, s)
            F = floor(P, m, s, pairs)
            R = R_of(None if c is None else c["gain"], F)
            okR = None if R is None else (R > r_min if strict_gt else R >= r_min)
            per[s] = {"cmp": c, "F": F, "R": R, "PASS": (None if c is None or okR is None else
                                                         bool(c["sep_better"] and okR))}
        return per, all(v["PASS"] for v in per.values()) if all(v["PASS"] is not None for v in per.values()) else None

    def reg_le_F(t, b, m, pairs):
        per = {}
        for s in SEEDS:
            c = cmp(t, b, m, s)
            F = floor(P, m, s, pairs)
            per[s] = {"gain": None if c is None else c["gain"], "F": F,
                      "PASS": None if c is None or F is None else bool(c["gain"] <= F)}
        oks = [v["PASS"] for v in per.values()]
        return per, (None if None in oks else all(oks))

    R1 = "R>1 (training-variance false-pass rate up to 0.17 at this floor, SPEC sec. 10.1)"
    # ---------------- L1 ----------------
    miss = have(RUNGS["L1"], "L1")
    if miss:
        out["rungs"]["L1"] = {"status": "INCOMPLETE", "missing": miss}
    else:
        cl = []
        per, ok = reg_le_F("V-R8d", "V0", "dir_correct_turn", L1_PAIRS)
        cl.append(clause("B-REG: V-R8d dir gain over V0 <= F", ok, per=per))
        per, ok = lever("V-R8", "V-R8d", "dir_correct_turn", L1_PAIRS, 1.0, strict_gt=True)
        cl.append(clause("B-REG: V-R8 - V-R8d dir separated, R>1", ok, per=per))
        reg_ok = all(c["PASS"] for c in cl) if all(c["PASS"] is not None for c in cl) else False
        rec = []
        for m in ("dir_correct_turn", "head15_all"):
            per, ok = lever("V-R8", "V0", m, L1_PAIRS, 1.0, strict_gt=True)
            rec.append(clause(f"B-RECIPE: {m} gain separated, {R1}", ok, per=per))
        per = {}
        for s in SEEDS:
            c = cmp("V-R8", "V0", "ade_straight", s)
            per[s] = {"cmp": c, "PASS": None if c is None else bool(c["delta"] <= 0.05 and c["ci"][1] is not None
                                                                     and c["ci"][1] <= 0.10)}
        rec.append(clause("B-RECIPE: GT-straight dADE <= +0.05 m, CI hi <= +0.10", all(v["PASS"] for v in per.values())
                          if all(v["PASS"] is not None for v in per.values()) else None, per=per))
        bad = {s: no_family_worse("V-R8", "V0", s, L1_PAIRS) for s in SEEDS}
        rec.append(clause("B-RECIPE: no family worse beyond F", not any(bad.values()), worse=bad))
        per = {}
        for s in SEEDS:
            c = cmp("V-R8", "V0", "ade_all", s, row_t="legal")
            per[s] = {"cmp": c, "PASS": None if c is None else bool(c["delta"] <= 0.05)}
        rec.append(clause("B-RECIPE: LEGAL row of V-R8 within +0.05 m ADE of V0", all(v["PASS"] for v in per.values())
                          if all(v["PASS"] is not None for v in per.values()) else None, per=per))
        rc_rows = {s: {r: {m: cmp("V-R8", "V-R8", m, s, row_t=r) for m in ("dir_correct_turn", "head15_all", "ade_all")}
                       for r in ("rc_off", "legal", "rc_shuf")} for s in SEEDS}
        out["rungs"]["L1"] = {"status": "SCORED", "B-REG": cl, "quotable": reg_ok, "B-RECIPE": rec,
                              "VERDICT": verdict_of(cl, rec),
                              "rc_rows_vs_base": rc_rows}
    # ---------------- L2 / L3 ----------------
    for rung, t, roll, rep, prem_key, prem_ok, metrics_ in (
            ("L2", "V-TACk", "V-TACk-roll", "V-TACkr", "gs_trunk_proj_tac_v6", lambda v: v >= 0.01,
             ("dir_correct_turn", "head15_all")),
            ("L3", "V-MAP4", "V-MAP4-roll", "V-MAP4r", "gs_bev025_proj_map_hires", lambda v: v > 0,
             ("map_iou_drivable", "map_iou_lane"))):
        miss = have(RUNGS[rung], rung)
        if miss:
            out["rungs"][rung] = {"status": "INCOMPLETE", "missing": miss}
            continue
        pairs = L1_PAIRS + ((rep, t),)
        prem = gs_median(read_metrics(w / "arms" / t / "run"), prem_key, 0.5, 1.0)
        prem_pass = None if prem is None else bool(prem_ok(prem["median"]))
        regs = []
        for m in metrics_:
            per, ok = reg_le_F(roll, "V-R8", m, pairs)
            regs.append(clause(f"B-REG: {roll} gain over V-R8 on {m} <= F", ok, per=per))
        reg_ok = all(c["PASS"] for c in regs)
        bars = [clause(f"B-PREM: median {prem_key} (last 50 %)", prem_pass, reading=prem)]
        for m in metrics_:
            per, ok = lever(t, "V-R8", m, pairs, 2.0)
            bars.append(clause(f"{m} gain separated, R>=2 (sec. 10.1)", ok, per=per))
        if rung == "L2":
            dr = inrun_drop(t, "V-R8", pairs)
            mp, map_ok = [], True
            for m in ("map_iou_drivable", "map_iou_lane"):
                for s in SEEDS:
                    c, F = cmp(t, "V-R8", m, s), floor(P, m, s, pairs)
                    ok = None if c is None or F is None else bool(-c["gain"] <= F and -c["gain"] <= 0.02)
                    mp.append({"metric": m, "seed": s, "cmp": c, "F": F, "PASS": ok})
                    map_ok = None if (map_ok is None or ok is None) else (map_ok and ok)
            bars.append(clause("box / agent AP drop <= F and <= 0.02", all(r_["PASS"] for r_ in dr)
                               if all(r_["PASS"] is not None for r_ in dr) else None, rows=dr))
            bars.append(clause("headline map IoU drop <= F and <= 0.02", map_ok, rows=mp))
        else:
            bad = {s: [b_ for b_ in no_family_worse(t, "V-R8", s, pairs) if b_["family"] in ("LATERAL", "PLAN")]
                   for s in SEEDS}
            dr = inrun_drop(t, "V-R8", pairs, cap=math.inf)
            bars.append(clause("box / agent AP no worse than F", all(r_["PASS"] for r_ in dr)
                               if all(r_["PASS"] is not None for r_ in dr) else None, rows=dr))
            bars.append(clause("route criteria no worse than F", not any(bad.values()), worse=bad))
        verdict = "PREMISE-UNMET" if (reg_ok and prem_pass is False) else verdict_of(regs, bars)
        out["rungs"][rung] = {"status": "SCORED", "B-REG": regs, "bars": bars, "VERDICT": verdict}
    # ---------------- L4 (B-X3) ----------------
    miss = have(RUNGS["L4"], "L4")
    if miss:
        out["rungs"]["L4"] = {"status": "INCOMPLETE", "missing": miss}
    else:
        bars = []
        per, ok = lever("V-R8", "V-VSHUF", "speed_mae_2_6s", L1_PAIRS, 1.0, strict_gt=True)
        bars.append(clause(f"speed MAE 2-6 s of V-R8 better than V-VSHUF, separated, {R1}", ok, per=per))
        per = {s: cmp("V-R8", "V-R8", "speed_mae_2_6s", s, row_t="vmax_shuf", row_b="vmax_off") for s in SEEDS}
        bars.append(clause("VMAX-SHUF - VMAX-OFF NOT separated better", not any(c and c["sep_better"]
                                                                                 for c in per.values()), per=per))
        per = {}
        for s in SEEDS:
            c = cmp("V-R8", "V-R8", "speed_mae_2_6s", s, row_t="vmax_off", row_b="base")
            per[s] = {"cmp": c, "PASS": None if c is None else bool(c["delta"] <= 0.10 and c["ci"][1] is not None
                                                                     and c["ci"][1] <= 0.20)}
        bars.append(clause("UNKNOWN - known <= +0.10 m/s (CI hi <= +0.20)", all(v["PASS"] for v in per.values())
                           if all(v["PASS"] is not None for v in per.values()) else None, per=per))
        out["rungs"]["L4"] = {"status": "SCORED", "bars": bars, "LEAK": "read by the D4 census (raw/x3_leak.json), not "
                              "by an arm", "VERDICT": verdict_of([], bars)}
    # ---------------- A1: L4b / L5 ----------------
    for rung, t, roll, m_main, guards in (("L4b", "V-R8-E8", "V-R8-E8-roll", "speed_mae_2_6s", None),
                                          ("L5", "V-R8-DRV", "V-R8-DRV-roll", "off_drivable",
                                           ("dir_correct_turn", "head15_all"))):
        miss = have(RUNGS[rung], rung)
        if miss:
            out["rungs"][rung] = {"status": "INCOMPLETE", "missing": miss}
            continue
        per, ok = reg_le_F(roll, "V-R8", m_main, L1_PAIRS)
        regs = [clause(f"B-REG: {roll} {m_main} gain over V-R8 <= F", ok, per=per)]
        bars = []
        per, ok = lever(t, "V-R8", m_main, L1_PAIRS, 2.0)
        bars.append(clause(f"{m_main} gain separated, R>=2", ok, per=per))
        if guards is None:
            bad = {s: no_family_worse(t, "V-R8", s, L1_PAIRS) for s in SEEDS}
            bars.append(clause("no family worse beyond F", not any(bad.values()), worse=bad))
            per = {s: cmp(t, "V-R8", "ceil_violation", s) for s in SEEDS}
            bars.append(clause("ceiling compliance (reported, A1 measure)", None, per=per, reported=True))
        else:
            for g in guards:
                bad = {}
                for s in SEEDS:
                    c, F = cmp(t, "V-R8", g, s), floor(P, g, s, L1_PAIRS)
                    bad[s] = {"cmp": c, "F": F, "PASS": None if c is None or F is None else bool(-c["gain"] <= F)}
                oks = [v["PASS"] for v in bad.values()]
                bars.append(clause(f"{g} no worse than F", None if None in oks else all(oks), per=bad))
        out["rungs"][rung] = {"status": "SCORED", "B-REG": regs, "bars": bars, "VERDICT": verdict_of(regs, bars)}
    return out


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items() if k != "bs"}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("size-k", "score"))
    ap.add_argument("--w", required=True)
    ap.add_argument("--v9-train", default=None)
    ap.add_argument("--v9-train-md5", default=None)
    a = ap.parse_args()
    w = Path(a.w)
    if a.cmd == "size-k":
        return cmd_size_k(w)
    cm, crec = (None, {"status": "UNAVAILABLE: no --v9-train given; constraint MAE reported without its constant"})
    if a.v9_train:
        cm, crec = cons_train_median(a.v9_train, a.v9_train_md5)
    res = _json_safe(score(w, cm, crec))
    txt = json.dumps(res, indent=1)
    (w / "LADDER_SCORE.json").write_text(txt, encoding="utf-8")
    print("LADDER_SCORE sha256", hashlib.sha256(txt.encode()).hexdigest()[:16])
    for r, v in (res.get("rungs") or {}).items():
        print(r, v.get("status"), v.get("VERDICT", ""), v.get("missing", "")[:4] if v.get("missing") else "")
    print(res.get("VERDICT", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
