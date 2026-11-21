#!/usr/bin/env python3
"""Amendment A2 reader (2026-10-16): does the head under-fit, or do the frozen visual features fall short?  Maps the 10,000-update V-s0 units to exactly ONE of A2's rows U1 / U2 / U3 / U4.

CONTRACT: raw/2026-10-16-scene-tap-a2/A2_CONTRACT_from_73db963.md (section "### A2" of PREREG_SCENE_TAP_PROBE.md as committed at 73db963).  Nothing here changes a threshold, the row order or the arm.

    H = U10k pooled held-out IoU [0, 40)                       (5 log-disjoint folds, each token predicted once)
    I = U10k in-sample IoU [0, 40)                             (each fold model on its own training tokens; the MEAN OVER THE 5 FOLDS of the per-fold ratio -- the banked probe's `insample_iou_0_40_mean`)
    D = H(U10k) - H(V seed 0, 2,000 updates, the banked units)  paired on the same tokens AND the same log resamples; "separated" = the probe's OWN rule (`scene_tap_probe.contrast`:
        the paired percentile CI excludes 0 AND |D| exceeds V's seed floor) -- the floor is read from the banked PROBE_RESULT.json (0.0035525..., "0.0036" in A2's text).

    U1  H >= 0.80
    U2  H <  0.80, I >= 0.80, D separated and > 0
    U3  H <  0.80, I <  0.80
    U4  H <  0.80, I >= 0.80, D not separated or <= 0           (applied in this order; thresholds on UNROUNDED values)

The bootstrap is the probe's own, IMPORTED (never reimplemented): paired log-cluster, `aux_s1_readout.boot_weights(n_logs, 10000, 20260927)`, `scene_tap_probe.pooled_p1` (the S1 readout's
`frame_stats` + `perception_reads`), `aux_s1_readout.by_log`, `aux_s1_readout._ci`, `scene_tap_probe.contrast`.  The only glue is `(W @ inter) / (W @ union)`, the line `assemble` uses, and the
IDENTITY TEST below proves the glue: running this reader on the BANKED 2,000-update V-s0 units must reproduce PROBE_RESULT.json's V seed-0 held-out IoU, its CI, its per-fold IoUs, its in-sample
mean and its SDF MAEs EXACTLY; and the ZERO CONTROL (V-s0(2k) against itself) must read a point of exactly 0.0 with CI [0, 0] and "not separated".  Either failing = INSTRUMENT FAILED, no row.

    python a2_read.py --out <dir> [--work-a2 D:/.../scene_tap_probe_a2] [--identity-only]

Reported, never gating: per-fold IoU, near-boundary and overall SDF MAE, the train-loss curve's last 10 % (slope, plateau), the wall time.  TIER: T0-like, never driving performance.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                    # the A2 COPY of scene_tap_probe (with the filter), not the banked one
import numpy as np  # noqa: E402
import scene_tap_probe as S  # noqa: E402

ASR = S.ASR
TOOL = "a2_read/1"
BANKED_DIR = os.path.join(os.path.dirname(HERE), "2026-10-15-scene-tap-probe")
BANKED_RESULT = os.path.join(BANKED_DIR, "PROBE_RESULT.json")
BANKED_WORK = S.ART_OUT                                     # D:/Projects/TanitAD-artifacts/scene_tap_probe   (the 2,000-update units, read-only here)
A2_WORK = "D:/Projects/TanitAD-artifacts/scene_tap_probe_a2"
BANKED_UPDATES, A2_UPDATES = 2000, 10000
BAR = 0.80                                                  # A2's H / I bar (= K2's)
FLOOR_TEXT = 0.0036                                         # the number A2's text quotes; the value USED is read from PROBE_RESULT.json

ROWS_A2 = {
    "U1": ("the 2,000-update budget was the limit; frozen visual features carry the road", "S1c: S1's visual head trained >= 5x longer (cost it on the dev box first, then the A40)"),
    "U2": ("longer training helps and the head can fit the features; generalisation is the limit", "S1c with more updates AND more label logs, each as its own registered pair"),
    "U3": ("the head cannot fit frozen visual features even at 5x updates => the frozen trunk is the limit", "the trunk: the graft (PI decision) and P6's trained trunk"),
    "U4": ("it memorises without generalising", "more label logs is the only frozen-trunk lever; the trunk otherwise"),
}


class InstrumentFailed(S.GuardFailed):
    """The identity test / zero control / a consistency check failed: the reader is not trustworthy, so NO row is produced."""


# =====================================================================================================================================================================
# the committed outcome table, mechanically
# =====================================================================================================================================================================
def decide_a2(h: float, i: float, d: dict) -> dict:
    """A2's table in A2's order, on unrounded values.  `d` is `scene_tap_probe.contrast`'s dict for D = H(U10k) - H(V s0 2k)."""
    sep_pos = bool(d["separated_beyond_floor"] and d["point"] > 0)
    if h >= BAR:
        row = "U1"
    elif h < BAR and i >= BAR and sep_pos:
        row = "U2"
    elif h < BAR and i < BAR:
        row = "U3"
    else:                                                    # h < BAR, i >= BAR, D not separated-positive: the only case left (the table is exhaustive)
        row = "U4"
    return {"row": row, "reading": ROWS_A2[row][0], "next_lever": ROWS_A2[row][1], "H": h, "I": i, "bar": BAR, "delta_point": d["point"], "delta_ci95": d["ci95"],
            "delta_separated_beyond_floor": bool(d["separated_beyond_floor"]), "delta_separated_positive": sep_pos}


def decide_a2_floor_only(h: float, i: float, d: dict) -> str:
    """The same table, with "separated" read as ONLY 'beyond V's seed floor' (A2's wording) instead of the probe's rule (CI excludes 0 AND beyond the floor).  Diagnostic: if the two differ,
    the row depends on the CI clause and the report must say so."""
    sep_pos = bool(d["abs_point_exceeds_all_floors"] and d["point"] > 0)
    if h >= BAR:
        return "U1"
    if i >= BAR and sep_pos:
        return "U2"
    if i < BAR:
        return "U3"
    return "U4"


# =====================================================================================================================================================================
# reading one (arm, seed) from its 5 unit files
# =====================================================================================================================================================================
def loss_curve_stats(curve: np.ndarray) -> dict:
    """The train-loss curve of one unit ([n, 2] = (bce, huber) per update).  Reported, never gating: the last-10 % OLS slope of bce per 1,000 updates (with its standard error), the means of the
    last 10 % vs the 10 % before it, and ten decile means of the whole curve.  The lr is cosine -> 0, so a flattening tail is expected; this says how much."""
    c = np.asarray(curve, np.float64)
    n = len(c)
    if n < 20:
        return {"n_updates": n, "note": "curve too short"}
    k = max(n // 10, 2)
    y = c[-k:, 0]
    x = np.arange(k, dtype=np.float64)
    xm = x - x.mean()
    slope = float((xm * (y - y.mean())).sum() / (xm ** 2).sum())
    resid = y - (y.mean() + slope * xm)
    se = float(math.sqrt((resid ** 2).sum() / max(k - 2, 1) / (xm ** 2).sum()))
    prev = c[-2 * k:-k, 0] if n >= 2 * k else None
    return {"n_updates": n, "last10pct_updates": k, "bce_last10pct_mean": float(y.mean()), "bce_prev10pct_mean": float(prev.mean()) if prev is not None else None,
            "bce_last10pct_slope_per_1000_updates": slope * 1000.0, "bce_last10pct_slope_se_per_1000": se * 1000.0,
            "bce_last10pct_change_vs_prev10pct": float(y.mean() - prev.mean()) if prev is not None else None,
            "bce_decile_means": [float(c[j * n // 10:(j + 1) * n // 10, 0].mean()) for j in range(10)]}


def arm_read(work: str, arm: str, seed: int, n_updates: int, U: "S.Universe", source, W: np.ndarray, logs_sorted: list) -> dict:
    """Pool one (arm, seed)'s 5 fold units (each token exactly once, config sha checked), read P1 / P2 through the readout, and form the per-resample IoU [0, 40) replicates."""
    cfg_sha = S.config_sha(n_updates)
    lg = np.full((U.n, S.ASR.NCX, S.ASR.NCY), np.nan, np.float32)
    sd = np.full_like(lg, np.nan)
    cover = np.zeros(U.n, np.int64)
    ins_f, infos, curves = [], [], []
    for f in range(S.K_FOLDS):
        p = S.unit_path(work, arm, seed, f)
        if not os.path.exists(p):
            raise S.GuardFailed(f"unit file missing: {p}")
        held, a, b, meta = S.load_unit(p)
        if meta["config_sha256"] != cfg_sha:
            raise S.GuardFailed(f"unit {arm} s{seed} f{f}: config_sha256 {meta['config_sha256'][:12]} != {cfg_sha[:12]} (n_updates {n_updates}); refusing to mix")
        if meta["info"]["updates"] != n_updates or meta["arm"] != arm or meta["seed"] != seed or meta["fold"] != f:
            raise S.GuardFailed(f"unit {p}: meta says {meta['arm']} s{meta['seed']} f{meta['fold']} updates {meta['info']['updates']}")
        if sorted(held.tolist()) != sorted(np.flatnonzero(U.fold == f).tolist()):
            raise S.GuardFailed(f"unit {arm} s{seed} f{f}: held-out indices are not fold {f}")
        lg[held], sd[held] = a, b
        cover[held] += 1
        ins_f.append(meta["insample"])
        infos.append(meta["info"])
        z = np.load(p, allow_pickle=False)
        curves.append(np.asarray(z["loss_curve"]) if "loss_curve" in z.files else None)
    if not (cover == 1).all():
        raise S.GuardFailed(f"{arm} s{seed}: tokens predicted {sorted(set(cover.tolist()))} times (every token exactly once)")
    res, st = S.pooled_p1(U.keys, lg, sd, source)
    if not st["valid"].all():
        raise S.GuardFailed("an unlabelled frame in the pooled read")
    ul, (inter, union) = ASR.by_log(U.logs, st[f"inter{S.K2_BAND_INDEX}"], st[f"union{S.K2_BAND_INDEX}"])
    if ul != logs_sorted:
        raise S.GuardFailed("log order differs from the bootstrap's")
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = (W @ inter) / (W @ union)                      # the very line `assemble` uses
    k2 = S.K2_BAND_INDEX
    per_fold_iou, per_fold_mae_all, per_fold_mae_near = {}, {}, {}
    for f in range(S.K_FOLDS):
        m = U.fold == f
        per_fold_iou[str(f)] = float(st[f"inter{k2}"][m].sum() / max(st[f"union{k2}"][m].sum(), 1.0))
        per_fold_mae_all[str(f)] = float(st["all_abs"][m].sum() / max(st["all_n"][m].sum(), 1.0))
        per_fold_mae_near[str(f)] = float(st["p2_abs"][m].sum() / max(st["p2_n"][m].sum(), 1.0))
    ins_iou = [i["inter_0-40"] / max(i["union_0-40"], 1.0) for i in ins_f]
    est = res["p1"]["all"]["0-40"]["est"]
    ci_probe = res["p1"]["all"]["0-40"]["ci95"]
    ci_reps = ASR._ci(reps)[0]
    if not (np.allclose(ci_reps, ci_probe, rtol=1e-12, atol=1e-15) and abs(float(inter.sum() / union.sum()) - est) <= 1e-15):
        raise InstrumentFailed(f"{arm} s{seed}: the replicate glue disagrees with the readout's own P1 (CI {ci_reps} vs {ci_probe}, point {inter.sum() / union.sum()} vs {est})")
    return {"arm": arm, "seed": seed, "n_updates": n_updates, "config_sha256": cfg_sha, "heldout_iou_0_40": est, "heldout_iou_0_40_ci95_logs_only": ci_probe,
            "heldout_iou_0_40_ci95_from_reps": ci_reps, "per_fold_heldout_iou_0_40": per_fold_iou,
            "insample_iou_0_40_per_fold": ins_iou, "insample_iou_0_40_mean": float(np.mean(ins_iou)),
            "insample_iou_0_40_pooled": float(sum(i["inter_0-40"] for i in ins_f) / max(sum(i["union_0-40"] for i in ins_f), 1.0)),
            "insample_sdf_mae_m_mean": float(np.mean([i["all_abs"] / max(i["all_n"], 1.0) for i in ins_f])),
            "heldout_p2": res["p2"], "heldout_overall_sdf_mae_m": res["p2"]["overall_mae_m"]["est"], "heldout_near_boundary_sdf_mae_m": res["p2"]["near_boundary_mae_m"]["est"],
            "per_fold_heldout_overall_sdf_mae_m": per_fold_mae_all, "per_fold_heldout_near_boundary_sdf_mae_m": per_fold_mae_near,
            "train": [{"fold": f, "wall_s": infos[f]["wall_s"], "loss_bce_last100": infos[f]["loss_bce_last100"], "loss_bce_mean_all": infos[f]["loss_bce_mean_all"],
                       "grad_norm_every50_mean": infos[f]["grad_norm_every50_mean"], "distinct_train_tokens_drawn": infos[f]["distinct_train_tokens_drawn"],
                       "curve": (loss_curve_stats(curves[f]) if curves[f] is not None else None)} for f in range(S.K_FOLDS)],
            "wall_s_total_train": float(sum(i["wall_s"] for i in infos)), "_reps": reps}


# =====================================================================================================================================================================
# the instrument checks
# =====================================================================================================================================================================
def read_banked(path: str) -> dict:
    r = json.load(open(path, encoding="utf-8"))
    v = r["arms"]["V"]
    return {"result": r, "floor": float(v["seed_floor"]), "v_s0": v["seeds"]["0"], "n_updates": int(r["n_updates"]), "config_sha256": r["config_sha256"]}


def identity_test(banked: dict, v2k: dict) -> dict:
    """Running the H computation on the BANKED 2,000-update V-s0 units must reproduce PROBE_RESULT.json's V seed-0 numbers EXACTLY (==, not allclose)."""
    b = banked["v_s0"]
    checks = {
        "heldout_iou_0_40": (v2k["heldout_iou_0_40"], b["heldout_iou_0_40"]),
        "heldout_iou_0_40_ci95_logs_only": (list(v2k["heldout_iou_0_40_ci95_logs_only"]), list(b["heldout_iou_0_40_ci95_logs_only"])),
        "per_fold_heldout_iou_0_40": (v2k["per_fold_heldout_iou_0_40"], b["per_fold_heldout_iou_0_40"]),
        "insample_iou_0_40_per_fold": (v2k["insample_iou_0_40_per_fold"], b["insample_iou_0_40_per_fold"]),
        "insample_iou_0_40_mean": (v2k["insample_iou_0_40_mean"], b["insample_iou_0_40_mean"]),
        "insample_sdf_mae_m_mean": (v2k["insample_sdf_mae_m_mean"], b["insample_sdf_mae_m_mean"]),
        "heldout_overall_sdf_mae_m": (v2k["heldout_overall_sdf_mae_m"], b["heldout_p2"]["overall_mae_m"]["est"]),
        "heldout_near_boundary_sdf_mae_m": (v2k["heldout_near_boundary_sdf_mae_m"], b["heldout_p2"]["near_boundary_mae_m"]["est"]),
        "heldout_overall_sdf_mae_ci95": (list(v2k["heldout_p2"]["overall_mae_m"]["ci95"]), list(b["heldout_p2"]["overall_mae_m"]["ci95"])),
    }
    out = {k: {"recomputed": a, "banked": bb, "exact": bool(a == bb)} for k, (a, bb) in checks.items()}
    ok = all(v["exact"] for v in out.values())
    return {"ok": ok, "checks": out}


def zero_control(reps: np.ndarray, point: float, floor: float) -> dict:
    """D of V-s0(2k) against itself: the point is exactly 0.0, the CI exactly [0, 0], and it is not separated."""
    c = S.contrast(reps, reps, point, point, {"V_seed_floor": floor})
    ok = bool(c["point"] == 0.0 and list(c["ci95"]) == [0.0, 0.0] and not c["separated_beyond_floor"] and not c["ci_excludes_zero"])
    return {"ok": ok, "contrast": c}


def check_floor(floor: float) -> None:
    """A2's text quotes V's seed floor as 0.0036; the value used is the file's.  They must agree to A2's four decimals, or the contract and the file are not about the same number."""
    if round(floor, 4) != FLOOR_TEXT:
        raise InstrumentFailed(f"V's seed floor in PROBE_RESULT.json is {floor!r}, A2 quotes {FLOOR_TEXT}")


# =====================================================================================================================================================================
# main
# =====================================================================================================================================================================
def run(out: str, work_a2: str, work_banked: str, banked_result: str, n_a2: int, n_banked: int, identity_only: bool = False, U=None, source=None, fh=None) -> dict:
    t0 = time.time()
    if not (S.N_BOOT == 10000 and S.SEED_BOOT == 20260927 and ASR.N_BOOT == 10000 and ASR.SEED == 20260927):
        raise InstrumentFailed("the bootstrap constants are not A2's (10,000 resamples, seed 20260927)")
    S.guard_p1_function()
    banked = read_banked(banked_result)
    if banked["n_updates"] != n_banked or banked["config_sha256"] != S.config_sha(n_banked):
        raise InstrumentFailed(f"the banked result's config (n_updates {banked['n_updates']}, sha {banked['config_sha256'][:12]}) is not this code's config at {n_banked} updates")
    check_floor(banked["floor"])
    if U is None:
        U = S.load_universe()
    if source is None:
        _lab, source = S.load_true_labels(U)
    logs_sorted = sorted(set(U.logs.tolist()))
    W = ASR.boot_weights(len(logs_sorted), S.N_BOOT, S.SEED_BOOT)
    S.log("reading the banked V-s0 (2,000 updates) units", fh)
    v2k = arm_read(work_banked, "V", 0, n_banked, U, source, W, logs_sorted)
    ident = identity_test(banked, v2k)
    zero = zero_control(v2k["_reps"], v2k["heldout_iou_0_40"], banked["floor"])
    S.log(f"IDENTITY {'OK' if ident['ok'] else 'FAILED'}: V-s0(2k) held-out IoU {v2k['heldout_iou_0_40']!r} vs banked {banked['v_s0']['heldout_iou_0_40']!r}; "
          f"ZERO CONTROL {'OK' if zero['ok'] else 'FAILED'}: point {zero['contrast']['point']!r} CI {zero['contrast']['ci95']}", fh)
    instrument = {"identity": ident, "zero_control": zero, "banked_result": banked_result, "banked_result_config_sha256": banked["config_sha256"],
                  "v_seed_floor_used": banked["floor"], "v_seed_floor_in_A2_text": FLOOR_TEXT}
    if not ident["ok"]:
        raise InstrumentFailed("IDENTITY TEST FAILED: " + json.dumps({k: v for k, v in ident["checks"].items() if not v["exact"]}, default=str)[:1500])
    if not zero["ok"]:
        raise InstrumentFailed("ZERO CONTROL FAILED: " + json.dumps(zero["contrast"], default=str)[:600])
    result = {"tool": TOOL, "tier": S.TIER, "contract": "A2_CONTRACT_from_73db963.md", "estimator": "paired log-cluster bootstrap over the %d navtest logs, %d resamples, seed %d "
              "(aux_s1_readout.boot_weights): per resample the pooled held-out IoU [0,40) of each arm, then D = U10k - V(2k). Answers 'another draw of LOGS' with the 5 fold models of EACH arm held "
              "fixed; it is blind to training variance (H-ESTIM-SEED-1), which is why 'separated' ALSO requires |D| beyond V's seed floor." % (len(logs_sorted), S.N_BOOT, S.SEED_BOOT),
              "n_tokens": U.n, "n_logs": len(logs_sorted), "instrument": instrument, "v_2k": {k: v for k, v in v2k.items() if k != "_reps"}}
    if identity_only:
        result["mode"] = "identity-only"
        result["wall_s"] = round(time.time() - t0, 1)
        return result
    S.log("reading the U10k (10,000 updates) units", fh)
    u10 = arm_read(work_a2, "V", 0, n_a2, U, source, W, logs_sorted)
    d = S.contrast(u10["_reps"], v2k["_reps"], u10["heldout_iou_0_40"], v2k["heldout_iou_0_40"], {"V_seed_floor": banked["floor"]})
    H, I = u10["heldout_iou_0_40"], u10["insample_iou_0_40_mean"]
    dec = decide_a2(H, I, d)
    alt = decide_a2_floor_only(H, I, d)
    result.update({"mode": "full", "u10k": {k: v for k, v in u10.items() if k != "_reps"}, "delta_u10k_minus_v2k": d, "decision": dec,
                   "row_if_separated_meant_floor_only": alt, "row_depends_on_the_ci_clause": bool(alt != dec["row"]),
                   "reported_never_gating": {"H_per_fold": u10["per_fold_heldout_iou_0_40"], "I_per_fold": u10["insample_iou_0_40_per_fold"],
                                             "near_boundary_sdf_mae_m": u10["heldout_near_boundary_sdf_mae_m"], "overall_sdf_mae_m": u10["heldout_overall_sdf_mae_m"],
                                             "v_2k_near_boundary_sdf_mae_m": v2k["heldout_near_boundary_sdf_mae_m"], "v_2k_overall_sdf_mae_m": v2k["heldout_overall_sdf_mae_m"],
                                             "train_wall_s_total": u10["wall_s_total_train"], "train_loss_curve": [t["curve"] for t in u10["train"]],
                                             "bce_last100": [t["loss_bce_last100"] for t in u10["train"]]},
                   "wall_s_read": round(time.time() - t0, 1)})
    return result


def summary_md(r: dict) -> str:
    if r.get("mode") != "full":
        return "# A2 identity check only: identity test %s, zero control %s" % ("EXACT" if r["instrument"]["identity"]["ok"] else "FAILED",
                                                                              "EXACT" if r["instrument"]["zero_control"]["ok"] else "FAILED") + chr(10)
    d, dec, u, v = r["delta_u10k_minus_v2k"], r["decision"], r["u10k"], r["v_2k"]
    cur = r["reported_never_gating"]["train_loss_curve"]
    sl = [c["bce_last10pct_slope_per_1000_updates"] for c in cur if c and "bce_last10pct_slope_per_1000_updates" in c]
    L = [f"# A2 RESULT: {dec['row']}", "", f"**{dec['reading']}**  ->  next lever: {dec['next_lever']}", "",
         "| quantity | value |", "|---|---|",
         f"| H = U10k held-out IoU [0,40) | {u['heldout_iou_0_40']:.6f}  (95% logs-only CI {u['heldout_iou_0_40_ci95_logs_only'][0]:.4f}..{u['heldout_iou_0_40_ci95_logs_only'][1]:.4f}) |",
         f"| I = U10k in-sample IoU [0,40) (mean of 5 folds) | {u['insample_iou_0_40_mean']:.6f}  (pooled {u['insample_iou_0_40_pooled']:.6f}) |",
         f"| H(V-s0, 2,000 updates), banked | {v['heldout_iou_0_40']:.6f}  (I {v['insample_iou_0_40_mean']:.6f}) |",
         f"| D = H(U10k) - H(V 2k), paired | {d['point']:+.6f}  (95% CI {d['ci95'][0]:+.4f}..{d['ci95'][1]:+.4f}); CI excludes 0: {d['ci_excludes_zero']}; beyond floor {d['floors_used']}: "
         f"{d['abs_point_exceeds_all_floors']}; SEPARATED (probe rule): {d['separated_beyond_floor']} |",
         f"| row if 'separated' meant floor-only | {r['row_if_separated_meant_floor_only']}  (row depends on the CI clause: {r['row_depends_on_the_ci_clause']}) |",
         f"| per-fold held-out IoU | {', '.join('%.4f' % x for x in u['per_fold_heldout_iou_0_40'].values())} |",
         f"| per-fold in-sample IoU | {', '.join('%.4f' % x for x in u['insample_iou_0_40_per_fold'])} |",
         f"| SDF MAE held-out overall / near-boundary (m) | {u['heldout_overall_sdf_mae_m']:.4f} / {u['heldout_near_boundary_sdf_mae_m']:.4f}  (2k: {v['heldout_overall_sdf_mae_m']:.4f} / {v['heldout_near_boundary_sdf_mae_m']:.4f}) |",
         f"| last-10% train-bce slope per 1,000 updates (5 folds) | {', '.join('%+.4f' % x for x in sl)} |",
         f"| train wall time (5 units) | {u['wall_s_total_train']:.0f} s |", "",
         f"Instrument: identity test {'EXACT' if r['instrument']['identity']['ok'] else 'FAILED'} (V-s0 2k held-out IoU {v['heldout_iou_0_40']!r} == banked); zero control "
         f"{'exact 0.0, CI [0,0]' if r['instrument']['zero_control']['ok'] else 'FAILED'}. V seed floor used {r['instrument']['v_seed_floor_used']!r}.",
         "Tier T0-like (teacher-forced head perception), never driving performance. Estimator: see A2_RESULT.json `estimator`."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-a2", default=A2_WORK)
    ap.add_argument("--work-banked", default=BANKED_WORK)
    ap.add_argument("--banked-result", default=BANKED_RESULT)
    ap.add_argument("--updates-a2", type=int, default=A2_UPDATES)
    ap.add_argument("--updates-banked", type=int, default=BANKED_UPDATES)
    ap.add_argument("--identity-only", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    fh = open(os.path.join(a.out, "a2_read.log"), "a", encoding="utf-8")
    try:
        r = run(a.out, a.work_a2, a.work_banked, a.banked_result, a.updates_a2, a.updates_banked, identity_only=a.identity_only, fh=fh)
        name = "A2_IDENTITY.json" if a.identity_only else "A2_RESULT.json"
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(S.dump_json_bytes(r))
        if not a.identity_only:
            with open(os.path.join(a.out, "A2_RESULT.md"), "w", encoding="utf-8", newline="\n") as f:
                f.write(summary_md(r))
            S.log(f"A2 READ OK: {r['decision']['row']}  H {r['decision']['H']!r}  I {r['decision']['I']!r}  D {r['decision']['delta_point']!r}", fh)
        else:
            S.log("A2 IDENTITY OK (identity test exact, zero control exact)", fh)
        return 0
    except BaseException as e:                                                    # noqa: BLE001
        S.log(f"A2_READ_FAILED {type(e).__name__}: {e}", fh)
        with open(os.path.join(a.out, "a2_read.error"), "w", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {type(e).__name__}: {e}\n\n{traceback.format_exc()}\n")
        return 3 if isinstance(e, InstrumentFailed) else 1


if __name__ == "__main__":
    sys.exit(main())
