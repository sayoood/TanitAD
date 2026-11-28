"""s6_lon_decomp.py -- WHERE does refcv8's S6 (0-6 s) longitudinal regression against refcv7-50,400 live, and which lever is cheapest?
CPU only, banked dumps only, no GPU, no retraining.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
      python -X utf8 s6_lon_decomp.py            # writes <PKG>/raw/s6_lon_decomp/S6_LON_DECOMP.{json,md}

WHAT.  Per S6 window (3,668 windows / 139 episodes, instants 1..6 s, dt 1.0) the per-window error of refcv8 minus the per-window
error of refcv7-50,400 (`b_refcv7_50400_s{seed}`, the banked arm embedded in every panel) is decomposed over PRE-REGISTERED window
partitions.  Per bin: n windows / n episodes, the bin's CONTRIBUTION to the total mean delta (sum of the bin's deltas / N, in the
metric's unit), its SHARE of the total (contribution / total), its enrichment (share / window fraction), the bin-mean delta, and the
signed longitudinal bias (too slow / too fast) of both arms.  Error metrics are recomputed by `pair_read_r8.components` -- the SAME
function that produced the banked PAIRED_READ -- so nothing is re-derived; the signed quantities are numpy and are cross-checked.

PRE-REGISTERED PARTITIONS (fixed here before any decomposed number was looked at):
  v0bin      GT measured speed at t0 (`v0`): <2 | 2-8 | 8-15 | 15-25 | >=25 m/s
  dvclass    GT speed change over the 6 s (v6 = `decisions/gt_future_ext[:, 59, 3]`, dv = v6 - v0), first matching rule wins:
             start_from_stop (v0 < 1 and v6 >= 2) | stopping (v0 >= 2 and v6 < 1) | slow_both (v0 < 2 and v6 < 2) |
             accelerating (dv >= +1.5) | decelerating (dv <= -1.5) | cruising (|dv| < 1.5)
  v0_x_dv    the cross of the two
  vmax_label the road-type max-speed label `v_max_ms` (4 classes: 30 / 50 / 100 / 120 km/h); ego-future derived (privileged)
  label_vs_gt r = gt_max_speed_6s / v_max_ms: GT_above_label (r > 1.0) | GT_near_label (0.7 <= r <= 1.0) | GT_well_below_label (r < 0.7)
  label_gap  ADDED 2026-10-18 AFTER control X showed the two arms consume DIFFERENT speed labels (refcv7-50,400's extras carry a per-clip
             CONTINUOUS `v_max_ms`, refcv8's the 4-class road-type label) and BEFORE any decomposed number was read:
             gap = refcv8 class value - refcv7 per-clip value: <0 | 0-2 | 2-5 | 5-10 | >=10 m/s
  lon_v7     the GT tactical longitudinal label `lon_v7` (index into vocab_v7.TACTICAL_LON_ACTIONS_V7; -100 = unlabelled)
  ceiling    the max-speed ceiling filter ACTS in that window for the refcv8 arm in the contrast (`nav_true.sel_idx != filter_off.sel_idx`)
  sign_2x2   sign of the signed along-track error at 6 s of (new arm, ref arm): slow = P_x(6 s) < G_x(6 s).  OUTCOME-defined, not exogenous.
METRICS decomposed: ADE 0-6 s, |along-track| at 6 s, speed MAE 0-6 s (error metrics, lower better) + signed along-track at 6 s and
signed mean speed error (bias; negative = refcv8 plans SLOWER / shorter than GT).

ESTIMATOR.  Episode-cluster bootstrap over the 139 eval episodes, models FIXED, B 2000, seed 0: the episode draws are EXACTLY
`taniteval.ci._draws` (same rng, same `rng.choice(uniq)` calls), so the all-window total reproduces the banked paired cell; every bin
statistic is a ratio of resampled sums on the SAME draws (so the bins' contributions add to the total in every draw).  95 % percentile.
It answers "another draw of EPISODES" only -- NOT another training run (single training seed per arm), NOT another inference run (seed 1
is shown beside seed 0 as the inference replicate).  TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).

CONTROLS (mandatory; the tool exits non-zero and writes CONTROL_FAILED.json if any fails):
  Z1  refcv7 os vs itself, Z2 refcv8 os vs itself: every bin's contribution, bin mean and CI must read EXACTLY 0.0.
  R   reproduction: the all-window delta / lo / hi of ADE, |along| 6 s and speed MAE for refcv8 ON vs refcv7 (each tree, seed 0 and seed 1
      for 30k EMA) must equal the banked PAIRED_READ_<tree>.json S6 cells; the OFF contrast must equal NR_MATCHED_INSTRUMENT trees/<tree>.json.
  X   independent derivations: numpy |signed along at 6 s| == components' LON_along_abs_final_m; numpy ADE == components' ade_m;
      the S6 `os` == extras `nav_true.traj[:, slots]` bit-exact (refcv8 and refcv7); refcv7 filter-off == ON (inert) on the S6 windows;
      GT end-point distance from `gt_future_ext[:, 59]` == |g(6 s)|; GT / window order identical across every tree.
  P   random-partition control: the v0bin labels permuted across windows -- shares must sit near the window fractions.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
OUT = HERE.parent / "raw" / "s6_lon_decomp"
NR_TREES = HERE.parent / "raw" / "nr_matched_instrument" / "trees"
TREES = {"ema30k": "step30000ema_unc", "ema20k": "step20000ema_unc", "raw20k": "step20000_unc"}
N_BOOT = 2000
BOOT_SEED = 0
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
QUESTION = ("episode-cluster bootstrap over the 139 eval episodes, models FIXED, B 2000, seed 0 (draws identical to taniteval.ci._draws): "
            "answers 'another draw of EPISODES' only; NOT another training run (single training seed per arm), NOT another inference run")
ERR_METRICS = [("ade_m", "ADE 0-6 s", "m"), ("LON_along_abs_final_m", "|along-track| at 6 s", "m"), ("LON_speed_mae_mps", "speed MAE 0-6 s", "m/s")]
SIGNED_METRICS = [("along_signed_final_m", "signed along-track at 6 s (P_x - G_x)", "m"), ("speed_err_signed_mps", "signed mean speed error (P - G)", "m/s")]
V0_EDGES = [(0.0, 2.0, "v0<2"), (2.0, 8.0, "v0 2-8"), (8.0, 15.0, "v0 8-15"), (15.0, 25.0, "v0 15-25"), (25.0, 1e9, "v0>=25")]
DV_ORDER = ["start_from_stop", "stopping", "slow_both", "accelerating", "decelerating", "cruising"]
VMAX_NAME = {8.333: "30 km/h (8.33)", 13.889: "50 km/h (13.89)", 27.778: "100 km/h (27.78)", 33.333: "120 km/h (33.33)"}
LABEL_ORDER = ["GT_above_label", "GT_near_label", "GT_well_below_label"]


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------ loading
def load_tree(root: str, seed: int, sha12, PR, ra) -> dict:
    """S6 panel arms (via pair_read_r8.load_panel_keyed) + per-window GT speed context from decisions/ + the extras arms (seed 0)."""
    pdir = os.path.join(root, f"s6_s{seed}")
    man, A, eid, keys = PR.load_panel_keyed(pdir, sha12)
    slots = list(man["grid"]["slots"])
    dt = float(man["grid"]["dt_s"])
    epi = {int(e["file_index"]): e for e in man["episodes"]}
    files = sorted(glob.glob(os.path.join(pdir, "ep*.npz")))
    v0, v6, ep_index, clip, ws, end_chk = [], [], [], [], [], 0.0
    for f in files:
        fi = int(os.path.basename(f)[2:5])
        z, d = np.load(f), np.load(os.path.join(pdir, "decisions", os.path.basename(f)))
        if not np.array_equal(z["ws"], d["ws"]):
            raise SystemExit(f"[s6_lon_decomp] {f}: panel ws != decisions ws")
        ge = d["gt_future_ext"]
        if ge.shape[1] != 60 or float(d["gt_future_valid_ext"].min()) < 1.0:
            raise SystemExit(f"[s6_lon_decomp] {f}: gt_future_ext not a full valid 60-step future")
        v0.append(z["v0"].astype(np.float64))
        v6.append(ge[:, 59, 3].astype(np.float64))
        # control X: the GT end-point distance from the world-frame future == |g(6 s)| in the ego frame
        dist_w = np.linalg.norm(ge[:, 59, :2].astype(np.float64) - d["pose_last"][:, :2].astype(np.float64), axis=-1)
        end_chk = max(end_chk, float(np.abs(dist_w - np.linalg.norm(z["g"][:, -1].astype(np.float64), axis=-1)).max()))
        ep_index += [int(epi[fi]["episode_index"])] * len(z["ws"])
        clip += [str(epi[fi]["clip_id"])] * len(z["ws"])
        ws += [int(w) for w in z["ws"]]
    T = {"root": root, "seed": seed, "man": man, "slots": slots, "dt": dt, "eid": np.asarray(eid), "keys": keys,
         "g": A["g"], "os": A["os"], "ref": A[f"b_refcv7_50400_s{seed}"], "v0": np.concatenate(v0), "v6": np.concatenate(v6),
         "ep_index": np.asarray(ep_index), "clip": clip, "ws": np.asarray(ws), "chk": {"gt_end_distance_world_vs_ego_max_abs_m": end_chk}}
    if len(T["v0"]) != len(eid):
        raise SystemExit("[s6_lon_decomp] window count mismatch between own loop and load_panel_keyed")
    if seed == 0:
        ex = np.load(os.path.join(root, "dump_s0", "refcv6_extras.npz"))
        key = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(ex["clip_index"], ex["ws"]))}
        rows = [key[(int(e), int(w))] for e, w in zip(T["ep_index"], T["ws"])]
        T["extras_rows"] = rows
        T["on_x"] = ex["nav_true.traj"][rows][:, slots].astype(np.float32)
        T["off"] = ex["filter_off.traj"][rows][:, slots].astype(np.float32)
        T["vz"] = ex["vmax_zero.traj"][rows][:, slots].astype(np.float32)
        T["ceil_acts"] = ex["nav_true.sel_idx"][rows] != ex["filter_off.sel_idx"][rows]
        T["vz_sel_changed"] = ex["nav_true.sel_idx"][rows] != ex["vmax_zero.sel_idx"][rows]
        T["v_max_ms"] = ex["v_max_ms"][rows].astype(np.float64)
        T["v_max_valid"] = ex["v_max_valid"][rows]
        T["gt_max_speed_6s"] = ex["gt_max_speed_6s"][rows].astype(np.float64)
        T["lon_v7"] = ex["lon_v7"][rows].astype(np.int64)
        T["tac_lon_pred"] = np.argmax(ex["nav_true.tacv6_lon_logits"][rows], axis=1).astype(np.int64)
        T["extras_v0_vs_panel_v0_max_abs"] = float(np.abs(ex["v0"][rows].astype(np.float64) - T["v0"]).max())
        T["chk"]["os_vs_extras_nav_true_slots_max_abs"] = float(np.abs(T["os"].astype(np.float64) - T["on_x"].astype(np.float64)).max())
    return T


def load_r7_extras(r7root: str, T: dict) -> dict:
    """refcv7-50,400 extras rows for T's windows, matched on (clip_id, ws) through the refcv7 dump's own manifest."""
    d = os.path.join(r7root, "dump_s0")
    man = jload(os.path.join(d, "manifest.json"))
    cid = {int(e["episode_index"]): str(e["clip_id"]) for e in man["episodes"]}
    ex = np.load(os.path.join(d, "refcv6_extras.npz"))
    key = {(cid[int(c)], int(w)): i for i, (c, w) in enumerate(zip(ex["clip_index"], ex["ws"]))}
    rows = [key[(c, int(w))] for c, w in zip(T["clip"], T["ws"])]
    sl = T["slots"]
    out = {"on_x": ex["nav_true.traj"][rows][:, sl].astype(np.float32), "off": ex["filter_off.traj"][rows][:, sl].astype(np.float32),
           "vz": ex["vmax_zero.traj"][rows][:, sl].astype(np.float32), "v_max_ms": ex["v_max_ms"][rows].astype(np.float64)}
    out["chk"] = {"r7_S6_ref_vs_extras_nav_true_slots_max_abs": float(np.abs(T["ref"].astype(np.float64) - out["on_x"].astype(np.float64)).max()),
                  "r7_filter_off_vs_on_S6_max_abs": float(np.abs(out["off"].astype(np.float64) - out["on_x"].astype(np.float64)).max()),
                  "r7_filter_changes_selection_S6": int((ex["nav_true.sel_idx"][rows] != ex["filter_off.sel_idx"][rows]).sum()),
                  "r7_vs_r8_vmax_label_max_abs": float(np.abs(out["v_max_ms"] - T["v_max_ms"]).max()),
                  "r7_vs_r8_vmax_label_n_windows_differ": int((np.abs(out["v_max_ms"] - T["v_max_ms"]) > 1e-3).sum()),
                  "r7_vmax_n_distinct_values": int(len(np.unique(out["v_max_ms"].round(3)))),
                  "r8_vmax_n_distinct_values": int(len(np.unique(T["v_max_ms"].round(3))))}
    # is refcv7's label constant within each clip?  (per-clip label vs per-window)
    per_clip = {}
    for c, v in zip(T["clip"], out["v_max_ms"]):
        per_clip.setdefault(c, set()).add(round(float(v), 3))
    out["chk"]["r7_vmax_constant_within_every_clip"] = bool(all(len(x) == 1 for x in per_clip.values()))
    out["chk"]["r7_vmax_minus_clip_max_gt_speed_6s_mean"] = float(np.mean([
        np.mean([v for c2, v in zip(T["clip"], out["v_max_ms"]) if c2 == c]) - max(g for c2, g in zip(T["clip"], T["gt_max_speed_6s"]) if c2 == c)
        for c in per_clip]))
    return out


# ------------------------------------------------------------------------------------------ per-window metrics
def window_metrics(PR, ra, P, G, dt) -> dict:
    c = PR.components(ra, P, G, dt)
    out = {k: np.asarray(c[k], dtype=np.float64) for k, _l, _u in ERR_METRICS}
    P64, G64 = P.astype(np.float64), G.astype(np.float64)
    out["along_signed_final_m"] = P64[:, -1, 0] - G64[:, -1, 0]

    def arclen(X):
        Z = np.concatenate([np.zeros_like(X[:, :1]), X], axis=1)
        return np.linalg.norm(np.diff(Z, axis=1), axis=-1).sum(1)
    out["speed_err_signed_mps"] = (arclen(P64) - arclen(G64)) / (P.shape[1] * dt)   # mean over the 6 steps of (|dP| - |dG|) / dt
    out["_x_along_abs_vs_components_max_abs"] = float(np.abs(np.abs(out["along_signed_final_m"]) - out["LON_along_abs_final_m"]).max())
    out["_x_ade_numpy_vs_components_max_abs"] = float(np.abs(np.linalg.norm(P64 - G64, axis=-1).mean(1) - out["ade_m"]).max())
    for k, _l, _u in ERR_METRICS + SIGNED_METRICS:
        if not np.isfinite(out[k]).all():
            raise SystemExit(f"[s6_lon_decomp] non-finite per-window {k}")
    return out


# ------------------------------------------------------------------------------------------ bootstrap
def boot_counts(eid, n_boot=N_BOOT, seed=BOOT_SEED):
    """Episode multiplicity per draw, from EXACTLY the draws of taniteval.ci._draws (same rng, same choice calls)."""
    e = np.asarray([str(x) for x in eid])
    uniq = np.unique(e)
    pos = {u: i for i, u in enumerate(uniq)}
    ep_idx = np.asarray([pos[x] for x in e])
    rng = np.random.default_rng(seed)
    C = np.zeros((n_boot, len(uniq)), dtype=np.float64)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        C[b] = np.bincount([pos[p] for p in pick], minlength=len(uniq))
    return C, ep_idx, len(uniq)


def pct(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def decompose(d, labels, order, BC) -> dict:
    """d: per-window delta; labels: per-window bin name.  Contribution / share / bin mean, joint episode-cluster draws."""
    C, ep_idx, n_ep = BC
    N = d.size
    Ne = np.bincount(ep_idx, minlength=n_ep).astype(np.float64)
    den = C @ Ne
    tot_b = (C @ np.bincount(ep_idx, weights=d, minlength=n_ep)) / den
    total = float(d.mean())
    t_lo, t_hi = pct(tot_b)
    res = {"total": {"point": total, "lo": t_lo, "hi": t_hi, "n_windows": int(N), "n_episodes": int(n_ep),
                     "frac_draws_total_le_0": float((tot_b <= 0).mean())}, "bins": {}}
    csum = 0.0
    for b in order:
        m = labels == b
        nb = int(m.sum())
        if nb == 0:
            res["bins"][b] = {"n_windows": 0}
            continue
        Sb = np.bincount(ep_idx, weights=np.where(m, d, 0.0), minlength=n_ep)
        Nb = np.bincount(ep_idx, weights=m.astype(np.float64), minlength=n_ep)
        contrib_b = (C @ Sb) / den
        with np.errstate(divide="ignore", invalid="ignore"):
            share_b = contrib_b / tot_b
            nbb = C @ Nb
            mean_b = np.where(nbb > 0, (C @ Sb) / np.where(nbb > 0, nbb, 1.0), np.nan)
        contrib = float(d[m].sum() / N)
        csum += contrib
        share = (contrib / total) if total != 0 else None
        c_lo, c_hi = pct(contrib_b)
        s_lo, s_hi = pct(share_b) if total != 0 else (None, None)
        m_lo, m_hi = pct(mean_b)
        res["bins"][b] = {"n_windows": nb, "n_episodes": int((Nb > 0).sum()), "frac_windows": nb / N,
                          "contrib": contrib, "contrib_lo": c_lo, "contrib_hi": c_hi,
                          "contrib_separated": bool(c_lo is not None and (c_lo > 0 or c_hi < 0)),
                          "share": share, "share_lo": s_lo, "share_hi": s_hi,
                          "enrichment": (share / (nb / N)) if share is not None else None,
                          "mean_delta": float(d[m].mean()), "mean_lo": m_lo, "mean_hi": m_hi,
                          "mean_separated": bool(m_lo is not None and (m_lo > 0 or m_hi < 0)),
                          "n_draws_bin_empty": int((nbb == 0).sum())}
    res["sum_of_contribs_minus_total"] = float(csum - total)
    ranked = sorted([(b, v["share"]) for b, v in res["bins"].items() if v.get("n_windows") and v.get("share") is not None],
                    key=lambda t: -t[1])
    acc, top = 0.0, []
    for b, s in ranked:
        if acc >= 0.5:
            break
        top.append(b)
        acc += s
    res["bins_reaching_50pct"] = {"bins": top, "cum_share": acc} if total != 0 else None
    return res


# ------------------------------------------------------------------------------------------ partitions
def partitions(T: dict) -> tuple:
    v0, v6 = T["v0"], T["v6"]
    dv = v6 - v0
    v0bin = np.empty(v0.size, dtype=object)
    for lo, hi, name in V0_EDGES:
        v0bin[(v0 >= lo) & (v0 < hi)] = name
    dvc = np.empty(v0.size, dtype=object)
    rules = [("start_from_stop", (v0 < 1.0) & (v6 >= 2.0)), ("stopping", (v0 >= 2.0) & (v6 < 1.0)), ("slow_both", (v0 < 2.0) & (v6 < 2.0)),
             ("accelerating", dv >= 1.5), ("decelerating", dv <= -1.5), ("cruising", np.abs(dv) < 1.5)]
    done = np.zeros(v0.size, dtype=bool)
    for name, cond in rules:
        sel = cond & ~done
        dvc[sel] = name
        done |= sel
    if not done.all():
        raise SystemExit("[s6_lon_decomp] dvclass rules are not exhaustive")
    cross = np.asarray([f"{a} | {b}" for a, b in zip(v0bin, dvc)], dtype=object)
    vm = np.asarray([VMAX_NAME.get(round(float(x), 3), f"other {x:.3f}") for x in T["v_max_ms"]], dtype=object)
    r = T["gt_max_speed_6s"] / T["v_max_ms"]
    lab = np.where(r > 1.0, "GT_above_label", np.where(r >= 0.7, "GT_near_label", "GT_well_below_label")).astype(object)
    from tanitad.models.vocab_v7 import TACTICAL_LON_ACTIONS_V7 as LON
    lon = np.asarray([("unlabelled (-100)" if k < 0 else f"{k} {LON[k]}") for k in T["lon_v7"]], dtype=object)
    gap = T["v_max_ms"] - T["r7_v_max_ms"]
    gp = np.where(gap < 0, "gap<0", np.where(gap < 2, "gap 0-2", np.where(gap < 5, "gap 2-5", np.where(gap < 10, "gap 5-10", "gap>=10")))).astype(object)
    P = {"v0bin": (v0bin, [n for _a, _b, n in V0_EDGES]), "dvclass": (dvc, DV_ORDER),
         "label_gap": (gp, ["gap<0", "gap 0-2", "gap 2-5", "gap 5-10", "gap>=10"]),
         "v0_x_dv": (cross, [f"{a} | {b}" for _x, _y, a in V0_EDGES for b in DV_ORDER]),
         "vmax_label": (vm, list(VMAX_NAME.values())), "label_vs_gt": (lab, LABEL_ORDER),
         "lon_v7": (lon, ["unlabelled (-100)"] + [f"{i} {n}" for i, n in enumerate(LON)])}
    return P, dv


def sign_partition(new_signed, ref_signed):
    a = np.where(new_signed < 0, "new slow", "new fast")
    b = np.where(ref_signed < 0, "ref slow", "ref fast")
    return np.asarray([f"{x} / {y}" for x, y in zip(a, b)], dtype=object), \
        ["new slow / ref slow", "new slow / ref fast", "new fast / ref slow", "new fast / ref fast"]


# ------------------------------------------------------------------------------------------ plan-derived longitudinal decision
LONCLS = ["decel", "cruise", "accel"]


def plan_lon_class(P, v0, dt):
    """decel / cruise / accel from the path's speed over its LAST second (|P6 - P5| / dt) minus the measured v0 (+-1.5 m/s, the dvclass
    threshold).  Applied identically to GT (the `g` arm) and to every plan, so the confusion compares like with like."""
    P = P.astype(np.float64)
    v_end = np.linalg.norm(P[:, -1] - P[:, -2], axis=-1) / dt
    dv = v_end - v0
    return np.where(dv <= -1.5, "decel", np.where(dv >= 1.5, "accel", "cruise")).astype(object)


def lon_confusion(PR, arms: dict, G, v0, dt, eid, tac_pred=None, lon_v7=None) -> dict:
    gt = plan_lon_class(G, v0, dt)
    out = {"gt_class_counts": {c: int((gt == c).sum()) for c in LONCLS}, "arms": {}, "paired": {}}
    ind = {}
    for a, P in arms.items():
        pc = plan_lon_class(P, v0, dt)
        out["arms"][a] = {"confusion_rows_gt_cols_plan": {g: {p: int(((gt == g) & (pc == p)).sum()) for p in LONCLS} for g in LONCLS},
                          "accuracy": float((pc == gt).mean())}
        ind[a] = {"false_decel_in_gt_cruise": np.where(gt == "cruise", (pc == "decel").astype(np.float64), np.nan),
                  "correct": (pc == gt).astype(np.float64),
                  "missed_decel_in_gt_decel": np.where(gt == "decel", (pc != "decel").astype(np.float64), np.nan)}
    for k in ("false_decel_in_gt_cruise", "correct", "missed_decel_in_gt_decel"):
        out["paired"][k] = PR.paired_cell(ind["ema30k_ON"][k], ind["r7_ON"][k], list(eid))
    if tac_pred is not None:
        from tanitad.models.vocab_v7 import TACTICAL_LON_ACTIONS_V7 as LONV
        p8, p7 = plan_lon_class(arms["ema30k_ON"], v0, dt), plan_lon_class(arms["r7_ON"], v0, dt)
        cells = {"GT cruise, refcv8 plan decel, refcv7 plan cruise": (gt == "cruise") & (p8 == "decel") & (p7 == "cruise"),
                 "GT cruise, refcv8 plan cruise": (gt == "cruise") & (p8 == "cruise"),
                 "GT decel, refcv8 plan decel, refcv7 plan cruise": (gt == "decel") & (p8 == "decel") & (p7 == "cruise"),
                 "all windows": np.ones(len(gt), bool)}
        out["refcv8_tac_lon_pred_by_cell"] = {c: {"n": int(m.sum()), **{LONV[k]: int(((tac_pred == k) & m).sum()) for k in range(len(LONV))}}
                                              for c, m in cells.items()}
        lab = lon_v7
        out["refcv8_tac_lon_pred_given_label"] = {LONV[g]: {"n": int((lab == g).sum()), **{LONV[k]: int(((tac_pred == k) & (lab == g)).sum()) for k in range(len(LONV))}}
                                                  for g in range(len(LONV)) if (lab == g).any()}
    return out


# ------------------------------------------------------------------------------------------ bias structure (offset vs compression)
def bias_structure(M: dict, G, arms: list, BC) -> dict:
    """Per arm: mean signed along at 6 s (episode-cluster CI) and the OLS of plan x(6 s) on GT x(6 s) (slope < 1 = speed profile
    compressed toward the mean; intercept = offset).  Slope differences vs refcv7 on the same draws."""
    C, ep_idx, n_ep = BC
    gx = G[:, -1, 0].astype(np.float64)
    out = {}
    W = C[:, ep_idx]                                       # [B, n] window weights per draw (episode multiplicity)

    def wols(y):
        sw = W.sum(1)
        mx, my = (W @ gx) / sw, (W @ y) / sw
        cov = (W @ (gx * y)) / sw - mx * my
        var = (W @ (gx * gx)) / sw - mx * mx
        return cov / var, my - (cov / var) * mx
    ref_b = None
    for a in arms:
        y = gx + M[a]["along_signed_final_m"]              # plan x at 6 s
        sl = np.polyfit(gx, y, 1)
        s_b, _i_b = wols(y)
        if a == "r7_ON":
            ref_b = s_b
        sg = M[a]["along_signed_final_m"]
        mb = (C @ np.bincount(ep_idx, weights=sg, minlength=n_ep)) / (C @ np.bincount(ep_idx, minlength=n_ep))
        out[a] = {"mean_signed_along_6s_m": float(sg.mean()), "ci": pct(mb), "frac_windows_slow": float((sg < 0).mean()),
                  "mean_signed_speed_err_mps": float(M[a]["speed_err_signed_mps"].mean()),
                  "ols_slope": float(sl[0]), "ols_intercept_m": float(sl[1]), "ols_slope_ci": pct(s_b), "_s_b": s_b}
    for a in arms:
        if a != "r7_ON" and ref_b is not None:
            d = out[a]["_s_b"] - ref_b
            out[a]["slope_minus_r7"] = {"point": out[a]["ols_slope"] - out["r7_ON"]["ols_slope"], "ci": pct(d)}
    for a in arms:
        del out[a]["_s_b"]
    return out


# ------------------------------------------------------------------------------------------ lever L1: post-hoc speed calibration
BETA_GRID = np.round(np.arange(-2.0, 2.0001, 0.05), 2)


def retime(P, beta, dt):
    """Move every waypoint along the plan's OWN polyline by beta * t metres (a uniform speed offset of beta m/s; arc length is
    clipped at 0 and kept monotone; beyond the end the last segment's direction is extrapolated).  retime(P, 0) == P."""
    P = P.astype(np.float64)
    n, K, _ = P.shape
    Z = np.concatenate([np.zeros((n, 1, 2)), P], axis=1)
    seg = np.diff(Z, axis=1)
    Ls = np.linalg.norm(seg, axis=-1)
    S = np.concatenate([np.zeros((n, 1)), np.cumsum(Ls, axis=1)], axis=1)
    u = np.tile(np.array([1.0, 0.0]), (n, 1))
    for j in range(K):
        ok = Ls[:, j] > 1e-3
        u[ok] = seg[ok, j] / Ls[ok, j, None]
    D = np.where(Ls[..., None] > 1e-6, seg / np.maximum(Ls, 1e-12)[..., None], u[:, None, :])
    t = dt * np.arange(1, K + 1)
    a = np.maximum.accumulate(np.maximum(S[:, 1:] + beta * t[None, :], 0.0), axis=1)
    idx = (S[:, None, 1:K] <= a[:, :, None]).sum(-1)       # [n, K] segment index 0..K-1
    r = np.arange(n)[:, None]
    return (Z[r, idx] + (a - S[r, idx])[..., None] * D[r, idx]).astype(np.float32)


def lever_calibration(PR, ra, arms: dict, G, dt, eid, v0bin_labels, dv_labels, BC) -> dict:
    """2-fold EPISODE cross-fitted speed offset (fit on odd episodes, applied to even, and vice versa), global and per v0 bin.
    Fit criterion: mean ADE on the FIT fold only; the scored fold is never tuned on.  DIAGNOSTIC (fit on eval episodes), not deployable."""
    fold = (np.asarray(eid) % 2).astype(int)
    G64 = G.astype(np.float64)
    out = {"beta_grid": [float(BETA_GRID[0]), float(BETA_GRID[-1]), 0.05], "folds": "episode position parity (eid % 2)",
           "criterion": "mean ADE 0-6 s on the fit fold", "arms": {}}
    out["known_value_retime_beta0_max_abs_m"] = max(float(np.abs(retime(P, 0.0, dt).astype(np.float64) - P.astype(np.float64)).max())
                                                    for P in arms.values())
    cal = {}
    for name, P in arms.items():
        R = np.stack([np.linalg.norm(retime(P, b, dt).astype(np.float64) - G64, axis=-1).mean(1) for b in BETA_GRID])   # [nb, n]
        rec = {}
        for mode in ("global", "per_v0bin"):
            beta_w = np.zeros(len(eid))
            chosen = {}
            for f in (0, 1):
                fit, app = fold != f, fold == f
                groups = [("all", np.ones(len(eid), bool))] if mode == "global" else [(b, v0bin_labels == b) for b in np.unique(v0bin_labels)]
                for gname, gm in groups:
                    fm = fit & gm
                    if fm.sum() == 0:
                        continue
                    b = float(BETA_GRID[int(np.argmin(R[:, fm].mean(1)))])
                    chosen[f"fold{f}:{gname}"] = b
                    beta_w[app & gm] = b
            Pc = np.empty_like(P)
            for b in np.unique(beta_w):
                m = beta_w == b
                Pc[m] = retime(P[m], float(b), dt)
            cal[(name, mode)] = Pc
            rec[mode] = {"betas_chosen_mps": chosen}
        out["arms"][name] = rec
    comp = {key: PR.components(ra, Pc, G, dt) for key, Pc in cal.items()}
    raw = {name: PR.components(ra, P, G, dt) for name, P in arms.items()}
    rows = {}
    for mode in ("global", "per_v0bin"):
        for k, _l, _u in ERR_METRICS:
            r8c, r7c = comp[("ema30k_ON", mode)][k], comp[("r7_ON", mode)][k]
            rows[f"{mode}:{k}"] = {
                "r8cal_minus_r7raw": PR.paired_cell(r8c, raw["r7_ON"][k], list(eid)),
                "r8cal_minus_r7cal": PR.paired_cell(r8c, r7c, list(eid)),
                "r8cal_minus_r8raw": PR.paired_cell(r8c, raw["ema30k_ON"][k], list(eid)),
                "r7cal_minus_r7raw": PR.paired_cell(r7c, raw["r7_ON"][k], list(eid))}
        d = np.asarray(comp[("ema30k_ON", mode)]["ade_m"], np.float64) - np.asarray(comp[("r7_ON", mode)]["ade_m"], np.float64)
        rows[f"{mode}:ade_by_dvclass_r8cal_minus_r7cal"] = decompose(d, dv_labels, DV_ORDER, BC)
    rows["raw:ade_m_r8raw_minus_r7raw"] = PR.paired_cell(raw["ema30k_ON"]["ade_m"], raw["r7_ON"]["ade_m"], list(eid))
    out["rows"] = rows
    return out


# ------------------------------------------------------------------------------------------ main
def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[s6_lon_decomp] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    t0 = time.time()
    import r8_battery_patch as BP
    L, Pm = BP.apply()
    ra = Pm.RR.ra3().ra
    sha12 = L.sha12
    import pair_read_r8 as PR
    OUT.mkdir(parents=True, exist_ok=True)

    trees = {lab: load_tree(os.path.join(ARCHIVE, t), 0, sha12, PR, ra) for lab, t in TREES.items()}
    s1 = load_tree(os.path.join(ARCHIVE, TREES["ema30k"]), 1, sha12, PR, ra)
    T = trees["ema30k"]
    r7x = load_r7_extras(BP.R7_BASE_ROOT, T)
    ctl = {"X": {}, "R": {}, "Z": {}}
    # --- X: window identity across trees and seeds; GT identical
    for lab, U in list(trees.items()) + [("ema30k_s1", s1)]:
        same_keys = U["keys"] == T["keys"]
        ctl["X"][f"{lab}_keys_equal_ema30k_s0"] = bool(same_keys)
        ctl["X"][f"{lab}_gt_max_abs_vs_ema30k_s0"] = float(np.abs(U["g"].astype(np.float64) - T["g"].astype(np.float64)).max()) if same_keys else None
        ctl["X"][f"{lab}_ref_r7_max_abs_vs_ema30k_s0_ref"] = (float(np.abs(U["ref"].astype(np.float64) - T["ref"].astype(np.float64)).max())
                                                             if same_keys else None)
        ctl["X"][f"{lab}_chk"] = U["chk"]
    ctl["X"]["r7_extras"] = r7x["chk"]
    ctl["X"]["extras_v0_vs_panel_v0_max_abs"] = T["extras_v0_vs_panel_v0_max_abs"]
    ctl["X"]["v_max_valid_min"] = float(T["v_max_valid"].min())

    dt = T["dt"]
    G = T["g"]
    M = {}   # arm name -> per-window metrics
    arms = {"r7_ON": T["ref"], "r7_vmaxzero": r7x["vz"], "ema30k_ON": T["os"], "ema30k_OFF": T["off"], "ema30k_vmaxzero": T["vz"],
            "ema20k_ON": trees["ema20k"]["os"], "ema20k_OFF": trees["ema20k"]["off"], "raw20k_ON": trees["raw20k"]["os"],
            "raw20k_OFF": trees["raw20k"]["off"]}
    for name, P in arms.items():
        M[name] = window_metrics(PR, ra, P, G, dt)
    arms_all = dict(arms)
    arms_all["ema30k_ON_s1"], arms_all["r7_ON_s1"] = s1["os"], s1["ref"]
    M["ema30k_ON_s1"] = window_metrics(PR, ra, s1["os"], s1["g"], dt)
    M["r7_ON_s1"] = window_metrics(PR, ra, s1["ref"], s1["g"], dt)
    ctl["X"]["numpy_vs_components"] = {n: {"along_abs": m["_x_along_abs_vs_components_max_abs"], "ade": m["_x_ade_numpy_vs_components_max_abs"]}
                                       for n, m in M.items()}
    T["r7_v_max_ms"] = r7x["v_max_ms"]
    BC = boot_counts(T["eid"])
    BC1 = boot_counts(s1["eid"])
    parts, dv = partitions(T)
    # --- R: reproduction of banked cells (all-window total) -- the bootstrap is the SAME as taniteval.ci's
    from taniteval import ci as CI

    def banked_cell(tree, seed, key):
        return jload(os.path.join(ARCHIVE, tree, f"PAIRED_READ_{tree}.json"))["surfaces"]["S6"]["seeds"][str(seed)]["rows"][key]["os_minus_refcv7_50400"]

    def repro(new, ref, eid, bank):
        r = CI.paired_episode_cluster_bootstrap(new, ref, list(eid), n_boot=N_BOOT, seed=BOOT_SEED)
        flds = ("delta", "lo", "hi", "n_windows", "n_episodes")
        return {"ci_reread": {f: r[f] for f in flds}, "banked": {f: bank.get(f) for f in flds},
                "exact": all(r[f] == bank.get(f) for f in flds)}
    rep = {}
    for lab, tree in TREES.items():
        for k, _l, _u in ERR_METRICS:
            rep[f"{lab}_s0_ON:{k}"] = repro(M[f"{lab}_ON"][k], M["r7_ON"][k], T["eid"], banked_cell(tree, 0, k))
    for k, _l, _u in ERR_METRICS:
        rep[f"ema30k_s1_ON:{k}"] = repro(M["ema30k_ON_s1"][k], M["r7_ON_s1"][k], s1["eid"], banked_cell(TREES["ema30k"], 1, k))
        nr = jload(str(NR_TREES / f"{TREES['ema30k']}.json"))["surfaces"]["S6"]["read_OFF"]["rows"][k]["os_minus_refcv7_50400"]
        rep[f"ema30k_s0_OFF:{k}"] = repro(M["ema30k_OFF"][k], M["r7_ON"][k], T["eid"], nr)
    # my joint-draw total must equal ci's unrounded total bounds (same draws)
    jd = decompose(M["ema30k_ON"]["ade_m"] - M["r7_ON"]["ade_m"], parts["v0bin"][0], parts["v0bin"][1], BC)["total"]
    r_un = CI.paired_episode_cluster_bootstrap(M["ema30k_ON"]["ade_m"], M["r7_ON"]["ade_m"], list(T["eid"]), n_boot=N_BOOT, seed=BOOT_SEED)
    rep["joint_draws_total_vs_ci_ade"] = {"joint": jd, "ci": {f: r_un[f] for f in ("delta", "lo", "hi")},
                                          "exact_at_ci_rounding": bool(round(jd["lo"], 4) == r_un["lo"] and round(jd["hi"], 4) == r_un["hi"]
                                                                       and round(jd["point"], 4) == r_un["delta"])}
    ctl["R"] = rep

    # --- contrasts
    CONTRASTS = [
        ("C0", "refcv8 EMA-30k ON minus refcv7-50,400 (seed 0) -- THE TARGET", "ema30k_ON", "r7_ON", "ema30k", 0),
        ("C0s1", "refcv8 EMA-30k ON minus refcv7-50,400 (inference seed 1 replicate)", "ema30k_ON_s1", "r7_ON_s1", None, 1),
        ("C1", "refcv8 EMA-30k OFF (ceiling filter off = matched instrument) minus refcv7-50,400", "ema30k_OFF", "r7_ON", "ema30k", 0),
        ("C2", "refcv8 EMA-30k ON minus OFF (what the ceiling filter does to refcv8)", "ema30k_ON", "ema30k_OFF", "ema30k", 0),
        ("C3", "refcv8 EMA-30k max-speed label WITHHELD minus refcv7-50,400 label WITHHELD", "ema30k_vmaxzero", "r7_vmaxzero", "ema30k", 0),
        ("C3b", "refcv8 EMA-30k ON minus refcv8 EMA-30k label WITHHELD (refcv8's label effect)", "ema30k_ON", "ema30k_vmaxzero", "ema30k", 0),
        ("C3c", "refcv7-50,400 ON minus refcv7 label WITHHELD (refcv7's label effect)", "r7_ON", "r7_vmaxzero", None, 0),
        ("C4", "refcv8 EMA-20k ON minus refcv7-50,400", "ema20k_ON", "r7_ON", "ema20k", 0),
        ("C5", "refcv8 RAW-20k ON minus refcv7-50,400", "raw20k_ON", "r7_ON", "raw20k", 0),
        ("C6", "refcv8 EMA-30k ON minus refcv8 EMA-20k ON (what moved 20k -> 30k)", "ema30k_ON", "ema20k_ON", "ema30k", 0),
        ("C7", "refcv8 RAW-20k ON minus refcv8 EMA-20k ON (raw vs EMA at 20k)", "raw20k_ON", "ema20k_ON", "raw20k", 0),
        ("Z1", "CONTROL refcv7 ON minus itself (must read exactly 0)", "r7_ON", "r7_ON", None, 0),
        ("Z2", "CONTROL refcv8 EMA-30k ON minus itself (must read exactly 0)", "ema30k_ON", "ema30k_ON", "ema30k", 0),
    ]
    res = {"tool": "s6_lon_decomp.py", "generated": time.strftime("%FT%T"), "tier": TIER, "estimator": QUESTION, "n_boot": N_BOOT,
           "boot_seed": BOOT_SEED, "surface": "S6 0-6 s (instants 1..6 s, dt 1.0)", "trees": {k: os.path.join(ARCHIVE, v) for k, v in TREES.items()},
           "refcv7_root": BP.R7_BASE_ROOT, "n_windows": int(len(T["eid"])), "n_episodes": int(len(set(T["eid"].tolist()))),
           "partition_rules": __doc__.split("PRE-REGISTERED PARTITIONS")[1].split("METRICS decomposed")[0].strip(),
           "window_fraction": {p: {b: float((lab == b).mean()) for b in order} for p, (lab, order) in parts.items()},
           "contrasts": {}, "controls": ctl}
    for cid, desc, new, ref, ceil_tree, seed in CONTRASTS:
        bc = BC1 if seed == 1 else BC
        ent = {"desc": desc, "new": new, "ref": ref, "inference_seed": seed, "partitions": {}}
        P = dict(parts)
        if ceil_tree is not None:
            ca = trees[ceil_tree]["ceil_acts"]
            P["ceiling"] = (np.where(ca, f"ceiling ACTS ({ceil_tree})", f"ceiling inert ({ceil_tree})").astype(object),
                            [f"ceiling ACTS ({ceil_tree})", f"ceiling inert ({ceil_tree})"])
            from tanitad.models.vocab_v7 import TACTICAL_LON_ACTIONS_V7 as LONV
            tp = trees[ceil_tree]["tac_lon_pred"]
            P["r8_tac_lon_pred"] = (np.asarray([f"pred {LONV[k]} ({ceil_tree})" for k in tp], dtype=object),
                                    [f"pred {n} ({ceil_tree})" for n in LONV])
            lv7 = T["lon_v7"]
            P["r8_tac_lon_correct"] = (np.where(lv7 < 0, "unlabelled", np.where(tp == lv7, "tac lon pred == lon_v7", "tac lon pred != lon_v7")).astype(object),
                                       ["tac lon pred == lon_v7", "tac lon pred != lon_v7", "unlabelled"])
        P["sign_2x2"] = sign_partition(M[new]["along_signed_final_m"], M[ref]["along_signed_final_m"])
        cn, cr = plan_lon_class(arms_all[new], T["v0"], dt), plan_lon_class(arms_all[ref], T["v0"], dt)
        P["plan_dec_new_ref"] = (np.asarray([f"new {x} / ref {y}" for x, y in zip(cn, cr)], dtype=object),
                                 [f"new {x} / ref {y}" for x in LONCLS for y in LONCLS])
        for pname, (lab, order) in P.items():
            ent["partitions"][pname] = {}
            for k, _l, _u in ERR_METRICS + SIGNED_METRICS:
                ent["partitions"][pname][k] = decompose(M[new][k] - M[ref][k], lab, order, bc)
            # signed levels per bin (both arms) + fraction of windows where the new arm is too slow
            lv = {}
            for b in order:
                m = lab == b
                if not m.any():
                    continue
                lv[b] = {"new_along_signed_mean": float(M[new]["along_signed_final_m"][m].mean()),
                         "ref_along_signed_mean": float(M[ref]["along_signed_final_m"][m].mean()),
                         "new_speed_err_signed_mean": float(M[new]["speed_err_signed_mps"][m].mean()),
                         "ref_speed_err_signed_mean": float(M[ref]["speed_err_signed_mps"][m].mean()),
                         "frac_new_slow": float((M[new]["along_signed_final_m"][m] < 0).mean()),
                         "frac_ref_slow": float((M[ref]["along_signed_final_m"][m] < 0).mean()),
                         "gt_v0_mean": float(T["v0"][m].mean()), "gt_dv_mean": float(dv[m].mean())}
            ent["partitions"][pname]["_levels"] = lv
        res["contrasts"][cid] = ent
        print(f"[{time.time() - t0:6.1f}s] {cid} done", flush=True)
    # --- P: random-partition control on C0 (v0bin labels permuted across windows)
    rng = np.random.default_rng(12345)
    perm = parts["v0bin"][0][rng.permutation(len(parts["v0bin"][0]))]
    ctl["P_random_partition_C0"] = {k: decompose(M["ema30k_ON"][k] - M["r7_ON"][k], perm, parts["v0bin"][1], BC)
                                    for k, _l, _u in ERR_METRICS}
    # --- Z verdict: every bin exactly 0
    zok = True
    for cid in ("Z1", "Z2"):
        for pname, pm in res["contrasts"][cid]["partitions"].items():
            for k, dd in pm.items():
                if k == "_levels":
                    continue
                for b, v in dd["bins"].items():
                    if v.get("n_windows"):
                        vals = [v["contrib"], v["contrib_lo"], v["contrib_hi"], v["mean_delta"], v["mean_lo"], v["mean_hi"]]
                        if any(x != 0.0 for x in vals):
                            zok = False
                if dd["total"]["point"] != 0.0 or dd["total"]["lo"] != 0.0 or dd["total"]["hi"] != 0.0:
                    zok = False
    ctl["Z"]["all_bins_exactly_zero"] = zok
    xok = (all(v is True for k, v in ctl["X"].items() if k.endswith("_keys_equal_ema30k_s0"))
           and all((v == 0.0) for k, v in ctl["X"].items() if k.endswith("_gt_max_abs_vs_ema30k_s0"))
           and all((v == 0.0) for k, v in ctl["X"].items() if k.endswith("_ref_r7_max_abs_vs_ema30k_s0_ref") and not k.startswith("ema30k_s1"))
           and all(U["chk"]["os_vs_extras_nav_true_slots_max_abs"] == 0.0 for U in trees.values())
           and r7x["chk"]["r7_S6_ref_vs_extras_nav_true_slots_max_abs"] == 0.0
           and r7x["chk"]["r7_filter_changes_selection_S6"] == 0
           and all(v["along_abs"] < 1e-4 and v["ade"] < 1e-4 for v in ctl["X"]["numpy_vs_components"].values())
           and all(U["chk"]["gt_end_distance_world_vs_ego_max_abs_m"] < 0.05 for U in list(trees.values()) + [s1]))
    ctl["X_PASS"] = bool(xok)
    ctl["R_PASS"] = bool(all(v.get("exact") for k, v in rep.items() if k != "joint_draws_total_vs_ci_ade")
                         and rep["joint_draws_total_vs_ci_ade"]["exact_at_ci_rounding"])
    ctl["Z_PASS"] = bool(zok)
    ctl["ALL_PASS"] = bool(ctl["X_PASS"] and ctl["R_PASS"] and ctl["Z_PASS"])
    res["lon_decision_confusion"] = lon_confusion(PR, {"r7_ON": T["ref"], "ema30k_ON": T["os"], "ema30k_OFF": T["off"], "ema20k_ON": trees["ema20k"]["os"]},
                                                  G, T["v0"], dt, T["eid"], tac_pred=T["tac_lon_pred"], lon_v7=T["lon_v7"])
    res["bias_structure"] = bias_structure(M, G, ["r7_ON", "ema30k_ON", "ema30k_OFF", "ema30k_vmaxzero", "r7_vmaxzero", "ema20k_ON", "raw20k_ON"], BC)
    res["lever_L1_speed_calibration"] = lever_calibration(PR, ra, {"ema30k_ON": T["os"], "r7_ON": T["ref"]}, G, dt, T["eid"],
                                                          parts["v0bin"][0], parts["dvclass"][0], BC)
    ctl["L1_retime_beta0_identity_PASS"] = bool(res["lever_L1_speed_calibration"]["known_value_retime_beta0_max_abs_m"] < 1e-4)
    ctl["ALL_PASS"] = bool(ctl["ALL_PASS"] and ctl["L1_retime_beta0_identity_PASS"])
    print(f"[{time.time() - t0:6.1f}s] bias structure + lever L1 done", flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    if not ctl["ALL_PASS"]:
        with open(OUT / "CONTROL_FAILED.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=str)
        print("CONTROL FAILED", json.dumps({"X": ctl["X_PASS"], "R": ctl["R_PASS"], "Z": ctl["Z_PASS"]}))
        return 3
    with open(OUT / "S6_LON_DECOMP.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    with open(OUT / "S6_LON_DECOMP.md", "w", encoding="utf-8") as f:
        f.write(render_md(res))
        va = OUT / "VERDICT_ANALYST.md"          # the analyst-authored verdict (written from this JSON), re-attached on every re-run
        if va.exists():
            f.write(va.read_text(encoding="utf-8"))
    print("WROTE", OUT / "S6_LON_DECOMP.json", OUT / "S6_LON_DECOMP.md", f"{res['wall_s']}s")
    return 0


# ------------------------------------------------------------------------------------------ markdown
def f4(x, dp=4):
    return "n/a" if x is None else f"{x:+.{dp}f}"


def cell(v, which="contrib", dp=4):
    if v.get(which) is None:
        return "n/a"
    lo, hi = v.get(f"{which}_lo" if which != "mean_delta" else "mean_lo"), v.get(f"{which}_hi" if which != "mean_delta" else "mean_hi")
    sep = ""
    if which == "contrib" and v.get("contrib_separated"):
        sep = " **SEP**"
    if which == "mean_delta" and v.get("mean_separated"):
        sep = " **SEP**"
    return f"{v[which]:+.{dp}f} [{f4(lo, dp)}, {f4(hi, dp)}]{sep}"


def share_cell(v):
    if v.get("share") is None:
        return "n/a"
    return f"{100 * v['share']:.0f} % [{100 * v['share_lo']:.0f}, {100 * v['share_hi']:.0f}]"


def table_primary(ent, pname, order_override=None) -> list:
    pm = ent["partitions"][pname]
    lv = pm["_levels"]
    L = ["| bin | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s share [CI] | abs-along@6s bin-mean delta [CI] m | speed-MAE share [CI] | "
         "signed along 6 s: refcv8 / refcv7 / delta [CI] m | refcv8 too slow (% win) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = order_override or list(pm["ade_m"]["bins"].keys())
    for b in order:
        a = pm["ade_m"]["bins"].get(b, {})
        if not a.get("n_windows"):
            continue
        al, sp, sg = pm["LON_along_abs_final_m"]["bins"][b], pm["LON_speed_mae_mps"]["bins"][b], pm["along_signed_final_m"]["bins"][b]
        L.append(f"| {b} | {a['n_windows']} / {a['n_episodes']} | {100 * a['frac_windows']:.1f} | {cell(a)} | {share_cell(a)} | {cell(a, 'mean_delta')} | "
                 f"{share_cell(al)} | {cell(al, 'mean_delta', 3)} | {share_cell(sp)} | {lv[b]['new_along_signed_mean']:+.2f} / {lv[b]['ref_along_signed_mean']:+.2f} / "
                 f"{cell(sg, 'mean_delta', 3)} | {100 * lv[b]['frac_new_slow']:.0f} |")
    t = pm["ade_m"]["total"]
    L.append(f"| **all** | {t['n_windows']} / {t['n_episodes']} | 100 | {t['point']:+.4f} [{f4(t['lo'])}, {f4(t['hi'])}] | 100 % | | | | | | |")
    r50 = pm["ade_m"].get("bins_reaching_50pct")
    if r50:
        L.append(f"\nADE: bins reaching >= 50 % of the delta (largest share first): **{', '.join(r50['bins'])}** (cumulative point share {100 * r50['cum_share']:.0f} %). "
                 f"|along| 6 s: **{', '.join(pm['LON_along_abs_final_m']['bins_reaching_50pct']['bins'])}** ({100 * pm['LON_along_abs_final_m']['bins_reaching_50pct']['cum_share']:.0f} %).")
    return L


def compact(ent, pname, metric="ade_m") -> str:
    pm = ent["partitions"][pname][metric]
    t = pm["total"]
    parts = [f"total {t['point']:+.4f} [{f4(t['lo'])}, {f4(t['hi'])}]"]
    for b, v in pm["bins"].items():
        if v.get("n_windows"):
            parts.append(f"{b}: {v['contrib']:+.4f}{' SEP' if v['contrib_separated'] else ''} ({share_cell(v)})")
    return "; ".join(parts)


def render_md(R: dict) -> str:
    C = R["contrasts"]
    ctl = R["controls"]
    L = ["# S6 (0-6 s) longitudinal decomposition: refcv8 EMA-30k vs refcv7-50,400 -- where the regression lives", "",
         f"* tier: **{R['tier']}**",
         f"* estimator: {R['estimator']}",
         f"* surface: {R['surface']}; n **{R['n_windows']} windows / {R['n_episodes']} episodes** (inference seed 0 unless marked); single TRAINING seed per arm.",
         "* arms: refcv8 `os` (ceiling filter ACTING, as banked) = **ON**; `filter_off.traj[:, slots]` (same DDIM draw, filter off) = **OFF** = the matched instrument; "
         "`vmax_zero.traj[:, slots]` = the max-speed (road-type) label WITHHELD; refcv7-50,400 = `b_refcv7_50400_s{seed}` (its filter is inert on S6, re-verified below).",
         "* **contribution** = sum of the bin's per-window deltas / N (adds up to the total mean delta, metric units); **share** = contribution / total; "
         "**enrichment** = share / window fraction. CIs: joint episode-cluster draws (B 2000), percentile 95 %. SEP = CI excludes 0.",
         "* signed along-track at 6 s = P_x(6 s) - G_x(6 s) in the t0 ego frame: **negative = the plan ends SHORT of GT (too slow)**, positive = overshoots (too fast).",
         "* privileged inputs: the per-frame nav and the road-type speed label are ego-future derived (optimistic by construction).",
         f"* artifacts: trees {', '.join(R['trees'].values())}; refcv7 {R['refcv7_root']}; wall {R['wall_s']} s.", "",
         "## Controls", "",
         f"* **Z (known-zero)**: refcv7 vs itself and refcv8 vs itself read exactly 0.0 (contribution, CI, bin mean) in every bin of every partition and metric: **{ctl['Z_PASS']}**.",
         f"* **R (reproduction)**: all-window deltas re-read through `taniteval.ci` equal the banked PAIRED_READ S6 cells (ON: 30k EMA s0+s1, 20k EMA, 20k RAW; OFF: NR_MATCHED_INSTRUMENT) "
         f"field-for-field, and the joint-draw total equals ci's at its rounding: **{ctl['R_PASS']}**.",
         f"* **X (independent derivations / identity)**: **{ctl['X_PASS']}** -- numpy |signed along| vs components max abs "
         f"{max(v['along_abs'] for v in ctl['X']['numpy_vs_components'].values()):.1e}; numpy ADE vs components {max(v['ade'] for v in ctl['X']['numpy_vs_components'].values()):.1e}; "
         f"S6 `os` vs extras `nav_true.traj[:, slots]` {ctl['X']['ema30k_chk']['os_vs_extras_nav_true_slots_max_abs']}; refcv7 S6 arm vs its extras "
         f"{ctl['X']['r7_extras']['r7_S6_ref_vs_extras_nav_true_slots_max_abs']}; refcv7 filter changes selection on {ctl['X']['r7_extras']['r7_filter_changes_selection_S6']} S6 windows; "
         f"refcv7 vs refcv8 speed label max abs {ctl['X']['r7_extras']['r7_vs_r8_vmax_label_max_abs']}; GT end-point world vs ego max abs "
         f"{ctl['X']['ema30k_chk']['gt_end_distance_world_vs_ego_max_abs_m']:.1e} m; window keys + GT identical across all trees and seed 1.",
         f"* **MEASURED, not a control failure -- the two arms consume DIFFERENT speed labels**: refcv7-50,400's `v_max_ms` differs from refcv8's on "
         f"{ctl['X']['r7_extras']['r7_vs_r8_vmax_label_n_windows_differ']} / {R['n_windows']} S6 windows (max abs {ctl['X']['r7_extras']['r7_vs_r8_vmax_label_max_abs']:.2f} m/s); refcv7: "
         f"{ctl['X']['r7_extras']['r7_vmax_n_distinct_values']} distinct values, constant within every clip = {ctl['X']['r7_extras']['r7_vmax_constant_within_every_clip']}, "
         f"mean (per-clip label minus the clip's max 6-s GT speed) {ctl['X']['r7_extras']['r7_vmax_minus_clip_max_gt_speed_6s_mean']:+.2f} m/s; refcv8: "
         f"{ctl['X']['r7_extras']['r8_vmax_n_distinct_values']} values (the 4-class road-type label).",
         "* **P (random partition)**: v0-bin labels permuted across windows; ADE shares vs window fractions: " +
         "; ".join(f"{b} {share_cell(v)} vs {100 * v['frac_windows']:.0f} %" for b, v in ctl["P_random_partition_C0"]["ade_m"]["bins"].items()), ""]
    c0 = C["C0"]
    L += ["## C0 -- THE TARGET: refcv8 EMA-30k ON minus refcv7-50,400, seed 0", ""]
    for pname, title in (("v0bin", "(a) GT initial speed"), ("dvclass", "(a) GT speed change over 6 s"), ("sign_2x2", "(b) sign of the along-track error at 6 s (new = refcv8, ref = refcv7; OUTCOME-defined)"),
                         ("ceiling", "(c) does the max-speed ceiling filter act (refcv8 EMA-30k)"), ("vmax_label", "(c) road-type max-speed label class"),
                         ("label_vs_gt", "(c) GT max speed over 6 s relative to the label"),
                         ("label_gap", "(c) speed-label INFORMATION gap: refcv8 4-class value minus refcv7's per-clip continuous value"),
                         ("lon_v7", "GT tactical longitudinal label (lon_v7)")):
        L += [f"### {title}", ""] + table_primary(c0, pname) + [""]
    cross = c0["partitions"]["v0_x_dv"]["ade_m"]["bins"]
    top = sorted([(b, v) for b, v in cross.items() if v.get("n_windows")], key=lambda t: -t[1]["contrib"])
    L += ["### (a) GT initial speed x speed change: the 8 largest ADE contributions (C0)", "",
          "| cell | n win / ep | win % | ADE contrib [CI] m | ADE share [CI] | ADE bin-mean delta [CI] m | abs-along@6s contrib [CI] m | signed along: refcv8 / refcv7 m |", "|---|---|---|---|---|---|---|---|"]
    lvx = c0["partitions"]["v0_x_dv"]["_levels"]
    for b, v in top[:8]:
        al = c0["partitions"]["v0_x_dv"]["LON_along_abs_final_m"]["bins"][b]
        L.append(f"| {b} | {v['n_windows']} / {v['n_episodes']} | {100 * v['frac_windows']:.1f} | {cell(v)} | {share_cell(v)} | {cell(v, 'mean_delta')} | {cell(al)} | "
                 f"{lvx[b]['new_along_signed_mean']:+.2f} / {lvx[b]['ref_along_signed_mean']:+.2f} |")
    L += ["", "## The same decomposition on the other contrasts (ADE contribution per bin, share in brackets)", ""]
    for cid in ("C0s1", "C1", "C2", "C3", "C3b", "C3c", "C4", "C5", "C6", "C7"):
        e = C[cid]
        L += [f"### {cid}: {e['desc']}", ""]
        for pname in ("v0bin", "dvclass", "label_gap", "label_vs_gt", "ceiling"):
            if pname in e["partitions"]:
                L.append(f"* **{pname}** ADE: {compact(e, pname)}")
                L.append(f"  * |along| 6 s: {compact(e, pname, 'LON_along_abs_final_m')}")
                L.append(f"  * signed along 6 s (bin mean delta): " + "; ".join(
                    f"{b} {cell(v, 'mean_delta', 3)}" for b, v in e["partitions"][pname]["along_signed_final_m"]["bins"].items() if v.get("n_windows")))
        L.append("")
    lc = R["lon_decision_confusion"]
    L += ["## Plan-derived longitudinal decision vs GT (decel / cruise / accel from the last-second speed minus v0, +-1.5 m/s; all windows, seed 0)", "",
          f"GT class counts: {lc['gt_class_counts']}", "",
          "| arm | accuracy | GT decel -> plan decel/cruise/accel | GT cruise -> plan decel/cruise/accel | GT accel -> plan decel/cruise/accel |", "|---|---|---|---|---|"]
    for a, v in lc["arms"].items():
        cm = v["confusion_rows_gt_cols_plan"]
        L.append(f"| {a} | {v['accuracy']:.3f} | " + " | ".join("/".join(str(cm[g][p]) for p in LONCLS) for g in LONCLS) + " |")
    fcl = lambda c: f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]{' SEP' if c['separated'] else ''} (refcv8 {c['mean_new']:.3f} vs refcv7 {c['mean_ref']:.3f}; n {c['n_windows']}/{c['n_episodes']})"  # noqa: E731
    L += ["", "Paired refcv8 EMA-30k minus refcv7-50,400 (per-window indicators, episode-cluster bootstrap): " +
          "; ".join(f"**{k}** {fcl(c)}" for k, c in lc["paired"].items()), ""]
    if "refcv8_tac_lon_pred_by_cell" in lc:
        L += ["refcv8's tactical lon head argmax (`nav_true.tacv6_lon_logits`) per decision cell (counts):", ""]
        for c, v in lc["refcv8_tac_lon_pred_by_cell"].items():
            L.append(f"* {c} (n {v['n']}): " + ", ".join(f"{k} {x}" for k, x in v.items() if k != "n" and x))
        L += ["", "refcv8's tactical lon head argmax given the GT `lon_v7` label (rows = label; counts):", ""]
        for g, v in lc["refcv8_tac_lon_pred_given_label"].items():
            L.append(f"* label {g} (n {v['n']}): " + ", ".join(f"{k} {x}" for k, x in v.items() if k != "n" and x))
        L.append("")
    for pname, title in (("plan_dec_new_ref", "C0 by plan-derived decision of refcv8 (new) and refcv7 (ref) -- OUTCOME-defined"),
                         ("r8_tac_lon_pred", "C0 by refcv8's tactical lon head argmax (`nav_true.tacv6_lon_logits`) -- OUTCOME-defined"),
                         ("r8_tac_lon_correct", "C0 by whether refcv8's tactical lon head matches the GT lon_v7 label")):
        if pname in C["C0"]["partitions"]:
            L += [f"### {title}", ""] + table_primary(C["C0"], pname) + [""]
    bs = R["bias_structure"]
    L += ["## Bias structure: OFFSET or COMPRESSION? (all 3,668 windows, seed 0)", "",
          "Plan x(6 s) regressed on GT x(6 s) (OLS; slope < 1 = the speed profile is compressed toward the mean; slope CI and slope minus refcv7 on the same draws).", "",
          "| arm | mean signed along 6 s [CI] m | windows too slow | mean signed speed err m/s | OLS slope [CI] | intercept m | slope minus refcv7 [CI] |", "|---|---|---|---|---|---|---|"]
    for a, v in bs.items():
        smr = v.get("slope_minus_r7")
        smr_s = "" if smr is None else f"{smr['point']:+.4f} [{f4(smr['ci'][0])}, {f4(smr['ci'][1])}]"
        L.append(f"| {a} | {v['mean_signed_along_6s_m']:+.3f} [{f4(v['ci'][0], 3)}, {f4(v['ci'][1], 3)}] | {100 * v['frac_windows_slow']:.0f} % | {v['mean_signed_speed_err_mps']:+.3f} | "
                 f"{v['ols_slope']:.4f} [{v['ols_slope_ci'][0]:.4f}, {v['ols_slope_ci'][1]:.4f}] | {v['ols_intercept_m']:+.3f} | {smr_s} |")
    lv = R["lever_L1_speed_calibration"]
    L += ["", "## Lever L1 RUN: a post-hoc uniform speed offset (zero GPU, zero training) -- DIAGNOSTIC, fit on eval episodes", "",
          f"* each waypoint moved along the plan's own polyline by beta*t (beta m/s, grid {lv['beta_grid']}); {lv['folds']} 2-fold cross-fit, criterion {lv['criterion']}; "
          f"the scored fold is never tuned on. Known-value control: beta = 0 reproduces the plan, max abs {lv['known_value_retime_beta0_max_abs_m']:.1e} m.",
          "* applied to BOTH arms (a calibrated refcv8 against a raw refcv7 is not a fair comparison); per-v0-bin uses only the measured v0 at t0 (admissible at inference).",
          f"* betas chosen: refcv8 {lv['arms']['ema30k_ON']}; refcv7 {lv['arms']['r7_ON']}", "",
          "| mode | metric | refcv8 cal - refcv7 RAW [CI] | refcv8 cal - refcv7 cal [CI] | refcv8 cal - refcv8 raw [CI] | refcv7 cal - refcv7 raw [CI] |", "|---|---|---|---|---|---|"]
    fc = lambda c: f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]{' SEP' if c['separated'] else ''}"  # noqa: E731
    for mode in ("global", "per_v0bin"):
        for k, lab, _u in ERR_METRICS:
            r = lv["rows"][f"{mode}:{k}"]
            L.append(f"| {mode} | {lab} | {fc(r['r8cal_minus_r7raw'])} | {fc(r['r8cal_minus_r7cal'])} | {fc(r['r8cal_minus_r8raw'])} | {fc(r['r7cal_minus_r7raw'])} |")
    L.append(f"\nreference (uncalibrated): refcv8 - refcv7 ADE {fc(lv['rows']['raw:ade_m_r8raw_minus_r7raw'])}.")
    for mode in ("global", "per_v0bin"):
        dd = lv["rows"][f"{mode}:ade_by_dvclass_r8cal_minus_r7cal"]
        L.append(f"* {mode}, refcv8 cal - refcv7 cal ADE by dvclass: total {dd['total']['point']:+.4f} [{f4(dd['total']['lo'])}, {f4(dd['total']['hi'])}]; " +
                 "; ".join(f"{b} {v['contrib']:+.4f}{' SEP' if v['contrib_separated'] else ''}" for b, v in dd["bins"].items() if v.get("n_windows")))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
