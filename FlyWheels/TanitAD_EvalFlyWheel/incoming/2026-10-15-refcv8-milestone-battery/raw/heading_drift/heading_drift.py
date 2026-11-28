"""heading_drift.py -- WHERE is refcv8's small, growing LATERAL heading-error regression?  CPU only, existing per-window dumps.

    python heading_drift.py --new-root E:/refcv8_eval_kit/battery/step10500ema_unc --label-new EMA-10.5k \
        --earlier EMA-5k=C:/Users/Admin/refcv8_eval_archive/battery/step5000ema_unc \
        --out-dir <dir> --tag HEADING_DRIFT_step10500ema

    # re-run on the step-15,000 dumps (the earlier points are repeatable, any number of them):
    python heading_drift.py --new-root E:/refcv8_eval_kit/battery/step15000ema_unc --label-new EMA-15k \
        --earlier EMA-10.5k=E:/refcv8_eval_kit/battery/step10500ema_unc EMA-5k=C:/Users/Admin/refcv8_eval_archive/battery/step5000ema_unc \
        --out-dir <dir> --tag HEADING_DRIFT_step15000ema

Reads the battery PANELS (`panel_s{s}` = S2, instants 0.5..2.0 s dt 0.5;  `s6_s{s}` = S6, instants 1..6 s dt 1.0) and their
`decisions/` sidecars, exactly as `code/pair_read_r8.py` does.  The comparison arm is the `b_refcv7_50400_s{s}` baseline that the
battery embedded in every panel (same windows, same inference seed, GT bit-identical -- re-asserted here).

METRIC.  `LAT_heading_mae_deg` = per window, the mean over path segments valid on BOTH the plan and the GT of
|wrap(heading_plan - heading_GT)|, in degrees (heading = atan2 of the chord between consecutive waypoints, origin prepended;
a segment is valid when its chord exceeds MIN_DS_MPS*dt).  Re-implemented here from `four_families._seq_geometry` (imported,
not re-derived) so the SIGNED per-segment error is available; the overall row is asserted against the battery's own
PAIRED_READ json (a control that must reproduce to the 4 dp it prints).

ESTIMATOR / QUESTION.  `taniteval.ci.paired_episode_cluster_bootstrap` (B 2000, seed 0, 95 % percentile), new minus ref, windows
finite on both sides.  It answers "another draw of EPISODES, models FIXED".  It does NOT answer "another TRAINING run" (a single
training run cannot: CLAUDE.md H-ESTIM-SEED-1) and only the s0-vs-s1 floor below speaks to "another INFERENCE run".
Heterogeneity ("is this bin worse than the other bins?") uses a vectorised episode-cluster bootstrap of (delta_bin - delta_rest),
reported at 95 % and at a Bonferroni level over every bin tested.

BINS are defined from model-independent quantities ONLY (GT path, measured v0, the clip's nav token) and their edges are fixed
constants below -- chosen from the GT / v0 distributions before any delta was looked at, never tuned on the deltas.

Writes <out-dir>/<tag>.json and <tag>.md.  NEVER writes a raw clip id (episodes are referred to by panel file index only).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

# ------------------------------------------------------------------------------------------------ constants
N_BOOT = 2000
SEED = 0
TURN_EDGES = (0.02, 0.06, 0.15)          # rad/s, GT mean |yaw rate| over valid chord pairs
TURN_LABELS = ("straight <0.02", "gentle 0.02-0.06", "turn 0.06-0.15", "sharp >=0.15", "undef (GT crawling)")
SPEED_EDGES = (2.0, 5.0, 10.0, 15.0, 25.0)   # m/s, measured v0
SPEED_LABELS = ("v0 <2", "v0 2-5", "v0 5-10", "v0 10-15", "v0 15-25", "v0 >=25")
SPEED_GROUP_EDGES = (5.0, 15.0)
SPEED_GROUP_LABELS = ("v0 <5", "v0 5-15", "v0 >=15")
NAV_LABELS = {0: "nav=follow", 1: "nav=left", 2: "nav=right"}   # sign of GT lateral displacement per code checked in the run
LOW_N_WIN, LOW_N_EP = 100, 15
PAIRED_READ_KEY = "LAT_heading_mae_deg"
EXTRA_ARMS = ("os_navzero", "os_navshuf", "os_navflip")   # the battery's nav-intervention arms (S2 and S6 panels carry them)
DEC_FIELDS = ("nav_cmd", "sel_idx", "sel_idx_base", "a_star", "anchor_acc", "goal_dist_sel", "sel_agrees_oracle")


# ------------------------------------------------------------------------------------------------ loading
def setup(taniteval_root: str, stack_root: str):
    for p in (taniteval_root, stack_root):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    from taniteval import ci, four_families as ff
    return torch, ci, ff


def panel_dir(root: str, surf: str, seed: int) -> str:
    return os.path.join(root, ("panel_s%d" if surf == "S2" else "s6_s%d") % seed)


def load_panel(root: str, surf: str, seed: int, ref_label: str) -> dict:
    """One (tree, surface, inference seed): GT, v0, the `os` plan, the embedded refcv7 baseline, and the decisions sidecar,
    all in the panel's own row order.  Every npz is READ (counted); an unreadable file raises -- a gap never passes silently."""
    pd = panel_dir(root, surf, seed)
    man = json.load(open(os.path.join(pd, "manifest.json"), encoding="utf-8"))
    cid = {int(e["file_index"]): str(e["clip_id"]) for e in man["episodes"]}
    files = sorted(glob.glob(os.path.join(pd, "ep*.npz")))
    out = {"dir": pd, "dt": float(man["grid"]["dt_s"]), "instants_s": man["grid"].get("instants_s"),
           "model_step": (man.get("model") or {}).get("step"), "ckpt": (man.get("model") or {}).get("ckpt"),
           "n_files": len(files), "n_unreadable": 0, "n_dec_missing": 0}
    keep = {k: [] for k in ("g", "v0", "os", "ref", "ws", "eid")} | {k: [] for k in DEC_FIELDS} | {k: [] for k in EXTRA_ARMS}
    keys = []
    for fi, f in enumerate(files):
        file_index = int(os.path.basename(f)[2:5])
        try:
            z = np.load(f)
            names = set(z.files)
            if ref_label not in names:
                raise KeyError(f"{ref_label} absent from {os.path.basename(f)}")
            arr = {k: z[k] for k in ("g", "v0", "os", "ws", ref_label)}
        except Exception as e:                                       # noqa: BLE001
            out["n_unreadable"] += 1
            raise RuntimeError(f"UNREADABLE panel file {f}: {e}") from e
        n = arr["g"].shape[0]
        keep["g"].append(arr["g"]); keep["v0"].append(arr["v0"]); keep["os"].append(arr["os"])
        keep["ref"].append(arr[ref_label]); keep["ws"].append(arr["ws"]); keep["eid"].append(np.full(n, fi))
        for ea in EXTRA_ARMS:
            keep[ea].append(z[ea] if ea in names else np.full(arr["os"].shape, np.nan, dtype=np.float32))
        keys += [(cid[file_index], int(w)) for w in arr["ws"]]
        dp = os.path.join(pd, "decisions", os.path.basename(f))
        try:
            d = np.load(dp)
            if not np.array_equal(d["ws"], arr["ws"]):
                raise ValueError("decisions ws differ from panel ws")
            for k in DEC_FIELDS:
                keep[k].append(d[k].astype(np.float64) if k in d.files else np.full(n, np.nan))
        except Exception:                                            # noqa: BLE001
            out["n_dec_missing"] += 1
            for k in DEC_FIELDS:
                keep[k].append(np.full(n, np.nan))
    for k, v in keep.items():
        out[k] = np.concatenate(v)
    out["keys"] = keys
    out["n_win"] = len(keys)
    out["n_ep"] = int(len(set(out["eid"].tolist())))
    return out


def align(a: dict, b: dict) -> np.ndarray:
    """index into `b` of every row of `a` (SET equality asserted; GT bit-identical asserted)."""
    pos = {k: i for i, k in enumerate(b["keys"])}
    if len(pos) != len(b["keys"]) or set(a["keys"]) != set(pos):
        raise RuntimeError(f"window sets differ: {len(a['keys'])} vs {len(b['keys'])}")
    idx = np.asarray([pos[k] for k in a["keys"]])
    d = float(np.abs(a["g"].astype(np.float64) - b["g"][idx].astype(np.float64)).max())
    if d != 0.0:
        raise RuntimeError(f"GT differs between trees (max abs {d}): not a paired comparison")
    return idx


# ------------------------------------------------------------------------------------------------ geometry
def seg_signed_heading(ff, torch, P: np.ndarray, G: np.ndarray, dt: float) -> np.ndarray:
    """[N,K] signed wrap(heading_plan - heading_GT) in degrees; NaN where the segment is not valid on BOTH (= the battery's mask)."""
    pt, gt = torch.as_tensor(P).float(), torch.as_tensor(G).float()
    Pg, Gg = ff._seq_geometry(pt, dt), ff._seq_geometry(gt, dt)
    both = Pg["valid"] & Gg["valid"]
    dh = (Pg["heading"] - Gg["heading"] + math.pi) % (2 * math.pi) - math.pi
    out = (dh * 180.0 / math.pi).numpy().astype(np.float64)
    out[~both.numpy()] = np.nan
    return out


def window_mae(dh: np.ndarray) -> np.ndarray:
    """the battery's LAT_heading_mae_deg: per-window mean of |dh| over its valid segments, NaN if none."""
    a = np.abs(dh)
    n = np.isfinite(a).sum(1)
    s = np.where(np.isfinite(a), a, 0.0).sum(1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def window_signed(dh: np.ndarray) -> np.ndarray:
    n = np.isfinite(dh).sum(1)
    s = np.where(np.isfinite(dh), dh, 0.0).sum(1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def gt_bins(ff, torch, G, v0, nav, dt):
    """model-independent bin labels (ints) + the signed GT yaw rate used for the turn direction."""
    Gg = ff._seq_geometry(torch.as_tensor(G).float(), dt)
    yr, pv = Gg["yaw_rate"], Gg["pair_valid"]
    nv = pv.sum(1)
    sgn = torch.where(nv > 0, (yr * pv).sum(1) / nv.clamp_min(1), torch.zeros_like(yr[:, 0])).numpy().astype(np.float64)
    ab = torch.where(nv > 0, (yr.abs() * pv).sum(1) / nv.clamp_min(1), torch.full_like(yr[:, 0], float("nan"))).numpy().astype(np.float64)
    turn = np.full(len(ab), len(TURN_LABELS) - 1, dtype=int)
    fin = np.isfinite(ab)
    turn[fin] = np.digitize(ab[fin], TURN_EDGES)
    speed = np.digitize(v0.astype(np.float64), SPEED_EDGES)
    sgrp = np.digitize(v0.astype(np.float64), SPEED_GROUP_EDGES)
    navc = np.where(np.isfinite(nav), nav, -1).astype(int)
    return {"turn": turn, "speed": speed, "speed_grp": sgrp, "nav": navc, "gt_yaw_abs": ab, "gt_yaw_signed": sgn}


# ------------------------------------------------------------------------------------------------ statistics
def cell(ci, x_new, x_ref, eid, mask, n_boot=N_BOOT) -> dict:
    """new - ref over `mask` windows finite on both sides: ci.py's PAIRED episode-cluster bootstrap (B, seed 0)."""
    m = np.asarray(mask, bool) & np.isfinite(x_new) & np.isfinite(x_ref)
    n = int(m.sum())
    if n == 0:
        return {"n_win": 0, "n_ep": 0, "status": "EMPTY"}
    r = ci.paired_episode_cluster_bootstrap(x_new[m], x_ref[m], eid[m], n_boot=n_boot, seed=SEED)
    row = {"n_win": n, "n_ep": int(r["n_episodes"]), "mean_new": float(x_new[m].mean()), "mean_ref": float(x_ref[m].mean()),
           "delta": float((x_new[m] - x_ref[m]).mean()), "lo": float(r["lo"]), "hi": float(r["hi"]),
           "separated": bool(r["separated"]), "p_delta_gt0": float(r.get("p_delta_gt0", float("nan")))}
    if n < LOW_N_WIN or row["n_ep"] < LOW_N_EP:
        row["low_n"] = True
    return row


def interaction(delta, eid, labels: dict, n_boot=N_BOOT, n_tests=1) -> dict:
    """per bin: (delta_bin - delta_rest) with a vectorised episode-cluster bootstrap, 95 % and Bonferroni(n_tests) percentile CIs.
    `delta` = per-window paired difference (NaN windows dropped).  labels = {bin_name: bool mask}."""
    m = np.isfinite(delta)
    eids = np.asarray([str(e) for e in eid[m]])
    uniq, inv = np.unique(eids, return_inverse=True)
    E = len(uniq)
    d = delta[m]
    rng = np.random.default_rng(SEED + 17)
    pick = rng.integers(0, E, size=(n_boot, E))
    W = np.zeros((n_boot, E))
    for b in range(n_boot):
        W[b] = np.bincount(pick[b], minlength=E)
    Stot = np.bincount(inv, weights=d, minlength=E)
    Ntot = np.bincount(inv, minlength=E).astype(float)
    out = {}
    a_lo, a_hi = 2.5, 97.5
    ab = 100.0 * 0.05 / n_tests / 2.0
    for name, mk in labels.items():
        mk = np.asarray(mk, bool)[m]
        nb = int(mk.sum())
        if nb == 0 or nb == len(d):
            out[name] = {"n_win": nb, "status": "EMPTY-OR-ALL"}
            continue
        S = np.bincount(inv[mk], weights=d[mk], minlength=E)
        N = np.bincount(inv[mk], minlength=E).astype(float)
        num_b, den_b = W @ S, W @ N
        num_r, den_r = W @ (Stot - S), W @ (Ntot - N)
        with np.errstate(invalid="ignore", divide="ignore"):
            stat = num_b / den_b - num_r / den_r
        stat = stat[np.isfinite(stat)]
        pt = float(d[mk].mean() - d[~mk].mean())
        lo, hi = np.percentile(stat, [a_lo, a_hi])
        blo, bhi = np.percentile(stat, [ab, 100 - ab])
        out[name] = {"n_win": nb, "excess_vs_rest": pt, "lo": float(lo), "hi": float(hi), "bonf_lo": float(blo), "bonf_hi": float(bhi),
                     "separated_95": bool(lo > 0 or hi < 0), "separated_bonferroni": bool(blo > 0 or bhi < 0),
                     "n_tests_bonferroni": n_tests}
    return out


def shares(delta, labels: dict) -> dict:
    m = np.isfinite(delta)
    N = int(m.sum())
    tot = float(delta[m].sum())
    out = {}
    for name, mk in labels.items():
        k = np.asarray(mk, bool) & m
        out[name] = {"window_share": float(k.sum() / N) if N else None,
                     "contrib_to_overall_delta": float(delta[k].sum() / N) if N else None,
                     "contrib_share": (float(delta[k].sum() / tot) if tot != 0 else None)}
    return out


def pooled_stats(a, b) -> dict:
    """paired-valid SEGMENTS pooled: signed bias, spread (SD), MAE, RMS for each arm and the L2 / L1 decomposition of new-minus-ref."""
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0:
        return {"n_seg": 0}
    sh = float((a - b).mean())
    st = {"n_seg": int(a.size),
          "bias_new": float(a.mean()), "bias_ref": float(b.mean()), "sd_new": float(a.std()), "sd_ref": float(b.std()),
          "mae_new": float(np.abs(a).mean()), "mae_ref": float(np.abs(b).mean()),
          "rms_new": float(np.sqrt((a ** 2).mean())), "rms_ref": float(np.sqrt((b ** 2).mean())),
          "shift_mean_new_minus_ref": sh, "sd_of_diff": float((a - b).std()),
          "corr_new_ref": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None}
    st["d_mae"] = st["mae_new"] - st["mae_ref"]
    st["d_mae_if_shift_removed"] = float(np.abs(a - sh).mean() - st["mae_ref"])
    st["d_mse"] = float((a ** 2).mean() - (b ** 2).mean())
    st["d_mse_from_bias"] = st["bias_new"] ** 2 - st["bias_ref"] ** 2
    st["d_mse_from_spread"] = st["sd_new"] ** 2 - st["sd_ref"] ** 2
    return st


# ------------------------------------------------------------------------------------------------ one comparison
def label_masks(B: dict) -> dict:
    """{dim: {bin_name: mask}} for the model-independent bins of one surface."""
    out = {"turn": {TURN_LABELS[i]: B["turn"] == i for i in range(len(TURN_LABELS))},
           "speed": {SPEED_LABELS[i]: B["speed"] == i for i in range(len(SPEED_LABELS))},
           "nav": {NAV_LABELS[i]: B["nav"] == i for i in NAV_LABELS}}
    return out


def compare(ci, name: str, mae_new, mae_ref, dh_new, dh_ref, eid, B: dict, n_boot: int) -> dict:
    """The full battery of splits for one (new, ref) pair on one surface.  mae_* [N]; dh_* [N,K] signed (NaN invalid)."""
    masks = label_masks(B)
    allm = np.ones(len(mae_new), bool)
    res = {"name": name, "overall": cell(ci, mae_new, mae_ref, eid, allm, n_boot), "dims": {}, "interaction": {}}
    delta = mae_new - mae_ref
    flat = {f"{d}|{n}": mk for d, bins in masks.items() for n, mk in bins.items()}
    n_tests = len(flat)
    inter = interaction(delta, eid, flat, n_boot, n_tests)
    for d, bins in masks.items():
        sh = shares(delta, bins)
        rows = {}
        for n, mk in bins.items():
            r = cell(ci, mae_new, mae_ref, eid, mk, n_boot)
            r.update(sh[n])
            r["vs_rest"] = inter[f"{d}|{n}"]
            rows[n] = r
        res["dims"][d] = rows
    # 2-D cross: turn x speed group
    cross = {}
    for ti, tn in enumerate(TURN_LABELS):
        for si, sn in enumerate(SPEED_GROUP_LABELS):
            mk = (B["turn"] == ti) & (B["speed_grp"] == si)
            cross[f"{tn} | {sn}"] = cell(ci, mae_new, mae_ref, eid, mk, n_boot)
    res["cross_turn_x_speed"] = cross
    # horizon instant (segment) x {all, straight, curved}
    K = dh_new.shape[1]
    straight = B["turn"] == 0
    curved = (B["turn"] >= 1) & (B["turn"] <= 3)
    inst = {}
    for k in range(K):
        a, b = np.abs(dh_new[:, k]), np.abs(dh_ref[:, k])
        for gname, gm in (("all", allm), ("straight", straight), ("curved", curved)):
            inst[f"seg{k + 1}|{gname}"] = {"abs": cell(ci, a, b, eid, gm, n_boot), "signed_shift": cell(ci, dh_new[:, k], dh_ref[:, k], eid, gm, n_boot)}
    res["by_instant"] = inst
    # signed error: bias vs spread, per bin (paired-valid segments pooled; turn-aligned sign inside turning bins)
    sw = np.where(B["turn"] >= 1, np.sign(B["gt_yaw_signed"]), 1.0)
    sw = np.where(B["turn"] == 4, 1.0, sw)
    sw = np.where(B["turn"] == 0, 1.0, sw)
    sw = np.where(sw == 0, 1.0, sw)
    sb_new, sb_ref = window_signed(np.where(np.isfinite(dh_ref), dh_new, np.nan)), window_signed(np.where(np.isfinite(dh_new), dh_ref, np.nan))
    signed = {}
    for d, bins in masks.items():
        for n, mk in bins.items():
            rows = {}
            for tag, mult in (("left_positive", np.ones_like(sw)), ("turn_aligned", sw)):
                A = (dh_new * mult[:, None])[mk]
                Bb = (dh_ref * mult[:, None])[mk]
                v = np.isfinite(A) & np.isfinite(Bb)
                st = pooled_stats(A[v], Bb[v])
                st["window_level_shift"] = cell(ci, sb_new * mult, sb_ref * mult, eid, mk, n_boot)
                rows[tag] = st
            signed[f"{d}|{n}"] = rows
    res["signed"] = signed
    # episode concentration
    m = np.isfinite(delta)
    ue, inv = np.unique(eid[m], return_inverse=True)
    sums = np.bincount(inv, weights=delta[m], minlength=len(ue))
    cnt = np.bincount(inv, minlength=len(ue))
    epm = sums / cnt
    order = np.argsort(-sums)
    tot = sums.sum()
    top5 = order[:5]
    keep = np.ones(len(ue), bool); keep[top5] = False
    res["episode_concentration"] = {
        "n_episodes": int(len(ue)), "frac_episodes_delta_gt0": float((epm > 0).mean()),
        "median_episode_mean_delta": float(np.median(epm)), "top5_episode_share_of_sum": float(sums[top5].sum() / tot) if tot != 0 else None,
        "top5_episode_file_index_and_mean_delta": [[int(ue[i]), float(epm[i]), int(cnt[i])] for i in top5],
        "overall_delta": float(delta[m].mean()),
        "delta_excluding_top5_episodes": float(sums[keep].sum() / cnt[keep].sum()),
    }
    return res



# ------------------------------------------------------------------------------------------------ extras (mechanism probes)
def shrinkage(ff, torch, dh_by_arm: dict, G, dt: float, eid, B: dict, groups: dict, n_boot: int, ref_lab: str) -> dict:
    """MULTIPLICATIVE test of the under-rotation: regress the turn-aligned heading error e on the turn-aligned GT chord heading h through the
    origin (slope = e.h / h.h; -0.10 means the plan turns 10 % less than GT at that point).  A constant offset would NOT scale with h.
    CI: episode-cluster bootstrap (vectorised) of each slope and of slope(arm) - slope(ref).  segments valid on every arm."""
    Gg = ff._seq_geometry(torch.as_tensor(G).float(), dt)
    hg = Gg["heading"].numpy().astype(np.float64) * 180.0 / math.pi
    sw = np.where((B["turn"] >= 1) & (B["turn"] <= 3), np.sign(B["gt_yaw_signed"]), 0.0)
    labs = list(dh_by_arm)
    valid = np.all([np.isfinite(dh_by_arm[l]) for l in labs], axis=0) & Gg["valid"].numpy()
    K = valid.shape[1]
    ue, inv = np.unique(np.asarray([str(e) for e in eid]), return_inverse=True)
    E = len(ue)
    rng = np.random.default_rng(SEED + 29)
    pick = rng.integers(0, E, size=(n_boot, E))
    W = np.stack([np.bincount(pick[b], minlength=E) for b in range(n_boot)]).astype(float)
    out = {}
    for gname, gm in groups.items():
        for sname, ks in (("last segment", [K - 1]), ("segments 2..K", list(range(1, K)))):
            m = np.zeros_like(valid)
            m[:, ks] = True
            m &= valid & gm[:, None] & (sw != 0)[:, None]
            h = hg * sw[:, None]
            if m.sum() < 30:
                continue
            ii = np.broadcast_to(inv[:, None], m.shape)[m]
            Sxx = np.bincount(ii, weights=(h ** 2)[m], minlength=E)
            sl, bs = {}, {}
            for l in labs:
                e = dh_by_arm[l] * sw[:, None]
                Sxy = np.bincount(ii, weights=(e * h)[m], minlength=E)
                sl[l] = float(Sxy.sum() / Sxx.sum())
                bs[l] = (W @ Sxy) / (W @ Sxx)
            row = {"n_seg": int(m.sum()), "n_win": int(m.any(1).sum()), "mean_gt_turn_aligned_heading_deg": float(h[m].mean()), "slope": {}, "slope_minus_ref": {}}
            for l in labs:
                lo, hi = np.percentile(bs[l], [2.5, 97.5])
                row["slope"][l] = {"slope": sl[l], "lo": float(lo), "hi": float(hi)}
                if l != ref_lab:
                    d = bs[l] - bs[ref_lab]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    row["slope_minus_ref"][l] = {"delta": sl[l] - sl[ref_lab], "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0)}
            out[f"{gname} | {sname}"] = row
    return out


def robust_bin_stats(delta, labels: dict) -> dict:
    """tail check: is a bin's mean delta carried by a few windows?  median, 10 % trimmed mean, sign split."""
    out = {}
    for n, mk in labels.items():
        d = delta[np.asarray(mk, bool) & np.isfinite(delta)]
        if d.size < 5:
            out[n] = {"n_win": int(d.size)}
            continue
        lo, hi = np.percentile(d, [10, 90])
        t = d[(d >= lo) & (d <= hi)]
        out[n] = {"n_win": int(d.size), "mean": float(d.mean()), "median": float(np.median(d)), "trimmed10_mean": float(t.mean()),
                  "frac_worse": float((d > 1e-9).mean()), "frac_better": float((d < -1e-9).mean()), "frac_tied": float((np.abs(d) <= 1e-9).mean()),
                  "sum_top1pct": float(np.sort(d)[::-1][:max(1, d.size // 100)].sum()), "sum_all": float(d.sum())}
    return out


def bias_profile(ci, dh_by_arm: dict, eid, B: dict, groups: dict, n_boot: int) -> dict:
    """TURN-ALIGNED signed heading error per horizon segment, per arm, on segments valid for ALL arms listed (apples to apples).
    + = plan heading rotated further INTO the GT turn than GT (over-rotation); - = under-rotation (plan lags / flattens the turn)."""
    sw = np.where((B["turn"] >= 1) & (B["turn"] <= 3), np.sign(B["gt_yaw_signed"]), 0.0)
    labs = list(dh_by_arm)
    valid = np.all([np.isfinite(dh_by_arm[l]) for l in labs], axis=0)
    out = {}
    for gname, gm in groups.items():
        K = valid.shape[1]
        rows = {}
        for k in range(K):
            m = gm & valid[:, k] & (sw != 0)
            if not m.any():
                continue
            rows[f"seg{k + 1}"] = {"n_win": int(m.sum()), **{f"bias_{l}": float((dh_by_arm[l][m, k] * sw[m]).mean()) for l in labs},
                                   **{f"mae_{l}": float(np.abs(dh_by_arm[l][m, k]).mean()) for l in labs}}
        out[gname] = rows
    return out


def top_episodes(delta, eid, B: dict, v0, k: int = 8) -> dict:
    m = np.isfinite(delta)
    ue, inv = np.unique(eid[m], return_inverse=True)
    sums = np.bincount(inv, weights=delta[m], minlength=len(ue))
    order = np.argsort(-sums)[:k]
    tot = float(sums.sum())
    rows = []
    for i in order:
        w = (eid == ue[i]) & m
        rows.append({"panel_file_index": int(ue[i]), "n_win": int(w.sum()), "sum_delta": float(sums[i]), "share_of_total": float(sums[i] / tot) if tot else None,
                     "mean_delta": float(delta[w].mean()), "frac_windows_sharp": float((B["turn"][w] == 3).mean()), "frac_windows_curved(0.02+)": float(((B["turn"][w] >= 1) & (B["turn"][w] <= 3)).mean()),
                     "mean_v0": float(v0[w].mean()), "nav_counts": {NAV_LABELS[c]: int((B["nav"][w] == c).sum()) for c in NAV_LABELS}})
    return {"total_sum_delta": tot, "n_episodes": int(len(ue)), "top": rows}


def nav_dependence(ci, ff, torch, Pn: dict, Pe: dict, dt: float, B: dict, eid, n_boot: int, newlab: str, earlab: str) -> dict:
    """does the model still USE nav for heading?  benefit(tree) = MAE(os_navzero) - MAE(os) (positive = nav helps heading), per bin;
    then benefit(new) - benefit(earlier).  Uses each tree's OWN intervention arms (no refcv7 sidecar exists in these panels)."""
    allm = np.ones(len(eid), bool)
    masks = {"overall": allm} | {f"{d}|{n}": mk for d, bins in label_masks(B).items() for n, mk in bins.items() if d != "nav"}
    out = {}
    mae = {}
    for tag, P in ((newlab, Pn), (earlab, Pe)):
        mae[tag] = {a: window_mae(seg_signed_heading(ff, torch, P[a], Pn["g"], dt)) for a in ("os",) + EXTRA_ARMS}
    for nm, mk in masks.items():
        row = {}
        for tag in (newlab, earlab):
            row[f"{tag}: MAE os / navzero / navshuf / navflip"] = [float(np.nanmean(np.where(mk, mae[tag][a], np.nan))) for a in ("os",) + EXTRA_ARMS]
            row[f"{tag}: benefit(navzero-os)"] = cell(ci, mae[tag]["os_navzero"], mae[tag]["os"], eid, mk, n_boot)
        bn = mae[newlab]["os_navzero"] - mae[newlab]["os"]
        be = mae[earlab]["os_navzero"] - mae[earlab]["os"]
        row["benefit_new_minus_earlier"] = cell(ci, bn, be, eid, mk, n_boot)
        out[nm] = row
    return out


def floor_check(sr: dict, newlab: str, earlier: list) -> dict:
    """per bin and per comparison: pooled 2-seed delta vs the inference floor (|s0-s1| of the two arms compared); same sign in both seeds;
    the programme's NOT-PROVEN rule: |delta| within 2x the floor => not proven.  INFERENCE floor only: a single training run cannot give a training floor."""
    out = {}
    fl = sr.get("inference_floor_s0_minus_s1", {})
    for cname, comp in sr.get("pooled_2seed", {}).items():
        a, b = cname.split(" - ")
        rows = {}
        names = ["overall"] + [f"{d}|{n}" for d, bins in comp["dims"].items() for n in bins]
        for nm in names:
            if nm == "overall":
                cell_p = comp["overall"]
                cs = [sr["seeds"][sd][cname]["overall"] for sd in sr["seeds"]]
            else:
                d, n = nm.split("|", 1)
                cell_p = comp["dims"][d][n]
                cs = [sr["seeds"][sd][cname]["dims"][d][n] for sd in sr["seeds"]]
            if "delta" not in cell_p or not all("delta" in c for c in cs):
                continue
            fa = abs((fl.get(a, {}).get(nm) or {}).get("delta", float("nan")))
            fb = abs((fl.get(b, {}).get(nm) or {}).get("delta", float("nan")))
            fmax = max([x for x in (fa, fb) if x == x] or [float("nan")])
            same = all(np.sign(c["delta"]) == np.sign(cs[0]["delta"]) for c in cs)
            ratio = abs(cell_p["delta"]) / fmax if fmax and fmax == fmax and fmax > 0 else None
            rows[nm] = {"delta_pooled": cell_p["delta"], "delta_by_seed": [c["delta"] for c in cs], "separated_by_seed": [c["separated"] for c in cs],
                        "separated_pooled": cell_p["separated"], "floor_a": fa, "floor_b": fb, "ratio_delta_to_max_floor": ratio, "same_sign_both_seeds": bool(same),
                        "verdict": ("CLEARS 2x INFERENCE FLOOR and separated (pooled and both seeds)" if (ratio and ratio > 2 and same and cell_p["separated"] and all(c["separated"] for c in cs)) else
                                    "CLEARS 2x floor + pooled-separated, not separated at each seed" if (ratio and ratio > 2 and same and cell_p["separated"]) else
                                    "NOT PROVEN (not separated and/or within 2x floor)")}
        out[cname] = rows
    return out


def run_surface(torch, ci, ff, surf: str, roots: list, seeds: list, ref_prefix: str, n_boot: int) -> dict:
    """roots[0] = (label, path) of the NEW tree; roots[1:] earlier trees.  Everything is aligned to the NEW tree's row order."""
    newlab, newroot = roots[0]
    out = {"surface": surf, "seeds": {}, "pooled_2seed": {}}
    per = {}                                                          # per[(label, seed)] -> dict of arrays aligned to new
    for s in seeds:
        refl = f"{ref_prefix}_s{s}"
        P0 = load_panel(newroot, surf, s, refl)
        dt = P0["dt"]
        if "meta" not in out:
            out["meta"] = {"dt_s": dt, "instants_s": P0["instants_s"], "n_windows": P0["n_win"], "n_episodes": P0["n_ep"],
                           "n_files_read": P0["n_files"] if "n_files_read" not in out else out["n_files_read"], "ref_label": refl}
        B = gt_bins(ff, torch, P0["g"], P0["v0"], P0["nav_cmd"], dt)
        if s == seeds[0]:
            out["bins_census"] = {d: {n: int(mk.sum()) for n, mk in b.items()} for d, b in label_masks(B).items()}
            out["bins_census_episodes"] = {d: {n: int(len(set(P0["eid"][mk].tolist()))) for n, mk in b.items()} for d, b in label_masks(B).items()}
        # nav sign control: mean GT lateral displacement at the last instant by nav code (documents the code->direction map)
        out.setdefault("nav_sign_check", {})[str(s)] = {NAV_LABELS[c]: float(P0["g"][B["nav"] == c, -1, 1].mean()) for c in NAV_LABELS if (B["nav"] == c).any()}
        per[(newlab, s)] = {"mae": window_mae(seg_signed_heading(ff, torch, P0["os"], P0["g"], dt)), "dh": seg_signed_heading(ff, torch, P0["os"], P0["g"], dt),
                            "P": P0, "B": B, "eid": P0["eid"]}
        per[("refcv7-50400", s)] = {"dh": seg_signed_heading(ff, torch, P0["ref"], P0["g"], dt)}
        per[("refcv7-50400", s)]["mae"] = window_mae(per[("refcv7-50400", s)]["dh"])
        for lab, root in roots[1:]:
            P1 = load_panel(root, surf, s, refl)
            idx = align(P0, P1)
            dh1 = seg_signed_heading(ff, torch, P1["os"][idx], P0["g"], dt)
            per[(lab, s)] = {"mae": window_mae(dh1), "dh": dh1, "P": {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (len(idx),) else v) for k, v in P1.items() if k != "keys"}}
            # baseline identity control: the embedded refcv7 arm must be bit-identical across trees
            out.setdefault("baseline_identity_control", {})[f"{lab}|s{s}"] = float(np.abs(P0["ref"].astype(np.float64) - P1["ref"][idx].astype(np.float64)).max())
    eid = per[(newlab, seeds[0])]["eid"]
    B = per[(newlab, seeds[0])]["B"]
    # comparison catalogue.  C1: new-ref; C2_x: new-earlier_x; C3_x: earlier_x-ref
    def arrays(lab, s):
        return per[(lab, s)]["mae"], per[(lab, s)]["dh"]
    for s in seeds:
        comps = {}
        mn, dn = arrays(newlab, s); mr, dr = arrays("refcv7-50400", s)
        comps[f"{newlab} - refcv7-50400"] = compare(ci, f"{newlab} - refcv7-50400 (seed {s})", mn, mr, dn, dr, eid, B, n_boot)
        for lab, _ in roots[1:]:
            me, de = arrays(lab, s)
            comps[f"{newlab} - {lab}"] = compare(ci, f"{newlab} - {lab} (seed {s})", mn, me, dn, de, eid, B, n_boot)
            comps[f"{lab} - refcv7-50400"] = compare(ci, f"{lab} - refcv7-50400 (seed {s})", me, mr, de, dr, eid, B, n_boot)
        out["seeds"][str(s)] = comps
    # ---- 2-seed pooled (inference noise averaged; says nothing about training variance)
    if len(seeds) == 2:
        s0, s1 = seeds
        def avg(lab, key):
            return 0.5 * (per[(lab, s0)][key] + per[(lab, s1)][key])
        mnp, mrp = avg(newlab, "mae"), avg("refcv7-50400", "mae")
        dnp = 0.5 * (per[(newlab, s0)]["dh"] + per[(newlab, s1)]["dh"]); drp = 0.5 * (per[("refcv7-50400", s0)]["dh"] + per[("refcv7-50400", s1)]["dh"])
        pooled = {f"{newlab} - refcv7-50400": compare(ci, f"{newlab} - refcv7-50400 (2-seed mean)", mnp, mrp, dnp, drp, eid, B, n_boot)}
        for lab, _ in roots[1:]:
            mep = avg(lab, "mae"); dep = 0.5 * (per[(lab, s0)]["dh"] + per[(lab, s1)]["dh"])
            pooled[f"{newlab} - {lab}"] = compare(ci, f"{newlab} - {lab} (2-seed mean)", mnp, mep, dnp, dep, eid, B, n_boot)
            pooled[f"{lab} - refcv7-50400"] = compare(ci, f"{lab} - refcv7-50400 (2-seed mean)", mep, mrp, dep, drp, eid, B, n_boot)
        out["pooled_2seed"] = pooled
        # ---- inference floor per bin: same checkpoint, seed 0 vs seed 1 (paired, same windows)
        masks = label_masks(B)
        floors = {}
        for lab in [newlab] + [l for l, _ in roots[1:]] + ["refcv7-50400"]:
            fl = {"overall": cell(ci, per[(lab, s0)]["mae"], per[(lab, s1)]["mae"], eid, np.ones(len(eid), bool), n_boot)}
            for d, bins in masks.items():
                for n, mk in bins.items():
                    fl[f"{d}|{n}"] = cell(ci, per[(lab, s0)]["mae"], per[(lab, s1)]["mae"], eid, mk, n_boot)
            floors[lab] = fl
        out["inference_floor_s0_minus_s1"] = floors
        out["floor_check"] = floor_check(out, newlab, [l for l, _ in roots[1:]])
        # ---- mechanism probes
        allm = np.ones(len(eid), bool)
        curved_m = (B["turn"] >= 1) & (B["turn"] <= 3)
        groups = {"curved (GT |yaw| >= 0.02)": curved_m, "sharp (>= 0.15)": B["turn"] == 3,
                  "curved & v0 5-10": curved_m & (B["speed"] == 2), "sharp & v0 5-10": (B["turn"] == 3) & (B["speed"] == 2)}
        pdh = {newlab: dnp, "refcv7-50400": drp}
        pmae = {newlab: mnp, "refcv7-50400": mrp}
        for lab, _ in roots[1:]:
            pdh[lab] = 0.5 * (per[(lab, s0)]["dh"] + per[(lab, s1)]["dh"])
            pmae[lab] = avg(lab, "mae")
        ex = {"bias_profile_turn_aligned_2seed_mean_dh": bias_profile(ci, pdh, eid, B, groups, n_boot),
              "robust_by_bin": {}, "top_episodes": {}}
        for lab, _ in roots[1:]:
            for (x, y, nm) in ((newlab, "refcv7-50400", f"{newlab} - refcv7-50400"), (newlab, lab, f"{newlab} - {lab}"), (lab, "refcv7-50400", f"{lab} - refcv7-50400")):
                if nm in ex["robust_by_bin"]:
                    continue
                dl = pmae[x] - pmae[y]
                ex["robust_by_bin"][nm] = robust_bin_stats(dl, {"ALL": allm} | {f"{d}|{n}": mk for d, bins in label_masks(B).items() for n, mk in bins.items()})
                ex["top_episodes"][nm] = top_episodes(dl, eid, B, per[(newlab, s0)]["P"]["v0"])
        ex["curvature_shrinkage"] = shrinkage(ff, torch, pdh, per[(newlab, s0)]["P"]["g"], out["meta"]["dt_s"], eid, B, groups, n_boot, "refcv7-50400")
        # oracle-agreement split of the new-vs-earlier delta, per seed
        ex["oracle_agreement_split"] = {}
        for sd in seeds:
            Pn_, mn_ = per[(newlab, sd)]["P"], per[(newlab, sd)]["mae"]
            for lab, _ in roots[1:]:
                Pe_, me_ = per[(lab, sd)]["P"], per[(lab, sd)]["mae"]
                an, ae = Pn_["sel_agrees_oracle"] > 0.5, Pe_["sel_agrees_oracle"] > 0.5
                cats = {"both agree with oracle": an & ae, "only new agrees": an & ~ae, "only earlier agrees": ~an & ae, "neither agrees": ~an & ~ae}
                rows = {}
                for cn, cm in cats.items():
                    rows[cn] = {"all": cell(ci, mn_, me_, eid, cm, n_boot), "sharp": cell(ci, mn_, me_, eid, cm & (B["turn"] == 3), n_boot)}
                ex["oracle_agreement_split"][f"{newlab} vs {lab}|seed{sd}"] = rows
        # nav dependence of heading, per seed
        ex["nav_dependence"] = {}
        for sd in seeds:
            for lab, _ in roots[1:]:
                ex["nav_dependence"][f"{newlab} vs {lab}|seed{sd}"] = nav_dependence(ci, ff, torch, per[(newlab, sd)]["P"], per[(lab, sd)]["P"], per[(newlab, sd)]["P"]["dt"] if "dt" in per[(newlab, sd)]["P"] else out["meta"]["dt_s"], B, eid, n_boot, newlab, lab)
        out["extras"] = ex
    # ---- selection / goal-proxy diagnostics: only possible where a decisions sidecar exists (the refcv8 trees, not refcv7)
    sel = {}
    for s in seeds:
        mn, dn = arrays(newlab, s)
        Pn = per[(newlab, s)]["P"]
        for lab, _ in roots[1:]:
            Pe = per[(lab, s)]["P"]
            mae_e = per[(lab, s)]["mae"]
            flip = np.isfinite(Pn["sel_idx"]) & np.isfinite(Pe["sel_idx"]) & (Pn["sel_idx"] != Pe["sel_idx"])
            ok = np.isfinite(Pn["sel_idx"]) & np.isfinite(Pe["sel_idx"])
            r = {"n_with_sidecar": int(ok.sum()), "flip_rate": float(flip[ok].mean()) if ok.any() else None}
            if ok.any():
                r["flip_rate_by_bin"] = {f"{d}|{n}": (float(flip[mk & ok].mean()) if (mk & ok).any() else None) for d, bins in label_masks(B).items() for n, mk in bins.items()}
                dl = mn - mae_e
                r["delta_heading_flip_windows"] = cell(ci, mn, mae_e, eid, flip & ok, n_boot)
                r["delta_heading_nonflip_windows"] = cell(ci, mn, mae_e, eid, (~flip) & ok, n_boot)
                r["flip_vs_nonflip_interaction"] = interaction(dl, eid, {"flip": flip & ok}, n_boot, 1)["flip"]
                gd_n, gd_e = Pn["goal_dist_sel"], Pe["goal_dist_sel"]
                r["goal_dist_sel_new_minus_earlier"] = cell(ci, gd_n, gd_e, eid, ok, n_boot)
                r["graft_override_rate_new"] = float((Pn["sel_idx"] != Pn["sel_idx_base"])[ok].mean())
                r["graft_override_rate_earlier"] = float((Pe["sel_idx"] != Pe["sel_idx_base"])[ok].mean())
                r["anchor_acc_new_minus_earlier"] = cell(ci, Pn["anchor_acc"], Pe["anchor_acc"], eid, ok, n_boot)
                r["sel_agrees_oracle_new_minus_earlier"] = cell(ci, Pn["sel_agrees_oracle"], Pe["sel_agrees_oracle"], eid, ok, n_boot)
                vv = ok & np.isfinite(dl) & np.isfinite(gd_n) & np.isfinite(gd_e)
                if vv.sum() > 10:
                    r["corr_window_delta_heading_vs_delta_goal_dist"] = float(np.corrcoef(dl[vv], (gd_n - gd_e)[vv])[0, 1])
                    r["corr_window_delta_heading_vs_goal_dist_new"] = float(np.corrcoef(dl[vv], gd_n[vv])[0, 1])
            sel[f"{newlab} vs {lab}|seed{s}"] = r
    out["selection_diagnostics"] = sel
    # level (not difference) of the goal proxy by bin, per tree
    lv = {}
    for s in seeds:
        for lab, _ in [(newlab, None)] + roots[1:]:
            gd = per[(lab, s)]["P"]["goal_dist_sel"]
            f = np.isfinite(gd)
            lv[f"{lab}|seed{s}"] = {"mean_goal_dist_sel_m": float(gd[f].mean()) if f.any() else None, "n": int(f.sum()),
                                    "by_bin": {f"{d}|{n}": (float(gd[mk & f].mean()) if (mk & f).any() else None) for d, bins in label_masks(B).items() for n, mk in bins.items()}}
    out["goal_proxy_levels"] = lv
    return out


# ------------------------------------------------------------------------------------------------ control against the battery
def control_vs_paired_read(new_root: str, res_s2: dict, seeds: list, n_boot: int = N_BOOT) -> dict:
    """the overall S2 and S6 heading cells must reproduce PAIRED_READ_*.json (the battery's own number) -- else this tool is wrong."""
    cands = sorted(glob.glob(os.path.join(new_root, "PAIRED_READ_*.json")))
    if not cands:
        return {"status": "NO PAIRED_READ json found; control not run"}
    pr = json.load(open(cands[0], encoding="utf-8"))
    out = {"source": cands[0], "rows": {}}
    for surf, rr in res_s2.items():
        for s in seeds:
            ref = (((pr.get("surfaces") or {}).get(surf) or {}).get("seeds") or {}).get(str(s), {}).get("rows", {}).get(PAIRED_READ_KEY, {}).get("os_minus_refcv7_50400")
            key = [k for k in rr["seeds"][str(s)] if k.endswith("- refcv7-50400") and k.split(" - ")[0] == rr["_newlab"]][0]
            mine = rr["seeds"][str(s)][key]["overall"]
            if not ref:
                out["rows"][f"{surf}|s{s}"] = {"status": "battery row absent"}
                continue
            ok_pt = mine["n_win"] == ref["n_windows"] and abs(mine["delta"] - ref["delta"]) < 6e-5
            ok_ci = abs(mine["lo"] - ref["lo"]) < 6e-5 and abs(mine["hi"] - ref["hi"]) < 6e-5
            ok = ok_pt and (ok_ci if n_boot == N_BOOT else True)
            out["rows"][f"{surf}|s{s}"] = {"mine": {k: mine[k] for k in ("n_win", "delta", "lo", "hi")},
                                          "battery": {"n_windows": ref["n_windows"], "delta": ref["delta"], "lo": ref["lo"], "hi": ref["hi"]},
                                          "point_and_n_reproduced": bool(ok_pt), "ci_reproduced": bool(ok_ci) if n_boot == N_BOOT else "n/a (n_boot != 2000)",
                                          "reproduced": bool(ok)}
    out["all_reproduced"] = all(v.get("reproduced") is True for v in out["rows"].values())
    return out


def training_goal_series(metrics8: str | None, metrics7: str | None) -> dict:
    out = {}
    for tag, path, key in (("refcv8_ema", metrics8, "ema_eval_goal2s_err_m"), ("refcv8_raw", metrics8, "eval_goal2s_err_m"), ("refcv7_raw", metrics7, "eval_goal2s_err_m")):
        if not path or not os.path.exists(path):
            continue
        rows = []
        for ln in open(path, encoding="utf-8"):
            try:
                d = json.loads(ln)
            except Exception:                                         # noqa: BLE001
                continue
            if key in d and d.get("step") is not None:
                rows.append([int(d["step"]), float(d[key])])
        out[tag] = {"key": key, "path": path, "n": len(rows), "rows": rows}
    return out


# ------------------------------------------------------------------------------------------------ rendering
def f4(x, nd=4):
    return "" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def row_md(name, r, extra=""):
    if not r or r.get("status") == "EMPTY" or "delta" not in r:
        return f"| {name} | 0 | 0 | | | | | |"
    sep = "**SEP**" if r["separated"] else ""
    low = " (low n)" if r.get("low_n") else ""
    return (f"| {name}{low} | {r['n_win']} | {r['n_ep']} | {f4(r['mean_new'])} | {f4(r['mean_ref'])} | {f4(r['delta'])} "
            f"[{f4(r['lo'])}, {f4(r['hi'])}] | {sep} | {extra} |")


def render_md(res: dict) -> str:
    L = []
    L.append(f"# {res['title']}\n")
    L.append(f"* tier: **{res['tier']}**; estimator: {res['estimator']}\n* generated {res['generated']}; trees: " + "; ".join(f"`{k}` = {v}" for k, v in res["trees"].items()))
    L.append("* BIN EDGES (fixed constants, GT/v0/nav only): turn = GT mean |yaw rate| rad/s " + str(res["bin_edges"]["turn"]) + "; speed = v0 m/s " + str(res["bin_edges"]["speed"]))
    ctl = res["control_vs_battery"]
    L.append(f"* CONTROL (my overall heading cell vs the battery's PAIRED_READ): all_reproduced = **{ctl.get('all_reproduced')}**")
    for surf in res["surfaces"]:
        sr = res["surfaces"][surf]
        L.append(f"\n## {surf} ({sr['meta']['instants_s']} s) -- {sr['meta']['n_windows']} windows / {sr['meta']['n_episodes']} episodes; heading = LAT_heading_mae_deg\n")
        L.append("bin census (windows): " + json.dumps(sr["bins_census"]))
        for cname, comp in sr["pooled_2seed"].items():
            L.append(f"\n### {surf} -- {cname}, 2-seed mean of the per-window metric\n")
            L.append("| bin | n win | n ep | mean new | mean ref | delta [95 % CI] | sep | window share / contrib share / excess-vs-rest [Bonf CI] |")
            L.append("|---|---|---|---|---|---|---|---|")
            L.append(row_md("ALL", comp["overall"]))
            for d, rows in comp["dims"].items():
                for n, r in rows.items():
                    v = r.get("vs_rest", {})
                    ex = (f"{f4(r.get('window_share'), 3)} / {f4(r.get('contrib_share'), 3)} / {f4(v.get('excess_vs_rest'))} "
                          f"[{f4(v.get('bonf_lo'))}, {f4(v.get('bonf_hi'))}]{' BONF-SEP' if v.get('separated_bonferroni') else ''}")
                    L.append(row_md(f"{d}: {n}", r, ex))
            ec = comp["episode_concentration"]
            L.append(f"\nepisode concentration: {ec['frac_episodes_delta_gt0']:.3f} of episodes have mean delta > 0; top-5 episodes carry {f4(ec['top5_episode_share_of_sum'], 3)} of the summed delta; "
                     f"delta excluding the top 5 = {f4(ec['delta_excluding_top5_episodes'])} (overall {f4(ec['overall_delta'])}).")
        for cname in list(sr["pooled_2seed"]):
            comp = sr["pooled_2seed"][cname]
            L.append(f"\n### {surf} -- {cname}: by horizon instant (segment k = chord ending at instant k), |dh| and signed shift\n")
            L.append("| segment | group | n win | n ep | mean new | mean ref | delta |dh| [CI] | sep | signed shift new-ref [CI] |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for k, v in comp["by_instant"].items():
                a, sgn = v["abs"], v["signed_shift"]
                if "delta" not in a:
                    continue
                L.append(f"| {k.split('|')[0]} | {k.split('|')[1]} | {a['n_win']} | {a['n_ep']} | {f4(a['mean_new'])} | {f4(a['mean_ref'])} | {f4(a['delta'])} [{f4(a['lo'])}, {f4(a['hi'])}] | {'**SEP**' if a['separated'] else ''} | "
                         f"{f4(sgn.get('delta'))} [{f4(sgn.get('lo'))}, {f4(sgn.get('hi'))}]{' SEP' if sgn.get('separated') else ''} |")
        for cname in list(sr["pooled_2seed"])[:1]:
            comp = sr["pooled_2seed"][cname]
            L.append(f"\n### {surf} -- {cname}: turn x speed cross\n")
            L.append("| cell | n win | n ep | mean new | mean ref | delta [CI] | sep | |")
            L.append("|---|---|---|---|---|---|---|---|")
            for n, r in comp["cross_turn_x_speed"].items():
                L.append(row_md(n, r))
        fl = sr.get("inference_floor_s0_minus_s1", {})
        if fl:
            L.append(f"\n### {surf} -- inference floor per bin: same checkpoint, seed 0 minus seed 1 (mean heading MAE, paired)\n")
            labs = list(fl)
            L.append("| bin | " + " | ".join(f"{l} s0-s1 [CI]" for l in labs) + " |")
            L.append("|---|" + "---|" * len(labs))
            keys = list(fl[labs[0]])
            for k in keys:
                cells = []
                for l in labs:
                    r = fl[l].get(k, {})
                    cells.append(f"{f4(r.get('delta'))} [{f4(r.get('lo'))}, {f4(r.get('hi'))}]{' SEP' if r.get('separated') else ''}" if "delta" in r else "")
                L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.extend(render_extras(surf, sr))
    return "\n".join(L) + "\n"


def render_extras(surf: str, sr: dict) -> list:
    L = []
    for cname, rows in sr.get("floor_check", {}).items():
        L.append(f"\n### {surf} -- {cname}: drift vs the INFERENCE floor (2x rule), per seed (inference floor only; a single training run gives no training floor)\n")
        L.append("| bin | delta pooled | delta s0 / s1 | sep s0 / s1 / pooled | floor a / b | ratio | verdict |")
        L.append("|---|---|---|---|---|---|---|")
        for nm, r in rows.items():
            rt = r["ratio_delta_to_max_floor"]
            L.append(f"| {nm} | {f4(r['delta_pooled'])} | {f4(r['delta_by_seed'][0])} / {f4(r['delta_by_seed'][-1])} | {r['separated_by_seed'][0]} / {r['separated_by_seed'][-1]} / {r['separated_pooled']} | "
                     f"{f4(r['floor_a'])} / {f4(r['floor_b'])} | {f4(rt, 1)} | {r['verdict']} |")
    ex = sr.get("extras", {})
    if not ex:
        return L
    L.append(f"\n### {surf} -- turn-aligned signed heading error per horizon segment (deg; + over-rotates into the turn, - under-rotates), 2-seed mean of dh, segments valid on all arms\n")
    for g, rows in ex["bias_profile_turn_aligned_2seed_mean_dh"].items():
        L.append(f"**{g}**\n")
        for k, r in rows.items():
            L.append(f"* {k} (n {r['n_win']}): " + ", ".join(f"{kk[5:]} bias {vv:+.3f}" for kk, vv in r.items() if kk.startswith("bias_"))
                     + "; " + ", ".join(f"{kk[4:]} MAE {vv:.3f}" for kk, vv in r.items() if kk.startswith("mae_")))
    L.append(f"\n### {surf} -- multiplicative under-rotation test: slope of turn-aligned heading error on turn-aligned GT heading (through origin; -0.10 = turns 10 % less than GT), 2-seed mean dh\n")
    L.append("| group | n win / seg | mean GT heading | slopes [95 % CI] | slope minus refcv7 [CI] |")
    L.append("|---|---|---|---|---|")
    for g, r in ex["curvature_shrinkage"].items():
        sl = "; ".join(f"{k} {v['slope']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" for k, v in r["slope"].items())
        sd = "; ".join(f"{k} {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]{' SEP' if v['separated'] else ''}" for k, v in r["slope_minus_ref"].items())
        L.append(f"| {g} | {r['n_win']} / {r['n_seg']} | {r['mean_gt_turn_aligned_heading_deg']:.1f} | {sl} | {sd} |")
    L.append(f"\n### {surf} -- tail check: per-window delta median / trimmed mean / sign split (is the mean carried by a few windows?)\n")
    for nm, rows in ex["robust_by_bin"].items():
        L.append(f"**{nm}**\n\n| bin | n | mean | median | 10%-trim mean | frac worse / better / tied | sum of top 1 % windows / sum all |\n|---|---|---|---|---|---|---|")
        for b, r in rows.items():
            if "mean" in r:
                L.append(f"| {b} | {r['n_win']} | {f4(r['mean'])} | {f4(r['median'], 5)} | {f4(r['trimmed10_mean'], 5)} | {r['frac_worse']:.3f} / {r['frac_better']:.3f} / {r['frac_tied']:.3f} | {f4(r['sum_top1pct'], 1)} / {f4(r['sum_all'], 1)} |")
    L.append(f"\n### {surf} -- the episodes carrying the summed delta (panel file index only)\n")
    for nm, te in ex["top_episodes"].items():
        L.append(f"**{nm}** (sum {f4(te['total_sum_delta'], 1)} over {te['n_episodes']} episodes)\n")
        for r in te["top"][:6]:
            L.append(f"* ep{r['panel_file_index']:03d}: n {r['n_win']}, share {f4(r['share_of_total'], 3)}, mean delta {f4(r['mean_delta'])}, sharp-window frac {r['frac_windows_sharp']:.2f}, "
                     f"curved frac {r['frac_windows_curved(0.02+)']:.2f}, mean v0 {r['mean_v0']:.1f}, nav {r['nav_counts']}")
    L.append(f"\n### {surf} -- selection agreement with the oracle anchor x heading delta (new minus earlier), per seed\n")
    for k, rows in ex["oracle_agreement_split"].items():
        L.append(f"**{k}**\n\n| category | all: n / delta [CI] | sharp: n / delta [CI] |\n|---|---|---|")
        for cn, r in rows.items():
            a, b = r["all"], r["sharp"]
            fa = f"{a['n_win']} / {f4(a['delta'])} [{f4(a['lo'])}, {f4(a['hi'])}]" if "delta" in a else "0"
            fb = f"{b['n_win']} / {f4(b['delta'])} [{f4(b['lo'])}, {f4(b['hi'])}]" if "delta" in b else "0"
            L.append(f"| {cn} | {fa} | {fb} |")
    L.append(f"\n### {surf} -- does the model still USE nav for heading?  benefit = MAE(os_navzero) - MAE(os); then benefit(new) - benefit(earlier)\n")
    for k, rows in ex["nav_dependence"].items():
        L.append(f"**{k}**\n\n| bin | n | benefit new [CI] | benefit earlier [CI] | benefit(new) - benefit(earlier) [CI] |\n|---|---|---|---|---|")
        labs = [kk for kk in next(iter(rows.values())) if kk.endswith("benefit(navzero-os)")]
        for b, r in rows.items():
            bd = r["benefit_new_minus_earlier"]
            if "delta" not in bd:
                continue
            c0, c1 = r[labs[0]], r[labs[1]]
            L.append(f"| {b} | {bd['n_win']} | {f4(c0.get('delta'))} [{f4(c0.get('lo'))}, {f4(c0.get('hi'))}] | {f4(c1.get('delta'))} [{f4(c1.get('lo'))}, {f4(c1.get('hi'))}] | "
                     f"{f4(bd['delta'])} [{f4(bd['lo'])}, {f4(bd['hi'])}]{' SEP' if bd['separated'] else ''} |")
    L.append(f"\n### {surf} -- selection / goal proxy (new vs earlier, per seed)\n")
    for k, v in sr.get("selection_diagnostics", {}).items():
        gd = v.get("goal_dist_sel_new_minus_earlier", {})
        fl, nf = v["delta_heading_flip_windows"], v["delta_heading_nonflip_windows"]
        L.append(f"* {k}: anchor flip rate {v['flip_rate']:.3f}; heading delta on flipped windows {f4(fl.get('delta'))} [{f4(fl.get('lo'))}, {f4(fl.get('hi'))}] (n {fl.get('n_win')}) vs unflipped "
                 f"{f4(nf.get('delta'))} [{f4(nf.get('lo'))}, {f4(nf.get('hi'))}]; goal_dist_sel proxy new-earlier {f4(gd.get('delta'))} [{f4(gd.get('lo'))}, {f4(gd.get('hi'))}]; "
                 f"corr(window dHeading, window dGoalDist) {f4(v.get('corr_window_delta_heading_vs_delta_goal_dist'), 3)}")
    return L


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new-root", required=True)
    ap.add_argument("--label-new", required=True)
    ap.add_argument("--earlier", nargs="*", default=[], help="LABEL=PATH of earlier refcv8 trees (same windows); first = most recent")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--surfaces", default="S2,S6")
    ap.add_argument("--ref-prefix", default="b_refcv7_50400")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--taniteval-root", default="E:/Projects/TanitAD/taniteval")
    ap.add_argument("--stack-root", default="E:/Projects/TanitAD/stack")
    ap.add_argument("--metrics8", default="C:/Users/Admin/qland/work/watch8/metrics.jsonl")
    ap.add_argument("--metrics7", default="C:/Users/Admin/qland/work/watch7/metrics.jsonl")
    a = ap.parse_args()
    torch, ci, ff = setup(a.taniteval_root, a.stack_root)
    seeds = [int(x) for x in a.seeds.split(",")]
    roots = [(a.label_new, a.new_root)]
    for e in a.earlier:
        lab, path = e.split("=", 1)
        roots.append((lab, path))
    t0 = time.time()
    res = {"title": f"refcv8 heading-drift localisation: {a.label_new} vs refcv7-50,400 and vs earlier refcv8 points",
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)",
           "estimator": "paired episode-cluster bootstrap (taniteval/ci.py, B %d, seed 0), models FIXED; answers 'another draw of EPISODES' only -- "
                        "NOT another training run (H-ESTIM-SEED-1), and the s0-vs-s1 floor is the only inference-run evidence" % a.n_boot,
           "generated": time.strftime("%FT%T"), "trees": dict(roots), "ref_arm": a.ref_prefix + "_s{seed}", "seeds": seeds,
           "bin_edges": {"turn": list(TURN_EDGES), "speed": list(SPEED_EDGES), "speed_group": list(SPEED_GROUP_EDGES)},
           "low_n_rule": f"flagged when n_win < {LOW_N_WIN} or n_ep < {LOW_N_EP}", "surfaces": {}}
    for surf in a.surfaces.split(","):
        sr = run_surface(torch, ci, ff, surf, roots, seeds, a.ref_prefix, a.n_boot)
        sr["_newlab"] = a.label_new
        res["surfaces"][surf] = sr
        print(f"[{time.time() - t0:7.1f}s] {surf} done", flush=True)
    res["control_vs_battery"] = control_vs_paired_read(a.new_root, res["surfaces"], seeds, a.n_boot)
    res["training_goal2s_series"] = training_goal_series(a.metrics8, a.metrics7)
    for sr in res["surfaces"].values():
        sr.pop("_newlab", None)
    os.makedirs(a.out_dir, exist_ok=True)
    jp, mp = os.path.join(a.out_dir, a.tag + ".json"), os.path.join(a.out_dir, a.tag + ".md")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=1, default=str)
    open(mp, "w", encoding="utf-8").write(render_md(res))
    print("WROTE", jp, mp, f"({time.time() - t0:.0f}s)")
    print("CONTROL all_reproduced =", res["control_vs_battery"].get("all_reproduced"))


if __name__ == "__main__":
    main()
