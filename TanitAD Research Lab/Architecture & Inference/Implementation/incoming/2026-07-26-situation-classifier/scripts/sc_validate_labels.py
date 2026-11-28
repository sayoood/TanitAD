"""Situation classifier — LABEL VALIDATION (dev box, CPU). Runs before any classifier is read.

Five checks, each of which CAN fail:

  V1  ⭐ the LANE-CHANGE heading-gate sensitivity. A sibling stream MEASURED that a naive lane-change
      candidate set is ~99 % road curvature — its rate collapses **9.51 % -> 0.107 % (x89)** as the
      heading gate tightens 10 deg -> 1 deg (INHERITED, `2026-07-26-situation-semantics/`). This
      reports the whole curve for OUR definition, with and without the S-shape clause (L5/L6), so
      the reader can see which clause does the work rather than taking a point estimate on trust.

  V2  ⭐ the ROUNDABOUT counter-clockwise purity, TRAIN (in-sample, the selection target) vs HELD-OUT
      (genuinely out-of-sample). The corpus has ZERO left-hand-traffic clips, so a true roundabout
      label must be ~100 % ccw. Also reports the MAXIMUM same-sign sweep anywhere in the corpus,
      which is the quantity a sibling stream used to conclude roundabouts do not occur here.

  V3  the TURN population's left/right balance — a junction-turn population must be ~50/50, and a
      curve-following artefact would not be.

  V4  ⭐ PRE-REG Sec 6.2: does the TURN half mark junctions, or is it a curve detector?
      P(perpendicular cross traffic | tight TURN) vs P(... | matched-heading LARGE-RADIUS curve).

  V5  the per-situation speed / duration / geometry profile, so the labels can be sanity-read.

usage:  python sc_validate_labels.py <poses_dir> <selection.parquet> <bundle> <cross_dir> <out.json>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..",
                                                "2026-07-26-h2-classifier", "scripts")))
import sc_situations as S  # noqa: E402
from taniteval.ci import _draws, episode_index  # noqa: E402


def lc_variants(K, dpsi_max, use_sshape):
    """The lane-change detector with the heading gate swept and the S-shape clause switchable."""
    T, W = K["T"], int(round(S.LC_W_S * S.HZ))
    if T < W + 2:
        return []
    x, y, psi, v = K["x"], K["y"], K["psi"], K["v"]
    hits = []
    for i in range(0, T - W):
        j = i + W
        if v[i] < S.LC_V_MIN or v[i:j + 1].min() < S.LC_V_MIN_ANY:
            continue
        if abs(np.degrees(psi[j] - psi[i])) > dpsi_max:
            continue
        c, s = np.cos(psi[i]), np.sin(psi[i])
        lat = -s * (x[i:j + 1] - x[i]) + c * (y[i:j + 1] - y[i])
        end = lat[-1]
        if not (S.LC_LAT_MIN <= abs(end) <= S.LC_LAT_MAX):
            continue
        if abs(end) < S.LC_MONO * np.abs(lat).max():
            continue
        if use_sshape:
            d = np.degrees(np.diff(psi[i:j + 1]))
            if min(d[d > 0].sum(), -d[d < 0].sum()) < S.LC_LOBE_DEG:
                continue
            cum = np.cumsum(d)
            k1 = int(np.argmax(np.abs(cum) >= S.LC_LOBE_DEG)) if (np.abs(cum) >= S.LC_LOBE_DEG).any() else 0
            if np.sign(cum[k1]) != np.sign(end):
                continue
        hits.append((i, j))
    return S._merge(hits)


def main():
    poses_dir, sel_path, bundle, cross_dir, out = sys.argv[1:6]
    Z = np.load(os.path.join(poses_dir, "poses.npz"))
    pm = json.load(open(os.path.join(poses_dir, "poses_meta.json")))
    meta = json.load(open(os.path.join(bundle, "sc_meta.json")))
    L = np.load(os.path.join(bundle, "sc_labels.npz"))
    sel = pd.read_parquet(sel_path)
    sel["chunk"] = sel["chunk"].astype(str).str.zfill(4)
    clips = sorted(sel["clip_id"].astype(str))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(clips), generator=g).tolist()
    vi = set(perm[:max(1, int(len(clips) * 0.2))])
    tr = [c for i, c in enumerate(clips) if i not in vi]
    f2i = {m["file"]: m["i"] for m in pm if m["cache"] == "train"}
    side = {m["k"]: m["side"] for m in meta}
    R = {}

    # ---------------------------------------------------------------- V1 heading-gate sensitivity
    gates = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0)
    curve = {"with_S_shape": {}, "without_S_shape": {}}
    lat_end = {}
    sub = [m for m in meta if m["side"] == "TRAIN"]
    for use in (True, False):
        for gt in gates:
            n_ev = n_cl = 0
            lats = []
            for m in sub:
                K = S.kinematics(Z[f"p{f2i[m['file']]}"])
                ev = lc_variants(K, gt, use)
                n_ev += len(ev)
                n_cl += int(bool(ev))
                for a, b in ev:
                    c, s = np.cos(K["psi"][a]), np.sin(K["psi"][a])
                    lats.append(abs(-s * (K["x"][b] - K["x"][a]) + c * (K["y"][b] - K["y"][a])))
            key = "with_S_shape" if use else "without_S_shape"
            curve[key][f"{gt:g}"] = {"events": n_ev, "clips": n_cl,
                                     "clip_rate": round(n_cl / max(len(sub), 1), 5),
                                     "median_lateral_m": round(float(np.median(lats)), 3)
                                     if lats else None}
            if use and gt == S.LC_DPSI_MAX:
                lat_end = {"median_m": round(float(np.median(lats)), 3) if lats else None,
                           "p10_m": round(float(np.quantile(lats, .1)), 3) if lats else None,
                           "p90_m": round(float(np.quantile(lats, .9)), 3) if lats else None}
    w = curve["with_S_shape"]
    wo = curve["without_S_shape"]
    R["V1_lane_change_heading_gate"] = {
        "note": ("the sibling stream MEASURED a x89 collapse (9.51 % -> 0.107 %) over 10->1 deg on a "
                 "NAIVE candidate set; the collapse ratio for OUR definition is reported here"),
        "operating_gate_deg": S.LC_DPSI_MAX, "curve": curve,
        "collapse_ratio_10_to_1_with_S_shape": round(
            w["10"]["clip_rate"] / max(w["1"]["clip_rate"], 1e-9), 2),
        "collapse_ratio_10_to_1_without_S_shape": round(
            wo["10"]["clip_rate"] / max(wo["1"]["clip_rate"], 1e-9), 2),
        "S_shape_rejection_at_operating_gate": round(
            1.0 - w[f"{S.LC_DPSI_MAX:g}"]["events"] / max(wo[f"{S.LC_DPSI_MAX:g}"]["events"], 1), 4),
        "lateral_offset_at_operating_gate_m": lat_end}

    # ------------------------------------------------- V2 roundabout ccw purity + max corpus sweep
    ccw = {"TRAIN": [], "HELDOUT": []}
    ccw_core = {"TRAIN": [], "HELDOUT": []}
    max_sweep, max_sweep_r50 = 0.0, 0.0
    for m in meta:
        K = S.kinematics(Z[f"p{f2i[m['file']]}"])
        psi, kap = K["psi"], K["kappa"]
        for a, b, s in S.curvature_runs(K, kappa_min=1.0 / 400.0):
            d = abs(np.degrees(psi[b] - psi[a]))
            max_sweep = max(max_sweep, d)
            seg = np.abs(kap[a:b + 1])
            seg = seg[seg > 0]
            if len(seg) and 1.0 / np.median(seg) <= 50.0:
                max_sweep_r50 = max(max_sweep_r50, d)
        for a, b in S.detect_roundabout(K, bracket=True):
            ccw[m["side"]].append(int(np.sign(kap[a:b + 1].mean())))
        for a, b in S.detect_roundabout(K, bracket=False):
            ccw_core[m["side"]].append(int(np.sign(kap[a:b + 1].mean())))
    R["V2_roundabout"] = {
        "left_hand_traffic_clips": 0,
        "why": "0 left-hand-traffic clips in the selection -> a TRUE roundabout label must be ~100 % ccw",
        "ccw_purity": {k: {"n": len(v), "ccw_frac": round(float(np.mean(np.array(v) > 0)), 4)
                           if v else None} for k, v in ccw.items()},
        "ccw_purity_core": {k: {"n": len(v), "ccw_frac": round(float(np.mean(np.array(v) > 0)), 4)
                                if v else None} for k, v in ccw_core.items()},
        "max_same_sign_sweep_deg_any_radius": round(max_sweep, 1),
        "max_same_sign_sweep_deg_radius_le_50m": round(max_sweep_r50, 1),
        "sibling_claim_INHERITED": "0 of 2482 clips reach a 270 deg sweep; corpus maximum 252 deg"}

    # ------------------------------------------------------------------- V3 turn left/right balance
    sgn = {"TRAIN": [], "HELDOUT": []}
    for m in meta:
        K = S.kinematics(Z[f"p{f2i[m['file']]}"])
        for a, b in S.detect_turns(K):
            sgn[m["side"]].append(int(np.sign(K["kappa"][a:b + 1].mean())))
    R["V3_turn_direction_balance"] = {
        k: {"n": len(v), "left_frac": round(float(np.mean(np.array(v) > 0)), 4) if v else None}
        for k, v in sgn.items()}
    R["V3_note"] = "a junction-turn population must be ~50/50; a curve-following artefact would not be"

    # --------------------------------------------- V4 turn vs matched-heading large-radius curve
    R["V4_turn_is_junction"] = _v4(cross_dir, L, meta)

    # ------------------------------------------------------------------------- V5 label profiles
    prof = {}
    for nm, key in (("lane_change", "lc_ab"), ("roundabout", "rb_ab"), ("turn", "turn_ab"),
                    ("curve_control", "curve_ab")):
        dur, dps, spd, rad = [], [], [], []
        for m in meta:
            arr = np.asarray(L[f"c{m['k']}_{key}"]).reshape(-1, 2)
            if not len(arr):
                continue
            K = S.kinematics(Z[f"p{f2i[m['file']]}"])
            for a, b in arr:
                a, b = int(a), int(b)
                dur.append((b - a + 1) / S.HZ)
                dps.append(abs(np.degrees(K["psi"][b] - K["psi"][a])))
                spd.append(float(K["v"][a:b + 1].mean()))
                sg = np.abs(K["kappa"][a:b + 1])
                sg = sg[sg > 0]
                rad.append(float(1.0 / np.median(sg)) if len(sg) else np.nan)
        prof[nm] = {"n": len(dur), **{f"{q}_{lbl}": round(float(np.nanquantile(arr_, qq)), 3)
                                      for lbl, arr_ in (("dur_s", dur), ("dpsi_deg", dps),
                                                        ("speed_ms", spd), ("radius_m", rad))
                                      for q, qq in (("p10", .1), ("p50", .5), ("p90", .9))}
                    } if dur else {"n": 0}
    R["V5_profiles"] = prof
    json.dump(R, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in R.items() if k != "V1_lane_change_heading_gate"}, indent=2))
    print("\nV1 collapse ratios:",
          R["V1_lane_change_heading_gate"]["collapse_ratio_10_to_1_with_S_shape"], "(with S-shape)",
          R["V1_lane_change_heading_gate"]["collapse_ratio_10_to_1_without_S_shape"], "(without)")


def _v4(cross_dir, L, meta):
    p = os.path.join(cross_dir, "sc_cross.npz")
    if not os.path.exists(p):
        return {"status": "no cross artifacts"}
    C = np.load(p)
    A, Ae, Bv, Be = [], [], [], []
    for m in meta:
        k = m["k"]
        if f"c{k}_cross" not in C.files:
            continue
        cr = C[f"c{k}_cross"].astype(bool)
        for key, hit, eid in (("turn_ab", A, Ae), ("curve_ab", Bv, Be)):
            msk = np.zeros(len(cr), bool)
            for a, b in np.asarray(L[f"c{k}_{key}"]).reshape(-1, 2):
                msk[int(a):min(int(b) + 1, len(cr))] = True
            if msk.any():
                hit.append(cr[msk])
                eid.append(np.full(int(msk.sum()), k))
    if not A or not Bv:
        return {"status": "empty population", "n_turn": len(A), "n_curve": len(Bv)}
    y = np.concatenate([np.concatenate(A), np.concatenate(Bv)]).astype(float)
    gflag = np.concatenate([np.ones(sum(map(len, A))), np.zeros(sum(map(len, Bv)))])
    eid = np.concatenate([np.concatenate(Ae), np.concatenate(Be)])
    uniq, idx_by_ep = episode_index(eid)

    def ratio(y, gflag):
        a = y[gflag > 0].mean() if (gflag > 0).any() else np.nan
        b = y[gflag < 1].mean() if (gflag < 1).any() else np.nan
        return float(a / max(b, 1e-9))
    b_ = [ratio(y[s], gflag[s]) for s in _draws(uniq, idx_by_ep, 2000, 0)]
    b_ = np.array([x for x in b_ if np.isfinite(x)])
    return {"P_cross_given_TURN": round(float(y[gflag > 0].mean()), 5),
            "P_cross_given_LARGE_RADIUS_CURVE": round(float(y[gflag < 1].mean()), 5),
            "n_turn_frames": int((gflag > 0).sum()), "n_curve_frames": int((gflag < 1).sum()),
            "ratio": round(ratio(y, gflag), 3),
            "ci95": [round(float(np.quantile(b_, .025)), 3), round(float(np.quantile(b_, .975)), 3)],
            "estimator": "paired episode-cluster bootstrap (taniteval.ci._draws, B=2000)",
            "separated_from_1": bool(float(np.quantile(b_, .025)) > 1.0)}


if __name__ == "__main__":
    main()
