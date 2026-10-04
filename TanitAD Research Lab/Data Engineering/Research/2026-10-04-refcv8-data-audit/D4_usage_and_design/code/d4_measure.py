#!/usr/bin/env python3
"""D4 zero-GPU measurements for the refcv8 label design (CPU only, dev box, < 1.5 GB RSS).

Inputs (all md5-pinned in the output):
  * labels  : s2_labels_v8_{train,eval}.jsonl.gz (train md5 b45377a1..., the blob refcv7 trained on)
  * poses   : the v2ep MANIFESTS of the exact caches refcv7 read (train copied from Thor, md5 3c9f8bc8...;
              eval139 local). poses[t] = (x, y, yaw, v) provider rows = raw rows - 2.
  * clock   : refcv6_clip_clock_sidecar.jsonl (the trainer's own --clip-clock-sidecar)
Outputs: raw/d4_measure.json. Clip ids are never written (sha12 only, and only in digests).

Measures (each is about a PROPOSED refcv8 label or about how refcv7 USED a channel; the existing-label
coverage census is D1's and the label-vs-geometry validity is D2's -- not repeated here):
  A. EVAL-DIAG grid reproduction (known values from the route RESULT: digest 92e36a1a, 107/588/105/312,
     per-clip nav side 0.439 on turn windows, lat_v7 IGNORE 72/107).
  B. Time-localised nav: per-window NavSim-equivalent command (path rule) and nav_30s entry rule;
     their agreement (+ origin-shift mutation); the nav-compliance predicate's correctness with the
     per-clip token vs the per-window command on TRAIN windows.
  C. Dense tactical labels: coverage, class shares, agreement with the 6-s GT class.
  D. Max-speed leak test: 5-fold clip-grouped OOF R^2 of the window's own future max speed
     (max v over [NOW+2, NOW+6] s) from v0 alone vs v0 + a candidate input.
  E. Traffic-light RED propagation over the stop episode (window gain).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d4_lib as L  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "raw")
SCR = os.environ["D4_SCRATCH"]
LAB = {"train": "D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz",
       "eval": "D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz"}
MAN = {"train": os.path.join(SCR, "train_v2manifest.pt"),
       "eval": "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt"}
SIDECAR = "D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl"
DT_NOM = 0.1007
LAT_SIDE = {"LANE_KEEP": 0, "NUDGE_L": 1, "NUDGE_R": -1, "TURN_L": 1, "TURN_R": -1,
            "LANE_CHANGE_L": 1, "LANE_CHANGE_R": -1, "ABORT_LC": 0}
NAV_SIDE = {"NAV_FOLLOW_ROAD": 0, "NAV_TURN_L": 1, "NAV_TURN_R": -1}
DENSE_SIDE = np.array([0, 1, -1, 1, -1, 1, -1])


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _lctext(lc):
    """EXECUTED + side L/R -> +1/-1; NOT_CITED -> 0; anything else (ANTICIPATED, NEGATED, MULTI) -> 9."""
    st, sd = lc.get("status"), lc.get("side")
    if st == "EXECUTED" and sd in ("L", "R"):
        return 1 if sd == "L" else -1
    return 0 if st in (None, "NOT_CITED") else 9


def load_labels(p):
    out = {}
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            g = (r.get("g_tac") or {}).get("goals") or {}
            sb = g.get("SPEED_BAND") or {}
            smi = r.get("speed_max_input") or {}
            nav = r.get("nav_command") or {}
            out[r["clip_id"]] = dict(
                t0=float(r["t0_s"]), band=tuple(r["bands"]["tactical_s"]),
                lat=(r.get("a_tac") or {}).get("lat"), lon=(r.get("a_tac") or {}).get("lon"),
                nav=nav.get("token"), nav_args=nav.get("args") or {},
                entries=(r.get("nav_30s") or {}).get("entries") or [],
                v_hi=sb.get("v_hi_ms"), v_lo=sb.get("v_lo_ms"), v_max=smi.get("v_max_ms"),
                country=(r.get("strata") or {}).get("country"),
                road=(r.get("strata") or {}).get("road_class"),
                lctext=_lctext(r.get("lane_change_text") or {}),
                tl={k: True for k in g if k.startswith("TRAFFIC_LIGHT_REACT")},
                tl_state={k: (g[k] or {}).get("state") for k in g if k.startswith("TRAFFIC_LIGHT_REACT")})
    return out


def load_clock():
    side = {}
    with open(SIDECAR, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                q = json.loads(line)
                side[int(q["sid"])] = (float(q["grid_start_s"]), float(q["dt_s"]))
    return side


def r2(y, yhat):
    return float(1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def oof_r2(y, X, folds, k=5):
    yhat = np.zeros_like(y)
    for f in range(k):
        tr, te = folds != f, folds == f
        w, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        yhat[te] = X[te] @ w
    return r2(y, yhat), float(np.abs(y - yhat).mean())


def onehot(idx, n):
    o = np.zeros((len(idx), n))
    o[np.arange(len(idx)), np.asarray(idx, int)] = 1.0
    return o


def run_split(split, labs, clock):
    man = torch.load(MAN[split], map_location="cpu", weights_only=False)
    n_clip = len(man["clip_id"])
    rec = {k: [] for k in ("ci", "t", "tnow", "inband", "full6", "gt", "th", "navclip", "navpath", "navpath3",
                           "navpath_ok", "naventry", "naventry_mut", "ent_def", "dist_next", "time_next", "dlat", "dlon",
                           "v0", "yvmax", "latv7side", "latv7", "pastmax", "clipmax", "rel",
                           "nxside", "vmax06", "kmax", "dpsimax", "lctext")}
    clips = []
    n_fallback_clock = 0
    for i in range(n_clip):
        cid = man["clip_id"][i]
        lab = labs.get(cid)
        if lab is None:
            continue
        P = man["poses"][i].numpy().astype(np.float64)
        sid = int(man["episode_uid"][i])
        g0, dt = clock.get(sid, (None, None))
        if g0 is None:
            g0, dt = 0.0, DT_NOM
            n_fallback_clock += 1
        B = L.window_block(P)
        n = B["n"]
        r = B["r"]
        tnow = g0 + (r + L.RAW_OFF) * dt
        lo, hi = lab["band"]
        inband = np.abs(tnow - lab["t0"]) <= (hi - lo) / 2.0
        full6 = B["valid"][:, 60]
        gt, th = L.gt_class(B)
        navclip = np.full(n, NAV_SIDE.get(lab["nav"], 0), np.int8)
        navpath, latoff, okp = L.nav_path_rule_xy(P, B)
        navpath3, _, _ = L.nav_path_rule_xy(P, B, lat_m=3.0)
        ra = int(round((lab["t0"] - g0) / dt)) - L.RAW_OFF
        ra = min(max(ra, 0), B["T"] - 1)
        s_now = B["S"][r] - B["S"][ra]
        nent, dnext, dend = L.nav_entry_rule(lab["entries"], s_now)
        nxside = L.nav_entry_rule.next_side.copy()
        nent_mut, _, _ = L.nav_entry_rule(lab["entries"], B["S"][r])
        ent_def = tnow >= lab["t0"] - 1e-6          # entries list only turns starting at/after the anchor
        # time to the next turn start (anchor-relative t_start -> raw) for the args
        tnext = np.full(n, np.inf)
        for e in sorted([q for q in lab["entries"] if q.get("t_start_s") is not None],
                        key=lambda q: q["t_start_s"]):
            if NAV_SIDE.get(e.get("token"), 0) == 0:
                continue
            ts = lab["t0"] + float(e["t_start_s"]) - tnow
            te = (lab["t0"] + float(e["t_end_s"]) - tnow) if e.get("t_end_s") is not None else np.full(n, np.inf)
            nxt = (te > 0) & ~np.isfinite(tnext)
            tnext = np.where(nxt, ts, tnext)
        dlat, dlon, _aux = L.dense_tactical(B)
        vmax06 = np.where(B["valid"], B["V"], -np.inf).max(1)
        lct = lab.get("lctext", 0)
        V = B["V"]
        yv = np.where(full6, np.where(B["valid"], V, -np.inf)[:, 20:61].max(1), np.nan)
        v = B["v"]
        pastmax = np.array([v[max(0, rr - 100):rr + 1].max() for rr in r])
        clipmax = np.full(n, v.max())
        if lab["lat"] in LAT_SIDE:
            lv = np.where(inband, LAT_SIDE[lab["lat"]], -9)
        else:
            lv = np.full(n, -9)
        rel = np.where(tnow + 6.0 < lab["t0"] + lo, 0, np.where(tnow > lab["t0"] + hi, 2, 1))
        ci = len(clips)
        clips.append(dict(sha=L.sha12(cid), sid=sid, lab=lab, ra=ra, v=v, g0=g0, dt=dt, T=B["T"]))
        for k, val in (("ci", np.full(n, ci)), ("t", B["t"]), ("tnow", tnow), ("inband", inband),
                       ("full6", full6), ("gt", gt), ("th", th), ("navclip", navclip), ("navpath", navpath),
                       ("navpath3", navpath3), ("navpath_ok", okp), ("naventry", nent), ("naventry_mut", nent_mut), ("ent_def", ent_def),
                       ("dist_next", dnext), ("time_next", tnext), ("dlat", dlat), ("dlon", dlon),
                       ("v0", B["v0"]), ("yvmax", yv), ("latv7side", lv),
                       ("latv7", np.where(inband, lab["lat"] or "", "IGNORE")),
                       ("pastmax", pastmax), ("clipmax", clipmax), ("rel", rel),
                       ("nxside", nxside), ("vmax06", vmax06), ("kmax", _aux["kmax"]),
                       ("dpsimax", _aux["dpsi_absmax"]), ("lctext", np.full(n, lct))):
            rec[k].append(val)
    W = {k: np.concatenate(v) for k, v in rec.items()}
    return W, clips, n_fallback_clock


def frac(a, b):
    return {"k": int(a), "n": int(b), "frac": (round(a / b, 4) if b else None)}


def main():
    t_start = time.time()
    res = {"_evidence": "MEASURED (ours); D4 refcv8 data audit; dev box CPU; ids sha12 only", "inputs": {}}
    for s in ("train", "eval"):
        res["inputs"][f"labels_{s}_md5"] = md5(LAB[s])
        res["inputs"][f"manifest_{s}_md5"] = md5(MAN[s])
    res["inputs"]["clock_sidecar_md5"] = md5(SIDECAR)
    clock = load_clock()
    out = {}
    for split in ("eval", "train"):
        labs = load_labels(LAB[split])
        W, clips, nfb = run_split(split, labs, clock)
        out[split] = (W, clips)
        res[f"{split}_n"] = {"clips": len(clips), "windows": int(len(W["t"])), "clock_fallback_clips": nfb,
                             "full6_windows": frac(W["full6"].sum(), len(W["t"]))}
        print(f"[d4] {split}: {len(clips)} clips {len(W['t'])} windows ({time.time()-t_start:.0f}s)", flush=True)

    # ======================== A. EVAL-DIAG grid reproduction (known values) =========================
    W, clips = out["eval"]
    by = {}
    for j in range(len(W["t"])):
        by.setdefault(int(W["ci"][j]), []).append((int(W["t"][j]), j))
    idx, keys = [], []
    for ci in sorted(by, key=lambda c: clips[c]["sha"]):
        ts = sorted(by[ci])
        for q in range(8):
            t_, j = ts[int((q + 0.5) * len(ts) / 8)]
            idx.append(j)
            keys.append((clips[ci]["sha"], t_))
    digest = hashlib.sha256(json.dumps(keys).encode()).hexdigest()
    idx = np.array(idx)
    gt = W["gt"][idx]
    gside = np.where(gt == "turnL", 1, np.where(gt == "turnR", -1, 0))
    turn = (gt == "turnL") | (gt == "turnR")
    straight = gt == "straight"
    A = {"digest16": digest[:16], "digest_known": "92e36a1a",
         "digest_pass": digest.startswith("92e36a1a"),
         "class_counts": {c: int((gt == c).sum()) for c in ("turnL", "turnR", "straight", "gentle", "unclassified")},
         "class_counts_known": {"turnL": 40, "turnR": 67, "straight": 588, "gentle": 105, "unclassified": 312}}
    A["class_pass"] = A["class_counts"] == A["class_counts_known"]
    nc = W["navclip"][idx]
    A["navclip_side_correct_on_turn"] = frac((nc[turn] == gside[turn]).sum(), turn.sum())
    A["navclip_side_correct_on_turn_known"] = 0.439
    lv = W["latv7side"][idx]
    A["latv7_ignore_on_turn"] = frac((lv[turn] == -9).sum(), turn.sum())
    A["latv7_ignore_on_turn_known"] = "72/107"
    A["latv7_side_agree_where_present_on_turn"] = frac(((lv == gside) & (lv != -9) & turn).sum(),
                                                       ((lv != -9) & turn).sum())
    A["latv7_side_agree_known"] = 0.486
    # ---- the PROPOSED per-window labels on the same 107 / 588 windows ----
    npth = W["navpath"][idx]
    nent = W["naventry"][idx]
    edef = W["ent_def"][idx]
    nloc = np.where(npth != -9, npth, np.where(edef, nent, -9))
    A["navlocal_defined_on_turn"] = frac((nloc[turn] != -9).sum(), turn.sum())
    A["navlocal_side_correct_on_turn"] = frac((nloc[turn] == gside[turn]).sum(), turn.sum())
    A["navlocal_straight_on_straight"] = frac((nloc[straight] == 0).sum(), straight.sum())
    A["navclip_straight_on_straight"] = frac((nc[straight] == 0).sum(), straight.sum())
    lrclip = nc != 0
    A["LRclip_windows_GTstraight_navclip"] = frac((lrclip & straight).sum(), (lrclip & (turn | straight | (gt == 'gentle'))).sum())
    A["LRclip_windows_GTstraight_navlocal_says_turn"] = frac((lrclip & straight & ((nloc == 1) | (nloc == -1))).sum(),
                                                          (lrclip & straight).sum())
    dl = W["dlat"][idx]
    dside = np.where(dl >= 0, DENSE_SIDE[np.clip(dl, 0, 6)], -9)
    A["dense_lat_labelled_on_turn"] = frac((dl[turn] != -100).sum(), turn.sum())
    A["dense_lat_side_correct_on_turn"] = frac((dside[turn] == gside[turn]).sum(), turn.sum())
    A["dense_lat_TURN_on_turn"] = frac(np.isin(dl[turn], [5, 6]).sum(), turn.sum())
    A["dense_lat_side_zero_on_straight"] = frac((dside[straight] == 0).sum(), straight.sum())
    A["dense_labelled_all_grid"] = frac((dl != -100).sum(), len(dl))
    res["A_eval_diag"] = A

    # ======================== B. nav on TRAIN windows ===============================================
    W, clips = out["train"]
    gt = W["gt"]
    cls_ok = gt != "unclassified"
    gside = np.where(gt == "turnL", 1, np.where(gt == "turnR", -1, 0))
    th = W["th"]
    nc = W["navclip"]
    npth = W["navpath"]
    nloc = np.where(npth != -9, npth, np.where(W["ent_def"], W["naventry"], -9))
    Bm = {}
    lr = nc != 0
    Bm["LRclip_windows"] = frac(lr.sum(), len(nc))
    Bm["LRclip_windows_GT_straight_of_classified"] = frac((lr & (gt == "straight")).sum(), (lr & cls_ok).sum())
    Bm["LRclip_windows_GT_turn_same_side_of_classified"] = frac((lr & (gside == nc) & ((gt == 'turnL') | (gt == 'turnR'))).sum(), (lr & cls_ok).sum())
    comp_clip = L.compliance(th, nc)
    Bm["compliance_predicate_true_on_LRclip_classified"] = frac((comp_clip & lr & cls_ok).sum(), (lr & cls_ok).sum())
    loclr = (nloc == 1) | (nloc == -1)
    comp_loc = L.compliance(th, nloc)
    Bm["compliance_predicate_true_on_localLR_classified"] = frac((comp_loc & loclr & cls_ok).sum(), (loclr & cls_ok).sum())
    Bm["followclip_windows_GT_turn"] = frac(((nc == 0) & ((gt == 'turnL') | (gt == 'turnR'))).sum(), ((nc == 0) & cls_ok).sum())
    Bm["navlocal_coverage"] = {"path_defined": frac((npth != -9).sum(), len(npth)),
                               "entry_only": frac(((npth == -9) & W["ent_def"]).sum(), len(npth)),
                               "undefined": frac((nloc == -9).sum(), len(npth))}
    Bm["navlocal_dist"] = {k: frac((nloc == v).sum(), (nloc != -9).sum()) for k, v in (("L", 1), ("S", 0), ("R", -1))}
    Bm["navclip_dist_windows"] = {k: frac((nc == v).sum(), len(nc)) for k, v in (("L", 1), ("S", 0), ("R", -1))}
    both = (npth != -9) & W["ent_def"]
    Bm["path_vs_entry_agree"] = frac((npth[both] == W["naventry"][both]).sum(), both.sum())
    Bm["path3m_vs_entry_agree"] = frac((W["navpath3"][both] == W["naventry"][both]).sum(), both.sum())
    turnish = both & ((npth != 0) | (W["naventry"] != 0))
    Bm["path_vs_entry_agree_on_turnish"] = frac((npth[turnish] == W["naventry"][turnish]).sum(), turnish.sum())
    # MUTATION (must go RED): the window's arc measured from the CLIP START instead of the anchor
    # (the documented "NOT from the recording start" bug class of nav_30s, 2026-09-09)
    Bm["MUTATION_origin_at_clip_start_agree"] = frac((npth[both] == W["naventry_mut"][both]).sum(), both.sum())
    Bm["MUTATION_origin_at_clip_start_agree_on_turnish"] = frac((npth[turnish] == W["naventry_mut"][turnish]).sum(), turnish.sum())
    Bm["args"] = {}
    dn = W["dist_next"]
    fin = np.isfinite(dn) & (nloc != -9)
    Bm["args"]["windows_with_a_next_turn_in_nav30"] = frac(fin.sum(), len(dn))
    tn = W["time_next"]
    Bm["args"]["time_to_next_turn_s_quantiles_(finite)"] = [round(float(q), 2) for q in np.quantile(tn[np.isfinite(tn)], [0.1, 0.5, 0.9])] if np.isfinite(tn).any() else None
    res["B_nav_train"] = Bm

    # ======================== C. dense tactical labels (TRAIN) ======================================
    dl, dlon = W["dlat"], W["dlon"]
    C = {"labelled": frac((dl != -100).sum(), len(dl)),
         "lat_dist": {L.LAT_NAMES[i]: frac((dl == i).sum(), (dl != -100).sum()) for i in range(7)},
         "lon_dist": {L.LON_NAMES[i]: frac((dlon == i).sum(), (dlon != -100).sum()) for i in range(6)}}
    tr = (gt == "turnL") | (gt == "turnR")
    dside = np.where(dl >= 0, DENSE_SIDE[np.clip(dl, 0, 6)], -9)
    C["side_agree_with_GT_on_GT_turn"] = frac((dside[tr] == gside[tr]).sum(), tr.sum())
    C["TURN_on_GT_turn"] = frac(np.isin(dl[tr], [5, 6]).sum(), tr.sum())
    C["TURN_on_GT_straight"] = frac(np.isin(dl[gt == "straight"], [5, 6]).sum(), (gt == "straight").sum())
    ib = W["inband"] & (dl != -100) & (W["latv7"] != "IGNORE")
    lab_side = W["latv7side"]
    C["inband_dense_side_vs_latv7_side_agree"] = frac((dside[ib] == lab_side[ib]).sum(), ib.sum())
    C["inband_latv7_side_vs_GT_side_on_GT_turn"] = frac(((lab_side == gside) & ib & tr).sum(), (ib & tr).sum())
    C["inband_dense_side_vs_GT_side_on_GT_turn"] = frac(((dside == gside) & ib & tr).sum(), (ib & tr).sum())
    res["C_dense_train"] = C

    # ======================== D. max-speed leak test (TRAIN, full-6 s windows) ======================
    m = W["full6"] & np.isfinite(W["yvmax"])
    y = W["yvmax"][m]
    v0 = W["v0"][m]
    ci = W["ci"][m]
    folds = np.array([int(clips[c]["sha"], 16) % 5 for c in range(len(clips))])[ci]
    one = np.ones((len(y), 1))
    vhi = np.array([clips[c]["lab"]["v_hi"] or 0.0 for c in range(len(clips))])[ci]
    vmx = np.array([clips[c]["lab"]["v_max"] or 0.0 for c in range(len(clips))])[ci]
    base = np.hstack([one, v0[:, None]])
    feats = {
        "v0_only": base,
        "refcv7_fed_clip_vhi_bin4": np.hstack([base, onehot(L.snap_up(vhi, L.LADDER4_KMH), 4)]),
        "clip_vmax_bin8_(E16)": np.hstack([base, onehot(L.snap_up(vmx, L.LADDER8_KMH), 8)]),
        "clip_vhi_raw": np.hstack([base, vhi[:, None]]),
        "window_future_max_bin8_(opt a)": np.hstack([base, onehot(L.snap_up(y, L.LADDER8_KMH), 8)]),
        "window_future_max_bin4": np.hstack([base, onehot(L.snap_up(y, L.LADDER4_KMH), 4)]),
        "clip_realised_max_bin8": np.hstack([base, onehot(L.snap_up(W["clipmax"][m], L.LADDER8_KMH), 8)]),
        "past10s_max_bin8_(admissible)": np.hstack([base, onehot(L.snap_up(W["pastmax"][m], L.LADDER8_KMH), 8)]),
        "past10s_max_raw_(admissible)": np.hstack([base, W["pastmax"][m][:, None]]),
    }
    countries = sorted({clips[c]["lab"]["country"] or "?" for c in range(len(clips))})
    cidx = np.array([countries.index(clips[c]["lab"]["country"] or "?") for c in range(len(clips))])[ci]
    feats["country_onehot"] = np.hstack([base, onehot(cidx, len(countries))[:, 1:]])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(clips))
    feats["CONTROL_shuffled_clip_bin4"] = np.hstack([base, onehot(L.snap_up(
        np.array([clips[c]["lab"]["v_hi"] or 0.0 for c in perm])[ci], L.LADDER4_KMH), 4)])
    feats["CONTROL_target_itself"] = np.hstack([base, y[:, None]])
    D = {"n_windows": int(len(y)), "n_clips": len(clips), "target": "max v over provider rows NOW+20..NOW+60 (raw [NOW+2 s, NOW+6 s])",
         "estimator": "5-fold OOF, folds = int(sha12,16) % 5 (clip-grouped), ordinary least squares", "rows": {}}
    r20, _ = oof_r2(y, base, folds)
    for k, X in feats.items():
        rr, mae = oof_r2(y, X, folds)
        D["rows"][k] = {"oof_r2": round(rr, 4), "mae_ms": round(mae, 3),
                        "future_info_recovered": round((rr - r20) / (1 - r20), 4) if k != "v0_only" else 0.0}
    # the same for the refcv7 fed value split by window position relative to the label band
    relm = W["rel"][m]
    D["refcv7_fed_by_position"] = {}
    for name, code in (("future_ends_before_band", 0), ("overlaps_band", 1), ("band_in_past", 2)):
        s = relm == code
        if s.sum() < 100:
            continue
        rb, _ = oof_r2(y[s], base[s], folds[s])
        rc, _ = oof_r2(y[s], feats["refcv7_fed_clip_vhi_bin4"][s], folds[s])
        D["refcv7_fed_by_position"][name] = {"n": int(s.sum()), "r2_v0": round(rb, 4), "r2_v0_bin4": round(rc, 4),
                                             "recovered": round((rc - rb) / (1 - rb), 4)}
    # anchor-level replication of the label file's own statement (v_hi <- v0 0.8789; <- (bin8, v0) 0.9702)
    va, yh, b8, b4, fa = [], [], [], [], []
    for c_i, c in enumerate(clips):
        lab = c["lab"]
        if lab["v_hi"] is None or lab["v_max"] is None:
            continue
        va.append(c["v"][min(max(c["ra"], 0), c["T"] - 1)])
        yh.append(lab["v_hi"])
        b8.append(L.snap_up(lab["v_max"], L.LADDER8_KMH))
        b4.append(L.snap_up(lab["v_hi"], L.LADDER4_KMH))
        fa.append(int(c["sha"], 16) % 5)
    va, yh, fa = np.array(va), np.array(yh), np.array(fa)
    one = np.ones((len(va), 1))
    bb = np.hstack([one, va[:, None]])
    ra0, _ = oof_r2(yh, bb, fa)
    ra8, _ = oof_r2(yh, np.hstack([bb, onehot(np.array(b8), 8)]), fa)
    ra4, _ = oof_r2(yh, np.hstack([bb, onehot(np.array(b4), 4)]), fa)
    # recompute v_hi from the 10 Hz poses at the anchor (alignment control)
    vre = []
    for c in clips:
        if c["lab"]["v_hi"] is None:
            continue
        a = c["ra"]
        seg = c["v"][a + 20:a + 61]
        vre.append(seg.max() if len(seg) else np.nan)
    vre = np.array(vre)
    okv = np.isfinite(vre)
    D["anchor_replication"] = {"n_clips": int(len(yh)), "r2_vhi_from_v0": round(ra0, 4),
                               "r2_vhi_from_v0_bin8": round(ra8, 4), "r2_vhi_from_v0_bin4": round(ra4, 4),
                               "recovered_bin8": round((ra8 - ra0) / (1 - ra0), 4),
                               "recovered_bin4": round((ra4 - ra0) / (1 - ra0), 4),
                               "known_values_label_file": {"r2_v0": 0.8789, "r2_v0_bin8": 0.9702, "recovered": 0.754},
                               "vhi_recomputed_from_poses_abs_err_median_ms": round(float(np.median(np.abs(vre[okv] - yh[okv]))), 3),
                               "vhi_recomputed_within_0p2ms": frac((np.abs(vre[okv] - yh[okv]) <= 0.2).sum(), okv.sum())}
    res["D_maxspeed_leak_train"] = D

    # ======================== E. traffic-light RED propagation over the stop episode =================
    E = {"n_clips_red": 0, "inband_windows_red": 0, "propagated_windows_red": 0, "clips_red_with_stop_overlap": 0}
    for ci_, c in enumerate(clips):
        lab = c["lab"]
        if "TRAFFIC_LIGHT_REACT_RED" not in lab["tl"]:
            continue
        E["n_clips_red"] += 1
        sel = W["ci"] == ci_
        tn = W["tnow"][sel]
        E["inband_windows_red"] += int(W["inband"][sel].sum())
        v = c["v"]
        stopped = v <= L.V_STOP
        # stop episodes in raw seconds
        rows = np.arange(len(v))
        traw = c["g0"] + (rows + L.RAW_OFF) * c["dt"]
        lo, hi = lab["band"]
        b0, b1 = lab["t0"] - 2.0, lab["t0"] + hi       # NOW in band and its 6-s future
        ep = []
        i = 0
        while i < len(v):
            if stopped[i]:
                j = i
                while j + 1 < len(v) and stopped[j + 1]:
                    j += 1
                ep.append((i, j))
                i = j + 1
            else:
                i += 1
        hit = [(a, b) for a, b in ep if traw[b] >= b0 and traw[a] <= b1]
        if not hit:
            continue
        E["clips_red_with_stop_overlap"] += 1
        a, b = hit[0]
        # approach: back to the last row before the stop where v >= 5 m/s (or the clip start), capped 8 s
        k = a
        while k > 0 and v[k] < 5.0 and (a - k) < 80:
            k -= 1
        t_lo, t_hi = traw[k], traw[b]          # until the launch (end of the stop episode)
        E["propagated_windows_red"] += int(((tn >= t_lo) & (tn <= t_hi)).sum())
    res["E_tl_red_propagation_train"] = E
    # ======================== F. extras ==============================================================
    F = {}
    # F1 eval-diag: 20-m token OR args (next turn of the correct side starting within the 6-s horizon)
    We, ce = out["eval"]
    gte = We["gt"][idx]
    gse = np.where(gte == "turnL", 1, np.where(gte == "turnR", -1, 0))
    tne = (gte == "turnL") | (gte == "turnR")
    nloc_e = np.where(We["navpath"][idx] != -9, We["navpath"][idx],
                      np.where(We["ent_def"][idx], We["naventry"][idx], -9))
    tn6 = (We["time_next"][idx] <= 6.0) & (We["nxside"][idx] != 0)
    combo = np.where((nloc_e == 1) | (nloc_e == -1), nloc_e, np.where(tn6, We["nxside"][idx], 0))
    F["evaldiag_token20m_or_args6s_side_correct_on_turn"] = frac((combo[tne] == gse[tne]).sum(), tne.sum())
    F["evaldiag_token20m_or_args6s_straight_on_straight"] = frac((combo[gte == "straight"] == 0).sum(),
                                                                 (gte == "straight").sum())
    F["evaldiag_entry_args_defined_(NOW>=anchor)_on_turn"] = frac(We["ent_def"][idx][tne].sum(), tne.sum())
    # F2 train: dense TURN misses on GT-turn windows -- big radius vs heading not reached
    tr = (gt == "turnL") | (gt == "turnR")
    miss = tr & ~np.isin(dl, [5, 6]) & (dl != -100)
    F["train_dense_turn_miss_on_GT_turn"] = frac(miss.sum(), (tr & (dl != -100)).sum())
    F["miss_heading_ge30_but_R_gt40m"] = frac((miss & (W["dpsimax"] >= 30.0) & (W["kmax"] < 1 / 40.0)).sum(), miss.sum())
    F["miss_heading_lt30_within_horizon"] = frac((miss & (W["dpsimax"] < 30.0)).sum(), miss.sum())
    # F3 lane-change TEXT (alpamayo chain_of_causation, an independent channel) vs dense LANE_CHANGE side
    cis = W["ci"]
    lc_side_any = {}
    for side, codes in ((1, [1]), (-1, [2])):
        hit = np.isin(dl, codes)
        lc_side_any[side] = np.bincount(cis[hit], minlength=len(clips)) > 0
    txt = np.array([clips[c]["lab"].get("lctext", 0) for c in range(len(clips))])
    for side, nm in ((1, "L"), (-1, "R")):
        has = txt == side
        F[f"lc_text_{nm}_clips_with_dense_LC_same_side"] = frac((lc_side_any[side] & has).sum(), has.sum())
        F[f"lc_text_{nm}_clips_with_dense_LC_opposite_side"] = frac((lc_side_any[-side] & has).sum(), has.sum())
    none = txt == 0
    F["no_lc_text_clips_with_any_dense_LC"] = frac(((lc_side_any[1] | lc_side_any[-1]) & none).sum(), none.sum())
    # F4 the human's own speed vs the FED ceiling (what SPEC_REFCV7 sec. 26.1 would enforce at E9)
    v6 = W["vmax06"]
    vhi_w = np.array([clips[c]["lab"]["v_hi"] or 0.0 for c in range(len(clips))])[cis]
    lim4 = np.array(L.LADDER4_KMH, float)[L.snap_up(vhi_w, L.LADDER4_KMH)] / 3.6
    over = v6 > lim4 + 1e-6
    F["human_max_speed_0_6s_exceeds_fed_clip_ceiling"] = frac(over.sum(), len(over))
    F["human_exceeds_fed_clip_ceiling_by_gt_2ms"] = frac((v6 > lim4 + 2.0).sum(), len(over))
    F["fed_clip_bin4_differs_from_window_own_0_6s_bin4"] = frac(
        (L.snap_up(vhi_w, L.LADDER4_KMH) != L.snap_up(np.maximum(v6, 0), L.LADDER4_KMH)).sum(), len(v6))
    # F5 deploy-matched ceiling: window [NOW, NOW+6] max snapped UP (8-ladder), +1 step w.p. 0.52,
    # UNKNOWN (all-zero) w.p. 0.45 -- the NavSim navtest map statistics (INHERITED, refcv6 NavSim RESULT 6.6)
    m = W["full6"] & np.isfinite(W["yvmax"])
    y = W["yvmax"][m]
    v0 = W["v0"][m]
    cim = cis[m]
    folds = np.array([int(clips[c]["sha"], 16) % 5 for c in range(len(clips))])[cim]
    base = np.hstack([np.ones((len(y), 1)), v0[:, None]])
    rng = np.random.default_rng(1)
    b = L.snap_up(np.maximum(W["vmax06"][m], 0), L.LADDER8_KMH)
    bump = rng.random(len(b)) < 0.52
    b2 = np.minimum(b + bump, 7)
    unk = rng.random(len(b)) < 0.45
    X = np.hstack([base, onehot(b2, 8) * (~unk)[:, None]])
    r0, _ = oof_r2(y, base, folds)
    rA, _ = oof_r2(y, np.hstack([base, onehot(b, 8)]), folds)
    rB, _ = oof_r2(y, np.hstack([base, onehot(b2, 8)]), folds)
    rC, _ = oof_r2(y, X, folds)
    b4 = L.snap_up(np.maximum(W["vmax06"][m], 0), L.LADDER4_KMH)
    rD, _ = oof_r2(y, np.hstack([base, onehot(b4, 4)]), folds)
    F["ceiling_candidates_leak"] = {
        "r2_v0": round(r0, 4),
        "win_0_6s_max_bin8": {"r2": round(rA, 4), "recovered": round((rA - r0) / (1 - r0), 4)},
        "win_0_6s_max_bin4": {"r2": round(rD, 4), "recovered": round((rD - r0) / (1 - r0), 4)},
        "win_0_6s_max_bin8_plus1_p0.52": {"r2": round(rB, 4), "recovered": round((rB - r0) / (1 - r0), 4)},
        "win_0_6s_max_bin8_plus1_p0.52_unknown_p0.45": {"r2": round(rC, 4), "recovered": round((rC - r0) / (1 - r0), 4)}}
    res["F_extras"] = F

    res["runtime_s"] = round(time.time() - t_start, 1)
    json.dump(res, open(os.path.join(OUT, "d4_measure.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
