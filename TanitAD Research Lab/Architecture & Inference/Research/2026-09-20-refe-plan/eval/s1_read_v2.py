#!/usr/bin/env python3
"""S1 reader v2 for PREREG_VDAC_P6_A5 (DRAFT, section 2.3).  CPU only.  Tier T0-like (head perception on banked frames; never driving performance).

v2 = the FROZEN S1a reader `s1_read.py` (sha256 58aa3a77b976a6fb4b08b4bc5126d572e8be8ef1fc37926d9c9ce0983eed718e) IMPORTED UNCHANGED (sha asserted at import) + exactly these reader changes
(A5 draft 2.3, nothing else):
  (i)   a MANIFEST-DRIVEN ARM TABLE: arm names are no longer hardcoded ("s0" / "shuf" / "init0" -> any names); each trained arm names its own SHUF control and its own initial head;
        the verdict function (frozen v1 `verdict`) is applied PER TRAINED ARM with that arm's own SHUF as C-SHUF and its own y-flip as C-FLIP.
  (ii)  PAIRED blocks X minus REF (HB-s0 minus HA-s0): S1-P1 and S1-Q differences through the SAME log resamples (`diff_block`); X-LIFT = both lower 95 % bounds > 0.
  (iii) HC's 1 m grid: an arm with cell 1.0 ([N,80,64] logits).  S1-P1 is read on the 2 m REFERENCE grid (the four 1 m logits of a 2 m cell averaged in PROBABILITY space, mask = mean sigmoid > 0.5, the
        same 2 m label mask); S1-Q uses the 1 m mask (EDT at cell 1.0, bilinear at 1 m); the 1 m IoU vs the 1 m label mask is reported (not deciding).
  (iv)  an arm may carry a supplied `in_sample_iou` (Stage 0's output, sha-verified) so the frozen `verdict()` can split S1-FIT / S1-WEAK (A5 draft 1).
Everything else (metrics, estimator, seed, bars, geometry, outcome table) is the frozen v1 code, called, not copied.  On the S1a inputs v2 must reproduce `s1a_read.json` digit for digit
(`regression_identity.py`).  Legacy mode: a manifest without an "arms" table, or run_read(d) with d["arms"] keys s0 / shuf / init0, is read exactly as v1 reads it.

Manifest v2 (superset of v1's): "arms": [{"name","role" in {trained,shuf,init},"files":[2 file keys],"cell":2.0|1.0,"shuf":<name>,"init":<name>,"in_sample_iou_file":<file key>}], "pairs": [{"x","ref"}].
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

TOOL = "vdac_s1_read/2"
V1_SHA256 = "58aa3a77b976a6fb4b08b4bc5126d572e8be8ef1fc37926d9c9ce0983eed718e"
V1_REL = os.path.join("raw", "2026-11-09-scorer1p6-prereg", "vdac", "s1a_read", "s1_read.py")
MUTATIONS_V2 = ("unpaired_delta",)       # deliberate regression of v2's own addition (the paired delta through DIFFERENT resamples); the tests switch it on, a real read never does
EPS_P = 1e-12


# ======================================================================== the frozen reader, imported with its sha asserted
def _sha(path, normalise_lf=False):
    with open(path, "rb") as f:
        b = f.read()
    if normalise_lf:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def find_v1(explicit=None):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [explicit, os.environ.get("S1_READ_V1"), os.path.join(here, "s1_read.py"), os.path.join(here, "..", V1_REL), os.path.join(here, "..", "..", V1_REL)]
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise SystemExit("REFUSED: the frozen S1a reader s1_read.py was not found (set S1_READ_V1 or place it beside this file): no fallback")


def load_v1(explicit=None):
    p = find_v1(explicit)
    raw, lf = _sha(p), _sha(p, True)
    if V1_SHA256 not in (raw, lf):
        raise SystemExit(f"REFUSED: {p} sha256 {raw} (LF-normalised {lf}) != the frozen s1_read.py {V1_SHA256}")
    name = "s1_read_v1_frozen"
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) and os.path.abspath(sys.modules[name].__file__) == p:
        return sys.modules[name], p
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m, p


_V1, V1_PATH = load_v1()
R = _V1


def sha256_file(p):
    return R.sha256_file(p)


# ======================================================================== (iii) the 1 m grid helpers
def cell_grid(cell):
    """(nx, ny, fine cells per coarse cell) of a head grid on the frozen extent x [-8, 72), y [-32, 32)."""
    b = round(cell / R.FINE_RES)
    if abs(b * R.FINE_RES - cell) > 1e-9 or R.NXF % b or R.NYF % b:
        raise ValueError(f"cell {cell} m does not tile the 0.5 m label grid")
    return R.NXF // b, R.NYF // b, b


def pool_to_2m(logit, sdf):
    """[N,80,64] 1 m head -> the 2 m reference grid: probability-space mean of the 2x2 sigmoids; the returned `logit` is p - 0.5 (a monotone stand-in: > 0 iff mean sigmoid > 0.5, the card's fixed rule);
    the SDF channel (reported-only near-boundary MAE) is the 2x2 block mean."""
    lg = np.asarray(logit, np.float64)
    n, nx, ny = lg.shape
    p = 1.0 / (1.0 + np.exp(-lg))
    p2 = p.reshape(n, nx // 2, 2, ny // 2, 2).mean((2, 4))
    s2 = np.asarray(sdf, np.float64).reshape(n, nx // 2, 2, ny // 2, 2).mean((2, 4))
    return (p2 - 0.5), s2


def block_mask(label_i8, b):
    """[N,160,128] int8 SDF -> [N, 160/b, 128/b] bool mask (>= half of the b x b fine cells have SDF > 0).  b = 4 is the frozen block_targets mask."""
    s = np.asarray(label_i8).astype(np.float32) * R.SDF_UNIT_M
    n = s.shape[0]
    frac = (s > 0).astype(np.float32).reshape(n, R.NXF // b, b, R.NYF // b, b).mean((2, 4))
    return frac >= 0.5


def dk_from_masks_cell(masks, pts, cell):
    """dk_from_masks of the frozen reader generalised to any grid: masks [N,nx,ny] bool at `cell` m -> (d_k [N,M] (+inf: no point inside), n_inside).  cell = 2.0 is byte-identical to the frozen call."""
    N, M = pts.shape[:2]
    dk = np.empty((N, M))
    n_in = np.empty((N, M), np.int64)
    for n in range(N):
        fld = R.sdf_from_mask(masks[n], cell)
        v, ok = R.sample_bilinear(fld, pts[n], R.X0, R.Y0, cell)
        dk[n] = np.round(np.where(ok, v, np.inf).min(axis=1), 9)
        n_in[n] = ok.sum(axis=1)
    return dk, n_in


def iou_band_cell(logit, label_i8, valid, cell, lo, hi):
    """per-frame inter / union [N] of `logit > 0` vs the `cell`-m label mask over cells whose x centre is in [lo, hi) (the 1 m IoU the draft reports for HC)."""
    nx, ny, b = cell_grid(cell)
    tm = block_mask(label_i8, b)
    pm = np.asarray(logit) > 0
    xc = R.X0 + (np.arange(nx) + 0.5) * cell
    k = np.broadcast_to(((xc >= lo) & (xc < hi))[None, :, None], pm.shape) & np.asarray(valid, bool)[:, None, None]
    return (tm & pm & k).sum((1, 2)).astype(np.float64), ((tm | pm) & k).sum((1, 2)).astype(np.float64)


# ======================================================================== (i) the arm table
def default_arm_spec(d):
    """legacy keys (v1): s0 -> HA-s0, shuf -> HA-SHUF, init0 -> HA-s0-initial."""
    spec = [{"key": "s0", "name": "HA-s0", "role": "trained", "cell": 2.0, "shuf": "shuf", "init": "init0" if "init0" in d["arms"] else None},
            {"key": "shuf", "name": "HA-SHUF", "role": "shuf", "cell": 2.0}]
    if "init0" in d["arms"]:
        spec.append({"key": "init0", "name": "HA-s0-initial", "role": "init", "cell": 2.0})
    return spec


def check_spec(spec, d):
    keys = [a["key"] for a in spec]
    names = [a["name"] for a in spec]
    if len(set(keys)) != len(keys) or len(set(names)) != len(names):
        raise ValueError("arm table: duplicate key or name")
    for a in spec:
        if a["role"] not in ("trained", "shuf", "init"):
            raise ValueError(f"arm {a['name']}: role {a['role']!r} is not trained / shuf / init")
        if a["key"] not in d["arms"]:
            raise ValueError(f"arm {a['name']}: no arrays under key {a['key']!r}")
        nx, ny, _ = cell_grid(a.get("cell", 2.0))
        if np.asarray(d["arms"][a["key"]]["logit"]).shape[1:] != (nx, ny):
            raise ValueError(f"arm {a['name']}: logit shape {np.asarray(d['arms'][a['key']]['logit']).shape} is not [N,{nx},{ny}] for cell {a.get('cell', 2.0)}")
    if not any(a["role"] == "trained" for a in spec):
        raise ValueError("arm table: no trained arm")
    for a in spec:
        if a["role"] == "trained":
            for ref in ("shuf", "init"):
                if a.get(ref) is not None and a[ref] not in keys:
                    raise ValueError(f"arm {a['name']}: {ref} -> {a[ref]!r} is not in the table")


def x_row(p1, q, f_hold, cshuf, cflip, lv_iou, city_ious, lift):
    """the A5 draft section 4 rows for an arm X, mechanically (reported beside the frozen S1 verdict; INCOMPLETE if an input is missing, never substituted)."""
    if cshuf is None or cflip is None:
        return "INCOMPLETE"
    if cshuf >= R.BAR_CSHUF or cflip >= R.BAR_CFLIP:
        return "INVALID"
    p1_ok, q_ok = p1 >= R.BAR_P1, q >= R.BAR_Q
    if p1_ok and q_ok:
        return "INCOMPLETE" if f_hold is None else ("X-GO" if f_hold else "UNNAMED")
    if q_ok:
        non_lv = [v for c, v in city_ious.items() if c != "LV"]
        return "X-COMP" if (lv_iou >= R.COMP_LV_MIN and any(v < R.COMP_NONLV_MAX for v in non_lv)) else "UNNAMED"
    if p1_ok:
        return "X-BOUND"
    if lift is None:
        return "INCOMPLETE"
    return "X-LIFT" if lift else "X-NULL"


# ======================================================================== the whole read, on arrays
def run_read(d, n_boot=R.N_BOOT, seed=R.SEED, mutate=None, arms=None, pairs=None):
    """d as v1 (frame_log, city_of_log, label, valid, poses20, dac, set_valid, all_logs) with d["arms"] = {key: {logit, sdf}}; `arms` the arm table (default: v1's s0 / shuf / init0);
    `pairs` [{"x": name, "ref": name}] -> the v1 result dict + "per_arm" and "paired"."""
    t0 = time.time()
    spec = arms if arms is not None else default_arm_spec(d)
    check_spec(spec, d)
    by_name = {a["name"]: a for a in spec}
    by_key = {a["key"]: a for a in spec}
    logs = np.asarray(d["frame_log"]).astype(str)
    grp = R.Groups(logs, d["city_of_log"])
    W = R.boot_weights(grp.G, n_boot, seed)
    res = {"tool": TOOL, "n_boot": n_boot, "seed": seed, "mutate": mutate, "n_frames": int(len(logs)), "n_groups": grp.G, "groups": grp.logs,
           "city_logs": {c: int((grp.city == c).sum()) for c in R.CITY_WEIGHT}, "missing_inputs": [], "arms": {}}
    pts = R.footprint_points(d["poses20"])
    frac_true, sdf_true = R.block_targets(d["label"])
    true_mask = frac_true >= 0.5
    mut = mutate if mutate in R.MUTATIONS else None
    dk_cache = {}
    reps_of = {}
    prim_spec = next(a for a in spec if a["role"] == "trained")

    def prep(a):
        arr = d["arms"][a["key"]]
        if abs(a.get("cell", 2.0) - R.CELL) < 1e-9:
            return arr["logit"], arr["sdf"]
        if abs(a["cell"] - 1.0) < 1e-9:
            return pool_to_2m(arr["logit"], arr["sdf"])
        raise ValueError(f"arm {a['name']}: cell {a['cell']} m is not 2.0 or 1.0")

    def arm_read(a, with_q=True, with_flip=False):
        arr = d["arms"][a["key"]]
        cell = a.get("cell", 2.0)
        lg2, sd2 = prep(a)
        st = R.frame_stats(lg2, sd2, d["label"], d["valid"])
        out = {"p1": {}, "p2": None, "q": None}
        for band in ("0-40", "0-20", "20-40", "40-72"):
            blk, reps, _ = R.p1_block(st, grp, W, band, mutate=mut)
            out["p1"][band] = blk
            if band == "0-40":
                reps_of[(a["name"], "p1")] = (reps, blk["point"])
        use = st["valid"]
        gi = grp.idx[use]
        pa, ps, pn = R.by_group(gi, grp.G, st["p2_abs"][use], st["p2_sq"][use], st["p2_n"][use])
        out["p2"] = {"near_boundary_mae_m": float(pa.sum() / pn.sum()) if pn.sum() else float("nan"), "near_boundary_rms_m": float(math.sqrt(ps.sum() / pn.sum())) if pn.sum() else float("nan"),
                     "n_cells": float(pn.sum()), "half_band_m": R.NEAR_BOUNDARY_M}
        if with_q:
            m = np.asarray(arr["logit"]) > 0
            dk, n_in = R.dk_from_masks(m, pts) if abs(cell - R.CELL) < 1e-9 else dk_from_masks_cell(m, pts, cell)
            dk_cache[a["name"]] = dk
            q, reps = R.q_block(dk, d["dac"], d["set_valid"], grp, W, grp.idx)
            q["outside_extent_point_share"] = float(1 - n_in.sum() / (n_in.size * pts.shape[2]))
            q["proposals_with_no_point_inside"] = int((n_in == 0).sum())
            out["q"] = q
            reps_of[(a["name"], "q")] = (reps, q["point"])
        if with_flip:
            mf = np.asarray(arr["logit"])[:, :, ::-1] > 0
            dkf, _ = R.dk_from_masks(mf, pts) if abs(cell - R.CELL) < 1e-9 else dk_from_masks_cell(mf, pts, cell)
            qf, _ = R.q_block(dkf, d["dac"], d["set_valid"], grp, W, grp.idx)
            out["q_yflip"] = qf
        if abs(cell - 1.0) < 1e-9:            # reported, not deciding: the 1 m IoU against the 1 m label mask, [0, 40) m, city-reweighted, log-cluster interval
            it, un = iou_band_cell(arr["logit"], d["label"], d["valid"], 1.0, *R.K2_BAND)
            ig, ug = R.by_group(gi, grp.G, it[use], un[use])
            pt1, rp1, pc1 = R.city_iou_read(ig, ug, grp, W, reweight=True)
            ci1, nn1 = R.ci95(rp1)
            out["p1_1m_grid"] = {"point": pt1, "ci95": ci1, "per_city": pc1, "n_nan_resamples": nn1, "note": "[0,40) m, `logit > 0` vs the 1 m label mask, city-reweighted (reported, not deciding)"}
        return out

    reads = {}
    for a in spec:
        reads[a["name"]] = arm_read(a, with_q=True, with_flip=(a["role"] == "trained"))
        res["arms"][a["name"]] = {k: v for k, v in reads[a["name"]].items()}
    primary = reads[prim_spec["name"]]

    # reported, not deciding: true-field AUC on the same proposals, sigma_b (primary trained arm, 2 m reference mask)
    dk_true, _ = R.dk_from_masks(true_mask, pts)
    qt, _ = R.q_block(dk_true, d["dac"], d["set_valid"], grp, W, grp.idx)
    res["true_field_auc_2m"] = qt
    pm = np.asarray(prep(prim_spec)[0]) > 0
    err = []
    for n in np.flatnonzero(d["valid"]):
        fp, ft = R.sdf_from_mask(pm[n], R.CELL), R.sdf_from_mask(true_mask[n], R.CELL)
        near = np.abs(ft) < R.NEAR_BOUNDARY_M
        err.append((fp - ft)[near])
    err = np.concatenate(err) if err else np.zeros(0)
    res["sigma_b_m"] = {"rms_mask_edt_field_vs_true_mask_edt_field": float(math.sqrt((err ** 2).mean())) if err.size else float("nan"), "n_cells": int(err.size),
                        "note": "mask-derived EDT field of the head vs the EDT field of the true 2 m mask, cells with |SDF_gt| < 3 m (card 5.1 step 5, same pipeline)"}

    # (ii) paired blocks, same resamples
    paired = []
    for pr in (pairs or []):
        xn, rn = pr["x"], pr["ref"]
        if xn not in by_name or rn not in by_name:
            raise ValueError(f"pair {pr}: an arm is not in the table")
        rep_x1, pt_x1 = reps_of[(xn, "p1")]
        rep_r1, pt_r1 = reps_of[(rn, "p1")]
        rep_xq, pt_xq = reps_of[(xn, "q")]
        rep_rq, pt_rq = reps_of[(rn, "q")]
        if mutate == "unpaired_delta":        # regression arm of v2's own addition: the reference read through ANOTHER set of resamples
            W2 = R.boot_weights(grp.G, n_boot, seed + 1)
            st_r = R.frame_stats(prep(by_name[rn])[0], prep(by_name[rn])[1], d["label"], d["valid"])
            _, rep_r1, _ = R.p1_block(st_r, grp, W2, "0-40")
            _, rep_rq = R.q_block(dk_cache[rn], d["dac"], d["set_valid"], grp, W2, grp.idx)
        p1d = R.diff_block(rep_x1, pt_x1, rep_r1, pt_r1)
        qd = R.diff_block(rep_xq, pt_xq, rep_rq, pt_rq)
        paired.append({"x": xn, "ref": rn, "p1": p1d, "q": qd, "x_lift": bool(p1d["lower95"] > 0 and qd["lower95"] > 0),
                       "note": "paired X minus REF on the same frames through the SAME log resamples; X-LIFT = both lower 95 % bounds > 0 (necessary, not sufficient: H-ESTIM-SEED-1)"})
    res["paired"] = paired

    # per trained arm: S1-F, controls, bars, frozen verdict, A5 section-4 row
    per_arm = {}
    primary_missing = []
    for a in spec:
        if a["role"] != "trained":
            continue
        nm = a["name"]
        rd = reads[nm]
        s1f = None
        if a.get("init") is not None:
            ini = by_key[a["init"]]
            s1f = {"p1": R.diff_block(*reps_of[(nm, "p1")], *reps_of[(ini["name"], "p1")]),
                   "q": R.diff_block(*reps_of[(nm, "q")], *reps_of[(ini["name"], "q")])}
            s1f["hold"] = bool(s1f["p1"]["lower95"] > 0 and s1f["q"]["lower95"] > 0)
        shuf_p1 = reads[by_key[a["shuf"]]["name"]]["p1"]["0-40"]["point"] if a.get("shuf") is not None else None
        p1 = rd["p1"]["0-40"]["point"]
        q = rd["q"]["point"]
        city = rd["p1"]["0-40"]["per_city"]
        cflip = rd["q_yflip"]["point"]
        ins = a.get("in_sample_iou")
        v, why, miss_v = R.verdict(p1, q, None if s1f is None else s1f["hold"], shuf_p1, cflip, city.get("LV", float("nan")), city, ins)
        lift = next((pp["x_lift"] for pp in paired if pp["x"] == nm), None)
        ctr = {"C-SHUF": None if shuf_p1 is None else {"value": shuf_p1, "bar": f"< {R.BAR_CSHUF}", "ok": bool(shuf_p1 < R.BAR_CSHUF)},
               "C-FLIP": {"value": cflip, "bar": f"< {R.BAR_CFLIP}", "ok": bool(cflip < R.BAR_CFLIP)}}
        bars = {"S1-P1": {"value": p1, "ci95": rd["p1"]["0-40"]["ci95"], "bar": f">= {R.BAR_P1}", "holds": bool(p1 >= R.BAR_P1)},
                "S1-Q": {"value": q, "ci95": rd["q"]["ci95"], "bar": f">= {R.BAR_Q}", "holds": bool(q >= R.BAR_Q)},
                "S1-F": None if s1f is None else {"p1_lower95": s1f["p1"]["lower95"], "q_lower95": s1f["q"]["lower95"], "bar": "both lower bounds > 0", "holds": s1f["hold"]},
                "S1-R": "not read (S1b = HA-s1 is trained only if HA-s0 reaches S1-GO)"}
        per_arm[nm] = {"S1_F": s1f, "controls": ctr, "bars": bars, "in_sample_iou": ins, "verdict": v, "verdict_reason": why, "verdict_missing": miss_v, "next": R.NEXT[v],
                       "x_row": x_row(p1, q, None if s1f is None else s1f["hold"], shuf_p1, cflip, city.get("LV", float("nan")), city, lift), "x_lift": lift}
        if a is prim_spec:
            primary_missing = miss_v
    res["per_arm"] = per_arm

    pn = prim_spec["name"]
    pa_ = per_arm[pn]
    if pa_["S1_F"] is None:
        res["missing_inputs"].append("MISSING-INPUT initial_head_predictions (the same seed's never-trained head on the H48 frames; needed for S1-F)")
    res["S1_F"] = pa_["S1_F"]
    if pa_["in_sample_iou"] is None:
        res["missing_inputs"].append("MISSING-INPUT in_sample_iou (head predictions on training frames were not exported; needed only to split S1-FIT from S1-WEAK)")
    res["missing_inputs"] += ["MISSING-INPUT p6_dac_logit (P6's own DAC logit is not in the export; descriptive AUC not computed)",
                              "MISSING-INPUT navtrain_metric_cache_t0_ego_states (S1-QT, tracked d_k, unavailable; reported, not deciding)"]

    # sensitivity: all-48 groups (logs without frames carry zero weight) -- CI only (primary trained arm)
    if d.get("all_logs"):
        g48 = R.Groups(logs, d["city_of_log"], group_logs=d["all_logs"])
        W48 = R.boot_weights(g48.G, n_boot, seed)
        lg_p, sd_p = prep(prim_spec)
        b48, _, _ = R.p1_block(R.frame_stats(lg_p, sd_p, d["label"], d["valid"]), g48, W48, "0-40")
        q48, _ = R.q_block(dk_cache[pn], d["dac"], d["set_valid"], g48, W48, g48.idx)
        res["sensitivity_all48_groups"] = {"n_groups": g48.G, "s1_p1_ci95": b48["ci95"], "s1_q_ci95": q48["ci95"]}

    res["controls"] = {k: v for k, v in pa_["controls"].items() if v is not None}
    res["bars"] = pa_["bars"]
    res["verdict"], res["verdict_reason"], res["verdict_missing"] = pa_["verdict"], pa_["verdict_reason"], pa_["verdict_missing"]
    res["next"] = pa_["next"]
    res["wall_s"] = time.time() - t0
    return res


# ======================================================================== real-data loading (v1's checks, arm table from the manifest)
def arms_from_manifest(man):
    """-> (spec list with `files`, pairs).  No "arms" in the manifest: v1's table (s0 / shuf / init0 by the head_* file keys)."""
    if man.get("arms"):
        spec = []
        for a in man["arms"]:
            e = {"key": a["name"], "name": a["name"], "role": a["role"], "cell": float(a.get("cell", 2.0)), "files": list(a["files"])}
            for k in ("shuf", "init", "in_sample_iou_file"):
                if a.get(k) is not None:
                    e[k] = a[k]
            spec.append(e)
        return spec, list(man.get("pairs", []))
    spec = []
    if "head_s0_a" in man["files"]:
        spec.append({"key": "s0", "name": "HA-s0", "role": "trained", "cell": 2.0, "files": ["head_s0_a", "head_s0_b"], "shuf": "shuf" if "head_shuf_a" in man["files"] else None,
                     "init": "init0" if "head_init0_a" in man["files"] else None})
    if "head_shuf_a" in man["files"]:
        spec.append({"key": "shuf", "name": "HA-SHUF", "role": "shuf", "cell": 2.0, "files": ["head_shuf_a", "head_shuf_b"]})
    if "head_init0_a" in man["files"]:
        spec.append({"key": "init0", "name": "HA-s0-initial", "role": "init", "cell": 2.0, "files": ["head_init0_a", "head_init0_b"]})
    return spec, []


def load_real(man):
    """v1's `load_real` with the arm loop driven by the manifest's arm table.  -> (d for run_read, audit).  d["_spec"], d["_pairs"] carry the table."""
    audit = {}
    cdir = man["cards_dir"]
    for f, sha in R.CARD_FILES.items():
        got = sha256_file(os.path.join(cdir, f))
        if got != sha:
            raise SystemExit(f"REFUSED: {f} sha256 {got} != frozen {sha}")
    audit["cards"] = "5 cards verified by sha256"
    for k, e in man["files"].items():
        got = sha256_file(e["path"])
        if got != e["sha256"]:
            raise SystemExit(f"REFUSED: {k} sha256 {got} != manifest {e['sha256']}")
    audit["inputs"] = f"{len(man['files'])} input files verified by sha256"
    h48_bytes = open(man["files"]["h48_logs"]["path"], "rb").read()
    if hashlib.sha256(h48_bytes).hexdigest() != R.H48_SHA:
        raise SystemExit("REFUSED: H48 list sha256")
    h48 = h48_bytes.decode().split()
    city_a3 = R.parse_a3_cities(open(os.path.join(cdir, "PREREG_VDAC_P6_A3.md"), encoding="utf-8").read())
    if sorted(city_a3) != sorted(h48):
        raise SystemExit("REFUSED: A3's table does not list exactly the H48 logs")
    sm = json.load(open(os.path.join(man["store"], "manifest.json")))
    if not str(sm["store_sha256"]).startswith(R.STORE_SHA_PREFIX):
        raise SystemExit("REFUSED: label store sha256")
    city_store = {l: R.MAP_CITY[sm["files"][f"shards/{l}.aux1"]["map"]] for l in h48}
    if city_store != city_a3:
        raise SystemExit("REFUSED: store manifest city != A3 city for some H48 log")
    audit["city_per_log"] = "A3 table == store manifest map, 48 logs"
    ex = np.load(man["files"]["export"]["path"], allow_pickle=False)
    exj = json.load(open(man["files"]["export_json"]["path"]))
    if exj["h48_sha256"] != R.H48_SHA or str(ex["h48_sha256"]) != R.H48_SHA:
        raise SystemExit("REFUSED: export H48 sha")
    if [str(x) for x in ex["comp_order"]] != ["nc", "dac", "ep", "ttc", "c", "ddc"]:
        raise SystemExit("REFUSED: comp_order")
    el, et = [str(x) for x in ex["log"]], [str(x) for x in ex["token"]]
    keys = list(zip(el, et))
    if len(set(keys)) != len(keys):
        raise SystemExit("REFUSED: duplicate (log, token) in the export")
    kpos = {k: i for i, k in enumerate(keys)}
    spec, pairs = arms_from_manifest(man)
    arms = {}
    for a in spec:
        nx, ny, _ = cell_grid(a["cell"])
        files = a["files"]
        if any(fk not in man["files"] for fk in files):
            raise SystemExit(f"REFUSED: arm {a['name']} names a file key absent from the manifest: {files}")
        lg = np.zeros((len(keys), nx, ny), np.float32)
        sd = np.zeros((len(keys), nx, ny), np.float32)
        got = np.zeros(len(keys), bool)
        for fk in files:
            z = np.load(man["files"][fk]["path"], allow_pickle=False)
            if not (np.asarray(z["step"]) == 0).all():
                raise SystemExit(f"REFUSED: {fk} holds rows with step != 0")
            if not (np.isfinite(z["logit"]).all() and np.isfinite(z["sdf_m"]).all()):
                raise SystemExit(f"REFUSED: {fk} holds non-finite outputs")
            if z["logit"].shape[1:] != (nx, ny):
                raise SystemExit(f"REFUSED: {fk} logit shape {z['logit'].shape} is not [N,{nx},{ny}] (arm {a['name']}, cell {a['cell']})")
            for i, k in enumerate(zip([str(x) for x in z["log"]], [str(x) for x in z["token"]])):
                j = kpos.get(k)
                if j is None:
                    raise SystemExit(f"REFUSED: {fk} key {k} is not in the export")
                if got[j]:
                    raise SystemExit(f"REFUSED: {fk} key {k} appears twice")
                got[j] = True
                lg[j], sd[j] = z["logit"][i], z["sdf_m"][i]
        if not got.all():
            raise SystemExit(f"REFUSED: arm {a['name']} does not cover the export's {len(keys)} keys ({int(got.sum())})")
        arms[a["key"]] = {"logit": lg, "sdf": sd}
        if a.get("in_sample_iou_file"):
            ent = man["files"].get(a["in_sample_iou_file"])
            if ent is None:
                raise SystemExit(f"REFUSED: arm {a['name']} in_sample_iou_file {a['in_sample_iou_file']!r} is not in the manifest")
            js = json.load(open(ent["path"]))
            if not isinstance(js.get("in_sample_iou"), float) or not math.isfinite(js["in_sample_iou"]):
                raise SystemExit(f"REFUSED: {ent['path']} carries no finite float in_sample_iou")
            a["in_sample_iou"] = js["in_sample_iou"]
    shard_logs = sorted(set(el))
    lab = np.zeros((len(keys), R.NXF, R.NYF), np.int8)
    valid = np.zeros(len(keys), bool)
    n_shard_tokens = 0
    for l in shard_logs:
        rel = f"shards/{l}.aux1"
        p = os.path.join(man["store"], rel)
        if sha256_file(p) != sm["files"][rel]["sha256"]:
            raise SystemExit(f"REFUSED: shard {rel} does not hash to the store manifest")
        _, labs = R.read_shard(p)
        n_shard_tokens += len(labs)
        for i in np.flatnonzero(np.asarray(el) == l):
            t = et[i]
            if t in labs:
                lab[i], valid[i] = labs[t], True
    audit["labelled_frames_in_the_48_shards"] = int(n_shard_tokens)
    dac = ex["comp"][:, :, 1].astype(np.float64)
    if not np.isin(dac, (0.0, 1.0)).all():
        raise SystemExit("REFUSED: DAC sub-scores are not exactly 0/1")
    d = {"frame_log": np.asarray(el), "city_of_log": dict(city_a3), "label": lab, "valid": valid,
         "poses20": ex["traj"].astype(np.float64), "dac": dac, "set_valid": np.ones(len(keys), bool), "arms": arms,
         "all_logs": sorted(h48), "_spec": spec, "_pairs": pairs}
    audit.update({"frames_forwarded": int(len(keys)), "logs_with_frames": len(shard_logs), "h48_logs_without_frames": sorted(set(h48) - set(shard_logs)),
                  "frames_with_label": int(valid.sum()), "frames_without_label": int((~valid).sum()), "export_banks": {str(b): int((ex["bank"] == b).sum()) for b in np.unique(ex["bank"])},
                  "sets_with_p6_pdm_label": int(len(keys)), "arms_present": sorted(arms),
                  "city_logs_with_frames": {c: int(sum(1 for l in shard_logs if city_a3[l] == c)) for c in R.CITY_WEIGHT}})
    if man.get("arms"):
        audit["arm_table"] = [{k: v for k, v in a.items() if k in ("name", "role", "cell", "files", "shuf", "init")} for a in spec]
        audit["pairs"] = pairs
    return d, audit


# ======================================================================== reporting
def tables_md(r):
    L = []
    if "HA-s0" in r["arms"] and "HA-SHUF" in r["arms"]:
        L.append(R.tables_md(r).rstrip("\n"))
    else:
        for nm, pa in r["per_arm"].items():
            a = r["arms"][nm]
            L += [f"### {nm}: verdict (mechanical, frozen v1 table): **{pa['verdict']}** - {pa['verdict_reason']}; A5 row **{pa['x_row']}**", "", "| bar | value | 95 % CI | bar | holds |", "|---|---|---|---|---|"]
            for k in ("S1-P1", "S1-Q"):
                b = pa["bars"][k]
                L.append(f"| {k} | {R.fmt(b['value'])} | [{R.fmt(b['ci95'][0])}, {R.fmt(b['ci95'][1])}] | {b['bar']} | {b['holds']} |")
            f = pa["bars"]["S1-F"]
            L.append("| S1-F | MISSING-INPUT | - | both lower bounds > 0 | unread |" if f is None else f"| S1-F | P1 lower {R.fmt(f['p1_lower95'])}, Q lower {R.fmt(f['q_lower95'])} | - | both > 0 | {f['holds']} |")
            for k, v in pa["controls"].items():
                L.append(f"| {k} | MISSING-INPUT | - | - | unread |" if v is None else f"| {k} | {R.fmt(v['value'])} | - | {v['bar']} | {v['ok']} |")
            L += ["", "per city (S1-P1 [0,40) IoU): " + ", ".join(f"{c} {R.fmt(v)}" for c, v in a["p1"]["0-40"]["per_city"].items()), ""]
    for nm, pa in r["per_arm"].items():
        if "HA-s0" in r["arms"] and "HA-SHUF" in r["arms"] and nm == "HA-s0":
            continue
        if ("HA-s0" in r["arms"] and "HA-SHUF" in r["arms"]):
            L += ["", f"### {nm}: verdict (mechanical, frozen v1 table): **{pa['verdict']}** - {pa['verdict_reason']}; A5 row **{pa['x_row']}**"]
    if r["paired"]:
        L += ["", "| paired X minus REF | S1-P1 delta | 95 % CI | S1-Q delta | 95 % CI | X-LIFT |", "|---|---|---|---|---|---|"]
        for p in r["paired"]:
            L.append(f"| {p['x']} - {p['ref']} | {R.fmt(p['p1']['point'])} | [{R.fmt(p['p1']['ci95'][0])}, {R.fmt(p['p1']['ci95'][1])}] | {R.fmt(p['q']['point'])} | [{R.fmt(p['q']['ci95'][0])}, {R.fmt(p['q']['ci95'][1])}] | {p['x_lift']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="inputs manifest json (cards_dir, store, files{key:{path,sha256}}, optional arms / pairs)")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--n-boot", type=int, default=R.N_BOOT)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    d, audit = load_real(man)
    print("COUNTS BEFORE THE READ", json.dumps(audit), flush=True)
    r = run_read(d, n_boot=a.n_boot, arms=d.pop("_spec"), pairs=d.pop("_pairs"))
    r["audit"] = audit
    r["reader_sha256"] = sha256_file(os.path.abspath(__file__))
    r["frozen_v1_reader_sha256"] = V1_SHA256
    r["manifest_sha256"] = sha256_file(a.manifest)
    os.makedirs(a.out, exist_ok=True)
    json.dump(r, open(os.path.join(a.out, "s1_read_v2.json"), "w"), indent=1, default=float)
    open(os.path.join(a.out, "s1_read_v2_tables.md"), "w", encoding="utf-8", newline="\n").write(tables_md(r))
    print(tables_md(r))
    print("MISSING:", *r["missing_inputs"], sep="\n  ")
    print("VERDICT", r["verdict"], "|", r["verdict_reason"], "| next:", r["next"])
    return 0


# every other public name of the frozen reader stays reachable as v2.<name> (the ported v1 tests call them), without shadowing what v2 defines
for _k, _v in vars(_V1).items():
    if not _k.startswith("__") and _k not in globals():
        globals()[_k] = _v

if __name__ == "__main__":
    sys.exit(main())
