#!/usr/bin/env python3
"""S1 read of PREREG_VDAC_P6.md (FROZEN) + A1..A4: perception of the vision drivable-area head on held-out navtrain (H48).  CPU only, no navtest value is read.

Tier T0-like (head perception on banked frames; never driving performance).  Evidence class of every number it prints: MEASURED (this reader on the banked exports).

Inputs (one manifest, every file re-hashed):  the head predictions of HA-s0 and HA-SHUF (two npz each: the 24 SFT held-out logs and the 24 extra H48 logs), the export of P6's proposals and
NAVSIM sub-scores of the H48 sets, the true-label store (shards of the 48 H48 logs only, each re-hashed against the store manifest).
Metrics (card section 'S1', every one computed here, none by hand):
  S1-P1  drivable IoU, x in [0, 40) m, mask `logit > 0` vs label mask, all H48 frames the head forwarded, REWEIGHTED to navtest's city mix (LV 4,064 / BOS 3,731 / PIT 2,437 / SG 1,914 of 12,146):
         IoU = sum_c w_c * (sum inter_c / sum union_c), w_c = n_c / 12,146.  (the card does not spell out the formula; this is the reader's reading of "city-reweighted", frozen before the run)
  S1-Q   pooled AUC of -d_k for NAVSIM DAC = 0 over P6's 64 proposals on H48 sets that carry P6 labels.  d_k = DESIGN 5.1: P6's 20 poses at 5 Hz -> NAVSIM's 8 at 0.5 s (to_navsim) -> 41 poses at
         0.1 s -> AUXVETO footprint, 4 corners = 164 points -> the head's MASK (logit > 0) on its 2 m grid -> Euclidean distance transform (half-cell corrected, + inside) -> bilinear samples,
         min over the points inside the extent; no point inside -> d_k = +inf (not vetoed).  POSE-ONLY.
  S1-F   (trained - the same seed's INITIAL head) on S1-P1 and on S1-Q, both 95 % lower bounds > 0, through the SAME log resamples.  No initial-head prediction in the export -> MISSING-INPUT.
  C-SHUF HA-SHUF's S1-P1 point < 0.50 else INVALID.     C-FLIP HA-s0's mask y-flipped, S1-Q point < 0.80 else INVALID.
Reported, not deciding: IoU by band 0-20 / 20-40 / 40-72 m and per city, near-boundary SDF MAE and RMS (|SDF_gt| < 3 m), sigma_b, the true-field AUC on the same proposals, P6's own DAC logit AUC
(MISSING-INPUT: P6's logits are not in the export), S1-QT (MISSING-INPUT: no navtrain metric cache), per-city, in-sample IoU (MISSING-INPUT: only H48 frames were forwarded).
Estimator: paired LOG-CLUSTER bootstrap, 10,000 resamples, seed 20260927, draw = proxy_eval.boot's (`default_rng(seed)`, one `integers(0, G, G)` per resample, logs sorted by name), 95 % percentile.
Groups = the H48 logs that carry >= 1 forwarded frame (the card says 48; a log with no frame cannot be resampled meaningfully); the all-48 version is reported as a sensitivity.
The outcome table is applied MECHANICALLY by `verdict()`.  An input the card needs that the exports do not hold is emitted as `MISSING-INPUT <name>` and is never substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
import zlib

import numpy as np

TOOL = "vdac_s1_read/1"
N_BOOT, SEED = 10000, 20260927
# ---- card bars (PREREG_VDAC_P6.md section 2, S1) ----
BAR_P1, BAR_Q, BAR_CSHUF, BAR_CFLIP = 0.80, 0.90, 0.50, 0.80
COMP_LV_MIN, COMP_NONLV_MAX, FIT_IN_SAMPLE = 0.80, 0.70, 0.85
CITY_N = {"LV": 4064, "BOS": 3731, "PIT": 2437, "SG": 1914}
CITY_TOTAL = 12146
CITY_WEIGHT = {c: n / CITY_TOTAL for c, n in CITY_N.items()}
MAP_CITY = {"us-nv-las-vegas-strip": "LV", "us-ma-boston": "BOS", "us-pa-pittsburgh-hazelwood": "PIT", "sg-one-north": "SG"}
BANDS = ((0.0, 20.0), (20.0, 40.0), (40.0, 72.0))
BAND_NAMES = ("0-20", "20-40", "40-72")
K2_BAND = (0.0, 40.0)
NEAR_BOUNDARY_M = 3.0
# ---- geometry (aux_label_store / aux_drivable_head / navsim_dac / s0_read, verbatim constants) ----
HALF_LENGTH = (4.049 + 1.127) / 2.0
HALF_WIDTH = 2.297 / 2.0
REAR_AXLE_TO_CENTER = (4.049 - 1.127) / 2.0
X0, Y0 = -8.0, -32.0
FINE_RES, NXF, NYF, SDF_UNIT_M = 0.5, 160, 128, 0.1
CELL, NCX, NCY, SDF_SCALE_M = 2.0, 40, 32, 6.0
BIG_M = 1.0e3
MUTATIONS = ("no_reweight", "frame_resample")          # deliberate regressions the tests switch on (never set in a real read)


# ======================================================================== small utilities
def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ======================================================================== bootstrap
def boot_weights(n_groups, n=N_BOOT, seed=SEED, mutate=None):
    """[n, G] multiplicity of each group in each resample (proxy_eval.boot's draw).  mutate='frame_resample' is the regression arm: it is NOT used by the reader."""
    rng = np.random.default_rng(seed)
    W = np.empty((n, n_groups), np.float64)
    for b in range(n):
        W[b] = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
    return W


def ci95(reps):
    r = np.asarray(reps, np.float64)
    ok = np.isfinite(r)
    if not ok.any():
        return [float("nan"), float("nan")], int((~ok).sum())
    return [float(np.percentile(r[ok], 2.5)), float(np.percentile(r[ok], 97.5))], int((~ok).sum())


def by_group(group_idx, n_groups, *arrays):
    return [np.bincount(group_idx, weights=np.asarray(a, np.float64), minlength=n_groups) for a in arrays]


# ======================================================================== label decoding (aux_label_store.read_shard / aux_drivable_head.block_stats)
def read_shard(path):
    with open(path, "rb") as f:
        if f.read(8) != b"AUXSDF1\n":
            raise ValueError("bad magic")
        (hl,) = struct.unpack("<I", f.read(4))
        head = json.loads(f.read(hl))
        base = f.tell()
        lab = {}
        for e in head["entries"]:
            f.seek(base + e["o"])
            lab[e["t"]] = np.frombuffer(zlib.decompress(f.read(e["l"])), np.int8).reshape(NXF, NYF).copy()
    return head, lab


def block_targets(label_i8):
    """[N, 160, 128] int8 SDF (0.1 m units) -> (mask fraction [N,40,32] = share of the 4x4 fine cells with SDF > 0, block-mean SDF clipped to +-6 m [N,40,32])."""
    s = label_i8.astype(np.float32) * SDF_UNIT_M
    n = s.shape[0]
    s = s.reshape(n, NXF // 4, 4, NYF // 4, 4)
    frac = (s > 0).astype(np.float32).mean((2, 4))
    sdf = np.clip(s, -SDF_SCALE_M, SDF_SCALE_M).mean((2, 4))
    return frac, sdf


def cell_x_centres():
    return X0 + (np.arange(NCX) + 0.5) * CELL


def band_cells(lo, hi):
    xc = cell_x_centres()
    return np.broadcast_to(((xc >= lo) & (xc < hi))[:, None], (NCX, NCY)).copy()


# ======================================================================== per-frame sufficient statistics (aux_s1_readout.frame_stats, numpy)
def frame_stats(logit, sdf_pred, label_i8, valid):
    """logit / sdf_pred [N,40,32]; label [N,160,128] int8; valid [N] bool (a frame without a label has no known cell).  -> per-frame arrays."""
    frac, sdf_t = block_targets(label_i8)
    tm = frac >= 0.5
    pm = np.asarray(logit) > 0
    kn = np.broadcast_to(np.asarray(valid, bool)[:, None, None], tm.shape)
    out = {"valid": np.asarray(valid, bool).copy()}
    bands = list(BANDS) + [K2_BAND]
    for b, band in enumerate(bands):
        k = kn & band_cells(*band)[None]
        out[f"inter{b}"] = (tm & pm & k).sum((1, 2)).astype(np.float64)
        out[f"union{b}"] = ((tm | pm) & k).sum((1, 2)).astype(np.float64)
    err = np.abs(np.asarray(sdf_pred, np.float64) - sdf_t.astype(np.float64))
    near = kn & (np.abs(sdf_t) < NEAR_BOUNDARY_M)
    out["p2_abs"], out["p2_sq"], out["p2_n"] = (err * near).sum((1, 2)), ((err ** 2) * near).sum((1, 2)), near.sum((1, 2)).astype(np.float64)
    return out


BAND_IDX = {"0-20": 0, "20-40": 1, "40-72": 2, "0-40": 3}


# ======================================================================== S1-P1: city-reweighted IoU through the log-cluster resamples
class Groups:
    """The resampling groups: sorted log names, each frame's group index, each group's city."""

    def __init__(self, frame_logs, city_of_log, group_logs=None):
        self.logs = sorted(set(map(str, frame_logs))) if group_logs is None else sorted(group_logs)
        pos = {l: i for i, l in enumerate(self.logs)}
        self.idx = np.asarray([pos[str(l)] for l in frame_logs], np.int64)
        self.G = len(self.logs)
        self.city = np.asarray([city_of_log[l] for l in self.logs])


def city_iou_read(inter_g, union_g, grp, W, reweight=True):
    """per-group sums -> (point, reps, per-city point).  reweight: sum_c w_c I_c/U_c ; no_reweight (regression arm): pooled sum I / sum U."""
    cities = list(CITY_WEIGHT)
    cm = {c: (grp.city == c).astype(np.float64) for c in cities}
    per_city = {}
    for c in cities:
        u = float((union_g * cm[c]).sum())
        per_city[c] = float((inter_g * cm[c]).sum() / u) if u > 0 else float("nan")
    if reweight:
        point = float(sum(CITY_WEIGHT[c] * per_city[c] for c in cities))
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = sum(CITY_WEIGHT[c] * (W @ (inter_g * cm[c])) / (W @ (union_g * cm[c])) for c in cities)
    else:
        point = float(inter_g.sum() / union_g.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = (W @ inter_g) / (W @ union_g)
    return point, np.asarray(reps), per_city


def p1_block(st, grp, W, band="0-40", mutate=None):
    """S1-P1 style read of one band for one arm -> dict (point, ci95, per_city, n_frames, n_logs)."""
    b = BAND_IDX[band]
    use = st["valid"]
    gi = grp.idx[use]
    if mutate == "frame_resample":                      # regression arm: every FRAME is its own resampling group instead of every log
        fr = np.flatnonzero(use)
        fg = type("G", (), {})()
        fg.G, fg.city = len(fr), grp.city[gi]
        inter_g, union_g = st[f"inter{b}"][use].astype(np.float64), st[f"union{b}"][use].astype(np.float64)
        point, reps, per_city = city_iou_read(inter_g, union_g, fg, boot_weights(fg.G, W.shape[0], SEED), reweight=True)
    else:
        inter_g, union_g = by_group(gi, grp.G, st[f"inter{b}"][use], st[f"union{b}"][use])
        point, reps, per_city = city_iou_read(inter_g, union_g, grp, W, reweight=(mutate != "no_reweight"))
    ci, n_nan = ci95(reps)
    return {"point": point, "ci95": ci, "per_city": per_city, "n_frames": int(use.sum()), "n_logs": int((np.bincount(gi, minlength=grp.G) > 0).sum()),
            "n_nan_resamples": n_nan}, reps, (inter_g, union_g)


def frame_resample_weights(n_frames, n=N_BOOT, seed=SEED):
    """REGRESSION ARM ONLY: per-frame multiplicities (what a wrong estimator would do)."""
    return boot_weights(n_frames, n, seed)


# ======================================================================== geometry for d_k
def to_navsim_vec(poses20, dt=0.2):
    """[..., 20, 3] at 0.2..4.0 s -> [..., 8, 3] at 0.5..4.0 s (navsim_dac.to_navsim, vectorised; the index / weight per target time are computed by the verbatim scalar rule)."""
    p = np.asarray(poses20, np.float64)
    t_src = dt * np.arange(1, p.shape[-2] + 1)
    out = np.zeros(p.shape[:-2] + (8, 3))
    for k in range(8):
        t = 0.5 * (k + 1)
        j = int(np.searchsorted(t_src, t - 1e-9))
        if j < p.shape[-2] and abs(t_src[j] - t) < 1e-9:
            out[..., k, :] = p[..., j, :]
            continue
        w = (t - t_src[j - 1]) / (t_src[j] - t_src[j - 1])
        out[..., k, :2] = (1 - w) * p[..., j - 1, :2] + w * p[..., j, :2]
        out[..., k, 2] = wrap(p[..., j - 1, 2] + w * wrap(p[..., j, 2] - p[..., j - 1, 2]))
    return out


def densify(poses8):
    """[..., 8, 3] at 0.5..4.0 s -> [..., 41, 3] at 0.0..4.0 s: linear in x, y and the UNWRAPPED heading, t0 pose = origin (s0_read.densify, verbatim)."""
    p = np.asarray(poses8, np.float64)
    lead = p.shape[:-2]
    full = np.concatenate([np.zeros(lead + (1, 3)), p], axis=-2)
    hd = np.unwrap(full[..., 2], axis=-1)
    t_q = np.arange(41) * 0.1
    j = np.minimum((t_q / 0.5 + 1e-9).astype(int), 7)
    w = (t_q - j * 0.5) / 0.5
    out = np.empty(lead + (41, 3))
    for c in (0, 1):
        out[..., c] = full[..., j, c] * (1 - w) + full[..., j + 1, c] * w
    out[..., 2] = hd[..., j] * (1 - w) + hd[..., j + 1] * w
    return out


def corners(poses):
    p = np.asarray(poses, np.float64)
    ch, sh = np.cos(p[..., 2]), np.sin(p[..., 2])
    cx, cy = p[..., 0] + REAR_AXLE_TO_CENTER * ch, p[..., 1] + REAR_AXLE_TO_CENTER * sh
    out = np.empty(p.shape[:-1] + (4, 2))
    for i, (lo, la) in enumerate(((HALF_LENGTH, HALF_WIDTH), (HALF_LENGTH, -HALF_WIDTH), (-HALF_LENGTH, HALF_WIDTH), (-HALF_LENGTH, -HALF_WIDTH))):
        out[..., i, 0] = cx + lo * ch - la * sh
        out[..., i, 1] = cy + lo * sh + la * ch
    return out


def sample_bilinear(field, pts, x0, y0, res):
    """Bilinear between cell centres, edge replicate in the half-cell border; NaN outside the extent (s0_read.sample_bilinear, verbatim)."""
    nx, ny = field.shape
    fx, fy = (pts[..., 0] - x0) / res - 0.5, (pts[..., 1] - y0) / res - 0.5
    ok = (fx >= -0.5) & (fx < nx - 0.5) & (fy >= -0.5) & (fy < ny - 0.5)
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - ix, fy - iy
    a0, a1 = np.clip(ix, 0, nx - 1), np.clip(ix + 1, 0, nx - 1)
    b0, b1 = np.clip(iy, 0, ny - 1), np.clip(iy + 1, 0, ny - 1)
    v = field[a0, b0] * (1 - tx) * (1 - ty) + field[a1, b0] * tx * (1 - ty) + field[a0, b1] * (1 - tx) * ty + field[a1, b1] * tx * ty
    return np.where(ok, v, np.nan), ok


def sdf_from_mask(mask, cell_m):
    """auxveto_read.sdf_from_mask / s0_read.sdf_from_mask: metres, + inside, half-cell corrected, no pad, no clip; degenerate masks read +-BIG_M."""
    from scipy.ndimage import distance_transform_edt as edt
    m = np.asarray(mask, bool)
    if not m.any():
        return np.full(m.shape, -BIG_M)
    if m.all():
        return np.full(m.shape, BIG_M)
    return np.where(m, edt(m) - 0.5, -(edt(~m) - 0.5)) * cell_m


def footprint_points(poses20):
    """[N, M, 20, 3] -> [N, M, 164, 2] (41 poses x 4 corners)."""
    p8 = to_navsim_vec(poses20)
    c = corners(densify(p8))
    return c.reshape(c.shape[0], c.shape[1], -1, 2)


def dk_from_masks(masks, pts):
    """masks [N,40,32] bool (head or true mask on the 2 m grid); pts [N,M,164,2] -> (d_k [N,M] (+inf if no point inside), n_inside [N,M])."""
    N, M = pts.shape[:2]
    dk = np.empty((N, M))
    n_in = np.empty((N, M), np.int64)
    for n in range(N):
        fld = sdf_from_mask(masks[n], CELL)
        v, ok = sample_bilinear(fld, pts[n], X0, Y0, CELL)
        dk[n] = np.round(np.where(ok, v, np.inf).min(axis=1), 9)        # 1e-9 m: bilinear fp noise on a constant (degenerate) field must not break ties
        n_in[n] = ok.sum(axis=1)
    return dk, n_in


# ======================================================================== pooled AUC through the log-cluster resamples (aux_s1_readout.auc_cluster_stats, verbatim)
def auc_cluster_stats(score, passes, cluster, n_groups):
    score = np.asarray(score, np.float64)
    passes = np.asarray(passes, bool)
    cluster = np.asarray(cluster, np.int64)
    pos = [np.sort(score[(cluster == g) & passes]) for g in range(n_groups)]
    neg = [np.sort(score[(cluster == g) & ~passes]) for g in range(n_groups)]
    S = np.zeros((n_groups, n_groups))
    Nm = np.zeros((n_groups, n_groups))
    for b in range(n_groups):
        if len(neg[b]) == 0:
            continue
        for a in range(n_groups):
            if len(pos[a]) == 0:
                continue
            lt = np.searchsorted(neg[b], pos[a], side="left").sum()
            le = np.searchsorted(neg[b], pos[a], side="right").sum()
            S[a, b] = lt + 0.5 * (le - lt)
            Nm[a, b] = len(pos[a]) * len(neg[b])
    return S, Nm


def auc_value(S, Nm, w=None):
    if w is None:
        w = np.ones(S.shape[0])
    d = float(w @ Nm @ w)
    return float(w @ S @ w) / d if d > 0 else float("nan")


def auc_reps(S, Nm, W):
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((W @ S) * W).sum(1) / ((W @ Nm) * W).sum(1)


def q_block(dk, dac, valid_set, grp, W, set_group):
    """S1-Q: pooled AUC of -d_k for DAC = 0.  dk [N,M], dac [N,M] in {0,1}, valid_set [N] (the set carries P6 labels), set_group [N] group idx."""
    m = np.broadcast_to(np.asarray(valid_set, bool)[:, None], dk.shape)
    cl = np.broadcast_to(np.asarray(set_group, np.int64)[:, None], dk.shape)
    passes = np.asarray(dac)[m] == 1.0
    S, Nm = auc_cluster_stats(dk[m], passes, cl[m], grp.G)
    point = auc_value(S, Nm)
    reps = auc_reps(S, Nm, W)
    ci, n_nan = ci95(reps)
    return {"point": point, "ci95": ci, "n_sets": int(np.asarray(valid_set).sum()), "n_pairs": float(Nm.sum()), "n_proposals_dac0": int((~passes).sum()),
            "n_proposals_dac1": int(passes.sum()), "n_nan_resamples": n_nan}, reps


def diff_block(reps_a, point_a, reps_b, point_b):
    d = np.asarray(reps_a) - np.asarray(reps_b)
    ci, n_nan = ci95(d)
    return {"point": point_a - point_b, "ci95": ci, "lower95": ci[0], "n_nan_resamples": n_nan}


# ======================================================================== the outcome table (card, S1) -- applied mechanically
def verdict(p1, q, f_hold, cshuf, cflip, lv_iou, city_ious, in_sample_iou):
    """Every argument None = its input is missing.  Returns (verdict string, the row's reason, missing list used).
    Order: INVALID (a control failed) -> the table rows.  A row whose condition needs a missing input is INCOMPLETE (never a substitute)."""
    miss = []
    if cshuf is None:
        miss.append("C-SHUF")
    if cflip is None:
        miss.append("C-FLIP")
    if (cshuf is not None and cshuf >= BAR_CSHUF) or (cflip is not None and cflip >= BAR_CFLIP):
        why = []
        if cshuf is not None and cshuf >= BAR_CSHUF:
            why.append(f"C-SHUF {cshuf:.4f} >= {BAR_CSHUF}")
        if cflip is not None and cflip >= BAR_CFLIP:
            why.append(f"C-FLIP {cflip:.4f} >= {BAR_CFLIP}")
        return "INVALID", "; ".join(why), miss
    if miss:
        return "INCOMPLETE", "a control could not be read: " + ", ".join(miss), miss
    p1_ok, q_ok = p1 >= BAR_P1, q >= BAR_Q
    if p1_ok and q_ok:
        if f_hold is None:
            return "INCOMPLETE", "S1-P1 and S1-Q hold; S1-F (needs the initial head's predictions) is unread: S1-GO only if S1-F holds", ["S1-F"]
        return ("S1-GO", "S1-P1, S1-Q, S1-F hold, controls valid", []) if f_hold else ("UNNAMED", "S1-P1 and S1-Q hold but S1-F fails: no S1 row matches", [])
    if q_ok and not p1_ok:
        non_lv = {c: v for c, v in city_ious.items() if c != "LV"}
        if lv_iou >= COMP_LV_MIN and any(v < COMP_NONLV_MAX for v in non_lv.values()):
            return "S1-COMP", f"S1-Q holds, S1-P1 fails, LV {lv_iou:.4f} >= {COMP_LV_MIN} and a non-LV city < {COMP_NONLV_MAX}", []
        return "UNNAMED", "S1-Q holds, S1-P1 fails, but not (LV >= 0.80 and a non-LV city < 0.70): no S1 row matches", []
    if p1_ok and not q_ok:
        return "S1-BOUND", "S1-P1 holds, S1-Q fails", []
    if in_sample_iou is None:
        return "INCOMPLETE", "S1-P1 and S1-Q both fail; S1-FIT vs S1-WEAK needs the in-sample IoU (not in the exports)", ["in_sample_iou"]
    return ("S1-FIT", f"both fail, in-sample IoU {in_sample_iou:.4f} >= {FIT_IN_SAMPLE}", []) if in_sample_iou >= FIT_IN_SAMPLE else ("S1-WEAK", f"both fail, in-sample IoU {in_sample_iou:.4f} < {FIT_IN_SAMPLE}", [])


NEXT = {"S1-GO": "S2 (the selection read on E3); S1b = HA-s1 (A2 section 4: only if HA-s0 reached S1-GO with valid controls; launching it is the MM's step)",
        "S1-COMP": "HA-CB (city-balanced draws), its own pair of seeds",
        "S1-BOUND": "HC (1 m cells, boundary-weighted Huber)",
        "S1-FIT": "HB (a LoRA-adapted copy of P6's trunk for the head only; P6's trunk untouched)",
        "S1-WEAK": "HB and HC as one registered pair under their own card",
        "INVALID": "fix the instrument / re-run the affected arm",
        "UNNAMED": "reported as unnamed (no card row matches); back to the MM",
        "INCOMPLETE": "supply the missing input(s) named in missing_inputs, then re-read; nothing is substituted"}


# ======================================================================== the whole read, on arrays (the tests call this on synthetic fixtures)
def run_read(d, n_boot=N_BOOT, seed=SEED, mutate=None):
    """d: dict of arrays.
      frame_log [N] str, frame_city [N]? (derived from city_of_log), city_of_log {log: city}, label [N,160,128] int8, valid [N] bool (frame has a label),
      poses20 [N,M,20,3], dac [N,M] float (1.0 = no violation), set_valid [N] bool (the set carries P6 labels),
      arms {name: {logit [N,40,32], sdf [N,40,32]}} with 's0' and 'shuf' (+ optional 'init0' = the initial head of HA-s0's seed).
    -> result dict (JSON-serialisable)."""
    t0 = time.time()
    logs = np.asarray(d["frame_log"]).astype(str)
    grp = Groups(logs, d["city_of_log"])
    W = boot_weights(grp.G, n_boot, seed)
    res = {"tool": TOOL, "n_boot": n_boot, "seed": seed, "mutate": mutate, "n_frames": int(len(logs)), "n_groups": grp.G, "groups": grp.logs,
           "city_logs": {c: int((grp.city == c).sum()) for c in CITY_WEIGHT}, "missing_inputs": [], "arms": {}}
    pts = footprint_points(d["poses20"])
    frac_true, sdf_true = block_targets(d["label"])
    true_mask = frac_true >= 0.5
    mut = mutate if mutate in MUTATIONS else None
    dk_cache = {}

    def arm_read(name, with_q=True, with_flip=False):
        a = d["arms"][name]
        st = frame_stats(a["logit"], a["sdf"], d["label"], d["valid"])
        out = {"p1": {}, "p2": None, "q": None}
        for band in ("0-40", "0-20", "20-40", "40-72"):
            blk, reps, _ = p1_block(st, grp, W, band, mutate=mut)
            out["p1"][band] = blk
            if band == "0-40":
                out["_p1_reps"], out["_p1_point"] = reps, blk["point"]
        use = st["valid"]
        gi = grp.idx[use]
        pa, ps, pn = by_group(gi, grp.G, st["p2_abs"][use], st["p2_sq"][use], st["p2_n"][use])
        out["p2"] = {"near_boundary_mae_m": float(pa.sum() / pn.sum()) if pn.sum() else float("nan"), "near_boundary_rms_m": float(math.sqrt(ps.sum() / pn.sum())) if pn.sum() else float("nan"),
                     "n_cells": float(pn.sum()), "half_band_m": NEAR_BOUNDARY_M}
        if with_q:
            m = np.asarray(a["logit"]) > 0
            dk, n_in = dk_from_masks(m, pts)
            dk_cache[name] = dk
            q, reps = q_block(dk, d["dac"], d["set_valid"], grp, W, grp.idx)
            q["outside_extent_point_share"] = float(1 - n_in.sum() / (n_in.size * pts.shape[2]))
            q["proposals_with_no_point_inside"] = int((n_in == 0).sum())
            out["q"], out["_q_reps"] = q, reps
        if with_flip:
            mf = np.asarray(a["logit"])[:, :, ::-1] > 0
            dkf, _ = dk_from_masks(mf, pts)
            qf, _ = q_block(dkf, d["dac"], d["set_valid"], grp, W, grp.idx)
            out["q_yflip"] = qf
        return out

    s0 = arm_read("s0", with_q=True, with_flip=True)
    shuf = arm_read("shuf", with_q=True)
    res["arms"]["HA-s0"] = {k: v for k, v in s0.items() if not k.startswith("_")}
    res["arms"]["HA-SHUF"] = {k: v for k, v in shuf.items() if not k.startswith("_")}

    # reported, not deciding: true-field AUC on the same proposals, sigma_b
    dk_true, _ = dk_from_masks(true_mask, pts)
    qt, _ = q_block(dk_true, d["dac"], d["set_valid"], grp, W, grp.idx)
    res["true_field_auc_2m"] = qt
    pm = np.asarray(d["arms"]["s0"]["logit"]) > 0
    err = []
    for n in np.flatnonzero(d["valid"]):
        fp, ft = sdf_from_mask(pm[n], CELL), sdf_from_mask(true_mask[n], CELL)
        near = np.abs(ft) < NEAR_BOUNDARY_M
        err.append((fp - ft)[near])
    err = np.concatenate(err) if err else np.zeros(0)
    res["sigma_b_m"] = {"rms_mask_edt_field_vs_true_mask_edt_field": float(math.sqrt((err ** 2).mean())) if err.size else float("nan"), "n_cells": int(err.size),
                        "note": "mask-derived EDT field of the head vs the EDT field of the true 2 m mask, cells with |SDF_gt| < 3 m (card 5.1 step 5, same pipeline)"}

    # S1-F : trained minus the same seed's INITIAL head, same resamples
    s1f = None
    if "init0" in d["arms"]:
        ini = arm_read("init0", with_q=True)
        res["arms"]["HA-s0-initial"] = {k: v for k, v in ini.items() if not k.startswith("_")}
        s1f = {"p1": diff_block(s0["_p1_reps"], s0["_p1_point"], ini["_p1_reps"], ini["_p1_point"]),
               "q": diff_block(s0["_q_reps"], s0["q"]["point"], ini["_q_reps"], ini["q"]["point"])}
        s1f["hold"] = bool(s1f["p1"]["lower95"] > 0 and s1f["q"]["lower95"] > 0)
    else:
        res["missing_inputs"].append("MISSING-INPUT initial_head_predictions (the same seed's never-trained head on the H48 frames; needed for S1-F)")
    res["S1_F"] = s1f
    res["missing_inputs"] += ["MISSING-INPUT in_sample_iou (head predictions on training frames were not exported; needed only to split S1-FIT from S1-WEAK)",
                              "MISSING-INPUT p6_dac_logit (P6's own DAC logit is not in the export; descriptive AUC not computed)",
                              "MISSING-INPUT navtrain_metric_cache_t0_ego_states (S1-QT, tracked d_k, unavailable; reported, not deciding)"]

    # sensitivity: all-48 groups (logs without frames carry zero weight) -- CI only
    if d.get("all_logs"):
        g48 = Groups(logs, d["city_of_log"], group_logs=d["all_logs"])
        W48 = boot_weights(g48.G, n_boot, seed)
        b48, _, _ = p1_block(frame_stats(d["arms"]["s0"]["logit"], d["arms"]["s0"]["sdf"], d["label"], d["valid"]), g48, W48, "0-40")
        q48, _ = q_block(dk_cache["s0"], d["dac"], d["set_valid"], g48, W48, g48.idx)
        res["sensitivity_all48_groups"] = {"n_groups": g48.G, "s1_p1_ci95": b48["ci95"], "s1_q_ci95": q48["ci95"]}

    # verdict
    p1 = s0["p1"]["0-40"]["point"]
    q = s0["q"]["point"]
    city = s0["p1"]["0-40"]["per_city"]
    cshuf = shuf["p1"]["0-40"]["point"]
    cflip = res["arms"]["HA-s0"]["q_yflip"]["point"]
    v, why, miss_v = verdict(p1, q, None if s1f is None else s1f["hold"], cshuf, cflip, city.get("LV", float("nan")), city, None)
    res["controls"] = {"C-SHUF": {"value": cshuf, "bar": f"< {BAR_CSHUF}", "ok": bool(cshuf < BAR_CSHUF)},
                       "C-FLIP": {"value": cflip, "bar": f"< {BAR_CFLIP}", "ok": bool(cflip < BAR_CFLIP)}}
    res["bars"] = {"S1-P1": {"value": p1, "ci95": s0["p1"]["0-40"]["ci95"], "bar": f">= {BAR_P1}", "holds": bool(p1 >= BAR_P1)},
                   "S1-Q": {"value": q, "ci95": s0["q"]["ci95"], "bar": f">= {BAR_Q}", "holds": bool(q >= BAR_Q)},
                   "S1-F": None if s1f is None else {"p1_lower95": s1f["p1"]["lower95"], "q_lower95": s1f["q"]["lower95"], "bar": "both lower bounds > 0", "holds": s1f["hold"]},
                   "S1-R": "not read (S1b = HA-s1 is trained only if HA-s0 reaches S1-GO)"}
    res["verdict"], res["verdict_reason"], res["verdict_missing"] = v, why, miss_v
    res["next"] = NEXT[v]
    res["wall_s"] = time.time() - t0
    return res


# ======================================================================== real-data loading
CARD_FILES = {"PREREG_VDAC_P6.md": "5594c2ae777a53ed4a8a873ca374c12943eadc74385cb07671c2ac90a3292ae4",
              "PREREG_VDAC_P6_A1.md": "8f121bdf9d6f19acec8927f12c50d3f7be2832c70c7f1fe40140d5eb8cc4ed70",
              "PREREG_VDAC_P6_A2.md": "c3d563586be6479ea11455f92d7a5295c9e1658e2d7ba1cc77b27e87e5c8b417",
              "PREREG_VDAC_P6_A3.md": "a7708573d9db16609bca0faf255ea4c879bda8723ce82e9df72cd27df763d570",
              "PREREG_VDAC_P6_A4.md": "045a10c8f7129c0ac2b168624a982915d50724e4d9b8960a91715fe083254cf5"}
H48_SHA = "2ae13935a3ead7f30b015f33b0bebc7a35e70eb2a4194161f8cbb66ac89b8da8"
STORE_SHA_PREFIX = "4a17db67"


def parse_a3_cities(a3_text):
    """the A3 card's list: 'log  CITY  n  origin' lines -> {log: city}"""
    out = {}
    for ln in a3_text.splitlines():
        m = re.match(r"^(\d{4}\.\d\d\.\d\d\.\d\d\.\d\d\.\d\d_veh-\d+_\d+_\d+)\s+(LV|BOS|PIT|SG)\s+(\d+)\s+(SFT24|extra-8)\s*$", ln)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def load_real(man):
    """man: the inputs manifest {dir_cards, store, files: {key: {path, sha256}}} -> the dict run_read wants (+ audit counts)."""
    audit = {}
    cdir = man["cards_dir"]
    for f, sha in CARD_FILES.items():
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
    if hashlib.sha256(h48_bytes).hexdigest() != H48_SHA:
        raise SystemExit("REFUSED: H48 list sha256")
    h48 = h48_bytes.decode().split()
    city_a3 = parse_a3_cities(open(os.path.join(cdir, "PREREG_VDAC_P6_A3.md"), encoding="utf-8").read())
    if sorted(city_a3) != sorted(h48):
        raise SystemExit("REFUSED: A3's table does not list exactly the H48 logs")
    sm = json.load(open(os.path.join(man["store"], "manifest.json")))
    if not str(sm["store_sha256"]).startswith(STORE_SHA_PREFIX):
        raise SystemExit("REFUSED: label store sha256")
    city_store = {l: MAP_CITY[sm["files"][f"shards/{l}.aux1"]["map"]] for l in h48}
    if city_store != city_a3:
        raise SystemExit("REFUSED: store manifest city != A3 city for some H48 log")
    audit["city_per_log"] = "A3 table == store manifest map, 48 logs"
    ex = np.load(man["files"]["export"]["path"], allow_pickle=False)
    exj = json.load(open(man["files"]["export_json"]["path"]))
    if exj["h48_sha256"] != H48_SHA or str(ex["h48_sha256"]) != H48_SHA:
        raise SystemExit("REFUSED: export H48 sha")
    if [str(x) for x in ex["comp_order"]] != ["nc", "dac", "ep", "ttc", "c", "ddc"]:
        raise SystemExit("REFUSED: comp_order")
    el, et = [str(x) for x in ex["log"]], [str(x) for x in ex["token"]]
    keys = list(zip(el, et))
    if len(set(keys)) != len(keys):
        raise SystemExit("REFUSED: duplicate (log, token) in the export")
    kpos = {k: i for i, k in enumerate(keys)}
    arms = {}
    for arm, files in (("s0", ("head_s0_a", "head_s0_b")), ("shuf", ("head_shuf_a", "head_shuf_b")), ("init0", ("head_init0_a", "head_init0_b"))):
        if files[0] not in man["files"]:
            continue
        lg = np.zeros((len(keys), NCX, NCY), np.float32)
        sd = np.zeros((len(keys), NCX, NCY), np.float32)
        got = np.zeros(len(keys), bool)
        for fk in files:
            z = np.load(man["files"][fk]["path"], allow_pickle=False)
            if not (np.asarray(z["step"]) == 0).all():
                raise SystemExit(f"REFUSED: {fk} holds rows with step != 0")
            if not (np.isfinite(z["logit"]).all() and np.isfinite(z["sdf_m"]).all()):
                raise SystemExit(f"REFUSED: {fk} holds non-finite outputs")
            for i, k in enumerate(zip([str(x) for x in z["log"]], [str(x) for x in z["token"]])):
                j = kpos.get(k)
                if j is None:
                    raise SystemExit(f"REFUSED: {fk} key {k} is not in the export")
                if got[j]:
                    raise SystemExit(f"REFUSED: {fk} key {k} appears twice")
                got[j] = True
                lg[j], sd[j] = z["logit"][i], z["sdf_m"][i]
        if not got.all():
            raise SystemExit(f"REFUSED: arm {arm} does not cover the export's {len(keys)} keys ({int(got.sum())})")
        arms[arm] = {"logit": lg, "sdf": sd}
    shard_logs = sorted(set(el))
    lab = np.zeros((len(keys), NXF, NYF), np.int8)
    valid = np.zeros(len(keys), bool)
    n_shard_tokens = 0
    for l in shard_logs:
        rel = f"shards/{l}.aux1"
        p = os.path.join(man["store"], rel)
        if sha256_file(p) != sm["files"][rel]["sha256"]:
            raise SystemExit(f"REFUSED: shard {rel} does not hash to the store manifest")
        _, labs = read_shard(p)
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
         "all_logs": sorted(h48)}
    audit.update({"frames_forwarded": int(len(keys)), "logs_with_frames": len(shard_logs), "h48_logs_without_frames": sorted(set(h48) - set(shard_logs)),
                  "frames_with_label": int(valid.sum()), "frames_without_label": int((~valid).sum()), "export_banks": {str(b): int((ex["bank"] == b).sum()) for b in np.unique(ex["bank"])},
                  "sets_with_p6_pdm_label": int(len(keys)), "arms_present": sorted(arms),
                  "city_logs_with_frames": {c: int(sum(1 for l in shard_logs if city_a3[l] == c)) for c in CITY_WEIGHT}})
    return d, audit


def fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def tables_md(r):
    s0, sh = r["arms"]["HA-s0"], r["arms"]["HA-SHUF"]
    L = [f"verdict (mechanical): **{r['verdict']}** - {r['verdict_reason']}", "", "| bar | value | 95 % CI | bar | holds |", "|---|---|---|---|---|"]
    for k in ("S1-P1", "S1-Q"):
        b = r["bars"][k]
        L.append(f"| {k} | {fmt(b['value'])} | [{fmt(b['ci95'][0])}, {fmt(b['ci95'][1])}] | {b['bar']} | {b['holds']} |")
    f = r["bars"]["S1-F"]
    L.append("| S1-F | MISSING-INPUT | - | both lower bounds > 0 | unread |" if f is None else f"| S1-F | P1 lower {fmt(f['p1_lower95'])}, Q lower {fmt(f['q_lower95'])} | - | both > 0 | {f['holds']} |")
    for k, v in r["controls"].items():
        L.append(f"| {k} | {fmt(v['value'])} | - | {v['bar']} | {v['ok']} |")
    L += ["", "per city (S1-P1 [0,40) IoU, HA-s0): " + ", ".join(f"{c} {fmt(v)}" for c, v in s0["p1"]["0-40"]["per_city"].items()), "",
          "| band IoU, HA-s0 (city-reweighted) | value | 95 % CI |", "|---|---|---|"]
    for band in ("0-20", "20-40", "40-72", "0-40"):
        b = s0["p1"][band]
        L.append(f"| {band} m | {fmt(b['point'])} | [{fmt(b['ci95'][0])}, {fmt(b['ci95'][1])}] |")
    L += ["", f"HA-SHUF IoU [0,40): {fmt(sh['p1']['0-40']['point'])} CI [{fmt(sh['p1']['0-40']['ci95'][0])}, {fmt(sh['p1']['0-40']['ci95'][1])}]; HA-SHUF S1-Q {fmt(sh['q']['point'])}",
          f"HA-s0 near-boundary SDF channel MAE {fmt(s0['p2']['near_boundary_mae_m'])} m, RMS {fmt(s0['p2']['near_boundary_rms_m'])} m (cells {s0['p2']['n_cells']:.0f})",
          f"sigma_b {fmt(r['sigma_b_m']['rms_mask_edt_field_vs_true_mask_edt_field'])} m; true-field 2 m AUC {fmt(r['true_field_auc_2m']['point'])}; y-flipped S1-Q {fmt(s0['q_yflip']['point'])}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="inputs manifest json (cards_dir, store, files{key:{path,sha256}})")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    d, audit = load_real(man)
    print("COUNTS BEFORE THE READ", json.dumps(audit), flush=True)
    r = run_read(d, n_boot=a.n_boot)
    r["audit"] = audit
    r["reader_sha256"] = sha256_file(os.path.abspath(__file__))
    r["manifest_sha256"] = sha256_file(a.manifest)
    os.makedirs(a.out, exist_ok=True)
    json.dump(r, open(os.path.join(a.out, "s1a_read.json"), "w"), indent=1, default=float)
    open(os.path.join(a.out, "s1a_read_tables.md"), "w", encoding="utf-8", newline="\n").write(tables_md(r))
    print(tables_md(r))
    print("MISSING:", *r["missing_inputs"], sep="\n  ")
    print("VERDICT", r["verdict"], "|", r["verdict_reason"], "| next:", r["next"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
