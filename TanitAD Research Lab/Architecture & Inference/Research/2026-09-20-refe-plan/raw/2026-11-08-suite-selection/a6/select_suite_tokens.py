#!/usr/bin/env python3
"""select_suite_tokens.py -- BEHAVIOUR_SCORECARD section 3.1, implemented as written (CPU only, offline).

Scorecard: E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/BEHAVIOUR_SCORECARD.md  (sha256 fac6c7f1...; FROZEN, not edited here)
Population: the 12,146 NAVSIM navtest tokens (navtest.yaml) read from their nuPlan DB logs.  Window = [t0 - 2 s, t0 + 15 s] on a 10 Hz grid
(every 2nd lidar_pc of the 20 Hz DB timeline).  Every predicate is computed from the DB's ego track and tracked objects (lidar_box) in the
HUMAN-PATH FRAME (arc length s and signed lateral offset to the polyline of the human ego rear-axle track over the window).

Every place where the frozen text leaves a choice is a switch in SPEC and is written up in AMBIGUITIES.md.  Nothing here was tuned on the output.

    python select_suite_tokens.py --out <dir> [--workers 8] [--limit-logs N] [--select-only]
Outputs (in --out): per_token_results.pkl, suite_features.csv.gz, SUITE_CANDIDATES.json, SUITE_PIN.json, run_report.json
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import pickle
import re
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------------------------------------------------------------------
# paths
SCORECARD = "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/BEHAVIOUR_SCORECARD.md"
SCORECARD_SHA256 = "fac6c7f150eb026de0708c6f6557df52481216f76b35f3c8c8b8045bcb8ec0c5"
CRUISE_JSON = "E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106/tokens_cruise.json"
CRUISE_JSON_SHA256 = "57db811e644218c00020cd6d25b417b604ffc9af7b81612976d0d52f02263f78"
DB_DIR = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
YAML = ("F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896/navsim/planning/script/config/common/train_test_split/"
        "scene_filter/navtest.yaml")
YAML_SHA256 = "61284edf5003c0291f843ce9817c822ba306609a62d54544223adae3fc7fc9cd"
OUT_DEFAULT = "E:/Projects/TanitAD-artifacts/tsf_bundle2_20261107/a6"      # A6 build (AMENDMENT_6): never overwrites a3/
SCORER_DIR = "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/code"   # HumanPath, for stop_line_s (AMENDMENT_3 section 5)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------------------------------------------------------------------
# SPEC: every constant.  `src` says where it comes from: SC1.. = scorecard section; AMB-n = an ambiguity reading (see AMBIGUITIES.md).
SPEC = dict(
    salt="|refe-behaviour-v1",                      # SC 3.1 item 2
    n_candidates=5,                                 # SC 3.1 item 2
    k_lo=-20, k_hi=150, step=2, dt_nominal=0.1,      # SC 3.1: window [t0-2 s, t0+15 s]; SC 1: 10 Hz grid (every 2nd 20 Hz lidar_pc)
    ego_front=4.049, ego_rear=1.127, ego_width=2.297,  # nuplan get_pacifica_parameters() (MEASURED by import 2026-11-07)
    corridor_half_width=1.5,                        # SC 1 (lead definition)
    free_gap_s=4.0, free_gap_min_m=15.0, free_ttc_s=8.0,   # SC 1 (free road)
    ttc_vrel_min=0.1,                               # SC 1 (TTC_t defined for v_rel > 0.1 m/s)
    human_min_ttc_s=1.0,                            # SC 3.1 item 1 (human safe)
    # cruise
    cruise_speed_band=0.15, cruise_max_curvature=0.02,
    # steady lead
    steady_min_run_s=8.0, steady_max_gap_m=60.0, steady_min_speed=3.0, steady_max_abs_acc=1.0,
    # lead braking
    brake_max_gap_m=50.0, brake_min_drop=4.0, brake_max_dur_s=6.0, brake_min_peak_decel=1.5, brake_min_ego_v0=5.0,
    # stop-go
    stopgo_stop_v=0.5, stopgo_go_v=2.0,
    # merge / cut-in
    merge_out_lat=2.0, merge_in_lat=0.5, merge_max_gap_m=30.0, merge_alt_max_heading_deg=30.0,   # the last one is the ALT reading only (AMB-10)
    # estimators that the frozen text does not fix (AMB-*)
    vel_half_baseline_ticks=5,                      # AMB-5: track velocity = central difference of positions over +-0.5 s
    acc_half_baseline_ticks=5,                      # AMB-5: track acceleration = central difference of that speed over +-0.5 s
    curv_half_baseline_ticks=5,                     # AMB-6: curvature = d(heading)/d(arc length) over +-0.5 s
    path_back_ext_m=60.0, path_fwd_ext_m=200.0, path_ext_step_m=4.0,   # AMB-2: straight extension of the human path
    cand_center_max_dist_m=20.0,                    # prefilter only (box half diagonal + corridor + vertex spacing << 20 m)
    overlap_prefilter_m=15.0,
    free_road_reading="sensible",                   # AMB-1: free = no lead within G AND TTC > 8 s (the literal reading is also computed)
)
STEP_N = SPEC["k_hi"] - SPEC["k_lo"] + 1            # 171 ticks
T0_I = -SPEC["k_lo"]                                # tick index of t0 (= 20)
EGO_COG_OFF = (SPEC["ego_front"] - SPEC["ego_rear"]) / 2.0     # centre of the ego box ahead of the rear axle
EGO_LEN = SPEC["ego_front"] + SPEC["ego_rear"]


def sha256_file(p, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_yaml(path=YAML):
    logs, tokens, sec = [], [], None
    for line in open(path, "r", encoding="utf-8"):
        s = line.rstrip("\n")
        if s.startswith("log_names:"):
            sec = "log"; continue
        if s.startswith("tokens:"):
            sec = "tok"; continue
        m = re.match(r"^\s+-\s+'([^']+)'\s*$", s)
        if m and sec == "log":
            logs.append(m.group(1))
        elif m and sec == "tok":
            tokens.append(m.group(1))
    return logs, tokens


def order_key(token):
    return hashlib.sha256((token + SPEC["salt"]).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------------------------------------------------
# per-log loading
class LogCtx:
    pass


def load_log(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    rows = cur.execute(
        "select lp.token, lp.timestamp, lp.scene_token, e.x, e.y, e.qw, e.qx, e.qy, e.qz, e.vx, e.vy "
        "from lidar_pc lp join ego_pose e on e.token = lp.ego_pose_token order by lp.timestamp").fetchall()
    c = LogCtx()
    c.n = len(rows)
    c.tok = [bytes(r[0]).hex() for r in rows]
    c.tok2rank = {t: i for i, t in enumerate(c.tok)}
    rank_of_blob = {bytes(r[0]): i for i, r in enumerate(rows)}
    c.T = np.array([r[1] for r in rows], dtype=np.float64) / 1e6
    c.scene = [bytes(r[2]).hex() if r[2] is not None else "" for r in rows]
    a = np.array([[r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]] for r in rows], dtype=np.float64)
    c.ex, c.ey = a[:, 0], a[:, 1]
    qw, qx, qy, qz = a[:, 2], a[:, 3], a[:, 4], a[:, 5]
    c.eyaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    c.ev = np.hypot(a[:, 6], a[:, 7])                    # ego_pose vx, vy are in the ego frame (checked against the pkl ego_dynamic_state)
    cat = {bytes(t): n for t, n in cur.execute("select track.token, category.name from track join category on track.category_token = category.token")}
    brow = cur.execute("select lidar_pc_token, track_token, x, y, yaw, width, length from lidar_box").fetchall()
    con.close()
    nb = len(brow)
    rk = np.fromiter((rank_of_blob[bytes(r[0])] for r in brow), dtype=np.int64, count=nb)
    trk_ids = {}
    trk = np.empty(nb, dtype=np.int64)
    isveh = np.empty(nb, dtype=bool)
    for i, r in enumerate(brow):
        tb = bytes(r[1])
        t = trk_ids.setdefault(tb, len(trk_ids))
        trk[i] = t
        isveh[i] = cat.get(tb) == "vehicle"
    num = np.array([[r[2], r[3], r[4], r[5], r[6]] for r in brow], dtype=np.float64).reshape(-1, 5)
    order = np.argsort(rk, kind="stable")
    c.bx, c.by, c.byaw, c.bw, c.bl = (num[order, j] for j in range(5))
    c.btrk, c.bveh = trk[order], isveh[order]
    c.bstart = np.searchsorted(rk[order], np.arange(c.n + 1))
    c.cat_of_row = None
    c.cat_names = [cat.get(b) for b in trk_ids]          # index = compact track id
    c.trk_hex = [bytes(b).hex() for b in trk_ids]
    return c


# ----------------------------------------------------------------------------------------------------------------------------------------
# geometry helpers
class Path:
    """Polyline of the human rear-axle track, straight-extended at both ends (AMB-2)."""

    def __init__(self, P, yaw):
        seg = np.hypot(*np.diff(P, axis=0).T)
        s_ego = np.concatenate([[0.0], np.cumsum(seg)])
        st = SPEC["path_ext_step_m"]
        nb, nf = int(SPEC["path_back_ext_m"] / st), int(SPEC["path_fwd_ext_m"] / st)
        d0 = np.array([math.cos(yaw[0]), math.sin(yaw[0])])
        d1 = np.array([math.cos(yaw[-1]), math.sin(yaw[-1])])
        back = P[0][None, :] - d0[None, :] * (st * np.arange(nb, 0, -1))[:, None]
        fwd = P[-1][None, :] + d1[None, :] * (st * np.arange(1, nf + 1))[:, None]
        self.V = np.vstack([back, P, fwd])
        self.S = np.concatenate([-st * np.arange(nb, 0, -1), s_ego, s_ego[-1] + st * np.arange(1, nf + 1)])
        self.s_ego = s_ego
        self.i0 = nb                                   # vertex index of the first human pose
        ab = np.diff(self.V, axis=0)
        L = np.hypot(ab[:, 0], ab[:, 1])
        self.L = L
        good = L > 1e-6
        dirs = np.zeros_like(ab)
        dirs[good] = ab[good] / L[good][:, None]
        # fill degenerate (stationary) segments with the nearest valid direction
        gi = np.where(good)[0]
        if len(gi) == 0:
            dirs[:] = [1.0, 0.0]
        else:
            nearest = gi[np.clip(np.searchsorted(gi, np.arange(len(L))), 0, len(gi) - 1)]
            prev = gi[np.clip(np.searchsorted(gi, np.arange(len(L)), side="right") - 1, 0, len(gi) - 1)]
            pick = np.where(np.abs(nearest - np.arange(len(L))) <= np.abs(prev - np.arange(len(L))), nearest, prev)
            dirs[~good] = dirs[pick[~good]]
        self.dir = dirs
        self.tree = cKDTree(self.V)

    def project(self, Q):
        """Q [N,2] -> s, lat (left +), tangent [N,2], dist"""
        d0, i = self.tree.query(Q)
        nseg = len(self.L)
        best = None
        for off in (-1, 0):
            j = np.clip(i + off, 0, nseg - 1)
            a = self.V[j]
            ab = self.V[j + 1] - a
            L2 = np.maximum((ab ** 2).sum(1), 1e-12)
            t = np.clip(((Q - a) * ab).sum(1) / L2, 0.0, 1.0)
            pr = a + t[:, None] * ab
            dist = np.hypot(*(Q - pr).T)
            cross = ab[:, 0] * (Q[:, 1] - a[:, 1]) - ab[:, 1] * (Q[:, 0] - a[:, 0])
            lat = np.where(cross >= 0, dist, -dist)
            s = self.S[j] + t * self.L[j]
            cur = (dist, s, lat, self.dir[j])
            if best is None:
                best = cur
            else:
                m = cur[0] < best[0]
                best = tuple(np.where(m[:, None] if x.ndim == 2 else m, x, y) for x, y in zip(cur, best))
        dist, s, lat, tg = best
        return s, lat, tg, dist


def rect_overlap_depth(c1, yaw1, hl1, hw1, c2, yaw2, hl2, hw2):
    """Minimum penetration depth over the 4 SAT axes (>0 => the two oriented rectangles overlap).  All arrays are pair-aligned [N,...]."""
    u1 = np.stack([np.cos(yaw1), np.sin(yaw1)], 1)
    v1 = np.stack([-np.sin(yaw1), np.cos(yaw1)], 1)
    u2 = np.stack([np.cos(yaw2), np.sin(yaw2)], 1)
    v2 = np.stack([-np.sin(yaw2), np.cos(yaw2)], 1)
    d = c2 - c1
    depth = np.full(len(d), np.inf)
    for ax in (u1, v1, u2, v2):
        r1 = hl1 * np.abs((u1 * ax).sum(1)) + hw1 * np.abs((v1 * ax).sum(1))
        r2 = hl2 * np.abs((u2 * ax).sum(1)) + hw2 * np.abs((v2 * ax).sum(1))
        depth = np.minimum(depth, r1 + r2 - np.abs((d * ax).sum(1)))
    return depth


def fnum(x, nd=4):
    if x is None:
        return None
    x = float(x)
    if math.isnan(x):
        return None
    if math.isinf(x):
        return 1e9 if x > 0 else -1e9
    return round(x, nd)


# ----------------------------------------------------------------------------------------------------------------------------------------
# one token
def analyze(ctx, r0, is_cruise_token, dbg=None):
    S = SPEC
    K = np.arange(S["k_lo"], S["k_hi"] + 1)
    idx = r0 + S["step"] * K
    if idx[0] < 0 or idx[-1] >= ctx.n:
        return dict(ok=False, reason="window_outside_log")
    nT = len(K)
    T = ctx.T[idx]
    dtk = np.diff(T)
    ex, ey, eyaw, ev = ctx.ex[idx], ctx.ey[idx], ctx.eyaw[idx], ctx.ev[idx]
    info = dict(ok=True, reason="", t_span_s=fnum(T[-1] - T[0]), tick_dt_max_dev=fnum(np.max(np.abs(dtk - S["dt_nominal"])), 4),
                n_scenes=len(set(ctx.scene[i] for i in idx[::10])), ego_v0=fnum(ev[T0_I]))
    P = np.stack([ex, ey], 1)
    path = Path(P, eyaw)
    s_ego = path.s_ego
    s_front = s_ego + S["ego_front"]
    info["path_len_m"] = fnum(s_ego[-1], 2)

    # ---- all tracked objects of the window -------------------------------------------------------------------------------------------
    lo, hi = ctx.bstart[idx], ctx.bstart[idx + 1]
    cnt = hi - lo
    tot = int(cnt.sum())
    tick_of = np.repeat(np.arange(nT), cnt)
    sel = np.repeat(lo - (np.cumsum(cnt) - cnt), cnt) + np.arange(tot)
    ox, oy, oyaw, ow, ol = ctx.bx[sel], ctx.by[sel], ctx.byaw[sel], ctx.bw[sel], ctx.bl[sel]
    otrk, oveh = ctx.btrk[sel], ctx.bveh[sel]

    # ---- human safety: overlap of the ego box with any tracked object ------------------------------------------------------------------
    egoc = np.stack([ex + np.cos(eyaw) * EGO_COG_OFF, ey + np.sin(eyaw) * EGO_COG_OFF], 1)
    oc = np.stack([ox, oy], 1)
    near = np.hypot(*(oc - egoc[tick_of]).T) < S["overlap_prefilter_m"]
    n_ov_ticks, ov_cat = 0, ""
    if near.any():
        ii = np.where(near)[0]
        tk = tick_of[ii]
        dep = rect_overlap_depth(egoc[tk], eyaw[tk], np.full(len(ii), EGO_LEN / 2), np.full(len(ii), S["ego_width"] / 2),
                                 oc[ii], oyaw[ii], ol[ii] / 2, ow[ii] / 2)
        hit = dep > 0.0
        if hit.any():
            n_ov_ticks = int(len(np.unique(tk[hit])))
            ov_cat = ctx.cat_names[otrk[ii[hit][0]]] or ""
    info["overlap_ticks"] = n_ov_ticks
    info["overlap_first_category"] = ov_cat

    # ---- vehicle candidates near the path ---------------------------------------------------------------------------------------------
    vsel = np.where(oveh)[0]
    cd, _ = path.tree.query(oc[vsel])
    ci = vsel[cd <= S["cand_center_max_dist_m"]]               # indices into the `sel` arrays
    nc = len(ci)
    ctick, ctrk = tick_of[ci], otrk[ci]
    if nc:
        hl, hw = ol[ci] / 2, ow[ci] / 2
        c_, s_ = np.cos(oyaw[ci]), np.sin(oyaw[ci])
        al = np.stack([c_, s_], 1)
        ac = np.stack([-s_, c_], 1)
        pts = [oc[ci]]
        for sl_, sw_ in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            pts.append(oc[ci] + (sl_ * hl)[:, None] * al + (sw_ * hw)[:, None] * ac)
        Q = np.concatenate(pts, 0)
        s_all, lat_all, tg_all, _ = path.project(Q)
        s_all, lat_all = s_all.reshape(5, nc), lat_all.reshape(5, nc)
        s_c, lat_c, tg_c = s_all[0], lat_all[0], tg_all[:nc]
        s_min, s_max = s_all[1:].min(0), s_all[1:].max(0)
        lat_min, lat_max = lat_all[1:].min(0), lat_all[1:].max(0)
        in_corr = (lat_min <= S["corridor_half_width"]) & (lat_max >= -S["corridor_half_width"])
        sf = s_front[ctick]
        ahead = s_max > sf
        gap_c = np.maximum(0.0, s_min - sf)

    # ---- track kinematics (positions -> velocity -> speed -> acceleration), only for tracks that are ever near the path ----------------------
    if nc:
        trk_list = np.unique(ctrk)
        loc = {int(t): i for i, t in enumerate(trk_list)}
        m_all = np.isin(otrk, trk_list) & oveh
        pos = np.full((len(trk_list), nT, 2), np.nan)
        li = np.array([loc[int(t)] for t in otrk[m_all]], dtype=np.int64)
        pos[li, tick_of[m_all]] = oc[m_all]
        h = S["vel_half_baseline_ticks"]
        vvec = np.full_like(pos, np.nan)
        for k in range(nT):
            a_, b_ = max(0, k - h), min(nT - 1, k + h)
            if a_ == b_:
                continue
            vvec[:, k] = (pos[:, b_] - pos[:, a_]) / (T[b_] - T[a_])
        spd = np.hypot(vvec[..., 0], vvec[..., 1])                       # [trk, nT]
        ha = S["acc_half_baseline_ticks"]
        acc = np.full_like(spd, np.nan)
        for k in range(ha, nT - ha):
            acc[:, k] = (spd[:, k + ha] - spd[:, k - ha]) / (T[k + ha] - T[k - ha])

    # ---- the lead at every tick (SC 1) -------------------------------------------------------------------------------------------------
    lead_trk = np.full(nT, -1, dtype=np.int64)
    lead_gap = np.full(nT, np.nan)
    lead_spd = np.full(nT, np.nan)
    lead_u = np.full(nT, np.nan)
    lead_acc = np.full(nT, np.nan)
    lead_lat = np.full(nT, np.nan)
    if nc:
        m = in_corr & ahead
        if m.any():
            mi = np.where(m)[0]
            order = np.lexsort((s_c[mi], gap_c[mi], ctick[mi]))
            mi = mi[order]
            _, first = np.unique(ctick[mi], return_index=True)
            for j in mi[first]:
                k = ctick[j]
                lt = loc[int(ctrk[j])]
                lead_trk[k] = ctrk[j]
                lead_gap[k] = gap_c[j]
                lead_spd[k] = spd[lt, k]
                v = vvec[lt, k]
                lead_u[k] = v[0] * tg_c[j, 0] + v[1] * tg_c[j, 1]
                lead_acc[k] = acc[lt, k]
                lead_lat[k] = lat_c[j]
    has = lead_trk >= 0
    vrel = ev - lead_u
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(has & (vrel > S["ttc_vrel_min"]), lead_gap / vrel, np.inf)
    ttc = np.where(has & np.isnan(lead_u), np.inf, ttc)
    G = np.maximum(S["free_gap_s"] * ev, S["free_gap_min_m"])
    near_lead = has & (lead_gap <= G)
    closing = has & (ttc <= S["free_ttc_s"])
    free_sensible = ~(near_lead | closing)                              # AMB-1 reading used by the selection
    free_literal = ~(has & near_lead & (ttc > S["free_ttc_s"]))         # AMB-1 literal text (kept for the sensitivity count only)
    min_ttc = float(np.min(ttc)) if has.any() else float("inf")
    info["min_ttc_s"] = fnum(min_ttc, 3)
    info["safe_ttc"] = bool(min_ttc >= S["human_min_ttc_s"])
    info["safe_overlap"] = bool(n_ov_ticks == 0)
    info["safe"] = bool(info["safe_ttc"] and info["safe_overlap"])
    info["lead_t0"] = bool(has[T0_I])
    info["lead_gap_t0"] = fnum(lead_gap[T0_I], 2)
    info["lead_ticks_frac"] = fnum(has.mean(), 3)

    # ---- CRUISE ------------------------------------------------------------------------------------------------------------------------
    med = float(np.median(ev))
    dev = float(np.max(np.abs(ev / max(med, 1e-6) - 1.0)))
    hk = S["curv_half_baseline_ticks"]
    psi = np.unwrap(eyaw)
    kap = np.full(nT, np.nan)
    for k in range(nT):
        a_, b_ = max(0, k - hk), min(nT - 1, k + hk)
        ds = s_ego[b_] - s_ego[a_]
        if ds > 1.0:
            kap[k] = (psi[b_] - psi[a_]) / ds
    kabs = float(np.nanmax(np.abs(kap))) if np.isfinite(kap).any() else float("nan")
    cr = dict(token_in_cruise_list=bool(is_cruise_token), v_median=fnum(med, 3), v_max_rel_dev=fnum(dev, 4), speed_band_ok=bool(dev <= S["cruise_speed_band"]),
              free_road_all=bool(free_sensible.all()), n_nonfree_ticks=int((~free_sensible).sum()),
              free_road_all_literal=bool(free_literal.all()), n_nonfree_ticks_literal=int((~free_literal).sum()),
              kappa_abs_max=fnum(kabs, 5), n_kappa_nan=int(np.isnan(kap).sum()), curvature_ok=bool(not (kabs > S["cruise_max_curvature"])))
    cr["pass"] = bool(is_cruise_token and cr["speed_band_ok"] and cr["free_road_all"] and cr["curvature_ok"] and info["safe"])
    cr["pass_literal_free"] = bool(is_cruise_token and cr["speed_band_ok"] and cr["free_road_all_literal"] and cr["curvature_ok"] and info["safe"])
    cr["pass_before_safety"] = bool(is_cruise_token and cr["speed_band_ok"] and cr["free_road_all"] and cr["curvature_ok"])

    # ---- the four event predicates, each computed under the LITERAL reading (primary) and under an ALT reading (see AMBIGUITIES.md AMB-7..10).
    # LITERAL: the event may lie anywhere in the window [t0-2 s, t0+15 s].   ALT: the event must lie inside [t0, t0+15 s] (and, for the merge slot, the
    # entering vehicle must run in the ego's direction, |heading difference| <= 30 deg = a lane change/merge rather than crossing traffic).
    safe = info["safe"]
    nc_ok = not is_cruise_token

    def find_steady(tmin):
        good = has & (lead_gap <= S["steady_max_gap_m"]) & (lead_spd >= S["steady_min_speed"]) & (np.abs(lead_acc) <= S["steady_max_abs_acc"])
        good = good & (np.arange(nT) >= tmin)
        best = None
        k = 0
        while k < nT:
            if not good[k]:
                k += 1
                continue
            e = k
            while e + 1 < nT and good[e + 1] and lead_trk[e + 1] == lead_trk[k]:
                e += 1
            dur = (e - k) * S["dt_nominal"]
            if best is None or dur > best[0] + 1e-9:
                best = (dur, k, e)
            k = e + 1
        st = dict(best_run_s=fnum(best[0], 2) if best else 0.0)
        if best:
            _, a_, b_ = best
            st.update(lead_track=int(lead_trk[a_]), run_start_rel_t0_s=fnum((a_ - T0_I) * S["dt_nominal"], 2), run_end_rel_t0_s=fnum((b_ - T0_I) * S["dt_nominal"], 2),
                      gap_max_m=fnum(lead_gap[a_:b_ + 1].max(), 2), gap_min_m=fnum(lead_gap[a_:b_ + 1].min(), 2),
                      lead_speed_min=fnum(lead_spd[a_:b_ + 1].min(), 2), lead_speed_max=fnum(lead_spd[a_:b_ + 1].max(), 2),
                      lead_abs_acc_max=fnum(np.abs(lead_acc[a_:b_ + 1]).max(), 3), lead_present_at_t0=bool(a_ <= T0_I <= b_))
        found = nc_ok and best is not None and best[0] >= S["steady_min_run_s"] - 1e-9
        st["pass"] = bool(found and safe)
        st["pass_before_safety"] = bool(found)
        return st

    def find_brake(tmin):
        br = dict(found=False)
        if nc and nc_ok and ev[T0_I] >= S["brake_min_ego_v0"]:
            nmax = int(round(S["brake_max_dur_s"] / S["dt_nominal"]))
            for a_ in range(tmin, nT):
                if not (has[a_] and lead_gap[a_] <= S["brake_max_gap_m"]):
                    continue
                lt = loc[int(lead_trk[a_])]
                v_a = spd[lt, a_]
                if np.isnan(v_a):
                    continue
                hit = None
                for b_ in range(a_ + 1, min(nT - 1, a_ + nmax) + 1):
                    v_b = spd[lt, b_]
                    if np.isnan(v_b) or v_a - v_b < S["brake_min_drop"]:
                        continue
                    seg_acc = acc[lt, a_:b_ + 1]
                    if np.isfinite(seg_acc).any() and -np.nanmin(seg_acc) >= S["brake_min_peak_decel"]:
                        hit = (b_, v_a - v_b, -float(np.nanmin(seg_acc)))
                        break
                if hit:
                    br = dict(found=True, lead_track=int(lead_trk[a_]), onset_rel_t0_s=fnum((a_ - T0_I) * S["dt_nominal"], 2),
                              t_end_rel_t0_s=fnum((hit[0] - T0_I) * S["dt_nominal"], 2), gap_onset_m=fnum(lead_gap[a_], 2), lead_v_onset=fnum(v_a, 2),
                              speed_drop=fnum(hit[1], 2), peak_decel=fnum(hit[2], 3), ego_v_t0=fnum(ev[T0_I], 2))
                    break
        br["pass"] = bool(br["found"] and safe)
        br["pass_before_safety"] = bool(br["found"])
        return br

    def find_stopgo(tmin):
        sg = dict(found=False)
        if nc and nc_ok:
            for f in range(max(1, tmin), nT):
                if lead_trk[f] < 0 or lead_trk[f] != lead_trk[f - 1]:
                    continue
                lt = loc[int(lead_trk[f])]
                if not (spd[lt, f - 1] >= S["stopgo_stop_v"] and spd[lt, f] < S["stopgo_stop_v"]):
                    continue
                g = None
                for j in range(f + 1, nT):
                    if lead_trk[j] != lead_trk[f]:
                        break
                    if spd[lt, j] > S["stopgo_go_v"]:
                        g = j
                        break
                if g is not None:
                    sg = dict(found=True, lead_track=int(lead_trk[f]), t_fall_rel_t0_s=fnum((f - T0_I) * S["dt_nominal"], 2),
                              t_go_rel_t0_s=fnum((g - T0_I) * S["dt_nominal"], 2), stopped_s=fnum((g - f) * S["dt_nominal"], 2),
                              gap_min_m=fnum(lead_gap[f - 1:g + 1].min(), 2), lead_v_before_fall=fnum(spd[lt, f - 1], 2))
                    break
        sg["pass"] = bool(sg["found"] and safe)
        sg["pass_before_safety"] = bool(sg["found"])
        return sg

    def find_merge(require_same_direction):
        mg = dict(found=False)
        if nc and nc_ok:
            ent = (ctick > T0_I) & (np.abs(lat_c) <= S["merge_in_lat"]) & ahead & (gap_c <= S["merge_max_gap_m"])
            if ent.any():
                ej = np.where(ent)[0]
                ej = ej[np.argsort(ctick[ej], kind="stable")]
                t0_rows = {int(otrk[i]): i for i in np.where((tick_of == T0_I) & oveh)[0]}
                tried = set()
                for j in ej:
                    tr = int(ctrk[j])
                    if tr in tried:
                        continue
                    if tr not in t0_rows:
                        tried.add(tr)
                        continue
                    i0 = t0_rows[tr]
                    s0, l0, _, _ = path.project(oc[i0:i0 + 1])
                    if abs(l0[0]) < S["merge_out_lat"]:
                        tried.add(tr)
                        continue
                    lt = loc[tr]
                    k = int(ctick[j])
                    v = vvec[lt, k]                    # heading of the vehicle's velocity vs the path tangent at entry (diagnostic; rule only in the ALT reading)
                    hd = float("nan")
                    if np.isfinite(v).all() and np.hypot(*v) > 1.0:
                        hd = math.degrees(abs(math.atan2(v[0] * tg_c[j, 1] - v[1] * tg_c[j, 0], v[0] * tg_c[j, 0] + v[1] * tg_c[j, 1])))
                    if require_same_direction and not (hd <= S["merge_alt_max_heading_deg"]):
                        continue                         # a LATER entry of the same vehicle may still qualify; do not mark it tried
                    tried.add(tr)
                    mg = dict(found=True, lead_track=tr, t_entry_rel_t0_s=fnum((k - T0_I) * S["dt_nominal"], 2), gap_entry_m=fnum(gap_c[j], 2),
                              lat_t0_m=fnum(l0[0], 2), lat_entry_m=fnum(lat_c[j], 2), vehicle_speed_entry=fnum(spd[lt, k], 2),
                              heading_diff_to_path_deg=fnum(hd, 1), n_entering_vehicles=int(len({int(ctrk[q]) for q in ej})))
                    break
        mg["pass"] = bool(mg["found"] and safe)
        mg["pass_before_safety"] = bool(mg["found"])
        return mg

    st, br, sg, mg = find_steady(0), find_brake(0), find_stopgo(0), find_merge(False)
    st_a, br_a, sg_a, mg_a = find_steady(T0_I), find_brake(T0_I), find_stopgo(T0_I), find_merge(True)

    if dbg is not None:       # arrays for the hand-verification script (not stored in the run outputs)
        dbg.update(T=T, ev=ev, s_ego=s_ego, lead_trk=lead_trk, lead_gap=lead_gap, lead_spd=lead_spd, lead_u=lead_u, lead_acc=lead_acc, ttc=ttc, free=free_sensible, kappa=kap,
                   track_names=ctx.cat_names, track_hex=ctx.trk_hex)
    info.update(cruise=cr, steady_lead=st, lead_braking=br, stop_go=sg, merge_cutin=mg,
                steady_lead_alt=st_a, lead_braking_alt=br_a, stop_go_alt=sg_a, merge_cutin_alt=mg_a)
    return info


# ----------------------------------------------------------------------------------------------------------------------------------------
def process_log(args):
    log, wanted, cruise_set = args
    t0 = time.time()
    res = dict(log=log, rows=[], status="ok", err="", n_wanted=len(wanted))
    try:
        ctx = load_log(f"{DB_DIR}/{log}.db")
        res["n_lidar_pc"] = ctx.n
        for tok in sorted(wanted):
            r0 = ctx.tok2rank.get(tok)
            if r0 is None:
                continue
            try:
                r = analyze(ctx, r0, tok in cruise_set)
            except Exception:
                r = dict(ok=False, reason="analyze_exception: " + traceback.format_exc()[-400:])
            r.update(token=tok, log=log, t0_timestamp_us=int(round(ctx.T[r0] * 1e6)), lidar_pc_rank=int(r0))
            res["rows"].append(r)
    except Exception:
        res["status"] = "failed"
        res["err"] = traceback.format_exc()[-600:]
    res["secs"] = round(time.time() - t0, 1)
    return res


SLOTS = ["cruise", "steady_lead", "lead_braking", "stop_go", "merge_cutin"]
SLOT_ORDER_FOR_DUP = SLOTS        # scorecard section 3 table order; used only to label duplicate-log collisions


def row_slot(r, slot, variant):
    """The predicate dict of a row for a slot under the LITERAL (primary) or ALT reading (cruise has no ALT)."""
    if variant == "alt" and slot != "cruise":
        return r[slot + "_alt"]
    return r[slot]


def _entry(r, rank, n, slot, variant):
    e = dict(rank=rank, token=r["token"], log=r["log"], order_sha256=order_key(r["token"]), is_candidate=bool(rank <= n))
    if rank <= 2 * n:
        e["predicate_values"] = dict(slot=row_slot(r, slot, variant),
                                     human_safety=dict(min_ttc_s=r["min_ttc_s"], safe_ttc=r["safe_ttc"], overlap_ticks=r["overlap_ticks"], safe_overlap=r["safe_overlap"]),
                                     ego_v0=r["ego_v0"], lead_t0=r["lead_t0"], lead_gap_t0=r["lead_gap_t0"], t0_timestamp_us=r["t0_timestamp_us"],
                                     n_scenes_in_window=r["n_scenes"], t_span_s=r["t_span_s"])
    return e


def _ordered(rows, passes):
    """SC 3.1 item 2: order the passing tokens by sha256(token + salt) ascending, keep the first token of each log."""
    passing = sorted([r for r in rows if passes(r)], key=lambda r: order_key(r["token"]))
    seen, ordered = set(), []
    for r in passing:
        if r["log"] in seen:
            continue
        seen.add(r["log"])
        ordered.append(r)
    return passing, ordered


def build_candidates(rows, variant="literal"):
    """SC 3.1 items 2 and 3 (duplicate-log part): first n_candidates of the ordered one-per-log list are the candidates."""
    out = {}
    ok_rows = [r for r in rows if r.get("ok")]
    n = SPEC["n_candidates"]
    for slot in SLOTS + ["lead_braking_or_stop_go"]:
        if slot == "lead_braking_or_stop_go":      # the scorecard TABLE has ONE slot "Lead braking / stop-go" (n = 2 / 4); the union pool, same ordering rule (AMB-12)
            ps = lambda r: row_slot(r, "lead_braking", variant)["pass"] or row_slot(r, "stop_go", variant)["pass"]
            pb = lambda r: row_slot(r, "lead_braking", variant)["pass_before_safety"] or row_slot(r, "stop_go", variant)["pass_before_safety"]
        else:
            ps = lambda r, slot=slot: row_slot(r, slot, variant)["pass"]
            pb = lambda r, slot=slot: row_slot(r, slot, variant)["pass_before_safety"]
        passing, ordered = _ordered(ok_rows, ps)
        entries = []
        for rank, r in enumerate(ordered, 1):
            if slot == "lead_braking_or_stop_go":
                e = _entry(r, rank, n, "lead_braking", variant)
                if "predicate_values" in e:
                    e["predicate_values"]["slot"] = dict(lead_braking=row_slot(r, "lead_braking", variant), stop_go=row_slot(r, "stop_go", variant))
            else:
                e = _entry(r, rank, n, slot, variant)
            entries.append(e)
        pre = [r for r in ok_rows if pb(r)]
        out[slot] = dict(
            n_tokens_tested=len(ok_rows), pool_tokens_passing=len(passing), pool_logs_passing=len(ordered),
            pool_tokens_before_human_safety=len(pre), pool_logs_before_human_safety=len({r["log"] for r in pre}),
            candidates=[e["token"] for e in entries[:n]], n_candidates=min(n, len(entries)), shortfall=max(0, n - len(entries)), ordered_one_per_log=entries)
    # duplicate-log label (SC 3.1 item 3, second bullet): another slot's candidate list holds the same log.  Informational: whether it is skipped depends on what the
    # sim session actually USES and on the fill order, so nothing is dropped here.
    cand_logs = {s: {e["log"]: e["rank"] for e in out[s]["ordered_one_per_log"][:n]} for s in SLOTS}
    for s in SLOTS:
        for e in out[s]["ordered_one_per_log"][:n]:
            e["same_log_in_other_slot_candidates"] = [dict(slot=t, rank=cand_logs[t][e["log"]]) for t in SLOTS if t != s and e["log"] in cand_logs[t]]
            e["same_token_in_other_slot_candidates"] = [t for t in SLOTS if t != s and e["token"] in out[t]["candidates"]]
    return out



# ----------------------------------------------------------------------------------------------------------------------------------------
# AMENDMENT_3 slots (J-UL, J-R, J-S, TL-G, TL-S): predicates in a3_slots.py (2 Hz OpenScene, AMENDMENT_3 section 2), human safe = this script's own flag
def build_a3_candidates(rows3, safe, A3):
    out = {}
    n = SPEC["n_candidates"]
    for slot in A3.SLOTS_A3:
        pre = [r for r in rows3 if A3.slot_pass(r, slot)]
        passing = sorted([r for r in pre if safe.get(r["token"])], key=lambda r: order_key(r["token"]))
        seen, ordered = set(), []
        for r in passing:
            if r["log"] in seen:
                continue
            seen.add(r["log"])
            ordered.append(r)
        entries = []
        for rank, r in enumerate(ordered, 1):
            e = dict(rank=rank, token=r["token"], log=r["log"], order_sha256=order_key(r["token"]), is_candidate=bool(rank <= n))
            if rank <= 2 * n:
                e["predicate_values"] = dict(city=r["city"], openscene_list_idx=r["openscene_list_idx"], map_fields=A3.slot_fields(r, slot), human_safe=True,
                                             slot_flags={k: r.get(k) for k in ([slot] if slot.startswith("J") else ["TL-G", "TL-G_c2code", "TL-G_strict", "gov_src", "gov_tied", "stop_gap_m", "tls_dur", "tls_frac", "sl_src", "tl_stage", "cross_on_red", "past_line_red"])})
            entries.append(e)
        out[slot] = dict(n_tokens_tested=len(rows3), pool_tokens_passing=len(passing), pool_logs_passing=len(ordered), pool_tokens_before_human_safety=len(pre),
                         pool_logs_before_human_safety=len({r["log"] for r in pre}), candidates=[e["token"] for e in entries[:n]], n_candidates=min(n, len(entries)),
                         shortfall=max(0, n - len(entries)), ordered_one_per_log=entries)
    return out


def stop_line_s_fields(a3c, log_ctx_cache=None):
    """AMENDMENT_3 section 5 `stop_line_s`: HumanPath.project (behaviour_score.py) of stop_line_xy over the BOX-CENTRE positions at the 10 Hz samples of the human window
    [t0-2 s, t0+15 s] (the same 171 ticks analyze() reads), with the scorer's straight extension (A-6).  Adds `stop_line_s` and `stop_line_s_diag` to each map_fields dict (ranks <= 2n)."""
    sys.path.insert(0, SCORER_DIR)
    os.environ.setdefault("BEHAV_VENDOR_DIR", SCORER_DIR)
    import behaviour_score as B
    by_log = collections.defaultdict(list)
    for slot, v in a3c.items():
        for e in v["ordered_one_per_log"]:
            if "predicate_values" in e and e["predicate_values"].get("map_fields"):
                by_log[e["log"]].append((slot, e))
    for log, lst in sorted(by_log.items()):
        ctx = load_log(f"{DB_DIR}/{log}.db")
        for slot, e in lst:
            r0 = ctx.tok2rank[e["token"]]
            K = np.arange(SPEC["k_lo"], SPEC["k_hi"] + 1)
            idx = r0 + SPEC["step"] * K
            yaw = ctx.eyaw[idx]
            cx, cy = ctx.ex[idx] + np.cos(yaw) * EGO_COG_OFF, ctx.ey[idx] + np.sin(yaw) * EGO_COG_OFF
            R = dict(x=cx, y=cy, yaw=np.unwrap(yaw), v=ctx.ev[idx])
            hp = B.HumanPath(R)
            mf = e["predicate_values"]["map_fields"]
            s_, lat_, _ = hp.project(np.array([mf["stop_line_xy"]]))
            mf["stop_line_s"] = float(s_[0])
            mf["stop_line_s_diag"] = dict(lateral_offset_m=float(lat_[0]), human_path_length_m=float(hp.length), s_front_at_t0=float(hp.project(np.array([[cx[T0_I] + 0.5 * 5.176 * np.cos(yaw[T0_I]), cy[T0_I] + 0.5 * 5.176 * np.sin(yaw[T0_I])]]))[0][0]),
                                          beyond_human_end=bool(s_[0] > hp.length), n_ticks=int(len(idx)))
    return a3c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-logs", type=int, default=0)
    ap.add_argument("--logs", default="", help="comma-separated log names (debug)")
    ap.add_argument("--select-only", action="store_true", help="reuse per_token_results.pkl")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    pkl_path = os.path.join(a.out, "per_token_results.pkl")

    # ---- input pins ------------------------------------------------------------------------------------------------------------------
    assert sha256_file(SCORECARD) == SCORECARD_SHA256, "scorecard changed"
    assert sha256_file(CRUISE_JSON) == CRUISE_JSON_SHA256, "tokens_cruise.json changed"
    assert sha256_file(YAML) == YAML_SHA256, "navtest.yaml changed"
    logs, tokens = parse_yaml()
    assert len(tokens) == len(set(tokens)) == 12146 and len(logs) == len(set(logs)) == 136, (len(tokens), len(logs))
    cruise = json.load(open(CRUISE_JSON, encoding="utf-8"))
    cruise_tokens = list(cruise["tokens"])
    assert len(cruise_tokens) == 154
    cruise_set = set(cruise_tokens)
    tokset = set(tokens)

    if not a.select_only:
        sel_logs = logs
        if a.logs:
            sel_logs = [x for x in a.logs.split(",") if x]
        if a.limit_logs:
            sel_logs = sel_logs[:a.limit_logs]
        missing_db = [l for l in sel_logs if not os.path.exists(f"{DB_DIR}/{l}.db")]
        assert not missing_db, missing_db
        # biggest DBs first so that no worker is left alone with a 3 GB log at the end
        sel_logs = sorted(sel_logs, key=lambda l: -os.path.getsize(f"{DB_DIR}/{l}.db"))
        jobs = [(l, tokset, cruise_set) for l in sel_logs]
        rows, reports = [], []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, res in enumerate(ex.map(process_log, jobs), 1):
                rows += res.pop("rows")
                reports.append(res)
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} {res['log']}: {res['status']} {res.get('secs')}s tokens so far {len(rows)}", flush=True)
                if res["status"] != "ok":
                    print(res["err"], flush=True)
        rows.sort(key=lambda r: r["token"])
        pickle.dump(dict(rows=rows, reports=reports, spec=SPEC), open(pkl_path, "wb"))
    else:
        d = pickle.load(open(pkl_path, "rb"))
        rows, reports = d["rows"], d["reports"]

    # ---- population accounting -------------------------------------------------------------------------------------------------------
    found = {r["token"] for r in rows}
    full_run = not (a.logs or a.limit_logs)
    unmapped = sorted(tokset - found) if full_run else []
    n_ok = sum(1 for r in rows if r.get("ok"))
    excl = collections.Counter(r.get("reason", "") for r in rows if not r.get("ok"))
    cands = build_candidates(rows, "literal")
    cands_alt = build_candidates(rows, "alt")

    import a3_slots as A3
    rows3, rep3 = A3.run_a3(tokset)
    safe = {r["token"]: bool(r.get("ok") and r.get("safe")) for r in rows}
    pickle.dump(dict(rows3=rows3, safe=safe), open(os.path.join(a.out, "rows3_a6.pkl"), "wb"))
    a3c = build_a3_candidates(rows3, safe, A3)
    a3c = stop_line_s_fields(a3c)

    pop = dict(n_navtest_tokens=len(tokens), n_rows=len(rows), n_window_ok=n_ok, excluded_reasons=dict(excl), n_tokens_unmapped=len(unmapped), unmapped_tokens_head=unmapped[:10],
               n_cruise_list_tokens=len(cruise_tokens), n_cruise_list_tokens_window_ok=sum(1 for r in rows if r.get("ok") and r["cruise"]["token_in_cruise_list"]))
    cj = dict(spec=SPEC, reading="LITERAL (primary): BEHAVIOUR_SCORECARD.md section 3.1 as written; every other choice is in AMBIGUITIES.md", scorecard_sha256=SCORECARD_SHA256,
              population=pop, slots=cands)
    json.dump(cj, open(os.path.join(a.out, "SUITE_CANDIDATES.json"), "w"), indent=1)
    cja = dict(spec=SPEC, reading="ALT (secondary, NOT the frozen rule): literal + the event must lie in [t0, t0+15 s]; merge slot also requires |heading diff| <= 30 deg. "
                                  "Cruise is identical to the literal list.", scorecard_sha256=SCORECARD_SHA256, population=pop, slots=cands_alt)
    json.dump(cja, open(os.path.join(a.out, "SUITE_CANDIDATES_ALT.json"), "w"), indent=1)

    a3j = dict(spec=SPEC, reading="AMENDMENT_3 (frozen, sha256 886e4465...) + section 9 rulings + AMENDMENT_6 (sha256 073635ce...: stop line = connector start; straight extension along the chord over the last 5 m of the moving part). `slots` = the six legacy slots, byte-for-byte the legacy build, plus J-UL J-R J-S TL-G TL-S",
               scorecard_sha256=SCORECARD_SHA256, population=pop, a3_population=rep3, slots=dict(cands, **a3c))
    json.dump(a3j, open(os.path.join(a.out, "SUITE_CANDIDATES_A6.json"), "w"), indent=1)
    mapf = {slot: {e["token"]: dict(rank=e["rank"], log=e["log"], is_candidate=e["is_candidate"], **e["predicate_values"]["map_fields"])
                   for e in v["ordered_one_per_log"] if "predicate_values" in e} for slot, v in a3c.items()}
    json.dump(dict(spec="AMENDMENT_3 section 5; stop_line_s on the box-centre 10 Hz human path (HumanPath.project); conflict_agent_ids are OpenScene track tokens (the sim maps them to agent_id); "
                        "conflict_zone_xy is the CCW convex hull of the union of the overlaps, no repeated closing vertex, [] = empty (0,2); heads/signal_governs_ego sorted ascending by connector id",
                   fields=mapf), open(os.path.join(a.out, "MAP_FIELDS_A6.json"), "w"), indent=1)

    # flat table
    import csv
    flat_path = os.path.join(a.out, "suite_features.csv.gz")
    cols = ["token", "log", "ok", "reason", "t0_timestamp_us", "ego_v0", "path_len_m", "n_scenes", "lead_t0", "lead_gap_t0", "lead_ticks_frac", "min_ttc_s", "safe_ttc",
            "overlap_ticks", "overlap_first_category", "safe"]
    extra = ([(s, k) for s in SLOTS for k in ("pass", "pass_before_safety")] + [("cruise", "pass_literal_free")]
             + [(s + "_alt", k) for s in SLOTS if s != "cruise" for k in ("pass", "pass_before_safety")])
    with gzip.open(flat_path, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + [f"{s}.{k}" for s, k in extra])
        for r in rows:
            w.writerow([r.get(c, "") for c in cols] + [r[s][k] if r.get("ok") else "" for s, k in extra])

    rep = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), secs=round(time.time() - t_start, 1), n_logs=len(reports),
               log_failures=[x["log"] for x in reports if x["status"] != "ok"], n_rows=len(rows), n_window_ok=n_ok, excluded_reasons=dict(excl),
               pools_literal={s: dict(tokens=v["pool_tokens_passing"], logs=v["pool_logs_passing"], before_safety_tokens=v["pool_tokens_before_human_safety"]) for s, v in cands.items()},
               pools_alt={s: dict(tokens=v["pool_tokens_passing"], logs=v["pool_logs_passing"], before_safety_tokens=v["pool_tokens_before_human_safety"]) for s, v in cands_alt.items()})
    json.dump(rep, open(os.path.join(a.out, "run_report.json"), "w"), indent=1)

    # ---- pins (only for a full run) ----------------------------------------------------------------------------------------------------
    if full_run:
        dbl = sorted((l, os.path.getsize(f"{DB_DIR}/{l}.db")) for l in logs)
        pin = dict(
            created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), python=sys.version.split()[0], numpy=np.__version__, scipy=__import__("scipy").__version__,
            script=dict(path="tsf_bundle2_20261107/code/select_suite_tokens.py", sha256=sha256_file(os.path.abspath(__file__))),
            spec_sha256=hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest(),
            inputs=dict(
                scorecard=dict(path=SCORECARD, sha256=SCORECARD_SHA256),
                tokens_cruise_json=dict(path=CRUISE_JSON, sha256=CRUISE_JSON_SHA256, n_tokens_list=len(cruise_tokens)),
                navtest_yaml=dict(path=YAML, sha256=YAML_SHA256, n_tokens=len(tokens), n_logs=len(logs),
                                  token_list_sha256_sorted_newline_joined=hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest()),
                nuplan_db=dict(dir=DB_DIR, n_files_used=len(dbl), identity="sha256 over sorted 'name:size' lines of the 136 navtest DB files (file CONTENT was not hashed: ~25 GB on a contended drive)",
                               identity_sha256=hashlib.sha256("\n".join(f"{l}.db:{s}" for l, s in dbl).encode()).hexdigest(), total_bytes=sum(s for _, s in dbl))),
            outputs=dict(SUITE_CANDIDATES_json_sha256=sha256_file(os.path.join(a.out, "SUITE_CANDIDATES.json")),
                         SUITE_CANDIDATES_ALT_json_sha256=sha256_file(os.path.join(a.out, "SUITE_CANDIDATES_ALT.json")),
                         suite_features_csv_gz_sha256=sha256_file(flat_path),
                         candidate_tokens_literal={s: cands[s]["candidates"] for s in SLOTS},
                         candidate_tokens_alt={s: cands_alt[s]["candidates"] for s in SLOTS}),
            note="Pinned by the run that wrote the outputs; the sim session re-runs `python -I select_suite_tokens.py --out <dir>` and compares SUITE_CANDIDATES.json.")
        pin["outputs"]["SUITE_CANDIDATES_A6_json_sha256"] = sha256_file(os.path.join(a.out, "SUITE_CANDIDATES_A6.json"))
        pin["outputs"]["MAP_FIELDS_A6_json_sha256"] = sha256_file(os.path.join(a.out, "MAP_FIELDS_A6.json"))
        pin["outputs"]["candidate_tokens_a6"] = {sl: a3c[sl]["candidates"] for sl in A3.SLOTS_A3}
        pin["a6"] = dict(amendment_6_sha256=sha256_file("E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/AMENDMENT_6_STOP_LINE.md"))
        pin["a3"] = dict(amendment_3_sha256=sha256_file("E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/AMENDMENT_3.md"),
                         a3_slots_py_sha256=sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "a3_slots.py")),
                         feasibility_a3_py_sha256=sha256_file("E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/amendment3/feasibility_a3.py"),
                         behaviour_score_py_sha256=sha256_file(SCORER_DIR + "/behaviour_score.py"), a3_report=rep3,
                         pools={sl: dict(tokens=a3c[sl]["pool_tokens_passing"], logs=a3c[sl]["pool_logs_passing"], shortfall=a3c[sl]["shortfall"]) for sl in A3.SLOTS_A3})
        pin["script"]["path"] = "tsf_bundle2_20261107/a6/select_suite_tokens.py"
        json.dump(pin, open(os.path.join(a.out, "SUITE_PIN_A6.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print("candidates literal:", {s: cands[s]["candidates"] for s in SLOTS})
    print("candidates a3:", {sl: a3c[sl]["candidates"] for sl in A3.SLOTS_A3})
    print("candidates alt:", {s: cands_alt[s]["candidates"] for s in SLOTS})


if __name__ == "__main__":
    main()
