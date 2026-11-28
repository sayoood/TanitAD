#!/usr/bin/env python
"""strat_roll_read.py -- reader of card PREREG_V8F_PROOF_ADD66 (STRATROLL10K): the strategic head's route checkpoints along the model's own rollout,
read from the banked tick files. Study instrument; CPU only; reads no model; starts no rollout.

Modes:  --selftest      control s4 only (synthetic tracks; one planted fault must fail); needs no data
        --controls-only controls s1 s2 s3 s6 s9 (+ s4, + the diagnostic s3b) on the real files; computes NO outcome cell
        (default)       the full read: controls first; only if s1-s6, s8, s9 pass are the cells computed and the names given (s7 gates ROLL only)
Definitions as coded: STRATROLL10K_DEFS.md. Frames: world frame of the tool's pose rows; grid step = the cache step (0.2 s), logged pose of step j = native track row 2j.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD66 (STRATROLL10K) with Amendment 1 (the route label is the programme's own label of the anchor step)"
TIER = ("first tick of each window = T0 (logged history only); later ticks = T1-kin (kinematic rollout on the model's own actions, logged camera frames, "
        "the lead does not react); never driving performance")
ARCS = (30.0, 50.0, 80.0)
CELLS = ("pos", "lat", "along", "head")
BASE_RUNS = ("R0", "R0C10", "R0G0", "R0G0X", "R0Bk10", "RLc", "RQ50", "RLcF5", "RQ50F5")
CAND_RUNS = ("RLcK", "RQ50K", "RLcKF5", "RQ50KF5")
SEEDS = (0, 1)
SEARCH = 25                 # anchor search half-width, steps
B_BOOT = 2000
TICK_DJ = (0, 10, 20)       # j - t0
TICK_AGE = (0, 4, 2)        # age_s
N_WIN, N_ROWS, N_EP = 2399, 7197, 141
DEF_W = "/workspace/v8proof/integ10k"
DEF_TRACKS = "/workspace/v8proof/data/inputs/native_tracks.pt"
DEF_JOIN = "/workspace/v8proof/hierrep/pins/join_eval141.npz"
DEF_CAND_JSON = "/workspace/v8proof/podread/out/read_cand/READ_CAND10K.json"
FORMS = (("LABEL", "RLc"), ("READER", "RQ50"))


def wrap(a):
    return (np.asarray(a, np.float64) + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------------------------------------ geometry (pure numpy, tested by s4)
def _finite_prefix(a):
    bad = ~np.isfinite(a).all(axis=1)
    return a[: int(np.argmax(bad))] if bad.any() else a


def label_card(G, a, arcs=ARCS):
    """Route label of anchor step ``a`` on the logged grid G [n, 3] (x, y, yaw), by the rule of ``v8f_mini_data.py`` lines 1213-1223 (cumulative arc over
    the positions AFTER the anchor, arc 0 at step a+1; first step whose arc reaches A; heading = that step's yaw). Returns pos [3, 2], yaw [3] (world), step [3], ok [3]."""
    pos, yaw, step, ok = np.full((len(arcs), 2), np.nan), np.full(len(arcs), np.nan), np.full(len(arcs), -1), np.zeros(len(arcs), bool)
    fut = _finite_prefix(G[a + 1:, :2]) if a + 1 < len(G) else np.zeros((0, 2))
    if len(fut) < 2:
        return pos, yaw, step, ok
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(fut, axis=0), axis=1))])
    for m, A in enumerate(arcs):
        if arc[-1] >= A:
            j = int(np.searchsorted(arc, A))
            s = a + 1 + j
            pos[m], yaw[m], step[m], ok[m] = G[s, :2], G[s, 2], s, True
    return pos, yaw, step, ok


def route_geom(G, a, pts, arcs=ARCS):
    """Distance of each point of pts [3, 2] to the logged route (polyline of the logged positions from the anchor ``a`` on): lat [3], lat_ok [3] (nearest
    point is not the last logged point), along [3] = arc from the first logged step after the anchor (the label's arc 0) to the foot, minus A."""
    n_m = len(pts)
    lat, along, ok = np.full(n_m, np.nan), np.full(n_m, np.nan), np.zeros(n_m, bool)
    poly = _finite_prefix(G[a:, :2])
    if len(poly) < 3:
        return lat, along, ok
    p0, d = poly[:-1], np.diff(poly, axis=0)
    L2 = (d ** 2).sum(1)
    L = np.sqrt(L2)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    nseg = len(L)
    for m in range(n_m):
        rel = pts[m][None, :] - p0
        u = np.where(L2 > 0, (rel * d).sum(1) / np.where(L2 > 0, L2, 1.0), 0.0)
        u = np.clip(u, 0.0, 1.0)
        foot = p0 + u[:, None] * d
        dist = np.linalg.norm(pts[m][None, :] - foot, axis=1)
        i = int(np.argmin(dist))
        lat[m] = dist[i]
        ok[m] = not (i == nseg - 1 and u[i] >= 1.0 - 1e-12)
        along[m] = (cum[i] + u[i] * L[i]) - cum[1] - arcs[m]
    return lat, along, ok


def label_join(J, c, G, a, sign=1.0):
    """Amendment 1 of the card: the programme's own route label of logged step ``a`` of clip index ``c`` (evaluation join ``rc`` / ``rc_ok``, given in the frame
    of the logged pose at ``a``), put into the world frame with that logged pose G[a]. Nothing is recomputed. Returns pos [3, 2], yaw [3] (world), ok [3].
    ``sign`` is 1.0; the planted fault of control s4 passes -1.0 (the heading of the wrong sign)."""
    n_m = len(ARCS)
    pos, yaw, ok = np.full((n_m, 2), np.nan), np.full(n_m, np.nan), np.zeros(n_m, bool)
    if a < 0 or a >= int(J["n"][c]) or a >= len(G) or not np.isfinite(G[a]).all():
        return pos, yaw, ok
    i = int(J["off"][c]) + a
    rc, rok = np.asarray(J["rc"][i], np.float64), np.asarray(J["rc_ok"][i], bool)
    cs, sn = math.cos(sign * G[a, 2]), math.sin(sign * G[a, 2])
    ok[:] = rok & np.isfinite(rc).all(axis=1)
    pos[:, 0] = G[a, 0] + cs * rc[:, 0] - sn * rc[:, 1]
    pos[:, 1] = G[a, 1] + sn * rc[:, 0] + cs * rc[:, 1]
    yaw[:] = wrap(G[a, 2] + rc[:, 2])
    pos[~ok] = np.nan
    yaw[~ok] = np.nan
    return pos, yaw, ok


def row_cells(G, a, ck_xy, ck_psi, label, arcs=ARCS):
    """The four cells at three arcs for one checkpoint triple (world frame) against ``label`` = (pos [3, 2], yaw [3], ok [3]) in the world frame. NaN where invalid.
    ``along`` (Amendment 1): the arc along the logged route of the foot of the checkpoint minus that of the foot of the label point."""
    lpos, lyaw, lok = label
    lok = np.asarray(lok, bool)
    pos = np.where(lok, np.linalg.norm(ck_xy - np.where(lok[:, None], lpos, 0.0), axis=1), np.nan)
    head = np.where(lok, np.abs(wrap(ck_psi - np.where(lok, lyaw, 0.0))) * 180.0 / math.pi, np.nan)
    lat, arc_ck, gok = route_geom(G, a, ck_xy, arcs)
    _, arc_lab, lgok = route_geom(G, a, np.where(lok[:, None], lpos, G[a, :2][None, :]), arcs)
    along = np.where(gok & lgok & lok, arc_ck - arc_lab, np.nan)
    return pos, np.where(gok, lat, np.nan), along, head


def ckpt_world(s_pose, ck):
    c, s = math.cos(s_pose[2]), math.sin(s_pose[2])
    x = s_pose[0] + c * ck[:, 0] - s * ck[:, 1]
    y = s_pose[1] + s * ck[:, 0] + c * ck[:, 1]
    return np.stack([x, y], 1), wrap(ck[:, 2] + s_pose[2])


def find_anchor(G32, center, sp_xy):
    """Logged step nearest to the position of s_pose within SEARCH steps of ``center``; ties (within 1e-6 m) go to the step nearest ``center``. G32: logged xy rounded to float32
    (s_pose is float32). Returns (step, distance)."""
    lo, hi = max(0, center - SEARCH), min(len(G32) - 1, center + SEARCH)
    d = np.linalg.norm(G32[lo:hi + 1].astype(np.float64) - sp_xy[None, :], axis=1)
    d = np.where(np.isfinite(d), d, np.inf)
    dm = d.min()
    cand = np.nonzero(d <= dm + 1e-6)[0] + lo
    s = int(cand[np.argmin(np.abs(cand - center))])
    return s, float(d[s - lo])


# ------------------------------------------------------------------------------------------------ s4: synthetic known values
def synth_line(n=100, step=2.0):
    i = np.arange(n, dtype=np.float64)
    return np.stack([step * i, 0 * i, 0 * i], 1)


def synth_circle(R=100.0, n=160, chord_arc=2.0):
    th = 2 * math.asin(chord_arc / (2 * R))
    i = np.arange(n, dtype=np.float64)
    return np.stack([R * np.sin(th * i), R * (1 - np.cos(th * i)), th * i], 1), th


def s4_checks(label_fn):
    """Control s4 on synthetic tracks. ``label_fn`` is the label rule under test (the real one, or the planted fault)."""
    out = []
    G = synth_line(100, 2.0)
    a = 5
    lpos, lyaw, lstep, lok = label_fn(G, a, ARCS)
    exp_step = [a + 1 + int(math.ceil(A / 2.0)) for A in ARCS]
    out.append(("line: labels at the first logged steps reaching 30/50/80 m", bool(lok.all() and list(lstep) == exp_step
               and np.allclose(lpos[:, 0], [2.0 * s for s in exp_step]) and np.allclose(lpos[:, 1], 0))))
    ck = lpos.copy()
    cps = np.zeros(3)
    pos = np.where(lok, np.linalg.norm(ck - lpos, axis=1), np.nan)
    lat, along, gok = route_geom(G, a, ck, ARCS)
    out.append(("line: checkpoint on the label -> pos 0, lat 0, head 0",
                bool(np.allclose(pos, 0) and np.allclose(lat, 0) and np.allclose(np.abs(wrap(cps - lyaw)), 0))))
    ck2 = lpos + np.array([0.0, 1.5])
    pos2 = np.linalg.norm(ck2 - lpos, axis=1)
    lat2, along2, _ = route_geom(G, a, ck2, ARCS)
    out.append(("line: checkpoint 1.5 m sideways -> lat 1.5, pos 1.5, along unchanged",
                bool(np.allclose(lat2, 1.5) and np.allclose(pos2, 1.5) and np.allclose(along2, along))))
    ck3 = lpos + np.array([4.0, 0.0])
    lat3, along3, _ = route_geom(G, a, ck3, ARCS)
    out.append(("line: checkpoint 4 m along the track -> along +4, lat 0", bool(np.allclose(along3 - along, 4.0) and np.allclose(lat3, 0))))
    out.append(("line: checkpoint on the label -> along = label arc minus A (30/50/80 are multiples of the 2 m step: 0)", bool(np.allclose(along, 0.0))))
    # circle
    Gc, th = synth_circle(100.0, 160, 2.0)
    carcs = (31.0, 51.0, 81.0)                     # not multiples of the 2 m chord: the first step reaching A is unambiguous in floating point
    lpos_c, lyaw_c, lstep_c, lok_c = label_fn(Gc, a, carcs)
    exp = [a + 1 + int(math.ceil(A / 2.0)) for A in carcs]
    ana = np.array([[100.0 * math.sin(th * s), 100.0 * (1 - math.cos(th * s))] for s in exp])
    out.append(("circle R=100: label positions equal the analytic ones to the step", bool(lok_c.all() and list(lstep_c) == exp
               and np.allclose(lpos_c, ana, atol=1e-9) and np.allclose(lyaw_c, [th * s for s in exp], atol=1e-12))))
    # short track
    Gs = synth_line(36, 2.0)       # positions to 70 m; from step 5 the arc after step 6 is 58 m -> 30 and 50 valid, 80 invalid
    _, _, _, lok_s = label_fn(Gs, a, ARCS)
    out.append(("short track: 30 and 50 valid, 80 invalid", bool(list(lok_s) == [True, True, False])))
    return out


def s4_join_checks(sign=1.0):
    """Amendment 1: the label path of the data (label_join -> row_cells) on synthetic known values. ``sign`` -1.0 is the planted fault."""
    out = []
    # a logged pose with a known heading: G[a] = (10, 5, pi/2); label (30, 2, 0.1) in its frame -> world (10 - 2, 5 + 30), heading pi/2 + 0.1
    G = np.zeros((40, 3))
    G[:, 0], G[:, 1], G[:, 2] = 10.0, 5.0 + 3.0 * np.arange(40), math.pi / 2          # a straight track heading +y, 3 m per step, step 0 at (10, 5)
    a = 0
    rc = np.zeros((40, 3, 3), np.float32)
    rc[:, 0], rc[:, 1], rc[:, 2] = (30.0, 2.0, 0.1), (51.0, 0.0, 0.0), (81.0, 0.0, 0.0)
    rok = np.ones((40, 3), bool)
    rok[:, 2] = False
    J = {"n": np.array([30]), "off": np.array([0]), "rc": rc, "rc_ok": rok}
    lpos, lyaw, lok = label_join(J, 0, G, a, sign)
    out.append(("join label: known pose and label -> analytic world position and heading",
                bool(np.allclose(lpos[0], [8.0, 35.0], atol=1e-9) and abs(float(wrap(lyaw[0] - (math.pi / 2 + 0.1)))) < 1e-6 and np.allclose(lpos[1], [10.0, 56.0], atol=1e-9))))
    out.append(("join label: validity passes through (third arc invalid), a step outside the clip's label rows reads invalid",
                bool(list(lok) == [True, True, False] and not label_join(J, 0, G, 35, sign)[2].any())))
    ck, cps = lpos.copy(), lyaw.copy()
    ck[2], cps[2] = (10.0, 86.0), math.pi / 2
    pos, lat, along, head = row_cells(G, a, ck, cps, (lpos, lyaw, lok))
    out.append(("join label: checkpoint on its label -> pos 0, head 0, along 0; invalid arc reads NaN",
                bool(np.allclose(pos[:2], 0) and np.allclose(head[:2], 0) and np.allclose(along[:2], 0) and np.isnan(pos[2]) and np.isnan(head[2]) and np.isnan(along[2]))))
    ck4 = ck.copy()
    ck4[1] = ck[1] + np.array([0.0, 4.0])
    pos4, lat4, along4, _ = row_cells(G, a, ck4, cps, (lpos, lyaw, lok))
    out.append(("join label: checkpoint 4 m farther along the straight track -> along 4, pos 4, lat 0",
                bool(abs(along4[1] - 4.0) < 1e-9 and abs(pos4[1] - 4.0) < 1e-9 and abs(lat4[1]) < 1e-9)))
    return out


def run_s4():
    good = s4_checks(label_card) + s4_join_checks(1.0)

    def fault(G, a, arcs=ARCS):                     # planted: the label taken one arc too far (each arc replaced by the next, the last by +40 m)
        return label_card(G, a, tuple(arcs[1:]) + (arcs[-1] + 40.0,))
    bad = s4_checks(fault)
    bad2 = s4_join_checks(-1.0)                     # planted (Amendment 1): the label put into the world frame with the heading of the wrong sign
    held = all(v for _, v in good)
    fault_caught = (not all(v for _, v in bad)) and (not all(v for _, v in bad2))
    return {"pass": bool(held and fault_caught), "checks": [{"name": n, "ok": v} for n, v in good], "all_checks_hold": held,
            "planted_fault_fails": fault_caught, "planted_fault_checks_failed": [n for n, v in bad if not v],
            "planted_fault_2_checks_failed": [n for n, v in bad2 if not v]}


# ------------------------------------------------------------------------------------------------ estimator
def boot(vals, eids, B=B_BOOT, seed=0):
    """Paired episode-cluster bootstrap of the mean of ``vals`` (list over model seeds of per-window differences, NaN dropped upstream), pooled over seeds; a drawn episode
    brings all its windows of both seeds. Returns dict(n, per_seed, mean, lo, hi, verdict) (lower is better)."""
    v = [np.asarray(x, np.float64) for x in vals]
    e = [np.asarray(x, np.int64) for x in eids]
    per_seed = [float(x.mean()) if len(x) else float("nan") for x in v]
    va, ea = np.concatenate(v) if v else np.zeros(0), np.concatenate(e) if e else np.zeros(0, np.int64)
    if len(va) == 0:
        return {"n": 0, "n_per_seed": [len(x) for x in v], "per_seed": per_seed, "mean": None, "lo": None, "hi": None, "verdict": "NO-DATA", "n_episodes": 0}
    ue, inv = np.unique(ea, return_inverse=True)
    S = np.bincount(inv, weights=va, minlength=len(ue))
    N = np.bincount(inv, minlength=len(ue)).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ue), size=(B, len(ue)))
    st = S[idx].sum(1) / N[idx].sum(1)
    lo, hi = np.percentile(st, 2.5), np.percentile(st, 97.5)
    return {"n": int(len(va)), "n_per_seed": [len(x) for x in v], "per_seed": per_seed, "mean": float(va.mean()), "lo": float(lo), "hi": float(hi),
            "verdict": verdict(per_seed, lo, hi), "n_episodes": int(len(ue))}


def verdict(per_seed, lo, hi):
    if any(not np.isfinite(x) for x in per_seed):
        return "NO-DATA"
    if all(x < 0 for x in per_seed) and hi < 0:
        return "BETTER"
    if all(x > 0 for x in per_seed) and lo > 0:
        return "WORSE"
    return "NOT-SEPARATED"


def family(verds):
    if all(x == "BETTER" for x in verds):
        return "BETTER"
    if all(x == "WORSE" for x in verds):
        return "WORSE"
    if all(x == "NOT-SEPARATED" for x in verds):
        return "NOT-SEPARATED"
    return "MIXED"


# ------------------------------------------------------------------------------------------------ data
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def tick_path(W, run, k):
    nm = f"I10{run}_s{k}"
    return os.path.join(W, "t1", nm, "ticks", f"ticks_{nm}_m10_causal_heads.npz")


FIELDS = ("tick", "j", "t0", "win", "ci", "age_s", "pose", "s_ckpt", "s_pose", "s_ckpt_w0", "pose_w0", "clip_names")


def load_ticks(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in FIELDS}


class Ctx:
    def __init__(self, a):
        self.a = a
        self.W = a.w
        self._grid = {}
        self.tracks = None
        self.J = None

    def load_static(self):
        import torch
        o = torch.load(self.a.tracks, map_location="cpu", weights_only=True)
        self.tracks = {c: t.double().numpy() for c, t in o["tracks"].items()}
        Jz = np.load(self.a.join, allow_pickle=True)
        self.J = {k: np.asarray(Jz[k]) for k in ("clip", "off", "n", "rc", "rc_ok")}      # in memory once (label_join reads it per row)

    def grid(self, name):
        g = self._grid.get(name)
        if g is None:
            tr = self.tracks[str(name)]
            G = np.asarray(tr, np.float64)[0::2, :3].copy()
            g = (G, G[:, :2].astype(np.float32))
            self._grid[name] = g
        return g


def structure(T, name, ref=None):
    """Control s6 for one tick file; returns (ok, detail)."""
    det = {}
    ok = True
    ok &= len(T["tick"]) == N_ROWS
    for k in range(3):
        m = T["tick"] == k
        ok &= int(m.sum()) == N_WIN
        ok &= bool((T["j"][m] - T["t0"][m] == TICK_DJ[k]).all()) and bool((T["age_s"][m] == TICK_AGE[k]).all())
    m0 = T["tick"] == 0
    o0 = np.argsort(T["win"][m0], kind="stable")
    ok &= bool((np.sort(T["win"][m0]) == np.arange(N_WIN)).all())
    ok &= len(np.unique(T["ci"])) == N_EP
    for k in (1, 2):                                   # the same windows at every tick
        mk = T["tick"] == k
        ok &= bool((T["ci"][mk][np.argsort(T["win"][mk])] == T["ci"][m0][o0]).all() and (T["t0"][mk][np.argsort(T["win"][mk])] == T["t0"][m0][o0]).all())
    if ref is not None:
        ok &= bool((ref["ci"] == T["ci"][m0][o0]).all() and (ref["t0"] == T["t0"][m0][o0]).all())
    det["rows"], det["episodes_clips"] = int(len(T["tick"])), int(len(np.unique(T["ci"])))
    return bool(ok), det


def window_order(T):
    """Per tick k: row indices ordered by window."""
    return [np.nonzero(T["tick"] == k)[0][np.argsort(T["win"][T["tick"] == k], kind="stable")] for k in range(3)]


def episode_map(W, T, ref_run="R0", seed=0):
    """eid per window from the dump files of the reference run: key (clip_index, ws)."""
    import glob
    d = os.path.join(W, "t1", f"I10{ref_run}_s{seed}", f"dump_I10{ref_run}_s{seed}_m10_causal_heads")
    key = {}
    for f in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        for e, w, c in zip(z["eid"], z["ws"], z["clip_index"]):
            key[(int(c), int(w))] = int(e)
    o = window_order(T)[0]
    eid = np.array([key.get((int(T["ci"][r]), int(T["t0"][r])), -1) for r in o], np.int64)
    return eid, len(key)


# ------------------------------------------------------------------------------------------------ controls s1 s2 s3 s6 s9
def s1_row_check(T):
    """Largest difference of the reader's own transform of s_ckpt (s_pose frame) into the window-origin frame (pose_w0) against the tool's s_ckpt_w0."""
    ck = T["s_ckpt"].astype(np.float64)
    sp, dp = T["s_pose"].astype(np.float64), T["pose_w0"].astype(np.float64)
    cs, ss = np.cos(sp[:, 2])[:, None], np.sin(sp[:, 2])[:, None]
    wx = sp[:, 0:1] + cs * ck[..., 0] - ss * ck[..., 1]
    wy = sp[:, 1:2] + ss * ck[..., 0] + cs * ck[..., 1]
    dx, dy = wx - dp[:, 0:1], wy - dp[:, 1:2]
    cd, sd = np.cos(dp[:, 2])[:, None], np.sin(dp[:, 2])[:, None]
    mine = np.stack([cd * dx + sd * dy, -sd * dx + cd * dy, wrap(ck[..., 2] + sp[:, 2:3] - dp[:, 2:3])], -1)
    tool = T["s_ckpt_w0"].astype(np.float64)
    dxy = float(np.abs(mine[..., :2] - tool[..., :2]).max())
    dps = float(np.abs(wrap(mine[..., 2] - tool[..., 2])).max())
    return dxy, dps


def anchors_for(ctx, T, rows_by_tick, ticks=(0, 1, 2)):
    """Anchor step and distance for every window of the given ticks. Returns dict k -> (step[N], dist[N])."""
    out = {}
    names = T["clip_names"]
    for k in ticks:
        r = rows_by_tick[k]
        st, ds = np.zeros(len(r), np.int64), np.zeros(len(r))
        for i, row in enumerate(r):
            G, G32 = ctx.grid(str(names[T["ci"][row]]))
            c = int(T["j"][row]) - int(T["age_s"][row])
            st[i], ds[i] = find_anchor(G32, c, T["s_pose"][row, :2].astype(np.float64))
        out[k] = (st, ds)
    return out


def s3_check(ctx, T, rows_by_tick, tree_ov=None):
    """Control s3 as Amendment 1 wrote it: on the first-tick rows, ``pos`` and ``head`` as the reader computes them (world frame, the join's label through the logged pose)
    equal ``pos_err_m`` / ``head_err_deg`` of the other session's ``checkpoint_errors`` (taniteval/v8f_consistency.py) on the raw ``s_ckpt`` and the join's ``rc`` / ``rc_ok``
    of step t0. Only differences are returned (no level)."""
    J, names = ctx.J, T["clip_names"]
    if not (np.asarray(J["clip"]) == np.asarray(names)).all():
        return {"pass": False, "computable": False, "reason": "join clip order differs from the tick file's clip_names"}
    fn, src = None, None
    for root in ([tree_ov] if tree_ov else []) + [os.path.join(DEF_W, "build", "tree_ov")]:
        try:
            p = os.path.join(root, "taniteval")
            if p not in sys.path:
                sys.path.insert(0, p)
            from taniteval import v8f_consistency as VC      # noqa: E402
            fn, src = VC.checkpoint_errors, os.path.join(p, "taniteval", "v8f_consistency.py")
            break
        except Exception:                                    # noqa: BLE001
            continue
    if fn is None:
        return {"pass": False, "computable": False, "reason": "taniteval.v8f_consistency could not be imported from the overlay tree"}
    r0 = rows_by_tick[0]
    ck = T["s_ckpt"][r0].astype(np.float64)
    idx = np.array([int(J["off"][int(T["ci"][row])]) + int(T["t0"][row]) for row in r0])
    ref = fn(ck, np.asarray(J["rc"][idx], np.float64), np.asarray(J["rc_ok"][idx], bool))
    pos, head = np.full((len(r0), 3), np.nan), np.full((len(r0), 3), np.nan)
    for i, row in enumerate(r0):
        c, a = int(T["ci"][row]), int(T["t0"][row])
        G, _ = ctx.grid(str(names[c]))
        sp = T["s_pose"][row].astype(np.float64)
        xy, psi = ckpt_world(sp, ck[i])
        pos[i], _, _, head[i] = row_cells(G, a, xy, psi, label_join(J, c, G, a))
    same_nan = bool((np.isnan(pos) == np.isnan(ref["pos_err_m"])).all() and (np.isnan(head) == np.isnan(ref["head_err_deg"])).all())
    v = ~np.isnan(pos) & ~np.isnan(ref["pos_err_m"])
    dpos = float(np.abs(pos - ref["pos_err_m"])[v].max()) if v.any() else float("nan")
    dh = np.abs(head - ref["head_err_deg"])
    dh = np.minimum(dh, 360.0 - dh)
    dhead = float(dh[v].max()) if v.any() else float("nan")
    ok = bool(same_nan and v.any() and dpos <= 1e-3 and dhead <= 1e-2)
    return {"pass": ok, "computable": True, "windows": int(len(r0)), "max_abs_diff_pos_m": dpos, "max_abs_diff_head_deg": dhead, "same_rows_invalid": same_nan,
            "n_valid_per_arc": v.sum(0).tolist(), "known_value_from": src, "note": "Amendment 1 of the card; differences only"}


def s3_frozen_check(ctx, T, rows_by_tick):
    """Control s3 as the card FIRST froze it (kept on record; it fails): the first-tick label of the rule the card cited (label_card) against the programme's own `rc`
    of the eval join (ego frame of the window origin). Returns dict."""
    J, names = ctx.J, T["clip_names"]
    if not (np.asarray(J["clip"]) == np.asarray(names)).all():
        return {"pass": None, "computable": False, "reason": "join clip order differs from the tick file's clip_names"}
    md, nval, nflag, nwin = np.zeros(3), np.zeros(3, int), 0, 0
    r0 = rows_by_tick[0]
    per_arc_flag = np.zeros(3, int)
    for row in r0:
        c, a = int(T["ci"][row]), int(T["t0"][row])
        G, _ = ctx.grid(str(names[c]))
        lpos, lyaw, _, lok = label_card(G, a)
        i = int(J["off"][c]) + a
        rc, rok = J["rc"][i].astype(np.float64), J["rc_ok"][i]
        cs, sn = math.cos(-G[a, 2]), math.sin(-G[a, 2])
        for m in range(3):
            if bool(lok[m]) != bool(rok[m]):
                nflag += 1
                per_arc_flag[m] += 1
                continue
            if not lok[m]:
                continue
            d = lpos[m] - G[a, :2]
            e = np.array([d[0] * cs - d[1] * sn, d[0] * sn + d[1] * cs])
            nval[m] += 1
            md[m] = max(md[m], float(np.abs(e - rc[m, :2]).max()), float(abs(wrap(lyaw[m] - G[a, 2] - rc[m, 2]))))
        nwin += 1
    ok = bool(md.max() <= 1e-3 and nflag == 0)
    return {"pass": ok, "computable": True, "windows": nwin, "max_diff_per_arc": md.tolist(), "n_valid_both_per_arc": nval.tolist(), "validity_disagreements": int(nflag),
            "validity_disagreements_per_arc": per_arc_flag.tolist(),
            "source_compared": "J['rc'] / J['rc_ok'] of the eval join file (the programme's own label, built from the v9 release by build_v9_labels.py: route smoothed in arc length, sigma 8 m)",
            "rule_mirrored": "v8f_mini_data.py lines 1213-1223 (cumulative arc over positions after the anchor; first step whose arc reaches A)"}


def s3b_smooth(ctx, T, rows_by_tick, tree):
    """Diagnostic only (not a card control): does the programme's SMOOTHED label (build_v9_labels.py smooth_path / rc_point, sigma 8 m, native track as the 10 Hz track) reproduce J['rc']?
    Sagitta validity is not reproduced (needs the raw log)."""
    try:
        sys.path.insert(0, os.path.join(tree, "stack"))
        sys.path.insert(0, os.path.join(tree, "stack", "scripts"))
        import build_v9_labels as BV
    except Exception as ex:                                   # noqa: BLE001
        return {"computed": False, "reason": f"import of build_v9_labels failed: {type(ex).__name__}: {ex}"[:200]}
    J, names = ctx.J, T["clip_names"]
    cache, md, nv, nflag = {}, np.zeros(3), np.zeros(3, int), np.zeros(3, int)
    for row in rows_by_tick[0]:
        c, a = int(T["ci"][row]), int(T["t0"][row])
        if c not in cache:
            pz = ctx.tracks[str(names[c])].copy()
            pz[:, 2] = np.unwrap(pz[:, 2])
            s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(pz[:, 0]), np.diff(pz[:, 1])))])
            tr10 = BV.Track10(np.arange(len(pz)) * 0.1, pz, s)
            cache[c] = (tr10, BV.smooth_path(tr10, 8.0))
        tr10, P = cache[c]
        k = 2 * a
        i = int(J["off"][c]) + a
        rc, rok = J["rc"][i].astype(np.float64), J["rc_ok"][i]
        for m, L in enumerate(ARCS):
            x, y, p, ok = BV.rc_point(P, tr10.s[k] + L, tr10.poses[k, 0], tr10.poses[k, 1], tr10.poses[k, 2])
            if ok != bool(rok[m]):
                nflag[m] += 1
            if ok and rok[m]:
                nv[m] += 1
                md[m] = max(md[m], abs(x - rc[m, 0]), abs(y - rc[m, 1]), abs(float(wrap(math.radians(p) - rc[m, 2]))))
    return {"computed": True, "max_diff_per_arc": md.tolist(), "n_valid_both_per_arc": nv.tolist(), "validity_disagreements_per_arc": nflag.tolist(),
            "note": "diagnostic; the card's s3 is the card-rule row above"}


def run_controls_a(ctx, a, runs, log):
    """s4 + s1 s2 s6 s9 (+ s3) over the real files. Returns (controls, loaded, rows)."""
    C = {}
    t = time.time()
    C["s4"] = run_s4()
    log(f"s4 synthetic known values: {'PASS' if C['s4']['pass'] else 'FAIL'}  (planted fault caught: {C['s4']['planted_fault_fails']})")
    inputs, avail, T_all = {}, {}, {}
    s1 = {"pass": True, "max_dxy_m": 0.0, "max_dpsi_rad": 0.0, "per_file": {}}
    s2 = {"pass": True, "max_first_tick_anchor_dist_m": 0.0, "n_rows_anchor_not_t0": 0, "max_first_tick_executed_minus_logged_pose_m": 0.0, "per_file": {}}
    s6 = {"pass": True, "per_file": {}}
    ref = None
    for run in runs:
        for k in SEEDS:
            p = tick_path(ctx.W, run, k)
            nm = f"{run}_s{k}"
            if not os.path.isfile(p):
                avail[nm] = False
                continue
            avail[nm] = True
            inputs[nm] = {"path": p, "sha256": sha256_file(p)}
            T = load_ticks(p)
            rbt = window_order(T)
            if ref is None:
                ref = {"ci": T["ci"][rbt[0]], "t0": T["t0"][rbt[0]]}
            ok6, det6 = structure(T, nm, ref)
            s6["per_file"][nm] = ok6
            s6["pass"] &= ok6
            dxy, dps = s1_row_check(T)
            s1["per_file"][nm] = [dxy, dps]
            s1["max_dxy_m"], s1["max_dpsi_rad"] = max(s1["max_dxy_m"], dxy), max(s1["max_dpsi_rad"], dps)
            an = anchors_for(ctx, T, rbt, ticks=(0,))
            st, ds = an[0]
            bad = int((st != T["t0"][rbt[0]]).sum())
            ex = 0.0
            for i, row in enumerate(rbt[0]):
                G, _ = ctx.grid(str(T["clip_names"][T["ci"][row]]))
                ex = max(ex, float(np.linalg.norm(T["pose"][row, :2] - G[int(T["t0"][row]), :2])))
            s2["per_file"][nm] = [float(ds.max()), bad]
            s2["max_first_tick_anchor_dist_m"] = max(s2["max_first_tick_anchor_dist_m"], float(ds.max()))
            s2["n_rows_anchor_not_t0"] += bad
            s2["max_first_tick_executed_minus_logged_pose_m"] = max(s2["max_first_tick_executed_minus_logged_pose_m"], ex)
            if nm == "R0_s0":
                T_ref = T
            T_all[nm] = None            # tick data is reloaded per use (memory)
            del T
    s1["pass"] = bool(s1["max_dxy_m"] <= 1e-3 and s1["max_dpsi_rad"] <= 1e-3 and len(s1["per_file"]) > 0)
    s2["pass"] = bool(s2["max_first_tick_anchor_dist_m"] <= 1e-6 and s2["n_rows_anchor_not_t0"] == 0 and len(s2["per_file"]) > 0)
    for kx, d in (("s1", s1), ("s2", s2), ("s6", s6)):
        C[kx] = d
    # s3 on R0 seed 0 (the windows are the same in every file by s6)
    if "R0_s0" in inputs:
        T0 = load_ticks(tick_path(ctx.W, "R0", 0))
        C["s3"] = s3_check(ctx, T0, window_order(T0))
        C["s3_as_first_frozen"] = s3_frozen_check(ctx, T0, window_order(T0))
        C["s3_as_first_frozen"]["on_record"] = "the control as the card first froze it FAILED (the card cited the synthetic-drive label rule); Amendment 1 replaced the label and this control"
        C["s3b_diagnostic"] = s3b_smooth(ctx, T0, window_order(T0), a.tree)
    else:
        C["s3"] = {"pass": False, "computable": False, "reason": "R0_s0 tick file absent"}
    C["s9"] = {"pass": len(inputs) > 0, "n_files": len(inputs), "tick_files_missing": [k for k, v in avail.items() if not v],
               "note": "sha256 of every tick file read recorded under inputs"}
    C["_elapsed_s"] = round(time.time() - t, 1)
    return C, inputs, avail


def print_controls_a(C, log):
    log(f"s1 frames (own transform vs the tool's s_ckpt_w0): {'PASS' if C['s1']['pass'] else 'FAIL'}  largest diff {C['s1']['max_dxy_m']:.3e} m / {C['s1']['max_dpsi_rad']:.3e} rad over {len(C['s1']['per_file'])} files")
    log(f"s2 anchor at the first tick: {'PASS' if C['s2']['pass'] else 'FAIL'}  largest distance {C['s2']['max_first_tick_anchor_dist_m']:.3e} m, rows with anchor != t0: {C['s2']['n_rows_anchor_not_t0']}; "
        f"executed minus logged pose at t0 largest {C['s2']['max_first_tick_executed_minus_logged_pose_m']:.3e} m")
    s3 = C["s3"]
    if s3.get("computable"):
        log(f"s3 (Amendment 1) first-tick pos / head vs the other session's checkpoint_errors: {'PASS' if s3['pass'] else 'FAIL'}  largest diff {s3['max_abs_diff_pos_m']:.3e} m / "
            f"{s3['max_abs_diff_head_deg']:.3e} deg, same rows invalid {s3['same_rows_invalid']}, n valid per arc {s3['n_valid_per_arc']}")
    else:
        log(f"s3 (Amendment 1): NOT COMPUTABLE (counts as FAIL): {s3.get('reason')}")
    s3 = C.get("s3_as_first_frozen", {})
    if s3.get("computable"):
        log(f"s3 as first frozen (on record, replaced by Amendment 1): {'PASS' if s3['pass'] else 'FAIL'}  largest diff per arc {[round(x, 4) for x in s3['max_diff_per_arc']]} (n valid both {s3['n_valid_both_per_arc']}), "
            f"validity disagreements {s3['validity_disagreements']} {s3['validity_disagreements_per_arc']}")
    b = C.get("s3b_diagnostic")
    if b:
        log("s3b (diagnostic, not a card control) programme's smoothed label from the native track: " + (
            f"largest diff per arc {[round(x, 4) for x in b['max_diff_per_arc']]} (n {b['n_valid_both_per_arc']}), validity disagreements {b['validity_disagreements_per_arc']}" if b.get("computed") else b.get("reason", "")))
    log(f"s6 structure (7,197 rows / 2,399 windows / ticks, steps, ages / 141 episodes) every file: {'PASS' if C['s6']['pass'] else 'FAIL'}  ({sum(C['s6']['per_file'].values())}/{len(C['s6']['per_file'])} files)")
    log(f"s9 inputs: {C['s9']['n_files']} tick files hashed; missing: {C['s9']['tick_files_missing']}")


# ------------------------------------------------------------------------------------------------ cells (outcomes) -- default mode only
def compute_cells(ctx, T):
    """pos/lat/along/head [3 ticks, N windows, 3 arcs] (NaN invalid) and the anchor (step, distance) [3, N]."""
    rbt = window_order(T)
    names = T["clip_names"]
    if not (np.asarray(ctx.J["clip"]) == np.asarray(names)).all():
        raise SystemExit("STOP: the clip order of the evaluation join differs from the tick file's clip_names")
    cells = {c: np.full((3, N_WIN, 3), np.nan) for c in CELLS}
    anc_st, anc_ds = np.zeros((3, N_WIN), np.int64), np.zeros((3, N_WIN))
    for k in range(3):
        for i, row in enumerate(rbt[k]):
            ci = int(T["ci"][row])
            G, G32 = ctx.grid(str(names[ci]))
            c = int(T["j"][row]) - int(T["age_s"][row])
            sp = T["s_pose"][row].astype(np.float64)
            a, d = find_anchor(G32, c, sp[:2])
            xy, psi = ckpt_world(sp, T["s_ckpt"][row].astype(np.float64))
            pos, lat, along, head = row_cells(G, a, xy, psi, label_join(ctx.J, ci, G, a))      # Amendment 1: the programme's own label of the anchor step
            cells["pos"][k, i], cells["lat"][k, i], cells["along"][k, i], cells["head"][k, i] = pos, lat, along, head
            anc_st[k, i], anc_ds[k, i] = a, d
    return cells, anc_st, anc_ds


def neighbour_index(t0, eid, d):
    """index of window w+d if it is in the same episode and its start is 4*d steps later, else -1."""
    n = len(t0)
    idx = np.full(n, -1, np.int64)
    w = np.arange(n - d)
    ok = (eid[w] == eid[w + d]) & (t0[w + d] - t0[w] == 4 * d)
    idx[w[ok]] = w[ok] + d
    return idx


def roll_diff(cell_k, cell_first, t0, eid, tick):
    """per window: cell at tick minus the mean of the first-tick cells of the two bracketing windows; valid only if all three are valid. [N, 3] with NaN."""
    d1, d2 = (1, 2) if tick == 1 else (4, 5)
    i1, i2 = neighbour_index(t0, eid, d1), neighbour_index(t0, eid, d2)
    out = np.full(cell_k.shape, np.nan)
    ok = (i1 >= 0) & (i2 >= 0)
    w = np.nonzero(ok)[0]
    ref = 0.5 * (cell_first[i1[w]] + cell_first[i2[w]])
    out[w] = cell_k[w] - ref
    return out


def pair_diffs(cA, cB, k, cell):
    """[N, 3] difference A - B at tick k on windows valid in both."""
    return cA[cell][k] - cB[cell][k]


def boot_cols(per_seed_arr, eid, B=B_BOOT):
    """per_seed_arr: list over seeds of [N, 3] diffs (NaN invalid) -> list over arcs of boot dict."""
    res = []
    for m in range(3):
        vals, eids = [], []
        for arr in per_seed_arr:
            v = arr[:, m]
            ok = np.isfinite(v)
            vals.append(v[ok])
            eids.append(eid[ok])
        res.append(boot(vals, eids, B))
    return res


def pair_read(cells, A, Bn, eid, ticks=(1, 2)):
    out = {}
    for k in ticks:
        out[f"tick{k}"] = {}
        for cell in CELLS + ("abs_along",):
            arrs = []
            for s in SEEDS:
                ca, cb = cells[(A, s)], cells[(Bn, s)]
                if cell == "abs_along":
                    arrs.append(np.abs(ca["along"][k]) - np.abs(cb["along"][k]))
                else:
                    arrs.append(ca[cell][k] - cb[cell][k])
            out[f"tick{k}"][cell] = boot_cols(arrs, eid)
    return out


def level_table(cells, anc_ds, runs):
    lv = {}
    for run in runs:
        for s in SEEDS:
            c = cells[(run, s)]
            ent = {}
            for k in range(3):
                for m, A in enumerate(ARCS):
                    for cell in CELLS:
                        v = c[cell][k][:, m]
                        ok = np.isfinite(v)
                        ent[f"tick{k}_{int(A)}m_{cell}"] = {"mean": float(v[ok].mean()) if ok.any() else None, "n_valid": int(ok.sum())}
            ent["anchor_distance_m"] = {f"tick{k}": {"mean": float(anc_ds[(run, s)][k].mean()), "median": float(np.median(anc_ds[(run, s)][k])),
                                                      "max": float(anc_ds[(run, s)][k].max())} for k in (1, 2)}
            lv[f"{run}_s{s}"] = ent
    return lv


def fam_name(f, kind):
    return {"better": {"BETTER": "BETTER-THAN-PINNED", "WORSE": "WORSE-THAN-PINNED", "NOT-SEPARATED": "NOT-SEPARATED-FROM-PINNED", "MIXED": "MIXED"},
            "cand": {"BETTER": "BETTER-THAN-CONVENTION", "WORSE": "WORSE-THAN-CONVENTION", "NOT-SEPARATED": "NOT-SEPARATED-FROM-CONVENTION", "MIXED": "MIXED"},
            "rule": {"BETTER": "BETTER", "WORSE": "WORSE", "NOT-SEPARATED": "NOT-SEPARATED", "MIXED": "MIXED"}}[kind][f]


def fam_of(pr, tick=2):
    return family([c["verdict"] for c in pr[f"tick{tick}"]["pos"]])


def run_s8():
    rng = np.random.default_rng(7)
    v, e = rng.normal(size=300), rng.integers(0, 30, 300)
    a = boot([v, v], [e, e])
    z = boot([v - v, v - v], [e, e])
    sw = boot([-(v + 1.0), -(v + 1.0)], [e, e])
    sw0 = boot([(v + 1.0), (v + 1.0)], [e, e])
    ok = bool(z["mean"] == 0.0 and z["lo"] == 0.0 and z["hi"] == 0.0 and z["verdict"] == "NOT-SEPARATED" and sw["mean"] < 0 and sw0["mean"] > 0 and sw["verdict"] == "BETTER" and sw0["verdict"] == "WORSE")
    return {"pass": ok, "self_pair": {k: z[k] for k in ("mean", "lo", "hi", "verdict")}, "swapped_read_mean": sw["mean"], "forward_read_mean": sw0["mean"]}


def full_read(ctx, a, log):
    t_all = time.time()
    runs = list(BASE_RUNS)
    cand_note = {"requested": bool(a.with_cand), "admitted": False, "reason": "flag --with-cand not given"}
    if a.with_cand:
        try:
            cj = json.load(open(a.cand_json, encoding="utf-8"))
            if cj.get("controls_all_pass") is True:
                cand_note.update(admitted=True, reason="READ_CAND10K.json controls_all_pass is true")
                runs += list(CAND_RUNS)
            else:
                cand_note["reason"] = f"{a.cand_json}: controls_all_pass is not true"
        except Exception as ex:                                  # noqa: BLE001
            cand_note["reason"] = f"{a.cand_json} unreadable or absent: {type(ex).__name__}"
    C, inputs, avail = run_controls_a(ctx, a, runs, log)
    print_controls_a(C, log)
    a_keys = ("s1", "s2", "s4", "s6", "s9")
    s3 = C["s3"]
    a_ok = all(C[k]["pass"] for k in a_keys) and (s3["pass"] is True)          # Amendment 1: s3 must pass; NOT COMPUTABLE counts as a fail
    if cand_note["admitted"] and not all(avail.get(f"{r}_s{k}") for r in CAND_RUNS for k in SEEDS):
        cand_note.update(admitted=False, reason="a tick file of a card ADD65 run is missing")
        runs = list(BASE_RUNS)
    res = {"card": CARD, "tier": TIER, "controls": C, "controls_all_pass": False, "names": [], "levels": None, "pairs": None, "roll": None, "reported_no_bar": {},
           "inputs": inputs, "candidate": cand_note, "strategic_decision_between_routes": "n = 0 (no window of this evaluation set carries a label of a route choice; not scored)"}
    if not a_ok:
        log("CONTROLS s1 s2 s3 s4 s6 s9 DID NOT ALL PASS: no outcome cell computed, nothing named")
        res["stopped"] = "stage-A controls failed; no cell computed"
        return res
    missing = [f"{r}_s{k}" for r in BASE_RUNS for k in SEEDS if not avail.get(f"{r}_s{k}")]
    if missing:
        res["stopped"] = f"base tick files missing: {missing}"
        log(res["stopped"])
        return res
    # stage B: cells
    cells, anc_ds, T0 = {}, {}, None
    for run in runs:
        for s in SEEDS:
            T = load_ticks(tick_path(ctx.W, run, s))
            if T0 is None:
                T0 = T
            c, st, ds = compute_cells(ctx, T)
            cells[(run, s)], anc_ds[(run, s)] = c, ds
            log(f"cells {run}_s{s} done")
    rbt = window_order(T0)
    t0 = T0["t0"][rbt[0]]
    eid, n_map = episode_map(ctx.W, T0)
    s6_extra = bool((eid >= 0).all() and len(np.unique(eid)) == N_EP)
    C["s6"]["episode_map"] = {"dump_windows": n_map, "all_windows_mapped": bool((eid >= 0).all()), "n_episodes": int(len(np.unique(eid)))}
    C["s6"]["pass"] = bool(C["s6"]["pass"] and s6_extra)
    # s5: first tick equals R0's
    d5 = 0.0
    for run in runs:
        for s in SEEDS:
            for cell in CELLS:
                x, y = cells[(run, s)][cell][0], cells[("R0", s)][cell][0]
                if not (np.isnan(x) == np.isnan(y)).all():
                    d5 = float("inf")
                else:
                    ok = np.isfinite(x)
                    d5 = max(d5, float(np.abs(x[ok] - y[ok]).max()) if ok.any() else 0.0)
    C["s5"] = {"pass": bool(d5 == 0.0), "largest_difference": d5}
    # s7: reference bias on R0 first-tick pos (windows w, w+2 against w+1)
    i1, i2 = neighbour_index(t0, eid, 1), neighbour_index(t0, eid, 2)
    s7 = {"per_arc": []}
    ok7 = True
    for m, A in enumerate(ARCS):
        vals, eids, lvl = [], [], []
        for s in SEEDS:
            p = cells[("R0", s)]["pos"][0][:, m]
            w = np.nonzero((i1 >= 0) & (i2 >= 0))[0]
            d = 0.5 * (p[w] + p[i2[w]]) - p[i1[w]]
            g = np.isfinite(d) & np.isfinite(p[w])
            vals.append(d[g]); eids.append(eid[w][g]); lvl.append(p[w][g])
        bt = boot(vals, eids)
        level = float(np.concatenate(lvl).mean())
        cond = bool((bt["lo"] <= 0.0 <= bt["hi"]) or abs(bt["mean"]) <= 0.02 * level)
        ok7 &= cond
        s7["per_arc"].append({"arc_m": A, "mean": bt["mean"], "lo": bt["lo"], "hi": bt["hi"], "first_tick_level": level, "share_of_level": bt["mean"] / level, "ok": cond, "n": bt["n"]})
    s7["pass"] = bool(ok7)
    C["s7"] = s7
    C["s8"] = run_s8()
    for k in ("s5", "s7", "s8"):
        log(f"{k}: {'PASS' if C[k]['pass'] else 'FAIL'}")
    all_pass = bool(a_ok and C["s5"]["pass"] and C["s8"]["pass"] and C["s6"]["pass"])
    res["controls_all_pass"] = all_pass
    res["controls_all_pass_note"] = "s1 s2 s3 (as Amendment 1 wrote it) s4 s5 s6 s8 s9; s7 gates ROLL only; s3 as first frozen is kept on record as failed"
    if not all_pass:
        log("CONTROLS DID NOT ALL PASS: nothing named; outcome cells stay unread")
        res["stopped"] = "controls failed at stage B; cells computed for the controls only are not reported"
        return res
    # reads
    roll = {}
    for run in runs:
        roll[run] = {}
        for tick in (1, 2):
            arrs, firsts = [], []
            for s in SEEDS:
                c = cells[(run, s)]
                d = roll_diff(c["pos"][tick], c["pos"][0], t0, eid, tick)
                arrs.append(d)
                firsts.append(np.where(np.isfinite(d), c["pos"][0], np.nan))
            bt = boot_cols(arrs, eid)
            lvl = [float(np.nanmean(np.concatenate([f[:, m] for f in firsts]))) for m in range(3)]
            roll[run][f"ROLL{tick}"] = [dict(b, first_tick_level_same_windows=lv, share_of_first_tick_level=(b["mean"] / lv if b["mean"] is not None else None), arc_m=A)
                                        for b, lv, A in zip(bt, lvl, ARCS)]
    res["roll"] = roll
    names = []
    if C["s7"]["pass"]:
        r1, r2 = [c["verdict"] for c in roll["R0"]["ROLL1"]], [c["verdict"] for c in roll["R0"]["ROLL2"]]
        if sum(v == "WORSE" for v in r2) >= 2:
            names.append("STRATROLL10K-STRATEGIC-DEGRADES-ON-ROLLOUT")
        elif not any(v == "WORSE" for v in r1 + r2):
            names.append("STRATROLL10K-STRATEGIC-HOLDS-ON-ROLLOUT")
        else:
            names.append("STRATROLL10K-STRATEGIC-MIXED-ON-ROLLOUT")
    else:
        res["roll_not_named_reason"] = "control s7 failed (reference bias)"
    PAIRS = [("R0G0X", "R0"), ("R0C10", "R0"), ("R0G0", "R0"), ("R0Bk10", "R0"), ("RLc", "R0"), ("RQ50", "R0"), ("RLcF5", "RLc"), ("RQ50F5", "RQ50")]
    if cand_note["admitted"]:
        PAIRS += [("RLcKF5", "RLc"), ("RQ50KF5", "RQ50"), ("RLcK", "RLc"), ("RQ50K", "RQ50"), ("RLcKF5", "RLcK"), ("RQ50KF5", "RQ50K")]
    pairs = {f"{A}-{B}": pair_read(cells, A, B, eid) for A, B in PAIRS}
    res["pairs"] = pairs
    names.append("STRATROLL10K-K-STRATEGIC-" + fam_name(fam_of(pairs["R0G0X-R0"]), "better"))
    names.append("STRATROLL10K-CAUSAL-STRATEGIC-" + fam_name(fam_of(pairs["R0C10-R0"]), "better"))
    for form, run in FORMS:
        names.append(f"STRATROLL10K-{form}-RULE-STRATEGIC-" + fam_name(fam_of(pairs[f"{run}-R0"]), "rule"))
    if cand_note["admitted"]:
        for form, run in FORMS:
            names.append(f"STRATROLL10K-{form}-CANDIDATE-STRATEGIC-" + fam_name(fam_of(pairs[f"{run}KF5-{run}"]), "cand"))
    res["names"] = names
    res["levels"] = level_table(cells, anc_ds, runs)
    res["reported_no_bar"] = {"levels_per_run_tick_arc_seed": "see levels", "anchor_distance_second_third_tick": "see levels[...]['anchor_distance_m']",
                              "windows_entering_roll": {run: {f"ROLL{t}": [b["n_per_seed"] for b in roll[run][f"ROLL{t}"]] for t in (1, 2)} for run in runs}}
    # print
    log("NAMES")
    for n in names:
        log("  " + n)
    if "roll_not_named_reason" in res:
        log("  ROLL not named: " + res["roll_not_named_reason"])
    if not cand_note["admitted"]:
        log("  CANDIDATE names not given: " + cand_note["reason"])
    log("pos cells (m): pair, tick, arc: diff [lo, hi] verdict | per-seed")
    for pn, pr in pairs.items():
        for tk in ("tick1", "tick2"):
            row = pr[tk]["pos"]
            log(f"  {pn:16s} {tk} " + " ; ".join(f"{int(A)}m {b['mean']:+.4f} [{b['lo']:+.4f},{b['hi']:+.4f}] {b['verdict']}" for A, b in zip(ARCS, row)))
    log("ROLL (R0), pos, m: ")
    for t in (1, 2):
        log(f"  ROLL{t} " + " ; ".join(f"{int(b['arc_m'])}m {b['mean']:+.4f} [{b['lo']:+.4f},{b['hi']:+.4f}] {b['verdict']} ({b['share_of_first_tick_level']*100:+.1f}% of level)" for b in roll["R0"][f"ROLL{t}"]))
    res["elapsed_s"] = round(time.time() - t_all, 1)
    return res


# ------------------------------------------------------------------------------------------------ main
def to_json(o):
    if isinstance(o, dict):
        return {str(k): to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_json(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return to_json(o.tolist())
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--with-cand", action="store_true")
    ap.add_argument("--w", default=DEF_W)
    ap.add_argument("--tracks", default=DEF_TRACKS)
    ap.add_argument("--join", default=DEF_JOIN)
    ap.add_argument("--tree", default=os.path.join(DEF_W, "tree"))
    ap.add_argument("--cand-json", default=DEF_CAND_JSON)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    def log(s):
        print(s, flush=True)
    if a.selftest:
        r = run_s4()
        for c in r["checks"]:
            log(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['name']}")
        log(f"  planted fault (label taken one arc too far) fails: {r['planted_fault_fails']} (checks failed: {len(r['planted_fault_checks_failed'])})")
        log(f"ZZ STRATROLL-SELFTEST all hold: {bool(r['pass'])}")
        return 0 if r["pass"] else 1
    ctx = Ctx(a)
    ctx.load_static()
    if a.controls_only:
        t = time.time()
        C, inputs, avail = run_controls_a(ctx, a, list(BASE_RUNS) + (list(CAND_RUNS) if a.with_cand else []), log)
        print_controls_a(C, log)
        ok = all(C[k]["pass"] for k in ("s1", "s2", "s4", "s6", "s9")) and (C["s3"].get("pass") is True)
        log("files hashed: " + ", ".join(f"{k}={v['sha256'][:12]}" for k, v in sorted(inputs.items())))
        log(f"elapsed {time.time() - t:.1f} s")
        log(f"ZZ STRATROLL-CONTROLS-ONLY s1 s2 s3 s4 s6 s9 all pass: {bool(ok)}")
        return 0
    res = full_read(ctx, a, log)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "READ_STRATROLL10K.json")
    if os.path.exists(p):
        log("REFUSED to overwrite " + p)
        return 2
    with open(p, "w", encoding="utf-8") as f:
        json.dump(to_json(res), f, indent=1)
    log(f"wrote {p}")
    log(f"ZZ STRATROLL-READ DONE controls_all_pass: {res['controls_all_pass']}  names: {len(res['names'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
