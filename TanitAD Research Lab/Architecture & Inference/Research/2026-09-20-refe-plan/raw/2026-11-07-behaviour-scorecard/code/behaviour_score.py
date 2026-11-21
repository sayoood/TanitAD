#!/usr/bin/env python
"""behaviour_score.py -- per-scenario metric code for BEHAVIOUR_SCORECARD.md (sha256 fac6c7f1...), stream BEHAV-CODE.

One rollout.npz (+ the human reference .npz) in, one JSON out: lead/gap/time-gap/TTC derived here, every
rule a1 a2 b1 b2 c1 c2 c3 c4 d1 d2 e1 e2 f1 f2 scored PASS / FAIL / INFORMATIONAL / NA, EPDMS (g) passed
through. Schema: ROLLOUT_SCHEMA.md. Open definitional questions: AMBIGUITIES.md (each is flagged "A-n" below).

CLI:  python behaviour_score.py score --rollout R.npz --human H.npz --out OUT.json [--dump-ticks T.npz]

Lines ending in "# MUT:<rule>" are the single line that decides each rule's verdict; mutation_check.py
neuters exactly that line in a COPY and requires the rule's deliberate-regression test to go RED.
CPU only, numpy + scipy (+ shapely for f1's vendored NAVSIM classifier), no randomness.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

SCORECARD_SHA256 = "fac6c7f150eb026de0708c6f6557df52481216f76b35f3c8c8b8045bcb8ec0c5"
SCHEMA_VERSION = "refe-rollout-v1"
TIER_STAMP = "T2 (re-perception closed-loop sim; MM ruling 2026-11-07)"      # AMENDMENT_1.md "Tier stamp"
EPS = 1e-9

CATEGORIES = ("cruise", "steady_lead", "lead_braking", "stop_go", "merge_cutin", "junction", "traffic_light")
RULES = ("a1", "a2", "b1", "b2", "c1", "c2", "c3", "c4", "d1", "d2", "e1", "e2", "f1", "f2")
FAMILIES = {
    "LONGITUDINAL": ("a1", "a2", "b1", "b2", "d1", "d2", "c4"),
    "LATERAL": ("e1", "e2", "d1"),
    "TACTICAL": ("c1", "c2", "c3", "c4"),
}
SLOTS = ("J-UL", "J-R", "J-S", "TL-G", "TL-S")     # AMENDMENT_3 CH-4: optional descriptive per-slot label (category pools J-UL, J-R, J-S)
SIGNAL_RED, SIGNAL_YELLOW, SIGNAL_GREEN = 1, 2, 3

# Every number is the scorecard's, except the ones marked "A-n" (see AMBIGUITIES.md) which are mine.
P = dict(
    DT=0.1, DT_IN=0.05, WARMUP_S=2.0,                          # section 1
    CORRIDOR_HALF_WIDTH=1.5, VREL_MIN=0.1, TG_VMIN=3.0,        # section 1
    FREE_HEADWAY_S=4.0, FREE_MIN_GAP_M=15.0, FREE_TTC_S=8.0, FREE_MIN_INTERVAL_S=2.0,
    PERSIST_TICKS=3,
    A1_LO=0.90, A1_HI=1.10, A1_SHARE=0.90, A1_VH_FLOOR=3.0,
    A2_WINDOW_S=4.0, A2_RATIO=0.85, A2_V_MIN=0.1,              # A2_V_MIN: A-9
    B1_STEADY_S=3.0, B1_ALEAD_MAX=1.0, B1_LO=1.5, B1_HI=2.5, B2_TG=1.0,
    C1_MINV=1.0, C1_APPROACH_M=40.0, C1_MODE="clear_persistent",  # C1_MODE: A-5 ("literal" = run to the crossing)
    C2_STOP_CAP_M=6.0, STOPPED_V=0.1, C4_STOP_V=1.0, C3_WITHIN_S=2.0,   # STOPPED_V, C4_STOP_V: A-7, A-8
    C4_LAUNCH_V=1.0, C4_MAX_S=3.0, C4_HUMAN_PLUS_S=1.5, C4_LEAD_PULL_V=2.0, STOP_MIN_S=1.0,  # A-8
    D1_WINDOW_S=4.0, D1_STRIDE_S=1.0, D1_SHARE=0.95, D1_HUMAN_MARGIN=0.05,
    D2_SMOOTH_S=0.5, D2_TURN=0.5, D2_SWING=1.0, D2_WITHIN_S=6.0,
    E1_CT=0.5, E1_RUN_S=2.0, E2_HEAD=0.087, E2_KAPPA=0.02, E2_V_MIN=1.0,   # E2_V_MIN: A-12
    F2_TTC=1.0, F2_V_STOPPED=0.005,
    PATH_KNOT_EPS=0.05, PATH_EXT_M=150.0, PATH_EXT_V_MIN=0.5, PATH_EXT_CHORD_M=5.0, ALEAD_HALF_S=0.5,    # PATH_*: A-6; ALEAD_HALF_S: A-3
    LAT_FAULT_CT=0.5, REAR_AXLE_TO_CENTER=1.461,               # A-10: lane-less stand-in for "ego in multiple lanes"; nuPlan Pacifica rear axle -> box centre
)


def nt(sec):
    """seconds -> ticks at the 10 Hz metric grid"""
    return int(round(sec / P["DT"]))


# ----------------------------------------------------------------------------------------------------------
# vendored NAVSIM / nuPlan functions (called as is, sha-pinned): ego_is_comfortable (d1); collision classifier (f1, A-10)
# ----------------------------------------------------------------------------------------------------------
_VENDOR_DIR = Path(os.environ.get("BEHAV_VENDOR_DIR", str(Path(__file__).resolve().parent)))
_VENDOR_SHA = {
    "navsim_pdm/pdm_comfort_metrics.py": "f8e64a9be7881ec7aabf751a882b2ee24bc503c2106e635eec896c58d6f4a010",
    "navsim_pdm/pdm_enums.py": "827e7338b8e553684f87a0fc27a16cff3e3ee382e2a18271145e7e549a6c52bc",
    "navsim_pdm/pdm_collision.py": "85f7e1c9d5169928b86030abeb1958537e38d9c3808dd5b727f0baf5fafb02db",
    "navsim_pdm/nuplan_shim.py": "82da81821b4becec867f80b5f7c25b6e647bd30e27605d11af17ff4080a8336c",
}
_NAVSIM = None


def _navsim():
    global _NAVSIM
    if _NAVSIM is None:
        for rel, want in _VENDOR_SHA.items():
            got = hashlib.sha256((_VENDOR_DIR / rel).read_bytes()).hexdigest()
            if got != want:
                raise RuntimeError(f"vendored NAVSIM file {rel} sha256 {got} != pinned {want}")
        sys.path.insert(0, str(_VENDOR_DIR))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from navsim_pdm import pdm_comfort_metrics as m
            from navsim_pdm.pdm_enums import StateIndex
        _NAVSIM = (m, StateIndex)
    return _NAVSIM


def _collision_mod():
    """the vendored nuPlan/NAVSIM collision classifier (A-10); the sha check in _navsim() has already covered pdm_collision.py and nuplan_shim.py"""
    _navsim()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from navsim_pdm import nuplan_shim, pdm_collision
    return pdm_collision, nuplan_shim


# ----------------------------------------------------------------------------------------------------------
# loading / schema validation / resampling to the 10 Hz metric grid
# ----------------------------------------------------------------------------------------------------------
class SchemaError(ValueError):
    pass


EGO_KEYS = ("t", "ego_x", "ego_y", "ego_yaw", "ego_v", "ego_a", "ego_steer")
AGENT_KEYS = ("agent_id", "agent_present", "agent_x", "agent_y", "agent_yaw", "agent_v", "agent_l", "agent_w", "agent_h")


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _str(d, k, default=None):
    if k not in d:
        return default
    return str(np.asarray(d[k]).reshape(-1)[0])


def validate(d, kind):
    """Return a list of problems (empty = ok). kind in {'rollout','human'}."""
    pr = []
    for k in EGO_KEYS:
        if k not in d:
            pr.append(f"missing key {k}")
    if pr:
        return pr
    t = np.asarray(d["t"])
    T = t.shape[0] if t.ndim == 1 else -1
    if T < 3:
        pr.append("t must be 1-D with >= 3 samples")
        return pr
    for k in EGO_KEYS[1:]:
        a = np.asarray(d[k])
        if a.shape != (T,):
            pr.append(f"{k} shape {a.shape} != ({T},)")
        elif not np.all(np.isfinite(a.astype(float))):
            pr.append(f"{k} has non-finite values")
    dt = np.diff(t.astype(float))
    if not np.allclose(dt, P["DT_IN"], atol=1e-6):
        pr.append(f"t must be uniform at {P['DT_IN']} s (20 Hz); got dt in [{dt.min():.6f}, {dt.max():.6f}]")
    if kind == "rollout":
        for k in ("scenario_id", "category", "synthetic"):
            if k not in d:
                pr.append(f"missing key {k}")
        if "category" in d and _str(d, "category") not in CATEGORIES:
            pr.append(f"category {_str(d, 'category')!r} not in {CATEGORIES}")
        if "human_ref_valid_until_s" in d:
            hv = np.asarray(d["human_ref_valid_until_s"], float).reshape(-1)
            if hv.size != 1 or not (np.isfinite(hv[0]) and hv[0] >= 0):
                pr.append("human_ref_valid_until_s must be one finite value >= 0 (seconds from the first tick)")
        if "slot" in d and _str(d, "slot") not in SLOTS:
            pr.append(f"slot {_str(d, 'slot')!r} not in {SLOTS}")
    if "agent_id" in d:
        for k in AGENT_KEYS:
            if k not in d:
                pr.append(f"agent block incomplete: missing {k}")
        if not pr:
            A = np.asarray(d["agent_id"]).shape[0]
            for k in ("agent_present", "agent_x", "agent_y", "agent_yaw", "agent_v"):
                if np.asarray(d[k]).shape != (A, T):
                    pr.append(f"{k} shape {np.asarray(d[k]).shape} != ({A},{T})")
            for k in ("agent_l", "agent_w", "agent_h"):
                if np.asarray(d[k]).shape != (A,):
                    pr.append(f"{k} shape {np.asarray(d[k]).shape} != ({A},)")
            if len(set(np.asarray(d["agent_id"]).tolist())) != A:
                pr.append("agent_id not unique")
    return pr


def _grid(t):
    n = int(np.floor((t[-1] - t[0]) / P["DT"] + 1e-9)) + 1
    return t[0] + P["DT"] * np.arange(n)


def _zoh_idx(tg, t):
    return np.clip(np.searchsorted(t, tg + 1e-9, side="right") - 1, 0, len(t) - 1)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _resample_ts(tg, t, x):  # (A,T) -> (A,n): linear where both neighbours present, else ZOH value
    i0 = _zoh_idx(tg, t)
    i1 = np.minimum(i0 + 1, len(t) - 1)
    fr = np.where(i1 > i0, (tg - t[i0]) / np.maximum(t[i1] - t[i0], 1e-12), 0.0)
    a0, a1 = x[:, i0], x[:, i1]
    ok = np.isfinite(a0) & np.isfinite(a1) & (fr[None] > 1e-12)
    return np.where(ok, a0 + fr[None] * (a1 - a0), a0)


def build_run(d, fallback=None):
    """Resample a rollout/human dict to the 10 Hz grid. `fallback` supplies agents/scenario fields if absent."""
    fb = fallback or {}
    t = np.asarray(d["t"], float)
    tg = _grid(t)
    n = len(tg)
    R = {"t": tg, "n": n, "t0": float(tg[0])}
    R["x"] = np.interp(tg, t, np.asarray(d["ego_x"], float))
    R["y"] = np.interp(tg, t, np.asarray(d["ego_y"], float))
    R["yaw"] = np.interp(tg, t, np.unwrap(np.asarray(d["ego_yaw"], float)))
    R["v"] = np.interp(tg, t, np.asarray(d["ego_v"], float))
    R["a"] = np.interp(tg, t, np.asarray(d["ego_a"], float))
    R["steer"] = np.interp(tg, t, np.asarray(d["ego_steer"], float))
    src = d if "ego_length" in d else fb
    R["L"] = float(np.asarray(src.get("ego_length", 5.176)).reshape(-1)[0])
    R["W"] = float(np.asarray(src.get("ego_width", 2.297)).reshape(-1)[0])
    R["RA"] = float(np.asarray(d.get("ego_rear_axle_to_center", fb.get("ego_rear_axle_to_center", P["REAR_AXLE_TO_CENTER"]))).reshape(-1)[0])
    ad = d if "agent_id" in d else fb
    if "agent_id" in ad:
        ids = np.asarray(ad["agent_id"]).astype(np.int64)
        at = np.asarray(ad["t"], float) if ad is not d else t
        order = np.argsort(ids, kind="stable")
        pres = np.asarray(ad["agent_present"]).astype(bool)[order]
        i0 = _zoh_idx(tg, at)
        R["ids"] = ids[order]
        R["present"] = pres[:, i0]
        mk = lambda k: np.asarray(ad[k], float)[order]
        R["ax"] = _resample_ts(tg, at, mk("agent_x"))
        R["ay"] = _resample_ts(tg, at, mk("agent_y"))
        R["av"] = _resample_ts(tg, at, mk("agent_v"))
        yw = mk("agent_yaw")
        i1 = np.minimum(i0 + 1, len(at) - 1)
        fr = np.where(i1 > i0, (tg - at[i0]) / np.maximum(at[i1] - at[i0], 1e-12), 0.0)
        y0, y1 = yw[:, i0], yw[:, i1]
        R["ayaw"] = np.where(np.isfinite(y0) & np.isfinite(y1), y0 + fr[None] * _wrap(y1 - y0), y0)
        R["al"] = np.asarray(ad["agent_l"], float)[order]
        R["aw"] = np.asarray(ad["agent_w"], float)[order]
        cls = ad.get("agent_cls")
        R["acls"] = (np.asarray(cls).astype(np.int64)[order] if cls is not None else np.zeros(len(ids), np.int64))
    else:
        z = np.zeros((0, n))
        R.update(ids=np.zeros(0, np.int64), present=z.astype(bool), ax=z, ay=z, av=z, ayaw=z,
                 al=np.zeros(0), aw=np.zeros(0), acls=np.zeros(0, np.int64))
    sd = d if "signal_state" in d else fb
    if "signal_state" in sd and np.asarray(sd["signal_state"]).size:
        st = np.asarray(sd["signal_state"]).astype(np.int64)
        ts = np.asarray(sd["t"], float) if sd is not d else t
        R["sig"] = st[_zoh_idx(tg, ts)]
        R["gov"] = np.asarray(sd["signal_governs_ego"]).astype(bool) if "signal_governs_ego" in sd else np.ones(st.shape[1], bool)
    else:
        R["sig"], R["gov"] = np.zeros((n, 0), np.int64), np.zeros(0, bool)
    sl = d.get("stop_line_s", fb.get("stop_line_s", np.nan))
    hv = d.get("human_ref_valid_until_s", fb.get("human_ref_valid_until_s"))   # AMENDMENT_5 item 2: float64 scalar from rollout.npz, seconds from the first tick; absent = every tick valid
    R["ref_until"] = float(np.asarray(hv).reshape(-1)[0]) if hv is not None and np.asarray(hv).size else math.inf
    R["stop_s"] = float(np.asarray(sl).reshape(-1)[0]) if np.asarray(sl).size else float("nan")
    zn = d.get("conflict_zone_xy", fb.get("conflict_zone_xy", np.zeros((0, 2))))
    zn = np.asarray(zn, float).reshape(-1, 2)
    R["zone"] = zn[np.all(np.isfinite(zn), axis=1)]
    ci = d.get("conflict_agent_ids", fb.get("conflict_agent_ids", np.zeros(0, np.int64)))
    R["conf_ids"] = np.asarray(ci).astype(np.int64).reshape(-1)
    if "call_t" in d and np.asarray(d["call_t"]).size:
        ct = np.asarray(d["call_t"], float)
        j = _zoh_idx(tg, ct)
        j = np.where(tg + 1e-9 < ct[0], -1, j)  # no call yet
        v4 = np.asarray(d["call_plan_v4"], float)
        v0 = np.asarray(d["call_plan_v0"], float)
        ratio = np.where(v0 > P["A2_V_MIN"], v4 / np.where(v0 > P["A2_V_MIN"], v0, 1.0), np.nan)
        R["plan_ratio"] = np.where(j >= 0, ratio[np.maximum(j, 0)], np.nan)
    else:
        R["plan_ratio"] = None
    return R


# ----------------------------------------------------------------------------------------------------------
# the human path (reference corridor, v_h(s), heading and curvature references)
# ----------------------------------------------------------------------------------------------------------
def _moving_chord_dir(x, y, v):
    """AMENDMENT_6 ruling 2: unit direction of the chord over the last PATH_EXT_CHORD_M of arc length of the path's moving part
    (samples with v >= PATH_EXT_V_MIN, in order; all of it if shorter). None when the moving part has no length."""
    m = np.asarray(v) >= P["PATH_EXT_V_MIN"]
    mx, my = np.asarray(x)[m], np.asarray(y)[m]
    if len(mx) < 2:
        return None
    a = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(mx), np.diff(my)))])
    if a[-1] <= 1e-9:
        return None
    s0 = max(0.0, a[-1] - P["PATH_EXT_CHORD_M"])
    x0, y0 = np.interp(s0, a, mx), np.interp(s0, a, my)
    d = np.array([mx[-1] - x0, my[-1] - y0])
    n = float(np.hypot(*d))
    return d / n if n > 1e-9 else None


class HumanPath:
    """Polyline through the human ego positions. Station s = cumulative Euclidean length of the RAW 10 Hz human
    positions from sample 0 (the definition ROLLOUT_SCHEMA.md gives for `stop_line_s`). Knots are the human
    samples that moved > PATH_KNOT_EPS from the previous knot; per-knot speed / yaw / curvature are group means.
    Both ends are extended by straight tangent lines of PATH_EXT_M (A-6). AMENDMENT_6 ruling 2: the END extension runs along the chord over the
    last PATH_EXT_CHORD_M of arc length of the MOVING part (v >= PATH_EXT_V_MIN), never a stopped human's last heading; the START extension is unchanged."""

    def __init__(self, R):
        x, y, yaw, v = R["x"], R["y"], R["yaw"], R["v"]
        n = len(x)
        raw_s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        kn, grp = [0], np.zeros(n, np.int64)
        for j in range(1, n):
            if math.hypot(x[j] - x[kn[-1]], y[j] - y[kn[-1]]) > P["PATH_KNOT_EPS"]:
                kn.append(j)
            grp[j] = len(kn) - 1
        kn = np.array(kn)
        K = len(kn)
        cnt = np.bincount(grp, minlength=K).astype(float)
        yr = np.gradient(yaw, P["DT"])
        kap = np.where(v >= P["E2_V_MIN"], yr / np.where(v >= P["E2_V_MIN"], v, 1.0), np.nan)
        fin = np.isfinite(kap)
        kc = np.bincount(grp, weights=fin.astype(float), minlength=K)
        self.k_s = raw_s[kn]
        self.k_v = np.bincount(grp, weights=v, minlength=K) / cnt
        self.k_yaw = np.bincount(grp, weights=yaw, minlength=K) / cnt
        self.k_kap = np.where(kc > 0, np.bincount(grp, weights=np.where(fin, kap, 0.0), minlength=K) / np.maximum(kc, 1), np.nan)
        px, py = x[kn], y[kn]
        if K >= 2:
            u0 = np.array([px[1] - px[0], py[1] - py[0]]); u0 /= np.hypot(*u0)
            u1 = np.array([px[-1] - px[-2], py[-1] - py[-2]]); u1 /= np.hypot(*u1)
        else:
            u0 = u1 = np.array([math.cos(yaw[0]), math.sin(yaw[0])])
        ext_dir = _moving_chord_dir(x, y, v)
        u1 = ext_dir if ext_dir is not None else u1  # MUT:pathext
        E = P["PATH_EXT_M"]
        self.G = np.vstack([[px[0] - E * u0[0], py[0] - E * u0[1]], np.c_[px, py], [px[-1] + E * u1[0], py[-1] + E * u1[1]]])
        self.GS = np.concatenate([[self.k_s[0] - E], self.k_s, [self.k_s[-1] + E]])
        self.length = float(self.k_s[-1])
        self.n_knots = K
        fk = np.isfinite(self.k_kap)
        self._kap_s, self._kap_v = self.k_s[fk], self.k_kap[fk]

    def project(self, Q):
        """Q (N,2) -> (s, signed lateral offset (left +), segment tangent heading)"""
        Q = np.nan_to_num(np.asarray(Q, float).reshape(-1, 2))
        A, B = self.G[:-1], self.G[1:]
        d = B - A
        L2 = (d ** 2).sum(1)
        w = Q[:, None, :] - A[None]
        u = np.clip((w * d[None]).sum(2) / L2[None], 0.0, 1.0)
        C = A[None] + u[..., None] * d[None]
        d2 = ((Q[:, None, :] - C) ** 2).sum(2)
        j = np.argmin(d2, axis=1)
        r = np.arange(len(Q))
        s = self.GS[j] + u[r, j] * (self.GS[j + 1] - self.GS[j])
        cross = d[j, 0] * (Q[:, 1] - A[j, 1]) - d[j, 1] * (Q[:, 0] - A[j, 0])
        lat = np.where(cross >= 0, 1.0, -1.0) * np.sqrt(d2[r, j])
        return s, lat, np.arctan2(d[j, 1], d[j, 0])

    def speed_at(self, s):
        return np.interp(s, self.k_s, self.k_v)

    def yaw_at(self, s):
        return np.interp(s, self.k_s, self.k_yaw)

    def kappa_at(self, s):
        return np.interp(s, self._kap_s, self._kap_v) if self._kap_s.size else np.full(np.shape(s), np.nan)


def _corners(x, y, yaw, l, w):
    """(n,) arrays -> (n,4,2) oriented-box corners"""
    lx = np.array([1, 1, -1, -1]) * 0.5 * l
    ly = np.array([1, -1, -1, 1]) * 0.5 * w
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    return np.stack([x[:, None] + c * lx - s * ly, y[:, None] + s * lx + c * ly], axis=-1)


def _sat_overlap(c1, c2):
    """(n,4,2) x (n,4,2) oriented rectangles -> (n,) bool, strict overlap (touching does not count)"""
    ov = np.ones(c1.shape[0], bool)
    for c in (c1, c2):
        for e in ((c[:, 1] - c[:, 0]), (c[:, 3] - c[:, 0])):
            e = e / np.maximum(np.hypot(e[:, 0], e[:, 1]), 1e-12)[:, None]
            p1, p2 = (c1 * e[:, None, :]).sum(2), (c2 * e[:, None, :]).sum(2)
            ov &= (p1.min(1) < p2.max(1) - 1e-9) & (p2.min(1) < p1.max(1) - 1e-9)
    return ov


def _in_poly(pts, poly):
    """pts (N,2), poly (K,2) -> (N,) bool, even-odd ray casting"""
    if len(poly) < 3:
        return np.zeros(len(pts), bool)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None], poly[:, 1][None]
    x2, y2 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    cond = (y1 > y) != (y2 > y)
    xi = x1 + (y - y1) * (x2 - x1) / np.where(y2 != y1, y2 - y1, 1.0)
    return (cond & (x < xi)).sum(1) % 2 == 1


def _box_in_zone(cx, cy, cyaw, l, w, zone):
    """(n,) arrays -> (n,) bool: box footprint overlaps the zone polygon (9 box sample points, plus zone vertices in the box)"""
    n = len(cx)
    if len(zone) < 3 or n == 0:
        return np.zeros(n, bool)
    gx = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1]) * 0.5 * l
    gy = np.array([1, 0, -1, 1, 0, -1, 1, 0, -1]) * 0.5 * w
    c, s = np.cos(cyaw)[:, None], np.sin(cyaw)[:, None]
    px = cx[:, None] + c * gx - s * gy
    py = cy[:, None] + s * gx + c * gy
    hit = _in_poly(np.stack([px, py], -1).reshape(-1, 2), zone).reshape(n, 9).any(1)
    for vx, vy in zone:  # zone vertex inside the box (zone smaller than the box sampling)
        lx = c[:, 0] * (vx - cx) + s[:, 0] * (vy - cy)
        ly = -s[:, 0] * (vx - cx) + c[:, 0] * (vy - cy)
        hit |= (np.abs(lx) <= 0.5 * l) & (np.abs(ly) <= 0.5 * w)
    return hit


def _runs(mask):
    """maximal True runs -> list of (start, end_inclusive)"""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0].tolist(), (np.where(d == -1)[0] - 1).tolist()))


def _run_len_same(ids):
    """consecutive-tick count of the same non-negative id, ending at each tick (inclusive)"""
    out = np.zeros(len(ids), np.int64)
    for i in range(len(ids)):
        if ids[i] >= 0:
            out[i] = out[i - 1] + 1 if i > 0 and ids[i] == ids[i - 1] else 1
    return out


# ----------------------------------------------------------------------------------------------------------
# derived per-tick quantities: station, cross-track, lead / gap / TTC / time gap, free road
# ----------------------------------------------------------------------------------------------------------
def derive(R, path):
    n = R["n"]
    warm = (R["t"] - R["t"][0]) >= P["WARMUP_S"] - EPS
    s_ego, ct, _ = path.project(np.c_[R["x"], R["y"]])
    fx = R["x"] + 0.5 * R["L"] * np.cos(R["yaw"])
    fy = R["y"] + 0.5 * R["L"] * np.sin(R["yaw"])
    s_front, _, _ = path.project(np.c_[fx, fy])
    yawrate = np.gradient(R["yaw"], P["DT"])
    ref_valid = ((R["t"] - R["t"][0]) <= R["ref_until"] + EPS) | ~np.isfinite(R["ref_until"])  # MUT:refvalid
    D = dict(n=n, warm=warm, ref_valid=ref_valid, s_ego=s_ego, ct=ct, s_front=s_front, yawrate=yawrate,
             vh=path.speed_at(s_ego), head_err=_wrap(R["yaw"] - path.yaw_at(s_ego)),
             beyond_path=(s_ego > path.length + 1e-6))
    D["kappa_ego"] = np.where(R["v"] >= P["E2_V_MIN"], yawrate / np.where(R["v"] >= P["E2_V_MIN"], R["v"], 1.0), np.nan)
    D["kappa_h"] = path.kappa_at(s_ego)
    A = len(R["ids"])
    gap_a = np.full((A, n), np.inf)
    hw = P["CORRIDOR_HALF_WIDTH"]
    for a in range(A):
        if R["acls"][a] != 0:  # lead = vehicle only (A-2)
            continue
        pr = R["present"][a] & np.isfinite(R["ax"][a])
        if not pr.any():
            continue
        cs = _corners(np.nan_to_num(R["ax"][a]), np.nan_to_num(R["ay"][a]), np.nan_to_num(R["ayaw"][a]), R["al"][a], R["aw"][a])
        s_c, d_c, _ = path.project(cs.reshape(-1, 2))
        s_c, d_c = s_c.reshape(n, 4), d_c.reshape(n, 4)
        # A-17: corner-projected corridor overlap; ahead = largest corner station beyond the front bumper
        in_corr = pr & (d_c.min(1) <= hw) & (d_c.max(1) >= -hw) & (s_c.max(1) > s_front)
        gap_a[a] = np.where(in_corr, np.maximum(0.0, s_c.min(1) - s_front), np.inf)
    lead_idx = np.full(n, -1, np.int64)
    gap = np.full(n, np.nan)
    if A:
        am = np.argmin(gap_a, axis=0)  # ties -> smaller id (agents are sorted by id)
        has = np.isfinite(gap_a[am, np.arange(n)])
        lead_idx = np.where(has, am, -1)
        gap = np.where(has, gap_a[am, np.arange(n)], np.nan)
    has = lead_idx >= 0
    li = np.maximum(lead_idx, 0)
    v_lead = np.where(has, R["av"][li, np.arange(n)] if A else np.nan, np.nan)
    a_lead = np.full(n, np.nan)
    if A:
        h = nt(P["ALEAD_HALF_S"])
        i_lo, i_hi = np.maximum(np.arange(n) - h, 0), np.minimum(np.arange(n) + h, n - 1)
        va = R["av"]
        a_all = (va[:, i_hi] - va[:, i_lo]) / ((i_hi - i_lo) * P["DT"])[None]
        a_lead = np.where(has, a_all[li, np.arange(n)], np.nan)
    v_rel = R["v"] - v_lead
    ttc = np.where(has & (v_rel > P["VREL_MIN"]), gap / np.where(v_rel > P["VREL_MIN"], v_rel, 1.0), np.inf)
    time_gap = np.where(has & (R["v"] >= P["TG_VMIN"]), gap / np.where(R["v"] >= P["TG_VMIN"], R["v"], 1.0), np.nan)
    lead_id = np.where(has, R["ids"][li] if A else -1, -1)
    G = np.maximum(P["FREE_HEADWAY_S"] * R["v"], P["FREE_MIN_GAP_M"])
    near = has & (gap <= G)
    closing = has & (ttc <= P["FREE_TTC_S"])
    hold = _signal_hold(R, D)   # AMENDMENT_3 CH-1 (Ruling 1): a held tick is not free road
    # A-1: "no lead within G, and TTC_t > 8 s"; AMENDMENT_3 CH-1: and not held
    free = ~(near | closing) & ~hold  # MUT:hold
    fw = free & warm & ref_valid   # a1, a2: AMENDMENT_5 item 2
    ints = [(i, j) for i, j in _runs(fw) if (j - i + 1) >= nt(P["FREE_MIN_INTERVAL_S"])]
    free_idx = np.concatenate([np.arange(i, j + 1) for i, j in ints]) if ints else np.zeros(0, np.int64)
    D.update(hold=hold, lead_idx=lead_idx, lead_id=lead_id, has_lead=has, gap=gap, v_lead=v_lead, a_lead=a_lead, v_rel=v_rel,
             ttc=ttc, time_gap=time_gap, lead_run=_run_len_same(lead_id), free=free, free_intervals=ints, free_idx=free_idx)
    return D


# ----------------------------------------------------------------------------------------------------------
# rules. Each returns dict(status raw, value, threshold, n, details). status in PASS/FAIL/NA
# ----------------------------------------------------------------------------------------------------------
def _refinfo(R, D):
    """AMENDMENT_5 item 2: report the NA count for the human-path rules; empty (output unchanged) when the key is absent"""
    if not np.isfinite(R["ref_until"]):
        return {}
    return dict(human_ref_valid_until_s=R["ref_until"], n_ticks_na_after_human_ref=int((~D["ref_valid"]).sum()))


def _na(why, **kw):
    return dict(status="NA", value=None, threshold=None, n=0, details=dict(reason=why, **kw))


def rule_a1(Rc, Dc, Rh, Dh, path):
    idx = Dc["free_idx"]
    if idx.size == 0:
        return _na("no free-road interval (>= 2.0 s after the 2.0 s warm-up)")
    ratio = Rc["v"][idx] / np.maximum(Dc["vh"][idx], P["A1_VH_FLOOR"])
    share = float(np.mean((ratio >= P["A1_LO"] - EPS) & (ratio <= P["A1_HI"] + EPS)))
    m = idx[idx < Rh["n"]]
    matched = float(np.mean((lambda r: (r >= P["A1_LO"] - EPS) & (r <= P["A1_HI"] + EPS))(Rc["v"][m] / np.maximum(Rh["v"][m], P["A1_VH_FLOOR"])))) if m.size else None
    status = "PASS" if share >= P["A1_SHARE"] - EPS else "FAIL"  # MUT:a1
    return dict(status=status, value=share, threshold=P["A1_SHARE"], n=int(idx.size),
                details=dict(matched_time_share=matched, n_free_intervals=len(Dc["free_intervals"]), min_ratio=float(ratio.min()), max_ratio=float(ratio.max()), **_refinfo(Rc, Dc)))


def rule_a2(Rc, Dc, Rh, Dh, path):
    if not Dc["free_intervals"]:
        return _na("no free-road interval")
    w = nt(P["A2_WINDOW_S"])
    v, vh = Rc["v"], Dc["vh"]
    starts = []
    for i0, i1 in Dc["free_intervals"]:
        for i in range(i0, i1 - w + 1):
            if v[i] > P["A2_V_MIN"] and vh[i] > P["A2_V_MIN"] and v[i + w] / v[i] < P["A2_RATIO"] and vh[i + w] / vh[i] >= P["A2_RATIO"]:
                starts.append(i)
    events = []  # windows that share a tick merge into one event
    for i in starts:
        if events and i <= events[-1][1] + w:  # windows [i,i+w] and [last,last+w] share a tick -> same event
            events[-1][1] = i
        else:
            events.append([i, i])
    n_ev = len(events)
    status = "PASS" if n_ev == 0 else "FAIL"  # MUT:a2
    slow = None
    if Rc["plan_ratio"] is not None and Dc["free_idx"].size:
        pr = Rc["plan_ratio"][Dc["free_idx"]]
        slow = float(np.mean(pr[np.isfinite(pr)] < P["A2_RATIO"])) if np.isfinite(pr).any() else None
    return dict(status=status, value=n_ev, threshold=0, n=len(Dc["free_intervals"]),
                details=dict(event_start_t=[round(float(Rc["t"][e[0]] - Rc["t"][0]), 3) for e in events][:20],
                             n_violating_windows=len(starts), plan_slow_share_free_road=slow, **_refinfo(Rc, Dc)))


def _steady(Rc, Dc):  # A-16: same lead id for >= 30 consecutive ticks ending here
    return (Dc["warm"] & Dc["has_lead"] & (Dc["lead_run"] >= nt(P["B1_STEADY_S"])) & (Rc["v"] >= P["TG_VMIN"])
            & np.isfinite(Dc["a_lead"]) & (np.abs(np.nan_to_num(Dc["a_lead"], nan=99.0)) <= P["B1_ALEAD_MAX"] + EPS))


def rule_b1(Rc, Dc, Rh, Dh, path):
    m = _steady(Rc, Dc)
    if not m.any():
        return _na("no steady-follow ticks")
    med = float(np.median(Dc["time_gap"][m]))
    status = "PASS" if (P["B1_LO"] - EPS <= med <= P["B1_HI"] + EPS) else "FAIL"  # MUT:b1
    return dict(status=status, value=med, threshold=[P["B1_LO"], P["B1_HI"]], n=int(m.sum()),
                details=dict(min_time_gap=float(Dc["time_gap"][m].min()), max_time_gap=float(Dc["time_gap"][m].max())))


def rule_b2(Rc, Dc, Rh, Dh, path):
    el = Dc["warm"] & Dc["has_lead"] & (Rc["v"] >= P["TG_VMIN"])
    if not el.any():
        return _na("no tick with a lead at v_ego >= 3 m/s after warm-up")
    bad = el & (Dc["time_gap"] < P["B2_TG"])
    runs = [r for r in _runs(bad) if r[1] - r[0] + 1 >= P["PERSIST_TICKS"]]
    status = "PASS" if len(runs) == 0 else "FAIL"  # MUT:b2
    return dict(status=status, value=len(runs), threshold=0, n=int(el.sum()),
                details=dict(min_time_gap=float(np.nanmin(Dc["time_gap"][el])), n_short_runs_lt3=len(_runs(bad)) - len(runs)))


def _gov_state(R):
    """most restrictive state among heads that govern the ego: red > yellow > green > none"""
    if R["sig"].shape[1] == 0 or not R["gov"].any():
        return np.zeros(R["n"], np.int64)
    s = R["sig"][:, R["gov"]]
    out = np.zeros(R["n"], np.int64)
    out = np.where((s == SIGNAL_GREEN).any(1), SIGNAL_GREEN, out)
    out = np.where((s == SIGNAL_YELLOW).any(1), SIGNAL_YELLOW, out)
    out = np.where((s == SIGNAL_RED).any(1), SIGNAL_RED, out)
    return out


def _signal_hold(R, D):
    """AMENDMENT_3 section 7 Ruling 1 + section 9 Q4. (n,) bool, True on every held tick.
    A hold STARTS at a tick where the governing signal is red, the front bumper is at or before the stop line and within C1_APPROACH_M (40 m, Q4)
    of it. It LATCHES until the first tick at or after a governing green at which v > C4_LAUNCH_V; that launch tick itself is not held, the ticks
    between the green and it are. A scenario without a governing signal and a stop line never holds (bit-identical to the pre-A3 scorer)."""
    n = R["n"]
    hold = np.zeros(n, bool)
    if not _scenario_has_signal(R):
        return hold
    g = _gov_state(R)
    dist = R["stop_s"] - D["s_front"]
    can_start = (g == SIGNAL_RED) & (D["s_front"] <= R["stop_s"] + EPS) & (dist <= P["C1_APPROACH_M"] + EPS)
    held, seen_green = False, False
    for i in range(n):
        if not held:
            if can_start[i]:
                held, seen_green = True, False
                hold[i] = True
            continue
        seen_green = seen_green or g[i] == SIGNAL_GREEN
        if seen_green and R["v"][i] > P["C4_LAUNCH_V"]:
            held = False
            if can_start[i]:   # a red right on the launch tick starts a new hold
                held, seen_green, hold[i] = True, False, True
        else:
            hold[i] = True
    return hold


def _conflict_in_zone(R):  # A-13
    """(n,) bool per tick: some conflicting agent's footprint overlaps the zone at this tick; also per-tick 'within 2 s' flag"""
    n = R["n"]
    ids = R["ids"]
    sel = np.ones(len(ids), bool) if R["conf_ids"].size == 0 else np.isin(ids, R["conf_ids"])
    now = np.zeros(n, bool)
    for a in np.where(sel)[0]:
        pr = R["present"][a] & np.isfinite(R["ax"][a])
        if not pr.any():
            continue
        h = _box_in_zone(np.nan_to_num(R["ax"][a]), np.nan_to_num(R["ay"][a]), np.nan_to_num(R["ayaw"][a]), R["al"][a], R["aw"][a], R["zone"])
        now |= (h & pr)
    k = nt(P["C3_WITHIN_S"])
    soon = np.zeros(n, bool)
    for i in range(n):
        soon[i] = now[i:i + k + 1].any()   # footprint in zone at some tick in [t, t+2.0 s]
    return now, soon


def _scenario_has_signal(R):
    return R["sig"].shape[1] > 0 and R["gov"].any() and np.isfinite(R["stop_s"])


def rule_c1(Rc, Dc, Rh, Dh, path):
    if not np.isfinite(Rc["stop_s"]):
        return _na("no stop line in the scenario")
    if _scenario_has_signal(Rc):
        clear = _gov_state(Rc) == SIGNAL_GREEN
    elif len(Rc["zone"]) >= 3:
        clear = ~_conflict_in_zone(Rc)[1]
    else:
        return _na("no signal and no conflict zone: 'clear' undefined")
    dist = Rc["stop_s"] - Dc["s_front"]
    cand = clear & (dist <= P["C1_APPROACH_M"]) & (dist >= 0)
    if not cand.any():
        return _na("ego never within 40 m of the line while clear")
    i0 = int(np.argmax(cand))
    stop_run = any(a <= i0 <= b and (b - a + 1) >= nt(P["STOP_MIN_S"]) for a, b in _runs(Rc["v"] <= P["C4_STOP_V"]))
    # AMENDMENT_3 CH-2 (Ruling 2, Q5): "stopped" is c4's definition, v <= 1.0 m/s inside a run of >= 1.0 s
    stopped_first = bool(Rc["v"][i0] <= P["C4_STOP_V"]) and stop_run  # MUT:c1na
    if stopped_first:
        return _na("stopped at the first clear tick; start-from-stop is judged by c4")
    after = np.arange(Rc["n"]) > i0
    stop_here = after & (Dc["s_front"] >= Rc["stop_s"])
    i1 = int(np.argmax(stop_here)) if stop_here.any() else Rc["n"] - 1
    if P["C1_MODE"] == "clear_persistent":
        unclear = after & ~clear
        if unclear.any():
            i1 = min(i1, int(np.argmax(unclear)) - 1)
    i1 = max(i1, i0)
    mv = float(Rc["v"][i0:i1 + 1].min())
    status = "PASS" if mv > P["C1_MINV"] else "FAIL"  # MUT:c1
    return dict(status=status, value=mv, threshold=P["C1_MINV"], n=int(i1 - i0 + 1),
                details=dict(window_t=[float(Rc["t"][i0] - Rc["t"][0]), float(Rc["t"][i1] - Rc["t"][0])], mode=P["C1_MODE"]))


def rule_c2(Rc, Dc, Rh, Dh, path):
    if not _scenario_has_signal(Rc):
        return _na("no governing signal / stop line")
    g = _gov_state(Rc)
    red = g == SIGNAL_RED
    if not red.any():
        return _na("signal never red")
    sf, s0 = Dc["s_front"], Rc["stop_s"]
    # AMENDMENT_3 CH-3 (Q3): a crossing EVENT on red, not any red tick past the line
    crossed = red[1:] & (sf[:-1] <= s0 + EPS) & (sf[1:] > s0 + EPS)  # MUT:c2x
    n_cross = int(crossed.sum())
    stop_gaps, open_end = [], False
    for i0, i1 in _runs(red & (sf <= s0 + EPS)):   # red episodes with the bumper still before the line
        stopped = np.where(Rc["v"][i0:i1 + 1] <= P["STOPPED_V"])[0]
        if stopped.size:
            stop_gaps.append(float(s0 - sf[i0 + stopped[0]]))
        elif i1 == Rc["n"] - 1:
            open_end = True
    far = [x for x in stop_gaps if x > P["C2_STOP_CAP_M"] + EPS]
    ok = (n_cross == 0) and (len(far) == 0)
    status = "PASS" if ok else "FAIL"  # MUT:c2
    return dict(status=status, value=n_cross, threshold=0, n=int(red.sum()),
                details=dict(stop_distances_before_line_m=stop_gaps, n_stops_beyond_6m_cap=len(far), red_open_at_end_not_stopped=open_end))


def rule_c3(Rc, Dc, Rh, Dh, path, f1, f2):
    if len(Rc["zone"]) < 3:
        return _na("no conflict zone")
    ego_in = _box_in_zone(Rc["x"], Rc["y"], Rc["yaw"], Rc["L"], Rc["W"], Rc["zone"])
    _, soon = _conflict_in_zone(Rc)
    viol = 0
    for i in range(1, Rc["n"]):
        if ego_in[i] and not ego_in[i - 1] and ego_in[i:i + P["PERSIST_TICKS"]].all() and soon[i]:
            viol += 1
    f_ok = f1["status"] != "FAIL" and f2["status"] != "FAIL"
    status = "PASS" if (viol == 0 and f_ok) else "FAIL"  # MUT:c3
    return dict(status=status, value=viol, threshold=0, n=int(ego_in.sum()), details=dict(rule_f_holds=bool(f_ok), ego_ever_in_zone=bool(ego_in.any())))


def rule_c4(Rc, Dc, Rh, Dh, path):
    v = Rc["v"]
    stopped = v <= P["C4_STOP_V"]
    g = _gov_state(Rc)
    green_edge = np.concatenate([[False], (g[1:] == SIGNAL_GREEN) & (g[:-1] != SIGNAL_GREEN)])
    lv = np.nan_to_num(Dc["v_lead"], nan=0.0)
    pull_edge = np.concatenate([[False], (Dc["has_lead"][1:] & (lv[1:] >= P["C4_LEAD_PULL_V"]) & ~(Dc["has_lead"][:-1] & (lv[:-1] >= P["C4_LEAD_PULL_V"])))])
    trig = (green_edge | pull_edge) & stopped & Dc["ref_valid"]   # AMENDMENT_5 item 2: a trigger after the human reference stops being valid is NA
    mstop = nt(P["STOP_MIN_S"])
    events = []
    for s0, s1 in _runs(stopped):
        if s1 - s0 + 1 < mstop:
            continue
        w = np.where(trig[s0:s1 + 1])[0]
        if w.size:
            events.append(s0 + int(w[0]))
    if not events:
        return _na("no stop followed by a green-light or lead-pull-away trigger", **_refinfo(Rc, Dc))
    worst, any_fail, cens = -1e9, False, 0
    det = []
    for tr in events:
        mv = np.where(v[tr:] > P["C4_LAUNCH_V"])[0]
        t_c = float(mv[0] * P["DT"]) if mv.size else math.inf
        hv = Rh["v"]
        if tr >= len(hv):
            hum = math.inf
        else:
            hm = np.where(hv[tr:] > P["C4_LAUNCH_V"])[0]
            hum = float(hm[0] * P["DT"]) if hm.size else math.inf
        allowed = max(P["C4_MAX_S"], hum + P["C4_HUMAN_PLUS_S"])
        remaining = (Rc["n"] - 1 - tr) * P["DT"]
        if not mv.size and remaining < allowed:
            cens += 1
            det.append(dict(trigger_t=round(float(Rc["t"][tr] - Rc["t"][0]), 3), censored=True))
            continue
        fail = t_c > allowed + EPS
        any_fail |= fail
        worst = max(worst, t_c - allowed)
        det.append(dict(trigger_t=round(float(Rc["t"][tr] - Rc["t"][0]), 3), launch_s=t_c if math.isfinite(t_c) else None, human_s=hum if math.isfinite(hum) else None, allowed_s=allowed if math.isfinite(allowed) else None))
    if cens == len(events):
        return _na("every launch event censored by the end of the scenario", events=det, **_refinfo(Rc, Dc))
    status = "FAIL" if any_fail else "PASS"  # MUT:c4
    return dict(status=status, value=worst, threshold=0.0, n=len(events) - cens, details=dict(events=det, n_censored=cens, **_refinfo(Rc, Dc)))


def _states(R):
    """realised ego states as the 11-column NAVSIM state array (n,11) (A-4: lateral accel = v * yaw rate)"""
    _, SI = _navsim()
    S = np.zeros((R["n"], SI.size()))
    yr = np.gradient(R["yaw"], P["DT"])
    S[:, 0], S[:, 1], S[:, 2] = R["x"], R["y"], R["yaw"]
    S[:, 3], S[:, 5], S[:, 6] = R["v"], R["a"], R["v"] * yr
    S[:, 7], S[:, 8] = R["steer"], np.gradient(R["steer"], P["DT"])
    S[:, 9], S[:, 10] = yr, np.gradient(yr, P["DT"])
    return S


def comfort_share(R, starts):
    """share of 4.0 s windows (41 samples at 10 Hz) on which NAVSIM ego_is_comfortable is True in all 6 checks"""
    m, _ = _navsim()
    if not len(starts):
        return None, np.zeros((0, 6), bool)
    w = nt(P["D1_WINDOW_S"]) + 1
    S = _states(R)
    batch = np.stack([S[i:i + w] for i in starts])
    res = m.ego_is_comfortable(batch, P["DT"] * np.arange(w))
    return float(np.mean(res.all(axis=1))), res


def rule_d1(Rc, Dc, Rh, Dh, path):
    w, stride, i0 = nt(P["D1_WINDOW_S"]), nt(P["D1_STRIDE_S"]), nt(P["WARMUP_S"])
    starts = list(range(i0, Rc["n"] - w, stride))
    if not starts:
        return _na("scenario shorter than warm-up + 4 s")
    share, res = comfort_share(Rc, starts)
    hs = [i for i in starts if i + w < Rh["n"]]
    h_share, _ = comfort_share(Rh, hs)
    thr = P["D1_SHARE"] if h_share is None else min(P["D1_SHARE"], h_share - P["D1_HUMAN_MARGIN"])
    status = "PASS" if share >= thr - EPS else "FAIL"  # MUT:d1
    return dict(status=status, value=share, threshold=thr, n=len(starts),
                details=dict(human_share=h_share, n_human_windows=len(hs), failing_checks_per_window=(~res).sum(0).tolist(),
                             check_order=["lon_acc", "lat_acc", "jerk_mag", "lon_jerk", "yaw_acc", "yaw_rate"]))


def _zigzag(v, thr):
    n = len(v)
    piv, d, ext = [0], 0, 0
    for i in range(1, n):
        if d == 0:
            if v[i] - v[0] >= thr:
                d, ext = 1, i
            elif v[0] - v[i] >= thr:
                d, ext = -1, i
        elif d == 1:
            if v[i] > v[ext]:
                ext = i
            elif v[ext] - v[i] >= thr:
                piv.append(ext); d, ext = -1, i
        else:
            if v[i] < v[ext]:
                ext = i
            elif v[i] - v[ext] >= thr:
                piv.append(ext); d, ext = 1, i
    if d != 0:
        piv.append(ext)
    return [p for k, p in enumerate(piv) if k == 0 or p != piv[k - 1]]


def rule_d2(Rc, Dc, Rh, Dh, path):  # A-18 scope, A-9 no persistence
    m = Dc["warm"]
    if m.sum() < 3:
        return _na("scenario shorter than the warm-up")
    t, v = Rc["t"][m], Rc["v"][m]
    k = nt(P["D2_SMOOTH_S"])
    ker = np.ones(k)
    vs = np.convolve(v, ker, mode="same") / np.convolve(np.ones(len(v)), ker, mode="same")
    piv = _zigzag(vs, P["D2_TURN"])
    dv = np.diff(vs[piv])
    qual = []
    for j in range(len(dv) - 1):
        if (dv[j] * dv[j + 1] < 0 and abs(dv[j]) >= P["D2_SWING"] - EPS and abs(dv[j + 1]) >= P["D2_SWING"] - EPS
                and t[piv[j + 2]] - t[piv[j]] <= P["D2_WITHIN_S"] + EPS):
            qual.append(j)
    n_ev = sum(1 for j in qual if (j - 1) not in qual)  # chains of overlapping qualifying pairs = one event
    status = "PASS" if n_ev == 0 else "FAIL"  # MUT:d2
    return dict(status=status, value=n_ev, threshold=0, n=len(piv) - 1,
                details=dict(swings=[round(float(x), 3) for x in dv][:30], first_event_t=(round(float(t[piv[qual[0]]] - Rc["t"][0]), 3) if qual else None)))


def rule_e1(Rc, Dc, Rh, Dh, path):
    if not Dc["ref_valid"].any():
        return _na("no tick with a valid human reference", **_refinfo(Rc, Dc))
    bad = (np.abs(Dc["ct"]) > P["E1_CT"]) & Dc["ref_valid"]   # AMENDMENT_5 item 2
    runs = [r for r in _runs(bad) if (r[1] - r[0] + 1) >= nt(P["E1_RUN_S"])]
    status = "PASS" if len(runs) == 0 else "FAIL"  # MUT:e1
    return dict(status=status, value=len(runs), threshold=0, n=int(Rc["n"]),
                details=dict(max_abs_cross_track=float(np.abs(Dc["ct"]).max()), longest_run_s=(max([(r[1] - r[0] + 1) * P["DT"] for r in _runs(bad)]) if bad.any() else 0.0), **_refinfo(Rc, Dc)))


def rule_e2(Rc, Dc, Rh, Dh, path):
    if not Dc["ref_valid"].any():
        return _na("no tick with a valid human reference", **_refinfo(Rc, Dc))
    rv = Dc["ref_valid"]
    he = np.abs(Dc["head_err"])[rv]
    ke = np.abs(Dc["kappa_ego"] - Dc["kappa_h"])[rv]
    ke = ke[np.isfinite(ke)]
    p95h = float(np.percentile(he, 95))
    p95k = float(np.percentile(ke, 95)) if ke.size else None
    ok_h = p95h <= P["E2_HEAD"] + EPS
    ok_k = True if p95k is None else p95k <= P["E2_KAPPA"] + EPS
    status = "PASS" if (ok_h and ok_k) else "FAIL"  # MUT:e2
    return dict(status=status, value=dict(p95_heading_err=p95h, p95_kappa_err=p95k), threshold=dict(heading=P["E2_HEAD"], kappa=P["E2_KAPPA"]),
                n=int(Rc["n"]), details=dict(heading_ok=bool(ok_h), kappa_ok=bool(ok_k), n_kappa=int(ke.size), **_refinfo(Rc, Dc)))


def _collisions(Rc, Dc):
    """Overlap events with every agent class, and NAVSIM's at-fault classification reproduced from SOURCE (A-10, AMENDMENT_1).

    Follows pdm_scorer.py:363-420 (`_calculate_no_at_fault_collision`) tick by tick, per track:
      * a track already overlapping the ego at the first tick is in `collided_track_ids` from the start (pdm_observation.py:209-220) -> exempt;
      * every other overlap tick is classified by the VENDORED `get_collision_type` (order STOPPED_EGO, STOPPED_TRACK, ACTIVE_REAR, ACTIVE_FRONT, else
        ACTIVE_LATERAL; ego "stopped" and track "stopped" both <= 5e-02 m/s; rear = agent centre > 150 deg off the ego REAR-AXLE heading; front = the
        front-bumper segment meets the track polygon);
      * at fault = STOPPED_TRACK or ACTIVE_FRONT, or ACTIVE_LATERAL while the ego is in multiple lanes / non-drivable area -- on the lane-less road
        that condition is the stand-in |cross-track| > LAT_FAULT_CT (flagged `lane_proxy`);
      * a tick that is NOT at fault puts the track into the exempt list: that track is skipped from then on (pdm_scorer.py:383, :418).
    Score of an at-fault hit: 0.0 for an Agent, 0.5 for any other object (pdm_scorer.py:402-405); f1 treats both as FAIL."""
    pdm_collision, shim = _collision_mod()
    CT = shim.CollisionType
    ev = []
    ego_c = _corners(Rc["x"], Rc["y"], Rc["yaw"], Rc["L"], Rc["W"])
    for a in range(len(Rc["ids"])):
        pr = Rc["present"][a] & np.isfinite(Rc["ax"][a])
        if not pr.any():
            continue
        ac = _corners(np.nan_to_num(Rc["ax"][a]), np.nan_to_num(Rc["ay"][a]), np.nan_to_num(Rc["ayaw"][a]), Rc["al"][a], Rc["aw"][a])
        ov = _sat_overlap(ego_c, ac) & pr
        exempt = bool(ov[0])                                   # in collided_track_ids from the start
        for i0, i1 in _runs(ov):
            first = hit = None                                 # (tick, type, fault, lateral) at the first classified tick / the first at-fault tick
            for i in range(i0, i1 + 1):
                if exempt:
                    break
                ct, fault = _classify_tick(Rc, Dc, a, i)
                rec = (i, ct, fault, ct == CT.ACTIVE_LATERAL_COLLISION)
                first = first or rec
                hit = hit or (rec if fault else None)
                if not fault:
                    exempt = True                              # first not-at-fault contact exempts the track from here on (also within this run)
            if first is None:                                  # whole run exempt
                ev.append(dict(agent_id=int(Rc["ids"][a]), start_t=round(float(Rc["t"][i0] - Rc["t"][0]), 3), n_ticks=int(i1 - i0 + 1), kind="EXEMPT",
                               at_fault=False, exempt=True, lane_proxy_used=False, navsim_no_at_fault_score=1.0))
                continue
            i, ct, fault, lateral = hit or first
            ev.append(dict(agent_id=int(Rc["ids"][a]), start_t=round(float(Rc["t"][i] - Rc["t"][0]), 3), n_ticks=int(i1 - i0 + 1),
                           kind=CT(ct).name.replace("_COLLISION", ""), at_fault=bool(fault), exempt=False, lane_proxy_used=bool(lateral),
                           navsim_no_at_fault_score=((0.5 if Rc["acls"][a] == 3 else 0.0) if fault else 1.0)))
    return ev


def _classify_tick(Rc, Dc, a, i):
    """NAVSIM's classification of ONE overlap tick of agent `a` (no exemption logic): (CollisionType, at_fault)."""
    from shapely import Polygon
    pdm_collision, shim = _collision_mod()
    CT = shim.CollisionType
    one = lambda k: np.array([k[i]], float)
    ego_c = _corners(one(Rc["x"]), one(Rc["y"]), one(Rc["yaw"]), Rc["L"], Rc["W"])[0]
    ag_c = _corners(one(np.nan_to_num(Rc["ax"][a])), one(np.nan_to_num(Rc["ay"][a])), one(np.nan_to_num(Rc["ayaw"][a])), Rc["al"][a], Rc["aw"][a])[0]
    state = np.zeros(11)                                                                  # NAVSIM state array (rear-axle pose, ego-frame velocity)
    state[0], state[1] = Rc["x"][i] - Rc["RA"] * math.cos(Rc["yaw"][i]), Rc["y"][i] - Rc["RA"] * math.sin(Rc["yaw"][i])     # X, Y
    state[2], state[3], state[4] = Rc["yaw"][i], Rc["v"][i], 0.0                         # HEADING, VELOCITY_X (along heading), VELOCITY_Y
    obj = shim.TrackedObject(Rc["ayaw"][a, i]) if Rc["acls"][a] == 3 else shim.Agent(Rc["ayaw"][a, i], Rc["av"][a, i])
    with np.errstate(all="ignore"):
        # our corner order is FL, FR, RR, RL; NAVSIM's ego polygon is FL, RL, RR, FR so exterior.coords[0] and [3] are the front bumper
        ct = pdm_collision.get_collision_type(state, Polygon(ego_c[[0, 3, 2, 1]]), obj, Polygon(ag_c))  # MUT:f1_cls
    fault = ct in (CT.ACTIVE_FRONT_COLLISION, CT.STOPPED_TRACK_COLLISION) or (ct == CT.ACTIVE_LATERAL_COLLISION and bool(abs(Dc["ct"][i]) > P["LAT_FAULT_CT"]))
    return CT(ct), bool(fault)


def rule_f1(Rc, Dc, Rh, Dh, path):
    ev = _collisions(Rc, Dc)
    n_fault = sum(1 for e in ev if e["at_fault"])
    status = "PASS" if n_fault == 0 else "FAIL"  # MUT:f1
    return dict(status=status, value=n_fault, threshold=0, n=len(Rc["ids"]), flags=dict(lane_proxy=True),
                details=dict(n_overlap_events_total=len(ev), events=ev[:20], lane_proxy=True,
                             source="NAVSIM pdm_scorer.py:363-420 + pdm_scorer_utils.py:12 via vendored navsim_pdm/pdm_collision.py (A-10)"))


def rule_f2(Rc, Dc, Rh, Dh, path):
    bad = Dc["has_lead"] & (Dc["ttc"] < P["F2_TTC"]) & (Rc["v"] > P["F2_V_STOPPED"])
    runs = [r for r in _runs(bad) if r[1] - r[0] + 1 >= P["PERSIST_TICKS"]]
    status = "PASS" if len(runs) == 0 else "FAIL"  # MUT:f2
    fin = Dc["ttc"][np.isfinite(Dc["ttc"])]
    return dict(status=status, value=len(runs), threshold=0, n=int(Dc["has_lead"].sum()),
                details=dict(min_ttc=(float(fin.min()) if fin.size else None), n_short_runs_lt3=len(_runs(bad)) - len(runs)))


def evaluate_rules(Rc, Dc, Rh, Dh, path):
    out = {}
    for r in ("a1", "a2", "b1", "b2", "c1", "c2"):
        out[r] = globals()["rule_" + r](Rc, Dc, Rh, Dh, path)
    f1, f2 = rule_f1(Rc, Dc, Rh, Dh, path), rule_f2(Rc, Dc, Rh, Dh, path)
    out["c3"] = rule_c3(Rc, Dc, Rh, Dh, path, f1, f2)
    for r in ("c4", "d1", "d2", "e1", "e2"):
        out[r] = globals()["rule_" + r](Rc, Dc, Rh, Dh, path)
    out["f1"], out["f2"] = f1, f2
    return out


# ----------------------------------------------------------------------------------------------------------
# one scenario
# ----------------------------------------------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def score_arrays(roll, hum):
    """roll, hum: dicts of numpy arrays (already loaded). Returns the per-scenario result dict."""
    for kind, d in (("rollout", roll), ("human", hum)):
        pr = validate(d, kind)
        if pr:
            raise SchemaError(f"{kind}: " + "; ".join(pr))
    Rc = build_run(roll)
    Rh = build_run(hum, fallback=roll)
    path = HumanPath(Rh)
    Dc, Dh = derive(Rc, path), derive(Rh, path)
    rc = evaluate_rules(Rc, Dc, Rh, Dh, path)
    rh = evaluate_rules(Rh, Dh, Rh, Dh, path)
    rules = {}
    for r in RULES:
        raw, hs = rc[r]["status"], rh[r]["status"]
        final = "INFORMATIONAL" if (raw in ("PASS", "FAIL") and hs == "FAIL") else raw
        rules[r] = dict(status=final, raw_status=raw, human_status=hs, value=rc[r]["value"], threshold=rc[r]["threshold"], n=rc[r]["n"],
                        human_value=rh[r]["value"], details=rc[r]["details"], **rc[r].get("flags", {}))
    ident = {k: _str(roll, k) for k in ("checkpoint_sha256", "dtype", "suite_tokens_sha256")}
    ident["seed"] = int(np.asarray(roll["seed"]).reshape(-1)[0]) if "seed" in roll else None
    ident["complete"] = all(v not in (None, "") for v in ident.values())
    ep = float(np.asarray(roll["epdms"]).reshape(-1)[0]) if "epdms" in roll else float("nan")
    fam = {f: {r: rules[r]["status"] for r in rs} for f, rs in FAMILIES.items()}
    fam["STRATEGIC"] = "PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE.md item 25): never a pass, never omitted"
    sc = dict(
        schema_version=SCHEMA_VERSION, scorecard_sha256=SCORECARD_SHA256,
        scenario_id=_str(roll, "scenario_id"), log_id=_str(roll, "log_id", _str(roll, "scenario_id")),
        category=_str(roll, "category"), synthetic=bool(np.asarray(roll["synthetic"]).reshape(-1)[0]),
        tier_stamp=TIER_STAMP, agent_cls_defaulted=bool("agent_id" in roll and np.asarray(roll["agent_id"]).size > 0 and "agent_cls" not in roll),
        identity=ident,
        epdms=dict(value=ep, terms=({str(k): float(v) for k, v in zip(np.asarray(roll["epdms_terms_names"]).tolist(), np.asarray(roll["epdms_terms"]).tolist())} if "epdms_terms" in roll else None)),
        rules=rules, families=fam,
        derived=dict(n_ticks_10hz=int(Rc["n"]), duration_s=float(Rc["t"][-1] - Rc["t"][0]), free_road_tick_share=float(np.mean(Dc["free"])),
                     n_free_intervals=len(Dc["free_intervals"]), lead_tick_share=float(np.mean(Dc["has_lead"])),
                     min_gap_m=(float(np.nanmin(Dc["gap"])) if Dc["has_lead"].any() else None),
                     min_time_gap_s=(float(np.nanmin(Dc["time_gap"])) if np.isfinite(Dc["time_gap"]).any() else None),
                     max_abs_cross_track_m=float(np.abs(Dc["ct"]).max()), n_ticks_beyond_human_path=int(Dc["beyond_path"].sum()),
                     human_path_length_m=path.length, human_path_knots=int(path.n_knots)),
        params={k: v for k, v in P.items()})
    if "human_ref_valid_until_s" in roll:   # AMENDMENT_5 item 2: emitted only when present
        sc["human_ref_valid_until_s"] = float(np.asarray(roll["human_ref_valid_until_s"]).reshape(-1)[0])
    if "slot" in roll:   # AMENDMENT_3 CH-4: emitted only when the rollout carries it, so every pre-A3 output stays byte-identical
        sc["slot"] = _str(roll, "slot")
    ticks = dict(t=Rc["t"] - Rc["t"][0], v=Rc["v"], s_ego=Dc["s_ego"], cross_track=Dc["ct"], lead_id=Dc["lead_id"], gap=Dc["gap"], ttc=Dc["ttc"],
                 time_gap=Dc["time_gap"], free=Dc["free"], vh_at_ego=Dc["vh"], head_err=Dc["head_err"])
    return _clean(sc), ticks


def score_files(rollout_path, human_path, out_path=None, dump_ticks=None):
    roll, hum = load_npz(rollout_path), load_npz(human_path)
    sc, ticks = score_arrays(roll, hum)
    sc["inputs"] = dict(rollout=str(rollout_path), rollout_sha256=hashlib.sha256(Path(rollout_path).read_bytes()).hexdigest(),
                        human=str(human_path), human_sha256=hashlib.sha256(Path(human_path).read_bytes()).hexdigest())
    txt = json.dumps(sc, sort_keys=True, indent=1, allow_nan=False)
    if out_path:
        Path(out_path).write_text(txt, encoding="utf-8")
    if dump_ticks:
        np.savez(dump_ticks, **ticks)
    return sc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--rollout", required=True)
    s.add_argument("--human", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--dump-ticks", default=None)
    a = ap.parse_args(argv)
    sc = score_files(a.rollout, a.human, a.out, a.dump_ticks)
    print(json.dumps({r: sc["rules"][r]["status"] for r in RULES}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
