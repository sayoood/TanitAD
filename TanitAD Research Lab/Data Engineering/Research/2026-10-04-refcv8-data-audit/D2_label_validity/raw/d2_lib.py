"""D2 label-validity audit -- shared library (refcv8 data audit, 2026-10-04).

INDEPENDENCE RULE (CLAUDE.md): the geometry below is authored from road geometry and
the brief, with every threshold written as a LITERAL here.  Nothing is imported from
`s2_geom_emit_v7.py` / `ego_manoeuvre.py`.  (The builder is imported ONLY in
d2_controls.py, as an object under test on synthetic tracks.)

Pose convention (checked, see RESULT.md): poses = (x, y, yaw, v); yaw CCW-positive, so
LEFT is +lateral and TURN_L is +dyaw -- the same convention the label file uses.
Time axis: provider row p sits at raw time  grid_start_s + (p + 2) * dt_s  (the trainer's
clock rule, clip_clock.py; n_stack - 1 = 2).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------------------
# LITERAL THRESHOLDS -- declared before any data was looked at (road geometry).
# ----------------------------------------------------------------------------------
TURN_DEG = 30.0            # a turn: >= 30 deg heading change inside the window (brief)
TURN_FAST_DEG_3S = 30.0    # variant B: >= 30 deg inside ANY 3 s sub-window (>= 10 deg/s mean)
NUDGE_MIN_M = 0.3          # nudge: lateral excursion 0.3 ..
NUDGE_MAX_M = 1.5          #        .. 1.5 m (brief)
NUDGE_MAX_DYAW_DEG = 15.0  #        with |net dyaw| < 15 deg (brief)
NUDGE_RETURN_FRAC = 0.5    #        "returning": |excursion at window end| < 0.5 x peak
MIN_PATH_M = 5.0           # shorter path in the window = effectively stationary
BRAKE_DV = -1.5            # brake: dv <= -1.5 m/s (brief)
ACCEL_DV = +1.5            # accelerate: dv >= +1.5 m/s (brief)
STOP_V = 0.5               # stop: v < 0.5 m/s (brief)
CREEP_V = 2.0              # walking-pace ceiling (a crawl never exceeds this)
CURVE_DEG = 20.0           # "in a curve" flag for the ADAPT_SPEED_FOR_CURVE check
DT_GRID = 0.1


# ----------------------------------------------------------------------------------
def sha12(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode("utf-8")).hexdigest()[:12]


def stable_episode_id(clip_id: str) -> int:
    """tanitad/data/v2_dataset.py:69 (re-stated, blake2b-8, >>1)."""
    return int.from_bytes(hashlib.blake2b(clip_id.encode("utf-8"), digest_size=8).digest(),
                          "big") >> 1


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@dataclass
class Track:
    sid: int
    clip_id: str
    t: np.ndarray      # raw seconds per provider row
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray    # unwrapped yaw, rad
    v: np.ndarray
    clock_src: str


def pose_dt(poses: np.ndarray):
    """clip_clock.pose_dt re-stated (sum|dxy| / sum mean v over moving steps)."""
    d = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1]))
    vb = 0.5 * (poses[1:, 3] + poses[:-1, 3])
    m = (poses[1:, 3] > 2.0) & (poses[:-1, 3] > 2.0)
    if m.sum() < 10:
        return None
    dt = float(d[m].sum() / vb[m].sum())
    return dt if 0.098 <= dt <= 0.104 else None


def load_clock_sidecar(path):
    tab = {}
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                tab[int(r["sid"])] = (float(r["grid_start_s"]), float(r["dt_s"]))
    return tab


def tracks_from_manifest(manifest_path, sidecar):
    import torch
    m = torch.load(manifest_path, map_location="cpu", weights_only=False)
    out = []
    for i, cid in enumerate(m["clip_id"]):
        P = m["poses"][i].double().numpy()
        sid = stable_episode_id(cid)
        assert sid == int(m["episode_uid"][i]), "stable id mismatch vs manifest"
        if sid in sidecar:
            g0, dt = sidecar[sid]
            src = "sidecar"
        else:
            d = pose_dt(P)
            g0, dt, src = (0.0, d, "pose_dt") if d is not None else (0.0, 0.1, "nominal")
        t = g0 + (np.arange(P.shape[0]) + 2) * dt
        out.append(Track(sid, cid, t, P[:, 0], P[:, 1], np.unwrap(P[:, 2]), P[:, 3], src))
    return out


def sample(tr: Track, times):
    ok = bool(times[0] >= tr.t[0] - 1e-9 and times[-1] <= tr.t[-1] + 1e-9)
    return (np.interp(times, tr.t, tr.x), np.interp(times, tr.t, tr.y),
            np.interp(times, tr.t, tr.psi), np.interp(times, tr.t, tr.v), ok)


# ----------------------------------------------------------------------------------
def lat_features(tr: Track, ta: float, lo: float, hi: float):
    times = ta + np.arange(lo, hi + 1e-9, DT_GRID)
    x, y, psi, v, ok = sample(tr, times)
    return lat_features_arrays(x, y, psi) | {"ok": ok}


def lat_features_arrays(x, y, psi):
    dpsi = np.degrees(psi - psi[0])
    i = int(np.argmax(np.abs(dpsi)))
    k = int(round(3.0 / DT_GRID))
    if len(dpsi) > k:
        sub = dpsi[k:] - dpsi[:-k]
        fast = float(sub[int(np.argmax(np.abs(sub)))])
    else:
        fast = float(dpsi[-1] - dpsi[0])
    c, s = math.cos(psi[0]), math.sin(psi[0])
    dx, dy = x - x[0], y - y[0]
    lat = -s * dx + c * dy                       # left = +
    arcs = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    L = float(arcs[-1])
    net = math.radians(float(dpsi[-1]))
    kap = net / L if L > 1e-6 else 0.0
    if abs(kap) > 1e-9:
        lat_arc = (1.0 - np.cos(kap * arcs)) / kap       # path of constant curvature
    else:
        lat_arc = np.zeros_like(arcs)
    res = lat - lat_arc
    j = int(np.argmax(np.abs(res)))
    return {"dpsi_peak": float(dpsi[i]), "dpsi_net": float(dpsi[-1]), "dpsi_fast3": fast,
            "L": L, "eps": float(abs(res[j])), "eps_sign": 1.0 if res[j] > 0 else -1.0,
            "eps_end": float(abs(res[-1])), "lat_raw_peak": float(lat[int(np.argmax(np.abs(lat)))])}


def classify_lat(f, variant="A"):
    """variant A  : turn = |dyaw_peak| >= 30 deg (brief), spec-complete rule A1 (below).
    variant B  : turn = >= 30 deg inside any 3 s sub-window, otherwise identical to A.
    variant A0 : the FIRST-RUN rule, kept only so the disclosed first-run numbers reproduce:
                 every 0.3..1.5 m excursion that does not return, and every excursion with
                 |net dyaw| >= 15 deg, was called SHIFT (no BEND class).  Superseded by A1
                 after the first eval139 run showed it mislabels road-curvature residuals.
    A1 order of tests (all literals above):
      1 path < 5 m                       -> LANE_KEEP (stationary)
      2 turn test                        -> TURN_{L,R}
      3 |net dyaw| >= 15 deg             -> BEND_{L,R}   (monotone heading change below the turn bar)
      4 eps < 0.3 m                      -> LANE_KEEP
      5 0.3 <= eps <= 1.5 and returning  -> NUDGE_{L,R}
      6 eps > 1.5 m                      -> SHIFT_{L,R}  (lane-change scale)
      7 else (0.3..1.5 m, not returning) -> LANE_KEEP    (sub-lane drift / curvature residual)
    """
    if f["L"] < MIN_PATH_M:
        return "LANE_KEEP"
    ang = f["dpsi_fast3"] if variant == "B" else f["dpsi_peak"]
    if abs(ang) >= TURN_DEG:
        return "TURN_L" if ang > 0 else "TURN_R"
    side = "L" if f["eps_sign"] > 0 else "R"
    if variant != "A0" and abs(f["dpsi_net"]) >= NUDGE_MAX_DYAW_DEG:
        # reordered after the analytic control `bend_L_R160_v14` (a pure 20 deg arc has eps = 0 and
        # so could never reach the BEND test when it sat after the eps < 0.3 test)
        return "BEND_" + ("L" if f["dpsi_net"] > 0 else "R")
    if f["eps"] < NUDGE_MIN_M:
        return "LANE_KEEP"
    returning = f["eps_end"] < NUDGE_RETURN_FRAC * f["eps"]
    if variant == "A0":
        if f["eps"] <= NUDGE_MAX_M and returning and abs(f["dpsi_net"]) < NUDGE_MAX_DYAW_DEG:
            return "NUDGE_" + side
        return "SHIFT_" + side
    if f["eps"] <= NUDGE_MAX_M and returning:
        return "NUDGE_" + side
    if f["eps"] > NUDGE_MAX_M:
        return "SHIFT_" + side
    return "LANE_KEEP"


def lon_features(tr: Track, ta: float, lo: float, hi: float):
    times = ta + np.arange(lo, hi + 1e-9, DT_GRID)
    x, y, psi, v, ok = sample(tr, times)
    return lon_features_arrays(v) | {"ok": ok}


def lon_features_arrays(v):
    return {"vstart": float(v[0]), "vend": float(v[-1]), "vmax": float(v.max()),
            "vmin": float(v.min()), "dv": float(v[-1] - v[0])}


def classify_lon(f):
    if f["vmax"] < STOP_V:
        return "HOLD"
    if f["vmax"] <= CREEP_V:
        return "CREEP"
    if f["dv"] <= BRAKE_DV or (f["vstart"] > STOP_V and f["vmin"] < STOP_V):
        return "BRAKE"
    if f["dv"] >= ACCEL_DV:
        return "ACCEL"
    return "CRUISE"


# expected-set mapping builder-class -> acceptable independent classes
LAT_OK = {"LANE_KEEP": {"LANE_KEEP", "BEND_L", "BEND_R"}, "NUDGE_L": {"NUDGE_L", "SHIFT_L"},
          "NUDGE_R": {"NUDGE_R", "SHIFT_R"}, "TURN_L": {"TURN_L"}, "TURN_R": {"TURN_R"}}
LON_OK = {"HOLD": {"HOLD"}, "CREEP": {"CREEP"}, "ACCELERATE": {"ACCEL"},
          "BRAKE_TO": {"BRAKE"}, "CRUISE": {"CRUISE"},
          "FOLLOW": {"BRAKE", "CRUISE"},          # builder FOLLOW = mild / partial slowing
          "ADAPT_SPEED_FOR_CURVE": {"HOLD", "CREEP", "BRAKE", "CRUISE", "ACCEL"}}  # any: scored separately


def confusion(rows, cols, pairs):
    M = np.zeros((len(rows), len(cols)), dtype=int)
    ri = {r: i for i, r in enumerate(rows)}
    ci = {c: i for i, c in enumerate(cols)}
    for a, b in pairs:
        M[ri[a], ci[b]] += 1
    return M


def fmt_matrix(M, rows, cols, title=""):
    w = max(len(c) for c in cols) + 1
    lines = [title, " " * 24 + "".join(c.rjust(w) for c in cols) + "   | n"]
    for i, r in enumerate(rows):
        lines.append(r.ljust(24) + "".join(str(int(M[i, j])).rjust(w) for j in range(len(cols)))
                     + f"   | {int(M[i].sum())}")
    return "\n".join(lines)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
