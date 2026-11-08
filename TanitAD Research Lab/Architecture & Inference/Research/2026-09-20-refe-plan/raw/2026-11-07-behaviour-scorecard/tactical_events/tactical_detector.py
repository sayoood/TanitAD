"""tactical_detector.py -- executed-decision detector for AMENDMENT_10 (DRAFT), TACTICAL family, A-14.

Pure functions of numpy arrays. CPU only. Imports nothing from the simulator and nothing from
behaviour_score.py (the reference path and lead logic are re-implemented here from the scorecard
text; equality with the scorer's HumanPath/lead is an OPEN ITEM, see the draft section 11).

The SAME functions are applied to the human log and to the driven trajectory:
    events = find_events(human, scene)              # scene/human-anchored, never uses the driven run
    label  = decide(event, traj, scene, path)       # traj = human or driven
All arrays are on the 10 Hz metric grid (dt = 0.1 s); use to_10hz() on 20 Hz rollout arrays.

traj  : dict t, x, y, yaw, v (T,)   [+ optional lane_offset (T,) signed m, + left, to the route-lane centre]
scene : dict, see ROLLOUT_SCHEMA (agent_* (A,T), signal_state (T,H), signal_governs_ego (H,),
        stop_line_s, conflict_zone_xy (K,2), conflict_agent_ids, human_ref_valid_until_s,
        ego_length, ego_width, log_id/scenario_id)
"""
from __future__ import annotations

import math

import numpy as np

DECISIONS = ("go", "yield", "stop", "follow", "change-lane")
NAMED = ("AMBIG_UNRESOLVED", "AMBIG_LEAD_LOST", "AMBIG_SLOW", "AMBIG_CONTESTED",
         "AMBIG_WRONG_SIDE", "NOT_ENGAGED", "NOT_COMPUTABLE")
LABELS = DECISIONS + NAMED
CLASSES = ("LINE_R", "LINE_G", "CONFLICT", "MERGE", "LEAD", "FREE", "LANE")

# Every value: source in AMENDMENT_10 DRAFT section 9. None is chosen from a T2 outcome.
P0 = dict(
    DT=0.1, WARMUP_S=2.0,                    # scorecard s1
    STOP_V=1.0, STOP_MIN_TICKS=10,           # A-8 creep<=1.0 is stopped; STOP_MIN_S=1.0 (behaviour_score.py:51)
    NEAR_M=40.0,                             # c1 approach window, 40 m
    CORRIDOR_HALF=1.5, EGO_L=5.176, EGO_W=2.297,
    PERSIST_TICKS=3,                         # scorecard s1 persistence
    LEAD_STEADY_TICKS=30, LEAD_GAP_MAX=60.0, LEAD_WINDOW_TICKS=80,   # s3.1 steady lead: 3 s, 60 m; 8 s
    FOLLOW_FRAC=0.9, GO_FRAC=0.9,            # the scorecard's 0.90 share (a1)
    GO_V=1.0,
    FREE_MIN_TICKS=20, FREE_G_S=4.0, FREE_G_MIN=15.0, FREE_TTC_S=8.0,   # s1 free road
    ZONE_TTZ_S=2.0,                          # c3
    LANE_HALF_M=1.75,                        # AMENDMENT_5 (half a lane)
    LANE_TOL_TICKS=40,                       # ESTIMATED 4.0 s (a2 horizon)
    MERGE_OFF_START=2.0, MERGE_OFF_ENTER=0.5, MERGE_GAP=30.0, MERGE_HEAD=math.radians(30.0),  # s3.1, AMENDMENT_4 AMB-10
    MIN_CLUSTERS=10, MIN_EVENTS=20,          # ESTIMATED
    BOOT_DRAWS=10000, BOOT_SEED=20260927,    # s4
    MUT_MERGE_FLIP=False, MUT_CROSS_IGNORED=False, MUT_LANE_BEFORE_STOP=False,
)


def to_10hz(a, axis=-1):
    """20 Hz rollout array -> 10 Hz grid (every second sample)."""
    sl = [slice(None)] * np.ndim(a)
    sl[axis] = slice(0, None, 2)
    return np.asarray(a)[tuple(sl)]


# ----------------------------------------------------------------------------- geometry
def _runs(mask):
    """maximal runs of True as (i0, i1_exclusive)."""
    m = np.asarray(mask, bool)
    out, i, n = [], 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


class RefPath:
    """Reference path = the human log (scorecard s1). 0.2 m distance filter (A-7), straight extension along the
    chord of the last 5.0 m of the MOVING part (A-6)."""

    def __init__(self, x, y, v):
        x, y, v = (np.asarray(a, float) for a in (x, y, v))
        keep = [0]
        for i in range(1, len(x)):
            if math.hypot(x[i] - x[keep[-1]], y[i] - y[keep[-1]]) >= 0.2:
                keep.append(i)
        kx, ky, kv = x[keep], y[keep], v[keep]
        mv = kv >= 0.5
        mx, my = (kx[mv], ky[mv]) if mv.sum() >= 2 else (kx, ky)
        if len(mx) >= 2:
            seg = np.hypot(np.diff(mx), np.diff(my))
            cs = np.concatenate([[0.0], np.cumsum(seg)])
            j = int(np.searchsorted(cs, cs[-1] - 5.0))
            j = min(max(j, 0), len(mx) - 2)
            d = np.array([mx[-1] - mx[j], my[-1] - my[j]])
            d = d / (np.linalg.norm(d) or 1.0)
        else:
            d = np.array([1.0, 0.0])
        ex, ey = kx[-1] + 300.0 * d[0], ky[-1] + 300.0 * d[1]
        self.px = np.append(kx, ex)
        self.py = np.append(ky, ey)
        seg = np.hypot(np.diff(self.px), np.diff(self.py))
        seg = np.maximum(seg, 1e-9)
        self.seglen = seg
        self.cs = np.concatenate([[0.0], np.cumsum(seg)])
        self.tx = np.diff(self.px) / seg
        self.ty = np.diff(self.py) / seg

    def project(self, qx, qy):
        """-> station s (m), signed lateral offset (m, + left), tangent heading at the foot point."""
        qx, qy = np.atleast_1d(np.asarray(qx, float)), np.atleast_1d(np.asarray(qy, float))
        ax, ay = self.px[:-1][None, :], self.py[:-1][None, :]
        dx, dy = qx[:, None] - ax, qy[:, None] - ay
        t = np.clip(dx * self.tx[None, :] + dy * self.ty[None, :], 0.0, self.seglen[None, :])
        fx, fy = ax + t * self.tx[None, :], ay + t * self.ty[None, :]
        d2 = (qx[:, None] - fx) ** 2 + (qy[:, None] - fy) ** 2
        k = np.argmin(d2, axis=1)
        r = np.arange(len(qx))
        s = self.cs[k] + t[r, k]
        off = -(qx - fx[r, k]) * self.ty[k] + (qy - fy[r, k]) * self.tx[k]   # left of the direction of travel = +
        return s, off, np.arctan2(self.ty[k], self.tx[k])


def _box_pts(x, y, yaw, l, w):
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.array([[sx * l / 2, sy * w / 2] for sx in (-1, 0, 1) for sy in (-1, 0, 1)])
    return np.stack([x + loc[:, 0] * c - loc[:, 1] * s, y + loc[:, 0] * s + loc[:, 1] * c], 1)


def _in_poly(pts, poly):
    inside = np.zeros(len(pts), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > pts[:, 1]) != (y2 > pts[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (pts[:, 1] - y1) / (y2 - y1) + x1
        inside ^= cond & (pts[:, 0] < xint)
    return inside


def _dist_poly(pt, poly):
    if _in_poly(np.asarray([pt]), poly)[0]:
        return 0.0
    best = 1e18
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        L2 = float(ab @ ab) or 1e-12
        t = min(max(float((pt - a) @ ab) / L2, 0.0), 1.0)
        best = min(best, float(np.linalg.norm(pt - (a + t * ab))))
    return best


def _ego_in_zone(traj, i, poly, l, w):
    """footprint (9 sample points of the ego box + zone vertices inside the box) intersects the zone."""
    if np.any(_in_poly(_box_pts(traj["x"][i], traj["y"][i], traj["yaw"][i], l, w), poly)):
        return True
    c, s = math.cos(traj["yaw"][i]), math.sin(traj["yaw"][i])
    d = poly - np.array([traj["x"][i], traj["y"][i]])
    lx, ly = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    return bool(np.any((np.abs(lx) <= l / 2) & (np.abs(ly) <= w / 2)))


# ----------------------------------------------------------------------------- scene-derived series
def _ego_dims(scene, P):
    return float(scene.get("ego_length", P["EGO_L"])), float(scene.get("ego_width", P["EGO_W"]))


def agent_gaps(traj, scene, path, P):
    """-> (gap (A,T) nan where the agent is not a vehicle in the corridor ahead, v_rel (A,T)). Scalar speeds (A-3)."""
    T = len(traj["t"])
    A = len(scene.get("agent_id", []))
    gap = np.full((A, T), np.nan)
    vrel = np.full((A, T), np.nan)
    if A == 0:
        return gap, vrel
    le, _ = _ego_dims(scene, P)
    s_e, _, _ = path.project(traj["x"], traj["y"])
    front = s_e + le / 2
    cls = scene.get("agent_cls", np.zeros(A, np.int8))
    for a in range(A):
        if cls[a] != 0:
            continue
        pres = np.asarray(scene["agent_present"][a], bool)
        if not pres.any():
            continue
        xa = np.nan_to_num(scene["agent_x"][a]); ya = np.nan_to_num(scene["agent_y"][a])
        s_a, off, _ = path.project(xa, ya)
        rear = s_a - scene["agent_l"][a] / 2
        inside = np.abs(off) - scene["agent_w"][a] / 2 <= P["CORRIDOR_HALF"]
        ok = pres & inside & (rear >= front)
        gap[a, ok] = (rear - front)[ok]
        vrel[a, ok] = (traj["v"] - np.nan_to_num(scene["agent_v"][a]))[ok]
    return gap, vrel


def lead_series(traj, scene, path, P):
    """-> lead_id (T,) (-1 none), gap (T,) nan none."""
    gap, _ = agent_gaps(traj, scene, path, P)
    T = len(traj["t"])
    lid = np.full(T, -1, np.int64)
    g = np.full(T, np.nan)
    if gap.shape[0]:
        has = np.any(np.isfinite(gap), axis=0)
        k = np.nanargmin(np.where(np.isfinite(gap), gap, np.inf), axis=0)
        lid[has] = np.asarray(scene["agent_id"])[k[has]]
        g[has] = gap[k[has], np.where(has)[0]]
    return lid, g


def free_road(traj, scene, path, P, held=None):
    """scorecard s1 / A-1: no lead within max(4 s*v, 15 m) AND no lead with TTC <= 8 s; held ticks excluded."""
    gap, vrel = agent_gaps(traj, scene, path, P)
    T = len(traj["t"])
    G = np.maximum(P["FREE_G_S"] * traj["v"], P["FREE_G_MIN"])
    free = np.ones(T, bool)
    if gap.shape[0]:
        fin = np.isfinite(gap)
        near = fin & (gap <= G[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where(fin & (vrel > 0.1), gap / vrel, np.inf)
        free &= ~np.any(near, axis=0) & ~np.any(ttc <= P["FREE_TTC_S"], axis=0)
    if held is not None:
        free &= ~held
    return free


_RANK = {0: 0, 3: 1, 2: 2, 1: 3}   # none < green < yellow < red


def governing_state(scene, T):
    ss = scene.get("signal_state")
    gv = np.asarray(scene.get("signal_governs_ego", []), bool)
    if ss is None or np.size(ss) == 0 or not gv.any():
        return np.zeros(T, np.int8)
    ss = np.asarray(ss)[:, gv]
    rk = np.vectorize(_RANK.get)(ss)
    inv = {v: k for k, v in _RANK.items()}
    return np.vectorize(inv.get)(rk.max(axis=1)).astype(np.int8)


def _stopped_run(v, i0, i1, P, mask=None):
    ok = (np.asarray(v)[i0:i1] <= P["STOP_V"])
    if mask is not None:
        ok &= mask[i0:i1]
    return any((b - a) >= P["STOP_MIN_TICKS"] for a, b in _runs(ok))


def lane_runs(traj, i0, i1, P):
    """lane-departure runs: |lane_offset| > 1.75 m for >= 3 ticks (AMENDMENT_5 + s1 persistence). -> [(a, b, sign)]"""
    lo = traj.get("lane_offset")
    if lo is None:
        return None
    lo = np.asarray(lo, float)
    out = []
    for a, b in _runs((np.abs(lo[i0:i1]) > P["LANE_HALF_M"]) & np.isfinite(lo[i0:i1])):
        if b - a >= P["PERSIST_TICKS"]:
            out.append((a + i0, b + i0, int(np.sign(np.nanmean(lo[a + i0:b + i0])))))
    return out


# ----------------------------------------------------------------------------- events
def find_events(human, scene, P=P0):
    """Scene/human-anchored events. Never reads a driven trajectory. -> (events, audit).
    event = dict(cls, i0, i1 (exclusive), key, meta). audit counts everything that was dropped."""
    T = len(human["t"])
    w0 = int(round(P["WARMUP_S"] / P["DT"]))
    le, _ = _ego_dims(scene, P)
    path = RefPath(human["x"], human["y"], human["v"])
    s_h, _, _ = path.project(human["x"], human["y"])
    front_h = s_h + le / 2
    audit = dict(n_ticks=T, ref_invalid_ticks=0, sub_persistence_runs=0, phases_without_trigger=0,
                 lane_change_available=human.get("lane_offset") is not None)
    ev = []
    lim = T
    vu = scene.get("human_ref_valid_until_s")
    if vu is not None and np.isfinite(vu):
        lim = int(min(T, math.floor(float(vu) / P["DT"]) + 1))
        audit["ref_invalid_ticks"] = T - lim

    def add(cls, i0, i1, **meta):
        ev.append(dict(cls=cls, i0=int(i0), i1=int(i1), key=f"{cls}:{int(i0)}:{meta.get('agent', '')}", meta=meta))

    # --- LINE (signal-controlled stop line); station-based => limited to the valid reference range
    sl = scene.get("stop_line_s", np.nan)
    gov = governing_state(scene, T)
    has_heads = np.size(scene.get("signal_state", [])) > 0
    if has_heads and sl is not None and np.isfinite(sl):
        before = (front_h <= sl) & (sl - front_h <= P["NEAR_M"])
        for cls, in_phase in (("LINE_R", gov == 1), ("LINE_G", (gov == 0) | (gov == 3))):
            for a, b in _runs(in_phase):
                trig = [i for i in range(max(a, w0), min(b, lim)) if before[i]]
                if not trig:
                    audit["phases_without_trigger"] += 1
                    continue
                add(cls, trig[0], min(b, lim), line_s=float(sl))
    # --- CONFLICT (c3 zone)
    zone = np.asarray(scene.get("conflict_zone_xy", np.zeros((0, 2))), float)
    A = len(scene.get("agent_id", []))
    if len(zone) >= 3 and A:
        ids = np.asarray(scene["agent_id"])
        cset = np.asarray(scene.get("conflict_agent_ids", []), np.int64)
        sel = [a for a in range(A) if (len(cset) == 0 or ids[a] in cset)]
        cond = np.zeros(T, bool)
        for i in range(T):
            for a in sel:
                if not scene["agent_present"][a][i]:
                    continue
                d = _dist_poly(np.array([scene["agent_x"][a][i], scene["agent_y"][a][i]]), zone)
                if d / max(float(scene["agent_v"][a][i]), 0.1) <= P["ZONE_TTZ_S"]:
                    cond[i] = True
                    break
        cond[:w0] = False
        for a, b in _runs(cond):
            if b - a >= P["PERSIST_TICKS"]:
                add("CONFLICT", a, b)
            else:
                audit["sub_persistence_runs"] += 1
    # --- station-based classes below are cut at the valid reference range
    lid, gap = lead_series(human, scene, path, P)
    held = np.zeros(T, bool)
    if has_heads and sl is not None and np.isfinite(sl):
        held = (gov == 1) & (sl - front_h >= 0) & (sl - front_h <= P["NEAR_M"])
    # --- MERGE / cut-in (s3.1 predicate, AMENDMENT_4 AMB-10)
    for a in range(A):
        pres = np.asarray(scene["agent_present"][a], bool)
        if scene.get("agent_cls", np.zeros(A, np.int8))[a] != 0 or pres.sum() < 2:
            continue
        xa, ya = np.nan_to_num(scene["agent_x"][a]), np.nan_to_num(scene["agent_y"][a])
        s_a, off, hd = path.project(xa, ya)
        idx = np.where(pres)[0]
        idx = idx[idx >= w0]
        if len(idx) == 0 or abs(off[idx[0]]) < P["MERGE_OFF_START"]:
            continue
        for i in idx:
            if i >= lim:
                break
            if abs(off[i]) <= P["MERGE_OFF_ENTER"]:
                dh = abs(math.atan2(math.sin(scene["agent_yaw"][a][i] - hd[i]), math.cos(scene["agent_yaw"][a][i] - hd[i])))
                gp = (s_a[i] - scene["agent_l"][a] / 2) - front_h[i]
                if dh <= P["MERGE_HEAD"] and -1e-9 <= gp <= P["MERGE_GAP"]:
                    add("MERGE", i, i + 1, agent=int(scene["agent_id"][a]), agent_idx=a)
                break
    # --- LEAD: same lead id continuously >= 3 s, gap <= 60 m; window = first 8 s of the run
    i = w0
    while i < lim:
        if lid[i] >= 0 and gap[i] <= P["LEAD_GAP_MAX"]:
            j = i
            while j < lim and lid[j] == lid[i] and np.isfinite(gap[j]) and gap[j] <= P["LEAD_GAP_MAX"]:
                j += 1
            if j - i >= P["LEAD_STEADY_TICKS"]:
                add("LEAD", i, min(j, i + P["LEAD_WINDOW_TICKS"]), agent=int(lid[i]))
            else:
                audit["sub_persistence_runs"] += 1
            i = j
        else:
            i += 1
    # --- FREE road intervals >= 2.0 s
    free = free_road(human, scene, path, P, held)
    free[:w0] = False
    free[lim:] = False
    for a, b in _runs(free):
        if b - a >= P["FREE_MIN_TICKS"]:
            add("FREE", a, b)
    # --- LANE: human-anchored lane departures (needs lane_offset)
    lr = lane_runs(human, 0, T, P)
    for a, b, sg in (lr or []):
        add("LANE", max(w0, a - P["LANE_TOL_TICKS"]), min(T, b + P["LANE_TOL_TICKS"]), sign=sg, run=(a, b))
    ev.sort(key=lambda e: (e["i0"], e["cls"]))
    return ev, audit


# ----------------------------------------------------------------------------- decisions
def _generic(traj, i0, i1, scene, path, P, lead_frac_any=None):
    """STOP > FOLLOW (any lead on >= 90 % of ticks) > GO (v > 1 on >= 90 %) > AMBIG_SLOW."""
    if _stopped_run(traj["v"], i0, i1, P):
        return "stop"
    if lead_frac_any is None:
        _, g = lead_series(traj, scene, path, P)
        lead_frac_any = float(np.mean(np.isfinite(g[i0:i1]))) if i1 > i0 else 0.0
    if lead_frac_any >= P["FOLLOW_FRAC"]:
        return "follow"
    if float(np.mean(traj["v"][i0:i1] > P["GO_V"])) >= P["GO_FRAC"]:
        return "go"
    return "AMBIG_SLOW"


def decide(event, traj, scene, P=P0, path=None, ref_path=None):
    """closed-form decision of ONE trajectory at ONE event. path = RefPath of the HUMAN log (shared by both runs)."""
    path = path or ref_path
    le, we = _ego_dims(scene, P)
    cls, i0, i1 = event["cls"], event["i0"], event["i1"]
    T = len(traj["t"])
    i1 = min(i1, T)
    if cls in ("LINE_R", "LINE_G"):
        sl = event["meta"]["line_s"]
        s, _, _ = path.project(traj["x"], traj["y"])
        front = s + le / 2
        crossed = bool(np.any(front[i0:i1] > sl)) and not P["MUT_CROSS_IGNORED"]
        engaged = bool(np.any(sl - front[i0:i1] <= P["NEAR_M"]))
        if not engaged:
            return "NOT_ENGAGED"
        if crossed:
            return "go"
        if _stopped_run(traj["v"], i0, i1, P, mask=front <= sl):
            return "stop"
        return "AMBIG_UNRESOLVED"
    if cls == "CONFLICT":
        zone = np.asarray(scene["conflict_zone_xy"], float)
        eng = False
        for i in range(i0, i1):
            if _ego_in_zone(traj, i, zone, le, we):
                return "go"
            if _dist_poly(np.array([traj["x"][i], traj["y"][i]]), zone) <= P["NEAR_M"]:
                eng = True
        return "yield" if eng else "NOT_ENGAGED"
    if cls == "MERGE":
        a = event["meta"]["agent_idx"]
        i = i0
        s_e, _, _ = path.project(traj["x"][i:i + 1], traj["y"][i:i + 1])
        s_a, _, _ = path.project(np.array([scene["agent_x"][a][i]]), np.array([scene["agent_y"][a][i]]))
        d = float(s_a[0] - s_e[0])
        half = (scene["agent_l"][a] + le) / 2
        if abs(d) > P["NEAR_M"]:
            return "NOT_ENGAGED"
        if P["MUT_MERGE_FLIP"]:
            d = -d
        return "yield" if d >= half else ("go" if d <= -half else "AMBIG_CONTESTED")
    lr = lane_runs(traj, i0, i1, P)
    if cls == "LEAD":
        lid, g = lead_series(traj, scene, path, P)
        ag = event["meta"]["agent"]
        frac = float(np.mean((lid[i0:i1] == ag) & np.isfinite(g[i0:i1])))
        stop = _stopped_run(traj["v"], i0, i1, P)
        if P["MUT_LANE_BEFORE_STOP"] and lr:
            return "change-lane"
        if stop:
            return "stop"
        if lr:
            return "change-lane"
        return "follow" if frac >= P["FOLLOW_FRAC"] else "AMBIG_LEAD_LOST"
    if cls == "FREE":
        if _stopped_run(traj["v"], i0, i1, P):
            return "stop"
        if lr:
            return "change-lane"
        return "go" if float(np.mean(traj["v"][i0:i1] > P["GO_V"])) >= P["GO_FRAC"] else "AMBIG_SLOW"
    if cls == "LANE":
        if lr is None:
            return "NOT_COMPUTABLE"
        sg = event["meta"]["sign"]
        same = [r for r in lr if r[2] == sg]
        if same:
            return "change-lane"
        if lr:
            return "AMBIG_WRONG_SIDE"
        return _generic(traj, i0, i1, scene, path, P)
    raise ValueError(cls)


def score_scenario(human, driven, scene, P=P0):
    """-> (rows, audit). One row per event: human label and driven label from the SAME decide()."""
    ev, audit = find_events(human, scene, P)
    path = RefPath(human["x"], human["y"], human["v"])
    rows = []
    cid = str(scene.get("log_id", scene.get("scenario_id", "scn")))
    for e in ev:
        rows.append(dict(cluster=cid, key=cid + "|" + e["key"], cls=e["cls"],
                         human=decide(e, human, scene, P, path), driven=decide(e, driven, scene, P, path)))
    return rows, audit


# ----------------------------------------------------------------------------- confusion table and estimators
def confusion(rows, cls=None):
    """-> dict(labels, counts[h][d], n, n_resolved, agree, n_diag)."""
    r = [x for x in rows if cls is None or x["cls"] == cls]
    M = np.zeros((len(LABELS), len(LABELS)), int)
    for x in r:
        M[LABELS.index(x["human"]), LABELS.index(x["driven"])] += 1
    k = len(DECISIONS)
    nres = int(M[:k, :k].sum())
    nd = int(np.trace(M[:k, :k]))
    return dict(labels=list(LABELS), counts=M.tolist(), n=len(r), n_resolved=nres, n_diag=nd,
                agreement=(nd / nres if nres else None),
                n_human_named=int(M[k:, :].sum()), n_driven_named_given_human_resolved=int(M[:k, k:].sum()))


def _cluster_stats(rows):
    d = {}
    k = len(DECISIONS)
    for x in rows:
        c = d.setdefault(x["cluster"], [0, 0])
        if x["human"] in DECISIONS and x["driven"] in DECISIONS:
            c[1] += 1
            c[0] += int(x["human"] == x["driven"])
    return d


def bootstrap_agreement(rows, P=P0):
    cs = _cluster_stats(rows)
    keys = [k for k, v in cs.items() if v[1] > 0]
    if len(keys) < P["MIN_CLUSTERS"]:
        return None
    arr = np.array([cs[k] for k in keys], float)
    rng = np.random.default_rng(P["BOOT_SEED"])
    idx = rng.integers(0, len(arr), size=(P["BOOT_DRAWS"], len(arr)))
    num, den = arr[idx, 0].sum(1), arr[idx, 1].sum(1)
    est = num / np.maximum(den, 1)
    return dict(lo=float(np.percentile(est, 2.5)), hi=float(np.percentile(est, 97.5)), n_clusters=len(keys),
                estimator="episode-cluster bootstrap (cluster = log/scenario), 10,000 draws, seed 20260927, 95 % percentile")


def paired_bootstrap_diff(rows_cand, rows_base, P=P0):
    """agreement(cand) - agreement(base), paired on event key, cluster = scenario. None if too few clusters."""
    kb = {r["key"] for r in rows_base}
    kc = {r["key"] for r in rows_cand}
    common = kb & kc
    c = _cluster_stats([r for r in rows_cand if r["key"] in common])
    b = _cluster_stats([r for r in rows_base if r["key"] in common])
    keys = [k for k in c if c[k][1] > 0 or b.get(k, [0, 0])[1] > 0]
    if len(keys) < P["MIN_CLUSTERS"]:
        return dict(status="INSUFFICIENT_N", n_clusters=len(keys), n_unpaired=len(kb ^ kc))
    arr = np.array([c[k] + b.get(k, [0, 0]) for k in keys], float)
    rng = np.random.default_rng(P["BOOT_SEED"])
    idx = rng.integers(0, len(arr), size=(P["BOOT_DRAWS"], len(arr)))
    s = arr[idx].sum(1)
    est = s[:, 0] / np.maximum(s[:, 1], 1) - s[:, 2] / np.maximum(s[:, 3], 1)
    pt = arr[:, 0].sum() / max(arr[:, 1].sum(), 1) - arr[:, 2].sum() / max(arr[:, 3].sum(), 1)
    return dict(status="OK", point=float(pt), lo=float(np.percentile(est, 2.5)), hi=float(np.percentile(est, 97.5)),
                n_clusters=len(keys), n_unpaired=len(kb ^ kc),
                estimator="paired episode-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile")


def report(rows, audits=None, P=P0):
    """Per-class tables (primary), pooled table (descriptive), and the status. Never a pass/fail."""
    out = dict(family="TACTICAL.executed_decision", classes={}, pooled=confusion(rows))
    for c in CLASSES:
        t = confusion(rows, c)
        if t["n"] == 0:
            continue
        t["interval"] = bootstrap_agreement([r for r in rows if r["cls"] == c], P) if t["n_resolved"] >= P["MIN_EVENTS"] else None
        t["status"] = "OK" if t["interval"] is not None else "INSUFFICIENT_N (counts only; no rate verdict)"
        out["classes"][c] = t
    if not rows:
        out["status"] = "NO_EVENTS (family N/A for this suite; reason and n=0 reported)"
    elif any(v["status"] == "OK" for v in out["classes"].values()):
        out["status"] = "OK_PARTIAL" if any(v["status"] != "OK" for v in out["classes"].values()) else "OK"
    else:
        out["status"] = "INSUFFICIENT_N"
    out["audit"] = audits or []
    return out


# ----------------------------------------------------------------------------- loader for rollout.npz / human.npz
def from_rollout(rollout, human, with_lane_offset=None):
    """Build (human_traj, driven_traj, scene) on the 10 Hz grid from the two schema dicts (np.load results).
    lane_offset is read from key 'lane_offset' of each file when present (ASSUMED field, open item)."""
    def tr(d):
        out = {k: to_10hz(d[k]) for k in ("t", "ego_x", "ego_y", "ego_yaw", "ego_v")}
        o = dict(t=out["t"], x=out["ego_x"], y=out["ego_y"], yaw=out["ego_yaw"], v=out["ego_v"])
        if "lane_offset" in d:
            o["lane_offset"] = to_10hz(d["lane_offset"])
        return o
    src = rollout
    sc = dict(scenario_id=str(rollout.get("scenario_id", "scn")), log_id=str(rollout.get("log_id", rollout.get("scenario_id", "scn"))))
    for k in ("agent_id", "agent_l", "agent_w", "agent_cls", "ego_length", "ego_width", "conflict_agent_ids",
              "signal_governs_ego", "stop_line_s", "conflict_zone_xy", "human_ref_valid_until_s"):
        if k in src:
            sc[k] = src[k]
    for k in ("agent_present", "agent_x", "agent_y", "agent_yaw", "agent_v"):
        if k in src:
            sc[k] = to_10hz(src[k], axis=-1)
    if "signal_state" in src:
        sc["signal_state"] = to_10hz(src["signal_state"], axis=0)
    return tr(human), tr(rollout), sc
