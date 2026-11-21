"""leadinfo_d.py -- PREREG_LEADINFO_D (sealed 2026-10-26 07:46): is there decel information in the GT LEAD AGENT that refcv8's lon head
does not carry?  DIAGNOSTIC on logged eval139 labels; no planner, no bar, GT features are an UPPER BOUND (never an inference input).

Order of work (every stage asserts before the next one runs):
  0. the prereg sha256 must equal the first token of PREREG_LEADINFO_D.sha256, else STOP;
  1. SYNTHETIC controls through the PRODUCTION code path (agent at (20,0) -> range 20 / lateral 0; negatives; a moving track; a rotated frame);
  2. eval139 S6 compliant windows + the two decel populations from l2_common.lon_indicators (the battery's definition, imported);
  3. the B1 join for the 139 eval clips (streamed once, cached on F:), keyed by (episode, ws) -- ws IS the NOW frame
     (refc_v3_train.py:4635 `_agent_item(ep, t + w - 1)`; manifest `ws_is`), the ego poses from the dump's decisions/ep_poses;
  4. alignment controls on REAL data (pose<->join frame sweep, pose speed == v0, lead-range-vs-ego-1s-travel count);
  5. features A / B / C, 5-fold episode-grouped cross-fit, AUC, episode-resampling bootstrap (B 2000, seed 0), the word.

Run:  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 python -X utf8 leadinfo_d.py          (PYTHONPATH as in the brief)
No clip id (UUID / hex prefix) is ever printed or written; windows are identified by (episode index, ws).
"""
from __future__ import annotations

import hashlib
import json
import lzma
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path("C:/Users/Admin/refcv8_pf/refcv9_leadinfo")
RES = HERE / "result"
SCR = Path("F:/refcv9_leadinfo_scratch")
PREREG = HERE / "PREREG_LEADINFO_D.md"
PREREG_SHA = HERE / "PREREG_LEADINFO_D.sha256"
JOIN_XZ = Path("E:/refcv6_eval_kit/data/joins/b1_train_plus_eval_agents.jsonl.xz")
JOIN_XZ_PREREG_PATH = "C:/Users/Admin/a40-rescue/b1_train_plus_eval_agents.jsonl.xz"
JOIN_XZ_ALT = Path("E:/Projects/TanitAD-artifacts/a40-rescue/b1_train_plus_eval_agents.jsonl.xz")
ARCH = Path("C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf")
SEED = 0
# ---- the prereg constants (section 2 / 3) -----------------------------------------------------------------------------------
CORR_HALF_M = 1.8       # |y| <= 1.8 m
RANGE_MAX_M = 80.0      # 0 < x <= 80 m
PAST_S = 1.0            # past window of the finite difference
N_FOLDS = 5
LOGREG_C = 1.0
BOOT_B = 2000
DELTA_BAR = 0.03


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(*a, flush=True)


# =====================================================================================================================
# 0. the seal
# =====================================================================================================================
def check_seal() -> dict:
    want = PREREG_SHA.read_text(encoding="utf-8").split()[0].strip().lower()
    got = sha256_file(PREREG)
    if got != want or not want.startswith("8f02a6dd"):
        raise SystemExit(f"STOP: prereg sha256 {got} != sealed {want}")
    return {"prereg_sha256": got, "sealed_equal": True}


# =====================================================================================================================
# 1. the lead extractor -- ONE implementation, used by the synthetic controls AND the real data
# =====================================================================================================================
class Frame:
    """One join record: agents in the EGO frame of that frame (+x fwd, +y left)."""
    __slots__ = ("cx", "cy", "l", "w", "occ", "tid", "t_s")

    def __init__(self, cx, cy, l, w, occ, tid, t_s):
        self.cx = np.asarray(cx, np.float64)
        self.cy = np.asarray(cy, np.float64)
        self.l = np.asarray(l, np.float64)
        self.w = np.asarray(w, np.float64)
        self.occ = np.asarray(occ, np.int64)
        self.tid = np.asarray(tid, dtype=object)
        self.t_s = float(t_s)


def frame_from_record(r: dict) -> Frame:
    ag = r["agents"]
    return Frame([a["cx"] for a in ag], [a["cy"] for a in ag], [a["l"] for a in ag], [a["w"] for a in ag],
                 [a.get("occ", 0) for a in ag], [str(a["track_id"]) for a in ag], r["t_s"])


class Episode:
    """frames: {frame_idx: Frame}  (absent key == NO_LABEL, never 'road clear');  poses: [N, >=3] (x, y, heading, ...) indexed by frame_idx."""

    def __init__(self, frames: dict, poses: np.ndarray):
        self.frames = frames
        self.poses = np.asarray(poses, np.float64)

    def frame(self, f: int):
        return self.frames.get(int(f))

    def pose(self, f: int):
        f = int(f)
        if f < 0 or f >= len(self.poses):
            return None
        return self.poses[f, :3]


def to_world(pose, cx, cy):
    x, y, h = pose
    c, s = np.cos(h), np.sin(h)
    return np.array([x + c * cx - s * cy, y + s * cx + c * cy])


LEAD_COLS = ("flag", "range_x", "lat_y", "range_rate", "lead_speed", "time_gap")


def lead_block(ep: Episode, ws: int, v0: float) -> dict | None:
    """The B lead block at the NOW frame ``ws`` (causal: frames <= ws only).  None == the NOW frame has no join record (NO_LABEL).

    corridor : the nearest (smallest x = cx) agent with 0 < cx <= 80 m and |cy| <= 1.8 m, in the ego frame of frame ws.
    range_x  : its cx;  lat_y: its cy.
    range_rate : (r(ws) - r(ws-k)) / (t_s(ws) - t_s(ws-k)),  r = hypot(cx, cy) of the SAME track_id (Euclidean range, frame-rotation
               invariant), k = 10 frames (1.0 s); if that frame / track is absent, the OLDEST frame k in 9..3 that carries the track
               (>= 0.3 s baseline).  No usable history -> rate 0 and ``hist_ok`` False (counted, reported).
    lead_speed : |world displacement of the track| / dt over the same interval (ego world poses from the dump's ep_poses);
               no usable history -> v0 (== rate 0, the prereg's no-rate default).  No lead -> 0.
    time_gap : range_x / max(v0, 1).   No lead: flag 0, range 80, lat 0, rate 0, speed 0 (the prereg's defaults), gap = 80 / max(v0, 1).
    """
    cur = ep.frame(ws)
    if cur is None:
        return None
    out = dict(flag=0.0, range_x=RANGE_MAX_M, lat_y=0.0, range_rate=0.0, lead_speed=0.0, time_gap=RANGE_MAX_M / max(float(v0), 1.0),
               hist_ok=False, hist_k=0, occ=0, length=0.0, dup=0)
    m = (cur.cx > 0.0) & (cur.cx <= RANGE_MAX_M) & (np.abs(cur.cy) <= CORR_HALF_M)
    if not m.any():
        return out
    idx = np.flatnonzero(m)
    j = int(idx[np.argmin(cur.cx[idx])])
    x, y = float(cur.cx[j]), float(cur.cy[j])
    out.update(flag=1.0, range_x=x, lat_y=y, time_gap=x / max(float(v0), 1.0), occ=int(cur.occ[j]), length=float(cur.l[j]),
               lead_speed=float(v0))
    tid = cur.tid[j]
    if np.sum(cur.tid == tid) != 1:
        out["dup"] = 1
        return out
    for k in [int(round(PAST_S * 10))] + list(range(int(round(PAST_S * 10)) - 1, 2, -1)):
        pr = ep.frame(ws - k)
        if pr is None:
            continue
        hit = np.flatnonzero(pr.tid == tid)
        if len(hit) != 1:
            continue
        i = int(hit[0])
        dt = cur.t_s - pr.t_s
        if dt <= 0.2:
            continue
        r_now, r_old = float(np.hypot(x, y)), float(np.hypot(pr.cx[i], pr.cy[i]))
        out["range_rate"] = (r_now - r_old) / dt
        p_now, p_old = ep.pose(ws), ep.pose(ws - k)
        if p_now is not None and p_old is not None:
            w_now, w_old = to_world(p_now, x, y), to_world(p_old, pr.cx[i], pr.cy[i])
            out["lead_speed"] = float(np.hypot(*(w_now - w_old)) / dt)
        else:
            out["lead_speed"] = float(v0 + out["range_rate"])
        out["hist_ok"], out["hist_k"] = True, k
        break
    return out


def synthetic_controls() -> dict:
    """The prereg's required control (agent at (20,0) must be found at range 20, lateral 0) plus negatives, run through lead_block()."""
    res, ok = {}, True

    def chk(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        res[name] = {"pass": bool(cond), "detail": detail}

    def ep_with(agents, past=None, ego_speed=0.0, n=40, ws=30, heading_rate=0.0):
        """agents: [(cx, cy, tid)] at ws;  past: {k: [(cx, cy, tid)]} at ws-k.  Ego moves straight at ego_speed (m/s), 0.1 s frames."""
        poses = np.zeros((n, 4))
        for f in range(n):
            poses[f] = [ego_speed * 0.1 * f, 0.0, heading_rate * 0.1 * f, ego_speed]
        frames = {}
        def mk(f, ag):
            frames[f] = Frame([a[0] for a in ag], [a[1] for a in ag], [4.5] * len(ag), [1.8] * len(ag), [0] * len(ag), [a[2] for a in ag], 0.1 * f)
        mk(ws, agents)
        for k, ag in (past or {}).items():
            mk(ws - k, ag)
        return Episode(frames, poses)

    # (1) THE required control: an agent at (20, 0) m, ego frame
    b = lead_block(ep_with([(20.0, 0.0, "a")]), 30, 10.0)
    chk("found_at_20_0", b["flag"] == 1.0 and abs(b["range_x"] - 20.0) < 1e-12 and abs(b["lat_y"]) < 1e-12 and abs(b["time_gap"] - 2.0) < 1e-12,
        {k: b[k] for k in ("flag", "range_x", "lat_y", "time_gap")})
    # (2) negatives: behind, outside corridor, beyond 80 m, exactly on the boundaries
    for nm, ag, want in (("behind_(-20,0)", [(-20.0, 0.0, "a")], 0.0), ("lateral_(20,3.0)", [(20.0, 3.0, "a")], 0.0),
                         ("beyond_(90,0)", [(90.0, 0.0, "a")], 0.0), ("x_zero_(0,0)", [(0.0, 0.0, "a")], 0.0),
                         ("edge_y_(20,1.8)", [(20.0, 1.8, "a")], 1.0), ("edge_y_(20,-1.8)", [(20.0, -1.8, "a")], 1.0),
                         ("edge_x_(80,0)", [(80.0, 0.0, "a")], 1.0), ("over_edge_y_(20,1.81)", [(20.0, 1.81, "a")], 0.0)):
        b = lead_block(ep_with(ag), 30, 10.0)
        chk(nm, b["flag"] == want, {"flag": b["flag"], "want": want})
    # (3) two agents in the corridor -> the nearest; an outside-corridor nearer agent is ignored
    b = lead_block(ep_with([(30.0, 0.0, "far"), (20.0, 1.0, "near"), (12.0, 4.0, "out")]), 30, 10.0)
    chk("nearest_in_corridor", b["flag"] == 1.0 and b["range_x"] == 20.0 and b["lat_y"] == 1.0, {"range_x": b["range_x"], "lat_y": b["lat_y"]})
    # (4) no lead -> prereg defaults
    b = lead_block(ep_with([]), 30, 10.0)
    chk("no_lead_defaults", b["flag"] == 0.0 and b["range_x"] == 80.0 and b["range_rate"] == 0.0 and b["lat_y"] == 0.0,
        {k: b[k] for k in ("flag", "range_x", "range_rate", "time_gap")})
    # (5) NO_LABEL is None, not 'road clear'
    chk("no_label_is_none", lead_block(Episode({}, np.zeros((40, 4))), 30, 10.0) is None, {})
    # (6) a moving lead: ego 10 m/s, lead 8 m/s world, 20 m ahead now -> 22 m ahead 1 s ago: rate -2.0, lead speed 8.0
    b = lead_block(ep_with([(20.0, 0.0, "a")], past={10: [(22.0, 0.0, "a")]}, ego_speed=10.0), 30, 10.0)
    chk("moving_lead_rate_-2_speed_8", b["hist_ok"] and abs(b["range_rate"] + 2.0) < 1e-9 and abs(b["lead_speed"] - 8.0) < 1e-9 and b["hist_k"] == 10,
        {k: b[k] for k in ("range_rate", "lead_speed", "hist_k")})
    # (7) a STATIONARY lead seen from a 10 m/s ego: ego-frame x shrinks 1 m / frame -> rate -10, world speed 0
    b = lead_block(ep_with([(20.0, 0.0, "a")], past={10: [(30.0, 0.0, "a")]}, ego_speed=10.0), 30, 10.0)
    chk("stationary_lead_rate_-10_speed_0", abs(b["range_rate"] + 10.0) < 1e-9 and abs(b["lead_speed"]) < 1e-9,
        {k: b[k] for k in ("range_rate", "lead_speed")})
    # (8) the same stationary lead from a TURNING ego (0.2 rad/s): world speed must still be ~0 (the ego-frame rotation is undone by the poses)
    ego_speed, hr, ws, n = 10.0, 0.2, 30, 40
    ep = ep_with([], ego_speed=ego_speed, heading_rate=hr)
    # build the poses of a circle-ish ego path consistently: x,y by integrating heading
    P = np.zeros((n, 4)); P[0, 3] = ego_speed
    for f in range(1, n):
        h = hr * 0.1 * (f - 1)
        P[f, 0] = P[f - 1, 0] + ego_speed * 0.1 * np.cos(h); P[f, 1] = P[f - 1, 1] + ego_speed * 0.1 * np.sin(h); P[f, 2] = hr * 0.1 * f; P[f, 3] = ego_speed
    Wp = np.array([P[ws, 0] + 20.0 * np.cos(P[ws, 2]), P[ws, 1] + 20.0 * np.sin(P[ws, 2])])    # a world-fixed point 20 m dead ahead of the ego at ws
    def inego(f):
        d = Wp - P[f, :2]; c, s = np.cos(-P[f, 2]), np.sin(-P[f, 2]); return (c * d[0] - s * d[1], s * d[0] + c * d[1])
    frames = {}
    for f in (ws, ws - 10):
        cx, cy = inego(f); frames[f] = Frame([cx], [cy], [4.5], [1.8], [0], ["a"], 0.1 * f)
    b = lead_block(Episode(frames, P), ws, ego_speed)
    chk("stationary_lead_turning_ego_world_speed_0", b["flag"] == 1.0 and b["hist_ok"] and abs(b["lead_speed"]) < 1e-9, {k: b[k] for k in ("flag", "range_x", "lat_y", "range_rate", "lead_speed")})
    # (9) history fallback: no frame ws-10, track present at ws-7 only -> k = 7;  no history at all -> rate 0, speed = v0, hist_ok False
    b = lead_block(ep_with([(20.0, 0.0, "a")], past={7: [(21.4, 0.0, "a")]}, ego_speed=10.0), 30, 10.0)
    chk("history_fallback_k7", b["hist_ok"] and b["hist_k"] == 7, {k: b[k] for k in ("hist_ok", "hist_k", "range_rate")})
    b = lead_block(ep_with([(20.0, 0.0, "a")], past={10: [(22.0, 0.0, "other")]}, ego_speed=10.0), 30, 10.0)
    chk("track_absent_no_history", (not b["hist_ok"]) and b["range_rate"] == 0.0 and b["lead_speed"] == 10.0, {k: b[k] for k in ("hist_ok", "range_rate", "lead_speed")})
    # (10) causality: a record at ws+5 with a closer lead must NOT change the answer at ws
    e1 = ep_with([(20.0, 0.0, "a")], past={10: [(22.0, 0.0, "a")]}, ego_speed=10.0)
    e2 = ep_with([(20.0, 0.0, "a")], past={10: [(22.0, 0.0, "a")]}, ego_speed=10.0)
    e2.frames[35] = Frame([5.0], [0.0], [4.5], [1.8], [0], ["z"], 3.5)
    chk("causal_future_frame_ignored", lead_block(e1, 30, 10.0) == lead_block(e2, 30, 10.0), {})
    return {"all_pass": bool(ok), "n": len(res), "cases": res}


# =====================================================================================================================
# 2./3. data
# =====================================================================================================================
def load_windows():
    sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
    import l2_common as C
    E = C.load_eval_pf(0)
    comp = np.asarray(E["compliant"], bool)
    # the battery's own definition; `os` is a stand-in plan -- only the GT class (4th return) and the population masks are used
    fd, md, gt, _pc = C.lon_indicators(E["os"], E, comp)
    pop_cruise, pop_decel = np.isfinite(fd), np.isfinite(md)
    assert not (pop_cruise & pop_decel).any()
    return C, E, comp, gt, pop_cruise, pop_decel


def stream_join(episodes_meta) -> tuple[dict, dict]:
    """Stream the B1 join once, keep the 139 eval clips (clip_id is the first key of every line), cache on F:."""
    cache = SCR / "join_eval139.pkl"
    want = {e["clip_id"]: e["episode_index"] for e in episodes_meta}
    uid = lambda cid: int.from_bytes(hashlib.blake2b(cid.encode("utf-8"), digest_size=8).digest(), "big") >> 1
    assert all(uid(e["clip_id"]) == e["episode_id"] for e in episodes_meta), "stable-id rule does not reproduce the manifest episode_id"
    meta = {"join_path": str(JOIN_XZ), "join_sha256": sha256_file(JOIN_XZ), "prereg_named_path": JOIN_XZ_PREREG_PATH,
            "prereg_named_path_exists": os.path.exists(JOIN_XZ_PREREG_PATH), "alt_copy": str(JOIN_XZ_ALT),
            "alt_copy_sha256": sha256_file(JOIN_XZ_ALT) if JOIN_XZ_ALT.exists() else None}
    if cache.exists():
        recs = pickle.load(open(cache, "rb"))
        meta["cache"] = "reused"
    else:
        recs, n_lines = {}, 0
        with lzma.open(JOIN_XZ, "rt", encoding="utf-8") as fh:
            for line in fh:
                n_lines += 1
                ei = want.get(line[12:48])
                if ei is None:
                    continue
                r = json.loads(line)
                assert r["clip_id"] in want
                recs.setdefault(ei, {})[int(r["frame_idx"])] = r
        pickle.dump(recs, open(cache, "wb"))
        meta["cache"] = f"built ({n_lines} join lines streamed)"
    meta["episodes_in_join"] = len(recs)
    meta["records_kept"] = int(sum(len(v) for v in recs.values()))
    return recs, meta


def load_poses(E, n_ep: int) -> dict:
    out = {}
    for ei in range(n_ep):
        d = np.load(ARCH / "dump_s0" / "decisions" / f"ep{ei:03d}.npz")
        out[ei] = (np.asarray(d["ep_poses"], np.float64), np.asarray(d["ws"], np.int64))
    return out


# =====================================================================================================================
# 4. real-data alignment controls
# =====================================================================================================================
def pose_join_shift_sweep(recs: dict, poses: dict, shifts=range(-3, 4), min_len=30) -> dict:
    """Which pose index f+s must transform the ego-frame agents of join frame f into the world frame?  A STATIONARY (parked) track has a
    constant world position only under the right shift; any other shift injects (x[f+s] - x[f]) ~ s * v * 0.1 m, which varies with the ego
    speed.  Metric: for tracks present >= min_len frames, the world-position std; report the 10th / 25th percentile across tracks (the
    stationary mass) per shift."""
    res = {int(s): [] for s in shifts}
    for ei, fr in recs.items():
        P, _ = poses[ei]
        by_tid: dict[str, list] = {}
        for f, r in fr.items():
            if f < 0 or f >= len(P):
                continue
            for a in r["agents"]:
                by_tid.setdefault(a["track_id"], []).append((f, a["cx"], a["cy"]))
        for tid, rows in by_tid.items():
            if len(rows) < min_len:
                continue
            arr = np.array(rows, np.float64)
            f = arr[:, 0].astype(int)
            for s in shifts:
                g = np.clip(f + s, 0, len(P) - 1)
                c, sn = np.cos(P[g, 2]), np.sin(P[g, 2])
                wx = P[g, 0] + c * arr[:, 1] - sn * arr[:, 2]
                wy = P[g, 1] + sn * arr[:, 1] + c * arr[:, 2]
                res[int(s)].append(float(np.sqrt(np.var(wx) + np.var(wy))))
    out = {}
    for s, v in res.items():
        v = np.asarray(v)
        out[s] = {"n_tracks": int(len(v)), "p10_std_m": float(np.percentile(v, 10)), "p25_std_m": float(np.percentile(v, 25)),
                  "min_std_m": float(v.min())}
    best = min(out, key=lambda s: out[s]["p10_std_m"])
    return {"shifts": out, "best_shift_by_p10": int(best), "aligned_at_zero": bool(best == 0)}


# =====================================================================================================================
# 5. estimator
# =====================================================================================================================
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def auc_np(y, s):
    """Mann-Whitney AUC with average ranks for ties."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class WAuc:
    """Weighted AUC for the episode-resampling bootstrap: window weight = the number of times its episode was drawn.  Scores are fixed
    (the out-of-fold predictions); only the weights move.  Ties get 1/2."""

    def __init__(self, y, s):
        self.y = np.asarray(y, bool)
        _, self.g = np.unique(np.asarray(s, np.float64), return_inverse=True)
        self.ng = int(self.g.max()) + 1

    def __call__(self, w):
        pos = np.bincount(self.g, weights=w * self.y, minlength=self.ng)
        neg = np.bincount(self.g, weights=w * (~self.y), minlength=self.ng)
        below = np.cumsum(neg) - neg
        return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def cross_fit(X, y, ep_idx, folds, C=LOGREG_C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    oof = np.full(len(y), np.nan)
    for k in range(N_FOLDS):
        te = folds == k
        tr = ~te
        assert not (set(ep_idx[tr]) & set(ep_idx[te])), "episode leaked across folds"
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=5000, solver="lbfgs").fit(sc.transform(X[tr]), y[tr])   # default penalty == L2
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    assert np.isfinite(oof).all()
    return oof


def make_folds(ep_idx, seed=SEED):
    ue = np.unique(ep_idx)
    perm = np.random.default_rng(seed).permutation(ue)
    fold_of = {int(e): i % N_FOLDS for i, e in enumerate(perm)}
    return np.array([fold_of[int(e)] for e in ep_idx]), {int(e): int(f) for e, f in fold_of.items()}


def cross_episode_permutation(ep_idx, seed=SEED):
    """A permutation pi of window rows such that ep_idx[pi[i]] != ep_idx[i] for EVERY i (the donor window is always in another episode).
    Random permutation, then each same-episode collision is repaired by swapping donors with a random row whose swap violates neither."""
    rng = np.random.default_rng(seed)
    n = len(ep_idx)
    pi = rng.permutation(n)
    for _ in range(100000):
        bad = np.flatnonzero(ep_idx[pi] == ep_idx)
        if len(bad) == 0:
            return pi
        i = int(bad[0])
        for j in rng.permutation(n)[:200]:
            if j != i and ep_idx[pi[j]] != ep_idx[i] and ep_idx[pi[i]] != ep_idx[j]:
                pi[i], pi[j] = pi[j], pi[i]
                break
    raise RuntimeError("could not build a cross-episode permutation")


def within_episode_permutation(ep_idx, seed=SEED):
    """DESCRIPTIVE extra (not in the prereg): shuffle the lead block among windows of the SAME episode (keeps episode-level information)."""
    rng = np.random.default_rng(seed)
    pi = np.arange(len(ep_idx))
    for e in np.unique(ep_idx):
        ii = np.flatnonzero(ep_idx == e)
        pi[ii] = rng.permutation(ii)
    return pi


def boot_cluster(ep_pos, n_ep, fns, B=BOOT_B, seed=SEED, mask=None):
    """Episode-resampling bootstrap with the OUT-OF-FOLD predictions held fixed.  fns: {name: WAuc}.  Returns {name: [B] AUCs}."""
    rng = np.random.default_rng(seed)
    out = {k: np.empty(B) for k in fns}
    for b in range(B):
        draw = rng.integers(0, n_ep, n_ep)
        cnt = np.bincount(draw, minlength=n_ep).astype(np.float64)
        w = cnt[ep_pos] if mask is None else cnt[ep_pos] * mask
        for k, f in fns.items():
            out[k][b] = f(w)
    return out


def cross_fit_hgb(X, y, folds):
    """DESCRIPTIVE extra (NOT the prereg estimator): one fixed, untuned gradient-boosted-tree config under the same episode-grouped folds."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    oof = np.full(len(y), np.nan)
    for k in range(N_FOLDS):
        te = folds == k
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=40, l2_regularization=1.0,
                                           early_stopping=False, random_state=0).fit(X[~te], y[~te])
        oof[te] = m.predict_proba(X[te])[:, 1]
    assert np.isfinite(oof).all()
    return oof


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def pr(x, d=4):
    return None if x is None else round(float(x), d)


# =====================================================================================================================
def main():
    t_start = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    SCR.mkdir(parents=True, exist_ok=True)
    out: dict = {"prereg": "PREREG_LEADINFO_D.md", "tier": "DIAGNOSTIC on logged eval139 labels (no planner); GT features are an UPPER BOUND, never an inference input",
                 "estimator": ("5-fold episode-grouped cross-fit (fold seed 0) of L2 logistic regression (C=1, standardised on the fit fold); score = AUC of the "
                               "out-of-fold P(decel). Delta-AUC CI: EPISODE-RESAMPLING bootstrap (B=2000, seed 0) -- draw the episodes with replacement, "
                               "recompute AUC(B)-AUC(A) on the resampled windows with the out-of-fold predictions held FIXED, 2.5/97.5 percentiles. "
                               "NOT taniteval.ci (a paired bootstrap of per-episode deltas is invalid for an AUC). Answers: would another draw of EPISODES "
                               "say this? -- it is blind to training-run and to inference-run variance (the models here are convex, so the former is nil)."),
                 "deviations": []}
    out["seal"] = check_seal()
    log("[0] seal OK", out["seal"]["prereg_sha256"][:8])

    # ---- 1. synthetic controls -----------------------------------------------------------------------------------
    syn = synthetic_controls()
    out["synthetic_controls"] = syn
    log(f"[1] synthetic controls: {syn['n']} cases, all_pass={syn['all_pass']}")
    if not syn["all_pass"]:
        bad = {k: v for k, v in syn["cases"].items() if not v["pass"]}
        (RES / "LEADINFO_D.json").write_text(json.dumps({"STOP": "synthetic control failed", "failed": bad}, indent=1), encoding="utf-8")
        raise SystemExit(f"STOP: synthetic control failed: {bad}")

    # ---- 2. windows ----------------------------------------------------------------------------------------------
    C, E, comp, gt, pop_cruise, pop_decel = load_windows()
    for k, v in E["checks"].items():
        if isinstance(v, bool):
            assert v, f"load_eval_pf check failed: {k}"
    assert E["checks"]["plan8_slots_equal_os"] == 0.0
    union = pop_cruise | pop_decel
    ui = np.flatnonzero(union)
    out["population"] = {"s6_windows": int(len(E["ws"])), "compliant": int(comp.sum()), "gt_decel_compliant": int(pop_decel.sum()),
                         "gt_cruise_compliant": int(pop_cruise.sum()), "gt_accel_compliant_excluded": int(((gt == "accel") & comp).sum()),
                         "union_n": int(len(ui)), "union_base_rate_decel": pr(pop_decel.sum() / len(ui)),
                         "source": "l2_common.lon_indicators(E['os'], E, compliant): fd finite == GT-cruise population, md finite == GT-decel population "
                                   "(the GT class does not depend on the plan argument)"}
    log("[2] population", out["population"])
    meta = json.load(open(ARCH / "s6_s0" / "manifest.json", encoding="utf-8"))
    eps_meta = meta["episodes"]
    n_ep_all = len(eps_meta)

    # ---- 3. join + poses -----------------------------------------------------------------------------------------
    recs, jmeta = stream_join(eps_meta)
    out["join"] = jmeta
    if not jmeta["prereg_named_path_exists"]:
        out["deviations"].append(f"D1 path: the prereg names {JOIN_XZ_PREREG_PATH}, which does not exist (the project drive D: is E: on this box). Used "
                                 f"{JOIN_XZ} (sha256 {jmeta['join_sha256'][:16]}...), BYTE-IDENTICAL to {JOIN_XZ_ALT} "
                                 f"(sha256 equal: {jmeta['alt_copy_sha256'] == jmeta['join_sha256']}).")
    log("[3] join", {k: v for k, v in jmeta.items() if k not in ("join_path", "alt_copy", "prereg_named_path")})
    poses = load_poses(E, n_ep_all)
    frames_by_ep = {ei: {f: frame_from_record(r) for f, r in fr.items()} for ei, fr in recs.items()}
    eps = {ei: Episode(frames_by_ep.get(ei, {}), poses[ei][0]) for ei in range(n_ep_all)}

    # window <-> pose rows: ws must be in the dump's ws list and ep_poses[ws, 3] must equal v0 (the speed at the NOW frame)
    ep_w, ws_w, v0_w = np.asarray(E["clip"]), np.asarray(E["ws"]), np.asarray(E["v0"], np.float64)
    assert np.array_equal(E["eid"], E["clip"])
    vdiff = np.array([abs(poses[int(e)][0][int(w), 3] - v) for e, w, v in zip(ep_w, ws_w, v0_w)])
    in_dump = np.array([int(w) in set(poses[int(e)][1].tolist()) for e, w in zip(ep_w, ws_w)])
    out["pose_alignment_controls"] = {
        "ep_poses_col3_equals_v0_at_ws": {"max_abs_diff": float(vdiff.max()), "p99": float(np.percentile(vdiff, 99)), "n": int(len(vdiff)),
                                           "reading": "ep_poses[ws, 3] (episode speed at the NOW frame) vs the battery's v0 -- ws indexes the pose rows"},
        "all_ws_listed_in_dump_decisions": bool(in_dump.all())}
    assert in_dump.all()
    log("[3] pose<->ws:", out["pose_alignment_controls"]["ep_poses_col3_equals_v0_at_ws"])
    sweep = pose_join_shift_sweep(recs, poses)
    out["pose_alignment_controls"]["pose_vs_join_frame_shift_sweep"] = sweep
    log("[3] pose<->join frame shift sweep: best shift", sweep["best_shift_by_p10"], {s: round(v["p10_std_m"], 3) for s, v in sweep["shifts"].items()})
    assert sweep["aligned_at_zero"], "pose index and join frame_idx are NOT aligned at shift 0"

    # ---- coverage ------------------------------------------------------------------------------------------------
    has_rec = np.array([ep_w[i] in eps and eps[int(ep_w[i])].frame(int(ws_w[i])) is not None for i in range(len(ws_w))])
    cov = {"union_windows": int(len(ui)), "with_join_record_at_ws": int(has_rec[ui].sum()), "without": int((~has_rec[ui]).sum()),
           "decel_pop": {"n": int(pop_decel.sum()), "covered": int((has_rec & pop_decel).sum())},
           "cruise_pop": {"n": int(pop_cruise.sum()), "covered": int((has_rec & pop_cruise).sum())},
           "fraction": pr(has_rec[ui].mean()),
           "episodes_in_union": int(len(np.unique(ep_w[ui]))), "episodes_with_any_covered_union_window": int(len(np.unique(ep_w[ui][has_rec[ui]])))}
    unc_eps = np.unique(ep_w[ui][~has_rec[ui]])
    cov["uncovered_union_windows_by_episode_index"] = {int(e): int(((ep_w == e) & union & ~has_rec).sum()) for e in unc_eps}
    out["coverage"] = cov
    log("[3] coverage", {k: v for k, v in cov.items() if k != "uncovered_union_windows_by_episode_index"})
    out["deviations"].append("D2 coverage rule: windows with no join record at ws are counted/reported and EXCLUDED from the B/C comparison "
                             "(NO_LABEL is not 'road clear'); A is fitted/scored on the SAME covered windows so A, B, C are paired. "
                             "AUC(A) and AUC(lon head) on the full union population are reported beside it.")

    # ---- features ------------------------------------------------------------------------------------------------
    sel = ui[has_rec[ui]]                                   # analysis windows (covered, in the union)
    y_all = pop_decel.astype(np.int64)
    lg6 = C.lon6(E["logits"])                               # [n, 6] in LON6 order
    P6 = softmax(lg6)
    A_all = np.column_stack([P6, v0_w])                     # A = softmax(6 LON6 logits) + v0
    p_dec_head = P6[:, list(C.G_DECEL)].sum(axis=1)         # lon head alone: P(decel classes) = BRAKE_TO + CREEP + HOLD
    blocks, hist_ok, rows_info = [], [], []
    for i in sel:
        b = lead_block(eps[int(ep_w[i])], int(ws_w[i]), float(v0_w[i]))
        assert b is not None
        blocks.append([b[c] for c in LEAD_COLS])
        rows_info.append(b)
    L = np.asarray(blocks, np.float64)
    ep_s, ws_s, y_s, v0_s = ep_w[sel], ws_w[sel], y_all[sel], v0_w[sel]
    A_s = A_all[sel]
    B_s = np.column_stack([A_s, L])
    pi = cross_episode_permutation(ep_s, SEED)
    assert (ep_s[pi] != ep_s).all()
    C_s = np.column_stack([A_s, L[pi]])
    pi_w = within_episode_permutation(ep_s, SEED)
    CW_s = np.column_stack([A_s, L[pi_w]])
    folds, fold_of = make_folds(ep_s)
    ue = np.unique(ep_s)
    n_ep = len(ue)
    ep_pos = np.searchsorted(ue, ep_s)

    flag = L[:, 0] == 1.0
    ok_hist = np.array([r["hist_ok"] for r in rows_info])
    dec = y_s == 1
    out["lead_features"] = {
        "columns": list(LEAD_COLS), "frame_convention": "join agents are in the EGO frame of their own frame (+x forward, +y left); the lead is the nearest "
        "centre (smallest cx) with 0 < cx <= 80 m and |cy| <= 1.8 m at the NOW frame ws (= the window origin t0; refc_v3_train.py:4635 lookup(eid, t + w - 1)); "
        "all join classes count (automobile, truck, bus, person, rider, ...), occluded or not (GT upper bound)",
        "lead_present_rate_all": pr(flag.mean()),
        "lead_present_rate_decel_windows": pr(flag[dec].mean()), "lead_present_rate_cruise_windows": pr(flag[~dec].mean()),
        "n_decel_windows": int(dec.sum()), "n_cruise_windows": int((~dec).sum()),
        "leads_with_usable_history": int((flag & ok_hist).sum()), "leads_total": int(flag.sum()),
        "leads_dup_track_id": int(sum(r["dup"] for r in rows_info)),
        "lead_occluded_share": pr(np.mean([r["occ"] for r, f in zip(rows_info, flag) if f])),
        "lead_range_x_quantiles_m_q05_q25_q50_q75_q95": [pr(q, 2) for q in np.percentile(L[flag, 1], [5, 25, 50, 75, 95])],
        "n_leads_range_lt_3m": int((flag & (L[:, 1] < 3.0)).sum()),
        "history_k_counts": {int(k): int(((np.array([r['hist_k'] for r in rows_info]) == k) & flag).sum()) for k in range(0, 11) if (((np.array([r['hist_k'] for r in rows_info]) == k) & flag).sum())},
    }
    # lead presence difference with an episode-resampling CI
    rng = np.random.default_rng(SEED)
    d_boot = np.empty(BOOT_B)
    for b in range(BOOT_B):
        draw = rng.integers(0, n_ep, n_ep)
        w = np.bincount(draw, minlength=n_ep).astype(np.float64)[ep_pos]
        d_boot[b] = (w * flag * dec).sum() / (w * dec).sum() - (w * flag * ~dec).sum() / (w * ~dec).sum()
    out["lead_features"]["presence_diff_decel_minus_cruise"] = {"point": pr(flag[dec].mean() - flag[~dec].mean()), "ci95": [pr(x) for x in ci(d_boot)],
                                                                "estimator": "episode-resampling bootstrap B2000 seed0"}
    # consistency: world lead speed vs v0 + range_rate (pose-free), among valid leads
    v_ok = flag & ok_hist
    if v_ok.sum() > 10:
        d = L[v_ok, 4] - (v0_s[v_ok] + L[v_ok, 3])
        out["lead_features"]["pose_free_consistency"] = {"n": int(v_ok.sum()), "median_abs_diff_ms": pr(np.median(np.abs(d)), 3),
                                                         "p90_abs_diff_ms": pr(np.percentile(np.abs(d), 90), 3),
                                                         "corr": pr(np.corrcoef(L[v_ok, 4], v0_s[v_ok] + L[v_ok, 3])[0, 1], 4),
                                                         "reading": "world-frame lead speed (ego poses) vs v0 + Euclidean range rate; they differ by the ego speed change over the past second"}
    log("[4] lead features", {k: v for k, v in out["lead_features"].items() if k not in ("frame_convention",)})

    # ---- REAL control: no in-path lead closer than the ego's 1 s travel on windows where GT keeps cruising at > 10 m/s ----------
    g1 = np.asarray(E["g"], np.float64)[sel, 0, :]                    # GT position at +1 s, ego frame at t0
    d1 = np.hypot(g1[:, 0], g1[:, 1])
    big = (d1 > 10.0) & (v0_s > 10.0) & (~dec)                        # GT cruise (population = cruise), v0 > 10, travels > 10 m in the first second
    rear = L[:, 1] - np.array([r["length"] for r in rows_info]) / 2.0
    out["real_control_lead_vs_ego_1s_travel"] = {
        "definition": "windows in the GT-cruise population with v0 > 10 m/s and GT displacement at +1 s > 10 m (the ego keeps cruising at > 10 m/s): "
                      "count windows with a lead whose range cx < the ego's GT 1 s travel (a lead inside the distance the ego covers in the next second "
                      "while the ego keeps cruising is physically implausible); NOT filtered on",
        "n_windows_cruising_gt10": int(big.sum()), "n_with_lead": int((big & flag).sum()),
        "n_lead_centre_closer_than_1s_travel": int((big & flag & (L[:, 1] < d1)).sum()),
        "n_lead_rear_edge_closer_than_1s_travel": int((big & flag & (rear < d1)).sum()),
        "min_over_windows_of_(range_x - travel_1s)_m": pr(np.min((L[:, 1] - d1)[big & flag]), 2) if (big & flag).any() else None,
        "same_count_all_covered_windows_with_travel_gt10": {"n": int((d1 > 10).sum()), "n_with_lead": int(((d1 > 10) & flag).sum()),
                                                            "n_closer": int(((d1 > 10) & flag & (L[:, 1] < d1)).sum())},
        "contrast_decel_windows_travel_gt10_closer": int(((d1 > 10) & dec & flag & (L[:, 1] < d1)).sum()),
    }
    # the stricter, physically grounded form (EXTRA): a lead inside the ego's next-second travel is only compatible with an ego that keeps
    # cruising if the lead moves at about the ego's speed (a short-gap FOLLOWER); a SLOWER lead there would be a collision course.
    cl = big & flag & (L[:, 1] < d1)
    ratio = L[cl, 4] / np.maximum(v0_s[cl], 1e-6)
    out["real_control_lead_vs_ego_1s_travel"]["extra_closer_cases_are_followers"] = {
        "n_closer": int(cl.sum()), "time_gap_s_range": [pr(L[cl, 5].min(), 3), pr(L[cl, 5].max(), 3)] if cl.any() else None,
        "lead_speed_over_v0_min_median_max": [pr(ratio.min(), 3), pr(np.median(ratio), 3), pr(ratio.max(), 3)] if cl.any() else None,
        "n_with_lead_slower_than_0.9_v0 (collision course while the GT ego keeps cruising)": int((ratio < 0.9).sum()),
        "n_episodes": int(len(set(ep_s[cl].tolist()))),
        "reading": "all of them are short-gap following (time gap < 1 s, lead speed ~ ego speed) -- compatible with the GT ego cruising; a misaligned join "
                   "or pose would instead give leads at ego-frame positions the GT path drives through at a closing speed"}
    log("[4] real control", out["real_control_lead_vs_ego_1s_travel"])

    # ---- AUCs ----------------------------------------------------------------------------------------------------
    oof = {"A": cross_fit(A_s, y_s, ep_s, folds), "B": cross_fit(B_s, y_s, ep_s, folds), "C": cross_fit(C_s, y_s, ep_s, folds)}
    oof["C_within_episode"] = cross_fit(CW_s, y_s, ep_s, folds)
    head = p_dec_head[sel]
    head8 = softmax(np.asarray(E["logits"], np.float64))                   # 8-logit softmax variant (descriptive)
    head8_dec = head8[:, [C.LON6_V7_IDS[j] for j in C.G_DECEL]].sum(axis=1)[sel]
    scores = {"A": oof["A"], "B": oof["B"], "C": oof["C"], "lon_head": head}
    # lead block ALONE (fit the same way) -- how much of the label the lead block carries without the lon head (descriptive)
    oof["lead_only"] = cross_fit(np.column_stack([v0_s, L]), y_s, ep_s, folds)
    oof["v0_only"] = cross_fit(v0_s.reshape(-1, 1), y_s, ep_s, folds)
    point = {k: auc_np(y_s, s) for k, s in scores.items()}
    point["lon_head_8logit_softmax"] = auc_np(y_s, head8_dec)
    point["C_within_episode"] = auc_np(y_s, oof["C_within_episode"])
    point["lead_only_with_v0"] = auc_np(y_s, oof["lead_only"])
    point["v0_only"] = auc_np(y_s, oof["v0_only"])
    fns = {k: WAuc(y_s, s) for k, s in {**scores, "C_within_episode": oof["C_within_episode"], "lead_only_with_v0": oof["lead_only"], "v0_only": oof["v0_only"]}.items()}
    # the weighted-AUC kernel must reproduce the Mann-Whitney AUC at unit weights
    for k, f in fns.items():
        pk = point[k]
        assert abs(f(np.ones(len(y_s))) - pk) < 1e-9, f"WAuc != auc_np for {k}"
    boots = boot_cluster(ep_pos, n_ep, fns)
    d = lambda a, b: np.asarray(boots[a]) - np.asarray(boots[b])
    res = {}
    for k in ("A", "B", "C", "lon_head", "C_within_episode", "lead_only_with_v0", "v0_only"):
        res[k] = {"auc": pr(point[k]), "ci95": [pr(x) for x in ci(boots[k])]}
    res["lon_head_8logit_softmax"] = {"auc": pr(point["lon_head_8logit_softmax"])}
    dBA, dCA, dBC = point["B"] - point["A"], point["C"] - point["A"], point["B"] - point["C"]
    res["delta_B_minus_A"] = {"point": pr(dBA), "ci95": [pr(x) for x in ci(d("B", "A"))]}
    res["delta_C_minus_A"] = {"point": pr(dCA), "ci95": [pr(x) for x in ci(d("C", "A"))]}
    res["delta_B_minus_C"] = {"point": pr(dBC), "ci95": [pr(x) for x in ci(d("B", "C"))]}
    res["delta_Cwithin_minus_A_DESCRIPTIVE"] = {"point": pr(point["C_within_episode"] - point["A"]), "ci95": [pr(x) for x in ci(d("C_within_episode", "A"))]}
    res["delta_A_minus_lon_head_DESCRIPTIVE"] = {"point": pr(point["A"] - point["lon_head"]), "ci95": [pr(x) for x in ci(d("A", "lon_head"))]}
    # A / lon head on the FULL union population (incl. windows with no join record), for the 0.685 reference
    uh = p_dec_head[ui]
    res["lon_head_full_union_population"] = {"auc": pr(auc_np(y_all[ui], uh)), "n": int(len(ui)), "reference": "0.685 measured earlier (as stated in the brief)"}
    # known-value control: the earlier 0.685 is REPRODUCED by the decel/(decel+cruise) renormalised head score on the same union population
    p_cru_head = P6[:, list(C.G_CRUISE)].sum(axis=1)
    renorm = p_dec_head / np.maximum(p_dec_head + p_cru_head, 1e-300)
    res["lon_head_renorm_decel_over_decel_plus_cruise"] = {
        "auc_full_union": pr(auc_np(y_all[ui], renorm[ui])), "auc_analysis_set": pr(auc_np(y_s, renorm[sel])), "n_full_union": int(len(ui)),
        "reading": "KNOWN-VALUE CONTROL: reproduces the 0.685 of the brief; the prereg's literal 'P(decel classes) from the softmax' is the "
                   "un-renormalised sum, reported as 'lon_head' above (0.6745 on the full union)"}
    assert abs(res["lon_head_renorm_decel_over_decel_plus_cruise"]["auc_full_union"] - 0.685) < 0.001, "0.685 known value NOT reproduced"

    # ---- DESCRIPTIVE extras (cannot change the word; nothing here may be used to tune anything scored on eval139) ---------------------------
    lp = flag.astype(np.float64)                       # lead-present subset: the only windows where a lead block can carry anything
    sub = {}
    # the masked weighted AUC is built on the FULL arrays with zero weights outside the subset (asserted equal to the subset AUC below)
    full_fns = {k: WAuc(y_s, s) for k, s in (("A", oof["A"]), ("B", oof["B"]), ("C", oof["C"]))}
    for k, f in full_fns.items():
        assert abs(f(lp) - auc_np(y_s[flag], {"A": oof["A"], "B": oof["B"], "C": oof["C"]}[k][flag])) < 1e-9, "masked WAuc != subset AUC"
    bsub = boot_cluster(ep_pos, n_ep, full_fns, mask=lp)
    sub["n_windows"] = int(flag.sum()); sub["n_decel"] = int((flag & dec).sum()); sub["n_episodes"] = int(len(set(ep_s[flag].tolist())))
    for k in ("A", "B", "C"):
        sub[k] = {"auc": pr(full_fns[k](lp)), "ci95": [pr(x) for x in ci(bsub[k])]}
    sub["delta_B_minus_A"] = {"point": pr(full_fns["B"](lp) - full_fns["A"](lp)), "ci95": [pr(x) for x in ci(bsub["B"] - bsub["A"])]}
    sub["delta_C_minus_A"] = {"point": pr(full_fns["C"](lp) - full_fns["A"](lp)), "ci95": [pr(x) for x in ci(bsub["C"] - bsub["A"])]}
    res["DESCRIPTIVE_lead_present_subset_logistic"] = sub
    # nonlinear probe (a negative from a LINEAR probe is not a negative about learnability): same folds, one fixed untuned config
    oof_h = {"A": cross_fit_hgb(A_s, y_s, folds), "B": cross_fit_hgb(B_s, y_s, folds), "C": cross_fit_hgb(C_s, y_s, folds)}
    fh = {k: WAuc(y_s, s) for k, s in oof_h.items()}
    bh = boot_cluster(ep_pos, n_ep, fh)
    hg = {k: {"auc": pr(fh[k](np.ones(len(y_s)))), "ci95": [pr(x) for x in ci(bh[k])]} for k in fh}
    hg["delta_B_minus_A"] = {"point": pr(fh["B"](np.ones(len(y_s))) - fh["A"](np.ones(len(y_s)))), "ci95": [pr(x) for x in ci(bh["B"] - bh["A"])]}
    hg["delta_C_minus_A"] = {"point": pr(fh["C"](np.ones(len(y_s))) - fh["A"](np.ones(len(y_s)))), "ci95": [pr(x) for x in ci(bh["C"] - bh["A"])]}
    hg["config"] = "HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=0); NOT tuned"
    res["DESCRIPTIVE_nonlinear_probe_hgb"] = hg
    out["results"] = res
    out["n"] = {"windows": int(len(y_s)), "episodes": int(n_ep), "n_decel": int(dec.sum()), "n_cruise": int((~dec).sum()),
                "decel_base_rate": pr(dec.mean()), "join_coverage_of_union": cov["fraction"], "bootstrap_B": BOOT_B, "bootstrap_seed": SEED,
                "folds": N_FOLDS, "fold_seed": SEED, "windows_per_fold": [int((folds == k).sum()) for k in range(N_FOLDS)],
                "episodes_per_fold": [int(len(set(ep_s[folds == k]))) for k in range(N_FOLDS)]}
    out["controls_C"] = {"permutation": "window-level seed-0 permutation of the lead block with EVERY donor in a DIFFERENT episode (asserted); "
                                         "v0 and the lon-head features stay with their own window",
                         "n_donor_in_other_episode": int((ep_s[pi] != ep_s).sum()), "n": int(len(pi))}

    # ---- the word (prereg section 4, literal) -------------------------------------------------------------------------
    lo_BA, hi_BA = res["delta_B_minus_A"]["ci95"]
    lo_CA, hi_CA = res["delta_C_minus_A"]["ci95"]
    c1 = bool(dBA >= DELTA_BAR and lo_BA > 0)
    c2 = bool(lo_CA <= 0.0 <= hi_CA)
    word = "LEAD-INFO PRESENT" if (c1 and c2) else "LEAD-INFO ABSENT"
    out["word"] = word
    out["word_clauses"] = {"clause1_dAUC_BA_ge_0.03_and_ci_lo_gt_0": c1, "clause2_dAUC_CA_ci_contains_0": c2,
                           "dAUC_BA": pr(dBA), "ci_BA": [pr(lo_BA), pr(hi_BA)], "dAUC_CA": pr(dCA), "ci_CA": [pr(lo_CA), pr(hi_CA)]}
    out["next_per_prereg"] = ("vision-only lead feature from refcv8's OWN box head (P0 post-processing), measured the same way on eval139, then a FRESH-split gate "
                              "(needs obstacle.offline for the 300 fresh clips: a download, sized first)") if word == "LEAD-INFO PRESENT" else \
        "the traffic-light state (the v7.2 GT TL labels as the upper bound), same diagnostic"
    out["deviations"].append(
        "D3 details the prereg leaves open (fixed in code BEFORE any AUC was computed, none tuned): range_x = the agent CENTRE cx (not the rear edge); the corridor is tested on "
        "the centre (|cy| <= 1.8, 0 < cx <= 80, both ends inclusive at 1.8 / 80); all join classes count (persons, riders, animals included) and occluded flags are ignored "
        "(GT upper bound; the selected leads are 0 % occluded anyway); range_rate uses the Euclidean range r = hypot(cx, cy) of the SAME track_id, baseline 1.0 s (10 frames) "
        "or, if that frame/track is absent, the oldest of 0.9..0.3 s; lead_speed = world displacement of the track over that baseline using the dump's ep_poses (pose index == join "
        "frame_idx, asserted by the shift sweep); no usable history (2 of 920 leads) -> rate 0 and lead_speed = v0; no lead -> lateral 0, lead_speed 0, range 80, rate 0, flag 0, "
        "time gap 80 / max(v0, 1). 'Past 1.0 s' is 10 provider frames (join t_s step 0.1007 s, used for dt).")
    out["deviations"].append(
        "D4 head reference: the prereg's literal 'P(decel classes) from the softmax' (BRAKE_TO+CREEP+HOLD of the 6-logit softmax) reads 0.6745 on the full union; the brief's 0.685 is "
        "reproduced (0.6851) by the renormalised score decel/(decel+cruise). Both are reported; the literal one is the primary 'lon head alone' row.")
    out["deviations"].append(
        "D5 (implementation note, not a departure): prereg s3's 'paired episode-cluster bootstrap with the out-of-fold predictions held fixed' is implemented as the brief says -- "
        "resample EPISODES (B 2000, seed 0), recompute AUC on the resampled windows for A, B, C on the SAME draw -- NOT taniteval.ci's per-episode-delta form, which is invalid for an AUC; "
        "the bootstrap universe is the 135 episodes with >= 1 covered union window (136 episodes have a union window, 139 exist).")
    out["fold_assignment_sha12"] =hashlib.sha256(json.dumps(sorted(fold_of.items())).encode()).hexdigest()[:12]
    out["inputs_sha256"] = {"prereg": out["seal"]["prereg_sha256"], "join_xz": jmeta["join_sha256"],
                            "extras_npz_dump_s0": sha256_file(ARCH / "dump_s0" / "refcv6_extras.npz"),
                            "s6_manifest_s0": sha256_file(ARCH / "s6_s0" / "manifest.json"), "code_leadinfo_d.py": sha256_file(__file__)}
    out["elapsed_s"] = round(time.time() - t_start, 1)
    log("[5] RESULT", word, json.dumps(res, indent=None)[:1600])

    np.savez_compressed(RES / "leadinfo_windows.npz", ep=ep_s, ws=ws_s, y=y_s, v0=v0_s, fold=folds, lead=L, lead_cols=np.array(LEAD_COLS),
                        oof_A=oof["A"], oof_B=oof["B"], oof_C=oof["C"], p_decel_head=head)
    (RES / "LEADINFO_D.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    write_md(out)
    log("done in", out["elapsed_s"], "s")


def write_md(o: dict):
    r, n, cov, lf = o["results"], o["n"], o["coverage"], o["lead_features"]
    L = []
    L.append(f"# LEADINFO_D -- {o['word']}\n")
    L.append("PREREG_LEADINFO_D.md (sha256 `%s`), DIAGNOSTIC on logged eval139 labels (S6, compliant, seed-0 battery draw `step48852ema_pf`); no planner; "
             "GT lead features are an UPPER BOUND, never an inference input.\n" % o["seal"]["prereg_sha256"])
    wc = o["word_clauses"]
    L.append(f"**Word:** {o['word']}.  Clause 1 (dAUC(B-A) >= +0.03 with CI lo > 0): **{wc['clause1_dAUC_BA_ge_0.03_and_ci_lo_gt_0']}** "
             f"(dAUC {wc['dAUC_BA']:+.4f}, CI [{wc['ci_BA'][0]:+.4f}, {wc['ci_BA'][1]:+.4f}]).  Clause 2 (dAUC(C-A) CI contains 0): "
             f"**{wc['clause2_dAUC_CA_ci_contains_0']}** (dAUC {wc['dAUC_CA']:+.4f}, CI [{wc['ci_CA'][0]:+.4f}, {wc['ci_CA'][1]:+.4f}]).\n")
    L.append(f"Next per prereg s4: {o['next_per_prereg']}.\n")
    L.append("## Estimator\n" + o["estimator"] + "\n")
    L.append("## n and coverage\n")
    L.append(f"* analysis set: **{n['windows']} windows, {n['episodes']} episodes** ({n['n_decel']} GT-decel, {n['n_cruise']} GT-cruise), decel base rate **{n['decel_base_rate']}**; "
             f"folds {n['episodes_per_fold']} episodes / {n['windows_per_fold']} windows.")
    L.append(f"* union population (compliant GT-decel + GT-cruise, S6): {cov['union_windows']} windows; **join record at ws: {cov['with_join_record_at_ws']} "
             f"({cov['fraction']})**, without: {cov['without']} (decel {cov['decel_pop']['covered']}/{cov['decel_pop']['n']} covered, cruise "
             f"{cov['cruise_pop']['covered']}/{cov['cruise_pop']['n']}); uncovered windows by episode index: {cov['uncovered_union_windows_by_episode_index']}.")
    L.append(f"* join: `{o['join']['join_path']}` sha256 `{o['join']['join_sha256']}` ({o['join']['records_kept']} records for the 139 eval clips).\n")
    L.append("## AUC (out-of-fold P(decel); CI = episode-resampling bootstrap, B 2000, seed 0)\n")
    L.append("| model | AUC | 95% CI |\n|---|---|---|")
    for k, lab in (("lon_head", "lon head alone, P(BRAKE_TO+CREEP+HOLD) from the 6-logit softmax"), ("A", "A = softmax(6 LON6 logits) + v0"),
                   ("B", "B = A + GT lead block"), ("C", "C = A + lead block permuted across episodes (control)"),
                   ("C_within_episode", "extra (not prereg): A + lead block shuffled WITHIN episode"),
                   ("lead_only_with_v0", "extra: v0 + lead block, no lon head"), ("v0_only", "extra: v0 only")):
        L.append(f"| {lab} | {r[k]['auc']:.4f} | [{r[k]['ci95'][0]:.4f}, {r[k]['ci95'][1]:.4f}] |")
    L.append(f"\nlon head alone, 8-logit softmax variant: {r['lon_head_8logit_softmax']['auc']:.4f};  lon head alone on the FULL union population "
             f"(n={r['lon_head_full_union_population']['n']}, incl. uncovered): {r['lon_head_full_union_population']['auc']:.4f} (ref. 0.685).\n")
    L.append("| delta AUC | point | 95% CI |\n|---|---|---|")
    for k, lab in (("delta_B_minus_A", "B - A"), ("delta_C_minus_A", "C - A"), ("delta_B_minus_C", "B - C"),
                   ("delta_Cwithin_minus_A_DESCRIPTIVE", "(within-episode shuffle) - A [extra]"), ("delta_A_minus_lon_head_DESCRIPTIVE", "A - lon head alone [extra]")):
        L.append(f"| {lab} | {r[k]['point']:+.4f} | [{r[k]['ci95'][0]:+.4f}, {r[k]['ci95'][1]:+.4f}] |")
    rn = r["lon_head_renorm_decel_over_decel_plus_cruise"]
    L.append(f"\n**Known-value control (passes):** the 0.685 of the brief is reproduced by the renormalised head score decel/(decel+cruise) on the full union: "
             f"**{rn['auc_full_union']:.4f}** (n={rn['n_full_union']}); on the analysis set {rn['auc_analysis_set']:.4f}. The prereg's literal un-renormalised P(decel classes) is "
             f"the 'lon head alone' row above (primary).\n")
    sb, hg = r["DESCRIPTIVE_lead_present_subset_logistic"], r["DESCRIPTIVE_nonlinear_probe_hgb"]
    L.append("## Descriptive extras (cannot change the word; not part of the prereg)\n")
    L.append(f"* lead-present windows only (n={sb['n_windows']}, {sb['n_decel']} decel, {sb['n_episodes']} episodes), same out-of-fold logistic predictions: "
             f"AUC A {sb['A']['auc']:.4f}, B {sb['B']['auc']:.4f}, C {sb['C']['auc']:.4f}; dAUC(B-A) {sb['delta_B_minus_A']['point']:+.4f} "
             f"[{sb['delta_B_minus_A']['ci95'][0]:+.4f}, {sb['delta_B_minus_A']['ci95'][1]:+.4f}], dAUC(C-A) {sb['delta_C_minus_A']['point']:+.4f} "
             f"[{sb['delta_C_minus_A']['ci95'][0]:+.4f}, {sb['delta_C_minus_A']['ci95'][1]:+.4f}].")
    L.append(f"* nonlinear probe (one fixed untuned gradient-boosted-tree config, same folds): AUC A {hg['A']['auc']:.4f}, B {hg['B']['auc']:.4f}, C {hg['C']['auc']:.4f}; "
             f"dAUC(B-A) {hg['delta_B_minus_A']['point']:+.4f} [{hg['delta_B_minus_A']['ci95'][0]:+.4f}, {hg['delta_B_minus_A']['ci95'][1]:+.4f}], "
             f"dAUC(C-A) {hg['delta_C_minus_A']['point']:+.4f} [{hg['delta_C_minus_A']['ci95'][0]:+.4f}, {hg['delta_C_minus_A']['ci95'][1]:+.4f}]. "
             f"({hg['config']})\n")
    L.append("\n## Lead presence\n")
    L.append(f"* lead present in **{lf['lead_present_rate_decel_windows']}** of GT-decel windows vs **{lf['lead_present_rate_cruise_windows']}** of GT-cruise windows; "
             f"difference {lf['presence_diff_decel_minus_cruise']['point']:+.4f} CI [{lf['presence_diff_decel_minus_cruise']['ci95'][0]:+.4f}, {lf['presence_diff_decel_minus_cruise']['ci95'][1]:+.4f}] "
             f"(episode-resampling bootstrap).")
    L.append(f"* leads {lf['leads_total']}, usable 1-s history {lf['leads_with_usable_history']}, occluded share {lf['lead_occluded_share']}, range quantiles q05/25/50/75/95 {lf['lead_range_x_quantiles_m_q05_q25_q50_q75_q95']} m, "
             f"leads closer than 3 m: {lf['n_leads_range_lt_3m']}.")
    L.append("* convention: " + lf["frame_convention"] + "\n")
    L.append("## Alignment controls\n")
    sc = o["synthetic_controls"]
    L.append(f"* synthetic (production code path): **{sc['n']} cases, all pass = {sc['all_pass']}** -- an agent at (20, 0) m is found at range 20, lateral 0 (and: behind/outside/over-80 m not found, "
             f"corridor edges inclusive, nearest wins, NO_LABEL = None, moving/stationary/turning-ego rates and world speed, history fallback, causality).")
    pa = o["pose_alignment_controls"]
    sw = pa["pose_vs_join_frame_shift_sweep"]
    L.append(f"* real, pose index vs join frame_idx: stationary-track world-position std (p10 over tracks) by shift {{{', '.join(f'{s}: {v['p10_std_m']:.3f} m' for s, v in sw['shifts'].items())}}} -> "
             f"best shift {sw['best_shift_by_p10']} (aligned at 0: {sw['aligned_at_zero']}).")
    L.append(f"* real, ep_poses[ws,3] vs battery v0: max |diff| {pa['ep_poses_col3_equals_v0_at_ws']['max_abs_diff']:.2e} m/s over {pa['ep_poses_col3_equals_v0_at_ws']['n']} windows.")
    rc = o["real_control_lead_vs_ego_1s_travel"]
    L.append(f"* real, lead range vs the ego's GT 1 s travel (GT cruising at > 10 m/s; {rc['n_windows_cruising_gt10']} windows, {rc['n_with_lead']} with a lead): "
             f"**{rc['n_lead_centre_closer_than_1s_travel']}** leads with centre range < 1 s travel, {rc['n_lead_rear_edge_closer_than_1s_travel']} with rear edge < 1 s travel "
             f"(min range - travel = {rc['min_over_windows_of_(range_x - travel_1s)_m']} m); all covered windows with travel > 10 m: {rc['same_count_all_covered_windows_with_travel_gt10']}; "
             f"contrast, GT-decel windows with travel > 10 m and a lead closer than the travel: {rc['contrast_decel_windows_travel_gt10_closer']}.")
    fo = rc["extra_closer_cases_are_followers"]
    L.append(f"* reading of those {fo['n_closer']} hits ({fo['n_episodes']} episodes): time gap range {fo['time_gap_s_range']} s, lead speed / v0 min-median-max "
             f"{fo['lead_speed_over_v0_min_median_max']}, hits with a lead SLOWER than 0.9 v0 (a collision course): "
             f"**{fo['n_with_lead_slower_than_0.9_v0 (collision course while the GT ego keeps cruising)']}**. "
             + ("Zero collision-course cases: the hits are short-gap followers moving with the ego, so the alignment is not contradicted; "
                if fo['n_with_lead_slower_than_0.9_v0 (collision course while the GT ego keeps cruising)'] == 0 else
                "NONZERO collision-course cases: the alignment is NOT cleanly supported; ")
             + "reported as written, nothing was filtered.\n")
    L.append("## Deviations\n" + "\n".join(f"* {d}" for d in o["deviations"]) + "\n")
    L.append("Tier: DIAGNOSTIC (T0-style logged labels, no planner) -- never driving performance. The CI answers 'another draw of EPISODES'; the logistic models are convex so "
             "training-seed variance is nil, and no stochastic planner is involved.\n")
    (RES / "LEADINFO_D.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
