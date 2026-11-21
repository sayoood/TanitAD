"""E5b -- the rows the scorer could not compute from arrays alone: DK-VIOL / DK-HW / DK-TG (a lead block from a scene's tracks) and S-POS@30/50/80
(S-head checkpoint error against the logged route, minus the CTRV baseline). Nothing is re-implemented: every step CALLS the programme's own function,
loaded by pinned path (sha256 checked):
  taniteval.lead_source.lead_block        the function that built the banked b1_eval_lead_block_k30.npz          (DK)
  stack/scripts/build_v9_labels           log_from_arrays / clip_geo / rc_fields: the v9 route checkpoints (S_8 path, arcs 30/50/80)   (S-POS GT)
  t1hd_read                               to_w0 / to_w0_f32 / mean_curvature / ctrv_straight / checkpoint_errors / per_window          (S-POS)
Runs in the scorer's OWN process (the readers bind the reader tree); the tests call it through `t2v8f_score.run_sub`.

SCENE INPUTS (the `SceneLog`): the rollout's logged rig poses `gt_T` [n,4,4] and times `gt_ts_us` [n] (10 Hz, the same arrays the E3 rollout is given) and the
NuRec `sequence_tracks.json` dict. Tracks are WORLD frame; each sample is converted to the RIG frame at its OWN timestamp (np.interp of the ego pose at the
sample time), the exact inverse of `lead_source.lead_track_in_window`, so `lead_block` receives what `obstacle.offline` would have given it.
DK horizon: 30 knots at 0.2 s (the banked block's grid), window origin = the start tick f0, ego speed v0 = backward difference at f0 (= the policy's `set_t0`).

ROLLOUT RECORD the scorer reads (what `t2v8f_exec.run_t2_rollout` returns, plus the policy's `extra` which now carries S):
  rec["f0"], rec["t0_step"], rec["raw_off"], rec["cl"] [30,2] executed, rec["g"] [30,2] logged, rec["v0"], rec["track"] [Nraw,4] executed 10 Hz (x,y,yaw,v),
  rec["extra"][k]["s_ckpt"] [3,3] (x,y,psi in S's tick frame), ["s_pose"] [3] (S's frame, world, float32 values), ["a_s"], ["step"]  for ticks k = 0, 1, 2.
  A start's identity is (scene index, f0); arms are PAIRED on that key (`pair_keys`), never on position.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import warnings

import numpy as np

import t2v8f_score as S

TREE = S.TREE
SRC = {"lead_source": (f"{TREE}/taniteval/taniteval/lead_source.py", "373f925ec54cca6aa6477cf849acfb9ccc1ae557700f5ee95226192ec1ce6a9f"),
       "build_v9_labels": (f"{TREE}/stack/scripts/build_v9_labels.py", "3a96dfd6a30c74d67f52f04f838c6b14108eadc7dee97ed040fc0b6bfd1e750a"),
       "s2_geom_emit_v7": (f"{TREE}/stack/scripts/s2_geom_emit_v7.py", "6b7b8af41330407c0343c1e98a80c0b0fef67e7d2ad771c05f47c02e65c31400"),
       "build_lead_block": (f"{TREE}/taniteval/tools/build_lead_block.py", "e21ca5b3e9f0ca1424af92b3f7fe624db2a177a3ce4871960764a9c59a1a2e34"),
       "v8f_consistency": (f"{TREE}/taniteval/taniteval/v8f_consistency.py", "16ed3586c9b6e503a8bce43eabb8d5464816dd5a60c342d871c5ee5a04041b90"),
       "t1hd_read": (f"{S.ART}/video5/ref/t1hd_read.py", "3c6d9f790f85511553c607a9d2102a7a425ac7c9a674be69ac23ea3a6cb1e9e2")}
K, DT_LEAD = 30, 0.2
LEAD_SPEED_TOL_MPS = 1e-3
_L = {}


def init_lead():
    """-> dict(LS=lead_source, B9=build_v9_labels, BLB=build_lead_block, T1H=t1hd_read (setup_imports done), DK=the reader's dk module). Idempotent."""
    if _L:
        return _L
    H = S.init()
    for nm, (p, want) in SRC.items():
        got = S.sha256_file(p)
        if got != want:
            raise RuntimeError(f"REFUSED: {nm} at {p} is {got}, not the pinned {want}")
    sys.path.insert(0, f"{TREE}/stack/scripts")
    from taniteval import lead_source as LS
    import build_v9_labels as B9

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    BLB = load("build_lead_block_pin", SRC["build_lead_block"][0])
    T1H = load("t1hd_read", SRC["t1hd_read"][0])
    T1H.setup_imports(TREE)
    for m in (LS, B9, BLB, T1H):
        if os.path.abspath(m.__file__).replace("\\", "/").upper().startswith("G:/"):
            raise RuntimeError("an instrument was imported from G:")
    _L.update(LS=LS, B9=B9, BLB=BLB, T1H=T1H, DK=H["T"].DK)
    return _L


# ============================================================================================================================ DK: the lead block from scene tracks
def ego_from_rig(T, tus):
    """ego dict (t s from the first tick, x, y, yaw UNWRAPPED, v) from the rig poses. v = backward difference |p[i]-p[i-1]|/0.1 (row 0 repeats row 1): the
    policy's `set_t0` speed at NOW, so the DK `speeds` equal the rollout's v0."""
    T = np.asarray(T, np.float64)
    tus = np.asarray(tus, np.int64)
    x, y = T[:, 0, 3], T[:, 1, 3]
    d = np.hypot(np.diff(x), np.diff(y)) / 0.1
    return {"t": (tus - tus[0]) / 1e6, "x": x, "y": y, "yaw": np.unwrap(np.arctan2(T[:, 1, 0], T[:, 0, 0])), "v": np.concatenate([[d[0]], d])}


def obs_from_tracks(seq, ego, t_first_us, mut=None):
    """`sequence_tracks.json` (world frame) -> the obs dict `lead_block` takes (t, track, center_x, center_y, size_x, is_vehicle), RIG frame at each sample's OWN time.
    size_x = the cuboid's length (cuboids_dims[i][0]); classes `lead_source.VEHICLE_CLASSES`. `mut`: deliberate-regression hooks for the pin tests only."""
    LS = init_lead()["LS"]
    top = seq[next(iter(seq))]
    td, dims = top["tracks_data"], top["cuboidtracks_data"]["cuboids_dims"]
    et, ex, ey, ew = ego["t"], ego["x"], ego["y"], ego["yaw"]
    cols = {k: [] for k in ("t", "track", "center_x", "center_y", "size_x", "is_vehicle")}
    for i, (tid, lab, P, ts) in enumerate(zip(td["tracks_id"], td["tracks_label_class"], td["tracks_poses"], td["tracks_timestamps_us"])):
        ts = np.asarray(ts, np.float64)
        if ts.size == 0:
            continue
        P = np.asarray(P, np.float64).reshape(-1, 7)
        t = (ts - float(t_first_us)) / 1e6
        tq = t + 0.05 if mut == "ego_time_shift" else t
        yaw, sx, sy = np.interp(tq, et, ew), np.interp(tq, et, ex), np.interp(tq, et, ey)
        c, s = (np.ones_like(yaw), np.zeros_like(yaw)) if mut == "no_rotation" else (np.cos(yaw), np.sin(yaw))
        dx, dy = P[:, 0] - sx, P[:, 1] - sy
        cols["t"].append(t)
        cols["track"].append(np.full(t.size, str(tid)))
        cols["center_x"].append(c * dx + s * dy)
        cols["center_y"].append(-s * dx + c * dy)
        cols["size_x"].append(np.full(t.size, float(dims[i][1 if mut == "width_as_length" else 0])))
        cols["is_vehicle"].append(np.full(t.size, str(lab) in LS.VEHICLE_CLASSES))
    if not cols["t"]:
        return None
    o = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.argsort(o["t"], kind="stable")
    return {k: v[order] for k, v in o.items()}


def tracks_from_obs(obs, ego, ids=None):
    """TEST HELPER: a `sequence_tracks.json`-shaped dict (WORLD frame) from rig-frame `obstacle.offline` cuboids, by the forward chain `lead_track_in_window`
    uses. Times are kept in seconds*1e6 so `obs_from_tracks(..., t_first_us=0)` recovers the same time base."""
    ids = sorted(set(obs["track"].tolist()))
    et, ex, ey, ew = ego["t"], ego["x"], ego["y"], ego["yaw"]
    td = {"tracks_id": [], "tracks_poses": [], "tracks_timestamps_us": [], "tracks_label_class": [], "tracks_flags": []}
    dims = []
    for tid in ids:
        m = obs["track"] == tid
        t = obs["t"][m]
        yaw, sx, sy = np.interp(t, et, ew), np.interp(t, et, ex), np.interp(t, et, ey)
        c, s = np.cos(yaw), np.sin(yaw)
        cx, cy = obs["center_x"][m], obs["center_y"][m]
        wx, wy = sx + cx * c - cy * s, sy + cx * s + cy * c
        td["tracks_id"].append(tid)
        td["tracks_poses"].append([[float(a), float(b), 0.0, 0.0, 0.0, 0.0, 1.0] for a, b in zip(wx, wy)])
        td["tracks_timestamps_us"].append([float(v) * 1e6 for v in t])
        td["tracks_label_class"].append("automobile" if bool(obs["is_vehicle"][m][0]) else "pedestrian")
        td["tracks_flags"].append("NONE")
        dims.append([float(obs["size_x"][m][0]), 1.9, 1.5])
    return {"dummy_chunk_id": {"tracks_data": td, "cuboidtracks_data": {"cuboids_dims": dims}}}


def lead_block_for_starts(obs, ego, t0s, **kw):
    """`lead_source.lead_block` for window origins `t0s` (clip seconds, ego time base) on the banked block's grid: 30 knots at 0.2 s."""
    LS = init_lead()["LS"]
    return LS.lead_block(np.asarray(t0s, np.float64), np.arange(1, K + 1, dtype=np.float64) * DT_LEAD, obs, ego, **kw)


def join_from_block(blk, v0=None):
    """The `Jd` that `DK.join_block` returns (same keys, same values for the LEAD rows), from a freshly built block. `v0` (the rollout's speed) replaces the block's
    ego speed after the same speed proof (|v0 - block speed| <= 1e-3 m/s) the reader applies."""
    DK = init_lead()["DK"]
    state = np.asarray(blk["state"]).astype(str).reshape(-1)
    n, k = state.size, np.asarray(blk["leads"]).shape[1]
    bsp = np.asarray(blk["speeds"], np.float64).reshape(-1)
    sp = bsp if v0 is None else np.asarray(v0, np.float64).reshape(-1)
    dmax = float(np.max(np.abs(sp - bsp))) if n else 0.0
    if not dmax <= LEAD_SPEED_TOL_MPS:
        raise RuntimeError(f"LABEL-FREE SPEED PROOF FAILED: max |v0 - block speed| = {dmax:.4g} m/s > {LEAD_SPEED_TOL_MPS}")
    unk = sorted(set(state.tolist()) - set(DK.STATES))
    if unk:
        raise RuntimeError(f"lead block carries states {unk} outside {DK.STATES}")
    is_lead = state == "LEAD"
    leads = np.full((n, k, 2), np.nan)
    leads[is_lead] = np.asarray(blk["leads"], np.float64)[is_lead]
    lens = np.full(n, np.nan)
    lens[is_lead] = np.asarray(blk["lead_lens"], np.float64).reshape(-1)[is_lead]
    out = {"rows": np.arange(n), "state": state, "leads": leads, "lead_lens": lens, "speeds": sp, "is_lead": is_lead, "speed_proof_max_mps": dmax, "k": int(k),
           "dt": DT_LEAD, "counts": {s: int((state == s).sum()) for s in DK.STATES}, "block_k": int(k)}
    for kk in DK.DK_KEYS:
        out[f"gt_{kk}"] = None
    return out


# ============================================================================================================================ S-POS: the logged route
class Route:
    """The logged route of ONE start on the 0.2 s grid anchored at the start tick: row q = logged tick f0 + 2q. `rows` [M,4] (x, y, yaw, v) world; `rc` [M,3,3]
    float32 (x, y, psi_rad in that row's pose frame, arcs 30/50/80); `rc_ok` [M,3] (v9's valid flag, row exists). Column 0 of `rc_ok` is NaN-safe: rc is 0 where not ok."""

    def __init__(self, rows, rc, rc_ok):
        self.rows, self.rc, self.rc_ok = np.asarray(rows, np.float64), np.asarray(rc, np.float32), np.asarray(rc_ok, bool)


def scene_rc_all(ego):
    """The v9 route checkpoints at EVERY logged tick of a scene, by the v9 builder's own functions on the ego log (`log_from_arrays` -> `clip_geo` ->
    `rc_fields`, RC-A 30/50/80 on the S_8 path, sagitta-checked), assembled exactly as `v8f_mini_data._rc_arrays` does. -> (rc [n,3,3] f32, ok [n,3] bool).
    A tick with less than the arc of logged route ahead is NOT ok (rc 0): the horizon's NaN, counted by the caller."""
    B9 = init_lead()["B9"]
    log = B9.log_from_arrays(ego["t"], ego["x"], ego["y"], ego["yaw"], ego["v"])
    G = B9.clip_geo(log)
    n = len(ego["t"])
    rc = np.full((n, 3, 3), np.nan, np.float32)
    ok = np.zeros((n, 3), bool)
    for i in range(n):
        f = B9.rc_fields(G.tr, G.log, G.paths, G.turns, float(ego["t"][i]), G.s_nat)
        for m, v in enumerate(("A30", "A50", "A80")):
            x, y, p = (float(f[f"rc_{v}_{c}"]) for c in ("x", "y", "psi"))
            va = float(f[f"rc_{v}_valid"])
            good = bool(np.isfinite(va) and va == 1 and np.isfinite(x) and np.isfinite(y) and np.isfinite(p))
            rc[i, m, 0], rc[i, m, 1], rc[i, m, 2] = x, y, np.radians(p)
            ok[i, m] = good
    rc[~ok] = 0.0
    return rc, ok


class SceneLog:
    """One scene: the logged rig track and the NuRec agent tracks. `jd(f0s)` -> the DK join for those starts; `route(f0)` -> the logged route from a start."""

    def __init__(self, gt_T, gt_ts_us, seq=None):
        self.T, self.tus, self.seq = np.asarray(gt_T, np.float64), np.asarray(gt_ts_us, np.int64), seq
        self.ego = ego_from_rig(self.T, self.tus)
        self._obs, self._rc = None, None

    @property
    def obs(self):
        if self._obs is None:
            self._obs = obs_from_tracks(self.seq, self.ego, int(self.tus[0])) if self.seq is not None else None
        return self._obs

    @property
    def rc(self):
        if self._rc is None:
            self._rc = scene_rc_all(self.ego)
        return self._rc

    def jd(self, f0s, v0s=None):
        """DK join for the starts `f0s` (rig ticks), always against the LOGGED agents (the human's lead): the `objects` / `empty` rendering condition changes what the car
        SEES, never what the distance-keeping rows are scored against. A scene with no tracks gives all-NO_LABEL (never free flow by omission), counted."""
        f0s = np.asarray(f0s, np.int64)
        return join_from_block(lead_block_for_starts(self.obs, self.ego, self.ego["t"][f0s]), v0s)

    def route(self, f0):
        idx = np.arange(int(f0), len(self.T), 2)
        e = self.ego
        rows = np.stack([e["x"][idx], e["y"][idx], e["yaw"][idx], e["v"][idx]], 1)
        rc, ok = self.rc
        return Route(rows, rc[idx], ok[idx])


def load_tracks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================================================================ S-POS: errors
def spos_errors(routes, s_ckpt, s_pose, pose_w0, tracks, r_s, mut=None):
    """Per-window S and CTRV checkpoint errors, the way `t1hd_read.analyze_run` computes them, from T2-shaped inputs.
    routes[N]: Route; s_ckpt [N,3,3,3] (tick, arc, x/y/psi) in S's tick frame; s_pose [N,3,3] and pose_w0 [N,3,3] (the same row per tick) world poses; tracks[N]: the executed
    10 Hz track [Nraw,4]; r_s [N,3]: S's cache step in that track's index (raw row 2*r_s).
    Match: the logged route row whose float32 xy is nearest the S pose (first minimum); on-route iff within 3 m. GT: the row's rc moved into the window-origin frame.
    -> dict(err={'S','CTRV','STRAIGHT'}: {'pos_err_m','head_err_deg'} [N,3,3], gt_ok [N,3,3], rS [N,3], dS [N,3], kappa [N,3]). `mut` = test hook."""
    H = init_lead()
    T1H = H["T1H"]
    N = len(routes)
    s_ckpt, s_pose, pose_w0 = np.asarray(s_ckpt, np.float32), np.asarray(s_pose, np.float32), np.asarray(pose_w0, np.float64)
    r_s = np.asarray(r_s, np.int64)
    s_xy = s_pose[..., :2].astype(np.float64)
    rS, dS = np.zeros((N, 3), np.int64), np.zeros((N, 3))
    for w, R in enumerate(routes):
        P = R.rows[:, :2].astype(np.float32).astype(np.float64)
        for k in range(3):
            dd = np.hypot(P[:, 0] - s_xy[w, k, 0], P[:, 1] - s_xy[w, k, 1])
            i = int(np.argmin(dd))
            rS[w, k], dS[w, k] = i, dd[i]
    onS = np.ones_like(dS, bool) if mut == "no_onroute" else dS <= T1H.OFF_ROUTE_M
    src = np.stack([np.stack([routes[w].rows[rS[w, k]][:3] for k in range(3)]) for w in range(N)])
    rc = np.stack([np.stack([routes[w].rc[rS[w, k]] for k in range(3)]) for w in range(N)]).astype(np.float64)
    gt_ok = np.stack([np.stack([routes[w].rc_ok[rS[w, k]] for k in range(3)]) for w in range(N)]) & onS[..., None]
    gt_w0 = rc if mut == "gt_frame" else T1H.to_w0(rc, src, pose_w0)
    kap = np.zeros((N, 3))
    for w in range(N):
        for k in range(3):
            kap[w, k] = T1H.mean_curvature(tracks[w], int(r_s[w, k]))[0]
    if mut == "kap_sign":
        kap = -kap
    ctrv, strt, _ = T1H.ctrv_straight(kap, s_pose.astype(np.float64), pose_w0)
    S_w0 = (s_ckpt.astype(np.float64) if mut == "s_frame" else
            T1H.to_w0_f32(s_ckpt.reshape(-1, 3, 3), s_pose.reshape(-1, 3), pose_w0.reshape(-1, 3)).reshape(N, 3, 3, 3).astype(np.float64))
    err = {nm: T1H.C.checkpoint_errors(arr, gt_w0, gt_ok) for nm, arr in (("S", S_w0), ("CTRV", ctrv), ("STRAIGHT", strt))}
    return {"err": err, "gt_ok": gt_ok, "rS": rS, "dS": dS, "kappa": kap, "onS": onS}


def spos_rows(E, mut=None):
    """{'S-POS@30/50/80': per-window (S - CTRV) position error, mean over ticks >= 1}; NaN where either side has no scored tick (the horizon is not reached, or off route).
    `n_scored[tick][arc]` counts the scored cells; windows with a value are `n_windows_ge1`."""
    T1H = init_lead()["T1H"]
    pS, pC = E["err"]["S"]["pos_err_m"], E["err"]["CTRV"]["pos_err_m"]
    if mut == "tick0_in":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            aS, aC = np.nanmean(pS, axis=1), np.nanmean(pC, axis=1)
    else:
        aS, aC = T1H.per_window(pS)[1], T1H.per_window(pC)[1]
    rows = {}
    for m, arc in enumerate(T1H.ARCS):
        ok = np.isfinite(aS[:, m]) & np.isfinite(aC[:, m])
        rows[f"S-POS@{arc}"] = np.where(ok, aS[:, m] - aC[:, m], np.nan)
    info = {"n_scored": {f"tick{k}": [int(E["gt_ok"][:, k, m].sum()) for m in range(3)] for k in range(3)},
            "n_windows_ge1": [int(np.isfinite(rows[f"S-POS@{a}"]).sum()) for a in T1H.ARCS],
            "off_route_cells": int((~E["onS"]).sum()), "mean_S_pos": [float(np.nanmean(aS[:, m])) for m in range(3)], "mean_CTRV_pos": [float(np.nanmean(aC[:, m])) for m in range(3)]}
    return rows, info


def spos_from_records(recs, scenes):
    """T2 rollout records -> (rows, info). `scenes[scene_idx]` is a SceneLog. Every record needs rec['scene'], rec['f0'], rec['track'], rec['extra'][0..2] with s_ckpt/s_pose/a_s/step."""
    routes, ck, sp, w0, trk, rs = [], [], [], [], [], []
    for r in recs:
        ex = r["extra"]
        routes.append(scenes[r["scene"]].route(r["f0"]))
        ck.append([e["s_ckpt"] for e in ex[:3]])
        sp.append([e["s_pose"] for e in ex[:3]])
        track = np.asarray(r["track"], np.float64)
        w0.append([track[2 * int(r["t0_step"]), :3]] * 3)
        trk.append(track)
        rs.append([int(e["step"]) - int(e["a_s"]) for e in ex[:3]])
    return spos_rows(spos_errors(routes, ck, sp, w0, trk, rs))


def pair_keys(recs):
    """The pairing identity of a rollout set: (scene, f0) per record, in order. Arms are contrasted only when these lists are IDENTICAL."""
    return [(int(r["scene"]), int(r["f0"])) for r in recs]


def lead_jd_for_records(recs, scenes):
    """The DK join for a rollout set, in RECORD order: starts are grouped by scene, each scene's block is built once, the speeds are the rollouts' own v0."""
    DK = init_lead()["DK"]
    n = len(recs)
    parts = {}
    for i, r in enumerate(recs):
        parts.setdefault(int(r["scene"]), []).append(i)
    J = None
    for sc, ii in parts.items():
        j = scenes[sc].jd([recs[i]["f0"] for i in ii], [recs[i]["v0"] for i in ii])
        if J is None:
            J = {"state": np.full(n, "NO_LABEL", dtype="<U12"), "leads": np.full((n, K, 2), np.nan), "lead_lens": np.full(n, np.nan), "speeds": np.zeros(n)}
        J["state"][ii], J["leads"][ii], J["lead_lens"][ii], J["speeds"][ii] = j["state"], j["leads"], j["lead_lens"], j["speeds"]
    J["rows"], J["is_lead"], J["k"], J["dt"], J["block_k"] = np.arange(n), J["state"] == "LEAD", K, DT_LEAD, K
    J["speed_proof_max_mps"] = 0.0
    J["counts"] = {s: int((J["state"] == s).sum()) for s in DK.STATES}
    for kk in DK.DK_KEYS:
        J[f"gt_{kk}"] = None
    return J


def rows_for_records(recs, scenes, name="X"):
    """T2 rollout records of ONE (arm, condition, agents) -> (rows {label: [N]}, missing {label: reason}, info). The 19 T1 rows from the reader's own functions."""
    G = np.stack([np.asarray(r["g"], np.float64) for r in recs])
    X = np.stack([np.asarray(r["cl"], np.float64) for r in recs])
    eid = np.array([int(r["scene"]) for r in recs])
    cx = S.context_from_lead(G, eid, lead_jd_for_records(recs, scenes))
    sp = spos_from_records(recs, scenes) if all("s_ckpt" in r["extra"][0] for r in recs) else None
    rows, missing = S.own_rows(X, G, cx, name, spos=None if sp is None else sp[0])
    return rows, missing, {"eid": eid, "keys": pair_keys(recs), "spos": None if sp is None else sp[1], "state_counts": cx.Jd["counts"]}


# ============================================================================================================================ pins (run in the scorer subprocess)
EGO_DIR = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo"
OBS_DIR = "C:/Users/Admin/tanitad-data/physicalai/labels/obstacle_offline_b1eval"
RAW = f"{TREE}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
SFRESH = f"{S.ART}/sfresh_inputs"
# DAGOACC_READ.json, seed 0, DAGOACC vs CTRV (ticks >= 1): delta, lo, hi at B 2000
BANKED_SPOS = {"S-POS@30": (-1.7428, -2.2737, -1.2861), "S-POS@50": (-4.5331, -5.6695, -3.5422), "S-POS@80": (-9.5624, -11.6843, -7.6124)}
SPOS_LABELS = ("S-POS@30", "S-POS@50", "S-POS@80")


def _eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "US" or b.dtype.kind in "US":
        return bool(np.array_equal(a.astype(str), b.astype(str)))
    return bool(np.array_equal(a.astype(np.float64), b.astype(np.float64), equal_nan=True))


def _dev(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return float("inf")
    f = np.isfinite(a)
    return float(np.abs(a[f] - b[f]).max()) if f.any() else 0.0


def _obs_dev(a, b):
    """Max |center_x|, |center_y| difference of two obs dicts after ordering both by (track, t, size_x): the adapter regroups by track."""
    oa, ob = (np.lexsort((o["center_x"], o["t"], o["track"])) for o in (a, b))
    if len(oa) != len(ob):
        return float("inf")
    return float(max(np.abs(a["center_x"][oa] - b["center_x"][ob]).max(), np.abs(a["center_y"][oa] - b["center_y"][ob]).max()))


def op_dk_pin(clips, arr, dump, floor_dump, mut=None):
    """DK pin on banked eval clips.
    Layer 1 (bit for bit): `lead_block` on the clip's own obstacle.offline + egomotion reproduces the banked block rows and the reader's `join_block` dict.
    Layer 2 (the scene adapter): the same cuboids as a WORLD-frame sequence_tracks dict, converted back to the rig frame, give the same states and leads to 1e-9 m.
    Layer 3 (rows): DK-VIOL / DK-HW / DK-TG from either join, on the banked RIC0_s0 windows of these clips, against the reader's banked arrays."""
    import pandas as pd
    H = init_lead()
    DK, BLB, LS = H["DK"], H["BLB"], H["LS"]
    R = S.init()["R"]
    blk = DK.load_block(S.LEAD_BLOCK)
    cid, frm = blk["clip_id"].astype(str), blk["frame"].astype(np.int64)
    out = {"clips": {}}
    ego_by, obs_by = {}, {}
    for c in clips:
        rows = np.flatnonzero(cid == c)
        ego = BLB.ego_series(pd.read_parquet(f"{EGO_DIR}/{c}.parquet"))
        ob = BLB.obs_series(pd.read_parquet(f"{OBS_DIR}/{c}.parquet"), LS.VEHICLE_CLASSES)
        obs = {k: ob[k] for k in ("t", "track", "center_x", "center_y", "size_x", "is_vehicle")}
        ego_by[c], obs_by[c] = ego, obs
        t0s, spd = blk["t0_s"][rows], blk["speeds"][rows]
        kw = {"max_stale_s": 1.0} if mut == "stale" else {}
        if mut == "ts_spacing":
            b1 = LS.lead_block(t0s, np.arange(1, K + 1) * 0.1, obs, ego)
        else:
            b1 = lead_block_for_starts(obs, ego, t0s, **kw)
        J1 = join_from_block(b1, spd)
        Jr = DK.join_block(blk, np.full(len(rows), c), frm[rows], spd, k=K, dt=DT_LEAD)
        l1 = {k: _eq(J1[k], Jr[k]) for k in ("state", "leads", "lead_lens", "speeds", "is_lead")}
        l1["counts"] = J1["counts"]
        l1["counts_equal"] = J1["counts"] == Jr["counts"]
        o2 = obs_from_tracks(tracks_from_obs(obs, ego), ego, 0, mut=mut)
        J2 = join_from_block(lead_block_for_starts(o2, ego, t0s), spd)
        l2 = {"state_equal": bool((J2["state"] == Jr["state"]).all()), "max_dev_leads_m": _dev(J2["leads"], Jr["leads"]),
              "max_dev_lens_m": _dev(J2["lead_lens"], Jr["lead_lens"]), "leads_bit_identical": _eq(J2["leads"], Jr["leads"]),
              "max_dev_obs_xy_m": _obs_dev(o2, obs)}
        out["clips"][c] = {"n_rows": int(len(rows)), "layer1": l1, "layer2": l2}
    z = np.load(arr, allow_pickle=True)
    labels = [str(q) for q in z["labels"]]
    d, dref = R.load_dir(dump), R.load_dir(floor_dump)
    join = np.load(f"{RAW}/join_eval141.npz", allow_pickle=False)
    wclip = join["clip"].astype(str)[dref["clip_index"]]
    frame = 2 * dref["ws"].astype(np.int64)
    sel = np.flatnonzero(np.isin(wclip, clips))
    sel = np.concatenate([sel[wclip[sel] == c] for c in clips])            # grouped by clip, in the order of `clips`
    out["layer3_n_windows"] = int(len(sel))
    res = {}
    for tag, layer in (("layer1", 1), ("layer2", 2)):
        parts = []
        for c in clips:
            m = sel[wclip[sel] == c]
            if not len(m):
                continue
            t0s = np.array([blk["t0_s"][np.flatnonzero((cid == c) & (frm == frame[i]))[0]] for i in m])
            ob = obs_by[c] if layer == 1 else obs_from_tracks(tracks_from_obs(obs_by[c], ego_by[c]), ego_by[c], 0, mut=mut)
            parts.append(join_from_block(lead_block_for_starts(ob, ego_by[c], t0s), dref["v0"][m]))
        J = {k: np.concatenate([p[k] for p in parts]) for k in ("state", "leads", "lead_lens", "speeds")}
        J.update(rows=np.arange(len(sel)), is_lead=J["state"] == "LEAD", k=K, dt=DT_LEAD, block_k=K, speed_proof_max_mps=0.0,
                 counts={s: int((J["state"] == s).sum()) for s in DK.STATES})
        for kk in DK.DK_KEYS:
            J[f"gt_{kk}"] = None
        cx = S.context_from_lead(dref["g"][sel], dref["eid"][sel], J)
        ra, _ = S.own_rows(d["cl"][sel], d["g"][sel], cx, "RIC0")
        per = {}
        for lab in ("DK-VIOL", "DK-HW", "DK-TG"):
            i = labels.index(lab)
            per[lab] = {"bit_identical": _eq(ra[lab], z[f"a_{i}"][sel]), "max_abs_dev": _dev(ra[lab], z[f"a_{i}"][sel]), "n_finite": int(np.isfinite(ra[lab]).sum())}
        per["state_counts"] = J["counts"]
        res[tag] = per
    out["layer3"] = res
    return out


def op_rc_pin(clips, mut=None):
    """GT pin: the v9 builder's functions on the clip's EGO LOG reproduce the join's `rc` / `rc_ok` (float32) at every cache step with a v9 row, bit for bit; also counts the cells
    that are NOT ok (the horizon the route does not reach), so the NaN path is exercised."""
    H = init_lead()
    B9 = H["B9"]
    J = H["T1H"].D.load_join(f"{RAW}/join_eval141.npz")
    out = {}
    for c in clips:
        ci = int(np.flatnonzero(J["clip"].astype(str) == c)[0])
        off, n = int(J["off"][ci]), int(J["n"][ci])
        log = B9.load_log(f"{EGO_DIR}/{c}.parquet")
        G = B9.clip_geo(log)
        ns, has = J["now_s"][off:off + n], J["has"][off:off + n]
        rc, ok = np.zeros((n, 3, 3), np.float32), np.zeros((n, 3), bool)
        for j in np.flatnonzero(has):
            f = B9.rc_fields(G.tr, G.log, G.paths, G.turns, float(ns[j]), G.s_nat)
            for m, v in enumerate(("A30", "A50", "A80")):
                x, y, p, va = (float(f[f"rc_{v}_{q}"]) for q in ("x", "y", "psi", "valid"))
                good = bool(np.isfinite(va) and va == 1 and np.isfinite(x) and np.isfinite(y) and np.isfinite(p))
                ok[j, m] = good
                if good:
                    rc[j, m] = np.array([x, y, p if mut == "psi_deg" else np.radians(p)], np.float32) + (np.float32(1e-6) if mut == "x_shift" else np.float32(0.0))
        ref_ok, ref_rc = J["rc_ok"][off:off + n] & has[:, None], J["rc"][off:off + n]
        out[c] = {"n_steps": int(has.sum()), "ok_equal": _eq(ok, ref_ok), "rc_bit_identical": _eq(rc * ok[..., None], ref_rc * ref_ok[..., None]),
                  "n_not_ok_cells": int((has[:, None] & ~ref_ok).sum())}
    return out


def op_spos_pin(ticks, mut=None, n_boot=2000):
    """S-POS pin: `t1hd_read.analyze_run` (the reader) on the banked --dump-heads ticks, against `spos_errors` fed the SAME inputs in the T2 record shape (routes from the join,
    the executed track from the reader's `rebuild_track`). Per-window S and CTRV errors must be bit-identical, and the pooled S-POS deltas must equal the banked read's."""
    import types
    T1H = init_lead()["T1H"]
    a = types.SimpleNamespace(eval_join=f"{RAW}/join_eval141.npz", eval_grid=f"{RAW}/eval_grid_windows.json", tracks=f"{SFRESH}/inputs/native_tracks.pt", eval_cache=f"{SFRESH}/cache")
    ctx = T1H.Ctx(a)
    tt = T1H.load_ticks(ticks)
    Rr = T1H.analyze_run(ctx, tt, a, "pin", 100, dump_dir=None)
    J = ctx.J
    N = tt["N"]
    routes, trk = [], []
    for w in range(N):
        ci, t0 = int(tt["ci"][w]), int(tt["t0"][w])
        ctx.clip(ci)
        off, n = int(J["off"][ci]), int(J["n"][ci])
        sl = slice(off + t0, off + n)
        routes.append(Route(ctx.PG[ci][t0:], J["rc"][sl], J["rc_ok"][sl] & J["has"][sl][:, None]))
        trk.append(T1H.rebuild_track(ctx, ci, t0, tt["plan"][w])[0])
    r_s = tt["j"] - tt["age_s"]
    E = spos_errors(routes, tt["s_ckpt"], tt["s_pose"], tt["pose_w0"], trk, r_s, mut=mut)
    rows, info = spos_rows(E, mut=mut)
    out = {"n_windows": int(N)}
    for nm in ("S", "CTRV"):
        for q in ("pos_err_m", "head_err_deg"):
            out[f"{nm}_{q}_bit_identical"] = _eq(E["err"][nm][q], Rr["err"][nm][q])
            out[f"{nm}_{q}_max_dev"] = _dev(E["err"][nm][q], Rr["err"][nm][q])
    out["kappa_bit_identical"] = _eq(E["kappa"], Rr["kappa"])
    out["gt_ok_equal"] = _eq(E["gt_ok"], Rr["gt_ok"])
    ref_rows = T1H.s_rows(Rr, tt["ci"], 100)["decisive"]
    pooled = {}
    for lab in SPOS_LABELS:
        ok = np.isfinite(rows[lab])                     # (S - CTRV) per window vs 0, NaN windows KEPT in the 141-episode pool: the reader's  (same resampling stream)
        pb = T1H.PB(rows[lab], np.where(ok, 0.0, np.nan), tt["ci"], n_boot=n_boot, seed=0)
        pooled[lab] = {"delta": float(pb["delta"]), "lo": float(pb["lo"]), "hi": float(pb["hi"]), "n_windows": int(ok.sum()),
                       "reader_n_windows": int(ref_rows[lab]["n_windows"]), "reader_delta_B100": float(ref_rows[lab]["delta"]), "banked_B2000": list(BANKED_SPOS[lab])}
    out["pooled"] = pooled
    out["info"] = info
    return out


def op_x12(mut=None):
    """The bootstrap over SCENES for X1 / X2 on synthetic starts (42 launch scenes + 20 guard scenes, 2 starts each, scene effects shared by a scene's two starts).
    Truth: C0 = P10 + 0.05 and RR = C0 - 0.03 on M1, so X1 -> +0.05 and X2 -> +0.03. Checks (returned, asserted by the test): clusters are SCENES, the mask selects launch scenes, the
    result equals the reader's own `paired` on the same arrays, it is deterministic, and misaligned starts are refused."""
    T = S.init()["T"]
    rng = np.random.default_rng(7)
    n_sc, per = 62, 2
    scene = np.repeat(np.arange(n_sc), per)
    launch = scene < 42
    eff = rng.normal(0.0, 0.05, n_sc)[scene]
    p10 = 0.20 + eff + rng.normal(0.0, 0.01, scene.size)
    c0 = p10 + 0.05 + rng.normal(0.0, 0.01, scene.size)
    rr = c0 - 0.03 + rng.normal(0.0, 0.01, scene.size)
    rows = lambda v: {S.M1: v}                                       # noqa: E731
    eid = np.arange(scene.size) if mut == "eid_window" else scene
    mask = None if mut == "mask_ignored" else launch
    out = S.x_contrasts(rows(c0), rows(p10), rows(rr), eid, mask, rows_blind=rows(c0 + 0.02 + rng.normal(0, 0.01, scene.size)))
    ref1 = T.paired(c0[launch], p10[launch], scene[launch])
    ref2 = T.paired(c0[launch], rr[launch], scene[launch])
    again = S.x_contrasts(rows(c0), rows(p10), rows(rr), eid, mask)
    guard = True
    keys = [(int(s), 10 * i) for i, s in enumerate(scene)]
    try:
        (lambda *a: None if mut == "no_guard" else S.pair_check(*a))(keys, keys[::-1])
        guard = False
    except ValueError:
        guard = True
    return {"X1": out["X1"], "X2": out["X2"], "BL": out["BL"], "ref_X1": {k: ref1[k] for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes")},
            "ref_X2": {k: ref2[k] for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes")}, "deterministic": out["X1"] == again["X1"] and out["X2"] == again["X2"],
            "guard_refuses_misaligned": guard, "n_launch_scenes": 42, "true_X1": 0.05, "true_X2": 0.03}


def load_rig(path, cam="camera_front_wide_120fov"):
    """`rig_trajectories.json` -> (gt_T [n,4,4], gt_ts_us [n]) at 10 Hz: the reading of `t2_scenes/code/t2lib.ego_from_rig` (the screening rule's own), the arrays the E3 rollout is given."""
    with open(path, "r", encoding="utf-8") as f:
        e = json.load(f)["rig_trajectories"][0]
    key = f"{cam}@{e['sequence_id']}"
    Ts, ts = e["cameras_frame_T_rig_worlds"][key], e["cameras_frame_timestamps_us"][key]
    stride = int(round(1e5 / (ts[1][1] - ts[0][1])))
    idx = range(0, len(Ts), stride)
    return np.stack([np.array(Ts[i][1], np.float64) for i in idx]), np.array([ts[i][1] for i in idx], np.int64)


def oracle_record(sc, scene_idx, f0, mut=None):
    """A rollout record in the T2 shape whose car EXECUTES THE LOGGED TRACK and whose S head emits the v9 route checkpoints of its own tick (an oracle): every error row must read 0 / its
    perfect value, which is how the whole scene -> record -> rows path is checked end to end. `mut` breaks one link."""
    e = sc.ego
    rows = np.stack([e["x"], e["y"], e["yaw"], e["v"]], 1)
    t0s = f0 // 2
    c0, s0 = np.cos(rows[f0, 2]), np.sin(rows[f0, 2])
    d = rows[f0 + 2:f0 + 61:2, :2] - rows[f0, :2][None]
    g = np.stack([c0 * d[:, 0] + s0 * d[:, 1], -s0 * d[:, 0] + c0 * d[:, 1]], 1)
    rc, _ = sc.rc
    extra = []
    for k in range(3):
        i = f0 + 20 * k
        ck = rc[i].astype(np.float64).copy()
        if mut == "ckpt_swapped_xy":
            ck = ck[:, [1, 0, 2]]
        extra.append({"s_ckpt": ck.tolist(), "s_pose": rows[i, :3].astype(np.float32).astype(np.float64).tolist(), "a_s": 0, "step": t0s + 10 * k})
    return {"scene": scene_idx, "f0": f0, "t0_step": t0s, "raw_off": 0, "g": g, "cl": g + (0.5 if mut == "exec_offset" else 0.0), "v0": float(rows[f0, 3]), "track": rows[:f0 + 61], "extra": extra}


def op_records_smoke(scenes, f0s, mut=None):
    """End to end on real downloaded scenes (rig + sequence_tracks): oracle records through `rows_for_records`. Reports the finite counts per row, the state counts of the DK join,
    and the worst deviation from the perfect value of the rows that must be perfect."""
    scs = {}
    for i, p in enumerate(scenes):
        T, tus = load_rig(f"{p}/rig_trajectories.json")
        scs[i] = SceneLog(T, tus, load_tracks(f"{p}/extracted/sequence_tracks.json"))
    recs = [oracle_record(scs[i], i, f0, mut) for i in scs for f0 in f0s]
    rows, missing, info = rows_for_records(recs, scs, "oracle")
    perfect = {"ADE6": 0.0, "speed MAE": 0.0, "along-track MAE": 0.0, "accel MAE": 0.0, "heading MAE": 0.0, "yaw-rate MAE": 0.0, "curvature MAE": 0.0, "cross-track MAE": 0.0,
               "TAC-LAT": 1.0, "TAC-LON": 1.0, "DK-HW": 0.0, "DK-TG": 0.0}
    dev = {k: float(np.nanmax(np.abs(rows[k] - v))) if np.isfinite(rows[k]).any() else None for k, v in perfect.items() if k in rows}
    sp = {k: {"n_finite": int(np.isfinite(rows[k]).sum()), "max": float(np.nanmax(rows[k])) if np.isfinite(rows[k]).any() else None} for k in SPOS_LABELS if k in rows}
    return {"n_records": len(recs), "n_valid": S.n_valid(rows), "missing": missing, "dev_from_perfect": dev, "spos": sp, "state_counts": info["state_counts"], "spos_info": info["spos"],
            "rows_present": sorted(rows)}
