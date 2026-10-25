"""v8F-mini data: the v9 label join onto the 0.2 s grid of the pooled DINOv3-L cache, and the window dataset.

Contract with the S4 cache (``Research/2026-10-17-refav1-hj-mini/code/pull_encode_pooled.py``, read, not edited):
one ``<cache>/<split>/clips/<clip>.pt`` per clip (``torch.load(weights_only=True)``) holding ``pool4x10`` fp8
``[T, 40, 1024]`` (``pool8x20`` optional), ``v / a / steer / yaw_rate`` ``[T]``, ``pose_grid`` ``[T, 4]`` = (x, y, yaw,
v) at v2ep frame ``2j``, ``kin3`` ``[T, 3]`` = (v0, a0, steer0) and ``p10`` ``[T, 8]`` (row 0 NaN), plus
``<cache>/<split>/manifest.jsonl`` (one line per clip: ``clip``, ``n_frames``, ...).

THE JOIN KEY (MEASURED 2026-10-17, ``Research/2026-10-17-v8f-mini/raw/join_check.json``): cache step ``j`` is v2ep
raw frame ``k = 2 j``, and a v9 row is addressed by ``(sid, k)`` with ``sid = blake2b-8(clip_id) >> 1`` -- the
release's ``k`` IS the v2ep frame index (``v_now_ms[k] - poses[k, 3]``: mean 0.0013, |max| 0.022 m/s on 204
windows). A window with NOW at cache step ``t`` therefore reads v9 row ``k = 2 t``.

KINEMATIC FLOOR UNITS (orchestrator correction 2026-10-17): ``actions[:, 0]`` is a road-wheel STEER ANGLE, so every
path integration here goes through ``taniteval.floors_kin`` with ``action_units="steer"``. The banked refav1 kd_x
/ P10 integrated it as a curvature (ha0_ext 5.356 m vs 2.838 m); they are NOT used as the residual base or as the
bar. Nothing here re-implements a floor.

THE YAW-RATE MODE (PI RULING D-PI-YAWRATE-1, 2026-10-17 -- ``kin_mode="yaw"``): *"the model should get measured ego motion
and its history, instead of steering angle which represents a sort of action, the model should get the measured yaw rate"*.
``actions[:, 0]`` is ``atan(2.9 * measured curvature)`` -- measured motion in an action's units -- so the steer-mode inputs
(``kin3[2]``, the three P10 features that multiply it, the kd_x base integrated from it, the ``steer`` action channel) are
RE-EXPRESSED, not deleted: the third kinematic channel becomes ``kappa0 = clamp(omega0 / max(v0, 2), +-0.3)`` with ``omega0`` the
measured yaw rate = the ONE-FRAME (0.1 s) BACKWARD wrapped difference of the pose-track heading, the action channel becomes ``omega``
itself, and the base is integrated as a CURVATURE (``action_units="kappa"``). ONE spelling: all of it comes from
``tanitad.models.ego_history.ego_channels_from_poses`` (+ ``kinematic_prior._pose_kappa0``), the functions ``p10_pose`` and the
ego-history encoder already use, on the NATIVE 10 Hz pose track (``a0`` is the one-frame difference too, exactly as ``p10_pose``).
Why the native track and not the cache's 0.2 s ``pose_grid`` (MEASURED 2026-10-17, ``p10_yaw_dt_ablation_540.json``): the P10
fitted on the grid-derived yaw scores 2.5944 m at 6 s on eval-141, the one on the native-track yaw 2.5627 m (steer P10-540: 2.5560)
-- the coarser difference costs 0.032 m, MORE than the P10 training-seed floor of 0.0213 m. The S4 cache holds only the even frames,
so the odd ones come from a ``native_tracks.pt`` sidecar (``stack/scripts/build_native_tracks.py``: the raw poses of every train
and eval clip, keyed by clip id); ``ClipStore`` / ``ra1_pooled.load_clip`` REFUSE a track that does not reproduce the cache's own
``pose_grid`` on its even frames. In yaw mode nothing here opens ``actions`` or the cache's ``steer`` / ``kin3`` / ``p10`` /
``yaw_rate`` (the cache's ``yaw_rate`` is a FORWARD difference: at the NOW frame it would read the first future pose).
The default stays ``"steer"`` so every banked artifact reproduces; the trainers default to ``yaw``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

N_TOK, D_FEAT, HORIZON = 40, 1024, 30
DT = 0.2
CKPT_ARCS = (30, 50, 80)
RC_VARIANTS = ("A30", "A50", "A80")
GOAL22 = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "YIELD", "STOP_POINT",
          "SPEED_BAND", "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR", "OVERTAKE_VEHICLE", "MERGE", "GAP_TARGET",
          "REACT_ON_ONCOMING", "TAKE_EXIT_L", "TAKE_EXIT_R", "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED",
          "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN", "LANE_CHANGE_L", "LANE_CHANGE_R")
LAT9 = ("LANE_KEEP", "TURN_L", "TURN_R", "LANE_CHANGE_L", "LANE_CHANGE_R")
LON9 = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP")
LAT_CONSTRAINTS = ("lat_theta_deg", "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m", "turn_dyaw_deg",
                   "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m", "turn_exit_psi_deg", "lc_t_cross_s", "lc_d_cross_m")
LON_CONSTRAINTS = ("lon_v_target_ms", "lon_t_reach_s", "lon_d_reach_m", "lon_t_start_s", "stop_x_m", "stop_y_m",
                   "lead_gap_m", "lead_tg_s", "lead_gap_min_m", "lead_tg_min_s")
SPEED_GOAL = ("v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
CONS_FIELDS = LAT_CONSTRAINTS + LON_CONSTRAINTS + SPEED_GOAL           # 26 regression cells (NaN = undefined)
CONS_IDX = {n: i for i, n in enumerate(CONS_FIELDS)}
IGNORE = -100
LIMITS_KMH = (30, 50, 100, 120)
ACCEPTED_TRAIN_MD5 = {"f63ece410b725febb8a5242cf2b01d3c": "v9 train, 4,369 clips",
                      "83e30bd3791f0767707eade5f4671759": "v9 train, 4,572 clips (superset; 4,369 bitwise identical)"}
ACCEPTED_EVAL_MD5 = {"6b5c7f207cffc3b7eb3cd527fd433599": "v9 eval139"}


def stable_episode_id(clip_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(clip_id.encode("utf-8"), digest_size=8).digest(), "big") >> 1


def sha12(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode("utf-8")).hexdigest()[:12]


def _fk():
    from taniteval import floors_kin
    return floors_kin


# ------------------------------------------------------------------------------------------------ small readers
def read_jsonl(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load_speed_sidecar(path) -> dict:
    """``refcv6_speed_max_v8_{train,eval}.jsonl`` -> ``{clip_id: (limit_kmh, one_hot[4], valid)}``. INPUT, never a
    training sign (the file's own meta says so); a row with ``valid == 0`` or ``one_hot`` missing is all-zero."""
    out = {}
    for r in read_jsonl(path):
        oh = np.asarray(r.get("one_hot") or [0, 0, 0, 0], np.float32)
        ok = int(r.get("valid", 1)) == 1 and oh.sum() > 0
        out[r["clip_id"]] = (int(r["limit_kmh"]) if ok else 0, oh if ok else np.zeros(4, np.float32), int(ok))
    return out


def read_manifest(cache_dir, split) -> list:
    """The S4 manifest -> ``[(clip, n_frames)]`` in file order (last line per clip wins)."""
    p = Path(cache_dir) / split / "manifest.jsonl"
    ents = {}
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln)
            ents[d["clip"]] = int(d["n_frames"])
    return list(ents.items())


def select_train_clips(clips, frac: float) -> list:
    """Nested subsets for H-DATA-SCALE-1: order by sha12, take the first ``round(frac * N)`` (25 % < 50 % < 100 %)."""
    order = sorted(clips, key=sha12)
    n = len(order) if frac >= 1.0 else max(1, int(round(frac * len(order))))
    return order[:n]


def assert_disjoint(train_clips, eval_clips) -> dict:
    """TRAIN/EVAL disjoint by clip id AND by stable sid. (Recording-session disjointness is enforced upstream by the
    S4 cache builder's parity gate -- the clip id does not carry the session.)"""
    t, e = set(train_clips), set(eval_clips)
    if t & e:
        raise AssertionError(f"train/eval clip ids overlap: {sorted(t & e)[:3]} ({len(t & e)})")
    ts, es = {stable_episode_id(c) for c in t}, {stable_episode_id(c) for c in e}
    if ts & es:
        raise AssertionError("train/eval stable sids overlap")
    return {"n_train": len(t), "n_eval": len(e), "overlap": 0}


# ------------------------------------------------------------------------------------------------ the join
def _rc_arrays(R, rows, ok_rows):
    """RC-A {30, 50, 80} -> ``[n, 3, 3]`` (x, y, psi_rad) and ``[n, 3]`` valid, for release rows ``rows``."""
    n = len(rows)
    rc = np.full((n, 3, 3), np.nan, np.float32)
    ok = np.zeros((n, 3), bool)
    for m, v in enumerate(RC_VARIANTS):
        x, y, p = (np.asarray(R[f"rc_{v}_{c}"], np.float64)[rows] for c in ("x", "y", "psi"))
        va = np.asarray(R[f"rc_{v}_valid"], np.float64)[rows]
        good = np.isfinite(va) & (va == 1) & np.isfinite(x) & np.isfinite(y) & np.isfinite(p) & ok_rows
        rc[:, m, 0], rc[:, m, 1], rc[:, m, 2] = x, y, np.radians(p)
        ok[:, m] = good
    rc[~ok] = 0.0
    return rc, ok


def build_label_join(clips_nframes, release, lead_npz=None, speed=None, *, require_labels: bool = False) -> dict:
    """Join a v9 release onto the cache grid. ``clips_nframes`` = ``[(clip_id, n_frames)]`` (cache order).

    Output (concatenated over clips; ``off[c]:off[c]+n[c]`` are clip ``c``'s steps j = 0..n-1):
      ``has`` (a v9 row exists at k = 2j), ``now_s``, ``nav`` ``[N, 5]`` = (FOLLOW, TURN_L, TURN_R one-hot,
      d_next/100 clipped to [0, 3], args_valid), ``nav_d`` (m, NaN if invalid), ``rc`` / ``rc_ok``, ``goal_y`` /
      ``goal_w`` (uint32), ``lat_cls`` / ``lat_allowed`` / ``lon_cls`` / ``lon_allowed`` (variant a, v9 ids),
      ``cons`` ``[N, 26]`` (NaN = undefined), ``lead_valid``, ``lead_gap`` (m at step j, NaN if no lead data),
      ``limit`` ``[C, 4]`` one-hot, ``limit_kmh`` ``[C]``, ``limit_ok`` ``[C]``.
    ONLY the INPUT fields of the nav block feed the model (``NAV_FORBIDDEN_AS_INPUT`` are never read)."""
    R = release.rows
    C = len(clips_nframes)
    n = np.array([nf for _, nf in clips_nframes], np.int64)
    off = np.concatenate([[0], np.cumsum(n)[:-1]])
    N = int(n.sum())
    J = {"clip": np.array([c for c, _ in clips_nframes]), "off": off, "n": n,
         "has": np.zeros(N, bool), "now_s": np.full(N, np.nan), "nav": np.zeros((N, 5), np.float32),
         "nav_d": np.full(N, np.nan, np.float32), "rc": np.zeros((N, 3, 3), np.float32), "rc_ok": np.zeros((N, 3), bool),
         "goal_y": np.zeros(N, np.uint32), "goal_w": np.zeros(N, np.uint32),
         "lat_cls": np.full(N, IGNORE, np.int16), "lat_allowed": np.zeros(N, np.uint8),
         "lon_cls": np.full(N, IGNORE, np.int16), "lon_allowed": np.zeros(N, np.uint8),
         "cons": np.full((N, len(CONS_FIELDS)), np.nan, np.float32), "lead_valid": np.zeros(N, np.int8),
         "lead_gap": np.full(N, np.nan, np.float32), "limit": np.zeros((C, 4), np.float32),
         "limit_kmh": np.zeros(C, np.int16), "limit_ok": np.zeros(C, bool)}
    lead_by = {}
    if lead_npz is not None:
        z = lead_npz if isinstance(lead_npz, dict) else np.load(lead_npz, allow_pickle=True)
        sh, kk, ld, gp = z["sha12"], z["k"], z["lead"], z["gap"]
        for s12 in np.unique(sh):
            m = sh == s12
            lead_by[str(s12)] = dict(zip(kk[m].tolist(), np.where(ld[m] == 1, gp[m], np.nan).tolist()))
    n_missing = 0
    for ci, (cid, nf) in enumerate(clips_nframes):
        sid = stable_episode_id(cid)
        if speed is not None and cid in speed:
            kmh, oh, ok = speed[cid]
            J["limit"][ci], J["limit_kmh"][ci], J["limit_ok"][ci] = oh, kmh, bool(ok)
        meta = release.clips.get(sid)
        if meta is None:
            n_missing += 1
            continue
        r0, nrows, k0 = meta
        j = np.arange(nf)
        k = 2 * j
        inside = (k >= k0) & (k < k0 + nrows)
        rows = r0 + np.clip(k - k0, 0, nrows - 1)
        sl = slice(int(off[ci]), int(off[ci]) + nf)
        assert np.all(np.asarray(R["k"])[rows[inside]] == k[inside]), f"v9 row/k mismatch in clip {cid}"
        J["has"][sl] = inside
        J["now_s"][sl] = np.where(inside, np.asarray(R["now_s"], np.float64)[rows], np.nan)
        tok = np.asarray(R["nav_token"], np.float64)[rows].astype(np.int64)
        valid = np.asarray(R["nav_args_valid"], np.float64)[rows] == 1
        dn = np.asarray(R["nav_d_next_m"], np.float64)[rows]
        dn_ok = valid & np.isfinite(dn)
        nav = np.zeros((nf, 5), np.float32)
        nav[np.arange(nf), np.clip(tok, 0, 2)] = 1.0
        nav[:, 3] = np.where(dn_ok, np.clip(dn / 100.0, 0.0, 3.0), 0.0)
        nav[:, 4] = dn_ok
        nav[~inside] = 0.0
        J["nav"][sl] = nav
        J["nav_d"][sl] = np.where(inside & dn_ok, dn, np.nan)
        rc, rok = _rc_arrays(R, rows, inside)
        J["rc"][sl], J["rc_ok"][sl] = rc, rok
        J["goal_y"][sl] = np.where(inside, np.asarray(R["goal_y"], np.uint64)[rows], 0).astype(np.uint32)
        J["goal_w"][sl] = np.where(inside, np.asarray(R["goal_w"], np.uint64)[rows], 0).astype(np.uint32)
        for nm, key in (("lat_cls", "lat_cls_a"), ("lon_cls", "lon_cls")):
            J[nm][sl] = np.where(inside, np.asarray(R[key])[rows], IGNORE)
        J["lat_allowed"][sl] = np.where(inside, np.asarray(R["lat_allowed_a"])[rows], 0)
        J["lon_allowed"][sl] = np.where(inside, np.asarray(R["lon_allowed"])[rows], 0)
        for q, f in enumerate(CONS_FIELDS):
            J["cons"][sl, q] = np.where(inside, np.asarray(R[f], np.float64)[rows], np.nan)
        J["lead_valid"][sl] = np.where(inside, np.asarray(R["lead_valid"])[rows], 0)
        lb = lead_by.get(sha12(cid))
        if lb is not None:
            J["lead_gap"][sl] = np.array([lb.get(int(kk_), np.nan) for kk_ in k], np.float32)
    J["n_clips_missing_labels"] = n_missing
    if require_labels and n_missing:
        raise ValueError(f"{n_missing} cache clips have no v9 rows")
    return J


def join_report(J: dict, release=None) -> dict:
    """Counts for the label-join report (exact overlap, coverage, goal base rates)."""
    has = J["has"]
    c_has = np.array([has[o:o + n].any() for o, n in zip(J["off"], J["n"])])
    gy, gw = J["goal_y"].astype(np.int64), J["goal_w"].astype(np.int64)
    rep = {"n_cache_clips": int(len(J["n"])), "n_clips_with_v9_rows": int(c_has.sum()),
           "n_clips_without": [str(c) for c in J["clip"][~c_has]], "n_steps": int(len(has)),
           "n_steps_with_row": int(has.sum()), "speed_sidecar_clips": int(J["limit_ok"].sum()),
           "goal_base_rate": {g: float((((gy >> i) & 1) * ((gw >> i) & 1))[has].mean()) if has.any() else float("nan")
                              for i, g in enumerate(GOAL22)},
           "rc_ok_rate": [float(J["rc_ok"][has][:, m].mean()) if has.any() else float("nan") for m in range(3)]}
    return rep


def save_join(J: dict, path):
    np.savez_compressed(path, **{k: v for k, v in J.items() if isinstance(v, np.ndarray)},
                        n_clips_missing_labels=np.int64(J["n_clips_missing_labels"]))


def load_join(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        J = {k: z[k] for k in z.files}
    J["n_clips_missing_labels"] = int(J["n_clips_missing_labels"])
    return J


# ------------------------------------------------------------------------------------------------ kinematic floors
KIN_MODES = ("steer", "yaw")
#: ``floors_kin`` action unit the third kinematic channel is integrated in, per mode
KIN_UNITS = {"steer": "steer", "yaw": "kappa"}
#: the ``meta['units']`` a P10 artifact must declare for each mode (``p10_prior.P10_UNITS_TO_VARIANT`` spelling)
P10_META_UNITS = {"steer": "steer", "yaw": "pose_kappa"}
#: the name of the lateral-control channel (action channel 1, the SENS read) per mode
CH1_NAMES = {"steer": "steer", "yaw": "yaw_rate"}


def check_kin_mode(kin_mode: str) -> str:
    if kin_mode not in KIN_MODES:
        raise ValueError(f"kin_mode must be one of {KIN_MODES}, got {kin_mode!r}")
    return kin_mode


DT_NATIVE = 0.1       # the v2ep / pose-track frame interval (the ego-history encoder's, p10_pose's)


def ego_channels_native(track) -> np.ndarray:
    """``[n, 3]`` float64 = (v, dv/dt, dyaw/dt) of the NATIVE 10 Hz pose track ``[n, 4]`` (x, y, yaw, v): THE ONE SPELLING --
    ``tanitad.models.ego_history.ego_channels_from_poses`` (one-frame BACKWARD differences, yaw wrapped, row 0 = 0 rate), called,
    not re-implemented. Row ``k`` reads track rows ``k`` and ``k - 1`` only. A NaN pose row gives NaN."""
    import torch
    from tanitad.models.ego_history import ego_channels_from_poses
    tr = torch.as_tensor(np.asarray(track, np.float64))[None]
    return ego_channels_from_poses(tr, tr.shape[1], dt=DT_NATIVE)[0].numpy()


def yaw_kin_from_track(track) -> tuple:
    """``(kin3 [T, 3], omega [T])`` in the YAW mode for a native track with ``n`` frames (``T = ceil(n / 2)`` cache steps; step ``j``
    is raw frame ``2 j``). ``kin3[j]`` = (v0, a0, kappa0) for a window whose last observed frame is ``2 j``: ``v0 = v[2j]``,
    ``a0`` = the ONE-frame backward speed difference, ``kappa0 = kinematic_prior._pose_kappa0(omega0, v0)`` =
    ``clamp(omega0 / max(v0, 2.0), +-0.3)`` with ``omega0`` the ONE-frame wrapped yaw difference / 0.1 s -- ``p10_pose``'s feature
    triple (``kinematic_prior.pose_kin3``), nothing recorded after frame ``2 j`` and no steer / action channel. Row 0 is NaN (as the
    cache's); ``omega[j]`` is the measured yaw rate (rad/s) at frame ``2 j`` (row 0 = 0.0, ``ego_channels_from_poses``' convention)."""
    import torch
    from tanitad.models.kinematic_prior import _pose_kappa0
    ch = ego_channels_native(track)[0::2]
    kap = _pose_kappa0(torch.from_numpy(ch[:, 2].copy()), torch.from_numpy(ch[:, 0].copy())).numpy()
    kin = np.stack([ch[:, 0], ch[:, 1], kap], 1)
    kin[0] = np.nan
    return kin, ch[:, 2].copy()


def load_native_tracks(path) -> dict:
    """``native_tracks.pt`` (``stack/scripts/build_native_tracks.py``) -> ``{clip_id: float64 [n_raw, 4]}`` (raw frame-indexed
    (x, y, yaw, v); the first two rows of a TRAIN clip are NaN: the v2 manifest starts at raw frame 2)."""
    import torch
    o = torch.load(path, map_location="cpu", weights_only=True)
    return {c: t.double().numpy() for c, t in o["tracks"].items()}


def check_track_matches_grid(track, pose_grid, clip: str = "?", tol: float = 1e-4) -> int:
    """REFUSE a native track that is not THIS clip's: its even frames must reproduce the cache's ``pose_grid`` wherever the track
    is finite (the track is float32 of the same raw poses => expected max |diff| 0). Returns the number of rows compared."""
    pg = np.asarray(pose_grid, np.float64)
    ev = np.asarray(track, np.float64)[0::2]
    if len(ev) != len(pg):
        raise ValueError(f"{clip}: native track has {len(track)} frames => {len(ev)} even rows, cache pose_grid has {len(pg)}")
    m = np.isfinite(ev).all(1)
    if m.sum() < 0.9 * len(pg):
        raise ValueError(f"{clip}: only {int(m.sum())}/{len(pg)} finite even track rows")
    err = float(np.abs(ev[m] - pg[m]).max())
    if not err <= tol:
        raise ValueError(f"{clip}: native track disagrees with the cache pose_grid (max |diff| {err:.3g} > {tol})")
    return int(m.sum())


def kdx_base(kin3, units: str = "steer") -> np.ndarray:
    """kd_x on the 30 x 0.2 s grid from ``kin3`` rows ``[n, 3]``: (v0, a0, steer0) with ``units="steer"`` (the default, CORRECT
    unit for the recorded channel) or (v0, a0, kappa0) with ``units="kappa"`` (the yaw mode). Row with NaN (cache row 0) -> zeros."""
    fk = _fk()
    kin3 = np.asarray(kin3, np.float64)
    bad = ~np.isfinite(kin3).all(1)
    k = np.where(bad[:, None], 0.0, kin3)
    ha0 = fk.ha0_paths(k[:, 0], DT, HORIZON)
    hx = fk.ha0_ext_paths(k, DT, HORIZON, action_units=units)
    base = fk.kd_x(ha0, hx).astype(np.float32)
    base[bad] = 0.0
    return base


def gt_future(pose_grid: np.ndarray) -> tuple:
    """GT ego-frame waypoints ``[T, 30, 2]`` at steps t+1..t+30 and an ok mask ``[T]`` (needs t + 30 <= T - 1)."""
    fk = _fk()
    T = len(pose_grid)
    out = np.zeros((T, HORIZON, 2), np.float32)
    ok = np.zeros(T, bool)
    if T > HORIZON:
        t0 = np.arange(T - HORIZON)
        out[t0] = fk.gt_waypoints(pose_grid, t0, np.arange(1, HORIZON + 1)).astype(np.float32)
        ok[t0] = True
    return out, ok


def human_arc(pose_grid: np.ndarray) -> np.ndarray:
    """``s_h[t, i]`` = the human's path length from step t to t+1+i (m), ``[T, 30]`` (NaN where the future ends)."""
    T = len(pose_grid)
    seg = np.linalg.norm(np.diff(pose_grid[:, :2].astype(np.float64), axis=0), axis=1)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    out = np.full((T, HORIZON), np.nan, np.float32)
    for t in range(max(T - HORIZON, 0)):
        out[t] = cs[t + 1:t + 1 + HORIZON] - cs[t]
    return out


def full_from_manifest(poses_m, actions_m):
    """A v2 MANIFEST row ``i`` is v2ep raw frame ``i + 2`` (n_stack - 1; MEASURED bit-identical on eval139): rebuild
    frame-indexed arrays with two NaN rows in front so that ``arr[k]`` is raw frame ``k`` (k >= 2 is real)."""
    P = np.concatenate([np.full((2, 4), np.nan), np.asarray(poses_m, np.float64)])
    A = np.concatenate([np.full((2, 2), np.nan), np.asarray(actions_m, np.float64)])
    return P, A


def window_kin(poses, actions, ts, kin_mode: str = "steer"):
    """(kin3 [n, 3], gt [n, 30, 2], pose_grid [Tg, 4]) for cache steps ``ts`` of one clip.
    ``kin_mode="steer"`` (default; computed EXACTLY as the S4 cache does): v0 = poses[2t, 3]; a0 = (v[2t] - v[2t-2]) / 0.2
    (backward); steer0 = actions[2t, 0]; GT from the 0.2 s pose grid ``poses[0::2]``.
    ``kin_mode="yaw"`` (PI ruling D-PI-YAWRATE-1): ``poses`` must be the NATIVE raw-frame track; the kinematics are
    ``yaw_kin_from_track`` (``p10_pose``'s triple: ONE-frame a0, kappa0 from the measured yaw rate); ``actions`` is NOT READ
    (may be ``None``)."""
    fk = _fk()
    check_kin_mode(kin_mode)
    ts = np.asarray(ts, np.int64)
    grid = poses[0::2]
    if kin_mode == "yaw":
        kin = yaw_kin_from_track(poses)[0][ts]
    else:
        k = 2 * ts
        kin = np.stack([poses[k, 3], (poses[k, 3] - poses[k - 2, 3]) / DT, actions[k, 0]], 1)
    gt = fk.gt_waypoints(grid, ts, np.arange(1, HORIZON + 1))
    return kin, gt, grid


# ------------------------------------------------------------------------------------------------ the clip store
class ClipStore:
    """Lazy per-clip cache reader with an LRU. ``get(clip)`` returns float32/float16 numpy views + derived arrays."""

    def __init__(self, cache_dir, split, lru: int = 64, grid: str = "pool4x10", kin_mode: str = "steer", tracks: dict | None = None):
        self.dir = Path(cache_dir) / split / "clips"
        self.lru, self.grid, self._c = int(lru), grid, {}
        self.kin_mode = check_kin_mode(kin_mode)
        self.tracks = tracks
        if self.kin_mode == "yaw" and not tracks:
            raise ValueError("kin_mode 'yaw' needs the native pose tracks (native_tracks.pt: --tracks); the cache holds only the even frames")

    def get(self, clip: str) -> dict:
        if clip in self._c:
            self._c[clip] = self._c.pop(clip)
            return self._c[clip]
        import torch
        o = torch.load(self.dir / f"{clip}.pt", map_location="cpu", weights_only=True)
        pg = o["pose_grid"].numpy().astype(np.float64)
        if self.kin_mode == "yaw":
            # PI ruling D-PI-YAWRATE-1: kinematics, P10 features and the lateral action channel come from the NATIVE POSE TRACK.
            # The cache's steer / kin3 / p10 / yaw_rate are NOT READ (kin3 / p10 carry the recorded steer; yaw_rate is a FORWARD
            # difference that would read the first future pose at the NOW frame).
            if clip not in self.tracks:
                raise KeyError(f"{clip}: no native pose track in the sidecar")
            check_track_matches_grid(self.tracks[clip], pg, clip)
            kin3, omega = yaw_kin_from_track(self.tracks[clip])
            p10 = np.full((len(pg), 8), np.nan, np.float32)
            p10[1:] = _fk().kin_features(kin3[1:]).astype(np.float32)
            d = {"feats": o[self.grid].to(torch.float16).numpy(), "pose": pg, "kin3": kin3, "p10f": p10, "a": o["a"].numpy(),
                 "ch1": np.nan_to_num(omega, nan=0.0).astype(np.float32), "v": o["v"].numpy()}   # undefined step -> 0.0 (see ra1_pooled)
            d["base"] = kdx_base(d["kin3"], KIN_UNITS["yaw"])
        else:
            d = {"feats": o[self.grid].to(torch.float16).numpy(), "pose": pg, "kin3": o["kin3"].numpy().astype(np.float64),
                 "p10f": o["p10"].numpy().astype(np.float32), "a": o["a"].numpy(), "steer": o["steer"].numpy(),
                 "v": o["v"].numpy()}
            d["ch1"] = d["steer"]
            d["base"] = kdx_base(d["kin3"])
        d["gt"], d["gt_ok"] = gt_future(pg)
        d["s_h"] = human_arc(pg)
        T = len(pg)
        assert d["feats"].shape[0] == T, f"{clip}: feats {d['feats'].shape[0]} != pose {T}"
        self._c[clip] = d
        while len(self._c) > self.lru:
            self._c.pop(next(iter(self._c)))
        return d


def _lab_pack(J, i):
    """One join row -> flat dict of numpy arrays."""
    gy, gw = int(J["goal_y"][i]), int(J["goal_w"][i])
    return {"has": bool(J["has"][i]), "nav": J["nav"][i],
            "goal_y": np.array([(gy >> b) & 1 for b in range(22)], np.float32),
            "goal_w": np.array([(gw >> b) & 1 for b in range(22)], np.float32),
            "lat_cls": int(J["lat_cls"][i]), "lat_al": np.array([(int(J["lat_allowed"][i]) >> b) & 1 for b in range(5)], bool),
            "lon_cls": int(J["lon_cls"][i]), "lon_al": np.array([(int(J["lon_allowed"][i]) >> b) & 1 for b in range(7)], bool),
            "cons": J["cons"][i], "rc": J["rc"][i], "rc_ok": J["rc_ok"][i], "lead_valid": int(J["lead_valid"][i])}


WINDOW_START_ABSENT = -1       # the trainer flag's / the config field's default: "not set" = hist - 1, today's behaviour


def effective_window_start(hist: int, window_start=None) -> int:
    """The cache step ``t`` of a clip's FIRST training window, and the ``t`` below which a held tick has no real frame behind it (PREREG_V8F_HIST Amendment 1).

    ABSENT (``None`` or ``WINDOW_START_ABSENT``) = ``hist - 1``, i.e. the first window whose ``hist`` frames are all real: exactly what ``train_windows`` and
    ``V8FMiniWindows.item`` did before the flag existed, so every arm that does not set it is bit-identical. SET, it must lie in ``[0, hist - 1]``: a start
    above ``hist - 1`` would be a different lever (dropping windows), a negative one has no meaning. ``--hist 10 --window-start 3`` therefore keeps the window list and
    the held-age clamp of ``--hist 4``, and the readers see 10 frames of which the ones before the clip are padded (``frame_ok`` False)."""
    hist = int(hist)
    if window_start is None or int(window_start) == WINDOW_START_ABSENT:
        return hist - 1
    ws = int(window_start)
    if not 0 <= ws <= hist - 1:
        raise ValueError(f"window_start must be {WINDOW_START_ABSENT} (absent = hist - 1) or in [0, hist - 1] = [0, {hist - 1}], got {ws}")
    return ws


class V8FMiniWindows:
    """Window dataset. A window = ``(clip_index, t, a_T, a_S)``; NOW = cache step ``t``.

    ``hmax`` frames ``t-hmax+1 .. t`` of scene tokens are returned (clamped at 0; ``frame_ok`` marks real ones);
    T's inputs are read at ``t - a_T`` and S's at ``t - a_S`` (SAMPLE-AND-HOLD, ages in 0.2 s cache steps; the true
    ticks are 0.5 s / 1.0 s -- see RESULT.md I-ages). A tick frame's labels come from the v9 row at that frame.

    ``window_start`` (PREREG_V8F_HIST Amendment 1; absent = ``hist - 1``) is the ``t`` from which a held age may reach back: ``room = max(t - window_start, 0)``.
    It is the SAME number ``train_windows(start=...)`` starts each clip at, so the window list and the held-age clamp move together."""

    def __init__(self, store: ClipStore, J: dict, windows, *, hist: int = 4, hmax: int = 10, max_age_t: int = 2,
                 max_age_s: int = 6, window_start=None):
        self.store, self.J, self.windows = store, J, list(windows)
        self.hist, self.hmax = int(hist), int(hmax)
        self.window_start = effective_window_start(self.hist, window_start)
        self.max_age_t, self.max_age_s = int(max_age_t), int(max_age_s)
        self.clip_index = {c: i for i, c in enumerate(J["clip"].tolist())}
        assert self.hmax >= self.hist + self.max_age_s, "hmax must cover hist + the largest S age"

    def __len__(self):
        return len(self.windows)

    def row(self, ci: int, j: int) -> int:
        return int(self.J["off"][ci]) + int(j)

    def item(self, ci: int, t: int, a_t: int = 0, a_s: int = 0) -> dict:
        clip = str(self.J["clip"][ci])
        d = self.store.get(clip)
        T = len(d["pose"])
        room = max(t - self.window_start, 0)                     # window_start = hist - 1 unless set: a tick frame needs `hist` real frames behind it
        a_t = int(min(a_t, room, self.max_age_t))
        a_s = int(max(a_t, min(a_s, room, self.max_age_s)))     # nested: S ticked no later than T did
        idx = np.arange(t - self.hmax + 1, t + 1)
        okf = idx >= 0
        idc = np.clip(idx, 0, T - 1)
        p10f = np.nan_to_num(d["p10f"][idc], nan=0.0)
        okf &= np.isfinite(d["p10f"][idc]).all(1) | (idx < 1)
        rows = [self.row(ci, t), self.row(ci, t - a_t), self.row(ci, t - a_s)]
        J = self.J
        it = {"feats": d["feats"][idc], "frame_ok": okf, "p10f": p10f,
              "act": np.stack([np.nan_to_num(d["a"][idc]), d["ch1"][idc]], -1).astype(np.float32),
              "nav": J["nav"][[self.row(ci, int(j)) for j in idc]], "pose": d["pose"][idc][:, :3].astype(np.float32),
              "limit": J["limit"][ci], "limit_ok": bool(J["limit_ok"][ci]),
              "gt": d["gt"][t], "gt_ok": bool(d["gt_ok"][t]), "base": d["base"][t], "kin3": d["kin3"][t].astype(np.float32),
              "p10_t": d["p10f"][t], "s_h": d["s_h"][t],
              "lead_gap_h": J["lead_gap"][self.row(ci, 0) + np.minimum(np.arange(t + 1, t + 1 + HORIZON), T - 1)],
              "a_t": a_t, "a_s": a_s, "t": t, "ci": ci, "T": T,
              "lab": _lab_pack(J, rows[0]), "lab_t": _lab_pack(J, rows[1]), "lab_s": _lab_pack(J, rows[2])}
        # geometry between the tick frames and NOW (measured odometry): used to re-express held outputs in the NOW frame
        it["pose_now"] = d["pose"][t][:3].astype(np.float32)
        it["pose_t"] = d["pose"][t - a_t][:3].astype(np.float32)
        it["pose_s"] = d["pose"][t - a_s][:3].astype(np.float32)
        return it

    def __getitem__(self, i):
        ci, t, a_t, a_s = self.windows[i]
        return self.item(ci, t, a_t, a_s)


def train_windows(store: ClipStore, J: dict, clips, *, hist: int = 4, stride: int = 1, start=None) -> list:
    """All 6 s-complete windows ``t in [start, T-31]`` of ``clips`` (age fields filled by the sampler); ``start`` absent = ``hist - 1``.

    PREREG_V8F_HIST Amendment 1: ``start`` is ``--window-start``. At ``hist`` 10 the default start (9) drops t = 3..8 of every clip; ``start=3`` keeps ``hist`` 4's list."""
    ci_of = {c: i for i, c in enumerate(J["clip"].tolist())}
    first = effective_window_start(hist, start)
    out = []
    for c in clips:
        ci = ci_of[c]
        T = int(J["n"][ci])
        for t in range(first, T - HORIZON, stride):
            out.append((ci, t, 0, 0))
    return out


def eval_grid_windows(grid_json, J: dict) -> list:
    """The trunk-probe grid (141 episodes, 2,399 windows): ``{clip_id: [t...]}`` -> ``[(ci, t, 0, 0)]``."""
    g = json.load(open(grid_json, encoding="utf-8"))
    ci_of = {c: i for i, c in enumerate(J["clip"].tolist())}
    return [(ci_of[c], int(t), 0, 0) for c in g["order"] for t in g["windows"][c]]


def collate(items: list) -> dict:
    """numpy items -> a dict of torch tensors (labels stacked per tick)."""
    import torch
    out = {}
    stack = lambda key, dt=None: torch.as_tensor(np.stack([np.asarray(it[key]) for it in items]), dtype=dt)
    if "feats" in items[0]:             # absent in --encoder lora batches: the frames are encoded on the fly (collate_frames)
        out["feats"] = stack("feats", torch.float16)
    for k in ("frame_ok", "limit_ok", "gt_ok"):
        out[k] = stack(k, torch.bool)
    for k in ("p10f", "act", "nav", "pose", "limit", "gt", "base", "kin3", "s_h", "lead_gap_h", "pose_now", "pose_t",
              "pose_s"):
        out[k] = stack(k, torch.float32)
    out["p10_t"] = torch.as_tensor(np.stack([np.nan_to_num(it["p10_t"]) for it in items]), dtype=torch.float32)
    for k in ("a_t", "a_s", "t", "ci"):
        out[k] = torch.as_tensor([it[k] for it in items], dtype=torch.long)
    for nm in ("lab", "lab_t", "lab_s"):
        L = [it[nm] for it in items]
        out[nm] = {"has": torch.as_tensor([x["has"] for x in L]), "nav": torch.as_tensor(np.stack([x["nav"] for x in L])),
                   "goal_y": torch.as_tensor(np.stack([x["goal_y"] for x in L])),
                   "goal_w": torch.as_tensor(np.stack([x["goal_w"] for x in L])),
                   "lat_cls": torch.as_tensor([x["lat_cls"] for x in L]), "lat_al": torch.as_tensor(np.stack([x["lat_al"] for x in L])),
                   "lon_cls": torch.as_tensor([x["lon_cls"] for x in L]), "lon_al": torch.as_tensor(np.stack([x["lon_al"] for x in L])),
                   "cons": torch.as_tensor(np.stack([x["cons"] for x in L])), "rc": torch.as_tensor(np.stack([x["rc"] for x in L])),
                   "rc_ok": torch.as_tensor(np.stack([x["rc_ok"] for x in L])),
                   "lead_valid": torch.as_tensor([x["lead_valid"] for x in L])}
    return out


# ------------------------------------------------------------------------------------------------ frames behind the cache (PREREG_V8F_LORA)
FRAME_HW = (256, 640)             # the cache builder's cylindrical frame (a 16 x 40 grid of 16-px patches); tests shrink it, a launch never does


class FrameStore:
    """Lazy per-clip reader of the v2 episode (``<root>/<clip>.v2ep.pt``: PNG frames of the 256x640 cylindrical view) with an LRU of the compressed
    payloads (~37 MB per clip). ``frames(clip, steps)`` decodes cache steps ``g`` (raw frame ``2 g``) with the cache builder's own decode."""

    def __init__(self, root, lru: int = 50):
        from collections import OrderedDict
        self.root, self.lru, self._c = Path(root), int(lru), OrderedDict()

    def path(self, clip: str) -> Path:
        return self.root / f"{clip}.v2ep.pt"

    def missing(self, clips) -> list:
        return [c for c in clips if not self.path(c).is_file()]

    def get(self, clip: str) -> dict:
        if clip in self._c:
            self._c.move_to_end(clip)
            return self._c[clip]
        import torch
        o = torch.load(self.path(clip), map_location="cpu", weights_only=False)        # our own file; the builder loads it the same way
        if o.get("clip_id") != clip:
            raise ValueError(f"{self.path(clip)}: clip_id {o.get('clip_id')!r} != file stem (the label join key would be wrong)")
        if str(o["codec"]) != "png" or (int(o["image_h"]), int(o["image_w"])) != tuple(FRAME_HW):
            raise ValueError(f"{self.path(clip)}: codec {o['codec']} / frame {(o['image_h'], o['image_w'])} is not the cache builder's png {FRAME_HW}")
        self._c[clip] = {k: o[k] for k in ("jpeg_buf", "jpeg_len", "codec")}
        while len(self._c) > self.lru:
            self._c.popitem(last=False)
        return self._c[clip]

    def n_cache_frames(self, clip: str) -> int:
        return (int(self.get(clip)["jpeg_len"].shape[0]) + 1) // 2

    def frames(self, clip: str, steps) -> "torch.Tensor":
        from tanitad.models.v8f_lora_encoder import decode_cache_frames
        return decode_cache_frames(self.get(clip), [int(g) for g in steps])


class V8FMiniFrameWindows(V8FMiniWindows):
    """``--encoder lora``: the same windows, but the batch carries the FRAMES behind ``feats`` instead of the cached tokens. ``item`` returns everything
    the base class does, with ``feats`` replaced by ``fclip`` / ``fidx`` (the ``hmax`` cache steps ``t - hmax + 1 .. t`` clamped into the clip, the SAME
    indices ``feats`` was gathered with). ``keep_cached`` keeps the cached ``feats`` too (the launch gate's identity probe)."""

    def __init__(self, *a, keep_cached: bool = False, **kw):
        super().__init__(*a, **kw)
        self.keep_cached = bool(keep_cached)

    def item(self, ci: int, t: int, a_t: int = 0, a_s: int = 0) -> dict:
        it = super().item(ci, t, a_t, a_s)
        idx = np.clip(np.arange(t - self.hmax + 1, t + 1), 0, it["T"] - 1)
        if not self.keep_cached:
            del it["feats"]
        it["fclip"], it["fidx"] = str(self.J["clip"][ci]), idx.astype(np.int64)
        return it


def unique_frames(clips, idx: np.ndarray):
    """``clips[b]`` and ``idx [B, F]`` (cache steps, already clamped into the clip) -> ``(keys, frame_map)``: ``keys`` the unique ``(clip, step)`` pairs in
    first-seen order, ``frame_map [B, F]`` the position of each frame in ``keys``. Frames shared by windows of one clip, and a clip's clamped frame 0, are
    encoded ONCE; autograd sums their gradients through the gather."""
    seen: dict = {}
    fm = np.empty(idx.shape, np.int64)
    for b, c in enumerate(clips):
        for f in range(idx.shape[1]):
            k = (c, int(idx[b, f]))
            j = seen.get(k)
            if j is None:
                j = seen[k] = len(seen)
            fm[b, f] = j
    return list(seen), fm


def collate_frames(items: list, fstore: FrameStore) -> dict:
    """``collate`` + the decoded frames: ``frames_u8`` uint8 ``[U, 3, 256, 640]`` (the unique frames of the batch) and ``frame_map`` long ``[B, hmax]``
    (``feats[b, f]`` = the encoding of ``frames_u8[frame_map[b, f]]``). If the items kept the cached tokens they are returned as ``feats`` too."""
    import torch
    keys, fm = unique_frames([it["fclip"] for it in items], np.stack([it["fidx"] for it in items]))
    by_clip: dict = {}
    for j, (c, g) in enumerate(keys):
        by_clip.setdefault(c, []).append((j, g))
    frames = torch.empty((len(keys), 3, *FRAME_HW), dtype=torch.uint8)
    T_of = {it["fclip"]: it["T"] for it in items}
    for c, lst in by_clip.items():
        if fstore.n_cache_frames(c) != T_of[c]:
            raise ValueError(f"{c}: the v2ep has {fstore.n_cache_frames(c)} cache steps, the cache clip has {T_of[c]}")
        fr = fstore.frames(c, [g for _, g in lst])
        frames[torch.as_tensor([j for j, _ in lst])] = fr
    b = collate(items)
    b["frames_u8"], b["frame_map"] = frames, torch.as_tensor(fm)
    return b


class ChunkedSampler:
    """Epoch order with I/O locality: shuffle clips, take ``chunk`` clips at a time, shuffle their windows. Also draws
    the sample-and-hold ages (a_t uniform in 0..2 steps = 0..0.4 s; S's tick uniform in 0..4 steps older than T's =
    0..0.8 s) -- seeded by (seed, epoch), so resumable."""

    def __init__(self, windows, batch: int, seed: int = 0, chunk: int = 48, max_age_t: int = 2, max_age_s: int = 6):
        self.windows, self.batch, self.seed, self.chunk = list(windows), int(batch), int(seed), int(chunk)
        self.max_age_t, self.max_age_s = int(max_age_t), int(max_age_s)       # a_s = a_t + (0..4): total age of S's tick

    def epoch(self, ep: int):
        rng = np.random.default_rng((self.seed, ep))
        by_clip = {}
        for w in self.windows:
            by_clip.setdefault(w[0], []).append(w)
        order = rng.permutation(sorted(by_clip))
        for c0 in range(0, len(order), self.chunk):
            pool = [w for c in order[c0:c0 + self.chunk] for w in by_clip[int(c)]]
            pool = [pool[i] for i in rng.permutation(len(pool))]
            for b0 in range(0, len(pool), self.batch):
                batch = []
                for ci, t, _, _ in pool[b0:b0 + self.batch]:
                    a_t = int(rng.integers(0, self.max_age_t + 1))
                    batch.append((ci, t, a_t, a_t + int(rng.integers(0, self.max_age_s - self.max_age_t + 1))))
                yield batch

    def n_batches(self):
        return math.ceil(len(self.windows) / self.batch)


def eval_context(grid: dict, kin_npz, J: dict, units: str | None = None, kin_mode: str = "steer") -> dict:
    """Everything the evaluator needs per window of the trunk-probe grid, WITHOUT any image feature (so the floors,
    the GT control and the nav-only baseline run before the cache exists). ``J`` is the eval join (cache order must be
    ``grid['order']``). Arrays are aligned window-for-window with ``[(clip, t) for clip in order for t in windows]``."""
    fk = _fk()
    check_kin_mode(kin_mode)
    units = KIN_UNITS[kin_mode] if units is None else units
    if units != KIN_UNITS[kin_mode]:
        raise ValueError(f"kin_mode {kin_mode!r} integrates in {KIN_UNITS[kin_mode]!r}, not {units!r}")
    z = np.load(kin_npz)
    off = np.concatenate([[0], np.cumsum(z["n"])])
    assert z["clip"].tolist() == list(grid["order"]) == J["clip"].tolist(), "eval clip order mismatch"
    C = {k: [] for k in ("eid", "t", "kin", "gt", "s_h", "lead_gap_h", "jrow")}
    for i, c in enumerate(grid["order"]):
        P = z["poses"][off[i]:off[i + 1]].astype(np.float64)
        A = None if kin_mode == "yaw" else z["actions"][off[i]:off[i + 1]].astype(np.float64)    # yaw mode never opens actions
        ts = np.asarray(grid["windows"][c], np.int64)
        kin, gt, pg = window_kin(P, A, ts, kin_mode)
        sh = human_arc(pg)
        T = len(pg)
        o = int(J["off"][i])
        idx = np.minimum(ts[:, None] + 1 + np.arange(HORIZON)[None], T - 1)
        C["eid"].append(np.full(len(ts), i)); C["t"].append(ts); C["kin"].append(kin); C["gt"].append(gt)
        C["s_h"].append(sh[ts]); C["lead_gap_h"].append(J["lead_gap"][o + idx]); C["jrow"].append(o + ts)
    X = {k: np.concatenate(v) for k, v in C.items()}
    X["kf8"] = fk.kin_features(X["kin"])
    ha0 = fk.ha0_paths(X["kin"][:, 0], DT, HORIZON)
    X["base"] = fk.kd_x(ha0, fk.ha0_ext_paths(X["kin"], DT, HORIZON, action_units=units)).astype(np.float32)
    r = X["jrow"]
    ci = X["eid"]
    X["has"] = J["has"][r]
    X["nav_tok"] = J["nav"][r][:, :3].argmax(1)
    X["nav_d"] = J["nav_d"][r]
    X["rc"], X["rc_ok"] = J["rc"][r], J["rc_ok"][r] & J["has"][r][:, None]
    X["limit_ms"] = np.where(J["limit_ok"][ci], J["limit"][ci] @ (np.array(LIMITS_KMH) / 3.6), np.nan)
    X["limit_kmh"] = J["limit_kmh"][ci]
    X["lat_cls"], X["lon_cls"] = J["lat_cls"][r], J["lon_cls"][r]
    X["lat_allowed"], X["lon_allowed"] = J["lat_allowed"][r], J["lon_allowed"][r]
    X["goal_y"], X["goal_w"] = J["goal_y"][r], J["goal_w"][r]
    X["cons"] = J["cons"][r]
    X["lead_valid"] = J["lead_valid"][r]
    return X


def v9_label_dict(X: dict) -> dict:
    """The names ``v8f_consistency.constraints_from_v9`` reads, from an :func:`eval_context`."""
    g = lambda n: X["cons"][:, CONS_IDX[n]]
    return {"goal_y": X["goal_y"], "goal_w": X["goal_w"], "v_hi_ms": g("v_hi_ms"), "lon_d_reach_m": g("lon_d_reach_m"),
            "lon_t_reach_s": g("lon_t_reach_s"), "turn_t_start_s": g("turn_t_start_s"), "turn_t_end_s": g("turn_t_end_s"),
            "lead_valid": X["lead_valid"], "lead_tg_min_s": g("lead_tg_min_s")}


# ------------------------------------------------------------------------------------------------ synthetic world
def make_synthetic_world(root, n_train=6, n_eval=3, T=70, seed=0, d_feat=D_FEAT) -> dict:
    """A tiny world in the exact S4/v9 formats, for tests and the 30-step smoke. Real driving-like kinematics; the
    features are random but ARE a function of (speed, curvature) so the model has something to learn."""
    import torch
    rng = np.random.default_rng(seed)
    root = Path(root)
    fk = _fk()
    names = {"train": [f"syn-train-{i:03d}" for i in range(n_train)], "eval": [f"syn-eval-{i:03d}" for i in range(n_eval)]}
    joins, windows, kin_eval, native = {}, {"order": names["eval"], "windows": {}}, [], {}
    GY = {g: i for i, g in enumerate(GOAL22)}
    for split, clips in names.items():
        (root / split / "clips").mkdir(parents=True, exist_ok=True)
        man = open(root / split / "manifest.jsonl", "w", encoding="utf-8")
        entries = []
        for cid in clips:
            v = np.clip(8 + np.cumsum(rng.normal(0, 0.25, T)), 0.0, 25.0)
            kap = np.cumsum(rng.normal(0, 0.004, T)) * 0.2
            yaw = np.cumsum(v * kap * DT)
            x = np.cumsum(v * np.cos(yaw) * DT)
            y = np.cumsum(v * np.sin(yaw) * DT)
            pose = np.stack([x, y, yaw, v], 1).astype(np.float32)
            steer = np.arctan(2.9 * kap).astype(np.float32)
            a = np.concatenate([(v[2:] - v[:-2]) / DT, [0.0, 0.0]]).astype(np.float32)
            kin3 = np.full((T, 3), np.nan)
            kin3[1:, 0] = v[1:]
            kin3[1:, 1] = (v[1:] - v[:-1]) / DT
            kin3[1:, 2] = steer[1:]
            p10 = np.full((T, 8), np.nan, np.float32)
            p10[1:] = fk.kin_features(kin3[1:]).astype(np.float32)
            f = rng.normal(0, 1, (T, N_TOK, d_feat)).astype(np.float32)
            f[:, :, 0] += v[:, None] / 10.0
            f[:, :, 1] += (kap * 30.0)[:, None]
            torch.save({"meta": {"clip_id": cid, "n_frames": T}, "pool4x10": torch.as_tensor(f).to(torch.float8_e4m3fn),
                        "v": torch.as_tensor(v, dtype=torch.float32), "a": torch.as_tensor(a),
                        "steer": torch.as_tensor(steer), "yaw_rate": torch.as_tensor(v * kap, dtype=torch.float32),
                        "pose_grid": torch.as_tensor(pose), "kin3": torch.as_tensor(kin3, dtype=torch.float32),
                        "p10": torch.as_tensor(p10)}, root / split / "clips" / f"{cid}.pt")
            man.write(json.dumps({"clip": cid, "n_frames": T}) + "\n")
            entries.append((cid, T, v, kap, pose))
            fr = np.zeros((2 * T - 1, 4), np.float32)             # the NATIVE 10 Hz pose track: odd frames = midpoints
            fr[0::2] = pose
            fr[1::2] = 0.5 * (pose[:-1] + pose[1:])
            native[cid] = torch.as_tensor(fr)
            if split == "eval":                                   # frame-level (10 Hz) poses/actions
                ac = np.zeros((2 * T - 1, 2), np.float32)
                ac[0::2, 0] = steer
                ac[1::2, 0] = 0.5 * (steer[:-1] + steer[1:])
                kin_eval.append((cid, fr, ac))
            if split == "eval":
                windows["windows"][cid] = list(range(3, T - HORIZON, 4))
        man.close()
        # v9-like join built directly (no release file): turn / stop / speed labels derived from the synthetic drive
        n = np.array([e[1] for e in entries])
        N = int(n.sum())
        J = build_label_join([(e[0], e[1]) for e in entries], _EmptyRelease())
        o = 0
        for ci, (cid, Tn, v, kap, pose) in enumerate(entries):
            sl = slice(o, o + Tn)
            J["has"][sl] = True
            fwd = np.arange(Tn)
            side = np.sign(np.convolve(kap, np.ones(15) / 15, "same"))
            tok = np.where(np.abs(np.convolve(kap, np.ones(15) / 15, "same")) > 0.002, np.where(side > 0, 1, 2), 0)
            nav = np.zeros((Tn, 5), np.float32)
            nav[np.arange(Tn), tok] = 1.0
            nav[:, 3] = rng.uniform(0, 1, Tn) * (tok > 0)
            nav[:, 4] = tok > 0
            J["nav"][sl], J["nav_d"][sl] = nav, np.where(tok > 0, nav[:, 3] * 100.0, np.nan)
            rc = np.zeros((Tn, 3, 3), np.float32)
            for t in range(Tn):
                fut = pose[min(t + 1, Tn - 1):, :2] - pose[t, :2]
                c_, s_ = math.cos(-pose[t, 2]), math.sin(-pose[t, 2])
                ego = np.stack([fut[:, 0] * c_ - fut[:, 1] * s_, fut[:, 0] * s_ + fut[:, 1] * c_], 1) if len(fut) else np.zeros((1, 2))
                arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ego, axis=0), axis=1))]) if len(ego) > 1 else np.zeros(1)
                for m, A in enumerate(CKPT_ARCS):
                    j = int(np.searchsorted(arc, A)) if arc[-1] >= A else len(arc) - 1
                    rc[t, m, :2] = ego[min(j, len(ego) - 1)]
                    rc[t, m, 2] = float(pose[min(t + 1 + j, Tn - 1), 2] - pose[t, 2])
                    J["rc_ok"][o + t, m] = arc[-1] >= A
            J["rc"][sl] = rc * J["rc_ok"][sl][:, :, None]
            gy = np.zeros(Tn, np.uint32)
            gy |= ((tok == 1).astype(np.uint32) << GY["TURN_L"]) | ((tok == 2).astype(np.uint32) << GY["TURN_R"])
            gy |= ((tok == 0).astype(np.uint32) << GY["FOLLOW_LANE"])
            J["goal_y"][sl] = gy
            J["goal_w"][sl] = (1 << GY["TURN_L"]) | (1 << GY["TURN_R"]) | (1 << GY["FOLLOW_LANE"]) | (1 << GY["STOP_POINT"])
            J["lat_cls"][sl] = tok
            J["lat_allowed"][sl] = (1 << tok).astype(np.uint8)
            vb = np.array([v[min(t + 10, Tn - 1)] for t in range(Tn)])
            J["lon_cls"][sl] = np.where(vb < 1, 0, 6)
            J["lon_allowed"][sl] = (1 << J["lon_cls"][sl]).astype(np.uint8)
            J["cons"][sl, CONS_IDX["v_hi_ms"]] = np.array([v[t + 10:t + 40].max() if t + 10 < Tn else np.nan for t in range(Tn)])
            J["cons"][sl, CONS_IDX["turn_t_start_s"]] = np.where(tok > 0, 1.0, np.nan)
            J["cons"][sl, CONS_IDX["turn_t_end_s"]] = np.where(tok > 0, 5.0, np.nan)
            J["limit"][ci], J["limit_kmh"][ci], J["limit_ok"][ci] = np.array([0, 1, 0, 0], np.float32), 50, True
            o += Tn
        save_join(J, root / f"join_{split}.npz")
        joins[split] = J
        with open(root / f"speed_{split}.jsonl", "w", encoding="utf-8") as fh:
            for cid, *_ in entries:
                fh.write(json.dumps({"sid": stable_episode_id(cid), "clip_id": cid, "v_hi_ms": 13.0, "bin": 1,
                                     "limit_kmh": 50, "one_hot": [0.0, 1.0, 0.0, 0.0], "valid": 1}) + "\n")
    json.dump(windows, open(root / "eval_grid_windows.json", "w"))
    torch.save({"tracks": native, "meta": {"schema": "native-tracks-v1", "synthetic": True}}, root / "native_tracks.pt")
    np.savez_compressed(root / "eval_kin.npz", clip=np.array([c for c, _, _ in kin_eval]),
                        n=np.array([len(f) for _, f, _ in kin_eval]), poses=np.concatenate([f for _, f, _ in kin_eval]),
                        actions=np.concatenate([a for _, _, a in kin_eval]))
    return {"root": str(root), "joins": joins, "names": names}


class _EmptyRelease:
    """A release with no clips (``build_label_join`` then returns an all-empty join to be filled by hand)."""
    rows, clips = {}, {}
