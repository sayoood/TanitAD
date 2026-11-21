#!/usr/bin/env python3
"""T1-HEADS read: PREREG_V8F_T1_HEADS.md (landed ae659d49) applied as written.

    STRATEGIC family (S-POS vs CTRV) and the T head's own decisions (T-LAT-F1 / T-LON-F1) at T1-kin, from the per-tick head dumps
    of `v8f_t1_eval.py run --dump-ticks --dump-heads` (81aa3e36).

Nothing here re-implements the codebase: the join / cache / tracks come from `tanitad.data.v8f_mini_data`, the frames from
`tanitad.models.v8f_mini.transform_ckpt`, the executed track is REBUILT with the tool's own `envelope` / `plan_to_world` /
`hermite_frames`, the curvature uses the programme's `_pose_kappa0` on `ego_channels_native`, the position / heading errors are
`taniteval.v8f_consistency.checkpoint_errors`, the F1 is `macro_f1` through the T0 eval's `_f1_reducer`, and every interval is
`taniteval.ci.paired_episode_cluster_bootstrap` (B 2000, seed 0, over the 141 episodes).

Modes
  real   the four runs of PREREG section 3 under --run-dir (refuses unless chain.out says ZZT1HD-DONE 4)
  smoke  ONE ticks file (the 24-window development run); every number is labelled SMOKE and is not outcome-bearing
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True                          # the tool tree is READ-ONLY (a live chain reads it)

import numpy as np  # noqa: E402

OFF_ROUTE_M = 3.0                                       # PREREG section 1
ARCS = (30, 50, 80)
SEED_FLOOR = {"lat": 0.0124, "lon": 0.0073}             # PREREG section 2 (MEASURED on the banked T0 fresh evals)
NCLS = {"lat": 5, "lon": 7}
TICKS = 3                                               # m = 10 => ticks 0, 1, 2
M_STEPS = 10
TOL_K2_M = 1e-4
TOL_K3_M = 1e-6
TOL_K4 = 1e-3
BANKED_QUOTED = {                                       # PREREG section 4, K4 (the values as quoted there)
    "s0": {"pos": [0.977, 2.296, 5.887], "lat": 0.8352, "lon": 0.5062},
    "s1": {"pos": [0.929, 2.204, 5.896], "lat": 0.8476, "lon": 0.5135}}
MAJORITY_QUOTED = {"lat": 0.3170, "lon": 0.0707}

NEXT_S = {"S ADDS ROUTE": "the S level uses the route while the model drives itself: support for the hierarchy claim at T1",
          "NO WORSE": "diagnose what S lacks at T1: the dead-reckoned nav's drift against the logged nav, then S's staleness at T1",
          "WORSE": "the same diagnosis; S is reported as worse than kinematics at T1"}
NEXT_T = {"HOLDS": "support: T's decision quality survives its own motion",
          "DEGRADES": "T is not robust to its own state. T supervision is folded into O-RECOVER (`PREREG_V8F_HACC_MC` §7; box 2, needs PI sign-off)",
          "INCONCLUSIVE": "n and the CI width are reported; next is m = 5 (more ticks per window), under its own read"}


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jenc(x):
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


# ---------------------------------------------------------------------------------------------------------------- arguments
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("real", "smoke"), required=True)
    ap.add_argument("--tool-root", required=True, help="the pinned export of 81aa3e36 (stack + taniteval); READ-ONLY")
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--tracks", required=True, help="native_tracks.pt")
    ap.add_argument("--eval-grid", required=True)
    ap.add_argument("--eval-join", required=True)
    ap.add_argument("--banked-dir", required=True, help="dir with eval_M2b_s{0,1}-sfresh-fresh.{json,npz}, dumps_HACC_s{0,1}_m10_causal.npz, dumps_manifest.json")
    ap.add_argument("--out", required=True, help="output JSON")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--run-dir", help="real: the t1hd_run dir (t1v8f_<tag>.json, ticks/ticks_<tag>.npz, dump_<tag>/, logs/chain.out)")
    ap.add_argument("--smoke-ticks", help="smoke: ticks_<tag>.npz")
    ap.add_argument("--smoke-dump-dir", help="smoke: the same run's dump_<tag> dir")
    ap.add_argument("--smoke-ref-dump-dir", help="smoke K7: the flags-OFF run's dump dir (the heads run must equal it bit for bit)")
    ap.add_argument("--smoke-seed", default="s0")
    ap.add_argument("--dev-fake-run", action="store_true", help="DEV ONLY: run the real-mode branch with --smoke-ticks / --smoke-dump-dir standing in for all four runs (REFUSED by construction)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------------------------------------------- tool imports
def setup_imports(root):
    for p in (os.path.join(root, "stack", "scripts"), os.path.join(root, "taniteval"), os.path.join(root, "stack")):
        sys.path.insert(0, p)
    global D, M, C, PB, EB, E0, T1, _pose_kappa0, torch
    import torch  # noqa: F401
    from tanitad.data import v8f_mini_data as D
    from tanitad.models import v8f_mini as M
    from tanitad.models.kinematic_prior import _pose_kappa0
    from taniteval import v8f_consistency as C
    from taniteval.ci import paired_episode_cluster_bootstrap as PB, episode_cluster_bootstrap as EB
    import v8f_mini_eval as E0
    import v8f_t1_eval as T1
    return root


# ---------------------------------------------------------------------------------------------------------------- inputs
class Ctx:
    """The join, the logged poses (cache pose_grid) and the native logged tracks of the clips the windows use."""

    def __init__(self, a):
        self.J = D.load_join(a.eval_join)
        self.wins_grid = D.eval_grid_windows(a.eval_grid, self.J)
        self.tracks = D.load_native_tracks(a.tracks)
        self.store = D.ClipStore(a.eval_cache, "eval", lru=2, grid="pool4x10", kin_mode="yaw", tracks=self.tracks)
        self.PG, self.PG32, self.TR = {}, {}, {}

    def clip(self, ci):
        if ci not in self.PG:
            name = str(self.J["clip"][ci])
            pg = np.asarray(self.store.get(name)["pose"], np.float64)       # float64 (x, y, yaw, v) per cache step: the cache pose_grid
            self.PG[ci] = pg
            self.PG32[ci] = pg[:, :2].astype(np.float32).astype(np.float64)  # the value the model was fed (batch['pose_s'] is float32)
            self.TR[ci] = self.tracks[name]
        return self.PG[ci]


def load_ticks(path):
    z = np.load(path, allow_pickle=False)
    tt = {k: z[k] for k in z.files}
    N = int(tt["n_windows"])
    win, tick = tt["win"], tt["tick"]
    row = np.full((N, TICKS), -1, np.int64)
    for r in range(len(win)):
        assert row[win[r], tick[r]] < 0, "duplicate (window, tick) row"
        row[win[r], tick[r]] = r
    assert (row >= 0).all(), "a (window, tick) row is missing"
    assert len(win) == N * TICKS
    at = lambda k: tt[k][row]                                       # [N, 3, ...]
    t0, ci = at("t0"), at("ci")
    assert (t0 == t0[:, :1]).all() and (ci == ci[:, :1]).all()
    assert (at("j") == t0 + M_STEPS * np.arange(TICKS)[None]).all(), "ticks are not m=10 apart"
    assert (at("tick") == np.arange(TICKS)[None]).all()
    assert (at("age_t")[:, 0] == 0).all() and (at("age_s")[:, 0] == 0).all(), "tick 0 must be fresh (ages 0, 0)"
    assert not tt["fired"].any()
    out = {"N": N, "ci": ci[:, 0].copy(), "t0": t0[:, 0].copy(), "row": row, "meta": {k: tt[k].item() for k in ("split", "tag", "ckpt", "replan_steps")}}
    for k in ("tau", "P", "plan", "gt", "gt_ok", "v0", "pose", "j", "age_t", "age_s", "s_ckpt", "s_ckpt_now", "s_ckpt_w0", "s_pose", "t_pose",
              "pose_w0", "t_goal_prob", "t_lat_logits", "t_lon_logits", "t_cons_pred"):
        out[k] = at(k)
    out["clip"] = np.asarray(tt["clip"])[row[:, 0]]
    return out


# ---------------------------------------------------------------------------------------------------------------- matching
def match_all(ctx, ci, t0, pose, f32):
    """r' = the logged row >= t0 of the same clip whose logged pose is nearest in world xy to ``pose`` [N, 3, 2+]; (row, distance).
    ``f32``: candidates are the float32 value the model was fed (the dump's own precision); else float64."""
    N = len(ci)
    r = np.zeros((N, TICKS), np.int64)
    d = np.zeros((N, TICKS))
    for w in range(N):
        ctx.clip(int(ci[w]))
        P = (ctx.PG32 if f32 else ctx.PG)[int(ci[w])][int(t0[w]):, :2]
        for k in range(TICKS):
            dd = np.hypot(P[:, 0] - float(pose[w, k, 0]), P[:, 1] - float(pose[w, k, 1]))
            i = int(np.argmin(dd))                                 # first minimum: ties go to the earlier row
            r[w, k], d[w, k] = int(t0[w]) + i, dd[i]
    return r, d


def jrow(ctx, ci, r):
    return ctx.J["off"][ci][:, None] + r


# ---------------------------------------------------------------------------------------------------------------- frames
def to_w0(ck, src, dst):
    """Re-express ck [..., M, 3] given in the ``src`` frame in the ``dst`` frame, in float64, by the model's own ``transform_ckpt``."""
    sh = ck.shape[:-2]
    c = torch.as_tensor(np.asarray(ck, np.float64).reshape(-1, ck.shape[-2], 3))
    s = torch.as_tensor(np.asarray(src, np.float64).reshape(-1, 3))
    d = torch.as_tensor(np.asarray(dst, np.float64).reshape(-1, 3))
    return M.transform_ckpt(c, s, d).numpy().reshape(*sh, ck.shape[-2], 3)


def to_w0_f32(ck, src, dst):
    """The tool's own arithmetic (`TickRecorder._ckpt_frames`): float32 tensors, the window-origin pose cast to float32."""
    return M.transform_ckpt(torch.as_tensor(ck), torch.as_tensor(src), torch.as_tensor(np.asarray(dst).astype(np.float32))).numpy()


def gt_frame_check(ctx, ci, t0, rS, gt_w0, gt_ok, pose_w0, src_override=None):
    """EXTRA (not in the PREREG): the GT checkpoints moved from the logged frame at r' into the window-origin frame must lie on the logged route
    (v9's RC-A is a sigma-8 m smoothed path of the human's own future): the distance from each scored GT (x, y) to the nearest logged pose at rows >= r',
    expressed in the window-origin frame. A wrong frame (e.g. the label read as if it were in the window-origin frame already) puts it metres off."""
    out = {f"tick{k}": [] for k in range(TICKS)}
    for w in range(len(ci)):
        pg = ctx.clip(int(ci[w]))
        c, s_ = np.cos(-pose_w0[w, 0, 2]), np.sin(-pose_w0[w, 0, 2])
        d = pg[:, :2] - pose_w0[w, 0, None, :2]
        route = np.stack([d[:, 0] * c - d[:, 1] * s_, d[:, 0] * s_ + d[:, 1] * c], 1)
        for k in range(TICKS):
            for m in range(3):
                if gt_ok[w, k, m]:
                    P = route[int(rS[w, k]):]
                    out[f"tick{k}"].append(float(np.hypot(P[:, 0] - gt_w0[w, k, m, 0], P[:, 1] - gt_w0[w, k, m, 1]).min()))
    return {k: ({"n": len(v), "median_m": float(np.median(v)), "frac_within_3m": float(np.mean(np.asarray(v) <= 3.0)),
                 "note": "tail = checkpoints beyond the cache clip's logged route (a 80 m arc near the clip end), also present at tick 0"} if v else {"n": 0})
            for k, v in out.items()}


# ---------------------------------------------------------------------------------------------------------------- the executed track
def rebuild_track(ctx, ci, t0, plans):
    """The executed 10 Hz track of one window, REBUILT with the tool's own functions from the dumped plans: ``commit_plan`` for ticks 0..2.
    Rows <= 2*t0 are the logged native track; rows after it are the executed Hermite frames. Returns (track, d_pose)."""
    pg = ctx.clip(ci)
    tr = np.full(ctx.TR[ci].shape, np.nan)
    tr[:2 * t0 + 1] = np.asarray(ctx.TR[ci], np.float64)[:2 * t0 + 1]
    poses = np.array(pg, copy=True)
    poses[t0 + 1:] = np.nan
    j = t0
    for k in range(TICKS):
        pose = poses[j]
        plan_e, _ = T1.envelope(plans[k])
        W = T1.plan_to_world(plan_e, pose)
        knots = np.vstack([pose[None, :2], W])
        xy, psi, v = T1.hermite_frames(tr[2 * (j - 1), :2], knots, M_STEPS, float(pose[2]))
        sl = slice(2 * j + 1, 2 * (j + M_STEPS) + 1)
        tr[sl, 0:2], tr[sl, 2], tr[sl, 3] = xy, psi, v
        for s in range(1, M_STEPS + 1):
            poses[j + s] = tr[2 * (j + s)]
        j += M_STEPS
    return tr, poses


def mean_curvature(track, r):
    """The executed track's mean curvature over the 1 s before cache row ``r``: the 10 raw frames 2r-9 .. 2r of
    ``kappa_frame = _pose_kappa0(omega, v)`` (the programme's measured-curvature feature: clamp(omega / max(v, 2 m/s), +-0.3), one-frame
    backward differences from `ego_channels_native`). Frames before the clip start do not exist and are not counted. Returns (kappa, n_frames)."""
    lo = max(0, 2 * r - 10)
    sl = track[lo:2 * r + 1]
    if len(sl) < 2:
        return 0.0, 0, 0.0
    ch = D.ego_channels_native(sl)[1:]                              # row 0 of a slice is the 0-rate convention, never a measurement
    k = _pose_kappa0(torch.from_numpy(ch[:, 2].copy()), torch.from_numpy(ch[:, 0].copy())).numpy()
    ok = np.isfinite(k) & np.isfinite(ch[:, 2]) & np.isfinite(ch[:, 0])
    if not ok.any():
        return 0.0, 0, 0.0
    # EXTRA sensitivity (never decisive): the unclamped ratio of the means, mean(omega) / max(mean(v), 0.5 m/s)
    k2 = float(ch[ok, 2].mean() / max(float(ch[ok, 0].mean()), 0.5))
    return float(k[ok].mean()), int(ok.sum()), k2


def ctrv_straight(kappa, s_pose, pose_w0):
    """CTRV (constant curvature = constant speed and yaw rate) and STRAIGHT checkpoints at arc 30 / 50 / 80 m from the pose S was computed from,
    expressed in the window-origin frame. kappa [N, 3]; s_pose [N, 3, 3]; pose_w0 [N, 3, 3]-broadcast."""
    N = kappa.shape[0]
    s = np.asarray(ARCS, np.float64)
    k = kappa[..., None]                                            # [N, 3, 1]
    small = np.abs(k) < 1e-9
    ks = np.where(small, 1.0, k)
    x = np.where(small, s, np.sin(ks * s) / ks)
    y = np.where(small, 0.0, (1.0 - np.cos(ks * s)) / ks)
    ps = np.where(small, 0.0, ks * s)
    ctrv_s = np.stack([x, y, ps], -1)                               # in S's tick frame
    str_s = np.stack([np.broadcast_to(s, x.shape), np.zeros_like(x), np.zeros_like(x)], -1)
    return to_w0(ctrv_s, s_pose, pose_w0), to_w0(str_s, s_pose, pose_w0), ctrv_s


# ---------------------------------------------------------------------------------------------------------------- estimator wrappers
PAIRS = []                                                          # every (name, a, b, eid, reduce) the read used: K1 re-runs each as (a, a)


def pbrow(name, a, b, eid, n_boot, reduce="mean", record=True, lower_is_better=True, **extra):
    a, b = np.asarray(a), np.asarray(b)
    r = PB(a, b, eid, n_boot=n_boot, seed=0, reduce=reduce)
    if record:
        PAIRS.append((name, a, b, eid, reduce))
    out = {"row": name, "delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": bool(r["separated"]),
           "n_episodes": r["n_episodes"], "estimator": r["estimator"], "n_boot": r["n_boot"]}
    if r.get("degenerate"):
        out["degenerate"] = True
    out["lower_is_better"] = bool(lower_is_better)                  # an error row: a positive delta is worse; an F1 row: a negative delta is worse
    out["worse"] = bool(r["separated"] and ((r["delta"] > 0) if lower_is_better else (r["delta"] < 0)))
    out["better"] = bool(r["separated"] and ((r["delta"] < 0) if lower_is_better else (r["delta"] > 0)))
    out.update(extra)
    return out


def srow(name, a, b, eid, n_boot, record=True, **extra):
    """An S error row on per-window arrays: both sides NaN on exactly the same windows (ineligible windows stay in the 141-episode pool)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(ok, a, np.nan), np.where(ok, b, np.nan)
    return pbrow(name, a, b, eid, n_boot, "mean", record, n_windows=int(ok.sum()), mean_a=float(np.nanmean(a)) if ok.any() else None,
                 mean_b=float(np.nanmean(b)) if ok.any() else None, **extra)


def _ints(v):
    """taniteval.ci casts every per-window array to float64; the encoded cells are small exact integers, so the cast back is lossless."""
    return np.rint(np.asarray(v, np.float64)).astype(np.int64)


def f1_red(n):
    base = E0._f1_reducer(n)                                        # the T0 eval's own reducer, unchanged; only the dtype is restored

    def red(v):
        return base(_ints(v))
    red.__name__ = f"macro_f1_{n}"
    return red


def acc_red(n):
    def red(v):
        v = _ints(v)
        v = v[v >= 0]
        return float(np.mean(v // (n + 1) == v % (n + 1))) if len(v) else float("nan")
    return red


def slots(enc):
    """enc [N, 3] encoded cells (-1 = unscored) -> (tick-0 side, ticks>=1 side, W) as slot arrays [2N] on the SAME windows W:
    W = windows with a scored tick-0 cell AND >= 1 scored cell at ticks 1, 2. Slot w*2+k: ticks>=1 side holds tick k+1, tick-0 side holds tick 0 in slot 0."""
    c0, ge = enc[:, 0], enc[:, 1:]
    W = (c0 >= 0) & (ge >= 0).any(1)
    a = np.where(W[:, None], ge, -1).reshape(-1)
    b = np.stack([np.where(W, c0, -1), np.full(len(c0), -1, enc.dtype)], 1).reshape(-1)
    return b, a, W


def trow(name, enc_a, enc_b_slots, W, eid2, n_boot, reduce, **extra):
    r = pbrow(name, enc_a, enc_b_slots, eid2, n_boot, reduce, **extra)
    return r


# ---------------------------------------------------------------------------------------------------------------- one run
def analyze_run(ctx, tt, a, label, n_boot, dump_dir=None, want_rows=True):
    """Everything one run (one ticks file) yields. Returns a dict of arrays and rows."""
    N, ci, t0 = tt["N"], tt["ci"], tt["t0"]
    J = ctx.J
    for c in np.unique(ci):
        ctx.clip(int(c))
    assert all(str(J["clip"][ci[w]]) == str(tt["clip"][w]) for w in range(N)), "tick-file clip ids disagree with the join"
    R = {"N": N, "label": label}
    s_xy = tt["s_pose"][..., :2].astype(np.float64)
    t_xy = tt["t_pose"][..., :2].astype(np.float64)
    rS, dS = match_all(ctx, ci, t0, s_xy, True)
    rT, dT = match_all(ctx, ci, t0, t_xy, True)
    rS64, dS64 = match_all(ctx, ci, t0, s_xy, False)
    rT64, dT64 = match_all(ctx, ci, t0, t_xy, False)
    R.update(rS=rS, dS=dS, rT=rT, dT=dT, dS64=dS64, dT64=dT64)
    onS, onT = dS <= OFF_ROUTE_M, dT <= OFF_ROUTE_M
    R["off_route"] = {"S": {f"tick{k}": int((~onS[:, k]).sum()) for k in range(TICKS)}, "T": {f"tick{k}": int((~onT[:, k]).sum()) for k in range(TICKS)},
                      "n_windows": int(N)}
    # ---- the GT at r' in the window-origin frame
    jS, jT = jrow(ctx, ci, rS), jrow(ctx, ci, rT)
    pose_w0 = tt["pose_w0"]                                         # [N, 3, 3] f64 (the same for every tick of a window)
    PGr = lambda r: np.stack([np.stack([ctx.PG[int(ci[w])][int(r[w, k])][:3] for k in range(TICKS)]) for w in range(N)])
    src = PGr(rS)                                                   # the logged pose of the label frame
    rc = J["rc"][jS].astype(np.float64)                             # [N, 3, 3(arc), 3]
    gt_w0 = to_w0(rc.reshape(N, TICKS, 3, 3), src, pose_w0)
    gt_ok = (J["rc_ok"][jS] & J["has"][jS][..., None]) & onS[..., None]       # [N, 3, 3]: scored only if the label exists AND on-route
    R["gt_ok"] = gt_ok
    R["gt_frame"] = gt_frame_check(ctx, ci, t0, rS, gt_w0, gt_ok, pose_w0)
    # ---- the executed track, rebuilt; the curvature of the second before S's frame
    kap = np.zeros((N, TICKS))
    kap2 = np.zeros((N, TICKS))
    nfr = np.zeros((N, TICKS), int)
    x1_cl = np.zeros(N)
    x1_pose = np.zeros(N)
    x1_s = np.zeros(N)
    x1_t = np.zeros(N)
    cl_dump = None
    if dump_dir:
        fs = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
        cl_dump = np.concatenate([np.load(f)["cl"] for f in fs])
        assert len(cl_dump) == N
    for w in range(N):
        tr, poses = rebuild_track(ctx, int(ci[w]), int(t0[w]), tt["plan"][w])
        for k in range(TICKS):
            j = int(tt["j"][w, k])
            r_s = j - int(tt["age_s"][w, k])
            kap[w, k], nfr[w, k], kap2[w, k] = mean_curvature(tr, r_s)
            if k > 0:                                               # the executed pose at the tick vs the dump (x, y, yaw, v)
                x1_pose[w] = max(x1_pose[w], float(np.abs(poses[j][:4] - tt["pose"][w, k][:4]).max()))
            x1_s[w] = max(x1_s[w], float(np.abs(poses[r_s][:3] - tt["s_pose"][w, k].astype(np.float64)).max()))
            x1_t[w] = max(x1_t[w], float(np.abs(poses[j - int(tt["age_t"][w, k])][:3] - tt["t_pose"][w, k].astype(np.float64)).max()))
        if cl_dump is not None:
            x1_cl[w] = float(np.abs(T1.exec_in_ego(tr, int(t0[w])) - cl_dump[w]).max())
    R["kappa"], R["kappa_frames"], R["kappa2"] = kap, nfr, kap2
    R["x1"] = {"max_cl_m": float(x1_cl.max()) if cl_dump is not None else None, "max_pose_tick_f64": float(x1_pose.max()),
               "max_s_pose_vs_rebuilt_m_f32_tol": float(x1_s.max()), "max_t_pose_vs_rebuilt_m_f32_tol": float(x1_t.max())}
    # ---- the S arms in the window-origin frame
    ctrv, strt, _ = ctrv_straight(kap, tt["s_pose"].astype(np.float64), pose_w0)
    ctrv2, _, _ = ctrv_straight(kap2, tt["s_pose"].astype(np.float64), pose_w0)
    S = tt["s_ckpt_w0"].astype(np.float64)
    mir_raw = tt["s_ckpt"] * np.array([1.0, -1.0, -1.0], np.float32)             # K5: y -> -y (and psi -> -psi) in S's own tick frame
    mir = to_w0_f32(mir_raw.reshape(-1, 3, 3), tt["s_pose"].reshape(-1, 3), tt["pose_w0"].reshape(-1, 3)).reshape(N, TICKS, 3, 3).astype(np.float64)
    err = {}
    for nm, arr in (("S", S), ("CTRV", ctrv), ("STRAIGHT", strt), ("MIRROR", mir), ("CTRV2", ctrv2)):
        e = C.checkpoint_errors(arr, gt_w0, gt_ok)                  # pos_err_m, head_err_deg: NaN where not scored
        err[nm] = e
    R["err"] = err
    R["gt_w0"] = gt_w0
    # ---- the T cells at r'_T
    hasT = J["has"][jT]
    cells = {}
    for hd in ("lat", "lon"):
        cls = J[f"{hd}_cls"][jT].astype(np.int64)
        true = np.where(hasT & (cls >= 0) & onT, cls, -1)
        pred = tt[f"t_{hd}_logits"].argmax(-1)
        cells[hd] = {"true": true, "pred": pred}
    gy = ((J["goal_y"][jT].astype(np.int64)[..., None] >> np.arange(22)) & 1).astype(bool)
    gw = ((J["goal_w"][jT].astype(np.int64)[..., None] >> np.arange(22)) & 1).astype(bool) & hasT[..., None] & onT[..., None]
    cells["goal"] = {"gy": gy, "gw": gw, "prob": tt["t_goal_prob"]}
    R["cells"] = cells
    v_gt = np.stack([np.stack([ctx.PG[int(ci[w])][int(rT[w, k]), 3] for k in range(TICKS)]) for w in range(N)])
    R["crawl"] = tt["v0"] < (v_gt - 2.0)                            # the tick's own speed vs the GT speed at r'_T, minus 2 m/s
    R["v0"] = tt["v0"]
    return R


# ---------------------------------------------------------------------------------------------------------------- rows
def per_window(e_arr):
    """[N, 3(tick), 3(arc)] -> (tick-0 [N, 3], mean over the scored ticks >= 1 [N, 3]; NaN where none)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return e_arr[:, 0], np.nanmean(e_arr[:, 1:], axis=1)


def s_rows(R, ci, n_boot, arms=("S",), tag=""):
    """The STRATEGIC rows of one run. Decisive: S-POS vs CTRV (ticks >= 1, mean per window). Beside: the rest of PREREG section 2."""
    out = {"decisive": {}, "beside": {}}
    eid = ci
    P = {nm: per_window(R["err"][nm]["pos_err_m"]) for nm in R["err"]}
    H = {nm: per_window(R["err"][nm]["head_err_deg"]) for nm in R["err"]}
    for m, arc in enumerate(ARCS):
        out["decisive"][f"S-POS@{arc}"] = srow(f"S-POS@{arc} (S - CTRV, ticks>=1)", P["S"][1][:, m], P["CTRV"][1][:, m], eid, n_boot, comparator="CTRV")
        out["beside"][f"S-POS@{arc} vs STRAIGHT"] = srow(f"S-POS@{arc} (S - STRAIGHT, ticks>=1)", P["S"][1][:, m], P["STRAIGHT"][1][:, m], eid, n_boot, comparator="STRAIGHT")
        out["beside"][f"S-HEAD@{arc} vs CTRV"] = srow(f"S-HEAD@{arc} (S - CTRV, ticks>=1)", H["S"][1][:, m], H["CTRV"][1][:, m], eid, n_boot, comparator="CTRV")
        out["beside"][f"S-HEAD@{arc} vs STRAIGHT"] = srow(f"S-HEAD@{arc} (S - STRAIGHT, ticks>=1)", H["S"][1][:, m], H["STRAIGHT"][1][:, m], eid, n_boot, comparator="STRAIGHT")
        out["beside"][f"S-POS@{arc} vs CTRV2 (EXTRA sensitivity: unclamped mean(omega)/max(mean(v), 0.5))"] = srow(
            f"S-POS@{arc} (S - CTRV2, ticks>=1) EXTRA sensitivity, never decisive", P["S"][1][:, m], P["CTRV2"][1][:, m], eid, n_boot, comparator="CTRV2")
        out["beside"][f"tick0 S-POS@{arc} vs CTRV"] = srow(f"tick-0 S-POS@{arc} (S - CTRV)", P["S"][0][:, m], P["CTRV"][0][:, m], eid, n_boot, comparator="CTRV")
        out["beside"][f"tick0 S-POS@{arc} vs STRAIGHT"] = srow(f"tick-0 S-POS@{arc} (S - STRAIGHT)", P["S"][0][:, m], P["STRAIGHT"][0][:, m], eid, n_boot, comparator="STRAIGHT")
        out["beside"][f"degradation S-POS@{arc}"] = srow(f"S-POS@{arc} degradation (ticks>=1) - (tick 0)", P["S"][1][:, m], P["S"][0][:, m], eid, n_boot)
        out["beside"][f"degradation CTRV-POS@{arc}"] = srow(f"CTRV-POS@{arc} (ticks>=1) - (tick 0)", P["CTRV"][1][:, m], P["CTRV"][0][:, m], eid, n_boot)
        out["beside"][f"degradation S-HEAD@{arc}"] = srow(f"S-HEAD@{arc} degradation (ticks>=1) - (tick 0)", H["S"][1][:, m], H["S"][0][:, m], eid, n_boot)
    # scored counts per tick and arc
    out["n_scored"] = {f"tick{k}": [int(R["gt_ok"][:, k, m].sum()) for m in range(3)] for k in range(TICKS)}
    out["n_windows_ge1"] = [int(np.isfinite(P["S"][1][:, m]).sum()) for m in range(3)]
    out["mean_pos_err_m"] = {nm: {"tick0": [float(np.nanmean(P[nm][0][:, m])) for m in range(3)],
                                 "ticks_ge1": [float(np.nanmean(P[nm][1][:, m])) for m in range(3)]} for nm in ("S", "CTRV", "STRAIGHT", "CTRV2")}
    return out


def t_rows(R, ci, n_boot, tag=""):
    """The TACTICAL rows of one run: T-LAT-F1 / T-LON-F1 (ticks >= 1) - (tick 0) on the same windows, the beside rows, the strata."""
    N = len(ci)
    eid2 = np.repeat(ci, 2)
    out = {"decisive": {}, "beside": {}, "confusion": {}, "crawl": {}}
    for hd, nm in (("lat", "T-LAT-F1"), ("lon", "T-LON-F1")):
        n = NCLS[hd]
        true, pred = R["cells"][hd]["true"], R["cells"][hd]["pred"]
        enc = E0._f1_encoded(true, pred, n)
        b, a, W = slots(enc)
        red = f1_red(n)
        row = pbrow(f"{nm} (ticks>=1) - (tick 0)", a, b, eid2, n_boot, red, lower_is_better=False, head=hd)
        v = a[a >= 0]
        v0_ = b[b >= 0]
        row.update(n_windows=int(W.sum()), n_cells_ge1=int((a >= 0).sum()), n_cells_tick0=int((b >= 0).sum()), f1_ge1=float(red(a)), f1_tick0=float(red(b)),
                   two_F=2 * SEED_FLOOR[hd], F=SEED_FLOOR[hd])
        row["n_cells_by_tick"] = [int((enc[:, k] >= 0).sum()) for k in range(TICKS)]
        out["decisive"][nm] = row
        out["beside"][f"{hd} accuracy"] = pbrow(f"{hd} accuracy (ticks>=1) - (tick 0)", a, b, eid2, n_boot, acc_red(n), lower_is_better=False, head=hd,
                                                 acc_ge1=float(acc_red(n)(a)), acc_tick0=float(acc_red(n)(b)))
        # the majority baselines: the banked T0 constants beside the baseline of THESE cells (the cells' own majority class)
        maj = C.majority_baseline(np.where(a >= 0, a // (n + 1), -1), n)
        majt0 = C.majority_baseline(np.where(b >= 0, b // (n + 1), -1), n)
        out["beside"][f"{hd} majority"] = {"banked_T0_quoted": MAJORITY_QUOTED[hd], "cells_ge1": float(maj["macro_f1"]), "cells_ge1_class": int(maj["majority_class"]),
                                           "cells_tick0": float(majt0["macro_f1"]), "cells_tick0_class": int(majt0["majority_class"])}
        out["confusion"][hd] = {"ticks_ge1": C.macro_f1(np.where(a >= 0, a // (n + 1), -1), np.where(a >= 0, a % (n + 1), 0), n)["confusion"],
                                "tick0": C.macro_f1(np.where(b >= 0, b // (n + 1), -1), np.where(b >= 0, b % (n + 1), 0), n)["confusion"],
                                "per_class_f1_ge1": C.macro_f1(np.where(a >= 0, a // (n + 1), -1), np.where(a >= 0, a % (n + 1), 0), n)["per_class_f1"],
                                "per_class_f1_tick0": C.macro_f1(np.where(b >= 0, b // (n + 1), -1), np.where(b >= 0, b % (n + 1), 0), n)["per_class_f1"],
                                "support_ge1": C.macro_f1(np.where(a >= 0, a // (n + 1), -1), np.where(a >= 0, a % (n + 1), 0), n)["support"]}
        # the crawl stratum: ticks whose own speed is below the GT speed at r'_T minus 2 m/s
        cr = R["crawl"][:, 1:].reshape(-1)
        a_cr, a_nc = np.where(cr, a, -1), np.where(~cr, a, -1)
        out["crawl"][hd] = {"n_cells_crawl": int((a_cr >= 0).sum()), "n_cells_noncrawl": int((a_nc >= 0).sum()),
                            "f1_crawl": float(red(a_cr)), "f1_noncrawl": float(red(a_nc)),
                            "ci_crawl": EB(a_cr, eid2, reduce=red, n_boot=n_boot, seed=0), "ci_noncrawl": EB(a_nc, eid2, reduce=red, n_boot=n_boot, seed=0),
                            "frac_ticks_ge1_crawl": float(R["crawl"][:, 1:].mean())}
    # goal macro-F1 (the tactical_metrics recipe) on cells: (ticks >= 1) - (tick 0), same windows
    g = R["cells"]["goal"]
    cid = np.arange(N * TICKS).reshape(N, TICKS)
    valid = g["gw"].any(-1)                                         # a cell with any weighted token
    cell = np.where(valid, cid, -1)
    b, a, W = slots(cell)
    gy, gw, pr = g["gy"].reshape(-1, 22), g["gw"].reshape(-1, 22), g["prob"].reshape(-1, 22)

    def goal_red(v):
        v = _ints(v)
        v = v[v >= 0]
        if not len(v):
            return float("nan")
        f1 = []
        for t in range(22):
            m = gw[v, t]
            if m.sum() == 0 or gy[v][m, t].sum() == 0:
                f1.append(np.nan)
                continue
            p = pr[v][m, t] > 0.5
            y = gy[v][m, t]
            tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
            f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        return float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan")

    out["beside"]["goal macro-F1"] = pbrow("goal macro-F1 (ticks>=1) - (tick 0)", a, b, np.repeat(ci, 2), n_boot, goal_red, record=False, lower_is_better=False,
                                          n_windows=int(W.sum()), f1_ge1=goal_red(a), f1_tick0=goal_red(b))
    return out


# ---------------------------------------------------------------------------------------------------------------- controls
def k1_identity(n_boot):
    """K1: every row's paired delta against ITSELF must be exactly +0 [0, 0] (S vs itself; T vs itself)."""
    bad, n = [], 0
    for name, a, b, eid, reduce in PAIRS:
        r = PB(a, a, eid, n_boot=n_boot, seed=0, reduce=reduce)
        n += 1
        if not (r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0):
            bad.append({"row": name, "delta": r["delta"], "lo": r["lo"], "hi": r["hi"]})
    return {"n_rows": n, "n_nonzero": len(bad), "nonzero": bad[:5], "PASS": n > 0 and not bad}


def k2_frames(tt):
    """K2: s_ckpt_w0 equals s_ckpt transformed by s_pose / pose_w0. Primary = the tool's own float32 arithmetic (xy, metres); beside = a float64
    recompute and the float32 rounding bound COMPUTED from the coordinate magnitudes (R74)."""
    N = tt["N"]
    ck, sp, w0 = tt["s_ckpt"].reshape(-1, 3, 3), tt["s_pose"].reshape(-1, 3), tt["pose_w0"].reshape(-1, 3)
    dump = tt["s_ckpt_w0"].reshape(-1, 3, 3)
    r32 = to_w0_f32(ck, sp, w0)
    r64 = to_w0(ck.astype(np.float64), sp.astype(np.float64), w0)
    d32 = np.abs(r32[..., :2] - dump[..., :2]).max()
    d64 = np.abs(r64[..., :2] - dump[..., :2].astype(np.float64)).max()
    mag = max(float(np.abs(sp[:, :2]).max()), float(np.abs(w0[:, :2]).max())) + float(np.abs(ck[..., :2]).max())
    bound = 8 * 2.0 ** -24 * mag                                    # ~8 float32 roundings at the largest world coordinate in play
    dpsi = float(np.abs(((r32[..., 2] - dump[..., 2]) + np.pi) % (2 * np.pi) - np.pi).max())
    return {"max_abs_xy_m_float32_replica": float(d32), "max_abs_xy_m_float64_recompute": float(d64), "max_abs_psi_rad_float32_replica": dpsi,
            "float32_rounding_bound_m": bound, "tolerance_m": TOL_K2_M, "n_rows": int(len(ck)),
            "PASS": bool(d32 <= TOL_K2_M and np.isfinite(d32)), "float64_within_tolerance": bool(d64 <= TOL_K2_M), "float64_within_computed_bound": bool(d64 <= bound)}


def k3_tick0(R, tt):
    """K3: r' = t0 at tick 0 for every window, matching distance <= 1e-6 m (S and T). The distance is taken in the dump's own precision (the logged pose cast to
    float32, which is what s_pose / t_pose are); the float64 distance and the float32 half-ulp it must sit under are reported beside."""
    t0 = tt["t0"]
    okS, okT = bool((R["rS"][:, 0] == t0).all()), bool((R["rT"][:, 0] == t0).all())
    dS, dT = float(R["dS"][:, 0].max()), float(R["dT"][:, 0].max())
    mag = float(np.abs(tt["s_pose"][:, 0, :2]).max())
    half_ulp = 2.0 ** -24 * mag
    return {"r_eq_t0_S": okS, "r_eq_t0_T": okT, "max_dist_S_m": dS, "max_dist_T_m": dT, "tolerance_m": TOL_K3_M, "dump_precision": "float32",
            "max_dist_S_m_float64_candidates": float(R["dS64"][:, 0].max()), "max_dist_T_m_float64_candidates": float(R["dT64"][:, 0].max()),
            "float32_half_ulp_at_max_coord_m": half_ulp, "float64_dist_within_2x_half_ulp": bool(max(R["dS64"][:, 0].max(), R["dT64"][:, 0].max()) <= 2 * np.sqrt(2) * half_ulp),
            "n_windows": int(len(t0)), "PASS": bool(okS and okT and dS <= TOL_K3_M and dT <= TOL_K3_M)}


def tick0_metrics(R, ci):
    """Tick-0 S pos_err means (windows scored at r' = t0) and T macro-F1: the K4 left-hand side."""
    pe = R["err"]["S"]["pos_err_m"][:, 0]
    pos = [float(np.nanmean(pe[:, m])) for m in range(3)]
    n_valid = [int(np.isfinite(pe[:, m]).sum()) for m in range(3)]
    f1 = {}
    for hd in ("lat", "lon"):
        n = NCLS[hd]
        enc = E0._f1_encoded(R["cells"][hd]["true"][:, 0], R["cells"][hd]["pred"][:, 0], n)
        f1[hd] = float(f1_red(n)(enc))
        f1[hd + "_n"] = int((enc >= 0).sum())
    return {"pos_err_m": pos, "n_valid": n_valid, "f1": f1}


def k4_tick0_is_T0(R, tt, ci, seed, banked_dir, smoke_index=None):
    j = json.load(open(os.path.join(banked_dir, f"eval_M2b_{seed}-sfresh-fresh.json"), encoding="utf-8"))["raw"]
    exp_pos = [j["strategic"][f"pos_err_m@{a}"]["mean"] for a in ARCS]
    exp = {"pos_err_m": exp_pos, "lat": j["tactical"]["lat"]["macro_f1"], "lon": j["tactical"]["lon"]["macro_f1"], "n_valid": j["strategic"]["n_valid"],
           "lat_n": j["tactical"]["lat"]["n"], "lon_n": j["tactical"]["lon"]["n"]}
    got = tick0_metrics(R, ci)
    q = BANKED_QUOTED[seed]
    d = {"pos": [abs(g - e) for g, e in zip(got["pos_err_m"], exp_pos)], "lat": abs(got["f1"]["lat"] - exp["lat"]), "lon": abs(got["f1"]["lon"] - exp["lon"])}
    dq = {"pos": [abs(g - e) for g, e in zip(got["pos_err_m"], q["pos"])], "lat": abs(got["f1"]["lat"] - q["lat"]), "lon": abs(got["f1"]["lon"] - q["lon"])}
    out = {"got": got, "banked_T0_fresh_json": exp, "quoted_in_prereg": q, "abs_diff_vs_json": d, "abs_diff_vs_quoted": dq, "tolerance": TOL_K4}
    full = smoke_index is None
    out["PASS"] = bool(full and max(d["pos"]) <= TOL_K4 and d["lat"] <= TOL_K4 and d["lon"] <= TOL_K4 and got["n_valid"] == exp["n_valid"]
                       and got["f1"]["lat_n"] == exp["lat_n"] and got["f1"]["lon_n"] == exp["lon_n"])
    # per-window diagnostic against the banked T0 npz (the cause, if it fails): S NOW-frame checkpoints and the T argmax
    z = np.load(os.path.join(banked_dir, f"eval_M2b_{seed}-sfresh-fresh.npz"))
    idx = np.arange(tt["N"]) if smoke_index is None else smoke_index
    out["per_window"] = {"max_abs_s_ckpt_now_xy_vs_T0_ckpt_m": float(np.abs(tt["s_ckpt_now"][:, 0, :, :2] - z["ckpt"][idx][..., :2]).max()),
                         "lat_argmax_equal": int((tt["t_lat_logits"][:, 0].argmax(-1) == z["lat_logits"][idx].argmax(-1)).sum()),
                         "lon_argmax_equal": int((tt["t_lon_logits"][:, 0].argmax(-1) == z["lon_logits"][idx].argmax(-1)).sum()), "n": int(tt["N"]),
                         "max_abs_lat_logit_diff": float(np.abs(tt["t_lat_logits"][:, 0] - z["lat_logits"][idx]).max()),
                         "max_abs_lon_logit_diff": float(np.abs(tt["t_lon_logits"][:, 0] - z["lon_logits"][idx]).max())}
    return out


def donor_index(ci, v0):
    """K6 donor per (window, tick): a window of ANOTHER episode at the SAME tick index with the nearest v0; ties go to the lowest window index."""
    N = len(ci)
    don = np.zeros((N, TICKS), np.int64)
    same = ci[:, None] == ci[None, :]
    for k in range(TICKS):
        dist = np.abs(v0[:, k][:, None] - v0[:, k][None, :])
        dist = np.where(same, np.inf, dist)
        don[:, k] = np.argmin(dist, axis=1)                         # argmin returns the FIRST minimum = the lowest index
    return don


def k5_k6(R, tt, ci, n_boot):
    """K5 (mirror S vs S) and K6 (cross-episode donor T vs T), each compared with the arm it degrades."""
    out = {}
    eid = ci
    P = {nm: per_window(R["err"][nm]["pos_err_m"]) for nm in ("S", "MIRROR")}
    k5 = {}
    for m, arc in enumerate(ARCS):
        k5[f"S-POS@{arc}"] = srow(f"K5 S-POS@{arc} (mirror - S, ticks>=1)", P["MIRROR"][1][:, m], P["S"][1][:, m], eid, n_boot, record=False)
    out["K5"] = {"rows": k5, "PASS": all(r["separated"] and r["delta"] > 0 for r in k5.values())}
    don = donor_index(ci, tt["v0"])
    eid2 = np.repeat(ci, 2)
    k6 = {}
    n_same = 0
    for hd, nm in (("lat", "T-LAT-F1"), ("lon", "T-LON-F1")):
        n = NCLS[hd]
        true, pred = R["cells"][hd]["true"], R["cells"][hd]["pred"]
        pred_d = np.stack([pred[don[:, k], k] for k in range(TICKS)], 1)
        b0, a_self, W = slots(E0._f1_encoded(true, pred, n))
        _, a_don, _ = slots(E0._f1_encoded(true, pred_d, n))
        red = f1_red(n)
        k6[nm] = pbrow(f"K6 {nm} (donor - self, ticks>=1)", a_don, a_self, eid2, n_boot, red, record=False, lower_is_better=False,
                       f1_donor=float(red(a_don)), f1_self=float(red(a_self)), n_windows=int(W.sum()))
    out["K6"] = {"rows": k6, "donor_cross_episode_all": bool((ci[don] != ci[:, None]).all()), "PASS": all(r["worse"] for r in k6.values())}
    return out


def k7_bit_exact(dump_dir, banked_dir, seed, ref_dir=None):
    """K7: the causal run's cl / fl / g dumps equal the banked causal dumps bit for bit (banked = float32 consolidation of the float64 originals).
    Also the manifest's sha256 over the float64 originals (every key, sorted). Smoke: against the flags-OFF run's dumps, every key, native dtype."""
    fs = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    zs = [np.load(f) for f in fs]
    d = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
    out = {"n_episode_files": len(fs), "n_windows": int(len(d["eid"]))}
    if ref_dir:
        rs = sorted(glob.glob(os.path.join(ref_dir, "ep*.npz")))
        rz = [np.load(f) for f in rs]
        r = {k: np.concatenate([z[k] for z in rz]) for k in rz[0].files}
        out["keys_equal"] = {k: bool(d[k].dtype == r[k].dtype and d[k].tobytes() == r[k].tobytes()) for k in r}
        out["PASS"] = all(out["keys_equal"].values()) and set(d) == set(r)
        return out
    bank = np.load(os.path.join(banked_dir, f"dumps_HACC_{seed}_m10_causal.npz"))
    out["bitwise_vs_banked_float32"] = {k: bool(np.ascontiguousarray(d[k].astype(np.float32)).tobytes() == np.ascontiguousarray(bank[k]).tobytes()) for k in ("cl", "fl", "g")}
    out["shapes_equal"] = {k: bool(d[k].shape == bank[k].shape) for k in ("cl", "fl", "g")}
    out["max_abs_diff_vs_banked_float32"] = {k: (float(np.abs(d[k].astype(np.float64) - bank[k].astype(np.float64)).max()) if d[k].shape == bank[k].shape else None)
                                             for k in ("cl", "fl", "g")}
    man = json.load(open(os.path.join(banked_dir, "dumps_manifest.json"), encoding="utf-8"))[f"HACC_{seed}_m10_causal"]
    h = hashlib.sha256(b"".join(np.ascontiguousarray(d[k]).tobytes() for k in sorted(d))).hexdigest()
    out["sha256_f64_concat"] = {"got": h, "manifest": man["sha256_f64_concat"], "equal": h == man["sha256_f64_concat"]}
    out["all_keys_float32_equal"] = {k: bool(k in d and np.ascontiguousarray(d[k].astype(np.float32) if d[k].dtype == np.float64 else d[k]).tobytes() == np.ascontiguousarray(bank[k]).tobytes()) for k in bank.files}
    out["PASS"] = all(out["bitwise_vs_banked_float32"].values())
    return out


# ---------------------------------------------------------------------------------------------------------------- outcomes
def strategic_outcome(rows_by_seed):
    flags, better_all, worse_any = {}, True, False
    for s, rows in rows_by_seed.items():
        for arc in ARCS:
            r = rows[f"S-POS@{arc}"]
            flags[f"{s}@{arc}"] = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "better": r["better"], "worse": r["worse"]}
            better_all &= r["better"]
            worse_any |= r["worse"]
    o = "WORSE" if worse_any else ("S ADDS ROUTE" if better_all else "NO WORSE")
    return {"OUTCOME": o, "committed_next": NEXT_S[o], "per_seed_arc": flags,
            "condition": {"S ADDS ROUTE": "S-POS separated-better than CTRV at all three arcs, at both seeds",
                          "NO WORSE": "no arc separated-worse at either seed, but not ADDS",
                          "WORSE": "any arc separated-worse at ≥ 1 seed"}[o]}


def tactical_outcome(rows_by_seed):
    flags, holds, degr = {}, True, False
    for s, rows in rows_by_seed.items():
        for hd, nm in (("lat", "T-LAT-F1"), ("lon", "T-LON-F1")):
            r, tf = rows[nm], 2 * SEED_FLOOR[hd]
            h = r["lo"] >= -tf
            d = (r["hi"] < 0) and (r["delta"] < -tf)
            flags[f"{s} {nm}"] = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "two_F": tf, "lo_ge_minus_2F": bool(h), "hi_lt_0_and_delta_lt_minus_2F": bool(d)}
            holds &= h
            degr |= d
    assert not (holds and degr), "HOLDS and DEGRADES cannot both be true (lo >= -2F and delta < -2F)"
    o = "DEGRADES" if degr else ("HOLDS" if holds else "INCONCLUSIVE")
    return {"OUTCOME": o, "committed_next": NEXT_T[o], "per_seed_head": flags,
            "condition": {"HOLDS": "the lower bound of Δ ≥ −2F, for lat and lon, at both seeds",
                          "DEGRADES": "the upper bound of Δ < 0 and the point Δ < −2F, for lat or lon, at ≥ 1 seed",
                          "INCONCLUSIVE": "otherwise"}[o]}


# ---------------------------------------------------------------------------------------------------------------- drivers
def run_one(ctx, a, path, dump_dir, label, seed, n_boot, banked_dir, smoke_index=None, ref_dump=None, full_rows=True, k7=True):
    t = time.time()
    tt = load_ticks(path)
    say(f"[{label}] ticks loaded: N={tt['N']} windows, {tt['N'] * TICKS} rows, tag={tt['meta']['tag']}")
    ci = tt["ci"]
    R = analyze_run(ctx, tt, a, label, n_boot, dump_dir)
    say(f"[{label}] matched + rebuilt + scored in {time.time() - t:.0f}s")
    res = {"label": label, "tag": tt["meta"]["tag"], "ticks_file": os.path.abspath(path), "ticks_sha256": sha256_file(path), "n_windows": tt["N"],
           "off_route": R["off_route"], "executed_track_rebuild_X1": R["x1"], "GT_frame_check_EXTRA": R["gt_frame"], "ctrv_curvature": {
               "frames_used_min": int(R["kappa_frames"].min()), "n_ticks_zero_frames": int((R["kappa_frames"] == 0).sum()),
               "kappa_abs_p50_p95_max": [float(np.percentile(np.abs(R["kappa"]), 50)), float(np.percentile(np.abs(R["kappa"]), 95)), float(np.abs(R["kappa"]).max())]}}
    res["K2"] = k2_frames(tt)
    res["K3"] = k3_tick0(R, tt)
    res["tick0"] = tick0_metrics(R, ci)
    if full_rows:
        res["STRATEGIC"] = s_rows(R, ci, n_boot)
        res["TACTICAL"] = t_rows(R, ci, n_boot)
        res.update(k5_k6(R, tt, ci, n_boot))
        res["K1"] = k1_identity(n_boot)
        PAIRS.clear()
    if k7 and (dump_dir is not None):
        res["K7"] = k7_bit_exact(dump_dir, banked_dir, seed, ref_dump)
    say(f"[{label}] done in {time.time() - t:.0f}s")
    return res, R, tt


def dev_t0_regression_effects(ctx, banked_dir, n_boot):
    """R75: compute the regression arms' expected effect on the rows they must move BEFORE the decisive run -- on the banked T0 fresh evals (all 2,399
    windows, tick 0 only: not outcome-bearing). K5: mirror the NOW-frame S checkpoints; K6: speed-matched cross-episode donor T (speed = the cache's v at t0)."""
    J = ctx.J
    wins = ctx.wins_grid
    ci = np.array([w[0] for w in wins])
    t0 = np.array([w[1] for w in wins])
    out = {"note": "DEV on the banked T0 fresh evals (tick 0, 2,399 windows); the rows of K5 / K6 at the decisive tick-0 level; NOT the decisive run"}
    for seed in ("s0", "s1"):
        z = np.load(os.path.join(banked_dir, f"eval_M2b_{seed}-sfresh-fresh.npz"))
        rc = J["rc"][J["off"][ci] + t0].astype(np.float64)
        ok = J["rc_ok"][J["off"][ci] + t0] & J["has"][J["off"][ci] + t0][:, None]
        ck = z["ckpt"].astype(np.float64)
        mir = ck * np.array([1.0, -1.0, -1.0])
        eS = C.checkpoint_errors(ck, rc, ok)["pos_err_m"]
        eM = C.checkpoint_errors(mir, rc, ok)["pos_err_m"]
        k5 = {f"S-POS@{arc}": srow(f"DEV K5 S-POS@{arc} (mirror - S, tick 0, T0 data)", eM[:, m], eS[:, m], ci, n_boot, record=False) for m, arc in enumerate(ARCS)}
        v0 = np.array([ctx.clip(int(c))[int(t), 3] for c, t in zip(ci, t0)])
        same = ci[:, None] == ci[None, :]
        don = np.argmin(np.where(same, np.inf, np.abs(v0[:, None] - v0[None, :])), axis=1)
        eid2 = ci
        k6 = {}
        for hd, nm in (("lat", "T-LAT-F1"), ("lon", "T-LON-F1")):
            n = NCLS[hd]
            row = J["off"][ci] + t0
            cls = J[f"{hd}_cls"][row].astype(np.int64)
            true = np.where(J["has"][row] & (cls >= 0), cls, -1)
            pred = z[f"{hd}_logits"].argmax(-1)
            ea, eb = E0._f1_encoded(true, pred[don], n), E0._f1_encoded(true, pred, n)
            red = f1_red(n)
            k6[nm] = pbrow(f"DEV K6 {nm} (donor - self, tick 0, T0 data)", ea, eb, eid2, n_boot, red, record=False, lower_is_better=False, f1_donor=float(red(ea)), f1_self=float(red(eb)))
        out[seed] = {"K5": k5, "K6": k6}
    PAIRS.clear()
    return out


def main(argv=None):
    a = parse(argv)
    setup_imports(a.tool_root)
    t_start = time.time()
    tool_file = os.path.join(a.tool_root, "stack", "scripts", "v8f_t1_eval.py")
    say(f"mode={a.mode}  tool={tool_file} sha256={sha256_file(tool_file)[:16]}  reader={sha256_file(os.path.abspath(__file__))[:16]}")
    say(f"numpy {np.__version__} torch {torch.__version__}  B={a.n_boot}")
    ctx = Ctx(a)
    res = {"prereg": "Project Steering/PREREG_V8F_T1_HEADS.md @ae659d49", "claim_row": "H-V8F-T1-HEADS-1", "mode": a.mode,
           "tier": "T1 (T1-kin: a self-action OPEN loop, never 'closed loop'); nav dead-reckoned; S and T sample-and-hold at the tool's ages",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0, over the 141 episodes (ineligible windows stay in the pool as NaN / -1 cells); "
                        "F1 rows through v8f_mini_eval._f1_reducer. It answers: another draw of EPISODES. The training-seed variance is answered only by the two seeds.",
           "tool_sha256": sha256_file(tool_file), "reader_sha256": sha256_file(os.path.abspath(__file__)), "seed_floor": SEED_FLOOR,
           "controls": {}, "runs": {}}
    if a.mode == "smoke":
        label = "SMOKE"
        res["LABEL"] = "SMOKE: 24 windows, logged bucket, seed 0 -- development only, NOT outcome-bearing"
        tt0 = np.load(a.smoke_ticks, allow_pickle=False)
        n_w = int(tt0["n_windows"])
        lin = np.linspace(0, len(ctx.wins_grid) - 1, n_w).round().astype(int)
        r, R, tt = run_one(ctx, a, a.smoke_ticks, a.smoke_dump_dir, "SMOKE", a.smoke_seed, a.n_boot, a.banked_dir, smoke_index=lin, ref_dump=a.smoke_ref_dump_dir)
        assert [(int(ctx.wins_grid[i][0]), int(ctx.wins_grid[i][1])) for i in lin] == [(int(c), int(t)) for c, t in zip(tt["ci"], tt["t0"])], "smoke windows are not the --max-windows subset"
        r["K4_smoke_per_window"] = k4_tick0_is_T0(R, tt, tt["ci"], a.smoke_seed, a.banked_dir, smoke_index=lin)
        res["runs"]["SMOKE"] = r
        res["dev_t0_regression_arm_effects"] = dev_t0_regression_effects(ctx, a.banked_dir, a.n_boot)
        res["controls"] = {k: r[k]["PASS"] for k in ("K1", "K5", "K6", "K7") if k in r}
        res["controls"]["K2"], res["controls"]["K3"] = r["K2"]["PASS"], r["K3"]["PASS"]
        pw = r["K4_smoke_per_window"]["per_window"]
        res["controls"]["K4 (SMOKE: per-window vs the T0 npz on the 24 windows; the aggregate K4 needs all 2,399)"] = bool(
            pw["lat_argmax_equal"] == pw["n"] and pw["lon_argmax_equal"] == pw["n"] and pw["max_abs_s_ckpt_now_xy_vs_T0_ckpt_m"] <= TOL_K4
            and pw["max_abs_lat_logit_diff"] <= TOL_K4 and pw["max_abs_lon_logit_diff"] <= TOL_K4)
    else:
        chain = open(os.path.join(a.run_dir, "logs", "chain.out"), encoding="utf-8").read() if not a.dev_fake_run else "ZZT1HD-DONE 4"
        if "ZZT1HD-STOP" in chain:
            raise SystemExit("REFUSED: chain.out has ZZT1HD-STOP; partial outputs are not read")
        if "ZZT1HD-DONE 4" not in chain:
            raise SystemExit("NOT READY: chain.out has no ZZT1HD-DONE 4")
        tags = {("causal", s): f"HACC_{s}_m10_causal_heads" for s in ("s0", "s1")}
        tags.update({("logged", s): f"HACC_{s}_m10_logged_heads" for s in ("s0", "s1")})
        if a.dev_fake_run:
            res["LABEL"] = "DEV FAKE RUN: the smoke ticks stand in for all four runs; REFUSED by construction; NOT a read"
        else:
            res["chain_out_sha256"] = sha256_file(os.path.join(a.run_dir, "logs", "chain.out"))
        for (cfg, s), tag in tags.items():
            path = a.smoke_ticks if a.dev_fake_run else os.path.join(a.run_dir, "ticks", f"ticks_{tag}.npz")
            dd = a.smoke_dump_dir if a.dev_fake_run else os.path.join(a.run_dir, f"dump_{tag}")
            dec = cfg == "causal"
            r, R, tt = run_one(ctx, a, path, dd, tag, s, a.n_boot, a.banked_dir, full_rows=dec, k7=dec)
            if not a.dev_fake_run:
                assert tt["N"] == len(ctx.wins_grid), "a decisive / K4 run must cover the 2,399 grid windows"
                assert [(int(c), int(t)) for c, t in zip(tt["ci"], tt["t0"])] == [(int(w[0]), int(w[1])) for w in ctx.wins_grid], "window list differs from the grid"
                r["run_json_sha256"] = sha256_file(os.path.join(a.run_dir, f"t1v8f_{tag}.json"))
            if not dec:
                r["K4"] = k4_tick0_is_T0(R, tt, tt["ci"], s, a.banked_dir)
            res["runs"][f"{cfg}_{s}"] = r
        # ---- the controls, PASS flags first
        ctl = res["controls"]
        dec = [res["runs"][f"causal_{s}"] for s in ("s0", "s1")]
        allr = list(res["runs"].values())
        ctl["K1"] = {"PASS": all(r["K1"]["PASS"] for r in dec), "runs": {r["label"]: r["K1"] for r in dec}}
        ctl["K2"] = {"PASS": all(r["K2"]["PASS"] for r in allr), "runs": {r["label"]: r["K2"] for r in allr}}
        ctl["K3"] = {"PASS": all(r["K3"]["PASS"] for r in allr), "runs": {r["label"]: r["K3"] for r in allr}}
        ctl["K4"] = {"PASS": all(res["runs"][f"logged_{s}"]["K4"]["PASS"] for s in ("s0", "s1")), "runs": {res["runs"][f"logged_{s}"]["label"]: res["runs"][f"logged_{s}"]["K4"] for s in ("s0", "s1")}}
        ctl["K5"] = {"PASS": dec[0]["K5"]["PASS"], "at": "causal s0 (decisive)", "s0": dec[0]["K5"], "s1_beside": dec[1]["K5"]}
        ctl["K6"] = {"PASS": dec[0]["K6"]["PASS"], "at": "causal s0 (decisive)", "s0": dec[0]["K6"], "s1_beside": dec[1]["K6"]}
        ctl["K7"] = {"PASS": all(r["K7"]["PASS"] for r in dec), "runs": {r["label"]: r["K7"] for r in dec}}
        res["dev_t0_regression_arm_effects"] = dev_t0_regression_effects(ctx, a.banked_dir, a.n_boot)
    pass_flags = {k: (v["PASS"] if isinstance(v, dict) else v) for k, v in res["controls"].items()}
    res["controls_PASS"] = pass_flags
    if a.mode == "real":
        refused = not all(pass_flags.values())
        res["REFUSED"] = bool(refused)
        if refused:                                                 # no outcome word is computed: the rows exist in `runs` and are NOT TO BE READ (R74)
            res["OUTCOME"] = {"STRATEGIC": "REFUSED", "TACTICAL": "REFUSED", "failed_controls": [k for k, v in pass_flags.items() if not v],
                              "note": "a control failed: PREREG section 5 'REFUSED: any control fails'. The rows in `runs` are written but NOT TO BE READ as outcomes."}
        else:
            rows_s = {s: res["runs"][f"causal_{s}"]["STRATEGIC"]["decisive"] for s in ("s0", "s1")}
            rows_t = {s: res["runs"][f"causal_{s}"]["TACTICAL"]["decisive"] for s in ("s0", "s1")}
            res["OUTCOME"] = {"STRATEGIC": strategic_outcome(rows_s), "TACTICAL": tactical_outcome(rows_t)}
    else:
        rows_s = {"s0": res["runs"]["SMOKE"]["STRATEGIC"]["decisive"]}
        rows_t = {"s0": res["runs"]["SMOKE"]["TACTICAL"]["decisive"]}
        res["OUTCOME_SMOKE_exercise_only"] = {"STRATEGIC": "(one seed, 24 windows: the outcome function ran as an exercise)", "rows_s": {k: {kk: v[kk] for kk in ("delta", "lo", "hi", "separated")} for k, v in rows_s["s0"].items()},
                                              "rows_t": {k: {kk: v[kk] for kk in ("delta", "lo", "hi", "separated")} for k, v in rows_t["s0"].items()}}
    res["wall_s"] = time.time() - t_start
    # ---- a compact table of the decisive rows (what the outcome functions read), and the extra checks that are NOT in the PREREG
    def compact(r):
        c = {k: r[k] for k in ("delta", "lo", "hi", "separated") if k in r}
        c["n"] = r.get("n_windows")
        if "n_cells_ge1" in r:
            c["n_cells_ge1"], c["n_cells_tick0"] = r["n_cells_ge1"], r["n_cells_tick0"]
        return c
    keys = ["SMOKE"] if a.mode == "smoke" else ["causal_s0", "causal_s1"]
    res["decisive_rows"] = {k: {"configuration": ("--limit logged, SMOKE" if a.mode == "smoke" else "--limit causal, m = 10"),
                               "rows": {**{n: compact(r) for n, r in res["runs"][k]["STRATEGIC"]["decisive"].items()},
                                        **{n: compact(r) for n, r in res["runs"][k]["TACTICAL"]["decisive"].items()}},
                               "off_route": res["runs"][k]["off_route"]} for k in keys}
    x = {}
    for k, r in res["runs"].items():
        e = r["executed_track_rebuild_X1"]
        x[k] = {"X1_executed_track_rebuilt_equals_dumped_cl": (None if e["max_cl_m"] is None else bool(e["max_cl_m"] <= 1e-9)),
                "X1_executed_pose_at_ticks_equals_dump_f64": bool(e["max_pose_tick_f64"] <= 1e-9),
                "X1_s_pose_t_pose_equal_rebuilt_to_float32": bool(max(e["max_s_pose_vs_rebuilt_m_f32_tol"], e["max_t_pose_vs_rebuilt_m_f32_tol"]) <= 1e-4), "detail": e}
    res["extra_checks_NOT_in_PREREG_not_gating"] = x
    ordered = {"controls_PASS": res.pop("controls_PASS")}
    for k in ("REFUSED", "OUTCOME", "OUTCOME_SMOKE_exercise_only", "LABEL", "decisive_rows"):
        if k in res:
            ordered[k] = res.pop(k)
    ordered.update(res)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(ordered, open(a.out, "w", encoding="utf-8"), indent=1, default=jenc)
    say("wrote", a.out)
    say("CONTROLS", json.dumps(ordered["controls_PASS"], default=jenc))
    if a.mode == "real":
        say("OUTCOME", json.dumps({k: (v["OUTCOME"] if isinstance(v, dict) else v) for k, v in ordered["OUTCOME"].items() if k in ("STRATEGIC", "TACTICAL")}, default=jenc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
