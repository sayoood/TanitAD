"""Gates of E3: T2g playback identity, the RR index, the live path (CPU). Every `check_*` takes `mut=None`; `code/run_mutations_e3.py` runs each
with a deliberate regression that must raise AssertionError (RED). Run from t2v8f/: CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests/test_t2v8f_e3.py -q"""
from __future__ import annotations

import hashlib
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.path.insert(0, HERE)
sys.path.insert(0, "F:/TanitAD-scratch/t2v8f/kit")
import numpy as np  # noqa: E402

import t2v8f_exec as X  # noqa: E402
import test_t2v8f_gates as G  # noqa: E402


class ReplayPolicy:
    """Returns the LOGGED plan at NOW (scaled by `scale`): the car then follows the human (scale 1) or at `scale` x the human's speed."""
    window = 17

    def __init__(self, gt_T, scale=1.0):
        self.rows, self.scale = X.logged_rows(gt_T), scale

    def set_t0(self, gt_T, f0, **k):
        r = self.rows[:f0 + 1]
        self.raw_off = (len(r) - 1) % 2
        self.track = r[self.raw_off:].copy()
        self.log = self.rows[self.raw_off:]

    def set_nav(self, **k):
        pass

    def set_history(self, h, n_new=None):
        self.track = np.vstack([self.track, np.asarray(h)[len(h) - n_new:]])

    def plan(self, frames, intr, v0, nav_cmd):
        assert (len(self.track) - 1) % 2 == 0
        return self.scale * X.logged_plan(self.log, len(self.track) - 1, pose=self.track[-1]), {}


def scene(i=0):
    """A real logged clip (eval-141 native track) as a scene: gt_T from the 10 Hz rows, 10 Hz timestamps. Window i of the screened picks."""
    W = G.world()
    win = G.pick(4)[i]
    clip = str(W["J"]["clip"][win[0]])
    r = W["tracks"][clip]
    gt_T = []
    for x, y, yaw, v in r:
        T = X.rz(yaw)
        T[0, 3], T[1, 3] = x, y
        gt_T.append(T)
    return clip, r, gt_T, 1.0e12 + 1e5 * np.arange(len(gt_T)), 2 * int(win[1])


def render_pose(T, t_us):
    return (np.array(T), float(t_us))


def check_t2g(mut=None, n=3):
    """T2g: the logged plan executed by playback reproduces the logged ego track (grid rows) to 1e-6 m; also: executor == the tool's commit_plan."""
    worst, odd = 0.0, 0.0
    for i in range(n):
        clip, r, gt_T, ts, f0 = scene(i)
        pol = ReplayPolicy(gt_T)
        rec = X.run_t2_rollout(pol, render_pose, gt_T, ts, f0, None, condition="RR", skew=1 if mut == "knot_shift" else 0)
        tr, t0 = rec["track"], rec["t0_step"]
        off = rec["raw_off"]
        L = r[off:off + len(tr)]
        worst = max(worst, float(np.abs(tr[0::2, :2] - L[0::2, :2]).max()))
        odd = max(odd, float(np.abs(tr[1::2, :2] - L[1::2, :2]).max()))
        assert rec["policy_track_equals_executor"]
    assert worst < 1e-6, f"T2g: grid rows differ from the logged track by {worst:.3g} m"
    return {"max_grid_dev_m": worst, "max_odd_row_dev_m": odd, "n_scenes": n}


def poses(rec):
    return [c for c in rec["calls"] if c["phase"] == "exec"]


def check_rr_index(mut=None):
    """RR with the car at the logged speed equals logged playback (every rendered pose and time); at half the logged speed it lags (a control that must differ)."""
    clip, r, gt_T, ts, f0 = scene(1)
    mode = "time" if mut == "rr_time_index" else "reindex"
    a = X.run_t2_rollout(ReplayPolicy(gt_T), render_pose, gt_T, ts, f0, None, condition="RR", rr_mode=mode)
    b = X.run_t2_rollout(ReplayPolicy(gt_T), render_pose, gt_T, ts, f0, None, condition="logged")
    dev = max(float(np.abs(x["pose"] - y["pose"]).max()) for x, y in zip(poses(a), poses(b)))
    dt = max(abs(x["t_us"] - y["t_us"]) for x, y in zip(poses(a), poses(b)))
    assert len(poses(a)) == len(poses(b)) == 30 and dev < 1e-9 and dt == 0.0, f"RR at the logged speed != logged playback (pose dev {dev:.3g}, time dev {dt} us)"
    h = X.run_t2_rollout(ReplayPolicy(gt_T, 0.5), render_pose, gt_T, ts, f0, None, condition="RR", rr_mode=mode)
    hl = X.run_t2_rollout(ReplayPolicy(gt_T, 0.5), render_pose, gt_T, ts, f0, None, condition="logged", rr_mode=mode)
    lag = max(float(np.abs(x["pose"][:3, 3] - y["pose"][:3, 3]).max()) for x, y in zip(poses(h), poses(hl)))
    taus = [c["tau"] for c in poses(h)]
    assert lag > 1.0 and taus[-1] < poses(h)[-1]["i"], f"RR at half speed does not lag the clock (lag {lag:.3g} m)"
    assert all(t1 >= t0 for t0, t1 in zip(taus, taus[1:])), "tau not monotone"
    return {"dev_at_logged_speed_m": dev, "half_speed_lag_m": lag, "tau_last_vs_i": [taus[-1], poses(h)[-1]["i"]]}


class HashLookupEncoder:
    """The stub renderer returns a LOGGED frame: frame bytes -> the banked tokens of that logged step. Nothing else is accepted."""
    snapshot = None
    canon = {"mode": "hash_lookup"}

    def __init__(self):
        self.tab, self.shift = {}, 0

    def add(self, img, tok):
        self.tab[hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()] = tok

    def tokens(self, frame, intr=None, already_cyl=False):
        return self.tab[hashlib.md5(np.ascontiguousarray(frame).tobytes()).hexdigest()]


def check_live_path(mut=None, n=2):
    """`plan(frames, intr)` through the stub renderer's real frames and an encoder that returns the banked tokens of the logged frame equals the cached-feature plan, bit for bit."""
    import stub_gsplat_renderer as S
    W = G.world()
    for win in G.pick(n):
        ci, t0 = win[0], win[1]
        clip = str(W["J"]["clip"][ci])
        o, nn = int(W["J"]["off"][ci]), int(W["J"]["n"][ci])
        tr = W["tracks"][clip]
        T0 = X.rz(tr[0, 2]) * 0 + np.eye(4)
        c, s = np.cos(tr[2 * t0 - 40, 2]), np.sin(tr[2 * t0 - 40, 2])
        sr = S.NuRecGsplatRenderer("stub")
        tok = G.banked_tokens(clip)
        enc = HashLookupEncoder()
        frames = [None] * 19
        for k in range(10):                                                    # grid steps t0-9 .. t0 at raw offsets -18 .. 0
            st = t0 - 9 + k
            row = tr[2 * st]
            p0 = tr[2 * (t0 - 9)]
            d = row[:2] - p0[:2]
            cc, ss = np.cos(p0[2]), np.sin(p0[2])
            T = X.rz(row[2] - p0[2])
            T[0, 3], T[1, 3] = cc * d[0] + ss * d[1], -ss * d[0] + cc * d[1]
            img = sr.render(T @ S._T_SENSOR_RIG, 0.0)[0]
            enc.add(img, tok[st - 1] if (mut == "lookup_off_by_one" and st == t0) else tok[st])
            frames[18 - 2 * (9 - k)] = img
        assert len({hashlib.md5(f.tobytes()).hexdigest() for f in frames if f is not None}) == 10, "stub frames collide"
        live, _, _ = G.make_policy(win, arm="model", schedule="periodic")
        cached, _, _ = G.make_policy(win, arm="model", schedule="periodic")
        live.tok.clear()
        live.encoder = enc
        v0 = float(live.track[-1, 3])
        pl, _ = live.plan(frames, None, v0, 0)
        pc, _ = cached.plan(None, None, v0, 0)
        assert np.array_equal(pl, pc), f"live-path plan differs from the cached-feature plan (max |d| {np.abs(pl - pc).max():.3g} m)"
    return n


class OffsetPolicy(ReplayPolicy):
    """The logged plan plus a lateral drift in the car's own frame (0 to 0.8 m left over the 6 s): the executed track leaves the logged path, so T2 and the logged pose differ."""

    def plan(self, frames, intr, v0, nav_cmd):
        p, e = super().plan(frames, intr, v0, nav_cmd)
        return p + np.stack([np.zeros(30), 0.8 * np.arange(1, 31) / 30.0], 1), e


def sloped_scene(i=0):
    """`scene(i)` with a real road surface: z rises 2 cm per tick, roll and pitch vary along the clip (a flat scene cannot tell the ground follower from the identity)."""
    clip, r, gt_T, ts, f0 = scene(i)
    out = []
    for k, T in enumerate(gt_T):
        yaw = X.yaw_of(T)
        roll, pitch = 0.02 * np.sin(0.07 * k), 0.015 * np.cos(0.05 * k)
        cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        U = np.eye(4)
        U[:3, :3] = X.rz3(yaw) @ Rx @ Ry
        U[:3, 3] = (T[0, 3], T[1, 3], 0.02 * k)
        out.append(U)
    return clip, r, out, ts, f0


def _bug_4x4(gt_T, xy, yaw):
    """The first version of `ground_correct` (4x4 `rz` times a 3x3 block): the deliberate-regression arm."""
    P_ = np.stack([T[:2, 3] for T in gt_T])
    G = gt_T[int(np.argmin(np.linalg.norm(P_ - np.asarray(xy)[None, :2], axis=1)))]
    out = np.eye(4)
    out[:3, :3] = X.rz(yaw) @ (X.rz(-X.yaw_of(G)) @ G[:3, :3])
    out[:3, 3] = (xy[0], xy[1], G[2, 3])
    return out


def _nearest(gt_T, xy):
    return gt_T[int(np.argmin(np.linalg.norm(np.stack([t[:2, 3] for t in gt_T]) - np.asarray(xy)[None, :2], axis=1)))]


_REAL = X.ground_correct                                                                       # the genuine function, captured before any mutation replaces the module attribute
T2_MUTATIONS = {
    "bug_4x4": _bug_4x4,                                                                       # raises on the first step
    "logged_pose": lambda g, xy, yaw: _nearest(g, xy).copy(),                                  # the logged pose, not the car's
    "z_from_car": lambda g, xy, yaw: _with_z(_REAL(g, xy, yaw), 0.0),               # no road height
    "yaw_from_log": lambda g, xy, yaw: _REAL(g, xy, X.yaw_of(_nearest(g, xy))),     # the logged heading
}


def _with_z(T, z):
    T = T.copy()
    T[2, 3] = z
    return T


def check_t2_condition(mut=None):
    """Condition T2 end to end through `run_t2_rollout` (E3's other tests ran RR and logged only, and T2 raised on its first step): every rendered exec pose is the HARNESS'
    `GroundFollower.correct` of the executed pose (x, y, yaw = the car's; z, roll, pitch = the nearest logged pose's) at time t_us = ts[f0] + 0.1 s * (raw step - 2 t0);
    the warm-up renders are the logged poses; the car left the logged path, and T2 renders differ from RR renders."""
    import closedloop_drive as cd
    clip, r, gt_T, ts, f0 = sloped_scene(2)
    saved = X.ground_correct
    if mut:
        X.ground_correct = T2_MUTATIONS[mut]
    try:
        try:
            rec = X.run_t2_rollout(OffsetPolicy(gt_T), render_pose, gt_T, ts, f0, None, condition="T2")
        except Exception as e:                                                     # noqa: BLE001
            raise AssertionError(f"the T2 rollout raised {type(e).__name__}: {str(e)[:80]}")
    finally:
        X.ground_correct = saved
    gf = cd.GroundFollower(gt_T)
    ex = [c for c in rec["calls"] if c["phase"] == "exec"]
    warm = [c for c in rec["calls"] if c["phase"] == "warm"]
    assert len(ex) == 30 and len(warm) >= 9, (len(ex), len(warm))
    tr, t0 = rec["track"], rec["t0_step"]
    worst, worst_t = 0.0, 0.0
    for q, c in enumerate(ex):
        n_now = 2 * t0 + 2 * (q + 1)
        car = X.rz(float(tr[n_now, 2]))
        car[0, 3], car[1, 3] = tr[n_now, 0], tr[n_now, 1]
        want = gf.correct(car)[0]
        worst = max(worst, float(np.abs(c["pose"] - want).max()))
        worst_t = max(worst_t, abs(c["t_us"] - (ts[f0] + 1e5 * (n_now - 2 * t0))))
    assert worst < 1e-12, f"T2 render pose differs from GroundFollower.correct of the executed pose by {worst:.3g}"
    assert worst_t == 0.0, f"T2 render time off by {worst_t} us"
    z_run = np.array([c["pose"][2, 3] for c in ex])
    assert z_run.min() > 0.02 * (f0 - 1), "the render height is not the logged road surface"
    drift = max(float(np.abs(c["pose"][:2, 3] - _nearest(gt_T, c["pose"][:2, 3])[:2, 3]).max()) for c in ex)
    assert drift > 0.05, f"the car never left the logged path ({drift:.3g} m): the check would not tell T2 from the logged pose"
    for c, f in zip(warm, [w["f"] for w in warm]):
        assert np.array_equal(c["pose"], gt_T[f]), "warm-up frames are not at the logged poses"
    rr = X.run_t2_rollout(OffsetPolicy(gt_T), render_pose, gt_T, ts, f0, None, condition="RR")
    dpose = max(float(np.abs(a["pose"] - b["pose"]).max()) for a, b in zip(ex, poses(rr)))
    assert dpose > 0.05, "T2 and RR render the same poses"
    return {"max_dev_vs_harness": worst, "drift_from_logged_path_m": drift, "t2_vs_rr_pose_dev_m": dpose, "n_exec": len(ex)}


def test_t2_condition():
    print(check_t2_condition())


def test_t2g():
    print(check_t2g())


def test_rr_index():
    print(check_rr_index())


def test_live_path():
    check_live_path()
