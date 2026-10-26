#!/usr/bin/env python3
"""v8F-mini T1-kin: a SELF-ACTION rollout on the refav1 trunk-probe grid (design: ``DESIGN_T1_V8F.md``).

TIER, in the vocabulary of ``Project Steering/EVAL_DOCTRINE.md`` (PI amendment 2026-09-02): **T1 = self-action OPEN loop.** The
model's kinematic state is its OWN executed motion, rebuilt at every replan; the camera stream is NOT re-rendered (that is
T2, not provisioned). So this tool never says "closed loop" and supports only open-loop trajectory-quality claims, not
"the model drives". Every output carries ``scene``, ``nav``, ``limit`` and ``replan_steps`` beside the numbers, because
those four decide how strict the T1 is (DESIGN section 4).

WHAT IS REPLACED by the model's own executed motion (everything the model reads that is built from the ego track):
``p10f`` (8 kinematic features), ``base`` (kd_x path), ``pose_now / pose_t / pose_s`` (odometry), and -- by default -- the
route countdown in ``nav``. WHAT STAYS LOGGED: ``feats`` / ``frame_ok`` (camera), ``limit`` / ``limit_ok`` (a declared
oracle, D-SPEEDBAND-EGO-ECHO-1), the sample-and-hold ages (a schedule). The model is handed ONLY the keys it reads
(``MODEL_KEYS``); a logged ego row after t0 does not exist in the per-window view (NaN), so it cannot be read.

NO MODEL CHANGE: ``V8FMini.forward_infer`` is called as is; no new input, no new weight.

The scoring is ``taniteval/tools/t1_eval.py::analyze`` (four families, S-curve, lag, paired episode-cluster bootstrap),
fed per-episode dumps in its schema -- not re-implemented.

Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from taniteval import floors_kin as fk  # noqa: E402
from tanitad.refs.refcv6_max_speed import speed_max_bin  # noqa: E402  (the sidecar's own containing-window rule)

DT = D.DT                   # 0.2 s: the cache step
H = D.HORIZON                # 30 steps = 6 s
EPS_M = 1e-3                 # a tangent shorter than 1 mm per 0.2 s interval holds the previous heading (plan_kin_t's convention)
#: the ONLY physical bound the executor applies: no executed 0.2 s step is longer than V_ENV_MS * 0.2 s (60 m/s = 216 km/h, beyond
#: any road speed in the corpus). It exists so that a diverging window stays finite instead of overflowing to NaN and corrupting a
#: whole run; it is COUNTED (``WindowState.clipped``) and reported, never silent. A sane plan never reaches it.
V_ENV_MS = 60.0
#: the allowlist: the model is handed EXACTLY the keys forward_infer reads. ``act`` and ``pose`` are in INFERENCE_INPUT_KEYS but
#: forward_infer never reads them (``act`` only inside ``loss``); dropping them makes a future read a loud KeyError.
NO_READ = frozenset({"act", "pose"})
MODEL_KEYS = frozenset(M.INFERENCE_INPUT_KEYS) - NO_READ
NAV_MODES = ("deadreckon", "logged")
SCENE_MODES = ("logged", "frozen")
LIMIT_MODES = ("logged", "off", "causal", "causal_all")
#: PREREG_V8F_T1KIN_CAUSALBUCKET: ``causal`` reads the max MEASURED ego speed over the last CAUSAL_STEPS + 1 rows [j - 20, j] (4 s; logged
#: up to t0, own executed after), ``causal_all`` over [0, j]; both snapped by ``speed_max_bin``. Rows after j are NaN by WindowState's
#: construction, and the slice ends at j anyway.
CAUSAL_STEPS = 20
CAUSAL_STATS = {}            # per run: ticks, ticks with no finite past, agreement with the logged bucket, 4 x 4 confusion (logged x causal)
SCHEDULES = ("periodic", "fresh")
TIER_DOCTRINE = ("EVAL_DOCTRINE.md, PI amendment 2026-09-02: T0 and T1 are both OPEN LOOP; T2 is the only closed loop and is not "
                 "provisioned. T1 here = the ego state is the model's OWN executed motion; the camera stream is logged. "
                 "It supports open-loop trajectory-quality claims only, never 'drives'.")


# ------------------------------------------------------------------------------------------------ geometry
def plan_to_world(plan, pose) -> np.ndarray:
    """``plan`` [N, 2] in the ego frame at ``pose`` = (x, y, yaw, ...) -> world [N, 2]."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    plan = np.asarray(plan, np.float64)
    return np.stack([pose[0] + c * plan[:, 0] - s * plan[:, 1], pose[1] + s * plan[:, 0] + c * plan[:, 1]], 1)


def hermite_frames(k_prev, knots, m: int, psi0: float):
    """Fill the 10 Hz raw frames of ``m`` executed cache steps from the plan's 0.2 s knots (DESIGN section 3, step 5).

    ``k_prev`` [2] = the previous executed knot (step j-1); ``knots`` [m+1.., 2] with ``knots[0]`` = the current knot (step j) and
    ``knots[i]`` = the plan's knot j+i. Cubic Hermite with Catmull-Rom tangents (a backward difference at the plan's last knot).
    Returns ``(xy [2m, 2], psi [2m], v [2m])`` for raw frames 2j+1 .. 2j+2m: the odd frame of interval i is the Hermite midpoint, the
    even frame the knot itself. Heading = tangent angle (held while the tangent is < ``EPS_M``); speed = |tangent| / 0.2 s."""
    K = np.vstack([np.asarray(k_prev, np.float64)[None], np.asarray(knots, np.float64)])      # K[i] = knot j+i-1
    nk = len(K)
    if nk < m + 2:
        raise ValueError(f"need at least m+1={m + 1} knots, got {nk - 1}")

    def tang(i):
        return 0.5 * (K[i + 1] - K[i - 1]) if i + 1 < nk else K[i] - K[i - 1]

    xy, psi, v = [], [], []
    cur = float(psi0)
    for i in range(1, m + 1):                                                    # interval K[i] -> K[i + 1]
        t0, t1 = tang(i), tang(i + 1)
        mid = 0.5 * (K[i] + K[i + 1]) + (t0 - t1) / 8.0
        dmid = 1.5 * (K[i + 1] - K[i]) - 0.25 * (t0 + t1)
        for pos, d in ((mid, dmid), (K[i + 1], t1)):
            nrm = float(np.hypot(d[0], d[1]))
            if nrm >= EPS_M:
                cur = math.atan2(d[1], d[0])
            xy.append(pos)
            psi.append(cur)
            v.append(nrm / DT)
    return np.asarray(xy), np.asarray(psi), np.asarray(v)


def envelope(plan):
    """Bound the plan's 0.2 s steps by ``V_ENV_MS * DT`` (direction kept) and zero a non-finite step. Returns ``(plan, hit)``.
    The origin is prepended, so step 0 is |plan[0]| itself. A plan inside the envelope is returned UNCHANGED (bit-identical)."""
    plan = np.asarray(plan, np.float64)
    steps = np.diff(np.vstack([np.zeros((1, 2)), plan]), axis=0)
    bad = ~np.isfinite(steps).all(1)
    nrm = np.linalg.norm(np.where(bad[:, None], 0.0, steps), axis=1)
    over = nrm > V_ENV_MS * DT
    if not (bad.any() or over.any()):
        return plan, False
    steps = np.where(bad[:, None], 0.0, steps)
    steps = steps * np.where(over, V_ENV_MS * DT / np.maximum(nrm, 1e-12), 1.0)[:, None]
    return np.cumsum(steps, axis=0), True


def exec_in_ego(track, t0: int) -> np.ndarray:
    """The executed cache-step knots t0+1 .. t0+30 in the t0 ego frame, by the SAME function that builds the GT (``gt_waypoints``)."""
    return fk.gt_waypoints(np.asarray(track, np.float64)[0::2], np.array([t0]), np.arange(1, H + 1))[0]


def ages_for(step: int, schedule: str, k_t: int = 3, k_s: int = 6):
    """Sample-and-hold ages (cache steps) at ``step`` = steps since t0. periodic: T ticks every ``k_t`` steps, S every ``k_s``
    (defaults 3 / 6: inside the trained support max_age_t=2, max_age_s=6 and nested as the trainer requires a_s >= a_t)."""
    if schedule == "fresh":
        return 0, 0
    if k_s % k_t:
        raise ValueError("k_s must be a multiple of k_t (S ticks no later than T)")
    return step % k_t, step % k_s


# ------------------------------------------------------------------------------------------------ the per-window state
class WindowState:
    """One window's view of its clip, with the ego arrays logged UP TO t0 and own-executed after.

    The constructor copies ONLY rows <= t0 of the logged ego arrays and of the native track (the future rows are NaN), so this
    object cannot leak the logged ego after t0 whatever the item builder does. ``d`` is a ClipStore-shaped dict that
    ``V8FMiniWindows.item`` reads unchanged. The logged track is passed to ``commit_logged`` only by the CONTROL executor."""

    EGO = ("pose", "kin3", "p10f", "base", "a", "ch1", "v")

    def __init__(self, base: dict, track, nav_rows, ci: int, t0: int, nav_mode: str = "deadreckon"):
        if nav_mode not in NAV_MODES:
            raise ValueError(f"nav_mode must be one of {NAV_MODES}")
        T = len(base["pose"])
        if not bool(base["gt_ok"][t0]):
            raise ValueError("window has no complete 6 s future (gt_ok False)")
        self.ci, self.t0, self.T, self.j, self.nav_mode = int(ci), int(t0), T, int(t0), nav_mode
        tr = np.full(np.asarray(track).shape, np.nan, np.float64)
        tr[:2 * t0 + 1] = np.asarray(track, np.float64)[:2 * t0 + 1]
        self.track = tr
        d = dict(base)                                                   # shallow: feats / gt / s_h stay shared and logged
        for k in self.EGO:
            arr = np.array(base[k], copy=True)
            arr[t0 + 1:] = np.nan
            d[k] = arr
        self.nav = np.array(nav_rows, np.float32, copy=True)
        self.nav0 = self.nav[t0].copy()
        if nav_mode == "deadreckon":
            self.nav[t0 + 1:] = np.nan
        self.arc = {t0: 0.0}
        self.d = d
        self.scene0 = None
        self.clipped = 0                  # ticks whose plan had a step beyond the envelope (or a non-finite value)

    # -- committing executed motion ------------------------------------------------------------------------------
    def commit_plan(self, plan_ego, m: int):
        """Execute the first ``m`` knots of ``plan_ego`` [30, 2] (ego frame at NOW), perfect tracking."""
        j = self.j
        pose = self.d["pose"][j]
        plan_ego, hit = envelope(plan_ego)
        self.clipped += int(hit)
        W = plan_to_world(plan_ego, pose)
        knots = np.vstack([pose[None, :2], W])
        xy, psi, v = hermite_frames(self.track[2 * (j - 1), :2], knots, m, float(pose[2]))
        sl = slice(2 * j + 1, 2 * (j + m) + 1)
        self.track[sl, 0:2], self.track[sl, 2], self.track[sl, 3] = xy, psi, v
        self._finish(j, j + m)

    def commit_logged(self, logged_track, m: int):
        """CONTROL executor: the 'own' motion IS the logged motion (DESIGN C1). Never used by a result."""
        j = self.j
        sl = slice(2 * j + 1, 2 * (j + m) + 1)
        self.track[sl] = np.asarray(logged_track, np.float64)[sl]
        self._finish(j, j + m)

    def _finish(self, j_old: int, j_new: int):
        sl = self.track[2 * j_old: 2 * j_new + 1]
        kin, om = D.yaw_kin_from_track(sl)                              # the ONE spelling, on the appended slice
        ch = D.ego_channels_native(sl)[0::2]
        d = self.d
        for s in range(1, j_new - j_old + 1):
            j = j_old + s
            d["kin3"][j] = kin[s]
            d["ch1"][j] = np.nan_to_num(om[s], nan=0.0)
            d["p10f"][j] = fk.kin_features(kin[s:s + 1])[0].astype(np.float32)
            d["base"][j] = D.kdx_base(kin[s:s + 1], D.KIN_UNITS["yaw"])[0]
            d["pose"][j] = self.track[2 * j]
            d["a"][j] = ch[s, 1]
            d["v"][j] = self.track[2 * j, 3]
            self.arc[j] = self.arc[j - 1] + float(np.linalg.norm(d["pose"][j, :2] - d["pose"][j - 1, :2]))
            if self.nav_mode == "deadreckon":
                row = self.nav0.copy()
                if row[4] > 0:                                           # route progress from OWN motion; token / valid held from t0
                    row[3] = np.clip(row[3] - self.arc[j] / 100.0, 0.0, 3.0)
                self.nav[j] = row
        self.j = j_new

    def executed_ego(self) -> np.ndarray:
        return exec_in_ego(self.track, self.t0)


class _OneClipStore:
    """A ClipStore stand-in that serves the dict of whichever window is being built."""

    def __init__(self):
        self.d = None

    def get(self, clip):
        return self.d


def build_item(ds, proxy, st: WindowState, tick: int, a_t: int, a_s: int, scene: str, limit: str, stats: dict | None = None) -> dict:
    proxy.d = st.d
    it = ds.item(st.ci, st.j, a_t, a_s)
    idc = np.clip(np.arange(st.j - ds.hmax + 1, st.j + 1), 0, st.T - 1)
    it["nav"] = st.nav[idc].astype(np.float32)
    if scene == "frozen":
        if tick == 0:
            st.scene0 = (it["feats"], it["frame_ok"])
        it["feats"], it["frame_ok"] = st.scene0
    if limit == "off":
        it["limit"], it["limit_ok"] = np.zeros(4, np.float32), False
    elif limit in ("causal", "causal_all"):
        logged = np.asarray(it["limit"], np.float32).copy() if bool(it["limit_ok"]) else None
        it["limit"], it["limit_ok"] = causal_bucket(st.d["v"], st.j, CAUSAL_STEPS if limit == "causal" else None)
        if stats is not None:
            stats["ticks"] = stats.get("ticks", 0) + 1
            if not it["limit_ok"]:
                stats["no_finite_past"] = stats.get("no_finite_past", 0) + 1
            elif logged is not None:
                cm = stats.setdefault("confusion_logged_x_causal", [[0] * 4 for _ in range(4)])
                cm[int(logged.argmax())][int(it["limit"].argmax())] += 1
    return it


def causal_bucket(v, j: int, steps: int | None):
    """The PREREG_V8F_T1KIN_CAUSALBUCKET bucket at step ``j``: ``speed_max_bin`` of the max finite speed over rows [j - steps, j] (or
    [0, j] when ``steps`` is None). Reads NOTHING after row j. No finite row -> (zeros, False), the ``--limit off`` format."""
    lo = 0 if steps is None else max(0, j - steps)
    w = np.asarray(v[lo:j + 1], np.float64)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return np.zeros(4, np.float32), False
    i, _ = speed_max_bin(float(max(w.max(), 0.0)))
    oh = np.zeros(4, np.float32)
    oh[i] = 1.0
    return oh, True


# ------------------------------------------------------------------------------------------------ planners
def model_planner(model, device):
    import torch

    def plan(batch):
        with torch.no_grad():
            out = model.forward_infer(batch)
        plan.keys_read = set(out["_keys_read"])
        return out["traj30"].float().cpu().numpy()
    plan.keys_read = set()
    return plan


def floor_planner(model, device):
    """The P10 floor driven the same way: the prior alone, the very call ``run_o`` makes (``v8f_mini.py:579``)."""
    import torch

    def plan(batch):
        with torch.no_grad():
            return model.prior(batch["p10f"][:, -1].float(), batch["base"].float()).float().cpu().numpy()
    return plan


# ------------------------------------------------------------------------------------------------ the rollout
def rollout(planner, states, ds, proxy, *, m: int = 1, schedule: str = "periodic", k_t: int = 3, k_s: int = 6,
            scene: str = "logged", limit: str = "logged", device="cpu", executor: str = "plan", logged_tracks=None, hook=None):
    """Roll every window of ``states`` for ``H // m`` ticks. Returns ``plan0`` [B, 30, 2] (the tick-0 plans) and the list of
    per-window executed paths in the t0 ego frame. ``hook(n, batch, plan, states)`` is the test seam."""
    import torch
    if H % m:
        raise ValueError(f"--replan-steps {m} must divide {H}")
    plan0 = None
    for n in range(H // m):
        step = n * m
        a_t, a_s = ages_for(step, schedule, k_t, k_s)
        stats = CAUSAL_STATS if hasattr(planner, "keys_read") else None          # the MODEL's ticks only (the floor reads no bucket)
        items = [build_item(ds, proxy, st, n, a_t, a_s, scene, limit, stats) for st in states]
        full = D.collate(items)
        batch = {k: full[k].to(device) for k in MODEL_KEYS}
        plan = planner(batch)
        if plan0 is None:
            plan0 = plan.copy()
        if hook is not None:
            hook(n, batch, plan, states)
        for st, pl in zip(states, plan):
            if executor == "logged":
                st.commit_logged(logged_tracks[st.ci], m)
            else:
                st.commit_plan(pl.astype(np.float64), m)
    return plan0, [st.executed_ego() for st in states]


def make_states(store, tracks, J, wins, nav_mode):
    out = []
    for ci, t0, _, _ in wins:
        clip = str(J["clip"][ci])
        base = store.get(clip)
        o, n = int(J["off"][ci]), int(J["n"][ci])
        out.append(WindowState(base, tracks[clip], J["nav"][o:o + n], ci, t0, nav_mode))
    return out


# ------------------------------------------------------------------------------------------------ scoring
def load_t1_eval():
    path = os.path.join(REPO, "taniteval", "tools", "t1_eval.py")
    spec = importlib.util.spec_from_file_location("t1_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_dumps(out_dir, wins, arms: dict, g, v0):
    """One ``ep*.npz`` per episode in ``t1_eval``'s schema; ``arms`` = {name: [N, 30, 2]}."""
    os.makedirs(out_dir, exist_ok=True)
    cis = np.array([w[0] for w in wins])
    ts = np.array([w[1] for w in wins])
    files = []
    for ci in dict.fromkeys(cis.tolist()):
        m = cis == ci
        f = os.path.join(out_dir, f"ep{ci:04d}.npz")
        np.savez(f, g=g[m], ws=ts[m], eid=cis[m], clip_index=cis[m], v0=v0[m], **{k: v[m] for k, v in arms.items()})
        files.append(f)
    return files


def analyze_dumps(files, header: dict, n_boot: int):
    t1 = load_t1_eval()
    res = t1.analyze(files, tiers={"fl": "T1", "ha0": "T1"}, n_boot=n_boot, dt=DT,
                     paired=[("ol", "cl", "paired_t1_minus_open"), ("fl", "cl", "paired_model_minus_floor_t1")])
    res["_tier_doctrine"] = TIER_DOCTRINE                                # t1_eval still prints the pre-amendment 'action-closed loop'
    res["t1v8f"] = header
    return res


# ------------------------------------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--tracks", required=True, help="native_tracks.pt (yaw mode needs the 10 Hz track)")
    p.add_argument("--eval-cache", default="C:/Users/Admin/refav1_hj_cache")
    p.add_argument("--eval-grid", required=True)
    p.add_argument("--eval-join", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--replan-steps", type=int, default=1, help="m: cache steps (0.2 s) executed per plan; must divide 30. m=30 == the one-shot open loop")
    p.add_argument("--schedule", choices=SCHEDULES, default="periodic")
    p.add_argument("--nav", choices=NAV_MODES, default="deadreckon")
    p.add_argument("--scene", choices=SCENE_MODES, default="logged")
    p.add_argument("--limit", choices=LIMIT_MODES, default="logged")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lru", type=int, default=32)
    p.add_argument("--max-windows", type=int, default=0, help="smoke: evenly-spaced subset of the grid (0 = all)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-analyze", action="store_true")
    p = sub.add_parser("analyze")
    p.add_argument("--dump-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    return ap


def cmd_run(a):
    import torch
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    if cfg.kin_mode != "yaw":
        raise SystemExit("REFUSED: T1-kin rebuilds the ego state from the executed pose track; a --kin-mode steer checkpoint reads the "
                         "recorded steering angle (PI ruling D-PI-YAWRATE-1) and has no own-motion equivalent.")
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    dev = torch.device(a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).eval()
    tracks = D.load_native_tracks(a.tracks)
    J = D.load_join(a.eval_join)
    store = D.ClipStore(a.eval_cache, "eval", lru=a.lru, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(a.eval_grid, J)
    if a.max_windows and a.max_windows < len(wins):
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, a.max_windows).round().astype(int)]
    proxy = _OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                          window_start=cfg.window_start)     # PREREG_V8F_HIST Amendment 1: the held-age clamp origin is the CHECKPOINT's (-1 = absent = hist - 1)
    kw = dict(m=a.replan_steps, schedule=a.schedule, scene=a.scene, limit=a.limit, device=dev)
    cl, ol, fl, ha0, g, v0, keys = [], [], [], [], [], [], set()
    n_clip = {"cl": 0, "fl": 0}
    t_start = time.time()
    for b0 in range(0, len(wins), a.batch):
        chunk = wins[b0:b0 + a.batch]
        mp = model_planner(model, dev)
        st_m = make_states(store, tracks, J, chunk, a.nav)
        plan0, exe = rollout(mp, st_m, ds, proxy, **kw)
        keys |= mp.keys_read
        st_f = make_states(store, tracks, J, chunk, a.nav)
        _, exe_f = rollout(floor_planner(model, dev), st_f, ds, proxy, **kw)
        n_clip["cl"] += sum(s.clipped > 0 for s in st_m)
        n_clip["fl"] += sum(s.clipped > 0 for s in st_f)
        cl += exe
        fl += exe_f
        ol.append(plan0)
        for ci, t0, _, _ in chunk:
            d = store.get(str(J["clip"][ci]))
            g.append(d["gt"][t0])
            v0.append(d["kin3"][t0][0])                                  # the ego speed AT t0 (logged history, allowed)
        print(f"[{b0 + len(chunk)}/{len(wins)}] {time.time() - t_start:.0f}s", flush=True)
    v0 = np.asarray(v0, np.float64)
    arms = {"cl": np.asarray(cl), "ol": np.concatenate(ol).astype(np.float64), "fl": np.asarray(fl),
            "ha0": fk.ha0_paths(v0, DT, H)}
    gt = np.asarray(g, np.float64)
    files = write_dumps(os.path.join(a.out_dir, f"dump_{a.tag}"), wins, arms, gt, v0)
    header = {"tag": a.tag, "ckpt": a.ckpt, "kin_mode": cfg.kin_mode, "tier": "T1 (self-action OPEN loop; ego state = own executed motion)",
              "replan_steps": a.replan_steps, "replan_hz": 5.0 / a.replan_steps, "schedule": a.schedule, "scene": a.scene, "nav": a.nav,
              "limit": a.limit, "n_windows": len(wins), "model_keys_read": sorted(keys),
              "windows_with_envelope_clip": n_clip, "envelope_ms": V_ENV_MS,
              "strict": {"kinematics": True, "route": a.nav == "deadreckon", "camera": a.scene == "frozen", "speed_bucket": a.limit != "logged"},
              "open_channels": ["camera stream (logged frames after t0 carry the human's ego-motion)" if a.scene == "logged" else "",
                                "route countdown from the logged human position" if a.nav == "logged" else "",
                                "speed bucket from the human's realised future max speed (D-SPEEDBAND-EGO-ECHO-1)" if a.limit == "logged" else ""],
              "acts": {k: fk.acts_check(v, DT) for k, v in arms.items() if k in ("cl", "fl", "ol")}}
    header["open_channels"] = [c for c in header["open_channels"] if c]
    if a.limit in ("causal", "causal_all"):                              # only then: --limit logged / off outputs stay byte-identical (C-L1)
        header["causal_bucket"] = {"rule": "speed_max_bin(max finite measured ego speed over rows "
                                           + (f"[j - {CAUSAL_STEPS}, j])" if a.limit == "causal" else "[0, j])"),
                                   "stats_model_ticks": CAUSAL_STATS}
    os.makedirs(a.out_dir, exist_ok=True)
    if a.no_analyze:
        json.dump(header, open(os.path.join(a.out_dir, f"t1v8f_{a.tag}.json"), "w"), indent=1)
        return 0
    res = analyze_dumps(files, header, a.n_boot)
    out = os.path.join(a.out_dir, f"t1v8f_{a.tag}.json")
    json.dump(res, open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", out)
    return 0


def cmd_analyze(a):
    files = sorted(os.path.join(a.dump_dir, f) for f in os.listdir(a.dump_dir) if f.startswith("ep") and f.endswith(".npz"))
    res = analyze_dumps(files, {"note": "re-scored from existing dumps; run header not available"}, a.n_boot)
    json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return 0


def main(argv=None):
    a = build_parser().parse_args(argv)
    return {"run": cmd_run, "analyze": cmd_analyze}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
