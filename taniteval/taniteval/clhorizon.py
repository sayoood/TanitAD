"""taniteval.clhorizon — the HORIZON-CAPABLE closed loop, and the v4 plan step.

WHY THIS MODULE EXISTS (the stranding the operating standard forbids)
--------------------------------------------------------------------
``Project Steering/Gates/flagship-v4-30k.card.json`` registers the gate
co-primary ``corridor_departure_rate`` at ``horizon_K = 185`` (18.5 s) on the
**closed-loop** surface. Until 2026-07-26 **nothing in ``taniteval`` could emit
it**:

* ``taniteval/closedloop.py`` fixed ``K_MAX = max(WP_STEPS) = 20`` — the *blind*
  horizon the co-primary exists to replace — and its ``run_and_save`` refused any
  arm without ``traj_capable`` **and** ``model.tactical_policy``. A v4 checkpoint
  has **neither**: it plans through ``FlagshipV4Head`` (dense-20 factorised
  LAT x LON x DIST selection).
* ``e1a_horizon.py``, which produced the card's REF-C reference at exactly
  K=185, is REF-C-specific (``RefCModel`` / ``--refc-ckpt``).

So the **registered co-primary was reachable only through a one-off driver in
``incoming/``** (``2026-07-26-v4-30k-gate/coprimary/v4_corridor_cl.py``). This
module is that driver's rollout and plan step, ported into the package.

WHAT IS PORTED, AND WHAT IS NOT
-------------------------------
:func:`corridor_rollout` is ``v4_corridor_cl.rollout`` — itself
``e1a_horizon.rollout``, itself ``lowood_lanekeep.cl_realfootage``, itself a
C6-clean minimal edit of ``taniteval.closedloop`` — with the per-step plan call
lifted out into an injected ``planner``. Loop body, window/stratum bookkeeping
and the reference-index geometry are reproduced verbatim so the ported number
lands on the SAME surface as the REF-C reference the card quotes.
``taniteval/tests/test_clhorizon.py::test_port_is_tensor_identical_to_the_driver``
asserts **bit-identical** ``lat`` / ``yaw`` / ``ade2s`` / ``de_fixed`` tensors
against the ``incoming/`` driver on the same inputs — the port cannot silently
change the measured behaviour.

:class:`V4Planner` is the driver's plan step: ``world.encode_window`` ->
``goal_modes.resolve_goal`` -> ``head(st, v0, lambda_plan=1.0, **goal_kw)``,
which is byte-for-byte the forward pass ``eval_flagship_v4.collect_planner``
runs, with the 0.5 s lookahead waypoint ``traj[:, LOOKAHEAD_STEP - 1]`` fed to
the same pure-pursuit controller.

NOT ported: the loaders (``eval_flagship_v4.load_v4_from_ck``) and the CLI. Those
stay in ``stack/scripts``, where the rest of the v4 harness lives; this module
takes already-built objects so it has no dependency on the stack layout and can
be tested with a stub planner on CPU.

THE OOD ACCOUNTING IS THE FIXED ONE. :func:`emit` uses :mod:`taniteval.ood`,
which implements **E1a's full disjunction** (ratio > ~1.5x **OR** steps leave the
measured envelope) instead of the ratio alone. The driver's own block emitted
"within the measured envelope on average" at K=185 while 54.63 % of steps were
outside it; that string cannot be produced here.

REPRODUCTION (MEASURED, ``INSTRUMENT_FIXES.md`` defect 3): on the committed
per-window tensors from the 30 k gate this module reproduces
**v4 K=185 overall 0.6388 / junction 0.8432** and **REF-C base 0.5833**, exactly.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from taniteval import ci as _ci
from taniteval import corridor as _corr
from taniteval import ood as _ood

__all__ = ["W", "DT", "WHEELBASE", "LOOKAHEAD_STEP", "WP_STEPS", "WP_IDX",
           "HORIZON_CEILING_K", "sampling_homography", "warp_batch",
           "wp_to_control", "wrap_angle", "V4Planner", "CallablePlanner",
           "corridor_rollout", "emit", "corridor_from_perwindow",
           "horizon_windows"]

# --- loop constants: e1a_horizon.py / lowood_lanekeep.py / closedloop.py ---- #
W = 8                     # predictor window
DT = 0.1                  # MEASURED — 10 Hz on every trainer and eval path
WHEELBASE = 2.7           # kinematic bicycle (kinematic.py)
LOOKAHEAD_STEP = 5        # pure-pursuit target = the 0.5 s waypoint
LD2_FLOOR = 0.25
STEER_CLAMP = 0.05
ACCEL_CLAMP = 3.0
SPEED_TC = 0.5
WP_STEPS = (5, 10, 15, 20)
WP_IDX = [k - 1 for k in WP_STEPS]
F_EFF = 266.0             # PhysicalAI front-wide effective focal length
CXY = 128.0
CAM_HEIGHT_M = 1.5
CAM_PITCH_DEG = 0.0
# Structural ceiling: a window exists at K only if T - W - K >= 1 and PhysicalAI
# clips are 190-199 frames. K=200 is IMPOSSIBLE on this corpus, not merely
# unmeasured. Same constant as run_gate.HORIZON_CEILING_K.
HORIZON_CEILING_K = 190
# Horizons at which a fixed-elapsed-time displacement error is recorded.
FIXED_STEPS = (20, 40, 80, 120, 160, 185, 190)


# =========================================================================== #
# geometry — e1a_horizon.py VERBATIM                                          #
# =========================================================================== #
def sampling_homography(dlat_m, dyaw_deg, h_cam=CAM_HEIGHT_M,
                        pitch_deg=CAM_PITCH_DEG, f=F_EFF, c=CXY):
    """Ground-plane homography that re-renders a real frame as if the camera had
    been displaced ``dlat_m`` laterally and rotated ``dyaw_deg``.

    ⚠️ Its fidelity was MEASURED only to ``|dlat| <= 3.0 m`` / ``|dyaw| <= 12
    deg`` (P1, ``lowood_flagship_ci.json``, on the flagship **v1** arm). Beyond
    that the warp degrades gracefully rather than faithfully, and the OOD ratio
    that would flag it SATURATES — see :mod:`taniteval.ood`."""
    Kk = torch.tensor([[f, 0, c], [0, f, c], [0, 0, 1.0]], dtype=torch.float64)
    Ki = torch.linalg.inv(Kk)
    p = math.radians(pitch_deg)
    n = torch.tensor([0.0, math.cos(p), math.sin(p)], dtype=torch.float64)
    d = float(h_cam)
    psi = math.radians(dyaw_deg)
    Ry = torch.tensor([[math.cos(-psi), 0, math.sin(-psi)],
                       [0, 1.0, 0],
                       [-math.sin(-psi), 0, math.cos(-psi)]], dtype=torch.float64)
    Cc = torch.tensor([dlat_m, 0.0, 0.0], dtype=torch.float64)
    t = -(Ry @ Cc)
    H_1to2 = Kk @ (Ry + torch.outer(t, n) / d) @ Ki
    return torch.linalg.inv(H_1to2)


def warp_batch(fw, Hs):
    """Apply one homography per batch element to a whole ``[b, W, C, H, W']``
    frame window. e1a_horizon.py VERBATIM."""
    b, Wn, C, Hh, Ww = fw.shape
    dev = fw.device
    ys, xs = torch.meshgrid(torch.arange(Hh, dtype=torch.float64, device=dev),
                            torch.arange(Ww, dtype=torch.float64, device=dev),
                            indexing="ij")
    ones = torch.ones_like(xs)
    P = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3).T
    src = Hs.to(dev).to(torch.float64) @ P
    su = (src[:, 0] / src[:, 2]).reshape(b, Hh, Ww)
    sv = (src[:, 1] / src[:, 2]).reshape(b, Hh, Ww)
    gx = 2.0 * su / (Ww - 1) - 1.0
    gy = 2.0 * sv / (Hh - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    grid = grid[:, None].expand(-1, Wn, -1, -1, -1).reshape(b * Wn, Hh, Ww, 2).float()
    out = F.grid_sample(fw.reshape(b * Wn, C, Hh, Ww), grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out.reshape(b, Wn, C, Hh, Ww)


def wp_to_control(w_look, v):
    """closedloop.py VERBATIM — pure pursuit + first-order speed tracking."""
    x, y = w_look[:, 0], w_look[:, 1]
    ld2 = (x * x + y * y).clamp_min(LD2_FLOOR)
    kappa = 2.0 * y / ld2
    steer = torch.atan(WHEELBASE * kappa).clamp(-STEER_CLAMP, STEER_CLAMP)
    v_target = x / (LOOKAHEAD_STEP * DT)
    accel = ((v_target - v) / SPEED_TC).clamp(-ACCEL_CLAMP, ACCEL_CLAMP)
    return steer, accel


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def horizon_windows(episode_T, K, window=W, stride=8):
    """How many windows an episode of ``T`` frames yields at horizon ``K``.

    ``starts = range(0, T - W - K, stride)`` — the structural reason n COLLAPSES
    at long horizon: on this corpus (T = 198-205) K=185 leaves ~1 window per
    episode. **n must be quoted with any number at this horizon.**"""
    return len(range(0, max(0, int(episode_T) - int(window) - int(K)), int(stride)))


# =========================================================================== #
# the plan step — the ONE thing that differs between arms                     #
# =========================================================================== #
class CallablePlanner:
    """Adapter for any ``f(frames, v0, goal) -> traj [b, >=LOOKAHEAD_STEP, 2]``.

    The rollout only ever needs a trajectory in the ego frame of the window's
    last observed frame; every arm-specific detail lives behind this."""

    def __init__(self, fn, name="callable"):
        self.fn, self.name = fn, name
        self.rec: dict = {}

    def traj(self, fw, v0, goal_batch):
        return self.fn(fw, v0, goal_batch)


class V4Planner:
    """The v4 plan step, ported from ``v4_corridor_cl.V4Planner``.

    ``world.encode_window`` -> ``goal_modes.resolve_goal`` ->
    ``head(st, v0, lambda_plan=1.0, **goal_kw)``. That is byte-for-byte the
    forward pass ``eval_flagship_v4.collect_planner`` runs, so a closed-loop
    number from this planner cannot drift from the open-loop gate primary.

    ``goal_modes`` is injected rather than imported: it lives in
    ``stack/scripts`` and importing it here would tie ``taniteval`` to the stack
    layout (and break this module's CPU tests). Callers pass the module.
    """

    def __init__(self, world, head, goal_head, goal_mode, goal_modes,
                 allow_fallback=False, lambda_plan=1.0):
        self.world, self.head = world, head
        self.goal_head, self.goal_mode = goal_head, goal_mode
        self.goal_modes = goal_modes
        self.allow_fallback = allow_fallback
        self.lambda_plan = float(lambda_plan)
        self.rec: dict = {}

    @torch.no_grad()
    def traj(self, fw, v0, goal_batch):
        """``fw [b, W, C, H, W']`` float in [0, 1]; ``v0 [b]`` -> ``traj [b, 20, 2]``."""
        st = self.world.encode_window(fw)
        goal_kw, rec = self.goal_modes.resolve_goal(
            self.goal_mode, head=self.head, batch=goal_batch, v0=v0, states=st,
            goal_head=self.goal_head, allow_fallback=self.allow_fallback)
        self.rec = rec
        return self.head(st, v0, lambda_plan=self.lambda_plan,
                         **goal_kw)["traj"].cpu()


# =========================================================================== #
# the HORIZON-CAPABLE closed loop                                             #
# =========================================================================== #
@torch.no_grad()
def corridor_rollout(planner, episodes, goals, device, K, *, stride=8, batch=16,
                     frames_of=None, verbose=False, progress=None):
    """Real-footage-in-the-loop rollout to an ARBITRARY horizon ``K``.

    Ported from ``v4_corridor_cl.rollout`` (= ``e1a_horizon.rollout``), with the
    plan call injected. At every 0.1 s tick the loop
    (a) finds the nearest logged reference pose ``mstar`` to the current ego
        pose and computes the lateral / heading deviation from it,
    (b) re-renders the real frame window at that deviation
        (:func:`sampling_homography` + :func:`warp_batch`),
    (c) asks ``planner.traj`` for a trajectory and takes the 0.5 s waypoint,
    (d) drives a kinematic bicycle under the resulting (steer, accel).

    ``goals`` is any object with ``get(episode_index, last_indices, device)``;
    pass ``None`` for arms that need no goal (it is then never called).
    ``frames_of(ep, a, b)`` returns the ``[W, C, H, W']`` float window; the
    default handles ``ep.frames`` uint8 or float.

    Returns the per-window record (``lat``/``yaw`` ``[n, K]``, ``ade2s``,
    ``hd2s``, ``hdK``, ``speed``, ``eid``, ``t0``, ``epi``, ``de_fixed``) or
    ``None`` when no episode is long enough — which is a NOT-MEASURED, never a
    pass. **Nothing here caps K**: the only limit is the structural one,
    ``T - W - K >= 1``.
    """
    K = int(K)
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if K > HORIZON_CEILING_K:
        raise ValueError(
            f"K={K} exceeds the structural ceiling K={HORIZON_CEILING_K} on this "
            f"corpus (clips are 190-199 frames; a window needs T - W - K >= 1). "
            f"No episode can produce a window at that horizon.")
    frames_of = frames_of or _default_frames

    rows = {k: [] for k in ("ade2s", "hd2s", "hdK", "speed", "eid", "t0", "epi")}
    lat_all, yaw_all, de_fixed = [], [], []
    fixed_steps = [s for s in FIXED_STEPS if s <= K]
    n_steps_done = 0
    for ep_i, ep in enumerate(episodes):
        poses = torch.as_tensor(ep.poses, dtype=torch.float32)
        T = int(poses.shape[0])
        starts = list(range(0, T - W - K, stride))
        if not starts:
            continue
        for bi in range(0, len(starts), batch):
            ch = starts[bi:bi + batch]
            b = len(ch)
            t0 = torch.tensor(ch)
            last = t0 + W - 1
            idx = last[:, None] + torch.arange(0, K + 1)[None]
            Pxy = poses[idx][..., :2]
            Pyaw = poses[idx][..., 2]
            oyaw = poses[last, 2]
            oxy = poses[last, :2]
            ex = poses[last, 0].clone(); ey = poses[last, 1].clone()
            eyaw = poses[last, 2].clone(); ev = poses[last, 3].clone()
            ego_ego = torch.zeros(b, K, 2)
            lat_t = torch.zeros(b, K); yaw_t = torch.zeros(b, K)
            ar = torch.arange(b)
            for k in range(K):
                d = (Pxy - torch.stack([ex, ey], -1)[:, None]).norm(dim=-1)
                mstar = d.argmin(dim=1)
                pref = Pxy[ar, mstar]; yref = Pyaw[ar, mstar]
                dx = ex - pref[:, 0]; dy = ey - pref[:, 1]
                dlat = -torch.sin(yref) * dx + torch.cos(yref) * dy
                dpsi = wrap_angle(eyaw - yref)
                lat_t[:, k] = dlat; yaw_t[:, k] = dpsi
                wins = [frames_of(ep, int(t0[i] + mstar[i]),
                                  int(t0[i] + mstar[i]) + W) for i in range(b)]
                fw = torch.stack(wins).to(device)
                Hs = torch.stack([
                    sampling_homography(float(dlat[i]),
                                        float(math.degrees(dpsi[i])))
                    for i in range(b)])
                fw = warp_batch(fw, Hs)
                # ---- the injected plan step (the ONLY arm-specific line) ----
                g = (goals.get(ep_i, (t0 + mstar + W - 1).numpy(), device)
                     if goals is not None else None)
                w_look = planner.traj(fw, ev.to(device), g)[:, LOOKAHEAD_STEP - 1].cpu()
                # -------------------------------------------------------------
                steer, accel = wp_to_control(w_look, ev)
                ex = ex + ev * torch.cos(eyaw) * DT
                ey = ey + ev * torch.sin(eyaw) * DT
                eyaw = eyaw + ev / WHEELBASE * torch.tan(steer) * DT
                ev = (ev + accel * DT).clamp_min(0.0)
                wdx = ex - oxy[:, 0]; wdy = ey - oxy[:, 1]
                ego_ego[:, k, 0] = (torch.cos(oyaw) * wdx + torch.sin(oyaw) * wdy)
                ego_ego[:, k, 1] = (-torch.sin(oyaw) * wdx + torch.cos(oyaw) * wdy)
                n_steps_done += 1
            gt2 = _gt_ego_waypoints(poses, last, WP_STEPS)
            rows["ade2s"].append(
                torch.linalg.norm(ego_ego[:, WP_IDX] - gt2, dim=-1).mean(1))
            gtf = _gt_ego_waypoints(poses, last, tuple(fixed_steps))
            de_fixed.append(torch.linalg.norm(
                ego_ego[:, [s - 1 for s in fixed_steps]] - gtf, dim=-1))
            lat_abs = lat_t.abs(); yaw_abs_deg = yaw_t.abs() * 180 / math.pi
            lat_all.append(lat_abs); yaw_all.append(yaw_abs_deg)
            rows["hd2s"].append(
                wrap_angle(poses[last + 20, 2] - poses[last, 2]).abs() * 180 / math.pi)
            rows["hdK"].append(
                wrap_angle(poses[last + K, 2] - poses[last, 2]).abs() * 180 / math.pi)
            rows["speed"].append(poses[last, 3])
            rows["eid"] += [str(ep_i)] * b
            rows["t0"].append(t0.clone())
            rows["epi"].append(torch.full((b,), ep_i))
        if verbose:
            print(f"    [cl] K={K} ep {ep_i + 1}/{len(episodes)} "
                  f"(rollout steps so far {n_steps_done})", flush=True)
        if progress is not None:
            progress(ep_i, len(episodes), n_steps_done)
    if not rows["eid"]:
        return None
    out = {k: (v if k == "eid" else torch.cat(v)) for k, v in rows.items()}
    out["lat"] = torch.cat(lat_all)
    out["yaw"] = torch.cat(yaw_all)
    out["de_fixed"] = torch.cat(de_fixed)
    out["fixed_steps"] = fixed_steps
    out["_rollout_steps_executed"] = n_steps_done
    return out


def _default_frames(ep, a, b):
    fr = ep.frames[a:b]
    fr = torch.as_tensor(fr)
    return fr.float().div(255.0) if fr.dtype == torch.uint8 else fr.float()


def _gt_ego_waypoints(poses, last, wp_steps):
    """``driving_diagnostic.gt_ego_waypoints``, inlined so this module has no
    stack dependency. Logged future poses expressed in the ego frame of
    ``last``; identical arithmetic, asserted against the driver in the tests."""
    last = torch.as_tensor(last)
    ox, oy, oyaw = poses[last, 0], poses[last, 1], poses[last, 2]
    out = []
    for s in wp_steps:
        px, py = poses[last + s, 0], poses[last + s, 1]
        dx, dy = px - ox, py - oy
        out.append(torch.stack([torch.cos(oyaw) * dx + torch.sin(oyaw) * dy,
                                -torch.sin(oyaw) * dx + torch.cos(oyaw) * dy],
                               dim=-1))
    return torch.stack(out, dim=1)


# =========================================================================== #
# aggregation — corridor.stratified + the FIXED OOD guard                     #
# =========================================================================== #
def emit(pw, K, *, ood_map=None, thresholds=_corr.CORRIDOR_GRID_M,
         primary=_corr.CORRIDOR_HALFWIDTH_M, junction_deg=_corr.JUNCTION_DEG,
         surface="closed_loop", n_boot=None, seed=0) -> dict:
    """The registered co-primary emitter + the honest OOD block.

    ``taniteval.corridor.stratified`` is THE registered emitter and is used
    unchanged. What differs from the ``incoming/`` driver is that the ``ood``
    node comes from :mod:`taniteval.ood`, which tests **both** clauses of E1a's
    rule — so this cannot emit "within the measured envelope" while a majority of
    steps are outside it.

    ``ood_map`` is optional (it needs the external P1 envelope JSON, which is not
    a library dependency). Without it the envelope FRACTIONS are still emitted —
    they need only the envelope constants and are model-independent — and the
    ratio is simply absent rather than silently assumed benign.
    """
    n_boot = _ci.DEFAULT_N_BOOT if n_boot is None else int(n_boot)
    lat = pw["lat"].numpy() if torch.is_tensor(pw["lat"]) else np.asarray(pw["lat"])
    yaw = pw["yaw"].numpy() if torch.is_tensor(pw["yaw"]) else np.asarray(pw["yaw"])
    eid = list(pw["eid"])
    hd = _np(pw["hd2s"])
    spd = _np(pw["speed"])
    ade2s = _np(pw["ade2s"]) if pw.get("ade2s") is not None else None

    out = _corr.stratified(lat, eid, hd, spd, thresholds=tuple(thresholds),
                           primary=primary, junction_deg=junction_deg,
                           yaw_abs_deg=yaw, ade2s=ade2s,
                           n_boot=n_boot, seed=seed, surface=surface)
    junc = _corr.junction_mask(hd, junction_deg)
    long_ = (~junc) & (spd >= np.median(spd))
    strata = {"overall": np.ones(len(hd), bool), "junction": junc,
              "longitudinal": long_, "other": (~junc) & (~long_)}
    ood_node = {"_envelope": {"lat_max_m": _ood.ENV_LAT_MAX,
                              "yaw_max_deg": _ood.ENV_YAW_MAX,
                              "provenance": "P1 MEASURED (lowood_flagship_ci"
                                            ".json), on the flagship v1 arm"},
                "_rule": _ood.RULE, "_saturation": _ood.SATURATION_NOTE,
                "_ratio_available": ood_map is not None}
    for name, mask in strata.items():
        ix = np.flatnonzero(mask)
        if ood_map is not None:
            ood_node[name] = _ood.verdict(lat[ix], yaw[ix],
                                          [eid[i] for i in ix], ood_map, K,
                                          n_boot=n_boot, seed=seed, stratum=name)
        else:
            frac = _ood.envelope_fractions(lat[ix], yaw[ix])
            node = {
                "stratum": name, "horizon_K": int(K),
                "n_windows": int(len(ix)),
                "EXTRAPOLATION_frac_steps_lat_over_3m": frac["frac_steps_lat_over_3m"],
                "EXTRAPOLATION_frac_steps_yaw_over_12deg": frac["frac_steps_yaw_over_12deg"],
                "EXTRAPOLATION_frac_steps_any": frac["frac_steps_any"],
                "EXTRAPOLATION_frac_windows_any_step_out_of_envelope":
                    frac["frac_windows_any_step_out_of_envelope"],
                "ood_peak_ratio": None,
                "ratio_is_lower_bound": bool(
                    frac["frac_windows_any_step_out_of_envelope"] > 0.0),
                "EXTRAPOLATION_VERDICT": _ood._verdict_string(
                    False, frac["frac_windows_any_step_out_of_envelope"],
                    frac["frac_steps_any"]),
                "_note": ("no P1 envelope JSON supplied, so no OOD RATIO is "
                          "reported. The envelope FRACTIONS are model-"
                          "independent and are the clause that actually fires "
                          "at long horizon.")}
            ood_node[name] = _ood.assert_envelope_verdict_consistent(
                node, _path=f"ood[{name}]")
    out["ood"] = ood_node

    if pw.get("de_fixed") is not None and pw.get("fixed_steps"):
        dfx = _np(pw["de_fixed"])
        out["de_at_elapsed_s"] = {
            f"{s * DT:g}": _ci.episode_cluster_bootstrap(
                np.asarray(dfx[:, i], float), eid, n_boot=n_boot, seed=seed)
            for i, s in enumerate(pw["fixed_steps"])}
    if pw.get("_rollout_steps_executed") is not None:
        out["rollout_steps_executed"] = int(pw["_rollout_steps_executed"])
    out["rollout_advanced_K_steps"] = bool(lat.shape[1] == int(K))
    out["horizon_K"] = int(K)
    out["horizon_s"] = round(int(K) * DT, 2)
    out["n_windows_note"] = (
        "n COLLAPSES with K (starts = range(0, T - W - K, stride)). A corridor "
        "number without its K AND its n is not admissible.")
    return out


def corridor_from_perwindow(path_or_pw, K=None, **kw) -> dict:
    """:func:`emit` on a persisted per-window dump.

    This is the arithmetic-only path that did not exist before 2026-07-26: with
    the per-window tensors on disk, a corridor block can be recomputed — a
    different half-width, a different stratification, a corrected OOD verdict —
    **without touching a GPU**. Its absence is why all five closed-loop artifacts
    had to be re-driven on GPU when the OOD rule was fixed.
    """
    pw = (path_or_pw if isinstance(path_or_pw, dict)
          else torch.load(str(path_or_pw), weights_only=False))
    if K is None:
        K = int(pw["lat"].shape[1])
    return emit(pw, K, **kw)


def _np(x):
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


# =========================================================================== #
# runnable entry point — so the co-primary is not stranded behind a driver     #
# =========================================================================== #
def run_v4(ckpt, val_dir, *, K=185, goal_mode="oracle", device="cuda",
           episodes=40, stride=8, batch=16, anchors_dense=None,
           p1_envelope=None, out=None, stack_paths=(), verbose=True) -> dict:
    """Emit the registered gate co-primary for a **v4** checkpoint, end to end.

    The loaders (``eval_flagship_v4.load_v4_from_ck``), the goal labeler
    (``v4_labels``) and ``goal_modes`` live in ``stack/scripts`` and are imported
    LAZILY here: ``taniteval`` must stay importable without the stack layout (its
    whole test suite runs on CPU with no stack model). Pass ``stack_paths`` when
    they are not already on ``sys.path``.

    ``--goal-mode oracle`` re-mints the goal at EVERY rollout step at the
    reference index the model is actually observing (``t0 + mstar + W - 1``), by
    the same ``v4_labels.mint_window`` call the dataset makes — the closed-loop
    analogue of the open-loop oracle, NOT a route handed over once at t=0. That
    policy is recorded in the output (GATE_PROTOCOL 0.8).
    """
    import json
    import sys
    for p in stack_paths:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import goal_modes                                              # noqa: E402
    from eval_flagship_v4 import load_v4_from_ck                   # noqa: E402
    from taniteval import data as _data                            # noqa: E402
    from taniteval.ood import OODMap                               # noqa: E402

    L = load_v4_from_ck(ckpt, device, anchors_dense=anchors_dense)
    world, head = L["world"], L["head"]
    planner = V4Planner(world, head, L.get("goal_head"), goal_mode, goal_modes)
    eps = _data.load_frames(_data.list_val_episodes(val_dir, episodes))
    goals = _v4_goal_cache(eps, stack_paths)
    pw = corridor_rollout(planner, eps, goals, device, K, stride=stride,
                          batch=batch, verbose=verbose)
    if pw is None:
        return {"skipped": f"no episode yields a window at K={K} "
                           f"(needs T - {W} - {K} >= 1)"}
    res = emit(pw, K, ood_map=OODMap(p1_envelope) if p1_envelope else None)
    res["ckpt"] = str(ckpt)
    res["ckpt_step"] = L.get("step")
    res["goal_provenance"] = goal_mode.upper()
    res["goal_index_policy"] = ("re-minted at EVERY rollout step at the "
                               "reference index the model observes "
                               "(t0 + mstar + W - 1)")
    res["goal_labeler_refusals"] = getattr(goals, "n_fail", None)
    if out:
        from pathlib import Path
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=2, default=str),
                             encoding="utf-8")
        pwp = str(Path(out).with_suffix("")) + f"_perwindow_K{int(K)}.pt"
        torch.save({k: (v if k in ("eid", "fixed_steps",
                                   "_rollout_steps_executed") else v)
                    for k, v in pw.items()}, pwp)
        res["per_window_dump"] = pwp
    return res


def _v4_goal_cache(episodes, stack_paths=()):
    """The v4 oracle goal cache (``route`` / ``route_graded`` / ``vt_band``),
    minted by the SAME ``v4_labels.mint_window`` call ``FlagshipV4Dataset``
    makes, so the goal the closed loop feeds is the object the open-loop gate
    primary feeds. Ported from ``v4_corridor_cl.GoalCache``."""
    import sys
    for p in stack_paths:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import refb_labels as rl                                       # noqa: E402
    import v4_labels                                               # noqa: E402
    from tanitad.lake.vtarget import savgol                        # noqa: E402
    VT_DROPPED = 23

    class _GoalCache:
        def __init__(self, eps, min_lookahead=50, use_net_dyaw=False):
            self.eps, self.min_lookahead = eps, min_lookahead
            self.use_net_dyaw = use_net_dyaw
            self._c, self.n_fail, self.n_total = {}, 0, 0

        def _build(self, e_i):
            poses = torch.as_tensor(self.eps[e_i].poses, dtype=torch.float32)
            vs = savgol(poses[:, 3].numpy().astype(np.float64))
            T = int(poses.shape[0])
            route = np.full(T, int(rl.ROUTE_UNKNOWN), dtype=np.int64)
            graded = np.zeros(T, dtype=np.float32)
            band = np.full(T, VT_DROPPED, dtype=np.int64)
            for L in range(T):
                self.n_total += 1
                try:
                    w = v4_labels.mint_window(poses, L, v_smoothed=vs,
                                              min_lookahead=self.min_lookahead,
                                              use_net_dyaw=self.use_net_dyaw)
                except Exception:
                    self.n_fail += 1
                    continue
                route[L] = int(w["route"])
                graded[L] = float(w["route_graded"])
                band[L] = int(w["vt_band"])
            self._c[e_i] = (route, graded, band)
            return self._c[e_i]

        def get(self, e_i, last_ix, device):
            r, g, bd = self._c.get(e_i) or self._build(e_i)
            li = np.clip(np.asarray(last_ix, dtype=np.int64), 0, len(r) - 1)
            return {"route": torch.as_tensor(r[li], device=device),
                    "route_graded": torch.as_tensor(g[li], device=device),
                    "vt_band": torch.as_tensor(bd[li], device=device)}

    return _GoalCache(episodes)
