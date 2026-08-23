#!/usr/bin/env python3
"""v4_corridor_cl.py — the flagship-v4 CO-PRIMARY: ``corridor_departure_rate``
at K=185 (18.5 s), CLOSED LOOP, on a ``FlagshipV4Head`` checkpoint.

WHY THIS FILE EXISTS
--------------------
The gate card ``Project Steering/Gates/flagship-v4-30k.card.json`` registers a
co-primary at ``horizon_K = 185`` on the ``closed-loop`` surface. Nothing in the
program could emit it for a v4 checkpoint:

* ``taniteval/closedloop.py`` caps at ``K_MAX = 20`` AND its ``run_and_save``
  loads arms through ``taniteval.registry.MODELS`` + ``loaders.load`` and
  requires ``traj_capable`` + ``model.tactical_policy`` — a v4 checkpoint has
  neither (it plans through ``FlagshipV4Head``, dense-20 factorised
  LAT x LON x DIST selection). That loader path REJECTS a v4 ckpt.
* ``e1a_horizon.py`` produced the card's reference numbers at exactly K=185 but
  is REF-C-specific (``--refc-ckpt`` / ``RefCModel`` / ``refc_config``).

This file is ``e1a_horizon.py`` with **one thing changed**: the per-step plan
call. Loop body, window/stratum bookkeeping, OOD envelope accounting and the
common-start PAIRED design are reproduced from it rather than reinvented, so the
v4 number lands on the SAME surface as the REF-C reference the card quotes.

THE ONE CHANGE
--------------
E1a's step is ``model(fw, nav_cmd=None, v0=ev, steps=2)["traj"][:, 0]`` (RefC's
traj is at wp_steps 5/10/15/20, so ``[:, 0]`` is the 0.5 s lookahead waypoint).
The v4 step is::

    st       = world.encode_window(fw)                 # [b, W, S]
    goal_kw  = goal_modes.resolve_goal(mode, head=head, batch=goal, v0=ev,
                                       states=st, goal_head=goal_head)
    traj     = head(st, ev, lambda_plan=1.0, **goal_kw)["traj"]   # [b, 20, 2]
    w_look   = traj[:, LOOKAHEAD_STEP - 1]             # step 5 = 0.5 s

i.e. byte-for-byte the forward pass ``eval_flagship_v4.collect_planner`` runs
(``st = world.encode_window(frames)`` -> ``goal_modes.resolve_goal`` ->
``head(st, v0, lambda_plan=1.0, **goal_kw)``), and the SAME 0.5 s lookahead index
E1a's pure-pursuit controller consumes. The loaders are imported FROM
``eval_flagship_v4`` (``load_v4_from_ck``), not reimplemented, so this cannot
drift from the open-loop gate primary.

GOAL PROVENANCE (stamped, per GATE_PROTOCOL 0.8)
-----------------------------------------------
``--goal-mode oracle`` (default) mints ``route`` / ``route_graded`` / ``vt_band``
from the ego's own FUTURE poses via ``v4_labels.mint_window`` — the SAME call
``FlagshipV4Dataset.__getitem__`` makes — so this surface matches the gate's
open-loop primary. In the closed loop the goal is re-minted at each step at the
REFERENCE index the model is actually observing (``t0 + mstar + W - 1``, the last
frame of the window fed in), which is the exact closed-loop analogue of the
open-loop oracle. That policy is recorded as ``goal_index_policy``.
``--goal-mode produced`` / ``neutral`` are available and use the identical
``goal_modes`` switch.

HONEST BOUNDS (stated, not hidden)
----------------------------------
* The real-footage source re-indexes along a 1-D manifold and warps by a
  ground-plane homography whose OOD envelope was MEASURED only to
  ``|dlat| <= 3.0 m`` / ``|dyaw| <= 12 deg`` (``lowood_flagship_ci.json``).
  ``np.interp`` CLAMPS beyond that, so the reported OOD ratio is a LOWER bound
  there. ``EXTRAPOLATION_*`` carries the out-of-envelope fraction. Any horizon
  whose peak OOD ratio exceeds ~1.5x is EXTRAPOLATION, not measurement (E1a's
  own rule).
* That envelope was measured on the FLAGSHIP v1 arm. Applied to v4 the *ratio*
  is a shift-magnitude proxy, not a v4 measurement; the ``EXTRAPOLATION_*``
  fractions need only the envelope CONSTANTS and are model-independent.
* An episode yields a window at K only if ``T - W - K >= 1``. On this corpus
  (T = 198-205) K=185 leaves ~1 window per episode: n is SMALL and is reported
  in every block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

for _p in ("/root/TanitAD/stack", "/root/TanitAD/stack/scripts",
           "/root/taniteval", "/workspace/TanitAD/stack",
           "/workspace/TanitAD/stack/scripts", "/workspace/taniteval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import goal_modes                                              # noqa: E402
import refb_labels as rl                                       # noqa: E402
import v4_labels                                               # noqa: E402
from driving_diagnostic import gt_ego_waypoints                # noqa: E402
from eval_flagship_v4 import load_v4_from_ck                   # noqa: E402
from tanitad.data.mixing import load_episode                   # noqa: E402
from tanitad.instruments.numerics import strict_numerics       # noqa: E402
from tanitad.lake.vtarget import savgol                        # noqa: E402
from taniteval import ci as _ci                                # noqa: E402
from taniteval import corridor as _corr                        # noqa: E402

# ---- loop constants: e1a_horizon.py / lowood_lanekeep.py / closedloop.py ----
W = 8
DT = 0.1
WHEELBASE = 2.7
LOOKAHEAD_STEP = 5
LD2_FLOOR = 0.25
STEER_CLAMP = 0.05
ACCEL_CLAMP = 3.0
SPEED_TC = 0.5
WP_STEPS = (5, 10, 15, 20)
WP_IDX = [k - 1 for k in WP_STEPS]

F_EFF = 266.0
CXY = 128.0

ENV_LAT_MAX = 3.0
ENV_YAW_MAX = 12.0
VT_DROPPED = 23


# --------------------------------------------------------------------------- #
# geometry — e1a_horizon.py VERBATIM                                            #
# --------------------------------------------------------------------------- #
def sampling_homography(dlat_m, dyaw_deg, h_cam, pitch_deg, f=F_EFF, c=CXY):
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


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class OODMap:
    """P1 MEASURED envelope |dlat|,|dpsi| -> ADE ratio (e1a_horizon.py verbatim)."""

    def __init__(self, ci_json):
        d = json.loads(Path(ci_json).read_text())
        self.base = d["baseline_real_frames"]["mean"]
        self.lat_x = np.array([r["amount"] for r in d["conditions"]["lat"]])
        self.lat_y = np.array([r["ade2s_ci"]["mean"] for r in d["conditions"]["lat"]])
        self.yaw_x = np.array([r["amount"] for r in d["conditions"]["yaw"]])
        self.yaw_y = np.array([r["ade2s_ci"]["mean"] for r in d["conditions"]["yaw"]])

    def ratio_arr(self, lat_abs, yaw_abs_deg):
        al = np.interp(lat_abs, self.lat_x, self.lat_y)      # CLAMPS beyond 3.0 m
        ay = np.interp(yaw_abs_deg, self.yaw_x, self.yaw_y)  # CLAMPS beyond 12 deg
        ex_l = np.clip((al - self.base) / self.base, 0.0, None)
        ex_y = np.clip((ay - self.base) / self.base, 0.0, None)
        return 1.0 + ex_l + ex_y


# --------------------------------------------------------------------------- #
# the v4 goal oracle — the SAME mint FlagshipV4Dataset.__getitem__ calls         #
# --------------------------------------------------------------------------- #
class GoalCache:
    """``route`` / ``route_graded`` / ``vt_band`` per (episode, t_last).

    Minted by ``v4_labels.mint_window(poses, t_last, v_smoothed=savgol(v))`` —
    the identical call ``FlagshipV4Dataset.__getitem__`` makes, so the goal the
    closed loop feeds is the same object the open-loop gate primary feeds.
    Indices for which the labeler refuses fall back to the sentinels the labeler
    itself emits (``ROUTE_UNKNOWN`` / ``vt_band = 23``) and are COUNTED."""

    def __init__(self, episodes, min_lookahead=50, use_net_dyaw=False):
        self.eps = episodes
        self.min_lookahead = min_lookahead
        self.use_net_dyaw = use_net_dyaw
        self._c: dict[int, tuple] = {}
        self.n_fail = 0
        self.n_total = 0

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


class V4Planner:
    """The ONE changed thing vs e1a_horizon.py: the per-step plan call."""

    def __init__(self, world, head, goal_head, goal_mode, allow_fallback=False):
        self.world, self.head = world, head
        self.goal_head, self.goal_mode = goal_head, goal_mode
        self.allow_fallback = allow_fallback
        self.rec: dict = {}

    @torch.no_grad()
    def traj(self, fw, v0, goal_batch):
        """fw [b, W, C, H, W'] float in [0,1]; v0 [b] -> traj [b, 20, 2]."""
        st = self.world.encode_window(fw)
        goal_kw, rec = goal_modes.resolve_goal(
            self.goal_mode, head=self.head, batch=goal_batch, v0=v0, states=st,
            goal_head=self.goal_head, allow_fallback=self.allow_fallback)
        rec.pop("scalars", None)
        self.rec = rec
        return self.head(st, v0, lambda_plan=1.0, **goal_kw)["traj"]


def _frames(ep, a, b):
    """uint8 [T,C,H,W] -> float [0,1] window, EXACTLY ``to_float_frames``.

    e1a materialises the whole episode as float up front; at 9x256x256 that is
    ~470 MB/episode here, so the identical conversion is done per slice. Same
    arithmetic, same dtype."""
    x = ep.frames[a:b]
    return x.float().div(255.0) if x.dtype == torch.uint8 else x.float()


# --------------------------------------------------------------------------- #
# open-loop canary — must reproduce the gate's MEASURED open-loop primary        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def openloop_canary(planner, episodes, goals, device, stride=8, batch=16):
    """v4 planner path open-loop, 4-waypoint convention (steps 5/10/15/20).

    The window set ``range(0, T - W - 20, stride)`` is IDENTICAL to
    ``eval_flagship_v4``'s (``FlagshipV4Dataset.index`` is
    ``range(T - window - max_horizon)`` with ``max_horizon = 20``, filtered by
    ``t % stride == 0``), so this is a direct reproduction test of that harness's
    ``wp4_ade_0_2s_selfcomputed`` — the check that the encode / goal / head /
    ego-frame plumbing in THIS file is the same plumbing that produced the
    gate's open-loop number."""
    ades, eids = [], []
    for ei, ep in enumerate(episodes):
        poses = torch.as_tensor(ep.poses, dtype=torch.float32)
        T = int(poses.shape[0])
        starts = list(range(0, T - W - max(WP_STEPS), stride))
        for bi in range(0, len(starts), batch):
            ch = starts[bi:bi + batch]
            frames = torch.stack([_frames(ep, t0, t0 + W) for t0 in ch]).to(device)
            last = torch.tensor([t0 + W - 1 for t0 in ch])
            v0 = poses[last, 3].to(device)
            g = goals.get(ei, last.numpy(), device)
            tr = planner.traj(frames, v0, g)[:, WP_IDX].cpu()
            gt = gt_ego_waypoints(poses, last, wp_steps=WP_STEPS)
            ades.append(torch.linalg.norm(tr - gt, dim=-1).mean(1))
            eids += [str(ei)] * len(ch)
    a = torch.cat(ades).numpy().astype(np.float64)
    out = _ci.episode_cluster_bootstrap(a, eids, n_boot=2000)
    out["plain_mean"] = float(a.mean())
    return out


# --------------------------------------------------------------------------- #
# the closed loop — e1a_horizon.rollout, plan call swapped                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout(planner, episodes, goals, device, K, stride=8, batch=16,
            verbose=True):
    rows = {k: [] for k in ("ade2s", "hd2s", "hdK", "speed", "eid", "t0", "epi")}
    lat_all, yaw_all, de_fixed = [], [], []
    fixed_steps = [s for s in (20, 40, 80, 120, 160, 185, 190) if s <= K]
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
                dpsi = _wrap(eyaw - yref)
                lat_t[:, k] = dlat; yaw_t[:, k] = dpsi
                wins = [_frames(ep, int(t0[i] + mstar[i]),
                                int(t0[i] + mstar[i]) + W) for i in range(b)]
                fw = torch.stack(wins).to(device)
                Hs = torch.stack([
                    sampling_homography(float(dlat[i]),
                                        float(math.degrees(dpsi[i])), 1.5, 0.0)
                    for i in range(b)])
                fw = warp_batch(fw, Hs)
                # ---- THE ONE CHANGED LINE vs e1a_horizon.py -----------------
                last_ref = (t0 + mstar + W - 1).numpy()
                g = goals.get(ep_i, last_ref, device)
                evd = ev.to(device)
                w_look = planner.traj(fw, evd, g)[:, LOOKAHEAD_STEP - 1].cpu()
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
            gt2 = gt_ego_waypoints(poses, last, wp_steps=WP_STEPS)
            rows["ade2s"].append(
                torch.linalg.norm(ego_ego[:, WP_IDX] - gt2, dim=-1).mean(1))
            gtf = gt_ego_waypoints(poses, last, wp_steps=tuple(fixed_steps))
            de_fixed.append(torch.linalg.norm(
                ego_ego[:, [s - 1 for s in fixed_steps]] - gtf, dim=-1))
            lat_abs = lat_t.abs(); yaw_abs_deg = yaw_t.abs() * 180 / math.pi
            lat_all.append(lat_abs); yaw_all.append(yaw_abs_deg)
            rows["hd2s"].append(
                _wrap(poses[last + 20, 2] - poses[last, 2]).abs() * 180 / math.pi)
            rows["hdK"].append(
                _wrap(poses[last + K, 2] - poses[last, 2]).abs() * 180 / math.pi)
            rows["speed"].append(poses[last, 3])
            rows["eid"] += [str(ep_i)] * b
            rows["t0"].append(t0.clone())
            rows["epi"].append(torch.full((b,), ep_i))
        if verbose:
            print(f"    [cl] K={K} ep {ep_i + 1}/{len(episodes)} "
                  f"(rollout steps so far {n_steps_done})", flush=True)
    if not rows["eid"]:
        return None
    out = {k: (v if k == "eid" else torch.cat(v)) for k, v in rows.items()}
    out["lat"] = torch.cat(lat_all)
    out["yaw"] = torch.cat(yaw_all)
    out["de_fixed"] = torch.cat(de_fixed)
    out["fixed_steps"] = fixed_steps
    out["_rollout_steps_executed"] = n_steps_done
    return out


def ood_block(pw, mask, ood, K):
    """The OOD/EXTRAPOLATION rows corridor_block deliberately omits (they need
    the external P1 envelope JSON, which is not a library dependency)."""
    m = np.flatnonzero(mask)
    if len(m) < 2:
        return None
    e = [pw["eid"][i] for i in m]
    lat = pw["lat"].numpy()[m]
    yaw = pw["yaw"].numpy()[m]
    ratio = ood.ratio_arr(lat, yaw)
    bo = lambda x: _ci.episode_cluster_bootstrap(np.asarray(x, float), e,  # noqa: E731
                                                 n_boot=2000)
    return {
        "horizon_K": K, "horizon_s": round(K * DT, 2),
        "n_windows": int(len(m)), "n_episodes": int(len(set(e))),
        "ood_peak_ratio": bo(ratio.max(1)),
        "ood_mean_ratio": bo(ratio.mean(1)),
        "frac_windows_ood_peak_under_1p16": round(float((ratio.max(1) <= 1.16).mean()), 4),
        "frac_windows_ood_peak_under_1p5": round(float((ratio.max(1) <= 1.5).mean()), 4),
        "EXTRAPOLATION_frac_steps_lat_over_3m": round(float((lat > ENV_LAT_MAX).mean()), 5),
        "EXTRAPOLATION_frac_steps_yaw_over_12deg": round(float((yaw > ENV_YAW_MAX).mean()), 5),
        "EXTRAPOLATION_frac_windows_any_step_out_of_envelope": round(
            float(((lat > ENV_LAT_MAX) | (yaw > ENV_YAW_MAX)).any(1).mean()), 4),
        "EXTRAPOLATION_VERDICT": (
            "EXTRAPOLATION — peak OOD ratio exceeds ~1.5x, beyond the MEASURED "
            "envelope; this is not a measurement"
            if float(ratio.max(1).mean()) > 1.5 else
            "within the measured envelope on average"),
    }


def emit(pw, ood, K, thresholds, primary, junction_deg, surface="closed_loop"):
    """corridor.stratified (THE registered emitter) + the OOD rows."""
    lat = pw["lat"].numpy()
    yaw = pw["yaw"].numpy()
    eid = pw["eid"]
    hd = pw["hd2s"].numpy()
    spd = pw["speed"].numpy()
    out = _corr.stratified(lat, eid, hd, spd, thresholds=tuple(thresholds),
                           primary=primary, junction_deg=junction_deg,
                           yaw_abs_deg=yaw, ade2s=pw["ade2s"].numpy(),
                           n_boot=2000, seed=0, surface=surface)
    junc = _corr.junction_mask(hd, junction_deg)
    long_ = (~junc) & (spd >= np.median(spd))
    out["ood"] = {
        "_envelope": "P1 MEASURED |dlat|<=3.0 m / |dyaw|<=12 deg "
                     "(lowood_flagship_ci.json, measured on the FLAGSHIP v1 "
                     "arm). np.interp CLAMPS beyond it, so the ratio is a "
                     "LOWER bound there; the EXTRAPOLATION_* fractions need "
                     "only the envelope CONSTANTS and are model-independent.",
        "overall": ood_block(pw, np.ones(len(hd), bool), ood, K),
        "junction": ood_block(pw, junc, ood, K),
        "longitudinal": ood_block(pw, long_, ood, K),
        "other": ood_block(pw, (~junc) & (~long_), ood, K),
    }
    fs = pw["fixed_steps"]
    dfx = pw["de_fixed"].numpy()
    out["de_at_elapsed_s"] = {
        f"{s * DT:g}": _ci.episode_cluster_bootstrap(
            np.asarray(dfx[:, i], float), eid, n_boot=2000)
        for i, s in enumerate(fs)}
    out["rollout_steps_executed"] = int(pw["_rollout_steps_executed"])
    out["rollout_advanced_K_steps"] = bool(pw["lat"].shape[1] == K)
    return out


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anchors-dense", required=True)
    ap.add_argument("--head-config", default=None)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--p1-json", default="/root/lanekeep/lowood_flagship_ci.json")
    ap.add_argument("--horizons", default="185,20")
    ap.add_argument("--pair-ref", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=999)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--corridor-grid", default="1.0,1.75,2.5")
    ap.add_argument("--corridor-halfwidth", type=float, default=1.75)
    ap.add_argument("--junction-deg", type=float, default=10.0)
    ap.add_argument("--goal-mode", choices=goal_modes.GOAL_MODES, default="oracle")
    ap.add_argument("--goal-fallback", action="store_true")
    ap.add_argument("--skip-canary", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    Ks = [int(x) for x in a.horizons.split(",")]
    thresholds = [float(x) for x in a.corridor_grid.split(",")]
    primary = a.corridor_halfwidth
    assert primary in thresholds, (primary, thresholds)
    ood = OODMap(a.p1_json)

    eps_files = sorted(Path(a.val_dir).glob("ep_*.pt"))[:a.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in eps_files]
    Ts = [int(e.poses.shape[0]) for e in episodes]
    surv = {K: int(sum(1 for T in Ts if T - W - K >= 1)) for K in Ks}
    nwin = {K: int(sum(len(range(0, T - W - K, a.stride)) for T in Ts)) for K in Ks}
    print(f"[v4cl] {len(episodes)} eps from {a.val_dir} | T in [{min(Ts)},{max(Ts)}]"
          f" | K sweep {Ks} | dev {device}", flush=True)
    print(f"[v4cl] episodes surviving each K: {surv}", flush=True)
    print(f"[v4cl] windows at stride {a.stride} per K: {nwin}", flush=True)
    assert surv[max(Ks)] > 0, f"K_max {max(Ks)} leaves no window (shortest {min(Ts)})"

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    world, grounding, head, step, hcfg, goal_head = load_v4_from_ck(
        ck, device, head_config_path=(a.head_config
                                      or Path(a.ckpt).parent / "config.json"),
        anchors_dense_path=a.anchors_dense)
    del ck
    planner = V4Planner(world, head, goal_head, a.goal_mode, a.goal_fallback)

    t = time.time()
    goals = GoalCache(episodes)
    for i in range(len(episodes)):
        goals._build(i)
    print(f"[v4cl] goal oracle minted for {goals.n_total} (episode, t_last) "
          f"indices in {time.time() - t:.0f}s; labeler refusals {goals.n_fail}",
          flush=True)

    res = {
        "_experiment": "flagship-v4 CO-PRIMARY corridor_departure_rate, CLOSED LOOP",
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
        "_design": ("real-footage-in-the-loop low-OOD closed loop. Loop body, "
                    "window/stratum bookkeeping, OOD accounting and the "
                    "common-start PAIRED design reproduced from e1a_horizon.py "
                    "(which produced the gate card's REF-C reference numbers at "
                    "the SAME K=185); the ONE change is the per-step plan call, "
                    "which is eval_flagship_v4.collect_planner's forward pass "
                    "(world.encode_window -> goal_modes.resolve_goal -> "
                    "head(st, v0, lambda_plan=1.0, **goal_kw)) with the 0.5 s "
                    "lookahead waypoint traj[:, 4] fed to the SAME pure-pursuit "
                    "controller."),
        "_emitter": "taniteval.corridor.stratified (the registered co-primary "
                    "emitter; corridor_block is e1a_horizon.block lifted)",
        "_estimator": ("episode_cluster_bootstrap (taniteval/ci.py), B=2000, "
                       "resampling unit = val EPISODE; PAIRED form for the "
                       "K-vs-K deltas. NEVER overlapping_holdout_se."),
        "ckpt": a.ckpt, "ckpt_md5": _md5(a.ckpt), "ckpt_step": step,
        "anchors_dense": a.anchors_dense, "anchors_md5": _md5(a.anchors_dense),
        "head_cfg": {"n_anchors": hcfg.n_anchors, "horizons": list(hcfg.horizons),
                     "factorised": hcfg.factorised,
                     "cond_vtarget": hcfg.cond_vtarget,
                     "cond_route": hcfg.cond_route,
                     "cond_imagination": hcfg.cond_imagination,
                     "goal_dropout": getattr(hcfg, "goal_dropout", None)},
        "goal_provenance": goal_modes.provenance(a.goal_mode, cfg=head.cfg),
        "goal_index_policy": (
            "FOLLOW — at every rollout step the oracle goal is re-minted at the "
            "REFERENCE index the model is actually observing (t0 + mstar + W - 1, "
            "the last frame of the warped window fed in), by the same "
            "v4_labels.mint_window call FlagshipV4Dataset.__getitem__ makes. "
            "This is the closed-loop analogue of the open-loop oracle; it is NOT "
            "a route given once at t=0."),
        "goal_labeler_refusals": goals.n_fail,
        "goal_labeler_indices": goals.n_total,
        "val_dir": a.val_dir, "n_episodes": len(episodes),
        "episode_T_min": min(Ts), "episode_T_max": max(Ts),
        "episodes_surviving_each_horizon": surv,
        "windows_at_each_horizon": nwin,
        "_horizon_ceiling_note": (
            f"an episode yields a window at K only if T-W-K>=1; T ranges "
            f"{min(Ts)}-{max(Ts)} frames here, so the ceiling is K = "
            f"{max(Ts) - W - 1} ({(max(Ts) - W - 1) * DT:.1f} s). At K=185 only "
            f"~1 window per episode survives — n is SMALL and is reported in "
            f"every block."),
        "stride": a.stride, "batch": a.batch,
        "corridor_thresholds_m": thresholds, "corridor_primary_m": primary,
        "junction_deg": a.junction_deg,
        "horizons_K": Ks, "horizons_s": [round(k * DT, 2) for k in Ks],
        "p1_envelope": a.p1_json, "p1_baseline_ade2s": ood.base,
        "reference_refc_base_K185_heldout44": {
            "_source": "Project Steering/Gates/flagship-v4-30k.card.json "
                       "co_primary.report_only_rationale + "
                       "e1a_horizon_heldout44_K185.json",
            "_evidence_class": "INHERITED (E1a, not re-run here)",
            "_caveat": "measured on physicalai-val-heldout-79d4e3d2d4c6 (44 eps, "
                       "43 windows at K=185), NOT this registered 40-episode val "
                       "cache — a scale reference, NOT a window-matched pair",
            "overall_cdr": [0.5877, 0.5107, 0.6622],
            "junction_cdr": [0.8414, 0.8144, 0.8667],
            "peak_xte_m": 38.94},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))

    with strict_numerics():
        if not a.skip_canary:
            t = time.time()
            c = openloop_canary(planner, episodes, goals, device, a.stride,
                                a.batch)
            c["_check"] = ("v4 planner path OPEN LOOP, 4-waypoint convention, "
                           "SAME window set as eval_flagship_v4. MUST reproduce "
                           "that harness's wp4_ade_0_2s_selfcomputed for this "
                           "ckpt/goal-mode, or the plumbing in this file differs "
                           "from the plumbing that produced the gate primary.")
            res["openloop_canary_ade_0_2s"] = c
            print(f"[v4cl] CANARY open-loop ADE@2s = {c['plain_mean']:.4f} "
                  f"(cluster mean {c['mean']:.4f} [{c['lo']:.4f},{c['hi']:.4f}]) "
                  f"n={c['n_windows']} ({time.time() - t:.0f}s)", flush=True)
            Path(a.out).write_text(json.dumps(res, indent=2, default=str))

        per_K = {}
        for K in Ks:
            t = time.time()
            pw = rollout(planner, episodes, goals, device, K, a.stride, a.batch)
            if pw is None:
                res.setdefault("all_windows", {})[str(K)] = {
                    "_skipped": "no window survives this horizon"}
                continue
            per_K[K] = pw
            blk = emit(pw, ood, K, thresholds, primary, a.junction_deg)
            res.setdefault("all_windows", {})[str(K)] = blk
            o = blk["overall"]
            oo = blk["ood"]["overall"]
            print(f"[v4cl] K={K:4d} ({K * DT:4.1f}s) n_win={o['n_windows']:5d} "
                  f"n_ep={o['n_episodes']:3d} "
                  f"CDR@{primary}={o['corridor_departure_rate']['mean']:.4f} "
                  f"[{o['corridor_departure_rate']['lo']:.4f},"
                  f"{o['corridor_departure_rate']['hi']:.4f}] "
                  f"winDEP={o['window_departure_rate']['mean']:.4f} "
                  f"peakXTE={o['peak_xte_m']['mean']:.3f} "
                  f"OODpeak={oo['ood_peak_ratio']['mean']:.3f} "
                  f"outEnv={oo['EXTRAPOLATION_frac_windows_any_step_out_of_envelope']:.3f}"
                  f" ({time.time() - t:.0f}s)", flush=True)
            Path(a.out).write_text(json.dumps(res, indent=2, default=str))
            torch.save({k: pw[k] for k in ("lat", "yaw", "ade2s", "hd2s", "hdK",
                                           "speed", "eid", "t0", "epi",
                                           "de_fixed", "fixed_steps")},
                       str(Path(a.out).with_suffix("")) + f"_perwindow_K{K}.pt")

    # ---- COMMON-START PAIRED curve (decision-grade) --------------------------
    ref = a.pair_ref
    key = lambda pw: [(int(x), int(y)) for x, y in zip(pw["epi"], pw["t0"])]  # noqa: E731
    if len(per_K) >= 2 and ref in per_K:
        keysets = [set(key(per_K[K])) for K in per_K]
        common = sorted(set.intersection(*keysets))
        node = {"_note": ("IDENTICAL windows at every horizon (the start set at "
                          "K_max is a subset of every smaller K's). Removes the "
                          "window-composition confound; deltas use the PAIRED "
                          "episode-cluster bootstrap."),
                "n_common_windows": len(common), "pair_reference_K": ref}
        if len(common) >= 8:
            sel = {}
            for K in per_K:
                order = {c: i for i, c in enumerate(key(per_K[K]))}
                sel[K] = np.array([order[c] for c in common])
            for K in per_K:
                pw = per_K[K]
                s = sel[K]
                sub = {"eid": [pw["eid"][i] for i in s],
                       "lat": pw["lat"][s], "yaw": pw["yaw"][s],
                       "ade2s": pw["ade2s"][s], "hd2s": pw["hd2s"][s],
                       "hdK": pw["hdK"][s], "speed": pw["speed"][s],
                       "de_fixed": pw["de_fixed"][s],
                       "fixed_steps": pw["fixed_steps"],
                       "_rollout_steps_executed": pw["_rollout_steps_executed"]}
                node[str(K)] = emit(sub, ood, K, thresholds, primary,
                                    a.junction_deg)
            hd = per_K[ref]["hd2s"].numpy()[sel[ref]]
            spd = per_K[ref]["speed"].numpy()[sel[ref]]
            eidc = [per_K[ref]["eid"][i] for i in sel[ref]]
            junc = _corr.junction_mask(hd, a.junction_deg)
            strata = {"overall": np.ones(len(hd), bool), "junction": junc,
                      "longitudinal": (~junc) & (spd >= np.median(spd)),
                      "other": (~junc) & (~((~junc) & (spd >= np.median(spd))))}
            deltas = {}
            for nm, mk in strata.items():
                m = np.flatnonzero(mk)
                if len(m) < 2:
                    continue
                e = [eidc[i] for i in m]
                r_lat = per_K[ref]["lat"][sel[ref]].numpy()[m]
                dd = {}
                for K in per_K:
                    if K == ref:
                        continue
                    k_lat = per_K[K]["lat"][sel[K]].numpy()[m]
                    dd[str(K)] = {
                        "d_corridor_departure_rate": _corr.paired_stratum_delta(
                            r_lat, k_lat, e, threshold=primary),
                        "d_peak_xte_m": _ci.paired_episode_cluster_bootstrap(
                            k_lat.max(1), r_lat.max(1), e, n_boot=2000),
                        "d_closed_ade2s_m": _ci.paired_episode_cluster_bootstrap(
                            per_K[K]["ade2s"][sel[K]].numpy()[m],
                            per_K[ref]["ade2s"][sel[ref]].numpy()[m], e,
                            n_boot=2000)}
                deltas[nm] = dd
            node[f"deltas_vs_K{ref}"] = {
                "_note": f"positive => the LONGER horizon is WORSE (higher). "
                         f"Reference K={ref}. Paired episode-cluster bootstrap, "
                         f"B=2000.", **deltas}
        else:
            node["_skipped"] = (f"only {len(common)} common windows; "
                                f"paired curve not emitted")
        res["paired_common_start"] = node

    try:
        res["_git_head"] = subprocess.check_output(
            ["git", "-C", "/root/TanitAD", "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        pass
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    print(f"[v4cl] wrote {a.out}", flush=True)
    print("V4CL_DONE", flush=True)


if __name__ == "__main__":
    main()
