"""TanitEval — P2 planner de-risk: CEM + cost over the FROZEN v1 world model.

The v3 thesis (V3_HIERARCHICAL_PLANNING_DESIGN.md §0/§8): the driving decision
should come from PLANNING — evaluating alternative action sequences against a
strategic goal via the world model's predicted consequences — NOT from a
supervised trajectory head (which degenerates: the v1 tactical head is 3.38 m
ADE@2s, worse than constant-velocity 0.82 m). P2 tests this at ZERO training
cost: a CEM planner + a hand-built cost roll candidate action sequences through
the frozen v1 flagship's OPERATIVE world model (the proven grounded step-readout
rollout, 0.45 m with true actions) and pick the lowest-cost trajectory. Nothing
is trained; the operative WM and step-readout are loaded frozen.

WHAT IS FROZEN / REUSED (never reinvented)
  * loaders.load(flagship-30k)  -> frozen v1 model + grounded step-readout
  * metric_dynamics.rollout_decode -> the EXACT gate/leaderboard operative
    rollout (encode window -> predictor K steps under an action sequence ->
    per-step metric Δpose -> SE(2) accumulate). The CEM decision variable is the
    future action sequence `fa` fed to this function.
  * data.load_frames / driving_diagnostic.gt_ego_waypoints / baseline_waypoints
  * closedloop.{wp_to_control,build_action,bicycle_integrate} + its
    imagination-in-the-loop harness (G4 reuses it verbatim, only the PLAN step is
    swapped: tactical head -> CEM).
  * pathspeed.{step_speed,metric_block} -> planned-speed profile for the cost +
    the honest longitudinal/lateral decomposition of the planner's error.
  * ci.episode_cluster_bootstrap / ci.paired_episode_cluster_bootstrap — the
    program's ONE decision-grade estimator. ⛔ MIGRATED 2026-08-16: G1_pass and
    G4_pass were adjudicated by `_jack_paired` / `_jack_scalar`
    (`overlapping_holdout_se`) until then. Those two functions survive for
    REPRODUCTION ONLY and their output is quarantined under
    `legacy_overlapping_holdout_se`; gates.split_by_episode is now used only to
    rebuild that legacy block.

THE COST  J(plan) = w_v·(v̂ − v_target)²                 [track the minted target]
                  + w_c·(accel² + jerk²) + w_s·steer_rate²  [comfort / smoothness]
                  − w_p·progress                          [along-track progress]
                  (+ gap/TTC barrier — SKIPPED in v0: no lead-agent labels in our
                   front-cam+pose data, per the spec's "skip gap term v0")
  v_target is minted OFFLINE (85th-pct free-flow future speed, §3(1)); v̂ is the
  planned trajectory's step speed. Weights are ENGINEERED from physical scales
  (m/s, m/s², m) and a sensitivity sweep is reported — they are NOT fit to GT ADE
  (that would make the G1 test circular).

HONEST SCOPE (stated, not hidden)
  * The P2 cost is LONGITUDINAL + comfort + progress only. It carries NO lateral
    / route / goal term (the strategic goal module is P3). So the planner is
    longitudinally guided and laterally defaults to the smoothest low-curvature
    option its proposal set + WM allow. This is the point of P2: it isolates
    whether planning-over-a-frozen-WM beats the degenerate heads on the strength
    of longitudinal control alone. The lateral ceiling here is a RESULT (it tells
    us what P3/P4 must add), not a bug.
  * v_source (WHY the target: sign vs flow) needs a VLM sign-read pass we don't
    run on the eval pod; v_target provenance is `kinematic` only (honest gap).

Run:  python3 -m taniteval.planner_p2 --arm flagship-30k --episodes 40 \
          [--closed-loop] [--cl-episodes 16] [--sweep]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

# ⛔ was: sys.path.insert(0, "/root/TanitAD/stack"[/scripts]) — that put a
# possibly PRE-v5 tree IN FRONT of the caller's PYTHONPATH and published a
# plausible wrong number instead of an error (STALE_IMPORT_GUARD.md).
from taniteval.stack_guard import ensure_stack_on_path as _ensure_stack  # noqa: E402
_ensure_stack()
sys.path.insert(0, "/root/taniteval")

from driving_diagnostic import (WP_STEPS, baseline_waypoints,  # noqa: E402
                                gt_ego_waypoints, net_heading_change_deg)
from tanitad.eval.gates import split_by_episode  # noqa: E402
from tanitad.models.metric_dynamics import rollout_decode  # noqa: E402
from taniteval import ci as _ci  # noqa: E402
from taniteval import closedloop as cl  # noqa: E402
from taniteval import driving as _drv  # noqa: E402
from taniteval import pathspeed as ps  # noqa: E402

# --- estimator policy — IMPORTED from driving.py, never restated ------------ #
# ⛔ MIGRATED 2026-08-16. Until then G1_pass (:458 -> :468) and G4_pass
# (:586 -> :595) were adjudicated by `_jack_paired` / `_jack_scalar`, i.e.
# `overlapping_holdout_se` — the estimator CLAUDE.md bans, which BIASES THE
# POINT ESTIMATE (mean-of-split-means, not full_set) and was measured on paired
# deltas at up to x-4.15 INCLUDING A SIGN FLIP. `g1_delta` is exactly a paired
# delta, so the gate could have read the wrong sign. See
# `…/incoming/2026-08-16-jack-in-gates/JACK_IN_GATES.md` for the re-decision on
# banked data (neither verdict flips) and for the measured per-arm corrections.
N_BOOT = _ci.DEFAULT_N_BOOT                        # 2000
DECISION_ESTIMATORS = _drv.DECISION_ESTIMATORS
DEPRECATED_ESTIMATOR = _drv.DEPRECATED_ESTIMATOR   # overlapping_holdout_se
ESTIMATOR_NOTE = _drv.ESTIMATOR_NOTE
LEGACY_BLOCK = cl.LEGACY_BLOCK                     # legacy_overlapping_holdout_se

# --- protocol constants (parity with rollout.py / closedloop.py) ------------ #
DT = 0.1
SPEED_SCALE = 10.0
WINDOW = 8
K_MAX = max(WP_STEPS)          # 20 = 2 s @ 10 Hz
IDX = [k - 1 for k in WP_STEPS]
STRIDE = 8

# --- action envelope (data: |steer|<=0.016, |accel|<=1.9; headroom for search) #
STEER_CLAMP = 0.03
ACCEL_CLAMP = 2.5

# --- CEM defaults (spec M6: N=64, 3 iters, elite-8) ------------------------- #
CEM = dict(N=64, iters=3, elite=8, K=K_MAX,
           sig_steer=0.006, sig_accel=0.5,
           min_steer=0.0008, min_accel=0.05)
CEM_CL = dict(N=48, iters=2, elite=8, K=K_MAX,       # lighter budget for the loop
              sig_steer=0.006, sig_accel=0.5,
              min_steer=0.0008, min_accel=0.05)

# --- G4 threshold: the v1 tactical head driving the SAME closed loop -------- #
# ⛔ The historical threshold 1.6852 is itself an `overlapping_holdout_se`
# heldout mean (`closedloop_flagship-30k.json`) — so the pre-2026-08-16 G4 gate
# compared a biased point estimate against a biased threshold and called the
# result a verdict. The decision-grade value is the FULL-SET mean of the same
# banked windows: 1.7318 (`closedloop_flagship-30k.CORRECTED.json`, reproduced
# bit-exactly from `raw_windows/clwin_flagship-30k.pt` in JACK_IN_GATES.md).
# Both are carried: the gate uses the corrected one, the legacy one stays
# quotable for reproduction and is never the decision.
G4_HEAD_BASELINE_ADE2S = 1.7318
G4_HEAD_BASELINE_ADE2S_LEGACY = 1.6852
G4_HEAD_BASELINE_SOURCE = (
    "closedloop_flagship-30k.CORRECTED.json closed_bike ade@2s, full_set mean "
    "over 881 windows / 40 episodes (episode_cluster_bootstrap); the legacy "
    "1.6852 is the same quantity under overlapping_holdout_se")

# --- v_target minting (spec §3(1) / V3_GOAL_VOCABULARY_V1 label minting) ----- #
VT_LOOK_LO = 100               # 10 s  (min lookahead for a valid free-flow read)
VT_LOOK_HI = 200               # 20 s  (max lookahead)
VT_MIN_STEPS = 30              # need >=3 s of free-flow samples else fall back
VT_PCTL = 0.85                 # 85th percentile
VT_HARD_DECEL = 1.5            # m/s^2: steps braking harder than this = constrained

# --- engineered cost weights (physical scales; NOT fit to GT ADE) ----------- #
W = dict(v=1.0, c=0.10, s=50.0, p=0.02)


# ======================================================================== #
# v_target — offline free-flow target-speed label (per window)              #
# ======================================================================== #
def vtarget_for(poses: torch.Tensor, last: torch.Tensor):
    """85th-pct free-flow future speed over the next 10-20 s, per window.

    poses [T,4] (x,y,yaw,v); last [b] window-end indices. For each window look
    ahead VT_LOOK_LO..HI steps, drop steps braking harder than VT_HARD_DECEL
    (constrained), take the VT_PCTL percentile of the remaining speeds. Falls
    back to the current speed when the free-flow sample is too short (episode end
    / sustained braking). Returns (v_target [b], valid [b] bool)."""
    T = poses.shape[0]
    v = poses[:, 3]
    vt = torch.empty(last.shape[0])
    valid = torch.zeros(last.shape[0], dtype=torch.bool)
    for i, L in enumerate(last.tolist()):
        hi = min(L + VT_LOOK_HI, T)
        fut = v[L + 1:hi]
        if fut.numel() >= VT_MIN_STEPS:
            acc = (fut[1:] - fut[:-1]) / DT                 # per-step accel
            keep = torch.ones(fut.numel(), dtype=torch.bool)
            keep[1:] = acc > -VT_HARD_DECEL                 # free-flow only
            ff = fut[keep]
            if ff.numel() >= VT_MIN_STEPS:
                vt[i] = torch.quantile(ff, VT_PCTL)
                valid[i] = True
                continue
        vt[i] = v[L]                                        # fallback: hold speed
    return vt, valid


# ======================================================================== #
# Cost                                                                      #
# ======================================================================== #
def cost_fn(traj: torch.Tensor, exec_act: torch.Tensor,
            v_target: torch.Tensor, w=W) -> torch.Tensor:
    """J for a batch of candidate rollouts. LOWER = better.

    traj [.,K,2] rolled-out ego waypoints; exec_act [.,K,2] the (steer,accel)
    executed per step; v_target [.]. Returns J [.]."""
    vhat = ps.step_speed(traj)                              # [.,K] planned speed
    speed_err = ((vhat - v_target[:, None]) ** 2).mean(dim=1)
    accel = exec_act[..., 1]
    steer = exec_act[..., 0]
    jerk = (accel[:, 1:] - accel[:, :-1]) / DT
    steer_rate = (steer[:, 1:] - steer[:, :-1]) / DT
    comfort = (accel ** 2).mean(1) + (jerk ** 2).mean(1)
    steer_smooth = (steer_rate ** 2).mean(1)
    seg = (traj[:, 1:] - traj[:, :-1]).norm(dim=-1)
    progress = seg.sum(1) + traj[:, 0].norm(dim=-1)         # arc length to 2 s
    return (w["v"] * speed_err + w["c"] * comfort
            + w["s"] * steer_smooth - w["p"] * progress)


# ======================================================================== #
# Rollout + candidate scoring (batched over windows x samples)              #
# ======================================================================== #
def _append_v0(fa_base: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """[.,H,2] (steer,accel) + v0 [.] -> [.,H,3] with the constant v0 channel."""
    v0c = (v0 / SPEED_SCALE).view(-1, 1, 1).expand(-1, fa_base.shape[1], 1)
    return torch.cat([fa_base, v0c], dim=-1)


@torch.no_grad()
def eval_candidates(model, step_readout, states, aw, cand, v_target, v0, w, K,
                    plan_first):
    """Roll every candidate action seq through the frozen operative WM, score.

    states [B,W,S], aw [B,W,A]; cand [B,n,K,2]; v_target/v0 [B]. Returns
    (J [B,n], traj [B,n,K,2]). `plan_first`=False keeps aw[:,-1] = the observed
    last action (apples-to-apples with the true-action gate rollout, fa = the
    plan); True overrides aw[:,-1] with the planned a0 so the planner controls
    the immediate step (closed-loop / execution convention)."""
    B, n = cand.shape[:2]
    st = states.repeat_interleave(n, dim=0)
    a_w = aw.repeat_interleave(n, dim=0).clone()
    v0e = v0.repeat_interleave(n)
    vte = v_target.repeat_interleave(n)
    fac = cand.reshape(B * n, K, 2)
    if plan_first:
        a_w[:, -1, :2] = fac[:, 0]                          # a0 drives step 1
        fa = _append_v0(fac[:, 1:], v0e)                    # a1..a_{K-1}
        exec_act = fac                                      # a0..a_{K-1}
    else:
        fa = _append_v0(fac, v0e)
        exec_act = torch.cat([a_w[:, -1:, :2], fac[:, :K - 1]], dim=1)
    traj, _ = rollout_decode(model.predictor, st, a_w, fa, step_readout, K)
    J = cost_fn(traj, exec_act, vte, w)
    return J.reshape(B, n), traj.reshape(B, n, K, 2)


def _clamp(cand: torch.Tensor) -> torch.Tensor:
    cand[..., 0].clamp_(-STEER_CLAMP, STEER_CLAMP)
    cand[..., 1].clamp_(-ACCEL_CLAMP, ACCEL_CLAMP)
    return cand


def build_seeds(v_target, v0, device, K, head_seed=None):
    """Coarse (steer x accel) constant-action proposal grid + the tactical head's
    immediate control as a learned proposal prior. Returns [B,M,K,2].

    accel grid is anchored on the accel that reaches v_target in ~2 s, so the
    longitudinal seeds already bracket the target; steer grid brackets
    left/straight/right so the lateral option space is covered before CEM."""
    B = v0.shape[0]
    steer_grid = torch.tensor([-0.012, -0.004, 0.0, 0.004, 0.012], device=device)
    a_reach = ((v_target - v0) / (K * DT)).clamp(-ACCEL_CLAMP, ACCEL_CLAMP)  # [B]
    accel_offsets = torch.tensor([-0.6, 0.0, 0.6], device=device)
    seeds = []
    for da in accel_offsets:
        acc = (a_reach + da).clamp(-ACCEL_CLAMP, ACCEL_CLAMP)               # [B]
        for st in steer_grid:
            row = torch.stack([st.expand(B), acc], dim=-1)                  # [B,2]
            seeds.append(row[:, None, :].expand(B, K, 2))
    # explicit coast-at-current-speed seed (a_reach=0, straight)
    seeds.append(torch.stack([torch.zeros(B, device=device),
                              torch.zeros(B, device=device)], dim=-1
                             )[:, None, :].expand(B, K, 2))
    seeds = torch.stack(seeds, dim=1).clone()                              # [B,M,K,2]
    if head_seed is not None:
        seeds = torch.cat([seeds, head_seed[:, None]], dim=1)
    return _clamp(seeds)


@torch.no_grad()
def cem_plan(model, step_readout, states, aw, v_target, v0, w, cfg,
             head_seed=None, plan_first=False):
    """CEM over future action sequences for a chunk of B windows. Returns
    (best_act [B,K,2], best_traj [B,K,2], best_cost [B])."""
    B = states.shape[0]
    dev = states.device
    K, N, elite = cfg["K"], cfg["N"], cfg["elite"]
    ar = torch.arange(B, device=dev)

    seeds = build_seeds(v_target, v0, dev, K, head_seed)                    # [B,M,K,2]
    cost_s, traj_s = eval_candidates(model, step_readout, states, aw, seeds,
                                     v_target, v0, w, K, plan_first)
    best_i = cost_s.argmin(dim=1)
    best_cost = cost_s[ar, best_i]
    best_act = seeds[ar, best_i].clone()
    best_traj = traj_s[ar, best_i].clone()

    mu = best_act.clone()
    sig = torch.empty(B, K, 2, device=dev)
    sig[..., 0] = cfg["sig_steer"]
    sig[..., 1] = cfg["sig_accel"]
    for _ in range(cfg["iters"]):
        eps = torch.randn(B, N, K, 2, device=dev)
        cand = _clamp(mu[:, None] + sig[:, None] * eps)
        cost_c, traj_c = eval_candidates(model, step_readout, states, aw, cand,
                                         v_target, v0, w, K, plan_first)
        cmin, cidx = cost_c.min(dim=1)
        imp = cmin < best_cost
        best_cost = torch.where(imp, cmin, best_cost)
        best_act = torch.where(imp[:, None, None], cand[ar, cidx], best_act)
        best_traj = torch.where(imp[:, None, None], traj_c[ar, cidx], best_traj)
        topk = cost_c.topk(elite, dim=1, largest=False).indices
        el = torch.gather(cand, 1, topk[..., None, None].expand(-1, -1, K, 2))
        mu = el.mean(dim=1)
        sig = el.std(dim=1)
        sig[..., 0].clamp_(min=cfg["min_steer"])
        sig[..., 1].clamp_(min=cfg["min_accel"])
    return best_act, best_traj, best_cost


# ======================================================================== #
# Head proposal prior — the v1 tactical head's immediate control as a seed  #
# ======================================================================== #
@torch.no_grad()
def head_action_seed(model, states, v0, K, ego=None):
    """Convert the frozen v1 tactical head's 0.5 s waypoint into a constant
    (steer,accel) seed via the harness pure-pursuit inverse — the single-mode
    v1 head reused as the planner's learned proposal prior (spec M3).

    ⛔ ``ego`` is the planner brains' ego port (E1, 2026-07-28); ``None`` is
    correct only for a checkpoint with no trained ``ego_emb`` and is refused
    otherwise by :func:`taniteval.ego_guard.assert_planner_ego`."""
    if getattr(model, "tactical_policy", None) is None:
        return None
    B = states.shape[0]
    nav = torch.zeros(B, dtype=torch.long, device=states.device)
    _eg.assert_planner_ego(model, ego, where="planner_p2.head_action_seed",
                           ego_source="observed pose at t and t-1")
    ctx = model.strategic_policy(states, nav, ego=ego)["ctx"]
    wp = model.tactical_policy(states, ctx, ego=ego)["waypoints"]
    steer, accel = cl.wp_to_control(wp[cl.LOOKAHEAD_STEP], v0)
    row = torch.stack([steer.clamp(-STEER_CLAMP, STEER_CLAMP),
                       accel.clamp(-ACCEL_CLAMP, ACCEL_CLAMP)], dim=-1)     # [B,2]
    return row[:, None, :].expand(B, K, 2).clone()


# ======================================================================== #
# Open-loop collection: planner vs head vs operative-rollout vs CV          #
# ======================================================================== #
@torch.no_grad()
def collect_openloop(model, step_readout, episodes, device, w=W, cfg=CEM,
                     window=WINDOW, stride=STRIDE, chunk=16, speed_input=True):
    """Every stride-window: CEM plan + the three baselines, all apples-to-apples.

    Returns dict of [N,K,2] full paths (planner/open_grnd/gt) + [N,4,2] waypoint
    sets (planner/open_grnd/cv/head/gt) + [N] meta (eid/speed/head_deg/v_target/
    vt_valid/plan_cost)."""
    wp_idx = torch.tensor(IDX, device=device)
    acc = {n: [] for n in
           ("plan_wp", "open_wp", "cv_wp", "head_wp", "gt_wp",
            "plan_full", "gt_full", "speed", "head_deg", "vt", "vt_valid",
            "plan_cost")}
    eid = []
    for ep in episodes:
        feats = ep.feats
        T = min(feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        starts = list(range(0, T - window - K_MAX, stride))
        for i in range(0, len(starts), chunk):
            ch = starts[i:i + chunk]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(feats[t:t + window])
                              for t in ch]).to(device)
            if fw.dtype == torch.uint8:
                fw = fw.float().div_(255.0)
            elif fw.dtype == torch.float16:
                fw = fw.float()
            aw = torch.stack([ep.actions[t:t + window] for t in ch]).to(device)
            fa_true = torch.stack([ep.actions[t + window:t + window + K_MAX]
                                   for t in ch]).to(device)
            v0 = ep.poses[last, 3].to(device).float()
            if speed_input:
                v0c = (v0 / SPEED_SCALE)[:, None, None]
                aw = torch.cat([aw, v0c.expand(-1, aw.shape[1], -1)], dim=-1)
                fa_true = torch.cat([fa_true,
                                     v0c.expand(-1, fa_true.shape[1], -1)], dim=-1)
            states = model.encode_window(fw)
            vt, vtv = vtarget_for(ep.poses, last)
            vt = vt.to(device)

            # --- planner: CEM over the frozen operative WM -------------------
            # ⛔ E1 (2026-07-28): None unless the ckpt owns trained ego weights.
            ego = (_eg.ego_from_poses(ep.poses, last, _eg.POSE_SCALE_DEFAULT,
                                      device)
                   if _eg.planner_ego_capability(model)["ego_input_on_planners"]
                   else None)
            hs = head_action_seed(model, states, v0, cfg["K"], ego=ego)
            _, plan_traj, plan_cost = cem_plan(
                model, step_readout, states, aw, vt, v0, w, cfg,
                head_seed=hs, plan_first=False)
            # --- operative rollout under TRUE actions (the gate rollout) -----
            open_traj, _ = rollout_decode(model.predictor, states, aw, fa_true,
                                          step_readout, K_MAX)
            # --- head tactical waypoints (the baseline being challenged) -----
            nav = torch.zeros(len(ch), dtype=torch.long, device=device)
            _eg.assert_planner_ego(model, ego, where="planner_p2.collect",
                                   ego_source="observed pose at t and t-1")
            ctx = model.strategic_policy(states, nav, ego=ego)["ctx"]
            hwp = model.tactical_policy(states, ctx, ego=ego)["waypoints"]
            head_wp = torch.stack([hwp[k] for k in WP_STEPS], dim=1)        # [b,4,2]

            gt_full = gt_ego_waypoints(ep.poses, last,
                                       tuple(range(1, K_MAX + 1))).to(device)
            acc["plan_wp"].append(plan_traj[:, wp_idx].cpu())
            acc["open_wp"].append(open_traj[:, wp_idx].cpu())
            acc["cv_wp"].append(
                baseline_waypoints(ep.poses, last)["constant_velocity"].cpu())
            acc["head_wp"].append(head_wp.cpu())
            acc["gt_wp"].append(gt_ego_waypoints(ep.poses, last).cpu())
            acc["plan_full"].append(plan_traj.cpu())
            acc["gt_full"].append(gt_full.cpu())
            acc["speed"].append(v0.cpu())
            acc["head_deg"].append(net_heading_change_deg(ep.poses, last))
            acc["vt"].append(vt.cpu())
            acc["vt_valid"].append(vtv)
            acc["plan_cost"].append(plan_cost.cpu())
            eid.extend([ep.episode_id] * len(ch))
    out = {n: torch.cat(v).float() for n, v in acc.items()}
    out["eid"] = eid
    return out


# ======================================================================== #
# Aggregation                                                               #
#                                                                           #
# ⛔ MIGRATED 2026-08-16. Every DECIDING number below is the episode-cluster #
# bootstrap over the val EPISODES (taniteval/ci.py); every arm-vs-arm delta  #
# on shared windows is the PAIRED form. The `_jack_*` pair is retained under #
# its true name for REPRODUCTION ONLY and its output is quarantined under    #
# `LEGACY_BLOCK`, never read by a verdict.                                   #
#                                                                           #
# The bug this closes: `G1_pass` was decided on `_jack_paired`, which is a   #
# PAIRED DELTA under `overlapping_holdout_se` — the exact statistic measured #
# on 2026-07-25 at up to x-4.15 INCLUDING A SIGN FLIP. `G4_pass` compared a  #
# mean-of-split-means against a threshold that was itself a mean-of-split-   #
# means. Re-decided on banked per-window data 2026-08-16: NEITHER FLIPS      #
# (JACK_IN_GATES.md), but the point estimates move -6.9 % to +5.9 % and the  #
# intervals were 1.20-2.17x too narrow.                                      #
# ======================================================================== #
def _ade2(pred, gt):
    """[N,4,2] pred vs GT -> [N] ADE over the 4 waypoints (to 2 s)."""
    return torch.linalg.norm(pred - gt, dim=-1).mean(dim=1)


def _eid_s(eids):
    """Episode ids as the strings ``ci.episode_index`` clusters on."""
    return [str(x) for x in eids]


def _interval(vals, eids, n_boot=None, seed=0):
    """Decision-grade single-arm interval: episode-cluster bootstrap.

    The point estimate is the **full_set** mean — the bootstrap supplies the
    interval and never moves the mean. That is the half of the correction that
    matters most here: `_jack_scalar`'s central value was a mean-of-SPLIT-means,
    so it was biased before any interval was drawn."""
    return _ci.episode_cluster_bootstrap(
        np.asarray(vals, dtype=float), _eid_s(eids),
        n_boot=N_BOOT if n_boot is None else n_boot, seed=seed)


def _paired(a, b, eids, n_boot=None, seed=0):
    """Decision-grade paired delta: `reduce(a) - reduce(b)` on the SAME episodes.

    ``mean`` is an ALIAS of ``delta`` (the delta IS a difference of means), kept
    so any consumer that printed the legacy ``d['mean'] ± d['ci95']`` reads the
    migrated block unchanged — the same aliasing `closedloop._paired` uses."""
    d = dict(_ci.paired_episode_cluster_bootstrap(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), _eid_s(eids),
        n_boot=N_BOOT if n_boot is None else n_boot, seed=seed))
    d["mean"] = d["delta"]
    return d


def _width_ratio(new, old):
    """new ci95 / legacy ci95 — the narrowing, RE-MEASURED in every artifact."""
    o = float(old.get("ci95", 0.0) or 0.0)
    n = float(new.get("ci95", 0.0) or 0.0)
    return round(n / o, 3) if o > 0 else None


def _point_shift_pct(new, old):
    """(legacy - corrected)/|corrected| in %, i.e. how wrong the legacy POINT
    estimate was. Signed, because the bias is bidirectional (11 arms inflated,
    16 deflated across the 27-arm blast radius) and a magnitude would hide that."""
    c = float(new.get("mean", new.get("delta", 0.0)))
    o = float(old.get("mean", 0.0))
    return round(100.0 * (o - c) / max(1e-12, abs(c)), 3) if c else None


def _jack_scalar(vals, eids, splits):
    """**DEPRECATED — `overlapping_holdout_se`. REPRODUCTION ONLY.**

    Mean-of-split-means over 8 OVERLAPPING random 20 % episode holdouts. Not a
    jackknife, not a valid SE, and it BIASES THE POINT ESTIMATE. ⛔ Never let
    this decide anything — `_interval` is the replacement. Output is self-
    labelling so a consumer that copies a number also copies its estimator."""
    v = np.asarray(vals, dtype=float)
    sm = np.asarray([float(np.nanmean(v[va])) for _t, va in splits if len(va)])
    mean = float(np.mean(sm))
    ci = float(_ci.overlapping_holdout_se(sm))
    return {"mean": round(mean, 4), "ci95": round(ci, 4), "n": int(v.size),
            "estimator": DEPRECATED_ESTIMATOR, "deprecated": True}


def _jack_paired(a, b, eids, splits):
    """**DEPRECATED — `overlapping_holdout_se` on a paired delta. REPRODUCTION
    ONLY.**

    ⛔ This is the single most dangerous form of the deprecated estimator: on
    paired deltas the 2026-07-25 blast radius measured errors up to **x-4.15,
    INCLUDING A SIGN FLIP**. It decided `G1_pass` until 2026-08-16. `_paired`
    is the replacement."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sm = np.asarray([float(np.nanmean(d[va])) for _t, va in splits if len(va)])
    mean = float(np.mean(sm))
    ci = float(_ci.overlapping_holdout_se(sm))
    return {"mean": round(mean, 4), "ci95": round(ci, 4),
            "separated": bool(abs(mean) - ci > 0),
            "estimator": DEPRECATED_ESTIMATOR, "deprecated": True}


def assert_no_deprecated_estimator(res):
    """Refuse to return a planner block whose DECIDING numbers are deprecated.

    Run before ``LEGACY_BLOCK`` is attached — that block exists precisely to
    carry the deprecated numbers and is the one documented exemption. Same
    single policy implementation as `driving`/`closedloop`, so a third copy of
    the rule cannot drift from the first two."""
    return _drv.assert_no_deprecated_estimator(
        {k: v for k, v in res.items() if k != LEGACY_BLOCK}, _path="planner_p2")


def analyze_openloop(col, n_splits=8, val_frac=0.2, n_boot=None, seed=0):
    eids = col["eid"]
    splits = [split_by_episode(eids, val_frac, s) for s in range(n_splits)]
    ade = {k: _ade2(col[f"{k}_wp"], col["gt_wp"]).numpy()
           for k in ("plan", "open", "cv", "head")}
    # DECIDING: full_set point estimate + episode-cluster bootstrap interval.
    boot = {k: _interval(ade[k], eids, n_boot, seed) for k in ade}
    # REPRODUCTION ONLY, quarantined under LEGACY_BLOCK at the bottom.
    heldout = {k: _jack_scalar(ade[k], eids, splits) for k in ade}

    # decoupled longitudinal / lateral of the planner path (pathspeed verbatim)
    sel_all = torch.arange(col["gt_full"].shape[0])
    mb_plan = ps.metric_block(col["plan_full"], col["gt_full"], sel_all)

    # straight vs curved stratification (where is the lateral-blind cost fine?)
    hd = col["head_deg"]
    straight = (hd < 5.0)
    curved = (hd >= torch.quantile(hd, 0.90))
    strat = {}
    for name, mask in (("straight_lt5deg", straight),
                       ("curved_top10pct", curved)):
        m = mask.nonzero(as_tuple=True)[0]
        if len(m) < 8:
            strat[name] = {"n": int(len(m))}
            continue
        strat[name] = {
            "n": int(len(m)),
            "plan_ade2s": round(float(_ade2(col["plan_wp"][m],
                                            col["gt_wp"][m]).mean()), 4),
            "open_ade2s": round(float(_ade2(col["open_wp"][m],
                                            col["gt_wp"][m]).mean()), 4),
            "cv_ade2s": round(float(_ade2(col["cv_wp"][m],
                                          col["gt_wp"][m]).mean()), 4),
            "head_ade2s": round(float(_ade2(col["head_wp"][m],
                                            col["gt_wp"][m]).mean()), 4),
        }

    # v_target quality: how well the planner tracks the minted target speed
    vt = col["vt"]
    v_plan = ps.step_speed(col["plan_full"])[:, -1]        # planned speed @2s
    v_gt = ps.step_speed(col["gt_full"])[:, -1]
    vt_block = {
        "vt_valid_frac": round(float(col["vt_valid"].mean()), 4),
        "vt_mean_mps": round(float(vt.mean()), 3),
        "gt_speed_at_last_mps": round(float(col["speed"].mean()), 3),
        "plan_speed_at_2s_mps": round(float(v_plan.mean()), 3),
        "gt_speed_at_2s_mps": round(float(v_gt.mean()), 3),
        "plan_vs_target_abs_mps": round(float((v_plan - vt).abs().mean()), 4),
        "gt_vs_target_abs_mps": round(float((v_gt - vt).abs().mean()), 4),
    }

    # ⛔ THE GATE. head - plan on the SAME windows -> PAIRED episode-cluster
    # bootstrap. Until 2026-08-16 this line was `_jack_paired`, whose measured
    # error on paired deltas reaches x-4.15 with sign flips.
    g1_delta = _paired(ade["head"], ade["plan"], eids, n_boot, seed)
    g1_legacy = _jack_paired(ade["head"], ade["plan"], eids, splits)
    res = {
        "n_windows": len(eids), "n_episodes": len(set(eids)),
        "ade2s": {
            "planner": boot["plan"],
            "operative_rollout_trueA": boot["open"],
            "constant_velocity": boot["cv"],
            "tactical_head": boot["head"],
        },
        "G1_head_minus_planner_ade2s": g1_delta,
        "G1_pass": bool(g1_delta["delta"] > 0 and g1_delta["separated"]),
        "planner_beats_cv": bool(boot["plan"]["mean"] < boot["cv"]["mean"]),
        "planner_vs_operative_gap_m": round(
            boot["plan"]["mean"] - boot["open"]["mean"], 4),
        "longitudinal_lateral": {
            "plan_long_rmse_2s_m": mb_plan["per_horizon"]["2s"]["long_rmse_m"],
            "plan_lat_rmse_2s_m": mb_plan["per_horizon"]["2s"]["lat_rmse_m"],
            "plan_long_frac_of_2s_sqerr":
                mb_plan["trajectory"]["long_frac_of_sqerr_2s"],
            "plan_speed_bias_mps": mb_plan["trajectory"]["speed_bias_mps"],
            "plan_path_geometry_crosstrack_rmse_m":
                mb_plan["trajectory"]["path_geometry_crosstrack_rmse_m"],
        },
        "straight_vs_curved": strat,
        "vtarget": vt_block,
        "cost_weights": dict(W),
        "estimator": {
            "interval": "episode_cluster_bootstrap",
            "delta": "paired_episode_cluster_bootstrap",
            "n_boot": N_BOOT if n_boot is None else int(n_boot),
            "seed": int(seed), "resampling_unit": "val episode",
            "deprecated_and_refused": DEPRECATED_ESTIMATOR,
            "legacy_block": LEGACY_BLOCK,
            "estimator_note": ESTIMATOR_NOTE},
    }
    assert_no_deprecated_estimator(res)
    # ---- QUARANTINE: the banned estimator, under its true name ------------- #
    #   Kept so every pre-2026-08-16 P2 number stays reproducible AND so the
    #   correction is RE-MEASURED in every artifact instead of cited from a doc.
    #   Never the headline; never read by G1_pass. The one guard exemption.
    res[LEGACY_BLOCK] = {
        "_what": "the pre-2026-08-16 open-loop numbers: mean-of-split-means "
                 "+- 1.96*std/sqrt(8) over 8 OVERLAPPING random 20% episode "
                 "holdouts (`overlapping_holdout_se`).",
        "_why_kept": "reproduction ONLY. ⛔ NOT admissible for any decision — "
                     "it biases the POINT ESTIMATE as well as the interval.",
        "_estimator": DEPRECATED_ESTIMATOR,
        "n_splits": n_splits, "val_frac": val_frac,
        "ade2s": {"planner": heldout["plan"],
                  "operative_rollout_trueA": heldout["open"],
                  "constant_velocity": heldout["cv"],
                  "tactical_head": heldout["head"]},
        "G1_head_minus_planner_ade2s": g1_legacy,
        "G1_pass_LEGACY": bool(g1_legacy["mean"] > 0 and g1_legacy["separated"]),
        "G1_verdict_flip_vs_decision_grade_LEGACY": bool(
            (g1_legacy["mean"] > 0 and g1_legacy["separated"])
            != res["G1_pass"]),
        "ci_width_ratio_new_over_legacy": {
            "_read": ">1 = the banned interval was too narrow BY THAT FACTOR",
            "planner": _width_ratio(boot["plan"], heldout["plan"]),
            "operative_rollout_trueA": _width_ratio(boot["open"], heldout["open"]),
            "constant_velocity": _width_ratio(boot["cv"], heldout["cv"]),
            "tactical_head": _width_ratio(boot["head"], heldout["head"]),
            "G1_delta": _width_ratio(g1_delta, g1_legacy)},
        "point_estimate_shift_pct_legacy_vs_corrected": {
            "_read": "(legacy - corrected)/|corrected| in %. Signed: the bias "
                     "is BIDIRECTIONAL, so a magnitude would hide the sign.",
            "planner": _point_shift_pct(boot["plan"], heldout["plan"]),
            "operative_rollout_trueA": _point_shift_pct(boot["open"], heldout["open"]),
            "constant_velocity": _point_shift_pct(boot["cv"], heldout["cv"]),
            "tactical_head": _point_shift_pct(boot["head"], heldout["head"]),
            "G1_delta": _point_shift_pct(g1_delta, g1_legacy)},
    }
    return res


# ======================================================================== #
# Closed-loop: swap the head PLAN step for CEM (G4, reuse cl harness)       #
# ======================================================================== #
@torch.no_grad()
def closed_loop_planner(model, step_readout, states0, aw, v0, v_target,
                        speed_input, w, cfg, replan_every=1, k=K_MAX):
    """Imagination-in-the-loop with a CEM planner. Same loop as
    closedloop.closed_loop_rollout (encode -> plan -> control -> imagine ->
    drive) but the PLAN is a CEM search over the frozen operative WM, executed
    action-first (a0 directly, no pure-pursuit inversion). Returns closed_bike
    [b,k,2] + executed steer/accel/speed."""
    b = states0.shape[0]
    win_s, win_a, v = states0.clone(), aw.clone(), v0.clone()
    steer_seq, accel_seq = [], []
    plan_act = None
    for tick in range(k):
        if tick % replan_every == 0:
            hs = head_action_seed(model, win_s, v, cfg["K"])
            plan_act, _, _ = cem_plan(model, step_readout, win_s, win_a,
                                      v_target, v, w, cfg, head_seed=hs,
                                      plan_first=True)                    # [b,K,2]
            step_in_plan = 0
        a = plan_act[:, step_in_plan]                                    # [b,2]
        step_in_plan += 1
        steer, accel = a[:, 0], a[:, 1]
        a_exec = cl.build_action(steer, accel, v, speed_input)           # [b,A]
        win_a_exec = win_a.clone()
        win_a_exec[:, -1] = a_exec
        z_next = model.predictor(win_s, win_a_exec)[1]
        steer_seq.append(steer)
        accel_seq.append(accel)
        v = (v + accel * DT).clamp_min(0.0)
        win_s = torch.cat([win_s[:, 1:], z_next.unsqueeze(1)], dim=1)
        win_a = torch.cat([win_a_exec[:, 1:], a_exec.unsqueeze(1)], dim=1)
    steer_seq = torch.stack(steer_seq, dim=1)
    accel_seq = torch.stack(accel_seq, dim=1)
    bike_pts, bike_spd = cl.bicycle_integrate(v0, steer_seq, accel_seq)
    return {"closed_bike": bike_pts, "steer": steer_seq, "accel": accel_seq,
            "speed": bike_spd}


@torch.no_grad()
def collect_closedloop(model, step_readout, episodes, device, w=W, cfg=CEM_CL,
                       window=WINDOW, stride=16, chunk=16, speed_input=True,
                       replan_every=1):
    """closed_bike (planner) + open_grnd (true actions) + cv + gt per window."""
    wp_idx = torch.tensor(IDX, device=device)
    acc = {n: [] for n in ("closed_bike", "open_grnd", "cv", "gt",
                           "speed", "head_deg")}
    eid = []
    for ep in episodes:
        feats = ep.feats
        T = min(feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        starts = list(range(0, T - window - K_MAX, stride))
        for i in range(0, len(starts), chunk):
            ch = starts[i:i + chunk]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(feats[t:t + window])
                              for t in ch]).to(device)
            if fw.dtype == torch.uint8:
                fw = fw.float().div_(255.0)
            elif fw.dtype == torch.float16:
                fw = fw.float()
            aw = torch.stack([ep.actions[t:t + window] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + window:t + window + K_MAX]
                              for t in ch]).to(device)
            v0 = ep.poses[last, 3].to(device).float()
            if speed_input:
                v0c = (v0 / SPEED_SCALE)[:, None, None]
                aw = torch.cat([aw, v0c.expand(-1, aw.shape[1], -1)], dim=-1)
                fa = torch.cat([fa, v0c.expand(-1, fa.shape[1], -1)], dim=-1)
            states0 = model.encode_window(fw)
            vt, _ = vtarget_for(ep.poses, last)
            vt = vt.to(device)
            open_wp, _ = rollout_decode(model.predictor, states0, aw, fa,
                                        step_readout, K_MAX)
            clp = closed_loop_planner(model, step_readout, states0, aw, v0, vt,
                                      speed_input, w, cfg, replan_every)
            acc["closed_bike"].append(clp["closed_bike"][:, wp_idx].cpu())
            acc["open_grnd"].append(open_wp[:, wp_idx].cpu())
            acc["cv"].append(
                baseline_waypoints(ep.poses, last)["constant_velocity"].cpu())
            acc["gt"].append(gt_ego_waypoints(ep.poses, last).cpu())
            acc["speed"].append(v0.cpu())
            acc["head_deg"].append(net_heading_change_deg(ep.poses, last))
            eid.extend([ep.episode_id] * len(ch))
    out = {n: torch.cat(v).float() for n, v in acc.items()}
    out["eid"] = eid
    return out


def analyze_closedloop(col, n_splits=8, val_frac=0.2, n_boot=None, seed=0):
    eids = col["eid"]
    splits = [split_by_episode(eids, val_frac, s) for s in range(n_splits)]
    gt = col["gt"]
    de = lambda k: torch.linalg.norm(col[k] - gt, dim=-1)     # [N,4]
    ade = {k: de(k).mean(dim=1).numpy() for k in ("closed_bike", "open_grnd", "cv")}
    fde = de("closed_bike")[:, -1].numpy()
    diverged = (de("closed_bike")[:, -1] > cl.DIVERGENCE_M).float().numpy()
    per_window = dict(ade, closed_bike_fde2s=fde, divergence_rate_gt5m=diverged)
    boot = {k: _interval(v, eids, n_boot, seed) for k, v in per_window.items()}
    heldout = {k: _jack_scalar(v, eids, splits) for k, v in per_window.items()}
    cb = boot["closed_bike"]
    res = {
        "n_windows": len(eids), "n_episodes": len(set(eids)),
        "closed_bike_ade2s": cb,
        "closed_bike_fde2s": boot["closed_bike_fde2s"],
        "open_grnd_ade2s": boot["open_grnd"],
        "cv_ade2s": boot["cv"],
        "divergence_rate_gt5m": boot["divergence_rate_gt5m"],
        "G4_head_baseline_ade2s": G4_HEAD_BASELINE_ADE2S,
        "G4_head_baseline_source": G4_HEAD_BASELINE_SOURCE,
        # ⛔ THE GATE, on the full_set point estimate vs the full_set threshold.
        "G4_pass": bool(cb["mean"] < G4_HEAD_BASELINE_ADE2S),
        # STRICTER, and the one to quote: the whole interval clears the bar.
        "G4_pass_ci_separated": bool(cb["hi"] < G4_HEAD_BASELINE_ADE2S),
        "_unpaired_warning":
            "the planner (stride 16, cl_episodes) and the head baseline "
            "(stride 8, 40 episodes) are DIFFERENT window sets, so this is an "
            "unpaired two-interval comparison. The planner's windows are the "
            "stride-16 subset of the baseline's; a PAIRED G4 on the aligned "
            "windows is in JACK_IN_GATES.md (delta -0.7375 [-0.9362, -0.5295], "
            "separated, n=221/20ep) and agrees.",
        "estimator": {
            "interval": "episode_cluster_bootstrap",
            "n_boot": N_BOOT if n_boot is None else int(n_boot),
            "seed": int(seed), "resampling_unit": "val episode",
            "deprecated_and_refused": DEPRECATED_ESTIMATOR,
            "legacy_block": LEGACY_BLOCK,
            "estimator_note": ESTIMATOR_NOTE},
    }
    assert_no_deprecated_estimator(res)
    res[LEGACY_BLOCK] = {
        "_what": "the pre-2026-08-16 closed-loop numbers under "
                 "`overlapping_holdout_se`, incl. the legacy G4 threshold "
                 "1.6852 which was ITSELF a mean-of-split-means.",
        "_why_kept": "reproduction ONLY. ⛔ NOT admissible for any decision.",
        "_estimator": DEPRECATED_ESTIMATOR,
        "n_splits": n_splits, "val_frac": val_frac,
        "closed_bike_ade2s": heldout["closed_bike"],
        "closed_bike_fde2s": heldout["closed_bike_fde2s"],
        "open_grnd_ade2s": heldout["open_grnd"],
        "cv_ade2s": heldout["cv"],
        "divergence_rate_gt5m": heldout["divergence_rate_gt5m"],
        "G4_head_baseline_ade2s_LEGACY": G4_HEAD_BASELINE_ADE2S_LEGACY,
        "G4_pass_LEGACY": bool(heldout["closed_bike"]["mean"]
                               < G4_HEAD_BASELINE_ADE2S_LEGACY),
        "G4_verdict_flip_vs_decision_grade_LEGACY": bool(
            (heldout["closed_bike"]["mean"] < G4_HEAD_BASELINE_ADE2S_LEGACY)
            != res["G4_pass"]),
        "ci_width_ratio_new_over_legacy": {
            k: _width_ratio(boot[k], heldout[k]) for k in boot},
        "point_estimate_shift_pct_legacy_vs_corrected": {
            k: _point_shift_pct(boot[k], heldout[k]) for k in boot},
    }
    return res


# ======================================================================== #
# Runner                                                                    #
# ======================================================================== #
def _load(arm, device):
    from taniteval import data, loaders
    from taniteval.registry import MODELS
    entry = [m for m in MODELS if m["key"] == arm][0]
    L = loaders.load(entry, device)
    files = data.list_val_episodes("/root/valdata/physicalai-val-0c5f7dac3b11",
                                   None)
    return entry, L, files, data


def run_and_save(arm="flagship-30k", device="cuda", episodes=40, cl_episodes=16,
                 closed_loop=False, sweep=False, replan_every=1,
                 out_dir="/root/taniteval/results"):
    import json
    from pathlib import Path
    entry, L, files, data = _load(arm, device)
    model, step_readout = L["model"], L["step_readout"]
    speed_input = bool(entry.get("speed_input"))
    t0 = time.time()

    files_ol = files[:episodes]
    eps = data.load_frames(files_ol)
    print(f"[p2] {arm}: open-loop CEM over {len(eps)} eps "
          f"(N={CEM['N']} iters={CEM['iters']} elite={CEM['elite']})...",
          flush=True)
    col = collect_openloop(model, step_readout, eps, device, w=W, cfg=CEM,
                           speed_input=speed_input)
    ol = analyze_openloop(col)
    print(f"[p2] {arm}: planner ade@2s={ol['ade2s']['planner']['mean']:.3f}"
          f"±{ol['ade2s']['planner']['ci95']:.3f} | head="
          f"{ol['ade2s']['tactical_head']['mean']:.3f} operative="
          f"{ol['ade2s']['operative_rollout_trueA']['mean']:.3f} cv="
          f"{ol['ade2s']['constant_velocity']['mean']:.3f} | G1_pass="
          f"{ol['G1_pass']} ({round(time.time()-t0,1)}s)", flush=True)

    res = {"arm": arm, "ckpt_step": L["step"],
           "protocol": {"episodes": episodes, "window": WINDOW, "stride": STRIDE,
                        "K": K_MAX, "hz": 10, "cem": CEM,
                        "ci": "episode_cluster_bootstrap over the val EPISODES "
                              "(paired form for the G1 delta) — taniteval/ci.py. "
                              "The legacy overlapping_holdout_se numbers are "
                              f"quarantined under `{LEGACY_BLOCK}`.",
                        "baselines_source": "closedloop_flagship-30k.json + "
                        "plan_flagship-30k.json (same harness)"},
        "open_loop": ol}

    if sweep:
        res["weight_sensitivity"] = _sweep(model, step_readout, eps, device,
                                           speed_input)

    if closed_loop:
        files_cl = files[:cl_episodes]
        eps_cl = data.load_frames(files_cl)
        t1 = time.time()
        print(f"[p2] {arm}: closed-loop CEM-in-the-loop over {len(eps_cl)} eps "
              f"(replan_every={replan_every})...", flush=True)
        colc = collect_closedloop(model, step_readout, eps_cl, device, w=W,
                                  cfg=CEM_CL, speed_input=speed_input,
                                  replan_every=replan_every)
        clr = analyze_closedloop(colc)
        res["closed_loop"] = clr
        print(f"[p2] {arm}: closed_bike ade@2s="
              f"{clr['closed_bike_ade2s']['mean']:.3f}"
              f"±{clr['closed_bike_ade2s']['ci95']:.3f} | head_baseline="
              f"{G4_HEAD_BASELINE_ADE2S} "
              f"| G4_pass={clr['G4_pass']} diverge="
              f"{clr['divergence_rate_gt5m']['mean']:.1%} "
              f"({round(time.time()-t1,1)}s)", flush=True)

    res["wall_s"] = round(time.time() - t0, 1)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    outp = Path(out_dir) / f"planner_p2_{arm}.json"
    outp.write_text(json.dumps(res, indent=2, default=str))
    print(f"[p2] {arm}: wrote {outp}", flush=True)
    return res


def _sweep(model, step_readout, eps, device, speed_input):
    """Weight sensitivity: vary w_c, w_p x{0.5,1,2} (NOT selecting on GT ADE)."""
    out = []
    for fc in (0.5, 1.0, 2.0):
        for fp in (0.5, 1.0, 2.0):
            w = dict(v=W["v"], c=W["c"] * fc, s=W["s"], p=W["p"] * fp)
            col = collect_openloop(model, step_readout, eps[:12], device, w=w,
                                   cfg=CEM, speed_input=speed_input)
            a = _ade2(col["plan_wp"], col["gt_wp"]).mean().item()
            out.append({"w_c": round(w["c"], 4), "w_p": round(w["p"], 4),
                        "planner_ade2s": round(a, 4)})
            print(f"[p2:sweep] w_c={w['c']:.3f} w_p={w['p']:.3f} "
                  f"-> ade2s={a:.3f}", flush=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser("taniteval.planner_p2")
    ap.add_argument("--arm", default="flagship-30k")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--cl-episodes", type=int, default=16)
    ap.add_argument("--closed-loop", action="store_true")
    ap.add_argument("--replan-every", type=int, default=1)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run_and_save(a.arm, a.device, a.episodes, a.cl_episodes, a.closed_loop,
                 a.sweep, a.replan_every)


if __name__ == "__main__":
    main()
