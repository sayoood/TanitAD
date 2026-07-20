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
  * gates.split_by_episode + bench CI protocol (8-split episode jackknife).

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

sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
sys.path.insert(0, "/root/taniteval")

from driving_diagnostic import (WP_STEPS, baseline_waypoints,  # noqa: E402
                                gt_ego_waypoints, net_heading_change_deg)
from tanitad.eval.gates import split_by_episode  # noqa: E402
from tanitad.models.metric_dynamics import rollout_decode  # noqa: E402
from taniteval import closedloop as cl  # noqa: E402
from taniteval import pathspeed as ps  # noqa: E402

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
def head_action_seed(model, states, v0, K):
    """Convert the frozen v1 tactical head's 0.5 s waypoint into a constant
    (steer,accel) seed via the harness pure-pursuit inverse — the single-mode
    v1 head reused as the planner's learned proposal prior (spec M3)."""
    if getattr(model, "tactical_policy", None) is None:
        return None
    B = states.shape[0]
    nav = torch.zeros(B, dtype=torch.long, device=states.device)
    ctx = model.strategic_policy(states, nav)["ctx"]
    wp = model.tactical_policy(states, ctx)["waypoints"]
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
            hs = head_action_seed(model, states, v0, cfg["K"])
            _, plan_traj, plan_cost = cem_plan(
                model, step_readout, states, aw, vt, v0, w, cfg,
                head_seed=hs, plan_first=False)
            # --- operative rollout under TRUE actions (the gate rollout) -----
            open_traj, _ = rollout_decode(model.predictor, states, aw, fa_true,
                                          step_readout, K_MAX)
            # --- head tactical waypoints (the baseline being challenged) -----
            nav = torch.zeros(len(ch), dtype=torch.long, device=device)
            ctx = model.strategic_policy(states, nav)["ctx"]
            hwp = model.tactical_policy(states, ctx)["waypoints"]
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
# Open-loop aggregation (8-split episode jackknife, bench CI protocol)      #
# ======================================================================== #
def _ade2(pred, gt):
    """[N,4,2] pred vs GT -> [N] ADE over the 4 waypoints (to 2 s)."""
    return torch.linalg.norm(pred - gt, dim=-1).mean(dim=1)


def _jack_scalar(vals, eids, splits):
    v = np.asarray(vals, dtype=float)
    sm = np.asarray([float(np.nanmean(v[va])) for _t, va in splits if len(va)])
    mean = float(np.mean(sm))
    ci = float(1.96 * np.std(sm) / max(1, len(sm)) ** 0.5)
    return {"mean": round(mean, 4), "ci95": round(ci, 4), "n": int(v.size)}


def _jack_paired(a, b, eids, splits):
    """Jackknifed mean of (a-b) per window + CI-separation from 0."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sm = np.asarray([float(np.nanmean(d[va])) for _t, va in splits if len(va)])
    mean = float(np.mean(sm))
    ci = float(1.96 * np.std(sm) / max(1, len(sm)) ** 0.5)
    return {"mean": round(mean, 4), "ci95": round(ci, 4),
            "separated": bool(abs(mean) - ci > 0)}


def analyze_openloop(col, n_splits=8, val_frac=0.2):
    eids = col["eid"]
    splits = [split_by_episode(eids, val_frac, s) for s in range(n_splits)]
    ade = {k: _ade2(col[f"{k}_wp"], col["gt_wp"]).numpy()
           for k in ("plan", "open", "cv", "head")}
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

    g1_delta = _jack_paired(ade["head"], ade["plan"], eids, splits)   # head - plan
    return {
        "n_windows": len(eids), "n_episodes": len(set(eids)),
        "ade2s": {
            "planner": heldout["plan"],
            "operative_rollout_trueA": heldout["open"],
            "constant_velocity": heldout["cv"],
            "tactical_head": heldout["head"],
        },
        "G1_head_minus_planner_ade2s": g1_delta,
        "G1_pass": bool(g1_delta["mean"] > 0 and g1_delta["separated"]),
        "planner_beats_cv": bool(heldout["plan"]["mean"] < heldout["cv"]["mean"]),
        "planner_vs_operative_gap_m": round(
            heldout["plan"]["mean"] - heldout["open"]["mean"], 4),
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
    }


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


def analyze_closedloop(col, n_splits=8, val_frac=0.2):
    eids = col["eid"]
    splits = [split_by_episode(eids, val_frac, s) for s in range(n_splits)]
    gt = col["gt"]
    de = lambda k: torch.linalg.norm(col[k] - gt, dim=-1)     # [N,4]
    ade = {k: de(k).mean(dim=1).numpy() for k in ("closed_bike", "open_grnd", "cv")}
    fde = de("closed_bike")[:, -1]
    diverged = (fde > cl.DIVERGENCE_M).float().numpy()
    heldout = {k: _jack_scalar(ade[k], eids, splits) for k in ade}
    return {
        "n_windows": len(eids), "n_episodes": len(set(eids)),
        "closed_bike_ade2s": heldout["closed_bike"],
        "closed_bike_fde2s": _jack_scalar(fde.numpy(), eids, splits),
        "open_grnd_ade2s": heldout["open_grnd"],
        "cv_ade2s": heldout["cv"],
        "divergence_rate_gt5m": _jack_scalar(diverged, eids, splits),
        "G4_head_baseline_ade2s": 1.6852,
        "G4_pass": bool(heldout["closed_bike"]["mean"] < 1.6852),
    }


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
                        "ci": "8-split episode jackknife",
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
              f"±{clr['closed_bike_ade2s']['ci95']:.3f} | head_baseline=1.685 "
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
