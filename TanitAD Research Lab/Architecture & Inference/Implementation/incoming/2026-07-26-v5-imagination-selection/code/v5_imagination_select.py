#!/usr/bin/env python3
"""E-V5-1 — IMAGINATION-SCORED SELECTION over the frozen v4 fan.

Pre-registration: ``V5_IMAGINATION_SELECTION.md`` §0 (staged BEFORE this ran).

THE HYPOTHESIS
--------------
Bar A proved a *discriminative* re-scorer over the existing latent features cannot
reach v1 even IN-SAMPLE (ceiling 0.4907 vs v1's 0.4271).  It did NOT test a
*simulative* scorer.  This script scores each of the 256 frozen fan candidates by
ROLLING IT FORWARD THROUGH THE WORLD MODEL and reading the imagined consequence.

Nothing is trained in the parameter-free arms (A0..A3, C1, C2, R, O).  Only A4
(the weighted combination) has free parameters, and it is reported BOTH in-sample
(comparable to Bar A's in-sample ceiling) and 5-fold episode-disjoint out-of-fold
(the deployable number).

THE INSTRUMENT
--------------
``tanitad.models.metric_dynamics.rollout_decode`` — reused verbatim, not rebuilt.
It advances the window with the model's OWN predicted latent (no new frame is
ever encoded), so ``k`` free steps is a genuine imagination roll-out.

CANDIDATE -> ACTION
-------------------
The fan is trajectories, the predictor eats actions.  ``traj_to_actions`` is the
exact inverse of the corpus's own action definition (``physicalai.py``:
WHEELBASE=2.9, steer=arctan(WHEELBASE*curvature), accel=d v/dt), so no new
convention is introduced.  Its fidelity is a pre-registered self-test (S3/S4).

Usage:
    python3 v5_imagination_select.py --wm v4          # scorer = v4's own WM
    python3 v5_imagination_select.py --wm v1          # scorer = v1's WM (PI dir. 1)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# provenance FIRST: six copies of eval_flagship_v4.py live on this box.        #
# --------------------------------------------------------------------------- #
STACK = os.environ.get("V5_STACK", "/root/v4eval/stack")
TANITEVAL = os.environ.get("V5_TANITEVAL", "/root/taniteval")
sys.path.insert(0, STACK)
sys.path.insert(0, str(Path(STACK) / "scripts"))
sys.path.insert(0, TANITEVAL)

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- pre-registered constants ---------------------------------------------
WHEELBASE = 2.9                 # physicalai.py:51 -- the corpus's own value
SPEED_SCALE = 10.0              # hard contract with the v1 trunk
MAX_STEER_RAD = 0.6             # cosmos_drive.py clip; guards atan blowup at v~0
DT = 0.1
K_MAX = 20                      # 2 s dense
WP_STEPS = (5, 10, 15, 20)      # the TanitEval 4 waypoints
V_FLOOR = 0.5                   # m/s floor in the curvature divide (v_safe idiom)
N_BOOT = 2000
SEED = 20260726

# ---- the committed numbers this experiment is judged against --------------
COMMITTED = {
    "v4_as_trained_ade_0_2s": 0.8563,      # BOOST_PROGRAM 3.2 / bar_a_produced.json
    "v4_oracle_in_fan_4wp": 0.2505,
    "v4_wm_canary_ade_2s": 1.1409,         # BOOST_PROGRAM 3.3 (Bar B)
    "bar_a_in_sample_ceiling_ce": 0.4907,  # THE BAR for CONFIRM
    "bar_a_in_sample_ceiling_regret": 0.5224,
    "v1_reference_40ep": 0.4271,           # THE BAR for STRONG (40-ep deployment)
    "bar_a_random_pick": 15.3622,          # the failing-input reference
    "wm_canary_bar": 0.55,
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def mod_prov(names):
    out = {}
    for n in names:
        m = sys.modules.get(n)
        if m is not None and getattr(m, "__file__", None):
            out[n] = {"path": m.__file__, "md5": md5(m.__file__)}
    return out


# ===========================================================================
# STAGE 0 -- candidate trajectory  ->  action sequence
# ===========================================================================
def traj_to_actions(traj: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """``traj`` [..., K, 2] dense ego waypoints @ DT -> actions [..., K, 3].

    The EXACT inverse of ``tanitad/data/physicalai.py``'s action definition:
    ``steer = arctan(WHEELBASE * curvature)``, ``accel = dv/dt``, 3rd channel
    ``v0 / SPEED_SCALE`` held constant (the leakage-safe v1 speed contract).

    ``v0`` [...] is the OBSERVED ego speed at t=0 and seeds ``v_0`` so the first
    accel is a real accel and not a step from zero.
    """
    lead = traj.shape[:-2]
    k = traj.shape[-2]
    zero = traj.new_zeros(*lead, 1, 2)
    p = torch.cat([zero, traj], dim=-2)                     # [..., K+1, 2]
    d = p[..., 1:, :] - p[..., :-1, :]                      # [..., K, 2]
    seg = d.norm(dim=-1)                                    # [..., K]
    v = seg / DT

    moving = seg > 1e-3
    psi_raw = torch.atan2(d[..., 1], d[..., 0])             # [..., K]
    # hold the previous heading through (near-)stationary segments; psi_0 = 0
    psi = torch.zeros_like(psi_raw)
    prev = torch.zeros_like(psi_raw[..., 0])
    cols = []
    for j in range(k):
        cur = torch.where(moving[..., j], psi_raw[..., j], prev)
        cols.append(cur)
        prev = cur
    psi = torch.stack(cols, dim=-1)

    psi_prev = torch.cat([torch.zeros_like(psi[..., :1]), psi[..., :-1]], dim=-1)
    dpsi = psi - psi_prev
    dpsi = (dpsi + math.pi) % (2 * math.pi) - math.pi        # wrap
    omega = dpsi / DT
    kappa = omega / v.clamp_min(V_FLOOR)
    steer = torch.atan(WHEELBASE * kappa).clamp(-MAX_STEER_RAD, MAX_STEER_RAD)

    v_prev = torch.cat([v0.unsqueeze(-1).expand(*lead, 1), v[..., :-1]], dim=-1)
    accel = (v - v_prev) / DT

    vch = (v0 / SPEED_SCALE).unsqueeze(-1).expand(*lead, k)
    return torch.stack([steer, accel, vch], dim=-1)          # [..., K, 3]


# ===========================================================================
# STAGE 1 -- the imagination roll-out, per candidate
# ===========================================================================
@torch.no_grad()
def imagine(predictor, step_readout, states: torch.Tensor,
            win_actions: torch.Tensor, cand_actions: torch.Tensor,
            k: int = K_MAX) -> torch.Tensor:
    """Roll the WM forward under each candidate's OWN action sequence.

    ``states`` [B, W, S], ``win_actions`` [B, W, 3], ``cand_actions`` [B, K, 3].
    Transition 1 is driven by ``win_a[:, -1]``, so the candidate takes control of
    it by OVERWRITING the last window action with ``a_c[0]``; ``a_c[1:]`` then
    drives transitions 2..K.  The candidate therefore controls the whole roll-out.
    Returns imagined ego waypoints [B, K, 2].  Reuses ``rollout_decode`` verbatim.
    """
    from tanitad.models.metric_dynamics import rollout_decode
    wa = win_actions.clone()
    wa[:, -1] = cand_actions[:, 0]
    wp, _ = rollout_decode(predictor, states, wa, cand_actions[:, 1:],
                           step_readout, k)
    return wp


# ===========================================================================
# STAGE 2 -- the pre-registered scoring rules.  LOWER cost = better candidate.
# ===========================================================================
def cost_A1_consistency(imag, fan, **_):
    """A1 (PRIMARY) -- does the WM agree that executing this plan yields it?"""
    return (imag - fan).norm(dim=-1).mean(dim=-1)                 # [B, N]


def cost_A2_goal_speed(imag, fan, v0=None, vt_speed=None, sel_accel_max=None,
                       **_):
    """A2 -- imagined terminal speed vs the PRODUCED target speed, clamped to the
    same reachable set ``FlagshipV15Head.select`` uses."""
    reach = sel_accel_max * K_MAX * DT
    v_goal = torch.max(torch.min(vt_speed, v0 + reach), (v0 - reach).clamp_min(0))
    v_term = (imag[..., -1, :] - imag[..., -2, :]).norm(dim=-1) / DT   # [B, N]
    return (v_term - v_goal[:, None]).abs()


def cost_A3_kinematic(imag, fan, **_):
    """A3 -- plausibility of the IMAGINED outcome: mean |2nd difference|, i.e.
    the imagined per-step acceleration magnitude."""
    d1 = imag[..., 1:, :] - imag[..., :-1, :]
    d2 = d1[..., 1:, :] - d1[..., :-1, :]
    return d2.norm(dim=-1).mean(dim=-1) / (DT * DT)


def cost_C1_ctrv(fan, v0=None, ctrv=None, **_):
    """C1 CONTROL (no WM) -- A1 with the world model replaced by a CTRV forward
    model rolled under the candidate's own actions.  If C1 matches A1 the world
    model contributed nothing."""
    return (ctrv - fan).norm(dim=-1).mean(dim=-1)


def cost_C2_ref(fan, imag_ref=None, **_):
    """C2 CONTROL (ONE imagination) -- distance to the WM roll-out under the
    OBSERVED (zero-order-hold) action.  No per-candidate rolling.  If C2 matches
    A1, per-candidate imagination adds nothing over one reference trajectory."""
    return (fan - imag_ref[:, None]).norm(dim=-1).mean(dim=-1)


def ctrv_rollout(v0: torch.Tensor, actions: torch.Tensor,
                 k: int = K_MAX) -> torch.Tensor:
    """Analytic bicycle/CTRV forward model under ``actions`` [B, N, K, 3].
    The WM-free control's simulator.  Returns [B, N, K, 2] ego waypoints."""
    steer, accel = actions[..., 0], actions[..., 1]
    b, n = steer.shape[0], steer.shape[1]
    x = steer.new_zeros(b, n)
    y = steer.new_zeros(b, n)
    psi = steer.new_zeros(b, n)
    v = v0[:, None].expand(b, n).clone()
    out = []
    for j in range(k):
        v = (v + accel[..., j] * DT).clamp_min(0.0)
        psi = psi + v / WHEELBASE * torch.tan(steer[..., j]) * DT
        x = x + v * torch.cos(psi) * DT
        y = y + v * torch.sin(psi) * DT
        out.append(torch.stack([x, y], dim=-1))
    return torch.stack(out, dim=-2)


# ===========================================================================
# STAGE 3 -- metrics (identical definitions to bar_a_selector.py)
# ===========================================================================
def wp4(traj, horizons, wp_steps=WP_STEPS):
    pos = [list(horizons).index(k) for k in wp_steps]
    return traj[:, pos]


def metrics_from_traj(traj, tgt, horizons):
    """4-wp ade_0_2s, miss@2m, dense along/cross split -- byte-identical
    definitions to Bar A's ``metrics_from_traj``."""
    p4, g4 = wp4(traj, horizons), wp4(tgt, horizons)
    d = (p4 - g4).norm(dim=-1)
    ade = d.mean(dim=1)
    miss = (d[:, -1] > 2.0).float()
    r = traj - tgt
    along = r[..., 0].abs().mean(dim=1)
    cross = r[..., 1].abs().mean(dim=1)
    return (ade.cpu().numpy(), miss.cpu().numpy(),
            along.cpu().numpy(), cross.cpu().numpy())


def pick_metrics(fan, idx, tgt, horizons):
    traj = fan[torch.arange(fan.shape[0], device=fan.device), idx]
    return metrics_from_traj(traj, tgt, horizons)


# ===========================================================================
# STAGE 4 -- intervals.  paired episode-cluster bootstrap ONLY.
# ===========================================================================
def _cluster_boot(vals_list, ep, n_boot=N_BOOT, seed=SEED):
    """Shared-resample episode-cluster bootstrap over several aligned arrays."""
    rng = np.random.default_rng(seed)
    eps = np.unique(ep)
    idx_by_ep = [np.where(ep == e)[0] for e in eps]
    draws = [[] for _ in vals_list]
    for _ in range(n_boot):
        pick = rng.integers(0, len(eps), len(eps))
        sel = np.concatenate([idx_by_ep[j] for j in pick])
        for i, v in enumerate(vals_list):
            draws[i].append(float(v[sel].mean()))
    return [np.asarray(d) for d in draws]


def single_ci(vals, ep, n_boot=N_BOOT, seed=SEED):
    (d,) = _cluster_boot([vals], ep, n_boot, seed)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": round(float(vals.mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "se": round(float(d.std(ddof=1)), 4),
            "n_windows": int(len(vals)), "n_episodes": int(len(np.unique(ep))),
            "n_boot": n_boot, "estimator": "episode_cluster_bootstrap",
            "reducer": "mean"}


def paired_ci(a, b, ep, n_boot=N_BOOT, seed=SEED):
    """Paired episode-cluster bootstrap of (a - b) on IDENTICAL windows.
    NEGATIVE = ``a`` is better.  NEVER overlapping_holdout_se."""
    (d,) = _cluster_boot([a - b], ep, n_boot, seed)
    lo, hi = np.percentile(d, [2.5, 97.5])
    delta = float((a - b).mean())
    return {"delta": round(delta, 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "ci95": round(float((hi - lo) / 2), 4),
            "p_delta_gt0": round(float((d > 0).mean()), 4),
            "separated": bool(lo > 0 or hi < 0),
            "n_windows": int(len(a)), "n_episodes": int(len(np.unique(ep))),
            "n_boot": n_boot, "estimator": "paired_episode_cluster_bootstrap",
            "_orientation": "a - b; NEGATIVE = a is BETTER"}


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="v4", choices=("v4", "v1"),
                    help="which world model SCORES the (always v4) fan")
    ap.add_argument("--v4-ckpt",
                    default="/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt")
    ap.add_argument("--v4-hcfg",
                    default="/workspace/_v4gate/flagship-v4-fromscratch-30k/config.json")
    ap.add_argument("--anchors", default="/root/models/flagship-v4-fromscratch-15k"
                                         "/flagship_v4_anchors_dense.pt")
    ap.add_argument("--val", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--v1-ckpt", default="/root/models/flagship-30k/ckpt.pt")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--goal-mode", default="produced",
                    choices=("produced", "oracle", "neutral"))
    ap.add_argument("--cache-batch", type=int, default=16)
    ap.add_argument("--imag-windows", type=int, default=4,
                    help="windows per imagination batch (x256 candidates)")
    ap.add_argument("--out", default="v5_imagination")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_start = time.time()
    R: dict = {
        "_experiment": "E-V5-1 -- IMAGINATION-SCORED SELECTION over the frozen "
                       "v4 fan (PI v5 synthesis, after Bar A REFUTED the "
                       "discriminative selector)",
        "_evidence_class": "MEASURED (ours)",
        "_scoring_wm": a.wm,
        "_goal_mode": a.goal_mode,
        "_primary_surface": "produced (deployable)" if a.goal_mode == "produced"
                            else a.goal_mode,
        "_estimator": "paired_episode_cluster_bootstrap (B=2000, unit = episode "
                      "cluster). NEVER overlapping_holdout_se.",
        "_committed_bars": COMMITTED,
        "_host": platform.node(),
        "_python": platform.python_version(),
        "_torch": torch.__version__,
        "_gpu": (torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"),
        "_stack_root": STACK,
        "_preregistration": "V5_IMAGINATION_SELECTION.md sec.0 (staged first)",
    }
    print(f"[v5] host={R['_host']} py={R['_python']} torch={R['_torch']} "
          f"gpu={R['_gpu']} stack={STACK}", flush=True)

    # ---- load the v4 arm (the FAN is always v4's) --------------------------
    import eval_flagship_v4 as E
    import goal_modes
    from tanitad.models.flagship_v15 import SPEED_SCALE as _SS
    assert _SS == SPEED_SCALE, f"SPEED_SCALE contract broken: {_SS}"

    cfg_e = E._eval_cfg()
    plan = E._plan(cfg_e)
    ds_val = E.build_val_dataset_v4(a.val, cfg_e, plan)
    ck4 = torch.load(a.v4_ckpt, map_location="cpu", weights_only=False)
    R["_ckpt_v4"] = {"path": a.v4_ckpt, "md5": md5(a.v4_ckpt),
                     "step": int(ck4.get("step", -1))}
    world4, grounding4, head, step4, hcfg, goal_head = E.load_v4_from_ck(
        ck4, DEV, head_config_path=a.v4_hcfg, anchors_dense_path=a.anchors)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    horizons = head.cfg.horizons
    del ck4

    # ---- the SCORING world model ------------------------------------------
    feas: dict = {"requested": a.wm}
    if a.wm == "v4":
        world_s, grounding_s = world4, grounding4
        feas["verdict"] = "N/A (v4 scores its own fan)"
    else:
        ck1 = torch.load(a.v1_ckpt, map_location="cpu", weights_only=False)
        R["_ckpt_v1"] = {"path": a.v1_ckpt, "md5": md5(a.v1_ckpt),
                         "step": int(ck1.get("step", -1))}
        world_s, grounding_s, step1 = E.load_v1_from_ck(ck1, DEV)
        del ck1
        feas.update({
            "v1_state_dim": int(world_s.state_dim),
            "v4_state_dim": int(world4.state_dim),
            "v1_pred_window": int(world_s.predictor.cfg.window),
            "v4_pred_window": int(world4.predictor.cfg.window),
            "v1_action_dim": int(world_s.predictor.cfg.action_dim),
            "v4_action_dim": int(world4.predictor.cfg.action_dim),
            "v1_step": step1,
            "_read": "the fan is METRIC trajectories, so a foreign WM only needs "
                     "to (a) consume this dataset's frames and (b) emit metres. "
                     "state_dim need NOT match v4's.",
        })
        ok = (feas["v1_pred_window"] == feas["v4_pred_window"]
              and feas["v1_action_dim"] == feas["v4_action_dim"])
        feas["verdict"] = "FEASIBLE" if ok else "INFEASIBLE"
        if not ok:
            feas["reason"] = ("predictor window or action_dim differ -- the same "
                              "frame window / action tensor cannot drive both")
        print(f"[v5] v1-scores-v4-fan feasibility: {feas['verdict']} {feas}",
              flush=True)
        if not ok:
            R["feasibility_v1_scores_v4_fan"] = feas
            Path(f"{a.out}_{a.wm}{a.tag}.json").write_text(json.dumps(R, indent=2))
            return
    R["feasibility_v1_scores_v4_fan"] = feas
    step_readout = grounding_s.step["op"]

    R["_module_provenance"] = mod_prov([
        "eval_flagship_v4", "train_flagship_v4", "goal_modes",
        "tanitad.models.flagship_v4", "tanitad.models.flagship_v15",
        "tanitad.models.metric_dynamics", "tanitad.models.fourbrain",
        "tanitad.refs.refc", "flagship_v4_data"])

    # =======================================================================
    # STAGE A -- build the eval cache (881 windows) + the fan
    # =======================================================================
    import refb_labels
    from torch.utils.data import default_collate
    from train_flagship_v4 import _to_device
    from tanitad.models.metric_dynamics import gt_ego_waypoints, rollout_decode

    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < a.episodes and t % a.stride == 0]
    print(f"[v5] {len(sel)} eval windows (episodes<{a.episodes}, "
          f"stride {a.stride})", flush=True)

    C = {k: [] for k in ("fan", "tgt", "v0", "vt_speed", "vt_keep", "ep", "t",
                         "ref_sel_idx", "ref_ade4", "states", "win_act",
                         "fut_act", "canary_wp", "gt4")}
    real_condition = head.condition
    stash: dict = {}

    def wrapped_condition(*x, **kw):
        m, tele, vt_keep = real_condition(*x, **kw)
        stash["vt_keep"] = vt_keep
        return m, tele, vt_keep

    head.condition = wrapped_condition                       # type: ignore
    t0 = time.time()
    try:
        for b0 in range(0, len(sel), a.cache_batch):
            idx = sel[b0:b0 + a.cache_batch]
            b = _to_device(default_collate([ds_val[i] for i in idx]), DEV)
            pl = b["pose_last"].float()
            v0 = pl[:, 3]
            fp = b["future_poses"].float()
            tgt = refb_labels.waypoint_targets(pl, fp[:, :max(horizons)], horizons)
            with torch.no_grad():
                st4 = world4.encode_window(b["frames"])
                goal_kw, _rec = goal_modes.resolve_goal(
                    a.goal_mode, head=head, batch=b, v0=v0, states=st4,
                    goal_head=goal_head, allow_fallback=False)
                out = head(st4, v0, lambda_plan=1.0, **goal_kw)
                # the SCORING model's own encoding + its true-action canary
                st_s = (st4 if a.wm == "v4" else world_s.encode_window(b["frames"]))
                aw2 = b["actions"].float()
                fa2 = b["future_actions"].float()
                vch = (v0 / SPEED_SCALE)[:, None, None]
                aw = torch.cat([aw2, vch.expand(-1, aw2.shape[1], -1)], dim=-1)
                fa = torch.cat([fa2, vch.expand(-1, fa2.shape[1], -1)], dim=-1)
                can_wp, _ = rollout_decode(world_s.predictor, st_s, aw, fa,
                                           step_readout, K_MAX)
            C["fan"].append(out["anchor_traj"].float().cpu())
            C["tgt"].append(tgt.float().cpu())
            C["ref_sel_idx"].append(out["sel_idx"].cpu())
            _p, _g = wp4(out["traj"], horizons), wp4(tgt, horizons)
            C["ref_ade4"].append((_p - _g).norm(dim=-1).mean(dim=1).float().cpu())
            C["v0"].append(v0.cpu())
            vts = goal_kw.get("vt_speed")
            C["vt_speed"].append((vts.float() if vts is not None else v0).cpu())
            vk = stash.get("vt_keep")
            C["vt_keep"].append(vk.cpu() if vk is not None
                                else torch.ones_like(v0, dtype=torch.bool).cpu())
            C["states"].append(st_s.float().cpu())
            C["win_act"].append(aw.float().cpu())
            C["fut_act"].append(fa.float().cpu())
            C["canary_wp"].append(can_wp.float().cpu())
            C["gt4"].append(gt_ego_waypoints(pl, fp, WP_STEPS).float().cpu())
            for i in idx:
                e_i, tt = ds_val.index[i]
                C["ep"].append(int(e_i))
                C["t"].append(int(tt))
            if b0 % (a.cache_batch * 10) == 0:
                print(f"  [cache] {b0 + len(idx)}/{len(sel)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        head.condition = real_condition                      # type: ignore

    C = {k: (torch.cat(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v))
         for k, v in C.items()}
    R["_cache"] = {"n_windows": int(C["fan"].shape[0]),
                   "n_candidates": int(C["fan"].shape[1]),
                   "n_episodes": int(len(torch.unique(C["ep"]))),
                   "state_dim": int(C["states"].shape[-1]),
                   "window": int(C["states"].shape[1]),
                   "wallclock_s": round(time.time() - t0, 1),
                   "held": "CPU RAM (/root is 99% full -- silent dd truncation)"}
    print(f"[v5] cache: {R['_cache']}", flush=True)

    ep = C["ep"].numpy()
    tgt_g = C["tgt"].to(DEV)
    fan_g = C["fan"].to(DEV)
    nW, nC = fan_g.shape[0], fan_g.shape[1]

    # =======================================================================
    # STAGE B -- SELF-TESTS (M3).  BOTH directions.  Abort on S1/S2 failure.
    # =======================================================================
    st_res: dict = {}

    # S1 -- cache fidelity vs the committed forward-pass numbers
    ade_as, miss_as, along_as, cross_as = pick_metrics(
        fan_g, C["ref_sel_idx"].to(DEV), tgt_g, horizons)
    fan_err4 = (wp4(fan_g.reshape(nW * nC, K_MAX, 2), horizons)
                .reshape(nW, nC, len(WP_STEPS), 2)
                - wp4(tgt_g, horizons)[:, None]).norm(dim=-1).mean(dim=-1)  # [W,N]
    oif = fan_err4.min(dim=1).values.mean().item()
    st_res["cache_fidelity"] = {
        "ade_0_2s_via_cache": round(float(ade_as.mean()), 4),
        "ade_0_2s_committed": COMMITTED["v4_as_trained_ade_0_2s"],
        "abs_diff": round(abs(float(ade_as.mean())
                              - COMMITTED["v4_as_trained_ade_0_2s"]), 5),
        "oracle_in_fan_4wp_via_cache": round(oif, 4),
        "oracle_in_fan_4wp_committed": COMMITTED["v4_oracle_in_fan_4wp"],
        "oif_abs_diff": round(abs(oif - COMMITTED["v4_oracle_in_fan_4wp"]), 5),
        "ref_ade4_identity_max_abs": round(
            float((torch.tensor(ade_as) - C["ref_ade4"]).abs().max()), 6),
        "_rule": "the cached path must reproduce the published forward-pass "
                 "number, or NOTHING computed on it is quotable",
    }
    st_res["cache_fidelity"]["PASS"] = bool(
        st_res["cache_fidelity"]["abs_diff"] < 5e-3
        and st_res["cache_fidelity"]["oif_abs_diff"] < 5e-3)

    # S2 -- reproduce the COMMITTED wm_canary_ade_2s with the true actions
    can_pred = C["canary_wp"].to(DEV).index_select(
        1, torch.tensor([k - 1 for k in WP_STEPS], device=DEV))
    can_err = (can_pred - C["gt4"].to(DEV)).norm(dim=-1).mean(dim=1)      # [W]
    st_res["canary_reproduction"] = {
        "wm_canary_ade_2s_here": round(float(can_err.mean()), 4),
        "wm_canary_ade_2s_committed": (COMMITTED["v4_wm_canary_ade_2s"]
                                       if a.wm == "v4" else None),
        "scoring_wm": a.wm,
        "n": int(can_err.shape[0]),
        "_note": "v4's committed 1.1409 was measured at the SAME episodes/stride; "
                 "for --wm v1 there is no committed value on THIS surface and the "
                 "number below is the v1 line established HERE, not inherited.",
    }
    if a.wm == "v4":
        st_res["canary_reproduction"]["abs_diff"] = round(
            abs(float(can_err.mean()) - COMMITTED["v4_wm_canary_ade_2s"]), 4)
        st_res["canary_reproduction"]["PASS"] = bool(
            st_res["canary_reproduction"]["abs_diff"] < 0.05)
    else:
        st_res["canary_reproduction"]["PASS"] = True

    # S3 -- inverse-map fidelity: traj_to_actions(GT) vs the dataset's own actions
    gt_dense = C["tgt"].to(DEV)                                  # [W, 20, 2]
    a_gt = traj_to_actions(gt_dense, C["v0"].to(DEV))            # [W, 20, 3]
    a_true = C["fut_act"].to(DEV)[:, :K_MAX]                     # [W, 20, 3]
    n_cmp = min(a_gt.shape[1], a_true.shape[1])

    def _corr(x, y):
        x, y = x.reshape(-1).double(), y.reshape(-1).double()
        x, y = x - x.mean(), y - y.mean()
        return float((x * y).sum() / (x.norm() * y.norm() + 1e-12))
    st_res["inverse_map_fidelity"] = {
        "steer_mae": round(float((a_gt[:, :n_cmp, 0]
                                  - a_true[:, :n_cmp, 0]).abs().mean()), 5),
        "steer_corr": round(_corr(a_gt[:, :n_cmp, 0], a_true[:, :n_cmp, 0]), 4),
        "accel_mae": round(float((a_gt[:, :n_cmp, 1]
                                  - a_true[:, :n_cmp, 1]).abs().mean()), 5),
        "accel_corr": round(_corr(a_gt[:, :n_cmp, 1], a_true[:, :n_cmp, 1]), 4),
        "n_steps_compared": int(n_cmp),
        "_read": "the derived actions are not expected to be IDENTICAL (the "
                 "dataset's steer/accel come from the raw ego signal, ours from "
                 "0.1 s finite differences of the pose target). Correlation is "
                 "the readable quantity; the MAE is the derivation's error budget.",
    }

    # S4 -- derivation error budget: roll under derived-GT actions vs true actions
    with torch.no_grad():
        derr = []
        for b0 in range(0, nW, 64):
            s = slice(b0, min(b0 + 64, nW))
            wp = imagine(world_s.predictor, step_readout,
                         C["states"][s].to(DEV), C["win_act"][s].to(DEV),
                         a_gt[s], K_MAX)
            p = wp.index_select(1, torch.tensor([k - 1 for k in WP_STEPS],
                                                device=DEV))
            derr.append((p - C["gt4"][s].to(DEV)).norm(dim=-1).mean(dim=1).cpu())
        derr = torch.cat(derr)
    st_res["derivation_error_budget"] = {
        "ade_2s_under_TRUE_actions": round(float(can_err.mean()), 4),
        "ade_2s_under_DERIVED_actions": round(float(derr.mean()), 4),
        "gap": round(float(derr.mean() - can_err.mean()), 4),
        "_read": "the gap is what the candidate->action inversion costs. It is "
                 "reported, not hidden: it bounds how much of any REFUTE could be "
                 "the inversion rather than the world model.",
    }

    # S5 -- failing inputs
    g = torch.Generator(device="cpu").manual_seed(SEED)
    rnd = torch.randint(0, nC, (nW,), generator=g)
    ade_rnd, *_ = pick_metrics(fan_g, rnd.to(DEV), tgt_g, horizons)
    anti = fan_err4.argmax(dim=1)
    ade_anti, *_ = pick_metrics(fan_g, anti, tgt_g, horizons)
    st_res["failing_input"] = {
        "random_pick_ade_0_2s": round(float(ade_rnd.mean()), 4),
        "bar_a_random_pick": COMMITTED["bar_a_random_pick"],
        "anti_oracle_ade_0_2s": round(float(ade_anti.mean()), 4),
        "as_trained_ade_0_2s": round(float(ade_as.mean()), 4),
        "PASS": bool(float(ade_rnd.mean()) > float(ade_as.mean())
                     and float(ade_anti.mean()) > float(ade_rnd.mean())),
        "_rule": "a random pick over the same frozen fan must score WORSE, and an "
                 "anti-oracle worse still; if not, the harness cannot detect a "
                 "bad selector",
    }
    R["selftest"] = st_res
    print(f"[v5] selftests: " + json.dumps(
        {k: v.get("PASS") for k, v in st_res.items() if "PASS" in v}), flush=True)
    if not (st_res["cache_fidelity"]["PASS"]
            and st_res["canary_reproduction"]["PASS"]):
        R["VERDICT"] = "ABORTED -- a pre-registered self-test failed"
        Path(f"{a.out}_{a.wm}{a.tag}.json").write_text(json.dumps(R, indent=2))
        print("[v5] ABORTED on self-test", flush=True)
        return

    # =======================================================================
    # STAGE C -- the imagination roll-outs (256 per window)
    # =======================================================================
    print(f"[v5] imagining {nW}x{nC} candidates x {K_MAX} steps "
          f"(scorer={a.wm})...", flush=True)
    t1 = time.time()
    imag_all = torch.empty(nW, nC, K_MAX, 2)
    imag_ref = torch.empty(nW, K_MAX, 2)
    ctrv_all = torch.empty(nW, nC, K_MAX, 2)
    bw = a.imag_windows
    with torch.no_grad():
        for b0 in range(0, nW, bw):
            s = slice(b0, min(b0 + bw, nW))
            m = s.stop - s.start
            fanb = C["fan"][s].to(DEV)                          # [m, N, 20, 2]
            v0b = C["v0"][s].to(DEV)
            stb = C["states"][s].to(DEV)
            wab = C["win_act"][s].to(DEV)
            acts = traj_to_actions(fanb, v0b[:, None].expand(m, nC))  # [m,N,20,3]
            ctrv_all[s] = ctrv_rollout(v0b, acts, K_MAX).cpu()
            wp = imagine(
                world_s.predictor, step_readout,
                stb.repeat_interleave(nC, 0), wab.repeat_interleave(nC, 0),
                acts.reshape(m * nC, K_MAX, 3), K_MAX)
            imag_all[s] = wp.reshape(m, nC, K_MAX, 2).cpu()
            # C2's single reference roll: zero-order-hold on the observed action
            wpr, _ = rollout_decode(world_s.predictor, stb, wab, None,
                                    step_readout, K_MAX)
            imag_ref[s] = wpr.cpu()
            if b0 % (bw * 25) == 0:
                print(f"  [imag] {s.stop}/{nW} ({time.time() - t1:.0f}s)",
                      flush=True)
    R["_imagination"] = {
        "wallclock_s": round(time.time() - t1, 1),
        "n_rollouts": int(nW * nC),
        "n_predictor_steps": int(nW * nC * K_MAX),
        "k": K_MAX,
        "convention": "last window action OVERWRITTEN with a_c[0]; a_c[1:] drives "
                      "transitions 2..K -- the candidate controls the whole roll",
    }
    print(f"[v5] imagination done in {R['_imagination']['wallclock_s']}s",
          flush=True)

    # =======================================================================
    # STAGE D -- score every pre-registered arm
    # =======================================================================
    imag_g = imag_all.to(DEV)
    ctrv_g = ctrv_all.to(DEV)
    ref_g = imag_ref.to(DEV)
    kw = dict(v0=C["v0"].to(DEV), vt_speed=C["vt_speed"].to(DEV),
              sel_accel_max=float(getattr(head.cfg, "sel_accel_max", 2.0)),
              ctrv=ctrv_g, imag_ref=ref_g)

    costs = {
        "A1_imag_consistency": cost_A1_consistency(imag_g, fan_g, **kw),
        "A2_imag_goal_speed": cost_A2_goal_speed(imag_g, fan_g, **kw),
        "A3_imag_kinematic": cost_A3_kinematic(imag_g, fan_g, **kw),
        "C1_ctrv_consistency": cost_C1_ctrv(fan_g, **kw),
        "C2_wm_ref_proximity": cost_C2_ref(fan_g, **kw),
    }
    picks = {"A0_as_trained": C["ref_sel_idx"].to(DEV),
             "R_random": rnd.to(DEV),
             "O_oracle_in_fan": fan_err4.argmin(dim=1)}
    for k, v in costs.items():
        picks[k] = v.argmin(dim=1)

    # A4 -- the weighted combination. z-scored per window so the terms are
    # commensurate; the base score enters as a NEGATIVE cost (higher = better).
    def _z(x):
        return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True)
                                                    + 1e-8)
    # the as-trained score is not cached as a vector; the head's own pick enters
    # as a one-hot NEGATIVE cost (i.e. a prior that favours what v4 would choose).
    prior = torch.zeros_like(fan_err4)
    prior.scatter_(1, C["ref_sel_idx"].to(DEV)[:, None], -1.0)
    terms = {k: _z(v) for k, v in costs.items()}
    terms["P_as_trained_prior"] = prior
    tnames = list(terms)
    T = torch.stack([terms[n] for n in tnames], dim=0)          # [T, W, N]

    grid = [0.0, 0.5, 1.0, 2.0]
    R["_a4_grid"] = {"terms": tnames, "weights": grid,
                     "n_combos": len(grid) ** len(tnames)}

    def _a4_best(win_idx):
        """exhaustive weight search on ``win_idx``; -> (best_w, best_ade)."""
        best, bw_ = None, None
        import itertools
        for combo in itertools.product(grid, repeat=len(tnames)):
            if all(c == 0 for c in combo):
                continue
            w = torch.tensor(combo, device=DEV, dtype=T.dtype)[:, None, None]
            sc = (T[:, win_idx] * w).sum(0)
            idx = sc.argmin(dim=1)
            ade = fan_err4[win_idx].gather(1, idx[:, None]).mean().item()
            if best is None or ade < best:
                best, bw_ = ade, combo
        return bw_, best

    all_idx = torch.arange(nW, device=DEV)
    w_ins, ade_ins = _a4_best(all_idx)
    R["A4_in_sample"] = {"weights": dict(zip(tnames, w_ins)),
                         "ade_0_2s_in_sample": round(float(ade_ins), 4),
                         "_read": "IN-SAMPLE: weights fitted and scored on the "
                                  "same windows. This is the number comparable to "
                                  "Bar A's 0.4907 in-sample ceiling."}

    # 5-fold EPISODE-DISJOINT out-of-fold for A4
    rng = np.random.default_rng(SEED)
    eps_u = np.unique(ep)
    perm = rng.permutation(eps_u)
    folds = np.array_split(perm, 5)
    oof_idx = torch.zeros(nW, dtype=torch.long, device=DEV)
    fold_rec = []
    for fi, test_eps in enumerate(folds):
        te = np.isin(ep, test_eps)
        tr_t = torch.tensor(np.where(~te)[0], device=DEV)
        te_t = torch.tensor(np.where(te)[0], device=DEV)
        wf, af = _a4_best(tr_t)
        w = torch.tensor(wf, device=DEV, dtype=T.dtype)[:, None, None]
        sc = (T[:, te_t] * w).sum(0)
        oof_idx[te_t] = sc.argmin(dim=1)
        fold_rec.append({"fold": fi, "test_ep": [int(x) for x in test_eps],
                         "n_test_windows": int(te.sum()),
                         "weights": dict(zip(tnames, wf)),
                         "fit_ade": round(float(af), 4)})
    picks["A4_imag_combo_oof"] = oof_idx
    R["_a4_folds"] = fold_rec

    # ---- metrics + intervals for every arm --------------------------------
    per_arm: dict = {}
    ade_by_arm: dict = {}
    for name, idx in picks.items():
        ade, miss, along, cross = pick_metrics(fan_g, idx, tgt_g, horizons)
        ade_by_arm[name] = ade
        per_arm[name] = {
            "ade_0_2s": round(float(ade.mean()), 4),
            "miss_at_2m": round(float(miss.mean()), 4),
            "along_abs_dense_LONGITUDINAL": round(float(along.mean()), 4),
            "cross_abs_dense_LATERAL": round(float(cross.mean()), 4),
            "frac_pick_equals_as_trained": round(
                float((idx == C["ref_sel_idx"].to(DEV)).float().mean()), 4),
            "n_distinct_picks": int(len(torch.unique(idx))),
            "ci": single_ci(ade, ep),
        }
    R["arms"] = per_arm

    base = ade_by_arm["A0_as_trained"]
    R["paired_vs_as_trained"] = {
        n: paired_ci(v, base, ep) for n, v in ade_by_arm.items()
        if n != "A0_as_trained"}
    # attribution: the primary against BOTH controls
    R["attribution"] = {
        "A1_minus_C1_ctrv": paired_ci(ade_by_arm["A1_imag_consistency"],
                                      ade_by_arm["C1_ctrv_consistency"], ep),
        "A1_minus_C2_ref": paired_ci(ade_by_arm["A1_imag_consistency"],
                                     ade_by_arm["C2_wm_ref_proximity"], ep),
        "_read": "NEGATIVE = per-candidate WM imagination beats the control. A "
                 "win the controls MATCH is a scoring-rule result, NOT an "
                 "imagination result.",
    }

    # ---- verdict against the pre-registered bars --------------------------
    par_free = {k: v for k, v in per_arm.items()
                if k.startswith(("A1", "A2", "A3"))}
    best_pf = min(par_free, key=lambda k: par_free[k]["ade_0_2s"])
    best_in_sample = min(par_free[best_pf]["ade_0_2s"],
                         R["A4_in_sample"]["ade_0_2s_in_sample"])
    best_oof = min(par_free[best_pf]["ade_0_2s"],
                   per_arm["A4_imag_combo_oof"]["ade_0_2s"])
    confirm = best_in_sample < COMMITTED["bar_a_in_sample_ceiling_ce"]
    strong = confirm and best_oof < COMMITTED["v1_reference_40ep"]
    R["VERDICT_BLOCK"] = {
        "best_parameter_free_imagination_arm": best_pf,
        "best_in_sample_ade_0_2s": round(float(best_in_sample), 4),
        "bar_CONFIRM_lt": COMMITTED["bar_a_in_sample_ceiling_ce"],
        "best_oof_ade_0_2s": round(float(best_oof), 4),
        "bar_STRONG_lt": COMMITTED["v1_reference_40ep"],
        "VERDICT": ("STRONG" if strong else "CONFIRM" if confirm else "REFUTE"),
    }

    # =======================================================================
    # STAGE E -- stratify by canary quality (Bar B: the WM failure is
    # CONCENTRATED, not uniform -- 22.7% of windows are already under the bar)
    # =======================================================================
    cq = can_err.cpu().numpy()
    q = np.quantile(cq, [0.25, 0.5, 0.75])
    strata = {
        "under_canary_bar_0.55": cq <= COMMITTED["wm_canary_bar"],
        "over_canary_bar_0.55": cq > COMMITTED["wm_canary_bar"],
        "q1_best_canary": cq <= q[0],
        "q2": (cq > q[0]) & (cq <= q[1]),
        "q3": (cq > q[1]) & (cq <= q[2]),
        "q4_worst_canary": cq > q[2],
    }
    strat: dict = {"_canary_quantiles": {"p25": round(float(q[0]), 4),
                                         "p50": round(float(q[1]), 4),
                                         "p75": round(float(q[2]), 4),
                                         "mean": round(float(cq.mean()), 4)}}
    for sname, mask in strata.items():
        if mask.sum() < 10:
            continue
        row = {"n_windows": int(mask.sum()),
               "n_episodes": int(len(np.unique(ep[mask]))),
               "frac_of_all": round(float(mask.mean()), 4)}
        for aname, v in ade_by_arm.items():
            row[aname] = round(float(v[mask].mean()), 4)
        row["paired_A1_minus_A0"] = paired_ci(
            ade_by_arm["A1_imag_consistency"][mask], base[mask], ep[mask])
        strat[sname] = row
    R["stratified_by_canary_quality"] = strat

    # ---- per-window dump so every bar is recomputable with NO GPU ----------
    # `imag`/`ctrv`/`fan` are persisted so E-V5-3's cost-vs-quality curve, and
    # any future scoring rule, are recomputable OFF-GPU. The k-step roll-out is
    # a strict PREFIX of the 20-step one (accumulate_se2 is causal), so
    # imag[..., :k, :] IS the k-step imagination -- no re-rolling needed.
    dump = {"ep": C["ep"], "t": C["t"], "canary_err": can_err.cpu(),
            "ref_sel_idx": C["ref_sel_idx"], "v0": C["v0"],
            "vt_speed": C["vt_speed"], "vt_keep": C["vt_keep"],
            "fan_err4": fan_err4.cpu(), "tgt": C["tgt"],
            "fan": C["fan"], "imag": imag_all, "ctrv": ctrv_all,
            "imag_ref": imag_ref,
            "picks": {k: v.cpu() for k, v in picks.items()},
            "ade_by_arm": {k: torch.tensor(v) for k, v in ade_by_arm.items()},
            "costs": {k: v.cpu() for k, v in costs.items()},
            "_note": "fan_err4 [W,256] is the 4-wp error of EVERY candidate: any "
                     "selection rule over this fan is recomputable from it alone. "
                     "imag[...,:k,:] is the k-step imagination (causal prefix)."}
    torch.save(dump, f"{a.out}_{a.wm}{a.tag}_windows.pt")

    # ---- E-V5-3: the raw timing half of the cost curve ---------------------
    if DEV == "cuda":
        import copy as _copy
        tim: dict = {}
        stb = C["states"][:8].to(DEV)
        wab = C["win_act"][:8].to(DEV)
        for nb in (256, 1024, 2048):
            s = stb[:1].expand(nb, -1, -1).contiguous()
            w = wab[:1].expand(nb, -1, -1).contiguous()
            with torch.no_grad():
                for _ in range(3):
                    world_s.predictor(s, w)
                torch.cuda.synchronize()
                t = time.time()
                for _ in range(10):
                    world_s.predictor(s, w)
                torch.cuda.synchronize()
            tim[f"predictor_step_ms_batch{nb}"] = round(
                (time.time() - t) / 10 * 1e3, 3)
        # the reference cost: ONE full camera pass (encode the 8-frame window)
        fr1 = _to_device(default_collate([ds_val[sel[0]]]), DEV)["frames"]
        with torch.no_grad():
            for _ in range(3):
                world_s.encode_window(fr1)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(10):
                world_s.encode_window(fr1)
            torch.cuda.synchronize()
        tim["encode_window_ms_batch1"] = round((time.time() - t) / 10 * 1e3, 3)
        tim["_read"] = ("predictor_step is ONE imagination step for the whole "
                        "batch; imagination cost = ceil(N/batch) * k * step_ms. "
                        "encode_window is the full camera pass this is measured "
                        "against.")
        R["_timing"] = tim
        print(f"[v5] timing: {tim}", flush=True)

    R["_wallclock_total_s"] = round(time.time() - t_start, 1)
    Path(f"{a.out}_{a.wm}{a.tag}.json").write_text(json.dumps(R, indent=2))
    print(json.dumps({"VERDICT": R["VERDICT_BLOCK"],
                      "arms": {k: v["ade_0_2s"] for k, v in per_arm.items()}},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
