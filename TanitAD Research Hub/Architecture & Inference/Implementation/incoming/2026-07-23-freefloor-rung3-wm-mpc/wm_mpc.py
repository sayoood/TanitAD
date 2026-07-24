"""Free inference-time floor — RUNG 3: WM-planning (MPPI / CEM) over the flagship v1
world model, on the no-renderer kinematic closed-loop harness (task #21).

WHAT THIS ADDS
--------------
The imagination proof (`incoming/2026-07-22-imagination-closedloop-proof/`) measured
two arms on the flagship v1 WM, paired on identical held-out windows:

    (A) single-shot open-loop   `open_plan_bike`  — plan ONCE, track frozen open-loop
    (B) imagination-in-the-loop `closed_bike`     — RE-PLAN the head every 0.1 s tick
                                                    and follow its 0.5 s waypoint with
                                                    a pure-pursuit inverse (SINGLE-STEP
                                                    re-plan)

    paired delta (B - A) ADE@2s = -0.213 m [-0.341, -0.053]  (IMAGINATION_HELPS)

RUNG 3 upgrades arm (B)'s ACTION SELECTION only. Arm (C) = `mpc_bike` still re-plans
the SAME strategic->tactical head anchor every tick, but instead of the deterministic
pure-pursuit inverse it runs **receding-horizon MPPI / CEM over the world model**:

    at each 0.1 s control tick:
      1. re-plan the head anchor on the current (imagined) latent  (== arm B)
      2. build a nominal action SEQUENCE over horizon H by pure-pursuit tracking the
         anchor (so candidate-0 == arm B's action; MPPI is a STRICT SUPERSET of B)
      3. sample K action sequences = nominal + Gaussian noise (clamped to envelope)
      4. ROLL EACH candidate through the WM operative predictor (imagination),
         decode the imagined metric path with the grounded step-readout
      5. score each imagined trajectory with a GT-FREE cost:
             track  = ||imagined path - head anchor||^2   (stay-on-intended-path /
                      lateral-deviation "off-road" proxy — the floor's target signal)
             progress = -forward reach at H                (small; anchor already
                                                            encodes the speed intent)
             comfort  = jerk + |steer| + |accel|          (the floor's comfort signal)
      6. MPPI: weight w_i = softmax(-J_i / temp); take the weighted first action.
         CEM : take the mean first action of the top-M elite (lowest-cost) candidates.
      7. EXECUTE that first action exactly like arm B (imagine one WM step, integrate
         the bicycle, slide the window) -> receding horizon.

Everything else (the head, the bicycle drive, the latent imagination bookkeeping, the
held-out windows) is byte-identical to arm B. So the paired delta (C - B) isolates
EXACTLY the sampling-based WM planning vs the single-step re-plan.

The cost NEVER sees ground truth (the planner cannot at deploy). We pre-register ONE
config (MPPI, K=8, H=8, grounded cost) as the headline BEFORE running, then report a
small secondary sweep clearly labelled as optimistic-if-selected.

CAVEATS (same as the proof / DAgger): self-referential (the WM is both the planner's
state estimator AND the simulator), kinematic no-renderer harness (off-road = large
lateral-deviation proxy, not a real safety rate), 12 val episodes. Within-harness
MECHANISM proof; AlpaSim is the decision-grade confirmer. This tool is VALIDATED for
inference-mechanism proofs (it produced the -0.213 win) and REFUTED for TRAINING
experiments (DAgger) — rung 3 is inference-only, no training, no reward.

Reuses (never reinvents) taniteval/taniteval/closedloop.py: wp_to_control, build_action,
bicycle_integrate, densify_plan, closed_loop_rollout (arm B), open_loop_plan_rollout
(arm A), _de, _comfort, analyze, and every protocol constant; taniteval/ci.py for the
paired episode-cluster bootstrap; metric_dynamics.accumulate_se2 for the grounded path.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from taniteval import closedloop as cl
from taniteval import ci as _ci
from driving_diagnostic import (baseline_waypoints, gt_ego_waypoints,
                                net_heading_change_deg)
from tanitad.models.metric_dynamics import accumulate_se2, rollout_decode

DT = cl.DT
K_MAX = cl.K_MAX
LOOKAHEAD = cl.LOOKAHEAD_STEP
WHEELBASE = cl.WHEELBASE
STEER_CLAMP = cl.STEER_CLAMP
ACCEL_CLAMP = cl.ACCEL_CLAMP
DIVERGENCE_M = cl.DIVERGENCE_M

# --------------------------------------------------------------------------- #
# PRE-REGISTERED default MPC config (locked BEFORE the measured run).          #
# Values chosen a priori from the action envelope + comfort bounds, NOT tuned  #
# against the closed-loop ADE (that would leak the test set).                  #
# --------------------------------------------------------------------------- #
MPC_DEFAULT = {
    "method": "mppi",       # "mppi" (softmax) | "cem" (elite mean)
    "K": 8,                 # candidate action sequences (registry timed ~20 ms @ K=8)
    "H": 8,                 # planning horizon in 0.1 s ticks (0.8 s) — short/on-manifold
    "sigma_steer": 0.008,   # rad   (data |steer|<=0.016; ~half the data range)
    "sigma_accel": 0.40,    # m/s^2 (data |accel|<=1.9)
    "temp": 0.7,            # MPPI softmax temperature on the standardized cost
    "elite_frac": 0.25,     # CEM: top 25% of K are elites
    "w_track": 1.0,         # anchor-tracking (dominant; the off-road proxy)
    "w_progress": 0.05,     # forward-reach reward (small; anchor already encodes speed)
    "w_comfort": 0.20,      # jerk + |steer| + |accel|
    "cost_path": "grnd",    # "grnd" = WM-imagined metric decode (true WM planning)
                            # "bike" = kinematic path (robustness check; WM unused in cost)
    "seed": 0,
}


# --------------------------------------------------------------------------- #
# Nominal action sequence: pure-pursuit tracking the head anchor over H ticks.  #
# Mirrors closedloop.open_loop_plan_rollout's per-tick controller EXACTLY, so   #
# the first nominal action == arm B's first action (MPPI superset of B).        #
# --------------------------------------------------------------------------- #
def _nominal_pp_seq(anchor_dense: torch.Tensor, v0: torch.Tensor, h: int):
    """anchor_dense [b,h,2] (current ego frame, steps 1..h) + v0 [b] -> nominal
    (steer_seq [b,h], accel_seq [b,h]) by pure-pursuit + bicycle, open-loop."""
    b = anchor_dense.shape[0]
    dev = anchor_dense.device
    x = torch.zeros(b, device=dev)
    y = torch.zeros(b, device=dev)
    yaw = torch.zeros(b, device=dev)
    v = v0.clone()
    ss, aa = [], []
    for kk in range(h):
        ti = min(kk + LOOKAHEAD - 1, h - 1)
        dx, dy = anchor_dense[:, ti, 0] - x, anchor_dense[:, ti, 1] - y
        c, s = torch.cos(-yaw), torch.sin(-yaw)
        w_look = torch.stack([dx * c - dy * s, dx * s + dy * c], dim=-1)
        steer, accel = cl.wp_to_control(w_look, v)
        ss.append(steer)
        aa.append(accel)
        x = x + v * torch.cos(yaw) * DT
        y = y + v * torch.sin(yaw) * DT
        yaw = yaw + v / WHEELBASE * torch.tan(steer) * DT
        v = (v + accel * DT).clamp_min(0.0)
    return torch.stack(ss, dim=1), torch.stack(aa, dim=1)          # [b,h]


@torch.no_grad()
def mppi_select(model, step_readout, win_s, win_a, v, anchor_dense, speed_input, cfg):
    """One MPPI/CEM control decision over the WM. Returns (steer0 [b], accel0 [b]).

    win_s [b,W,S] current imagined latent window; win_a [b,W,A] action window; v [b]
    current speed; anchor_dense [b,H,2] the head's densified plan (current ego frame).
    ONLY the action selection differs from arm B — the caller executes steer0/accel0
    exactly as closed_loop_rollout does.
    """
    b, W, S = win_s.shape
    A = win_a.shape[-1]
    K, H = cfg["K"], cfg["H"]
    dev = win_s.device
    g = torch.Generator(device=dev).manual_seed(cfg["seed"])

    # --- nominal sequence + K sampled perturbations (candidate 0 = nominal) ----
    nom_st, nom_ac = _nominal_pp_seq(anchor_dense, v, H)           # [b,H]
    eps_st = torch.randn(b, K, H, generator=g, device=dev)
    eps_ac = torch.randn(b, K, H, generator=g, device=dev)
    steer_k = (nom_st[:, None] + cfg["sigma_steer"] * eps_st).clamp(-STEER_CLAMP, STEER_CLAMP)
    accel_k = (nom_ac[:, None] + cfg["sigma_accel"] * eps_ac).clamp(-ACCEL_CLAMP, ACCEL_CLAMP)
    steer_k[:, 0] = nom_st                                         # elite anchor
    accel_k[:, 0] = nom_ac

    # --- roll every candidate through the WM (imagination) ---------------------
    ws = win_s[:, None].expand(b, K, W, S).reshape(b * K, W, S).clone()
    wa = win_a[:, None].expand(b, K, W, A).reshape(b * K, W, A).clone()
    vv = v[:, None].expand(b, K).reshape(b * K).clone()
    st_f = steer_k.reshape(b * K, H)
    ac_f = accel_k.reshape(b * K, H)
    grnd_dp = []
    for hh in range(H):
        a_exec = cl.build_action(st_f[:, hh], ac_f[:, hh], vv, speed_input)   # [bK,A]
        wa_exec = wa.clone()
        wa_exec[:, -1] = a_exec
        z_next = model.predictor(ws, wa_exec)[1]                  # [bK,S] 1-step head
        grnd_dp.append(step_readout(ws[:, -1], z_next))           # [bK,3] metric dpose
        vv = (vv + ac_f[:, hh] * DT).clamp_min(0.0)
        ws = torch.cat([ws[:, 1:], z_next.unsqueeze(1)], dim=1)
        wa = torch.cat([wa_exec[:, 1:], a_exec.unsqueeze(1)], dim=1)
    grnd_path = accumulate_se2(torch.stack(grnd_dp, dim=1)).reshape(b, K, H, 2)

    if cfg["cost_path"] == "bike":
        # kinematic path under the candidate actions (WM NOT used in the cost —
        # robustness check that isolates "does the WM consequence matter?")
        bike, _ = cl.bicycle_integrate(vv.new_zeros(b * K) + v.repeat_interleave(K),
                                       st_f, ac_f)
        cand_path = bike.reshape(b, K, H, 2)
    else:
        cand_path = grnd_path

    # --- GT-free cost: track anchor + progress + comfort -----------------------
    track = ((cand_path - anchor_dense[:, None]) ** 2).sum(-1).mean(-1)        # [b,K]
    progress = cand_path[:, :, -1, 0]                                          # [b,K]
    jerk = (accel_k[:, :, 1:] - accel_k[:, :, :-1]).abs().mean(-1)             # [b,K]
    comfort = jerk + steer_k.abs().mean(-1) + 0.1 * accel_k.abs().mean(-1)     # [b,K]

    def _z(t):                       # per-window standardization across the K candidates
        return (t - t.mean(1, keepdim=True)) / (t.std(1, keepdim=True) + 1e-6)

    J = cfg["w_track"] * _z(track) - cfg["w_progress"] * _z(progress) \
        + cfg["w_comfort"] * _z(comfort)                                       # [b,K]

    if cfg["method"] == "cem":
        m = max(1, int(round(cfg["elite_frac"] * K)))
        elite = J.topk(m, dim=1, largest=False).indices                        # [b,m] lowest cost
        st0 = torch.gather(steer_k[:, :, 0], 1, elite).mean(1)
        ac0 = torch.gather(accel_k[:, :, 0], 1, elite).mean(1)
    else:  # mppi
        w = torch.softmax(-J / cfg["temp"], dim=1)                             # [b,K]
        st0 = (w * steer_k[:, :, 0]).sum(1)
        ac0 = (w * accel_k[:, :, 0]).sum(1)
    return st0, ac0


@torch.no_grad()
def mpc_rollout(model, step_readout, states0, aw, v0, speed_input, cfg, k=K_MAX):
    """Arm (C): receding-horizon WM-MPPI/CEM closed loop. Same bookkeeping as
    closedloop.closed_loop_rollout (arm B) — ONLY the per-tick action is chosen by
    mppi_select instead of the pure-pursuit inverse. Returns the mpc bicycle path
    + executed controls, and the total MPC-selection wall time (for the compute cost).
    """
    b = states0.shape[0]
    nav = torch.zeros(b, dtype=torch.long, device=states0.device)
    win_s = states0.clone()
    win_a = aw.clone()
    v = v0.clone()
    steer_seq, accel_seq = [], []
    sel_s = 0.0
    for _ in range(k):
        # (a) re-plan the head anchor on the current imagined latent (== arm B)
        ctx = model.strategic_policy(win_s, nav)["ctx"]
        wp = model.tactical_policy(win_s, ctx)["waypoints"]
        anchor = cl.densify_plan(wp, cfg["H"])                    # [b,H,2] current frame
        # (b) ACTION SELECTION: MPPI/CEM over the WM (the ONLY change vs arm B)
        if states0.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        steer, accel = mppi_select(model, step_readout, win_s, win_a, v,
                                   anchor, speed_input, cfg)
        if states0.is_cuda:
            torch.cuda.synchronize()
        sel_s += time.perf_counter() - t0
        # (c) EXECUTE the selected first action exactly like arm B
        a_exec = cl.build_action(steer, accel, v, speed_input)
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
    return {"mpc_bike": bike_pts, "steer": steer_seq, "accel": accel_seq,
            "speed": bike_spd, "select_s": sel_s, "n_decisions": b * k}


# --------------------------------------------------------------------------- #
# Collection — one aligned pass building arms A, B, C on IDENTICAL windows.     #
# Arms A and B come from the UNMODIFIED closedloop functions, so they reproduce #
# the -0.213 proof exactly (a built-in sanity check, cf. DAgger's A0).          #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_all(model, step_readout, episodes, device, speed_input, cfg,
                window=cl.WINDOW, stride=cl.STRIDE, batch=cl.BATCH, k=K_MAX):
    wp_idx = torch.tensor(cl.IDX, device=device)
    names = ("gt", "cv", "closed_bike", "open_plan_bike", "plan_direct",
             "mpc_bike", "open_grnd", "open_bike", "closed_grnd",
             "speed", "head_deg", "steer", "accel", "vseq",
             "mpc_steer", "mpc_accel", "mpc_vseq")
    acc = {n: [] for n in names}
    eid = []
    sel_s, n_dec = 0.0, 0
    for ep in episodes:
        feats = ep.feats
        T = feats.shape[0]
        starts = list(range(0, T - window - k, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(feats[t:t + window]) for t in ch]).to(device)
            if fw.dtype == torch.uint8:
                fw = fw.float().div_(255.0)
            elif fw.dtype == torch.float16:
                fw = fw.float()
            aw = torch.stack([ep.actions[t:t + window] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + window:t + window + k] for t in ch]).to(device)
            v0 = ep.poses[last, 3].to(device).float()
            if speed_input:
                v0c = (v0 / cl.SPEED_SCALE)[:, None, None]
                aw = torch.cat([aw, v0c.expand(-1, aw.shape[1], -1)], dim=-1)
                fa = torch.cat([fa, v0c.expand(-1, fa.shape[1], -1)], dim=-1)
            states0 = model.encode_window(fw)

            open_wp, _ = rollout_decode(model.predictor, states0, aw, fa, step_readout, k)
            true_steer = torch.cat([aw[:, -1:, 0], fa[:, :k - 1, 0]], dim=1)
            true_accel = torch.cat([aw[:, -1:, 1], fa[:, :k - 1, 1]], dim=1)
            open_bike, _ = cl.bicycle_integrate(v0, true_steer, true_accel)
            clb = cl.closed_loop_rollout(model, step_readout, states0, aw, v0,
                                         speed_input, k)               # arm B
            olb = cl.open_loop_plan_rollout(model, states0, v0, speed_input, k)  # arm A
            mpc = mpc_rollout(model, step_readout, states0, aw, v0,
                              speed_input, cfg, k)                      # arm C
            sel_s += mpc["select_s"]
            n_dec += mpc["n_decisions"]

            acc["gt"].append(gt_ego_waypoints(ep.poses, last).cpu())
            acc["cv"].append(baseline_waypoints(ep.poses, last)["constant_velocity"].cpu())
            acc["closed_bike"].append(clb["closed_bike"][:, wp_idx].cpu())
            acc["closed_grnd"].append(clb["closed_grnd"][:, wp_idx].cpu())
            acc["open_plan_bike"].append(olb["open_plan_bike"][:, wp_idx].cpu())
            acc["plan_direct"].append(olb["plan_direct"][:, wp_idx].cpu())
            acc["mpc_bike"].append(mpc["mpc_bike"][:, wp_idx].cpu())
            acc["open_grnd"].append(open_wp[:, wp_idx].cpu())
            acc["open_bike"].append(open_bike[:, wp_idx].cpu())
            acc["speed"].append(v0.cpu())
            acc["head_deg"].append(net_heading_change_deg(ep.poses, last))
            acc["steer"].append(clb["steer"].cpu())
            acc["accel"].append(clb["accel"].cpu())
            acc["vseq"].append(clb["speed"].cpu())
            acc["mpc_steer"].append(mpc["steer"].cpu())
            acc["mpc_accel"].append(mpc["accel"].cpu())
            acc["mpc_vseq"].append(mpc["speed"].cpu())
            eid.extend([ep.episode_id] * len(ch))
    out = {n: torch.cat(v).float() for n, v in acc.items()}
    out["eid"] = eid
    out["_mpc_select_s"] = sel_s
    out["_mpc_n_decisions"] = n_dec
    return out


def analyze_mpc(win, cfg):
    """Paired C-vs-B (the headline), C-vs-A, plus arm C's own drift / off-road /
    comfort. Reuses cl._de + taniteval.ci; reproduces the A-vs-B proof block via
    cl.analyze on the shared arms."""
    eid = win["eid"]
    gt = win["gt"]

    # --- reproduce the A-vs-B imagination proof (sanity: must match -0.213) -----
    ab_win = {kk: win[kk] for kk in ("gt", "cv", "closed_bike", "closed_grnd",
                                     "open_grnd", "open_bike", "open_plan_bike",
                                     "plan_direct", "speed", "head_deg",
                                     "steer", "accel", "vseq")}
    ab_win["eid"] = eid
    ab = cl.analyze(ab_win)
    ic = ab["imagination_comparison"]

    # --- per-window arm scalars (ADE 0-2s, FDE@2s, divergence, lateral@2s) ------
    ade_A = cl._de(win["open_plan_bike"], gt).mean(dim=1).numpy()
    ade_B = cl._de(win["closed_bike"], gt).mean(dim=1).numpy()
    ade_C = cl._de(win["mpc_bike"], gt).mean(dim=1).numpy()
    fde_B = cl._de(win["closed_bike"], gt)[:, -1].numpy()
    fde_C = cl._de(win["mpc_bike"], gt)[:, -1].numpy()
    div_B = (cl._de(win["closed_bike"], gt)[:, -1] > DIVERGENCE_M).float().numpy()
    div_C = (cl._de(win["mpc_bike"], gt)[:, -1] > DIVERGENCE_M).float().numpy()
    lat_B = (win["closed_bike"][:, -1, 1] - gt[:, -1, 1]).abs().numpy()   # off-road proxy@2s
    lat_C = (win["mpc_bike"][:, -1, 1] - gt[:, -1, 1]).abs().numpy()

    # --- the pre-registered readings (paired episode-cluster bootstrap) ---------
    d_CB_ade = _ci.paired_episode_cluster_bootstrap(ade_C, ade_B, eid)   # C - B
    d_CA_ade = _ci.paired_episode_cluster_bootstrap(ade_C, ade_A, eid)   # C - A
    d_CB_fde = _ci.paired_episode_cluster_bootstrap(fde_C, fde_B, eid)
    d_CB_div = _ci.paired_episode_cluster_bootstrap(div_C, div_B, eid)
    d_CB_lat = _ci.paired_episode_cluster_bootstrap(lat_C, lat_B, eid)

    d = d_CB_ade["delta"]
    if d_CB_ade["separated"] and d < 0:
        verdict = ("MPC_HELPS: WM-MPPI/CEM planning (C) has LOWER closed-loop ADE@2s "
                   "than the single-step re-plan (B), CI-separated (paired "
                   "episode-cluster bootstrap) -> rung 3 is a real lever beyond the "
                   "-0.213 imagination win; PROMOTE to AlpaSim for the decision-grade test")
    elif d_CB_ade["separated"] and d > 0:
        verdict = ("MPC_HURTS: WM-MPPI/CEM planning (C) has HIGHER closed-loop ADE@2s "
                   "than the single-step re-plan (B), CI-separated -> sampling+selecting "
                   "on THIS self-referential WM's imagined latents degrades vs simply "
                   "re-planning the head; keep the single-step re-plan (WM-exploitation "
                   "signature in miniature; needs a faithful sim / pessimism)")
    else:
        verdict = ("TIE: WM-MPPI/CEM planning (C) and the single-step re-plan (B) are "
                   "NOT CI-separated on closed-loop ADE@2s -> the single-step re-plan "
                   "already captures the imagination benefit at this fidelity; MPPI adds "
                   "no measurable closed-loop gain -> report plainly, do NOT promote on "
                   "this evidence (AlpaSim could still separate them)")

    return {
        "_headline": "paired C-vs-B is the rung-3 reading; A-vs-B reproduces the "
                     "-0.213 imagination proof as a sanity check",
        "arms": {
            "A_open_plan_bike": "single-shot open-loop (plan once, track frozen)",
            "B_closed_bike": "single-step re-plan (imagination-in-the-loop, "
                             "pure-pursuit inverse) = the -0.213 baseline",
            "C_mpc_bike": f"receding-horizon WM {cfg['method'].upper()} "
                          f"(K={cfg['K']}, H={cfg['H']}, {cfg['cost_path']} cost) "
                          "over the flagship v1 world model",
        },
        "config": cfg,
        "A_open_plan_bike_ade@2s": _ci.episode_cluster_bootstrap(ade_A, eid),
        "B_closed_bike_ade@2s": _ci.episode_cluster_bootstrap(ade_B, eid),
        "C_mpc_bike_ade@2s": _ci.episode_cluster_bootstrap(ade_C, eid),
        "paired_delta_C_minus_B_ade@2s": d_CB_ade,
        "paired_delta_C_minus_A_ade@2s": d_CA_ade,
        "paired_delta_C_minus_B_fde@2s": d_CB_fde,
        "paired_delta_C_minus_B_divergence_gt5m@2s": d_CB_div,
        "paired_delta_C_minus_B_lateral_offroad@2s": d_CB_lat,
        "B_divergence_gt5m@2s": _ci.episode_cluster_bootstrap(div_B, eid),
        "C_divergence_gt5m@2s": _ci.episode_cluster_bootstrap(div_C, eid),
        "B_lateral_offroad@2s": _ci.episode_cluster_bootstrap(lat_B, eid),
        "C_lateral_offroad@2s": _ci.episode_cluster_bootstrap(lat_C, eid),
        "C_comfort": cl._comfort({"accel": win["mpc_accel"], "steer": win["mpc_steer"],
                                  "vseq": win["mpc_vseq"]}),
        "reproduced_A_vs_B": {
            "A_open_plan_bike_ade@2s": ic["A_open_plan_bike_ade@2s"]["mean"],
            "B_closed_bike_ade@2s": ic["B_closed_bike_ade@2s"]["mean"],
            "paired_delta_B_minus_A_ade@2s": ic["paired_delta_B_minus_A_ade@2s"]["delta"],
            "expected_proof": {"A": 1.9325, "B": 1.7196, "B_minus_A": -0.213},
        },
        "verdict": verdict,
    }
