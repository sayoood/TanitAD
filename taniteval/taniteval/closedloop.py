"""TanitEval — NO-RENDERER closed-loop evaluation (imagination-in-the-loop).

WHY THIS EXISTS
---------------
AlpaSim's NuRec renderer is unrunnable on this eval pod (unprivileged container,
seccomp blocks user namespaces — see the 2026-07-19 AlpaSim INTAKE). Given our
data (front-wide camera + ego poses, NO HD map / agent boxes), a collision /
drivable-area PDM is not computable — we lack the geometry. The HONEST closed-loop
test we CAN run is *imagination-in-the-loop*: use the flagship world model as its
OWN neural simulator and measure how far the ego drifts from the logged path when
the model consumes its OWN predictions (distribution shift), plus how that drift
compounds vs a matched teacher-forced open-loop rollout.

THE LOOP (model = its own sim)
------------------------------
Start from a real observed window -> encode -> latent window z_{t-7..t} (+ ego
state). Then for each 0.1 s control tick k = 0..K-1 (K=20 -> 2 s @ 10 Hz):
  (a) PLAN: run the trained strategic->tactical hierarchy on the CURRENT latent
      window (deploy path: follow nav, state-only) -> 2 s ego-frame waypoints
      {0.5,1,1.5,2}s. The plan is computed on the IMAGINED latent, not pixels.
  (b) CONTROL: derive the executed action a_k=(steer,accel) from the near-term
      (0.5 s) waypoint via a pure-pursuit steer + progress-matched speed
      controller (the "waypoint->control" bicycle mapping from the drafted
      AlpaSim adapter).
  (c) IMAGINE: step the latent with the OPERATIVE brain z_{k+1}=predictor(z_k,a_k)
      — the model consumes its OWN prediction (NOT ground-truth next latent, NOT
      true future actions). Intent-free (the deployed operative rollout is
      intent-free by design; the step-readout is calibrated intent-free — threading
      intent off-manifolds it, see taniteval.hierarchy). Slide the latent window.
  (d) DRIVE: propagate the ego pose by a kinematic bicycle model under a_k.

Two closed-loop ego paths fall out and BOTH are reported (they answer different
questions honestly):
  * closed_bicycle : kinematic integration of the executed controls (task (d)).
                     Independent of the model's metric decode -> the HEADLINE
                     closed-loop ADE/FDE, and the source of comfort/divergence.
  * closed_grounded: SE(2) accumulation of the operative predictor's own metric
                     step-readout on the imagined transitions. Apples-to-apples
                     with the teacher-forced open-loop rollout (metric_dynamics.
                     rollout_decode) -> used for the closed-MINUS-open delta, which
                     isolates the pure distribution-shift cost (only the action /
                     latent SOURCE changes; the path construction is identical).

MATCHED OPEN-LOOP baselines (compounding-error reference):
  * open_grounded : rollout_decode under the TRUE future actions (the exact gate /
                    leaderboard grounded rollout) — the same [b,K,2] path builder
                    as closed_grounded, so closed_grounded - open_grounded is a
                    clean compounding-error delta.
  * open_bicycle  : bicycle integration under the TRUE actions — the KINEMATIC
                    FIDELITY FLOOR (how well the bicycle+action convention alone
                    reconstructs the logged path; ~0.38 m ADE@2s on physicalai).
                    closed_bicycle - open_bicycle is the policy+imagination cost.

HONEST LIMITATIONS (stated, not hidden)
---------------------------------------
  * NO collision / drivable-area / PDM score — we have no map or agent boxes.
    This is NOT a safety closed-loop; it is a drift / stability closed-loop.
  * SELF-REFERENTIAL: the world model is BOTH the planner's state estimator and
    the simulator. Failures the world model is blind to (things it never learned
    to imagine) are invisible here — the loop cannot surprise itself with an event
    outside its own imagination. A photoreal external sim (NuRec/HUGSIM) is the
    only cure; this is the honest on-pod proxy.
  * The waypoint->control inverse and the bicycle are a HARNESS controller, not
    part of the model; open_bicycle quantifies its fidelity floor so closed-loop
    drift stays attributable.
  * Actions correlate ~0.87 with bicycle-consistent controls (CAN signals, not
    perfectly kinematic) — the residual shows up in the open_bicycle floor.

Reuses (never reinvents): loaders.load / data.load_frames (arm + episodes),
metric_dynamics.rollout_decode + accumulate_se2 + StepDisplacementReadout (grounded
path), driving_diagnostic.gt_ego_waypoints / baseline_waypoints / _ego /
net_heading_change_deg (geometry + CV floor), gates.split_by_episode (CI protocol).
Run: python3 -m taniteval.closedloop --arm flagship-30k [--episodes 40]
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
sys.path.insert(0, "/root/taniteval")

from driving_diagnostic import (WP_STEPS, baseline_waypoints,  # noqa: E402
                                gt_ego_waypoints, net_heading_change_deg)
from tanitad.eval.gates import split_by_episode  # noqa: E402
from tanitad.models.metric_dynamics import accumulate_se2, rollout_decode  # noqa: E402

# --- protocol constants (parity with bench.py / rollout.py) ----------------- #
SPEED_SCALE = 10.0          # v0 action-channel scale (every trainer)
DT = 0.1                    # 10 Hz operative tick
WHEELBASE = 2.7             # kinematic bicycle wheelbase (kinematic.py)
WINDOW = 8                  # predictor window W
K_MAX = max(WP_STEPS)       # 20 steps = 2 s
IDX = [k - 1 for k in WP_STEPS]
HORIZONS_S = {5: "0.5s", 10: "1s", 15: "1.5s", 20: "2s"}
STRIDE = 8
BATCH = 32

# --- harness controller (waypoint -> control) constants -------------------- #
LOOKAHEAD_STEP = 5          # plan the 0.5 s waypoint as the pure-pursuit target
LD2_FLOOR = 0.25            # (0.5 m)^2 lookahead-distance floor (low-speed guard)
STEER_CLAMP = 0.05          # rad (data |steer|<=0.016; 3x head-room)
ACCEL_CLAMP = 3.0           # m/s^2 (data |accel|<=1.9)
SPEED_TC = 0.5              # s: time-constant of the P speed controller

# --- honest thresholds ------------------------------------------------------ #
DIVERGENCE_M = 5.0          # closed-loop FDE@2s beyond this = "diverged"
A_LON_COMFORT = 2.0         # m/s^2  (comfort, not safety, bounds — documented)
A_LAT_COMFORT = 3.0         # m/s^2
JERK_COMFORT = 2.0          # m/s^3


# --------------------------------------------------------------------------- #
# Harness controller + bicycle (pure functions, vectorised over the batch)     #
# --------------------------------------------------------------------------- #
def wp_to_control(w_look: torch.Tensor, v: torch.Tensor):
    """Ego-frame lookahead waypoint (0.5 s) + current speed -> (steer, accel).

    ``w_look`` [b,2] = (x fwd, y left) target 0.5 s ahead of the CURRENT pose;
    ``v`` [b] current speed. Pure-pursuit curvature kappa = 2 y / L_d^2 ->
    steer = atan(wheelbase * kappa); speed P-controller toward the plan's implied
    forward speed x/0.5. Both clamped to the observed action envelope."""
    x, y = w_look[:, 0], w_look[:, 1]
    ld2 = (x * x + y * y).clamp_min(LD2_FLOOR)
    kappa = 2.0 * y / ld2
    steer = torch.atan(WHEELBASE * kappa).clamp(-STEER_CLAMP, STEER_CLAMP)
    v_target = x / (LOOKAHEAD_STEP * DT)               # implied mean fwd speed
    accel = ((v_target - v) / SPEED_TC).clamp(-ACCEL_CLAMP, ACCEL_CLAMP)
    return steer, accel


def build_action(steer, accel, v, speed_input):
    """(steer, accel) [+ v0=v/SPEED_SCALE] -> action row [b, A] for the predictor."""
    base = torch.stack([steer, accel], dim=-1)          # [b,2]
    if speed_input:
        return torch.cat([base, (v / SPEED_SCALE).unsqueeze(-1)], dim=-1)
    return base


def bicycle_integrate(v0: torch.Tensor, steer_seq: torch.Tensor,
                      accel_seq: torch.Tensor):
    """Integrate the kinematic bicycle from ego-frame origin (0,0,0,v0).

    ``steer_seq``/``accel_seq`` [b,K]; returns (pts [b,K,2], speed [b,K]) where
    pts are ego-frame positions (matching gt_ego_waypoints' frame). Same math as
    kinematic.rollout_bicycle (x fwd, y left), stepped one action at a time."""
    b, K = steer_seq.shape
    x = torch.zeros(b, device=v0.device)
    y = torch.zeros(b, device=v0.device)
    yaw = torch.zeros(b, device=v0.device)
    v = v0.clone()
    pts, spd = [], []
    for k in range(K):
        x = x + v * torch.cos(yaw) * DT
        y = y + v * torch.sin(yaw) * DT
        yaw = yaw + v / WHEELBASE * torch.tan(steer_seq[:, k]) * DT
        v = (v + accel_seq[:, k] * DT).clamp_min(0.0)
        pts.append(torch.stack([x, y], dim=-1))
        spd.append(v)
    return torch.stack(pts, dim=1), torch.stack(spd, dim=1)


# --------------------------------------------------------------------------- #
# The closed loop — plan -> control -> imagine -> drive, per 0.1 s tick         #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def closed_loop_rollout(model, step_readout, states0, aw, v0, speed_input,
                        k=K_MAX):
    """Roll the imagination-in-the-loop closed loop for a batch of windows.

    states0 [b,W,S] encoded latent window; aw [b,W,A] observed action window
    (v0 channel already appended if speed_input); v0 [b] initial speed. Returns a
    dict with the closed bicycle path, the closed grounded path, and per-step
    executed controls (steer/accel/speed) for the comfort + divergence metrics."""
    b = states0.shape[0]
    nav_follow = torch.zeros(b, dtype=torch.long, device=states0.device)
    win_s = states0.clone()
    win_a = aw.clone()
    v = v0.clone()
    steer_seq, accel_seq, grnd_dp = [], [], []
    for _ in range(k):
        # (a) PLAN on the current (imagined) latent window — deploy path, state-only
        ctx = model.strategic_policy(win_s, nav_follow)["ctx"]
        wp = model.tactical_policy(win_s, ctx)["waypoints"]
        w_look = wp[LOOKAHEAD_STEP]                          # [b,2] 0.5 s target
        # (b) CONTROL: waypoint -> (steer, accel)
        steer, accel = wp_to_control(w_look, v)
        a_exec = build_action(steer, accel, v, speed_input)  # [b,A]
        # (c) IMAGINE: step the latent under a_exec (intent-free, on-manifold)
        win_a_exec = win_a.clone()
        win_a_exec[:, -1] = a_exec
        z_next = model.predictor(win_s, win_a_exec)[1]       # 1-step head [b,S]
        grnd_dp.append(step_readout(win_s[:, -1], z_next))   # [b,3] metric dpose
        # (d) DRIVE: record the executed control; bicycle integrated after the loop
        steer_seq.append(steer)
        accel_seq.append(accel)
        v = (v + accel * DT).clamp_min(0.0)                  # track speed for v0
        # slide the windows: append the imagined latent + executed action
        win_s = torch.cat([win_s[:, 1:], z_next.unsqueeze(1)], dim=1)
        win_a = torch.cat([win_a_exec[:, 1:], a_exec.unsqueeze(1)], dim=1)
    steer_seq = torch.stack(steer_seq, dim=1)                # [b,K]
    accel_seq = torch.stack(accel_seq, dim=1)                # [b,K]
    bike_pts, bike_spd = bicycle_integrate(v0, steer_seq, accel_seq)
    grnd_wp = accumulate_se2(torch.stack(grnd_dp, dim=1))    # [b,K,2]
    return {"closed_bike": bike_pts, "closed_grnd": grnd_wp,
            "steer": steer_seq, "accel": accel_seq, "speed": bike_spd}


# --------------------------------------------------------------------------- #
# Collection — one aligned pass building every path per window                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect(model, step_readout, episodes, device, speed_input=False,
            window=WINDOW, stride=STRIDE, batch=BATCH, k=K_MAX):
    """Every stride-window of every episode -> all six paths [N,4,2] + meta.

    Paths (ego frame at the window's last frame, waypoints at steps 5/10/15/20):
      gt, cv, closed_bike, closed_grnd, open_grnd, open_bike.
    Plus per-window: speed, head_deg, eid, and the closed-loop executed controls
    steer/accel/speed [N,K] for comfort + divergence."""
    wp_idx = torch.tensor(IDX, device=device)
    acc = {n: [] for n in ("gt", "cv", "closed_bike", "closed_grnd",
                           "open_grnd", "open_bike", "speed", "head_deg",
                           "steer", "accel", "vseq")}
    eid = []
    for ep in episodes:
        feats = ep.feats
        T = feats.shape[0]
        starts = list(range(0, T - window - k, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(feats[t:t + window])
                              for t in ch]).to(device)
            if fw.dtype == torch.uint8:
                fw = fw.float().div_(255.0)
            elif fw.dtype == torch.float16:
                fw = fw.float()
            aw = torch.stack([ep.actions[t:t + window] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + window:t + window + k]
                              for t in ch]).to(device)
            v0 = ep.poses[last, 3].to(device).float()        # initial speed [b]
            if speed_input:
                v0c = (v0 / SPEED_SCALE)[:, None, None]
                aw = torch.cat([aw, v0c.expand(-1, aw.shape[1], -1)], dim=-1)
                fa = torch.cat([fa, v0c.expand(-1, fa.shape[1], -1)], dim=-1)
            states0 = model.encode_window(fw)                # [b,W,S]

            # --- open-loop grounded (TRUE actions) = the gate rollout ---------
            open_wp, _ = rollout_decode(model.predictor, states0, aw, fa,
                                        step_readout, k)      # [b,k,2]
            # --- open-loop bicycle (TRUE actions) = kinematic fidelity floor --
            true_steer = torch.cat([aw[:, -1:, 0], fa[:, :k - 1, 0]], dim=1)
            true_accel = torch.cat([aw[:, -1:, 1], fa[:, :k - 1, 1]], dim=1)
            open_bike, _ = bicycle_integrate(v0, true_steer, true_accel)
            # --- closed loop (imagination in the loop) -----------------------
            cl = closed_loop_rollout(model, step_readout, states0, aw, v0,
                                     speed_input, k)

            acc["gt"].append(gt_ego_waypoints(ep.poses, last).cpu())
            acc["cv"].append(
                baseline_waypoints(ep.poses, last)["constant_velocity"].cpu())
            acc["closed_bike"].append(cl["closed_bike"][:, wp_idx].cpu())
            acc["closed_grnd"].append(cl["closed_grnd"][:, wp_idx].cpu())
            acc["open_grnd"].append(open_wp[:, wp_idx].cpu())
            acc["open_bike"].append(open_bike[:, wp_idx].cpu())
            acc["speed"].append(v0.cpu())
            acc["head_deg"].append(net_heading_change_deg(ep.poses, last))
            acc["steer"].append(cl["steer"].cpu())
            acc["accel"].append(cl["accel"].cpu())
            acc["vseq"].append(cl["speed"].cpu())
            eid.extend([ep.episode_id] * len(ch))
    out = {n: torch.cat(v).float() for n, v in acc.items()}
    out["eid"] = eid
    return out


# --------------------------------------------------------------------------- #
# Aggregation — bench.py CI protocol (8-split episode-disjoint jackknife)       #
# --------------------------------------------------------------------------- #
def _de(pred, gt):
    """[N,4,2] -> [N,4] per-waypoint Euclidean point error."""
    return torch.linalg.norm(pred - gt, dim=-1)


def _suite(pred, gt):
    """Per-horizon ade@/de@ + fde@2s for a waypoint set [N,4,2]."""
    de = _de(pred, gt)
    out = {}
    for j, (step, name) in enumerate(sorted(HORIZONS_S.items())):
        out[f"de@{name}"] = float(de[:, j].mean())
        out[f"ade@{name}"] = float(de[:, :j + 1].mean())
    out["ade_0_2s"] = out["ade@2s"]
    out["fde@2s"] = float(de[:, -1].mean())
    return out


def _agg(dicts):
    """List of per-split scalar dicts -> {metric: mean/ci95/std over splits}."""
    out = {}
    for kk in dicts[0]:
        v = np.array([d[kk] for d in dicts], dtype=float)
        out[kk] = {"mean": round(float(np.nanmean(v)), 4),
                   "ci95": round(float(1.96 * np.nanstd(v) / max(1, len(v)) ** .5), 4),
                   "std": round(float(np.nanstd(v)), 4)}
    return out


def _jack(vals, eids, splits):
    """Episode-jackknife of a per-window scalar array -> mean/ci95/n/separated.

    'separated' = |mean| - ci95 > 0 (the delta / rate is CI-resolved from 0)."""
    v = np.asarray(vals, dtype=float)
    sm = [float(np.nanmean(v[va])) for _tr, va in splits if len(va)]
    sm = np.asarray(sm)
    mean = float(np.mean(sm))
    ci = float(1.96 * np.std(sm) / max(1, len(sm)) ** 0.5)
    return {"mean": round(mean, 4), "ci95": round(ci, 4), "n": int(v.size),
            "separated": bool(abs(mean) - ci > 0)}


def _speed_labels(speed):
    q = torch.quantile(speed, torch.tensor([1 / 3, 2 / 3]))
    lo, hi = float(q[0]), float(q[1])
    lab = ["low" if float(s) < lo else "high" if float(s) >= hi else "med"
           for s in speed]
    return lab, [round(lo, 3), round(hi, 3)]


def _comfort(win):
    """Comfort / smoothness of the CLOSED-LOOP executed controls.

    accel [N,K] m/s^2, steer [N,K] rad, speed [N,K] m/s. jerk = d(accel)/dt;
    lateral accel = v^2/L*tan(steer). Reports mean magnitudes and the fraction of
    steps that exceed documented COMFORT (not safety) bounds."""
    a = win["accel"]
    st = win["steer"]
    v = win["vseq"]
    jerk = (a[:, 1:] - a[:, :-1]).abs() / DT                 # [N,K-1]
    a_lat = v.pow(2) / WHEELBASE * st.tan().abs()            # [N,K]
    viol_lon = (a.abs() > A_LON_COMFORT).float().mean()
    viol_lat = (a_lat > A_LAT_COMFORT).float().mean()
    viol_jerk = (jerk > JERK_COMFORT).float().mean()
    within_all = ((a.abs() <= A_LON_COMFORT) &
                  (a_lat <= A_LAT_COMFORT)).float().mean()
    return {
        "mean_abs_accel_mps2": round(float(a.abs().mean()), 4),
        "mean_abs_lat_accel_mps2": round(float(a_lat.mean()), 4),
        "mean_abs_jerk_mps3": round(float(jerk.mean()), 4),
        "mean_abs_steer_rad": round(float(st.abs().mean()), 5),
        "frac_steps_exceed_lon_comfort": round(float(viol_lon), 4),
        "frac_steps_exceed_lat_comfort": round(float(viol_lat), 4),
        "frac_steps_exceed_jerk_comfort": round(float(viol_jerk), 4),
        "frac_steps_within_accel_bounds": round(float(within_all), 4),
        "bounds": {"a_lon": A_LON_COMFORT, "a_lat": A_LAT_COMFORT,
                   "jerk": JERK_COMFORT, "kind": "comfort (not safety)"},
    }


def analyze(win, n_splits=8, val_frac=0.2, seed=0):
    """Compose all four honest metric blocks from a collect() window set."""
    eids = win["eid"]
    gt = win["gt"]
    splits = [split_by_episode(eids, val_frac, s)
              for s in range(seed, seed + n_splits)]

    def suite_ci(path):
        return _agg([_suite(win[path][va], gt[va]) for _tr, va in splits])

    # ---- 1. Closed-loop ADE/FDE (+ open-loop + baselines) ------------------ #
    paths = {"closed_bike": "closed_bike (headline: kinematic exec of the plan)",
             "closed_grnd": "closed_grnd (grounded step-readout of imagined roll)",
             "open_grnd": "open_grnd (TRUE actions, gate grounded rollout)",
             "open_bike": "open_bike (TRUE actions, bicycle fidelity FLOOR)",
             "cv": "cv (constant-velocity trivial baseline)"}
    heldout = {p: suite_ci(p) for p in paths}

    # ---- 2. Compounding-error delta (closed MINUS open), grounded path ----- #
    #   apples-to-apples: only the action/latent SOURCE differs; same SE(2)
    #   step-readout path builder. Per-horizon jackknifed mean difference.
    de_cg = _de(win["closed_grnd"], gt)
    de_og = _de(win["open_grnd"], gt)
    de_cb = _de(win["closed_bike"], gt)
    de_ob = _de(win["open_bike"], gt)
    compounding = {"_definition": "closed_grounded point-error - open_grounded "
                   "point-error per horizon (distribution-shift cost the open-loop "
                   "ADE hides); grounded path both sides so only the action source "
                   "differs",
                   "_caveat": "the step-readout is calibrated on TRUE-action rolled "
                   "latents; under self-generated actions the imagined latents drift "
                   "off that manifold, so this delta bundles genuine trajectory drift "
                   "WITH step-readout off-manifold decode degradation. The bicycle "
                   "delta below is controller-clean (no learned decode)."}
    comp_bike = {"_definition": "closed_bicycle_ADE - open_bicycle_ADE per horizon "
                 "(policy+imagination cost above the kinematic fidelity floor)"}
    for j, (step, name) in enumerate(sorted(HORIZONS_S.items())):
        compounding[f"delta@{name}"] = _jack(
            (de_cg[:, j] - de_og[:, j]).numpy(), eids, splits)
        comp_bike[f"delta@{name}"] = _jack(
            (de_cb[:, j] - de_ob[:, j]).numpy(), eids, splits)

    # ---- 3. Stability / comfort / divergence ------------------------------- #
    fde_bike = de_cb[:, -1]                                   # closed_bike @2s
    diverged = (fde_bike > DIVERGENCE_M).float()
    lat_dev = {name: round(float((win["closed_bike"][:, j, 1] -
                                  gt[:, j, 1]).abs().mean()), 4)
               for j, (step, name) in enumerate(sorted(HORIZONS_S.items()))}
    stability = {
        "divergence_rate_gt5m@2s": _jack(diverged.numpy(), eids, splits),
        "divergence_threshold_m": DIVERGENCE_M,
        "lateral_deviation_growth_m": lat_dev,
        "comfort": _comfort(win),
    }

    # ---- 4. Speed-stratified closed-loop drift ----------------------------- #
    spd_lab, spd_thr = _speed_labels(win["speed"])
    by_speed = {}
    for lab in ("low", "med", "high"):
        m = np.array([i for i, l in enumerate(spd_lab) if l == lab])
        if not len(m):
            continue
        sel = torch.tensor(m)
        by_speed[lab] = {
            "n": int(len(m)),
            "closed_bike_ade@2s": round(float(de_cb[sel].mean()), 4),
            "closed_grnd_ade@2s": round(float(de_cg[sel].mean()), 4),
            "open_grnd_ade@2s": round(float(de_og[sel].mean()), 4),
            "open_bike_ade@2s": round(float(de_ob[sel].mean()), 4),
            "closed_minus_open_grnd_ade@2s":
                round(float((de_cg[sel] - de_og[sel]).mean()), 4),
            "divergence_rate_gt5m@2s": round(float(diverged[sel].mean()), 4),
            "mean_speed_mps": round(float(win["speed"][sel].mean()), 3),
        }

    cb2 = heldout["closed_bike"]["ade_0_2s"]
    summary = {
        "closed_bike_ade@2s": cb2["mean"], "closed_bike_ade@2s_ci95": cb2["ci95"],
        "closed_bike_fde@2s": heldout["closed_bike"]["fde@2s"]["mean"],
        "closed_grnd_ade@2s": heldout["closed_grnd"]["ade_0_2s"]["mean"],
        "open_grnd_ade@2s": heldout["open_grnd"]["ade_0_2s"]["mean"],
        "open_bike_ade@2s_kinematic_floor": heldout["open_bike"]["ade_0_2s"]["mean"],
        "cv_ade@2s": heldout["cv"]["ade_0_2s"]["mean"],
        "closed_minus_open_grnd_de@2s": compounding["delta@2s"]["mean"],
        "closed_minus_open_bike_de@2s": comp_bike["delta@2s"]["mean"],
        "divergence_rate_gt5m@2s": stability["divergence_rate_gt5m@2s"]["mean"],
        "high_vs_low_speed_closed_bike_ade@2s": [
            by_speed.get("low", {}).get("closed_bike_ade@2s"),
            by_speed.get("high", {}).get("closed_bike_ade@2s")],
    }

    return {
        "n_windows": int(gt.shape[0]),
        "n_episodes": len(set(eids)),
        "summary": summary,
        "protocol": {"window": WINDOW, "stride": STRIDE, "hz": 10,
                     "wp_steps": list(WP_STEPS), "K_steps": K_MAX,
                     "n_splits": n_splits, "val_frac": val_frac,
                     "ci": "8-split episode-disjoint jackknife (bench.py)",
                     "nav": "follow (deploy-realistic, no route command)",
                     "operative_step": "intent-free (deployed regime)"},
        "closedloop_ade_fde": {
            "_path_legend": paths,
            "_headline": "closed_bike is the headline closed-loop path; "
                         "closed_grnd is the model's own metric decode",
            "heldout": heldout},
        "compounding_error_grounded": compounding,
        "compounding_error_bicycle": comp_bike,
        "stability": stability,
        "speed_stratified": {"thresholds_mps": spd_thr, "by_speed": by_speed,
                             "_read": "is closed-loop drift worse at high speed "
                             "(the known longitudinal weakness)?"},
        "limitations": [
            "NO collision / drivable-area / PDM — no HD map or agent boxes in our "
            "data; this is a drift/stability closed loop, not a safety one.",
            "SELF-REFERENTIAL: the world model is both state estimator and "
            "simulator; failures it cannot imagine are invisible (needs an "
            "external photoreal sim to cure).",
            "waypoint->control + bicycle are a HARNESS controller (not the model); "
            "open_bike is its kinematic fidelity floor so drift stays attributable.",
            "dataset actions correlate ~0.87 with bicycle-consistent controls "
            "(CAN signals) — residual shows up in open_bike.",
            "open-loop L2 is a weak claim (arXiv:2605.00066); the closed-loop delta "
            "is the point, not the absolute open-loop number.",
        ],
    }


# --------------------------------------------------------------------------- #
# Standalone runner                                                            #
# --------------------------------------------------------------------------- #
def run_and_save(key, device="cuda", episodes=40,
                 out_dir="/root/taniteval/results"):
    """Load an arm, run the closed loop, write results/closedloop_<key>.json.

    Reuses loaders.load + data (read-only) exactly like bench/planning panels."""
    import json
    import time
    from pathlib import Path
    from taniteval import data, loaders
    from taniteval.registry import MODELS
    entry = [m for m in MODELS if m["key"] == key]
    if not entry:
        print(f"[cl] unknown arm {key}", flush=True)
        return {"key": key, "skipped": "unknown arm"}
    entry = entry[0]
    t0 = time.time()
    L = loaders.load(entry, device)
    model = L["model"]
    if not L["traj_capable"] or getattr(model, "tactical_policy", None) is None:
        msg = ("arm has no operative step-readout + tactical policy — the "
               "imagination-in-the-loop harness needs a WorldModel 4-brain arm "
               "(flagship / REF-A); REF-B is a direct planner, not applicable")
        print(f"[cl] {key}: SKIP ({msg})", flush=True)
        return {"key": key, "skipped": msg}
    files = data.list_val_episodes(
        "/root/valdata/physicalai-val-0c5f7dac3b11", episodes)
    if entry.get("train_ids"):                       # replicate runner leak guard
        from tanitad.data.mixing import load_episode
        tid = set(Path(entry["train_ids"]).read_text().split())
        files = [f for f in files
                 if str(load_episode(str(f), mmap=True).episode_id) not in tid]
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], device))
    win = collect(model, L["step_readout"], eps, device,
                  speed_input=bool(entry.get("speed_input")))
    res = analyze(win)
    res["model"] = {k: entry.get(k) for k in
                    ("key", "name", "arch", "encoder", "speed_input")}
    res["ckpt_step"] = L["step"]
    res["wall_s"] = round(time.time() - t0, 1)
    outp = Path(out_dir) / f"closedloop_{key}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2, default=str))
    ho = res["closedloop_ade_fde"]["heldout"]
    cb, og = ho["closed_bike"], ho["open_grnd"]
    d2 = res["compounding_error_grounded"]["delta@2s"]
    dv = res["stability"]["divergence_rate_gt5m@2s"]
    print(f"[cl] {key} step={L['step']} n={res['n_windows']}: "
          f"closed_bike ade@2s={cb['ade_0_2s']['mean']:.3f}±{cb['ade_0_2s']['ci95']:.3f} "
          f"fde@2s={cb['fde@2s']['mean']:.3f} | open_grnd ade@2s={og['ade_0_2s']['mean']:.3f} "
          f"| closed-open Δ@2s={d2['mean']:.3f}±{d2['ci95']:.3f} "
          f"| diverge={dv['mean']:.1%} ({res['wall_s']}s) -> {outp.name}", flush=True)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser("taniteval.closedloop")
    ap.add_argument("--arm", default="flagship-30k",
                    help="registry key (default flagship-30k = v1 FINAL)")
    ap.add_argument("--all-flagships", action="store_true",
                    help="run flagship-30k, flagship-speed, flagship-nospeed")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    keys = (["flagship-30k", "flagship-speed", "flagship-nospeed"]
            if a.all_flagships else [a.arm])
    for key in keys:
        try:
            run_and_save(key, a.device, a.episodes)
        except Exception as e:
            import traceback
            print(f"[cl] {key} FAILED: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
