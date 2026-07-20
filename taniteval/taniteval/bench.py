"""TanitEval — benchmark suite.

Computes, fresh from rollout windows, for model AND the CV baseline:
  * per-horizon ADE/DE @ {0.5, 1, 1.5, 2} s          (established gate metrics)
  * FDE@2s, RMSE, miss-rate@2m                        (hub trajectory seam)
  * TMS-openloop (trajectory smoothness, hub formula on the predicted path)
  * 8-split episode-disjoint jackknife -> mean ± CI95 (established protocol)
  * strata: curvature (straight/gentle/sharp) and speed terciles
Open-loop numbers are weak claims per arXiv:2605.00066 — recorded in the output.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
sys.path.insert(0, "/root/taniteval")

from driving_diagnostic import (WP_STEPS, baseline_waypoints,  # noqa: E402
                                curvature_bucket, gt_ego_waypoints,
                                net_heading_change_deg, scalar_metrics,
                                _ego, _wrap)
from tanitad.eval.gates import split_by_episode  # noqa: E402
from tanitad.models.metric_dynamics import rollout_decode  # noqa: E402
from tanitad.models.readout import RidgeProbe  # noqa: E402
from taniteval.tanitad_metrics import (ade, fde, miss_rate,  # noqa: E402
                                       rmse_xy, TMS_ALPHA, TMS_BETA)
from taniteval import rollout as _rollout  # noqa: E402  (canonical ego append)

HORIZONS_S = {5: "0.5s", 10: "1s", 15: "1.5s", 20: "2s"}
DT = 0.1

# --- diagnostic-panel constants (kinematic floor / ego-status ceiling) ------ #
SPEED_SCALE = 10.0              # v0 action-channel scale (matches every trainer)
WINDOW = 8                      # rollout.collect default window (parity)
STRIDE = 8                      # rollout.collect default stride (parity)
K_MAX = max(WP_STEPS)           # 20 steps = 2 s @ 10 Hz
EGO_HIST = 4                    # ego-status: past-K ego-frame displacements
RIDGE_ALPHAS = (1.0, 10.0, 100.0)   # ceiling ridge alpha ladder (dd parity)
KFOLD = 5                       # episode-disjoint folds for held-out ceilings
FLOOR_BASELINES = ("constant_velocity", "go_straight", "constant_yaw_rate")
STRAIGHT_TRIVIAL_SKILL = 3.0    # skill_score<=this on straights => near-trivial


def tms_openloop(pred, alpha=TMS_ALPHA, beta=TMS_BETA):
    """Hub TMS formula applied to predicted waypoint paths [N, H, 2].

    jerk = |d3 pos/dt3| (longitudinal proxy: speed 2nd derivative), steer_rate =
    |d heading/dt|; integrals over the predicted horizon. In (0, 1]."""
    p = pred.numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
    if p.shape[1] < 4:
        return float("nan")
    d = np.diff(p, axis=1)                      # [N, H-1, 2] step displacements
    v = np.linalg.norm(d, axis=-1) / DT         # speeds
    acc = np.diff(v, axis=1) / DT
    jerk = np.abs(np.diff(acc, axis=1) / DT)    # [N, H-3]
    hd = np.arctan2(d[..., 1], d[..., 0])
    sr = np.abs(np.diff(np.unwrap(hd, axis=1), axis=1) / DT)
    ji = jerk.sum(axis=1) * DT
    si = sr.sum(axis=1) * DT
    return float(np.mean(1.0 / (1.0 + alpha * ji + beta * si)))


def _suite(pred, gt):
    """All scalar metrics for waypoint sets [N, 4, 2] (steps 5/10/15/20)."""
    de = torch.linalg.norm(pred - gt, dim=-1)        # [N, 4]
    out = {}
    for j, (step, name) in enumerate(sorted(HORIZONS_S.items())):
        out[f"de@{name}"] = float(de[:, j].mean())
        out[f"ade@{name}"] = float(de[:, :j + 1].mean())
    out["ade_0_2s"] = out["ade@2s"]
    out["fde@2s"] = fde(pred.numpy(), gt.numpy())
    out["rmse"] = rmse_xy(pred.numpy(), gt.numpy())
    out["miss_rate@2m"] = miss_rate(pred.numpy(), gt.numpy())
    out["tms_openloop"] = tms_openloop(pred)
    return out


def _agg(dicts):
    keys = dicts[0].keys()
    out = {}
    for k in keys:
        v = np.array([d[k] for d in dicts], dtype=float)
        out[k] = {"mean": round(float(np.nanmean(v)), 4),
                  "ci95": round(float(1.96 * np.nanstd(v) / max(1, len(v)) ** .5), 4),
                  "std": round(float(np.nanstd(v)), 4)}
    return out


def _strata(labels, de_m, de_c):
    out = {}
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        if not idx:
            continue
        sel = torch.tensor(idx)
        out[lab] = {
            "model_ade@1s": round(float(de_m[sel, :2].mean()), 4),
            "cv_ade@1s": round(float(de_c[sel, :2].mean()), 4),
            "model_ade@2s": round(float(de_m[sel].mean()), 4),
            "cv_ade@2s": round(float(de_c[sel].mean()), 4),
            "n": len(idx)}
    return out


def run(data, n_splits=8, val_frac=0.2, seed=0):
    """Full benchmark over rollout windows. Returns the results dict."""
    pred, gt, cv = data["pred"], data["gt"], data["cv"]
    splits = [split_by_episode(data["eid"], val_frac, s)
              for s in range(seed, seed + n_splits)]
    model_split = [_suite(pred[va], gt[va]) for _t, va in splits]
    cv_split = [_suite(cv[va], gt[va]) for _t, va in splits]
    de_m = torch.linalg.norm(pred - gt, dim=-1)
    de_c = torch.linalg.norm(cv - gt, dim=-1)
    curv = [curvature_bucket(float(h)) for h in data["head_deg"]]
    q = torch.quantile(data["speed"], torch.tensor([1 / 3, 2 / 3]))
    spd = ["low" if float(s) < float(q[0]) else
           "high" if float(s) >= float(q[1]) else "med"
           for s in data["speed"]]
    beats = _agg(model_split)["ade_0_2s"]["mean"] < _agg(cv_split)["ade_0_2s"]["mean"]
    return {
        "n_windows": int(pred.shape[0]),
        "heldout": {"model": _agg(model_split), "cv": _agg(cv_split)},
        "full_set": {"model": _suite(pred, gt), "cv": _suite(cv, gt)},
        "beats_cv_ade_0_2s": bool(beats),
        "by_curvature": _strata(curv, de_m, de_c),
        "by_speed": _strata(spd, de_m, de_c),
        "protocol": {"n_splits": n_splits, "val_frac": val_frac,
                     "wp_steps": data.get("wp_steps"),
                     "claim_strength": "open-loop / weak (arXiv:2605.00066)"},
    }


# =========================================================================== #
# DIAGNOSTIC PANEL — kinematic FLOOR + ego-status CEILING + skill_score        #
# (Benchmarks & Eval P0 FLEET DIRECTIVE, 2026-07-17, item 1)                   #
#                                                                              #
# The vetted kinematics/L2 math is IMPORTED from scripts/driving_diagnostic.py #
# (baseline_waypoints / gt_ego_waypoints / scalar_metrics / RidgeProbe) — not  #
# reinvented. Everything here is ADDITIVE; run()/_suite()/_strata() unchanged. #
# =========================================================================== #
def ego_status_features(poses, last, hist=EGO_HIST):
    """Perception-free ego-status vector per window — the AD-MLP "ego-status is
    all you need" input (arXiv:2305.10430 repro). No pixels, no latent: only the
    ego kinematic history at ``last``. [b, 2*hist+4].

    Cols: ``hist`` past ego-frame step displacements (captures v0 + curvature
    history), then scalar v0, a0 (accel), omega0 (yaw-rate), alpha0 (yaw-accel).
    ``last`` must be >= hist+1 (rollout windows have last>=WINDOW-1=7)."""
    disp = torch.cat([_ego(poses[last - j, :2] - poses[last - j - 1, :2],
                           poses[last, 2]) for j in range(hist)], dim=-1)
    v0 = (poses[last, :2] - poses[last - 1, :2]).norm(dim=-1, keepdim=True)
    vm1 = (poses[last - 1, :2] - poses[last - 2, :2]).norm(dim=-1, keepdim=True)
    om0 = _wrap(poses[last, 2] - poses[last - 1, 2]).unsqueeze(-1)
    om1 = _wrap(poses[last - 1, 2] - poses[last - 2, 2]).unsqueeze(-1)
    return torch.cat([disp, v0, v0 - vm1, om0, om0 - om1], dim=-1)


@torch.no_grad()
def collect_full(model, step_readout, episodes, device, speed_input=False,
                 need_states=True, window=WINDOW, stride=STRIDE, batch=8,
                 fwd_k=K_MAX, yaw_input=False, dyn_input=False):
    """rollout.collect (verbatim protocol) PLUS the diagnostic aux in ONE aligned
    pass: all 3 kinematic baselines, ego-status features, and the last-frame
    latent state. Returns a dict that is a strict superset of rollout.collect's
    (pred/gt/cv/eid/speed/head_deg/wp_steps) so bench.run() consumes it as-is.
    Extra keys: go_straight, constant_yaw_rate, ego_status, states."""
    wp_idx = torch.tensor([k - 1 for k in WP_STEPS])
    acc = {k: [] for k in ("pred", "gt", "constant_velocity", "go_straight",
                           "constant_yaw_rate", "ego_status", "states",
                           "speed", "head_deg")}
    eid = []
    for ep in episodes:
        feats = ep.feats
        T = min(feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        starts = list(range(0, T - window - K_MAX, stride))
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
            fa = torch.stack([ep.actions[t + window:t + window + fwd_k]
                              for t in ch]).to(device)
            aw, fa = _rollout.append_ego(aw, fa, ep.poses, last, speed_input,
                                         yaw_input, dyn_input, device)
            states = model.encode_window(fw)                        # [b, W, S]
            wp_full, _ = rollout_decode(model.predictor, states, aw, fa,
                                        step_readout, fwd_k)
            acc["pred"].append(
                wp_full.index_select(1, wp_idx.to(device)).cpu().float())
            acc["gt"].append(gt_ego_waypoints(ep.poses, last))
            bp = baseline_waypoints(ep.poses, last)
            for n in FLOOR_BASELINES:
                acc[n].append(bp[n])
            acc["ego_status"].append(ego_status_features(ep.poses, last))
            acc["states"].append(states[:, -1].cpu().float() if need_states
                                 else torch.empty(len(ch), 0))
            acc["speed"].append(ep.poses[last, 3])
            acc["head_deg"].append(net_heading_change_deg(ep.poses, last))
            eid.extend([ep.episode_id] * len(ch))
    out = {k: torch.cat(v).float() for k, v in acc.items()}
    out["cv"] = out["constant_velocity"]              # rollout.collect alias
    out["eid"] = eid
    out["wp_steps"] = list(WP_STEPS)
    return out


def _de(pred, gt):
    """[N,4,2] pred/gt -> [N,4] per-waypoint Euclidean error."""
    return torch.linalg.norm(pred - gt, dim=-1)


def _speed_labels(speed):
    q = torch.quantile(speed, torch.tensor([1 / 3, 2 / 3]))
    lo, hi = float(q[0]), float(q[1])
    lab = ["low" if float(s) < lo else "high" if float(s) >= hi else "med"
           for s in speed]
    return lab, [round(lo, 3), round(hi, 3)]


def _l2_conventions(de):
    """Both open-loop L2 averaging conventions from a [N,4] point-error tensor.

    cumulative : mean over ALL waypoints<=T (ST-P3 style)      -> ade@Ts
    endpoints  : mean over the horizon endpoints only          -> mean(de@Ts)
    The nuScenes literature reports both; they diverge at intermediate T. We
    surface the per-horizon breakdown (scalar_metrics, the gate convention) too."""
    m = scalar_metrics(de)
    endpoints = float(sum(m[f"de@{k/10:g}s"] for k in WP_STEPS) / len(WP_STEPS))
    return {
        "cumulative_avg_0_2s": round(m["ade_0_2s"], 4),   # ade convention
        "endpoint_avg": round(endpoints, 4),              # de convention
        "l2@1s": round(m["de@1s"], 4), "l2@2s": round(m["de@2s"], 4),
        "ade@1s": round(m["ade@1s"], 4), "ade@2s": round(m["ade@2s"], 4),
    }


def _kfold_ridge(feats, gt_flat, eid, alpha, k=KFOLD, seed=0):
    """Episode-disjoint k-fold held-out ridge readout. Every window is predicted
    exactly once (from a fold that excludes its episode). Returns held-out pred
    [N,8] and the mean in-fold train R^2."""
    import random
    N = feats.shape[0]
    uniq = sorted(set(eid))
    kk = max(2, min(k, len(uniq)))
    rng = random.Random(seed)
    order = uniq[:]
    rng.shuffle(order)
    grp = {e: idx % kk for idx, e in enumerate(order)}
    folds = [[] for _ in range(kk)]
    for i, e in enumerate(eid):
        folds[grp[e]].append(i)
    pred = torch.zeros(N, gt_flat.shape[1], dtype=torch.float32)
    r2s = []
    for va in folds:
        if not va:
            continue
        va_t = torch.tensor(va)
        mask = torch.ones(N, dtype=torch.bool)
        mask[va_t] = False
        tr_t = mask.nonzero(as_tuple=True)[0]
        if len(tr_t) < feats.shape[1] // 4 or len(tr_t) < 8:
            tr_t = torch.arange(N)          # degenerate: fall back to in-sample
        pr = RidgeProbe(alpha=alpha).fit(feats[tr_t], gt_flat[tr_t])
        pred[va_t] = pr.predict(feats[va_t]).float()
        r2s.append(pr.r2(feats[tr_t], gt_flat[tr_t]))
    return pred, float(np.mean(r2s)) if r2s else float("nan")


def _best_ridge_ceiling(feats, gt, eid, seed=0):
    """Held-out ridge readout, alpha chosen by lowest held-out ADE. Returns
    (pred_wp [N,4,2], de [N,4], chosen_alpha, fit_r2)."""
    gt_flat = gt.reshape(gt.shape[0], -1)
    best = None
    for a in RIDGE_ALPHAS:
        pred_flat, r2 = _kfold_ridge(feats, gt_flat, eid, a, seed=seed)
        pred_wp = pred_flat.reshape(-1, len(WP_STEPS), 2)
        de = _de(pred_wp, gt)
        ade = float(de.mean())
        if best is None or ade < best[0]:
            best = (ade, pred_wp, de, a, r2)
    _, pred_wp, de, a, r2 = best
    return pred_wp, de, a, round(r2, 4)


def kinematic_floor(win):
    """Best-of-3 trivial floor (CV / go-straight / CTRV), overall + speed-gated +
    curvature-gated. Per-window floor = min error across the three baselines (the
    strongest trivial predictor available at that window)."""
    gt = win["gt"]
    have = [n for n in FLOOR_BASELINES if n in win]
    de_by = {n: _de(win[n], gt) for n in have}
    floor_de = torch.stack([de_by[n] for n in have]).min(0).values  # [N,4]
    win_counts = {n: 0 for n in have}
    if len(have) > 1:
        which = torch.stack([de_by[n].mean(1) for n in have]).argmin(0)
        for j, n in enumerate(have):
            win_counts[n] = int((which == j).sum())
    spd_lab, spd_thr = _speed_labels(win["speed"])
    curv_lab = [curvature_bucket(float(h)) for h in win["head_deg"]]

    def strata(labels, de):
        out = {}
        for lab in sorted(set(labels)):
            idx = torch.tensor([i for i, l in enumerate(labels) if l == lab])
            out[lab] = {"ade_0_2s": round(float(de[idx].mean()), 4),
                        "n": int(len(idx))}
        return out
    return {
        "per_baseline_ade_0_2s": {n: round(float(de_by[n].mean()), 4)
                                  for n in have},
        "best_of_3_ade_0_2s": round(float(floor_de.mean()), 4),
        "best_of_3_l2": _l2_conventions(floor_de),
        "which_baseline_wins": win_counts,
        "by_speed": strata(spd_lab, floor_de),          # speed-gated (headline)
        "by_curvature": strata(curv_lab, floor_de),
        "speed_tertile_thresholds_mps": spd_thr,
        "_floor_de": floor_de,                          # internal (stripped)
        "baselines_used": have,
    }


def _skill_strata(labels, model_de, floor_de, ceil_de):
    """Per-stratum skill_score = model_L2 / denominator (dimensionless; <1 the
    model beats the denominator). Reported vs the kinematic floor and vs the
    learned readout ceiling, per stratum. ADE-0-2s convention."""
    out = {}
    for lab in sorted(set(labels)):
        idx = torch.tensor([i for i, l in enumerate(labels) if l == lab])
        mL = float(model_de[idx].mean())
        fL = float(floor_de[idx].mean())
        row = {"model_l2": round(mL, 4), "floor_l2": round(fL, 4),
               "skill_vs_floor": round(mL / max(fL, 1e-9), 3), "n": int(len(idx))}
        if ceil_de is not None:
            cL = float(ceil_de[idx].mean())
            row["ceiling_l2"] = round(cL, 4)
            row["skill_vs_ceiling"] = round(mL / max(cL, 1e-9), 3)
        out[lab] = row
    return out


def diagnostic(win, seed=0):
    """Compose the FLOOR + CEILING + skill_score panel from a collect_full() win
    (or any win carrying the aux keys). Degrades gracefully: latent/ego ceilings
    are emitted null-with-note if their inputs are absent. ADDITIVE — returns a
    self-contained block; callers merge it beside run()'s output."""
    gt = win["gt"]
    model_de = _de(win["pred"], gt)
    n_ep = len(set(win["eid"]))
    low_conf = int(win["pred"].shape[0]) < 200 or n_ep < KFOLD + 3
    floor = kinematic_floor(win)
    floor_de = floor.pop("_floor_de")

    # --- ego-status ceiling (AD-MLP repro; perception-free learned readout) --- #
    ego_block, ego_de = None, None
    if "ego_status" in win and win["ego_status"].shape[1] > 0:
        ego_wp, ego_de, ego_a, ego_r2 = _best_ridge_ceiling(
            win["ego_status"], gt, win["eid"], seed)
        ctrv_ade = (round(float(_de(win["constant_yaw_rate"], gt).mean()), 4)
                    if "constant_yaw_rate" in win else None)
        ego_ade = round(float(ego_de.mean()), 4)
        beats = (ctrv_ade is not None and ego_ade < ctrv_ade)
        ego_block = {
            "held_out_ade_0_2s": ego_ade, "l2": _l2_conventions(ego_de),
            "ridge_alpha": ego_a, "fit_r2": ego_r2,
            "ctrv_ade_0_2s": ctrv_ade, "ridge_beats_ctrv": bool(beats),
            "note": ("learned ego-status readout BEATS hand-coded CTRV — a "
                     "learned kinematic shortcut exists"
                     if beats else
                     "learned ego-status readout does NOT beat CTRV — no "
                     "learned shortcut over trivial kinematics; keep CTRV as "
                     "the bar (falsifier)"),
            "input": "ego kinematic history only — NO pixels/latent (AD-MLP)",
        }

    # --- latent ceiling (best linear readout of the world-model latent) ------ #
    lat_block, lat_de = None, None
    if "states" in win and win["states"].shape[1] > 0:
        lat_wp, lat_de, lat_a, lat_r2 = _best_ridge_ceiling(
            win["states"], gt, win["eid"], seed)
        lat_block = {
            "held_out_ade_0_2s": round(float(lat_de.mean()), 4),
            "l2": _l2_conventions(lat_de), "ridge_alpha": lat_a,
            "fit_r2": lat_r2, "state_dim": int(win["states"].shape[1]),
            "note": ("best linear readout of the frozen latent (upper bound on "
                     "what the operative head could extract WITHOUT the "
                     "action-privilege the grounded rollout enjoys)"),
        }

    # --- skill_score per stratum (headline: speed-gated) --------------------- #
    ceil_de = lat_de if lat_de is not None else ego_de
    spd_lab, _ = _speed_labels(win["speed"])
    curv_lab = [curvature_bucket(float(h)) for h in win["head_deg"]]
    skill = {"denominator_ceiling": ("latent" if lat_de is not None else
                                     "ego_status" if ego_de is not None else None),
             "by_speed": _skill_strata(spd_lab, model_de, floor_de, ceil_de),
             "by_curvature": _skill_strata(curv_lab, model_de, floor_de, ceil_de)}

    # --- falsifier read-outs (report honestly) ------------------------------- #
    straight = skill["by_curvature"].get("straight", {})
    straight_skill = straight.get("skill_vs_floor")
    falsifiers = {
        "ridge_ceiling_vs_ctrv": (ego_block["note"] if ego_block else
                                  "no ego-status ceiling computed"),
        "skill_on_straights": {
            "skill_vs_floor": straight_skill,
            "near_trivial_competitive": (straight_skill is not None and
                                         straight_skill <= STRAIGHT_TRIVIAL_SKILL),
            "threshold": STRAIGHT_TRIVIAL_SKILL,
            "note": ("model within {}x of the trivial floor on straights — "
                     "near-trivial-competitive".format(STRAIGHT_TRIVIAL_SKILL)
                     if (straight_skill is not None and
                         straight_skill <= STRAIGHT_TRIVIAL_SKILL)
                     else "model clears the near-trivial bar on straights"),
        },
    }
    return {
        "n_windows": int(win["pred"].shape[0]),
        "n_episodes": n_ep,
        "low_confidence": bool(low_conf),
        "low_confidence_note": (
            "few windows/episodes (<200 win or <8 ep) — the held-out ridge "
            "ceilings overfit and skill_score CIs are wide; treat as indicative"
            if low_conf else "decision-grade sample"),
        "model_ade_0_2s": round(float(model_de.mean()), 4),
        "model_l2": _l2_conventions(model_de),
        "kinematic_floor": floor,
        "ego_status_ceiling": ego_block,
        "latent_ceiling": lat_block,
        "open_loop_l2_both_conventions": {
            "model": _l2_conventions(model_de),
            "floor_best_of_3": floor["best_of_3_l2"],
            "ego_status_ceiling": ego_block["l2"] if ego_block else None,
            "latent_ceiling": lat_block["l2"] if lat_block else None,
            "convention_note": "cumulative = mean over waypoints<=T (ST-P3); "
                               "endpoint = mean of per-horizon final errors",
        },
        "skill_score": skill,
        "falsifiers": falsifiers,
        "protocol": {"window": WINDOW, "stride": STRIDE,
                     "wp_steps": list(WP_STEPS), "kfold": KFOLD,
                     "ridge_alphas": list(RIDGE_ALPHAS),
                     "action_privilege_caveat":
                     "model rollout uses TRUE future actions; ceilings/floor do "
                     "not — skill_vs_ceiling<1 does not imply the head beats a "
                     "fair readout"},
    }


def run_diagnostic_panel(entry, device="cuda", episodes=40,
                         out_dir="/root/taniteval/results", need_states=True):
    """Standalone callable (no runner edit): load an arm, collect the full
    aligned window set, and write results/diag_<key>.json with the FLOOR /
    CEILING / skill_score panel. Reuses loaders.load + data (read-only)."""
    import json
    import time
    from pathlib import Path
    sys.path.insert(0, "/root/taniteval")
    from taniteval import data, loaders             # read-only use (adf3 owns)
    key = entry["key"]
    t0 = time.time()
    L = loaders.load(entry, device)
    if not L["traj_capable"]:
        return {"key": key, "skipped": "no trajectory surface (planner arm) — "
                "floor/ceiling need a grounded rollout to score model_l2"}
    files = data.list_val_episodes(
        "/root/valdata/physicalai-val-0c5f7dac3b11", episodes)
    if entry.get("train_ids"):                      # replicate runner leak guard
        from tanitad.data.mixing import load_episode
        tid = set(Path(entry["train_ids"]).read_text().split())
        files = [f for f in files
                 if str(load_episode(str(f), mmap=True).episode_id) not in tid]
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], device))
    win = collect_full(L["model"], L["step_readout"], eps, device,
                       speed_input=bool(entry.get("speed_input")),
                       need_states=need_states,
                       yaw_input=bool(entry.get("yaw_input")),
                       dyn_input=bool(entry.get("dyn_input")))
    diag = diagnostic(win)
    res = {"model": {k: entry.get(k) for k in
                     ("key", "name", "arch", "encoder", "speed_input")},
           "ckpt_step": L["step"], "n_windows": diag["n_windows"],
           "wall_s": round(time.time() - t0, 1), "diagnostic": diag}
    outp = Path(out_dir) / f"diag_{key}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2, default=str))
    f, s = diag["kinematic_floor"], diag["skill_score"]["by_speed"]
    print(f"[diag] {key} step={L['step']} n={diag['n_windows']}: "
          f"model={diag['model_ade_0_2s']} floor={f['best_of_3_ade_0_2s']} "
          f"ego-ceil={diag['ego_status_ceiling']['held_out_ade_0_2s'] if diag['ego_status_ceiling'] else '—'} "
          f"lat-ceil={diag['latent_ceiling']['held_out_ade_0_2s'] if diag['latent_ceiling'] else '—'} "
          f"-> diag_{key}.json ({res['wall_s']}s)", flush=True)
    return res


def main():
    import argparse
    sys.path.insert(0, "/root/taniteval")
    from taniteval.registry import MODELS
    ap = argparse.ArgumentParser("taniteval.bench")
    ap.add_argument("--model", help="registry key; omit with --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    keys = ([m["key"] for m in MODELS] if a.all else [a.model])
    for key in keys:
        entry = [m for m in MODELS if m["key"] == key]
        if not entry:
            print(f"[diag] unknown model {key}", flush=True)
            continue
        try:
            r = run_diagnostic_panel(entry[0], a.device, a.episodes)
            if r.get("skipped"):
                print(f"[diag] {key}: skipped ({r['skipped']})", flush=True)
        except Exception as e:
            import traceback
            print(f"[diag] {key} FAILED: {type(e).__name__}: {str(e)[:160]}",
                  flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
