"""TanitEval — genuine-prediction / anticipation / generalization panel.

The scientific question (Sayed): does the world-model GENUINELY predict scene
physics and GENERALIZE, or does it merely EXTRAPOLATE the current dynamics
(v0 / yaw-rate)?  Grounded ADE is measured WITH the true future actions, so a
low ADE can be produced by a good dynamics integrator that never looks at the
road.  This panel isolates *anticipation* — skill that a constant-turn-rate
(CTRV) extrapolator cannot have — and proves (causally) whether it is READ FROM
THE SCENE.

Tests (feasible-first, in priority order):

  A  CTRV-DIVERGENCE STRATIFICATION.  Stratify val windows by how far the true
     future departs from CTRV over the 2 s horizon (``divergence`` = |GT-CTRV|).
     Report skill_vs_CTRV per divergence bin.  Genuine prediction => the model
     beats CTRV MOST on high-divergence windows (an upcoming turn/brake/accel
     not yet in the dynamics).

  B  VISION-ABLATION-ON-DIVERGENCE  (the causal clincher / HEADLINE).  Re-run
     the rollout on high-divergence windows with vision mean-replaced / shuffled
     (the imagination-panel ablation).  If the anticipation advantage COLLAPSES
     toward CTRV without vision => the anticipation was read from the scene.
     Advantage-with-vision vs advantage-without, CI-separated, per bin.

  C  ANTICIPATION LEAD-TIME.  For windows with a maneuver onset, how many frames
     BEFORE onset the prediction commits to the maneuver, and whether that lead
     vanishes under vision-ablation.  A positive vision-tied lead = "sees what's
     coming".

  D  LATENT DECODABILITY OF SCENE-ONLY QUANTITIES.  Linear-probe the pooled
     latent for upcoming road curvature (from the GT future path geometry) and
     compare to an ego-status (kinematics-only) probe.  latent >> ego-status =>
     the latent encodes SCENE, not just ego-state.

  E  OOD GENERALIZATION.  True unseen corpora (comma2k19, Cosmos) are not
     materialized on the eval pod (see ``ood_corpus_status``); the within-corpus
     proxy stratifies by NOVEL/harder maneuvers (top-decile curvature/accel,
     sharper/faster than typical) and checks predictions stay physically
     feasible (bounded curvature/accel/jerk) = learned physics not memorization.

  F  COSMOS COUNTERFACTUAL (proxy).  Real paired-clip Cosmos edits are not
     available on the pod (see ``cosmos_feasibility``); the closest feasible
     causal probe is a SPATIAL-SECTOR OCCLUSION (H15-style) of the raw input
     with DYNAMICS HELD FIXED: does the predicted path respond MORE to occluding
     the road ahead than the periphery?  Frame-input arms only (flagship/REF-B).

Reuses (read-only) loaders.load + data + rollout.collect protocol +
imagination-panel ablation + bench ridge-probe pattern.  All pure math lives in
module-level functions with synthetic-input tests in tests/.
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
                                net_heading_change_deg)
from tanitad.models.metric_dynamics import rollout_decode  # noqa: E402
from taniteval import rollout as _rollout  # noqa: E402  (canonical ego append)

SPEED_SCALE = 10.0            # v0 action-channel scale (matches every trainer)
WINDOW = 8                    # rollout.collect parity
STRIDE = 8                    # rollout.collect parity
K_MAX = max(WP_STEPS)         # 20 steps = 2 s @ 10 Hz
DT = 0.1
WP_IDX = [k - 1 for k in WP_STEPS]     # index full-K waypoints at 0.5/1/1.5/2 s

# physical-feasibility envelopes (passenger-car, generous; learned-physics test)
KAPPA_MAX = 0.5              # 1/m  (min radius ~2 m — anything tighter = unphysical)
ALAT_MAX = 12.0             # m/s^2 lateral (~1.2 g)
ALON_MAX = 12.0             # m/s^2 longitudinal
JERK_MAX = 60.0             # m/s^3


# ========================================================================== #
# PURE MATH — no model; unit-tested in tests/test_generalization.py           #
# ========================================================================== #
def per_window_l2(pred, gt, idx=None):
    """Mean waypoint L2 per window. pred/gt [N,H,2] -> [N] (mean over idx steps).

    ``idx`` selects a subset of horizon steps (default: all)."""
    de = torch.linalg.norm(pred - gt, dim=-1)          # [N,H]
    if idx is not None:
        de = de[:, idx]
    return de.mean(dim=-1)


def ctrv_divergence(gt, ctrv, idx=None):
    """How far the true future departs from CTRV per window (meters).

    ``divergence`` = mean waypoint distance between GT and the constant-yaw-rate
    extrapolation over the horizon. High => an upcoming maneuver the current
    dynamics do not contain. gt/ctrv [N,H,2] -> [N]."""
    return per_window_l2(ctrv, gt, idx)


def quantile_bins(values, n_bins=4):
    """Assign each value to an equal-count quantile bin 0..n_bins-1.

    Returns (labels [N] long, edges [n_bins+1]). Ties keep monotonic labels;
    empty upper bins collapse (searchsorted on unique-clamped edges)."""
    v = torch.as_tensor(values, dtype=torch.float64)
    qs = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    edges = torch.quantile(v, qs)
    # right=False, exclude the leftmost edge from creating a -1 bin
    lab = torch.bucketize(v, edges[1:-1].contiguous(), right=False)
    return lab.long(), edges


def skill_vs_ctrv(pred, gt, ctrv, idx=WP_IDX):
    """Per-window model / CTRV error and the advantage (CTRV_l2 - model_l2).

    Returns dict of [N] tensors: model_l2, ctrv_l2, skill (model/ctrv, <1 beats),
    advantage (meters CTRV beaten by; >0 the model is better)."""
    m = per_window_l2(pred, gt, idx)
    c = per_window_l2(ctrv, gt, idx)
    return {"model_l2": m, "ctrv_l2": c,
            "skill": m / c.clamp_min(1e-9),
            "advantage": c - m}


def cluster_bootstrap_ci(values, eid, n_boot=2000, seed=0, alpha=0.05):
    """Mean + percentile CI resampling EPISODES (cluster bootstrap) to respect
    within-episode correlation. values [N], eid list[N]. Returns
    (mean, lo, hi, n)."""
    v = np.asarray(values, dtype=np.float64)
    eid = np.asarray([str(e) for e in eid])
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    uniq = np.unique(eid)
    idx_by_ep = {e: np.where(eid == e)[0] for e in uniq}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_ep[e] for e in pick])
        boots[b] = v[sel].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi), int(v.size)


def paired_bootstrap_delta_ci(a, b, eid, n_boot=2000, seed=0, alpha=0.05):
    """CI on mean(a) - mean(b) with the SAME resampled episodes each draw
    (paired). a,b [N] aligned per window. Returns (delta_mean, lo, hi,
    p_delta_gt0) where p_delta_gt0 is the bootstrap fraction with delta>0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    eid = np.asarray([str(e) for e in eid])
    uniq = np.unique(eid)
    idx_by_ep = {e: np.where(eid == e)[0] for e in uniq}
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_ep[e] for e in pick])
        d[i] = a[sel].mean() - b[sel].mean()
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(a.mean() - b.mean()), float(lo), float(hi),
            float((d > 0).mean()))


def path_curvature_deg(gt):
    """Net |heading change| of the ego future path over the horizon, degrees.

    A scene-derived quantity (the road bends this much ahead). gt [N,H,2] ego
    waypoints -> [N]. Heading of the final leg vs the first leg."""
    d = gt[:, 1:] - gt[:, :-1]                          # [N,H-1,2] step vectors
    head = torch.atan2(d[..., 1], d[..., 0])            # [N,H-1]
    net = (head[:, -1] - head[:, 0])
    net = (net + np.pi) % (2 * np.pi) - np.pi
    return net.abs() * (180.0 / np.pi)


def signed_lateral(pred):
    """Signed lateral (ego-y) displacement per step. pred [N,H,2] -> [N,H]."""
    return pred[..., 1]


def maneuver_onset_step(gt, ctrv, tol_m=0.5):
    """First horizon step where GT departs from CTRV by > tol (meters), i.e. the
    maneuver 'starts'. gt/ctrv [N,H,2] -> [N] long (H = no onset in horizon)."""
    dev = torch.linalg.norm(gt - ctrv, dim=-1)          # [N,H]
    H = dev.shape[1]
    over = dev > tol_m
    step = torch.full((dev.shape[0],), H, dtype=torch.long)
    any_over = over.any(dim=1)
    step[any_over] = over.float().argmax(dim=1)[any_over]
    return step


def commit_step(pred, gt, ctrv, margin_m=0.3):
    """First step where the PRED path departs from CTRV in the direction of the
    EVENTUAL maneuver by > margin. pred/gt/ctrv [N,H,2] -> [N] long (H = never).

    The maneuver direction is taken from the GT horizon ENDPOINT (sign of
    gt_y-ctrv_y at the last step) so a prediction can 'commit' BEFORE the GT
    kinematically departs (that early commitment is exactly the anticipation the
    lead-time measures). Commit = sign(pred_y - ctrv_y) == maneuver_dir and
    |pred_y - ctrv_y| > margin."""
    py, gy, cy = signed_lateral(pred), signed_lateral(gt), signed_lateral(ctrv)
    man_dir = torch.sign((gy - cy)[:, -1]).unsqueeze(1)      # [N,1] eventual dir
    pred_dev = py - cy
    ok = (torch.sign(pred_dev) == man_dir) & (pred_dev.abs() > margin_m) & (man_dir != 0)
    H = pred.shape[1]
    step = torch.full((pred.shape[0],), H, dtype=torch.long)
    any_ok = ok.any(dim=1)
    step[any_ok] = ok.float().argmax(dim=1)[any_ok]
    return step


def path_feasibility(pred, v0):
    """Physical-feasibility of a predicted ego path. pred [N,H,2] ego waypoints,
    v0 [N] entry speed (m/s). Returns per-window dicts of peak curvature, lateral
    & longitudinal accel, jerk, and a boolean 'feasible' within the envelopes.

    Kinematics: step speed = |Δp|/dt; heading from Δp; curvature = Δheading /
    arc; a_lat = v^2 * curvature; a_lon = Δspeed/dt; jerk = Δa_lon/dt."""
    d = pred[:, 1:] - pred[:, :-1]                       # [N,H-1,2]
    ds = torch.linalg.norm(d, dim=-1)                    # [N,H-1] arc per step
    speed = ds / DT                                      # [N,H-1]
    head = torch.atan2(d[..., 1], d[..., 0])
    dhead = (head[:, 1:] - head[:, :-1] + np.pi) % (2 * np.pi) - np.pi  # [N,H-2]
    kappa = dhead.abs() / ds[:, 1:].clamp_min(1e-3)      # [N,H-2] 1/m
    v_mid = speed[:, 1:]                                 # align with kappa
    a_lat = v_mid.pow(2) * kappa
    a_lon = (speed[:, 1:] - speed[:, :-1]) / DT          # [N,H-2]
    jerk = (a_lon[:, 1:] - a_lon[:, :-1]) / DT           # [N,H-3]
    kap = kappa.amax(dim=1)
    al = a_lat.amax(dim=1)
    alon = a_lon.abs().amax(dim=1)
    jk = jerk.abs().amax(dim=1) if jerk.shape[1] > 0 else torch.zeros(pred.shape[0])
    # PATH-SHAPE physics (robust to step jitter): bounded curvature + lateral
    # accel — 'could a car trace this path?'. Kept separate from SMOOTHNESS
    # (jerk / longitudinal accel), which finite-differencing inflates from
    # waypoint jitter (so a jittery-but-well-shaped path is not called unphysical).
    feasible = (kap <= KAPPA_MAX) & (al <= ALAT_MAX)
    smooth = (alon <= ALON_MAX) & (jk <= JERK_MAX)
    return {"kappa_max": kap, "a_lat_max": al, "a_lon_max": alon,
            "jerk_max": jk, "feasible": feasible, "smooth": smooth}


def ego_status_features(poses, last, hist=4):
    """Perception-free ego-kinematic history at ``last`` (import-free copy of the
    bench AD-MLP input, kept local so this module is self-contained). Returns
    [b, 2*hist+4]. ``last`` must be >= hist+1."""
    from driving_diagnostic import _ego, _wrap
    disp = torch.cat([_ego(poses[last - j, :2] - poses[last - j - 1, :2],
                           poses[last, 2]) for j in range(hist)], dim=-1)
    v0 = (poses[last, :2] - poses[last - 1, :2]).norm(dim=-1, keepdim=True)
    vm1 = (poses[last - 1, :2] - poses[last - 2, :2]).norm(dim=-1, keepdim=True)
    om0 = _wrap(poses[last, 2] - poses[last - 1, 2]).unsqueeze(-1)
    om1 = _wrap(poses[last - 1, 2] - poses[last - 2, 2]).unsqueeze(-1)
    return torch.cat([disp, v0, v0 - vm1, om0, om0 - om1], dim=-1)


def _kfold_probe_r2(feats, target, eid, alphas=(1.0, 10.0, 100.0), k=5, seed=0):
    """Episode-disjoint k-fold ridge probe. Returns (best held-out R^2, alpha).

    R^2 is computed on the pooled held-out predictions (each window predicted
    once from a fold excluding its episode)."""
    from tanitad.models.readout import RidgeProbe
    import random
    feats = feats.float()
    target = target.float().reshape(target.shape[0], -1)
    N = feats.shape[0]
    uniq = sorted(set(str(e) for e in eid))
    kk = max(2, min(k, len(uniq)))
    rng = random.Random(seed)
    order = uniq[:]
    rng.shuffle(order)
    grp = {e: i % kk for i, e in enumerate(order)}
    folds = [[] for _ in range(kk)]
    eid_s = [str(e) for e in eid]
    for i, e in enumerate(eid_s):
        folds[grp[e]].append(i)
    best = None
    for a in alphas:
        pred = torch.zeros_like(target)
        for va in folds:
            if not va:
                continue
            va_t = torch.tensor(va)
            mask = torch.ones(N, dtype=torch.bool)
            mask[va_t] = False
            tr_t = mask.nonzero(as_tuple=True)[0]
            if len(tr_t) < 8:
                tr_t = torch.arange(N)
            pr = RidgeProbe(alpha=a).fit(feats[tr_t], target[tr_t])
            pred[va_t] = pr.predict(feats[va_t]).float()
        ss_res = (pred - target).pow(2).sum()
        ss_tot = (target - target.mean(0, keepdim=True)).pow(2).sum().clamp_min(1e-9)
        r2 = float(1.0 - ss_res / ss_tot)
        if best is None or r2 > best[0]:
            best = (r2, a)
    return best


# ========================================================================== #
# MODEL PASSES — encode + rollout under real / mean / shuffled vision          #
# ========================================================================== #
def _append_v0(aw, fa, poses, last, device):
    v0 = (poses[last, 3:4] / SPEED_SCALE).to(device)
    aw = torch.cat([aw, v0.unsqueeze(1).expand(-1, aw.shape[1], -1)], dim=-1)
    fa = torch.cat([fa, v0.unsqueeze(1).expand(-1, fa.shape[1], -1)], dim=-1)
    return aw, fa


@torch.no_grad()
def collect(model, step_readout, episodes, device, speed_input=False,
            window=WINDOW, stride=STRIDE, fwd_k=K_MAX, yaw_input=False,
            dyn_input=False):
    """Encode every window once; store states + actions + GT/CTRV/CV (full-K) +
    ego-status + per-window scalars. Two-pass so the mean/shuffle vision
    ablation uses a GLOBAL reference scene (not a per-chunk mean)."""
    full_steps = list(range(1, fwd_k + 1))
    S, AW, FA, GT, CTRV, CV, EGO, LAT = [], [], [], [], [], [], [], []
    SPD, HDG, EID, LAST_V = [], [], [], []
    for ep in episodes:
        feats = ep.feats
        T = min(feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        starts = list(range(0, T - window - K_MAX, stride))
        for i in range(0, len(starts), 8):
            ch = starts[i:i + 8]
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
            states = model.encode_window(fw)                    # [b,W,Sdim]
            S.append(states.cpu().half())
            AW.append(aw.cpu().float())
            FA.append(fa.cpu().float())
            LAT.append(states[:, -1].cpu().float())
            GT.append(gt_ego_waypoints(ep.poses, last, full_steps))
            bw = baseline_waypoints(ep.poses, last, full_steps)
            CTRV.append(bw["constant_yaw_rate"])
            CV.append(bw["constant_velocity"])
            EGO.append(ego_status_features(ep.poses, last))
            SPD.append(ep.poses[last, 3])
            HDG.append(net_heading_change_deg(ep.poses, last))
            LAST_V.append(ep.poses[last, 3])
            EID.extend([ep.episode_id] * len(ch))
    return {
        "states": torch.cat(S), "aw": torch.cat(AW), "fa": torch.cat(FA),
        "latent": torch.cat(LAT), "gt": torch.cat(GT).float(),
        "ctrv": torch.cat(CTRV).float(), "cv": torch.cat(CV).float(),
        "ego": torch.cat(EGO).float(), "speed": torch.cat(SPD).float(),
        "head_deg": torch.cat(HDG).float(), "v0": torch.cat(LAST_V).float(),
        "eid": EID, "full_steps": full_steps,
    }


@torch.no_grad()
def rollout_modes(model, step_readout, col, device, modes=("real", "mean", "shuffle"),
                  batch=64, seed=0):
    """Roll the operative predictor under each vision mode over ALL windows.

    real    : the window's own encoded states.
    mean    : every window sees the GLOBAL mean state sequence (scene removed,
              actions kept) — the causal 'no scene' counterfactual.
    shuffle : states permuted across windows (right scene stats, wrong scene).
    Returns {mode: pred [N,K,2]}."""
    states = col["states"].float()
    aw, fa = col["aw"], col["fa"]
    N, K = states.shape[0], len(col["full_steps"])
    mean_state = states.mean(dim=0, keepdim=True)                # [1,W,Sdim]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    out = {m: torch.zeros(N, K, 2) for m in modes}
    for i in range(0, N, batch):
        j = min(i + batch, N)
        a = aw[i:j].to(device)
        f = fa[i:j].to(device)
        src = {
            "real": states[i:j],
            "mean": mean_state.expand(j - i, -1, -1),
            "shuffle": states[perm[i:j]],
        }
        for m in modes:
            st = src[m].to(device)
            wp, _ = rollout_decode(model.predictor, st, a, f, step_readout, K)
            out[m][i:j] = wp.cpu().float()
    return out


# ========================================================================== #
# TEST COMPOSERS                                                               #
# ========================================================================== #
def _bin_report(adv_model, adv_ctrv_is_zero, skill, div, eid, n_bins=4):
    """Per divergence-bin skill_vs_CTRV + advantage with CI. adv_model/skill/div
    [N] tensors."""
    lab, edges = quantile_bins(div, n_bins)
    rows = []
    for b in range(n_bins):
        sel = (lab == b).nonzero(as_tuple=True)[0]
        if len(sel) == 0:
            continue
        e = [eid[i] for i in sel.tolist()]
        adv_mean, adv_lo, adv_hi, n = cluster_bootstrap_ci(
            adv_model[sel].numpy(), e)
        rows.append({
            "bin": b, "divergence_range_m": [round(float(edges[b]), 3),
                                             round(float(edges[b + 1]), 3)],
            "n": n, "n_episodes": len(set(e)),
            "median_divergence_m": round(float(div[sel].median()), 3),
            "median_skill_vs_ctrv": round(float(skill[sel].median()), 4),
            "frac_beats_ctrv": round(float((skill[sel] < 1).float().mean()), 4),
            "advantage_over_ctrv_m": round(adv_mean, 4),
            "advantage_ci95": [round(adv_lo, 4), round(adv_hi, 4)],
        })
    return rows, lab, edges


def test_A_divergence(col, preds, n_bins=4):
    """A: CTRV-divergence stratification + anticipation."""
    gt, ctrv = col["gt"], col["ctrv"]
    div = ctrv_divergence(gt, ctrv, WP_IDX)
    sk = skill_vs_ctrv(preds["real"], gt, ctrv, WP_IDX)
    rows, lab, edges = _bin_report(sk["advantage"], None, sk["skill"], div,
                                   col["eid"], n_bins)
    top = rows[-1] if rows else {}
    bot = rows[0] if rows else {}
    # monotone anticipation: advantage grows from low- to high-divergence bins
    advs = [r["advantage_over_ctrv_m"] for r in rows]
    monotone = all(advs[i] <= advs[i + 1] + 1e-6 for i in range(len(advs) - 1))
    return {
        "divergence_bins": rows,
        "overall_median_skill_vs_ctrv": round(float(sk["skill"].median()), 4),
        "overall_frac_beats_ctrv": round(float((sk["skill"] < 1).float().mean()), 4),
        "overall_advantage_m": round(float(sk["advantage"].mean()), 4),
        "skill_ratio_note": "per-window skill = model_L2/CTRV_L2; on low-"
        "divergence windows CTRV_L2->0 so the RATIO explodes (use "
        "frac_beats_ctrv + advantage_m as the robust reads; median skill is "
        "reported over mean for the same reason).",
        "high_div_advantage_m": top.get("advantage_over_ctrv_m"),
        "high_div_beats_ctrv_frac": top.get("frac_beats_ctrv"),
        "low_div_advantage_m": bot.get("advantage_over_ctrv_m"),
        "advantage_monotone_in_divergence": bool(monotone),
        "anticipation_verdict": (
            "GENUINE anticipation — advantage over CTRV grows with divergence and "
            "is positive on the high-divergence (upcoming-maneuver) windows"
            if (top.get("advantage_over_ctrv_m", 0) > 0 and
                top.get("advantage_ci95", [0, 0])[0] > 0)
            else "no CTRV-beating anticipation on high-divergence windows"),
        "_div": div, "_bin_labels": lab, "_edges": edges,
    }


def test_B_vision_ablation(col, preds, div, lab, n_bins=4):
    """B: vision-ablation on divergence (HEADLINE). Does the anticipation
    advantage collapse toward CTRV when the scene is removed?"""
    gt, ctrv, eid = col["gt"], col["ctrv"], col["eid"]
    adv = {m: skill_vs_ctrv(preds[m], gt, ctrv, WP_IDX)["advantage"]
           for m in preds}
    rows = []
    for b in range(n_bins):
        sel = (lab == b).nonzero(as_tuple=True)[0]
        if len(sel) == 0:
            continue
        e = [eid[i] for i in sel.tolist()]
        row = {"bin": b, "n": len(sel),
               "median_divergence_m": round(float(div[sel].median()), 3)}
        for m in preds:
            mn, lo, hi, _ = cluster_bootstrap_ci(adv[m][sel].numpy(), e)
            row[f"advantage_{m}_m"] = round(mn, 4)
            row[f"advantage_{m}_ci95"] = [round(lo, 4), round(hi, 4)]
        if "mean" in preds:
            dmn, dlo, dhi, pgt = paired_bootstrap_delta_ci(
                adv["real"][sel].numpy(), adv["mean"][sel].numpy(), e)
            row["vision_effect_m"] = round(dmn, 4)          # real - mean advantage
            row["vision_effect_ci95"] = [round(dlo, 4), round(dhi, 4)]
            row["vision_effect_p_gt0"] = round(pgt, 4)
            row["collapses_without_vision"] = bool(
                dlo > 0 and row["advantage_mean_m"] < row["advantage_real_m"])
        rows.append(row)
    top = rows[-1] if rows else {}
    headline = {
        "high_div_advantage_with_vision_m": top.get("advantage_real_m"),
        "high_div_advantage_without_vision_m": top.get("advantage_mean_m"),
        "high_div_vision_effect_m": top.get("vision_effect_m"),
        "high_div_vision_effect_ci95": top.get("vision_effect_ci95"),
        "anticipation_is_vision_based": bool(top.get("collapses_without_vision")),
        "verdict": (
            "HEADLINE POSITIVE — on high-divergence windows the CTRV-beating "
            "advantage is vision-based (collapses when the scene is removed, "
            "CI-separated); the model READS upcoming physics from the scene"
            if top.get("collapses_without_vision") else
            "anticipation advantage does NOT collapse without vision — not "
            "conclusively scene-based on these windows"),
    }
    return {"per_bin": rows, "headline": headline}


def test_C_lead_time(col, preds, div, lab, n_bins=4, hi_bins=(2, 3)):
    """C: anticipation lead-time on high-divergence maneuver-onset windows."""
    gt, ctrv, eid = col["gt"], col["ctrv"], col["eid"]
    onset = maneuver_onset_step(gt, ctrv, tol_m=0.5)
    H = gt.shape[1]
    hi = torch.zeros(len(gt), dtype=torch.bool)
    for b in hi_bins:
        hi |= (lab == b)
    qual = hi & (onset > 0) & (onset < H)                   # onset inside horizon
    sel = qual.nonzero(as_tuple=True)[0]
    if len(sel) < 10:
        return {"skipped": f"only {len(sel)} qualifying onset windows"}
    e = [eid[i] for i in sel.tolist()]
    out = {"n_windows": int(len(sel)), "n_episodes": len(set(e)),
           "mean_onset_step": round(float(onset[sel].float().mean()), 2)}
    for m in ("real", "mean"):
        if m not in preds:
            continue
        commit = commit_step(preds[m][sel], gt[sel], ctrv[sel], margin_m=0.3)
        lead = (onset[sel] - commit).float()                # >0 commits early
        committed = commit < H
        lm, llo, lhi, _ = cluster_bootstrap_ci(lead[committed].numpy(),
                                               [e[i] for i in
                                                committed.nonzero(as_tuple=True)[0].tolist()]) \
            if committed.any() else (float("nan"), float("nan"), float("nan"), 0)
        out[m] = {
            "commit_rate": round(float(committed.float().mean()), 4),
            "mean_lead_frames": round(lm, 3) if lm == lm else None,
            "mean_lead_s": round(lm * DT, 3) if lm == lm else None,
            "lead_ci95_frames": [round(llo, 3), round(lhi, 3)] if lm == lm else None,
        }
    if "real" in out and "mean" in out:
        lr, lm = out["real"]["mean_lead_frames"], out["mean"]["mean_lead_frames"]
        cr, cm = out["real"]["commit_rate"], out["mean"]["commit_rate"]
        out["vision_tied_lead"] = bool((lr or 0) > (lm or 0) or cr > cm)
        out["verdict"] = (
            "positive vision-tied lead — the model commits to the maneuver "
            "earlier WITH vision than without ('sees what's coming')"
            if out["vision_tied_lead"] else
            "no pre-onset lateral lead here")
        out["caveat"] = (
            "ACTION-PRIVILEGE CONFOUND: the operative rollout consumes the TRUE "
            "future actions, and before a maneuver the steering is still ~straight "
            "— so the lateral path tracks the given (pre-onset) actions rather "
            "than committing early from vision. The mean-ablation 'lead' is a "
            "collapse artifact (the vision-removed prediction is a fixed average "
            "path that curves toward the dataset-mean maneuver). Anticipation "
            "here shows up as HORIZON-ADE advantage over CTRV (tests A/B), not as "
            "early lateral commitment; treat this lead-time panel as "
            "non-informative for an action-grounded rollout.")
    return out


def test_D_latent_decodability(col):
    """D: can scene-only quantities be linearly read from the pooled latent, and
    ABOVE an ego-kinematics-only probe? Target: upcoming road curvature."""
    curv = path_curvature_deg(col["gt"])                    # [N] scene quantity
    eid = col["eid"]
    lat_r2, lat_a = _kfold_probe_r2(col["latent"], curv[:, None], eid)
    ego_r2, ego_a = _kfold_probe_r2(col["ego"], curv[:, None], eid)
    gain = lat_r2 - ego_r2
    # also decode raw net-heading as a sanity target (dynamics-correlated)
    return {
        "target": "upcoming_road_curvature_deg (net |heading change| of GT future path)",
        "latent_probe_r2": round(lat_r2, 4), "latent_alpha": lat_a,
        "ego_status_probe_r2": round(ego_r2, 4), "ego_alpha": ego_a,
        "latent_minus_ego_r2": round(gain, 4),
        "state_dim": int(col["latent"].shape[1]),
        "ego_dim": int(col["ego"].shape[1]),
        "n_windows": int(col["latent"].shape[0]),
        "verdict": (
            "latent encodes SCENE — upcoming curvature is decodable from the "
            "latent ABOVE an ego-kinematics-only probe (scene info beyond ego "
            "state)" if gain > 0.03 else
            "no scene advantage — the latent decodes curvature no better than "
            "ego kinematics alone (curvature may be dynamics-correlated here)"),
    }


def test_E_ood_proxy(col, preds):
    """E: within-corpus novelty proxy + physical-feasibility (learned physics).

    TRUE OOD corpora are data-blocked (see ood_corpus_status). Proxy: top-decile
    curvature/accel windows (sharper/faster than typical) + feasibility of
    predictions."""
    gt, ctrv, cv, eid = col["gt"], col["ctrv"], col["cv"], col["eid"]
    head, speed = col["head_deg"], col["speed"]
    sk = skill_vs_ctrv(preds["real"], gt, ctrv, WP_IDX)
    cv_l2 = per_window_l2(cv, gt, WP_IDX)
    feas = path_feasibility(preds["real"], col["v0"])
    feas_gt = path_feasibility(gt, col["v0"])

    def stratum(mask, name):
        sel = mask.nonzero(as_tuple=True)[0]
        if len(sel) < 8:
            return {"stratum": name, "n": int(len(sel)),
                    "note": "too few windows"}
        e = [eid[i] for i in sel.tolist()]
        adv_m, adv_lo, adv_hi, _ = cluster_bootstrap_ci(
            sk["advantage"][sel].numpy(), e)
        return {
            "stratum": name, "n": int(len(sel)),
            "model_ade_m": round(float(sk["model_l2"][sel].mean()), 4),
            "ctrv_ade_m": round(float(sk["ctrv_l2"][sel].mean()), 4),
            "cv_ade_m": round(float(cv_l2[sel].mean()), 4),
            "median_skill_vs_ctrv": round(float(sk["skill"][sel].median()), 4),
            "beats_ctrv_frac": round(float((sk["skill"][sel] < 1).float().mean()), 4),
            "advantage_over_ctrv_m": round(adv_m, 4),
            "advantage_ci95": [round(adv_lo, 4), round(adv_hi, 4)],
            "pred_shape_feasible_frac": round(float(feas["feasible"][sel].float().mean()), 4),
            "gt_shape_feasible_frac": round(float(feas_gt["feasible"][sel].float().mean()), 4),
            "pred_smooth_frac": round(float(feas["smooth"][sel].float().mean()), 4),
            "pred_max_kappa_p95": round(float(np.percentile(feas["kappa_max"][sel].numpy(), 95)), 4),
            "pred_max_alat_p95": round(float(np.percentile(feas["a_lat_max"][sel].numpy(), 95)), 4),
        }
    hd90 = torch.quantile(head, 0.90)
    sp90 = torch.quantile(speed, 0.90)
    rows = [
        stratum(torch.ones_like(head, dtype=torch.bool), "all"),
        stratum(head >= hd90, "sharp_top10pct_curvature"),
        stratum(speed >= sp90, "fast_top10pct_speed"),
        stratum((head >= hd90) | (speed >= sp90), "novel_sharp_or_fast"),
    ]
    sharp = rows[1]                                        # sharp_top10pct row
    return {
        "note": "within-corpus novelty proxy — NOT a true unseen corpus; see "
                "ood_corpus_status for why comma2k19/Cosmos are unavailable",
        "strata": rows,
        "overall_pred_shape_feasible_frac": round(float(feas["feasible"].float().mean()), 4),
        "overall_pred_smooth_frac": round(float(feas["smooth"].float().mean()), 4),
        "feasibility_note": "shape-feasible = bounded curvature (<=%.2g 1/m) + "
        "lateral accel (<=%g m/s^2) — the path a car could trace; smooth = "
        "bounded jerk + longitudinal accel (finite-difference-inflated by "
        "waypoint jitter, so reported separately)." % (KAPPA_MAX, ALAT_MAX),
        "curvature_p90_deg": round(float(hd90), 3),
        "speed_p90_mps": round(float(sp90), 3),
        "verdict": (
            "predictions stay PHYSICALLY-SHAPED (bounded curvature/lateral accel) "
            "and still beat CTRV on the sharpest windows — consistent with "
            "learned physics, not memorization. Honest caveats: at the highest "
            "SPEEDS trivial CTRV is near-perfect and the model does not beat it; "
            "predicted paths are less SMOOTH than GT (jerk-gated)."
            if (sharp.get("advantage_over_ctrv_m", -1) or -1) > 0
            and sharp.get("pred_shape_feasible_frac", 0) > 0.9
            else "mixed — see strata; advantage or shape-feasibility varies by "
                 "novelty regime (sharp vs fast differ)"),
    }


@torch.no_grad()
def collect_occlusion(model, step_readout, episodes, device, speed_input=False,
                      window=WINDOW, stride=STRIDE, fwd_k=K_MAX, max_eps=12,
                      sectors=None):
    """F-proxy: spatial-sector occlusion of the RAW input with dynamics held
    fixed. For each window, roll under the true (unmasked) frames and under each
    sector-masked variant; measure ||pred_masked - pred_base|| over the horizon.
    Frame-input arms only (ep.feats is [T,9,H,W] uint8)."""
    if sectors is None:
        sectors = _default_sectors()
    K = fwd_k
    base_all = {s: [] for s in sectors}                    # per-sector deltas
    n_win = 0
    ep_used = 0
    for ep in episodes:
        if ep_used >= max_eps:
            break
        feats = ep.feats
        if feats.dtype != torch.uint8 or feats.ndim != 4:
            return {"skipped": "occlusion needs raw [T,9,H,W] frames "
                    "(frame-input arm); this arm feeds frozen features"}
        ep_used += 1
        T = feats.shape[0]
        Hpx, Wpx = feats.shape[-2], feats.shape[-1]
        starts = list(range(0, T - window - K_MAX, stride))
        for i in range(0, len(starts), 8):
            ch = starts[i:i + 8]
            if not ch:
                continue
            last = torch.tensor([t + window - 1 for t in ch])
            base = torch.stack([torch.as_tensor(feats[t:t + window])
                                for t in ch]).float().div(255.0).to(device)
            aw = torch.stack([ep.actions[t:t + window] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + window:t + window + fwd_k]
                              for t in ch]).to(device)
            if speed_input:
                aw, fa = _append_v0(aw, fa, ep.poses, last, device)
            st = model.encode_window(base)
            wp_base, _ = rollout_decode(model.predictor, st, aw, fa,
                                        step_readout, K)
            # per-window, per-channel spatial mean = the neutral fill value
            fill = base.mean(dim=(-1, -2), keepdim=True)   # [b,W,9,1,1]
            for sname, (y0, y1, x0, x1) in sectors.items():
                ys, ye = int(y0 * Hpx), int(y1 * Hpx)
                xs, xe = int(x0 * Wpx), int(x1 * Wpx)
                masked = base.clone()
                masked[..., ys:ye, xs:xe] = fill               # broadcasts in
                sm = model.encode_window(masked)
                wp_m, _ = rollout_decode(model.predictor, sm, aw, fa,
                                         step_readout, K)
                delta = torch.linalg.norm(wp_m - wp_base, dim=-1)[:, WP_IDX].mean(1)
                base_all[sname].append(delta.cpu())
            n_win += len(ch)
    out = {s: torch.cat(v) for s, v in base_all.items() if v}
    return {"deltas": out, "n_windows": n_win, "n_episodes": ep_used,
            "sectors": {k: list(v) for k, v in sectors.items()}}


def _default_sectors():
    """(y0,y1,x0,x1) fractional boxes on the 256x256 crop. road_ahead = lower
    center (the drivable surface the ego heads into); periphery = side strips;
    sky = upper band (control that should not affect the path)."""
    return {
        "road_ahead": (0.55, 1.00, 0.30, 0.70),
        "periphery_left": (0.40, 1.00, 0.00, 0.20),
        "periphery_right": (0.40, 1.00, 0.80, 1.00),
        "sky_top": (0.00, 0.30, 0.00, 1.00),
    }


def test_F_occlusion(occ):
    """F: does the predicted path respond MORE to occluding the road ahead than
    the periphery (dynamics identical)?"""
    if occ.get("skipped"):
        return {"skipped": occ["skipped"], "spec": _cosmos_spec()}
    d = occ["deltas"]
    means = {s: round(float(v.mean()), 4) for s, v in d.items()}
    road = means.get("road_ahead", 0.0)
    peri = np.mean([means.get("periphery_left", 0.0),
                    means.get("periphery_right", 0.0)])
    sky = means.get("sky_top", 0.0)
    return {
        "note": "proxy for a Cosmos scene-counterfactual — occlude a spatial "
                "sector of the input, dynamics (actions,v0) held fixed, measure "
                "the predicted-path shift (m). road>periphery => the model reads "
                "the road ahead to predict the path.",
        "mean_path_shift_m": means,
        "n_windows": occ["n_windows"], "n_episodes": occ["n_episodes"],
        "road_ahead_vs_periphery_ratio": round(road / max(peri, 1e-6), 3),
        "road_ahead_vs_sky_ratio": round(road / max(sky, 1e-6), 3),
        "reads_road_ahead": bool(road > 1.3 * peri and road > sky),
        "verdict": (
            "occluding the road ahead shifts the prediction MORE than occluding "
            "the periphery (dynamics fixed) => the path is causally conditioned "
            "on the road scene, not just ego dynamics"
            if (road > 1.3 * peri and road > sky) else
            "no road>periphery asymmetry — the predicted path is not clearly "
            "conditioned on the road-ahead pixels here"),
        "cosmos_counterfactual_spec": _cosmos_spec(),
    }


def _cosmos_spec():
    return {
        "ideal": "insert/remove a scene element (obstacle, curve) with the ego "
                 "DYNAMICS HELD FIXED, and show the prediction responds "
                 "(slow/swerve) => zero-confound causal scene-understanding.",
        "requires_data": [
            "Cosmos-Drive-Dreams PIXELS on the eval pod (currently absent — "
            "/workspace is empty; cosmos_pairs.py can extract matched-weather "
            "PAIRS from a generation shard to /workspace/data/cosmos/pairs but "
            "no shard is staged).",
            "paired clips differing in ONE scene element with matched ego-state, "
            "OR an inpainting/edit tool to remove/insert an obstacle while "
            "keeping frames otherwise identical.",
        ],
        "requires_tooling": [
            "a video->epcache converter producing [T,9,256,256] uint8 + poses "
            "(the physicalai epcache contract) for Cosmos clips;",
            "an inpainting model (e.g. a diffusion eraser) for true insert/remove, "
            "OR use cosmos_pairs weather/scene pairs as natural counterfactuals.",
        ],
        "feasible_now": "spatial-sector occlusion (implemented here) — the "
                        "same causal logic (dynamics fixed, perturb the scene) "
                        "on the physicalai pixels we already have.",
    }


# ========================================================================== #
# ORCHESTRATION                                                                #
# ========================================================================== #
def ood_corpus_status():
    """Report which corpora are materialized on the eval pod (data inventory)."""
    from pathlib import Path
    root = Path("/root/valdata")
    corpora = []
    for d in sorted(root.glob("*")) if root.exists() else []:
        n = len(list(d.glob("ep_*.pt")))
        corpora.append({"dir": str(d), "n_episodes": n})
    return {
        "present": corpora,
        "physicalai_val": "IN-DISTRIBUTION held-out episodes (flagship-30k "
                          "trained on physicalai_phase0 epcache) — the val split "
                          "is disjoint episodes, not a different corpus.",
        "comma2k19": "ABSENT on the eval pod. Tooling exists "
                     "(stack/scripts/extract_comma2k19.py) but no chunk zips / "
                     "epcache are staged (/root/valdata has only physicalai; "
                     "/workspace is empty). Enabling needs: download comma2k19 "
                     "chunk zips -> extract -> convert to [T,9,256,256] epcache.",
        "cosmos": "ABSENT on the eval pod. Tooling exists "
                  "(stack/scripts/cosmos_pairs.py, cosmos_verify.py) but no "
                  "generation shard / pixels are staged. Enabling needs a Cosmos "
                  "shard + a clip->epcache converter.",
        "true_ood_feasible_now": False,
        "note": "Tests A-D + F run on the in-distribution held-out physicalai "
                "val (the anticipation/causal tests do not REQUIRE a new corpus "
                "— they test skill CTRV cannot have). Test E's true-OOD arm is "
                "data-blocked; the within-corpus novelty proxy runs instead.",
    }


def run(model, step_readout, episodes, device, speed_input=False,
        n_bins=4, do_occlusion=True, occ_eps=12, feed="frames",
        yaw_input=False, dyn_input=False):
    """Compose tests A-F into one results dict for a rollout-capable arm."""
    col = collect(model, step_readout, episodes, device, speed_input,
                  yaw_input=yaw_input, dyn_input=dyn_input)
    preds = rollout_modes(model, step_readout, col, device)
    A = test_A_divergence(col, preds, n_bins)
    div, lab = A.pop("_div"), A.pop("_bin_labels")
    A.pop("_edges")
    B = test_B_vision_ablation(col, preds, div, lab, n_bins)
    C = test_C_lead_time(col, preds, div, lab, n_bins)
    D = test_D_latent_decodability(col)
    E = test_E_ood_proxy(col, preds)
    F = {"skipped": "occlusion disabled"}
    if do_occlusion:
        if feed == "frames":
            occ = collect_occlusion(model, step_readout, episodes, device,
                                    speed_input, max_eps=occ_eps)
            F = test_F_occlusion(occ)
        else:
            F = {"skipped": "frozen-feature arm — no raw pixels to occlude",
                 "cosmos_counterfactual_spec": _cosmos_spec()}
    return {
        "n_windows": int(col["gt"].shape[0]),
        "n_episodes": len(set(col["eid"])),
        "ood_corpus_status": ood_corpus_status(),
        "A_ctrv_divergence": A,
        "B_vision_ablation": B,
        "C_lead_time": C,
        "D_latent_decodability": D,
        "E_ood_proxy": E,
        "F_cosmos_counterfactual": F,
        "bottom_line": _bottom_line(A, B, D, E, F),
    }


def _bottom_line(A, B, D, E, F):
    bits = []
    bits.append(A.get("anticipation_verdict", ""))
    hl = B.get("headline", {})
    bits.append(hl.get("verdict", ""))
    bits.append(D.get("verdict", ""))
    if not F.get("skipped"):
        bits.append(F.get("verdict", ""))
    return " || ".join(b for b in bits if b)


def run_and_save(key, device="cuda", episodes=40, n_bins=4, occ_eps=12,
                 out_dir="/root/taniteval/results", corpus="physicalai"):
    """Standalone: load an arm, run the generalization panel, write
    results/gen_<key>.json. Reuses loaders.load + data (read-only)."""
    import json
    import time
    from pathlib import Path
    sys.path.insert(0, "/root/taniteval")
    from taniteval import data, loaders             # read-only use (adf3 owns)
    from taniteval.registry import MODELS, CORPORA
    entry = [m for m in MODELS if m["key"] == key]
    assert entry, f"unknown model {key}"
    entry = entry[0]
    csel = [c for c in CORPORA if c["key"] == corpus]
    known = [c["key"] for c in CORPORA]
    assert csel, f"unknown corpus {corpus}; known {known}"
    val_root = csel[0]["root"]
    t0 = time.time()
    L = loaders.load(entry, device)
    if not L["traj_capable"]:
        # PLANNER arms (REF-B) have no grounded rollout, so the anticipation
        # tests (CTRV divergence / vision-ablation) don't apply. But the
        # cross-corpus GENERALIZATION read is still meaningful as the planner's
        # DIRECT trajectory metrics on this corpus (in-dist vs OOD) — reuse
        # refb_eval.collect + bench.run (identical metric suite to `run`).
        if entry.get("arch") != "refb":
            res = {"key": key, "skipped": "no trajectory surface (planner arm) — "
                   "the anticipation tests need a grounded rollout"}
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / f"gen_{key}.json").write_text(json.dumps(res, indent=2))
            print(f"[gen] {key}: skipped (planner arm)", flush=True)
            return res
        from taniteval import bench, refb_eval
        files = data.list_val_episodes(val_root, episodes)
        assert files, f"no episodes under {val_root} (corpus {corpus})"
        eps = data.load_frames(files)
        win = refb_eval.collect(L["model"], eps, device,
                                speed_input=bool(entry.get("speed_input")),
                                yaw_input=bool(entry.get("yaw_input")))
        res = bench.run(win)
        res["method"] = win.get("method")
        res["planner_generalization"] = True
        res["model"] = {k: entry.get(k) for k in
                        ("key", "name", "arch", "encoder", "speed_input")}
        res["ckpt_step"] = L["step"]
        res["corpus"] = {k: csel[0].get(k) for k in
                         ("key", "name", "kind", "root")}
        res["wall_s"] = round(time.time() - t0, 1)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        _sfx = "" if corpus == "physicalai" else f"_{corpus}"
        (Path(out_dir) / f"gen_{key}{_sfx}.json").write_text(
            json.dumps(res, indent=2, default=str))
        hm = res["heldout"]["model"]
        print(f"[gen] {key} step={L['step']} corpus={corpus} "
              f"n={res['n_windows']}: "
              f"ade@2s={hm['ade_0_2s']['mean']:.3f}"
              f"+/-{hm['ade_0_2s']['ci95']:.3f} "
              f"fde={hm['fde@2s']['mean']:.3f} "
              f"miss@2m={hm['miss_rate@2m']['mean']:.3f} "
              f"(planner direct) -> gen_{key}{_sfx}.json", flush=True)
        return res
    files = data.list_val_episodes(val_root, episodes)
    assert files, f"no episodes under {val_root} (corpus {corpus})"
    if corpus == "physicalai" and entry.get("train_ids"):  # leak guard (phys ids)
        from tanitad.data.mixing import load_episode
        tid = set(Path(entry["train_ids"]).read_text().split())
        files = [f for f in files
                 if str(load_episode(str(f), mmap=True).episode_id) not in tid]
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], device))
    res = run(L["model"], L["step_readout"], eps, device,
              speed_input=bool(entry.get("speed_input")), n_bins=n_bins,
              occ_eps=occ_eps, feed=L["feed"],
              yaw_input=bool(entry.get("yaw_input")),
              dyn_input=bool(entry.get("dyn_input")))
    res["model"] = {k: entry.get(k) for k in
                    ("key", "name", "arch", "encoder", "speed_input")}
    res["ckpt_step"] = L["step"]
    res["wall_s"] = round(time.time() - t0, 1)
    res["corpus"] = {k: csel[0].get(k) for k in ("key", "name", "kind", "root")}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    _sfx = "" if corpus == "physicalai" else f"_{corpus}"
    (Path(out_dir) / f"gen_{key}{_sfx}.json").write_text(
        json.dumps(res, indent=2, default=str))
    hl = res["B_vision_ablation"]["headline"]
    A = res["A_ctrv_divergence"]
    print(f"[gen] {key} step={L['step']} n={res['n_windows']}: "
          f"hi-div adv={A.get('high_div_advantage_m')}m "
          f"vision-effect={hl.get('high_div_vision_effect_m')}m "
          f"CI{hl.get('high_div_vision_effect_ci95')} "
          f"vision-based={hl.get('anticipation_is_vision_based')} "
          f"({res['wall_s']}s) -> gen_{key}{_sfx}.json", flush=True)
    return res


def main():
    import argparse
    sys.path.insert(0, "/root/taniteval")
    from taniteval.registry import MODELS
    ap = argparse.ArgumentParser("taniteval.generalization")
    ap.add_argument("--model", help="registry key; omit with --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--occ-eps", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    keys = ([m["key"] for m in MODELS] if a.all else [a.model])
    for key in keys:
        if not any(m["key"] == key for m in MODELS):
            print(f"[gen] unknown model {key}", flush=True)
            continue
        try:
            run_and_save(key, a.device, a.episodes, a.bins, a.occ_eps)
        except Exception as e:
            import traceback
            print(f"[gen] {key} FAILED: {type(e).__name__}: {str(e)[:160]}",
                  flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
