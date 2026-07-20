"""TanitEval — imagination panel.

Grounded ADE is measured WITH the true future action sequence, which
over-determines the pose — a good score can hide an inert encoder. This panel
ISOLATES the vision/imagination contribution for any rollout-capable world-model
arm (flagship, REF-A), via a 2x2 (+ latent fidelity):

  vision {real, mean, shuffle}  x  future-actions {true, withheld}

    A real+trueA  baseline (the grounded-ADE regime)
    B mean+trueA  scene content removed, actions kept
    C shuf+trueA  WRONG scene, right stats, actions kept
    D real+noA    imagination regime: coast on last action, predict from vision
    E mean+noA    imagination floor

Headline metrics:
  vision_use_pct     = (B-A)/A * 100   ADE cost of removing scene (actions on)
  imagination_pct    = (E-D)/E * 100   ADE real vision recovers (actions off)
  latent_fidelity    = cos(z_hat, z_true)     imagined latents vs true future
  latent_vision_gain = cos_real - cos_mean    fidelity lost when vision ablated

Applies to traj-capable predictor+step_readout arms; NOT REF-B (planner)."""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from tanitad.models.metric_dynamics import (rollout_decode,  # noqa: E402
                                            rollout_transitions)
from driving_diagnostic import WP_STEPS, gt_ego_waypoints  # noqa: E402
from taniteval import rollout as _rollout  # noqa: E402  (canonical ego append)

SPEED_SCALE = 10.0
K = max(WP_STEPS)
WIN = 8
IDX = [k - 1 for k in WP_STEPS]


def _ablate(states, mode, gen):
    if mode == "real":
        return states
    if mode == "mean":
        return states.mean(dim=0, keepdim=True).expand_as(states).clone()
    if mode == "shuffle":
        return states[torch.randperm(states.shape[0], generator=gen,
                                     device=states.device)]
    raise ValueError(mode)


@torch.no_grad()
def run(model, step_readout, episodes, device, speed_input=False, max_eps=12,
        stride=8, yaw_input=False, dyn_input=False):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(0)
    conds = {"A": ("real", True), "B": ("mean", True), "C": ("shuffle", True),
             "D": ("real", False), "E": ("mean", False)}
    ade = {c: [] for c in conds}
    cos_real, cos_mean = [], []
    fidelity_ok = True

    for ep in episodes[:max_eps]:
        fr = ep.feats                                  # [T,...] uint8 or fp16
        T = min(fr.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        for i in range(0, T - WIN - K, stride * 8):
            ch = list(range(i, min(i + stride * 8, T - WIN - K), stride))
            if not ch:
                continue
            last = torch.tensor([t + WIN - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + WIN]) for t in ch]
                             ).to(device).float()
            if ep.feats.dtype == torch.uint8:
                fw = fw.div_(255.0)
            aw = torch.stack([ep.actions[t:t + WIN] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + WIN:t + WIN + K] for t in ch]
                             ).to(device)
            aw, fa = _rollout.append_ego(aw, fa, ep.poses, last, speed_input,
                                         yaw_input, dyn_input, device)
            gt = gt_ego_waypoints(ep.poses, last).to(device)
            states = model.encode_window(fw)

            for c, (vmode, use_fa) in conds.items():
                st = _ablate(states, vmode, gen)
                wp, _ = rollout_decode(model.predictor, st, aw,
                                       fa if use_fa else None, step_readout, K)
                ade[c].append(torch.linalg.norm(wp[:, IDX] - gt, dim=-1)
                              .mean(1).cpu())

            if fidelity_ok:                            # best-effort per arch
                try:
                    tr = rollout_transitions(model.predictor, states, aw, fa, K)
                    zh = torch.stack([p[1] for p in tr], dim=1)
                    fut = torch.stack([torch.as_tensor(fr[t + WIN:t + WIN + K])
                                       for t in ch]).to(device).float()
                    if ep.feats.dtype == torch.uint8:
                        fut = fut.div_(255.0)
                    b, k = fut.shape[0], fut.shape[1]
                    z_true = model.encode(fut.reshape(b * k, *fut.shape[2:])
                                          ).reshape(b, k, -1)
                    tr_m = rollout_transitions(
                        model.predictor, _ablate(states, "mean", gen), aw, fa, K)
                    zh_m = torch.stack([p[1] for p in tr_m], dim=1)
                    cos_real.append(
                        F.cosine_similarity(zh, z_true, dim=-1).mean().cpu())
                    cos_mean.append(
                        F.cosine_similarity(zh_m, z_true, dim=-1).mean().cpu())
                except Exception:
                    fidelity_ok = False

    m = {c: float(torch.cat(ade[c]).mean()) for c in conds}
    n = sum(t.numel() for t in ade["A"])
    lat = None
    if fidelity_ok and cos_real:
        cr, cm = float(torch.stack(cos_real).mean()), \
            float(torch.stack(cos_mean).mean())
        lat = {"real": round(cr, 4), "mean_vision": round(cm, 4),
               "vision_gain": round(cr - cm, 4)}
    return {
        "ade": {k: round(v, 4) for k, v in m.items()},
        "ade_labels": {"A": "real+trueActions", "B": "meanVision+trueActions",
                       "C": "shuffleVision+trueActions", "D": "real+noActions",
                       "E": "meanVision+noActions"},
        "vision_use_pct": round(100 * (m["B"] - m["A"]) / max(m["A"], 1e-9), 1),
        "shuffle_cost_pct": round(100 * (m["C"] - m["A"]) / max(m["A"], 1e-9), 1),
        "imagination_pct": round(100 * (m["E"] - m["D"]) / max(m["E"], 1e-9), 1),
        "latent_fidelity": lat,
        "n_windows": n,
        "verdict": _verdict(m, lat),
    }


def _verdict(m, lat):
    vu = 100 * (m["B"] - m["A"]) / max(m["A"], 1e-9)
    im = 100 * (m["E"] - m["D"]) / max(m["E"], 1e-9)
    if vu < 3 and im < 3:
        return "dynamics integrator — vision inert"
    if vu >= 3 and im >= 5:
        return "genuine imagination — vision used AND predicts w/o actions"
    if vu >= 3:
        return "vision used, but leans on given actions (weak imagination)"
    return "mixed / weak vision signal"
