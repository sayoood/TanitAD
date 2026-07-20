"""Does flagship-19k actually USE vision, or is it a dynamics integrator?

The grounded ADE is measured WITH the true future action sequence, which
over-determines the pose — so a good score can hide an inert encoder. This
probe isolates vision with a 2x2 (+ latent fidelity):

  vision  x  future-actions
  ----------------------------------------------------------------
  A real     true    -> baseline (the 0.628 regime)
  B mean     true    -> ablate scene, keep true actions
  C shuffle  true    -> WRONG scene, right stats, true actions
  D real     NONE    -> imagination regime: coast on last action, predict
                        the future from vision alone
  E mean     NONE    -> imagination floor (no scene, no future actions)

Reads:
  B~A and C~A  => pose ignores scene content when actions are given
                  (dynamics integrator; actions carry the trajectory)
  D << E       => with actions withheld, REAL vision predicts far better than
                  no-vision => genuine visual imagination
  D ~ E        => even without actions, vision adds nothing => no imagination
  latent cos(real) >> cos(mean) => imagined latents track the true scene
                  evolution BECAUSE of vision (the world-model 'imagination edge')
"""
import sys
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

import torch
import torch.nn.functional as F
from pathlib import Path

from taniteval import loaders, data
from taniteval.registry import MODELS
from tanitad.models.metric_dynamics import rollout_decode, rollout_transitions
from driving_diagnostic import WP_STEPS, gt_ego_waypoints

DEV = "cuda"
SPEED_SCALE = 10.0
K = max(WP_STEPS)                      # 20
WIN = 8
IDX = [k - 1 for k in WP_STEPS]        # 0-based horizon indices into k-rollout


def ablate(states, mode, gen):
    if mode == "real":
        return states
    if mode == "mean":                 # remove scene identity, keep global stats
        return states.mean(dim=0, keepdim=True).expand_as(states).clone()
    if mode == "shuffle":              # pair each rollout with a WRONG scene
        return states[torch.randperm(states.shape[0], generator=gen,
                                     device=states.device)]
    raise ValueError(mode)


@torch.no_grad()
def run():
    e = [m for m in MODELS if m["key"] == "flagship-speed"][0]
    L = loaders.load(e, DEV)
    model, step_readout = L["model"], L["step_readout"]
    model.eval()
    files = data.list_val_episodes(
        "/root/valdata/physicalai-val-0c5f7dac3b11", 40)[:12]
    eps = data.load_frames(files)
    gen = torch.Generator(device=DEV).manual_seed(0)

    conds = {"A real+trueA": ("real", True), "B mean+trueA": ("mean", True),
             "C shuf+trueA": ("shuffle", True), "D real+noA": ("real", False),
             "E mean+noA": ("mean", False)}
    ade = {c: [] for c in conds}
    cos_real, cos_mean = [], []

    for ep in eps:
        fr = ep.feats                                  # [T,9,H,W] uint8 (RawEp)
        T = fr.shape[0]
        for i in range(0, T - WIN - K, 8 * 8):         # coarse stride, batch below
            ch = list(range(i, min(i + 8 * 8, T - WIN - K), 8))
            if not ch:
                continue
            last = torch.tensor([t + WIN - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + WIN]) for t in ch]
                             ).to(DEV).float().div_(255.0)
            aw = torch.stack([ep.actions[t:t + WIN] for t in ch]).to(DEV)
            fa = torch.stack([ep.actions[t + WIN:t + WIN + K] for t in ch]
                             ).to(DEV)
            v0 = (ep.poses[last, 3:4] / SPEED_SCALE).to(DEV)
            aw = torch.cat([aw, v0[:, None].expand(-1, aw.shape[1], -1)], -1)
            fa = torch.cat([fa, v0[:, None].expand(-1, fa.shape[1], -1)], -1)
            gt = gt_ego_waypoints(ep.poses, last).to(DEV)    # [b,4,2]
            states = model.encode_window(fw)                 # [b,W,S]

            for c, (vmode, use_fa) in conds.items():
                st = ablate(states, vmode, gen)
                wp, _ = rollout_decode(model.predictor, st, aw,
                                       fa if use_fa else None, step_readout, K)
                pred = wp[:, IDX]                             # [b,4,2]
                ade[c].append(torch.linalg.norm(pred - gt, dim=-1).mean(1).cpu())

            # latent fidelity: do imagined latents track the TRUE future scene?
            # rollout_transitions -> list of k (z_prev, z_hat); stack the z_hats.
            tr = rollout_transitions(model.predictor, states, aw, fa, K)
            zh = torch.stack([p[1] for p in tr], dim=1)          # [b,k,S]
            fut = torch.stack([torch.as_tensor(fr[t + WIN:t + WIN + K])
                               for t in ch]).to(DEV).float().div_(255.0)
            b, k = fut.shape[0], fut.shape[1]
            z_true = model.encode(fut.reshape(b * k, *fut.shape[2:])
                                  ).reshape(b, k, -1)
            tr_m = rollout_transitions(
                model.predictor, ablate(states, "mean", gen), aw, fa, K)
            zh_m = torch.stack([p[1] for p in tr_m], dim=1)
            cos_real.append(F.cosine_similarity(zh, z_true, dim=-1).mean().cpu())
            cos_mean.append(F.cosine_similarity(zh_m, z_true, dim=-1).mean().cpu())

    print("=" * 60)
    print("flagship-speed @19k — vision x action ablation (ADE 0-2s, m)")
    print("=" * 60)
    base = float(torch.cat(ade["A real+trueA"]).mean())
    for c in conds:
        v = float(torch.cat(ade[c]).mean())
        print(f"  {c:16} ADE {v:.3f}   (Δ vs A {v - base:+.3f})")
    print("-" * 60)
    cr, cm = float(torch.stack(cos_real).mean()), float(torch.stack(cos_mean).mean())
    print(f"  latent fidelity cos(z_hat, z_true):  real {cr:.3f}   "
          f"mean-vision {cm:.3f}   (Δ {cr - cm:+.3f})")
    n = int(sum(len(x) for x in ade["A real+trueA"]) if
            isinstance(ade["A real+trueA"][0], torch.Tensor) else 0)
    print(f"  n_windows ~ {sum(t.numel() for t in ade['A real+trueA'])}")


if __name__ == "__main__":
    run()
