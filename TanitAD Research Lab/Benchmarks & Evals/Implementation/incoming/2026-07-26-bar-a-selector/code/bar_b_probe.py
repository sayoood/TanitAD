"""BAR B — the cheapest FALSIFYING probe of the off-path-augmentation hypothesis.

STATUS: **HYPOTHESIS-grade.** Nothing is trained. This probe cannot confirm the
hypothesis; it can only weaken it or leave it standing.

THE HYPOTHESIS (from the brief, 2026-07-26, HYPOTHESIS-grade as received)
-------------------------------------------------------------------------
`wm_canary_ade_2s` = 1.1409 against a <= 0.55 bar, IDENTICAL in both goal modes,
so it is a genuine world-model deficiency the selector fix does not touch. It
must fall 2.07x and no Bar-B lever has ever been identified. A candidate arrived
today: the closed-loop envelope is NOT a renderer-fidelity limit -- the yaw warp
is geometrically exact (max|dH| = 0.000e+00 over 30 conditions) -- so roughly half
of what it measures is our arm's own OOD sensitivity. That points at TRAINING-TIME
OFF-PATH AUGMENTATION as a world-model lever.

WHAT THIS PROBE ACTUALLY TESTS, AND WHAT IT CANNOT
---------------------------------------------------
The canary rolls the operative predictor under TRUE actions, so it stays ON-path
by construction. There is no off-path distance to correlate against on these
rollouts. What IS testable, and is a NECESSARY condition for the OOD reading:

    does world-model error CONCENTRATE on dynamically-unusual / off-nominal
    states, or is it a flat error floor?

If the canary error is roughly flat across turning / speed / latent-novelty
strata, then the world model is uniformly inaccurate and off-path augmentation
addresses a tail that is not where the error lives -- which weakens the lever.
If it rises steeply with novelty, the lever survives its first falsification
attempt (it is still not confirmed).

PRE-REGISTERED READING (committed before the numbers were computed)
-------------------------------------------------------------------
  SURVIVES   -- the strongest covariate shows top-decile / bottom-decile canary
                ratio >= 1.5x AND |Spearman rho| >= 0.20, with the decile gap
                separated by episode-cluster bootstrap.
  WEAKENED   -- best ratio < 1.5x or |rho| < 0.20: WM error does not concentrate
                on unusual states, so "our arm's own OOD sensitivity" is not
                where the 1.1409 lives.
Either way the output is HYPOTHESIS-grade and may not authorise a GPU-week.

COVARIATES (all computed from data that already exists; none needs training)
  head_deg_abs       |net heading change| over the observation window  (turning)
  v0                 observed speed                                    (regime)
  gt_cross_2s_abs    |lateral| GT displacement at 2 s                  (path departure)
  dv_window          speed change across the observation window        (accel/brake)
  latent_novelty     ||state_last - mean(state_last)|| / std           (DIRECT novelty
                     in the encoder's own representation -- the closest thing to
                     "OOD" that exists without a training-set density model)
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

STACK = "/root/v4eval/stack"
sys.path.insert(0, STACK + "/scripts")
sys.path.insert(0, STACK)
sys.path.insert(0, "/root/taniteval")

import driving_diagnostic as dd  # noqa: E402
import eval_flagship_v4 as E  # noqa: E402
from taniteval.ci import (episode_cluster_bootstrap,  # noqa: E402
                          paired_episode_cluster_bootstrap)

CKPT = "/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt"
HCFG = "/workspace/_v4gate/flagship-v4-fromscratch-30k/config.json"
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
ANCH = "/root/models/flagship-v4-fromscratch-15k/flagship_v4_anchors_dense.pt"
DEV = "cuda"
OUT = Path("/root/bara/bar_b_probe.json")
HORIZONS = (5, 10, 15, 20)
K_MAX = 20
RATIO_BAR, RHO_BAR = 1.5, 0.20


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


@torch.no_grad()
def main():
    from tanitad.models.flagship_v15 import SPEED_SCALE
    from tanitad.models.metric_dynamics import gt_ego_waypoints, rollout_decode

    R = {"_experiment": "BAR B probe -- does WM canary error concentrate on "
                        "dynamically-unusual states?",
         "_evidence_class": "MEASURED (ours) applied to a HYPOTHESIS",
         "_tier": "PROVISIONAL -- one path, one probe; may not authorise a GPU-week",
         "_nothing_trained": True,
         "_prereg": {"SURVIVES": f"best decile ratio >= {RATIO_BAR} AND "
                                 f"|rho| >= {RHO_BAR}, decile gap separated",
                     "WEAKENED": "otherwise -- WM error does not live on unusual "
                                 "states"},
         "_caveat": "the canary rolls under TRUE actions and is therefore ON-path "
                    "by construction; this tests a NECESSARY condition for the OOD "
                    "reading, not off-path distance itself",
         "_host": platform.node(), "_python": sys.version.split()[0],
         "_ckpt": {"path": CKPT, "md5": md5(CKPT)}, "_stack_root": STACK}

    cfg = E._eval_cfg()
    plan = E._plan(cfg)
    ds = E.build_val_dataset_v4(VAL, cfg, plan)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    world, grounding, head, step, hcfg, goal_head = E.load_v4_from_ck(
        ck, DEV, head_config_path=HCFG, anchors_dense_path=ANCH)
    del ck
    R["_ckpt_step"] = int(step)

    step_readout = grounding.step["op"]
    sel = [i for i, (e, t) in enumerate(ds.index) if e < 40 and t % 8 == 0]
    wp_idx = torch.tensor([k - 1 for k in HORIZONS], device=DEV)
    E_, V0, HD, CR, DV, ST, EID = [], [], [], [], [], [], []
    pose_cache: dict[int, torch.Tensor] = {}
    t0 = time.time()
    for b0 in range(0, len(sel), 16):
        idx = sel[b0:b0 + 16]
        items = [ds[i] for i in idx]
        fr = torch.stack([x["frames"] for x in items]).to(DEV)
        aw2 = torch.stack([x["actions"] for x in items]).to(DEV).float()
        fa2 = torch.stack([x["future_actions"] for x in items]).to(DEV).float()
        fp = torch.stack([x["future_poses"] for x in items]).to(DEV).float()
        pl = torch.stack([x["pose_last"] for x in items]).to(DEV).float()
        v0 = pl[:, 3]
        vch = (v0 / SPEED_SCALE)[:, None, None]
        aw = torch.cat([aw2, vch.expand(-1, aw2.shape[1], -1)], dim=-1)
        fa = torch.cat([fa2, vch.expand(-1, fa2.shape[1], -1)], dim=-1)
        gt = gt_ego_waypoints(pl, fp, HORIZONS)
        states = world.encode_window(fr)
        wp_full, _ = rollout_decode(world.predictor, states, aw, fa,
                                    step_readout, K_MAX)
        pred = wp_full.index_select(1, wp_idx).float()
        E_.append((pred - gt).norm(dim=-1).mean(dim=1).cpu())    # canary ADE@2s
        V0.append(v0.cpu())
        CR.append(gt[:, -1, 1].abs().cpu())                      # |cross| @2s
        ST.append(states[:, -1].float().cpu())
        for i in idx:
            e_i, t = ds.index[i]
            po = pose_cache.get(e_i)
            if po is None:
                po = torch.as_tensor(ds.episodes[e_i].poses, dtype=torch.float32)
                pose_cache[e_i] = po
            last = torch.tensor([t + ds.window - 1])
            HD.append(dd.net_heading_change_deg(po, last))
            DV.append(torch.tensor([float(po[t + ds.window - 1, 3]
                                          - po[t, 3])]))
            EID.append(int(ds.episodes[e_i].episode_id))
        if b0 % 320 == 0:
            print(f"  [bar-b] {b0 + len(idx)}/{len(sel)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    err = torch.cat(E_).numpy()
    st = torch.cat(ST)
    cov = {
        "head_deg_abs": torch.cat(HD).abs().numpy().astype(float),
        "v0": torch.cat(V0).numpy().astype(float),
        "gt_cross_2s_abs": torch.cat(CR).numpy().astype(float),
        "dv_window": torch.cat(DV).abs().numpy().astype(float),
        "latent_novelty": ((st - st.mean(0, keepdim=True)).norm(dim=-1)
                           / st.std()).numpy().astype(float),
    }
    eids = [str(x) for x in EID]
    R["canary"] = {
        "n_windows": int(err.shape[0]),
        "n_episodes": len(set(eids)),
        "wm_canary_ade_2s_mean": round(float(err.mean()), 4),
        "committed_value": 1.1409,
        "abs_diff": round(abs(float(err.mean()) - 1.1409), 4),
        "bar": 0.55,
        "must_fall_by": round(float(err.mean()) / 0.55, 3),
        "interval": episode_cluster_bootstrap(err, eids, n_boot=2000, seed=0),
        "distribution": {q: round(float(np.quantile(err, q / 100)), 4)
                         for q in (5, 25, 50, 75, 95, 99)},
        "frac_windows_already_under_bar": round(float((err < 0.55).mean()), 4),
    }

    R["covariates"] = {}
    for name, x in cov.items():
        rho = spearman(x, err)
        order = np.argsort(x)
        n = len(x)
        lo, hi = order[:n // 10], order[-(n // 10):]
        lo_m, hi_m = float(err[lo].mean()), float(err[hi].mean())
        d = np.zeros(n)
        d[hi] = 1.0
        # decile gap with an episode-cluster interval on the top decile alone
        R["covariates"][name] = {
            "spearman_rho": round(rho, 4),
            "bottom_decile_canary": round(lo_m, 4),
            "top_decile_canary": round(hi_m, 4),
            "ratio_top_over_bottom": round(hi_m / lo_m, 3) if lo_m > 0 else None,
            "deciles": [round(float(err[order[k * (n // 10):(k + 1) * (n // 10)]]
                                    .mean()), 4) for k in range(10)],
            "covariate_deciles": [round(float(x[order[k * (n // 10):
                                                     (k + 1) * (n // 10)]].mean()), 4)
                                  for k in range(10)],
            "top_decile_interval": episode_cluster_bootstrap(
                err[hi], [eids[i] for i in hi], n_boot=2000, seed=0),
            "bottom_decile_interval": episode_cluster_bootstrap(
                err[lo], [eids[i] for i in lo], n_boot=2000, seed=0),
        }

    best = max(R["covariates"].items(),
               key=lambda kv: (kv[1]["ratio_top_over_bottom"] or 0))
    R["READING"] = {
        "strongest_covariate": best[0],
        "ratio": best[1]["ratio_top_over_bottom"],
        "rho": best[1]["spearman_rho"],
        "verdict": ("SURVIVES" if (best[1]["ratio_top_over_bottom"] or 0) >= RATIO_BAR
                    and abs(best[1]["spearman_rho"]) >= RHO_BAR else "WEAKENED"),
        "_tier": "PROVISIONAL / HYPOTHESIS-grade -- this probe cannot confirm a "
                 "lever and may not authorise a restart",
    }
    OUT.write_text(json.dumps(R, indent=2, default=str))
    print(json.dumps(R, indent=2, default=str))


if __name__ == "__main__":
    main()
