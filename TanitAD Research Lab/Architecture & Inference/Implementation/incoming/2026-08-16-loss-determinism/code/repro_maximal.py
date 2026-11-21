"""ALL-LEVERS-ON RNG enumeration — covers the branch-gated code the minimal
build skips (selector, anchor head, o3 band rows, plan, goal cat args).

C74: an enumeration verified only on the paths a minimal batch happens to take
is not an enumeration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_SP = Path(__file__).parent
sys.path.insert(0, str(_SP))
from repro_nondet import (_install, _ACTIVE, _HITS, _sub_cfgs, T,  # noqa
                          V6Config, V6Stack)


def build_max(seed=0):
    torch.manual_seed(seed)
    cfg = V6Config(d_tac=32, d_str=16, adapter_hidden=32, f_hidden_tac=32,
                   f_hidden_str=32, f_blocks=1, aux_hidden=16, sigreg_slices=8,
                   plan_steps=6, dt=0.1, op_band_s=(0.0, 0.2),
                   tac_band_s=(0.2, 0.6), hz_op=10.0, hz_tac=2.0, hz_str=0.5,
                   d_plan_feat=16, emission_hidden=16, d_goal_embed=128,
                   n_candidates=8, selector="goal", anchor_goal="snap_lat",
                   goal_cat_args=True, goal_factored=True,
                   goal_multilabel=True, n_anchors=8, n_lat_bins=4,
                   plan_wta_eps=0.1, sigreg_free_dims=4, **_sub_cfgs())
    s = V6Stack(cfg)
    g = torch.Generator().manual_seed(3)
    steps = list(range(1, cfg.plan_steps + 1))
    s.anchor_head.load_anchor_table(
        torch.randn(cfg.n_anchors, len(steps), 2, generator=g) * 3.0,
        horizons=steps, dt=cfg.dt)
    return s


def main():
    _install()
    rep = {}
    for stage in ("S-W", "S-T", "S-S", "S-J"):
        s = build_max()
        s.train()
        b = T.synthetic_train_batch(s, batch=2, k=4, seed=7)
        b["gt_wp"] = torch.zeros(2, 2, 2)
        w = T.V6LossWeights(lambda_plan=1.0, w_select=1.0, w_anchor=1.0)
        kw = dict(stage=stage, o1_k=2, o5_k=2, weights=w, o3_band_rows=1,
                  o3_mode="context", o5_mode="linear-decay")
        torch.manual_seed(123)
        st0 = torch.random.get_rng_state().clone()
        _ACTIVE[0] = True
        _HITS.clear()
        r = T.v6_loss_step(s, b, generator=torch.Generator().manual_seed(11),
                           **kw)
        _ACTIVE[0] = False
        rep[stage] = {
            "terms": r["log"]["terms"],
            "global_rng_consumed": not torch.equal(
                st0, torch.random.get_rng_state()),
            "sites": sorted(_HITS),
        }
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
