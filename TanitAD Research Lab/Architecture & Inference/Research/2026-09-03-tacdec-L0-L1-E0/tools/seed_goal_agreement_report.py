#!/usr/bin/env python
"""L0's BEFORE/AFTER on the 64 token pairs — measured on RUNNING `plan()`.

⛔ THIS DOES NOT REIMPLEMENT `seed_goal_mismatch.py`. That tool is READ-ONLY
and is re-run UNMODIFIED for the BEFORE number (`raw/
seed_goal_mismatch_AFTER_L0_default.json` — the same 62/64 it banked before the
edit, which is the deliberate-regression evidence that the default path did not
move). What this tool adds is the half the arithmetic tool cannot reach: it
drives a REAL `RefAV1.plan()` once per (lat, lon) token pair per
`cost_time_grid`, captures the action feed the GOAL rollout actually receives,
and compares it ELEMENT FOR ELEMENT with the feed `_cost_chunk` gives the
canonical seed — `seed_goal_mismatch.py`'s own metric, applied to running code
rather than to a transcription of it.

It also reports the numerical consequence: `1 - cos` between the seed's own
terminal field and the goal, which is the quantity the planner's cost actually
pays.

⚠️ The model is a TINY RANDOM-INIT rig (the fixture of
`stack/tests/test_refa_v1_plan_goal.py`). That is exact for this question: the
identity is a property of which ACTIONS are fed, not of what the weights know,
and the decoded token is pinned so all 64 pairs can be exercised.

TIER: n/a (a wiring identity, not a capability). EVIDENCE CLASS: MEASURED.
GPU: none.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v
from tanitad.refs.refa_v1 import (COST_TIME_GRIDS, GOAL_TIME_GRIDS, RefAV1,
                                  RefAV1Config)
from tanitad.refs.refa_v1_plan import PlanConfig

COS32_TOL = 4.0 * 1.1920928955078125e-07     # 4 x spacing(1f)


class _OneHotHead(torch.nn.Module):
    def __init__(self, i: int, n: int):
        super().__init__()
        self.i, self.n = int(i), int(n)

    def forward(self, x):
        out = torch.full((x.shape[0], self.n), -10.0, dtype=x.dtype,
                         device=x.device)
        out[:, self.i] = 10.0
        return out


def _model(seed: int = 0) -> RefAV1:
    torch.manual_seed(seed)
    cfg = RefAV1Config(
        tac_vocab_version="v7.0", d_enc=16, d_state=16, n_tokens=8,
        op_layers=1, op_heads=2, op_window=2, tac_layers=1, tac_queries=4,
        str_dim=8, str_layers=1,
        strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1, n_heads=2,
                                            d_ctx=16, d_cmd=8),
        tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1, n_heads=2,
                                          d_intent=16))
    m = RefAV1(cfg).eval()
    m.std.fit(torch.randn(256, cfg.d_enc))
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v0", type=float, default=10.0)
    a = ap.parse_args()

    m = _model(a.seed)
    c = m.cfg
    g = torch.Generator().manual_seed(a.seed)
    feats = torch.randn(1, c.op_window, c.n_tokens, c.d_enc, generator=g)
    nav = torch.randint(0, 4, (1,), generator=g)
    pc = PlanConfig(horizon=c.plan_steps, dt=c.op_dt, seed=0, n_samples=4,
                    n_iters=1, n_elites=2)
    lats = list(tactical_lat_actions(c.tac_vocab_version))
    lons = list(tactical_lon_actions_v(c.tac_vocab_version))

    with torch.no_grad():
        field = m.encode(feats)
        last = m._last_state(field)
        brains = m._run_brains(field.mean(dim=-2), nav)
    intent = brains["intent"]
    z0 = m._tac_field(last)
    stride = int(round(c.tac_dt / c.op_dt))
    reg = torch.tensor([min(j * stride, c.plan_steps - 1)
                        for j in range(c.tac_steps)], dtype=torch.long)

    res = {"tool": "seed_goal_agreement_report.py", "tier": "n/a",
           "evidence_class": "MEASURED (running plan() on a tiny rig)",
           "config": {"op_steps": c.op_steps, "op_dt": c.op_dt,
                      "tac_dt": c.tac_dt, "tac_steps": c.tac_steps,
                      "stride": stride, "plan_horizon_s": c.plan_horizon_s,
                      "plan_steps": c.plan_steps,
                      "plan_level": c.plan_level,
                      "tac_vocab_version": c.tac_vocab_version},
           "goal_time_grids": list(GOAL_TIME_GRIDS),
           "cost_time_grids": list(COST_TIME_GRIDS),
           "v0_used_m_s": a.v0, "n_token_pairs": len(lats) * len(lons),
           "by_arm": {}, "tokens": []}

    t0 = time.time()
    for gtg in GOAL_TIME_GRIDS:
        for ctg in COST_TIME_GRIDS:
            n_agree = 0
            worst_gt = 0.0
            worst_tok = None
            for lat in lats:
                for lon in lons:
                    m.lat_head = _OneHotHead(lats.index(lat), m.n_lat)
                    m.lon_head = _OneHotHead(lons.index(lon), m.n_lon)
                    seen = []
                    orig = m.tactical.rollout

                    def spy(fld, acts, intent=None, last_only=False):
                        o = orig(fld, acts, intent=intent, last_only=last_only)
                        if int(acts.shape[0]) == 1 and last_only:
                            seen.append((acts.detach().clone(),
                                         o.detach().clone()))
                        return o

                    m.tactical.rollout = spy
                    try:
                        r = m.plan(feats, v0=a.v0, nav_cmd=nav, plan_cfg=pc,
                                   cost_time_grid=ctg, goal_time_grid=gtg)
                    finally:
                        m.tactical.rollout = orig
                    g_acts, g_z = seen[-1]
                    ctrl = r.goal_action["controls"]
                    s_acts = m._model_actions(
                        ctrl[:c.plan_steps][None],
                        torch.tensor([a.v0], dtype=torch.float32), "kappa")
                    if ctg == "tactical":
                        s_acts = s_acts.index_select(1, reg)
                    same = (g_acts.shape == s_acts.shape
                            and torch.equal(g_acts, s_acts))
                    with torch.no_grad():
                        z_seed = orig(z0, s_acts, intent=intent,
                                      last_only=True)
                        gt = float(1.0 - F.cosine_similarity(
                            z_seed.flatten(1), g_z.flatten(1), dim=-1))
                    n_agree += int(same)
                    if abs(gt) > abs(worst_gt):
                        worst_gt, worst_tok = gt, (lat, lon)
                    res["tokens"].append({
                        "goal_time_grid": gtg, "cost_time_grid": ctg,
                        "lat": lat, "lon": lon,
                        "feeds_equal": bool(same),
                        "n_tactical_steps_differing":
                            int((g_acts != s_acts).any(-1).sum())
                            if g_acts.shape == s_acts.shape else None,
                        "seed_goal_term_1mcos_f32": gt,
                        "goal_source": r.goal_source,
                        "goal_time_grid_on_result": r.goal_time_grid})
            res["by_arm"][f"{gtg}__{ctg}"] = {
                "goal_time_grid": gtg, "cost_time_grid": ctg,
                "is_default": gtg == "full" and ctg == "dense",
                "n_token_pairs": len(lats) * len(lons),
                "n_agreeing_feeds": n_agree,
                "n_mismatching_feeds": len(lats) * len(lons) - n_agree,
                "worst_seed_goal_term_1mcos_f32": worst_gt,
                "worst_token": worst_tok,
                "identity_reached": bool(abs(worst_gt) <= COS32_TOL),
                "fp32_cosine_resolution_4ulp": COS32_TOL,
            }
            print(f"[{gtg}/{ctg}] agree {n_agree}/64  worst 1-cos {worst_gt:.3e}"
                  f"  ({time.time()-t0:.1f}s)", flush=True)

    d = res["by_arm"]
    res["headline"] = (
        f"DEFAULT (goal_time_grid='full', cost_time_grid='dense'): "
        f"{d['full__dense']['n_agreeing_feeds']}/64 token pairs feed the goal "
        f"the seed's own actions. REPAIR (goal_time_grid='plan'): "
        f"{d['plan__dense']['n_agreeing_feeds']}/64 dense and "
        f"{d['plan__tactical']['n_agreeing_feeds']}/64 tactical, worst "
        f"seed-goal term {max(abs(d['plan__dense']['worst_seed_goal_term_1mcos_f32']), abs(d['plan__tactical']['worst_seed_goal_term_1mcos_f32'])):.3e} "
        f"against a float32 cosine resolution of {COS32_TOL:.3e}.")
    res["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"by_arm": res["by_arm"], "headline": res["headline"]},
                     indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
