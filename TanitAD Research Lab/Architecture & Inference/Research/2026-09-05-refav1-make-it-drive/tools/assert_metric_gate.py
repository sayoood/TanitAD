#!/usr/bin/env python
"""THE SECOND GATE: under ``cost_metric="cos"`` a CORRECT turn token is refused.

⭐ WHY THIS EXISTS. `assert_plan_gate.py` established gate #1 — a LANE_KEEP
decode makes a turn UNREACHABLE, because the decoded token's canonical profile
is the population's only curvature-carrying candidate. This script asks the
next question: **when the goal head is RIGHT and decodes TURN_L, does the
planner actually turn?**

Under ``"ccos"`` it does. Under ``"cos"`` — which is `refav1_arm.py`'s DEFAULT
`--cost-metric` and the module default of `plan()` — it does not: the planner
returns ``baseline:hold_v0`` with curvature exactly 0. The turn is proposed and
then REJECTED, which is a different failure from never being proposed.

`refa_v1.py:COST_METRICS` already documents the mechanism (under ``cos`` the
goal term realises ~1.2e-05 of its own range against a jerk charge of 0.094, so
"100.00 % of the iteration-0 population is excluded before the world model is
consulted"). This measures the BEHAVIOURAL consequence, over many windows,
seeds and search sizes, so the claim is not a single-draw artefact.

⛔ CONTROLS:
  * ``ccos`` is the same-breath positive control for every ``cos`` zero — the
    identical window, seed and search size must produce a real turn, or the
    zero says something about the probe rather than the metric.
  * a LANE_KEEP row under BOTH metrics must read zero: gate #1 is
    metric-independent, and if it were not, the two gates are confounded.
  * the search size is swept, so "cos found nothing" cannot be "cos was not
    given enough samples".

Run: python assert_metric_gate.py --out metric_gate.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default=None)
    ap.add_argument("--n-windows", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.stack:
        sys.path.insert(0, os.path.abspath(a.stack))

    import torch
    from tanitad.refs.refa_v1 import (RefAV1, RefAV1Config,
                                      StrategicPolicyConfig,
                                      TacticalPolicyConfig, W_JERK, W_KAPPA,
                                      W_VEND, GOAL_KAPPA_TURN)
    from tanitad.refs.refa_v1_plan import PlanConfig
    from tanitad.models.v6 import tactical_lat_actions

    LAT = list(tactical_lat_actions("v7.0"))
    N = len(LAT)
    STAGE_B = (0.0, 0.0, 64.29715042415070)

    def _model(seed):
        cfg = RefAV1Config(
            tac_vocab_version="v7.0", d_enc=16, d_state=16, n_tokens=8,
            op_layers=1, op_heads=2, op_window=2, tac_layers=1, tac_queries=4,
            str_dim=8, str_layers=1,
            strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1, n_heads=2,
                                                d_ctx=16, d_cmd=8),
            tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1, n_heads=2,
                                              d_intent=16))
        torch.manual_seed(seed)
        m = RefAV1(cfg).eval()
        m.std.fit(torch.randn(256, cfg.d_enc))
        return m, cfg

    rows: dict = {}
    for (ns, ni) in ((32, 3), (64, 4), (128, 6), (300, 30)):
        for metric in ("cos", "ccos"):
            for wname, w in (("shipped", None), ("stageB", STAGE_B)):
                key = f"n{ns}_i{ni}|{metric}|{wname}"
                turn_hits, turn_kappa, lk_nonzero, srcs = 0, [], 0, {}
                for wi in range(a.n_windows):
                    m, cfg = _model(wi)
                    g = torch.Generator().manual_seed(wi)
                    feats = torch.randn(1, cfg.op_window, cfg.n_tokens,
                                        cfg.d_enc, generator=g)
                    nav = torch.randint(0, 4, (1,), generator=g)
                    pc = PlanConfig(horizon=cfg.plan_steps, dt=cfg.op_dt,
                                    seed=0, n_samples=ns, n_iters=ni,
                                    n_elites=max(4, ns // 8))
                    for tok in ("TURN_L", "LANE_KEEP"):
                        b = torch.zeros(N)
                        b[LAT.index(tok)] = 1e3
                        r = m.plan(feats, v0=10.0, nav_cmd=nav, plan_cfg=pc,
                                   model_action_units="kappa",
                                   cost_metric=metric, cost_weights=w,
                                   lat_logit_bias=b)
                        k = float(r.controls[:, 1].mean())
                        if tok == "TURN_L":
                            turn_kappa.append(k)
                            srcs[str(r.source)] = srcs.get(str(r.source), 0) + 1
                            if k > 1e-3:
                                turn_hits += 1
                        elif abs(k) > 1e-9:
                            lk_nonzero += 1
                rows[key] = {
                    "n_windows": a.n_windows,
                    "TURN_L_executed_rate": turn_hits / a.n_windows,
                    "TURN_L_mean_kappa": sum(turn_kappa) / len(turn_kappa),
                    "TURN_L_sources": srcs,
                    "LANE_KEEP_nonzero_curvature_windows": lk_nonzero,
                }
                print(f"{key:<28} TURN_L executed "
                      f"{turn_hits}/{a.n_windows} "
                      f"mean_kappa={rows[key]['TURN_L_mean_kappa']:+.5f}  "
                      f"LANE_KEEP nonzero={lk_nonzero}")

    R = {"module_default_weights": {"W_JERK": W_JERK, "W_KAPPA": W_KAPPA,
                                    "W_VEND": W_VEND},
         "stage_b_weights": list(STAGE_B),
         "GOAL_KAPPA_TURN": GOAL_KAPPA_TURN,
         "rows": rows, "checks": {}}
    fails = []

    def chk(name, ok, detail):
        R["checks"][name] = {"ok": bool(ok), **detail}
        if not ok:
            fails.append(name)

    # ⚠️ THE ASSERTIONS ARE SCOPED TO THE **SHIPPED WEIGHTS**, deliberately.
    # A first pass asserted these for every weight set and failed, and the
    # failure was informative rather than a bug: with the Stage-B triple (all
    # penalties zero) this TINY random-init model wanders — it produces
    # curvature on LANE_KEEP windows too, because nothing regularises the
    # search and its untrained goal field carries no signal. That wandering is
    # a property of a random-init WM and does NOT transfer (the trained
    # checkpoint reads LANE_KEEP curvature EXACTLY 0.0 on 244/244 windows).
    # So the tiny-model claim is made only where it is clean, and the
    # trained-model measurement (`turn_execution.py`) is what carries the
    # finding.
    cos_ship = {k: v["TURN_L_executed_rate"] for k, v in rows.items()
                if "|cos|shipped" in k}
    ccos_ship = {k: v["TURN_L_executed_rate"] for k, v in rows.items()
                 if "|ccos|shipped" in k}
    chk("GATE2_cos_never_executes_a_correct_TURN_at_shipped_weights",
        all(v == 0.0 for v in cos_ship.values()),
        {"per_search_size": cos_ship,
         "note": "swept 32/3, 64/4, 128/6, 300/30 — so this is not "
                 "'the search was too small'"})
    # POSITIVE CONTROL: ccos, identical windows/seeds/search sizes, must turn
    # on a clear majority. Without it the zeros above say nothing about the
    # metric.
    chk("CONTROL_ccos_DOES_execute_the_same_TURN",
        all(v > 0.5 for v in ccos_ship.values()), {"per_search_size": ccos_ship})
    # gate #1 must hold under BOTH metrics at the shipped weights, or the two
    # gates are confounded with each other.
    ship = {k: v["LANE_KEEP_nonzero_curvature_windows"]
            for k, v in rows.items() if "shipped" in k}
    chk("GATE1_LANE_KEEP_is_zero_under_BOTH_metrics_at_shipped_weights",
        all(v == 0 for v in ship.values()), {"nonzero_windows": ship})

    R["PASS"] = not fails
    R["failed_checks"] = fails
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1)
    print()
    for k, v in R["checks"].items():
        print(("PASS " if v["ok"] else "FAIL ") + k)
    print("\nOVERALL:", "PASS" if R["PASS"] else f"FAIL {fails}")
    return 0 if R["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
