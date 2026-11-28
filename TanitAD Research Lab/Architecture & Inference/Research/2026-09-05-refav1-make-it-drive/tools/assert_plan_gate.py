#!/usr/bin/env python
"""The curvature gate, measured END-TO-END through the real ``plan()``.

⭐ WHY THIS EXISTS. `assert_gate.py` proves the gate COMPONENT BY COMPONENT
(`canonical_controls`, `_baseline_controls`, `colored_noise`). That is an
argument about parts. This script drives the ACTUAL `RefAV1.plan()` — the same
function `refav1_arm.py` calls — and measures what comes out of it when the
goal head's decoded token is forced to each lateral token in turn. It answers
the only question that matters for driving: *does the token the goal head emits
DECIDE the planner's steering?*

It also settles a second question the Stage-B command raises. That arm runs
``--cost-weights 0.0,0.0,64.297`` (W_JERK=0, W_KAPPA=0) and `refav1_arm.py`
never passes ``target_speed``, while `_cost_chunk` guards the W_VEND term on
``if target_speed is not None``. If that reading is right, ALL THREE cost
weights are inert under the shipped arm and the cost is the goal term ALONE.
Read-and-conclude is exactly the error class CLAUDE.md warns about, so this
measures it: sweep W_VEND over 18 orders of magnitude and compare the returned
controls BIT-FOR-BIT.

⛔ CONTROLS. Every "no change" assertion is paired with a same-breath control
that MUST show a change, or the comparison is vacuous (a probe that always
returns the same thing would pass every inertness test):

  * W_VEND inert with target_speed=None  ..vs..  W_VEND ACTIVE when a
    target_speed IS supplied (the probe can see a difference)
  * LANE_KEEP gives ~zero curvature      ..vs..  TURN_L / TURN_R give
    large curvature of the CORRECT SIGN (the planner can steer at all)

Run: python assert_plan_gate.py --out plan_gate.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--n-iters", type=int, default=4)
    a = ap.parse_args()
    if a.stack:
        sys.path.insert(0, os.path.abspath(a.stack))

    import torch
    import torch.nn as nn
    from tanitad.refs.refa_v1 import (RefAV1, RefAV1Config, GOAL_KAPPA_TURN,
                                      StrategicPolicyConfig,
                                      TacticalPolicyConfig)
    from tanitad.refs.refa_v1_plan import PlanConfig

    R: dict = {"torch": torch.__version__, "checks": {}, "PASS": None}
    fails: list[str] = []

    def chk(name, ok, detail):
        R["checks"][name] = {"ok": bool(ok), **detail}
        if not ok:
            fails.append(name)

    # ---- a tiny REAL RefAV1, with the hierarchy (so the goal path runs) --- #
    # Shape/config helper mirrors stack/tests/test_cost_ccos.py::_cfg.
    cfg = RefAV1Config(
        tac_vocab_version="v7.0", d_enc=16, d_state=16, n_tokens=8,
        op_layers=1, op_heads=2, op_window=2, tac_layers=1, tac_queries=4,
        str_dim=8, str_layers=1,
        strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1, n_heads=2,
                                            d_ctx=16, d_cmd=8),
        tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1, n_heads=2,
                                          d_intent=16))
    torch.manual_seed(0)
    model = RefAV1(cfg).eval()
    model.std.fit(torch.randn(256, cfg.d_enc))
    LAT = ["LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC",
           "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R"]
    R["lat_vocab"] = LAT

    g = torch.Generator().manual_seed(0)
    feats = torch.randn(1, cfg.op_window, cfg.n_tokens, cfg.d_enc, generator=g)
    nav = torch.randint(0, 4, (1,), generator=g)
    V0 = 10.0
    pc = PlanConfig(horizon=cfg.plan_steps, dt=cfg.op_dt, seed=0,
                    n_samples=a.n_samples, n_iters=a.n_iters, n_elites=8)

    class _Forced(nn.Module):
        """Replaces `lat_head` so the DECODED TOKEN is the independent
        variable. Everything else in plan() is untouched — this is the one
        lever under test."""
        def __init__(self, idx: int, n: int = 8):
            super().__init__()
            self.idx, self.n = idx, n

        def forward(self, x):
            o = torch.full((*x.shape[:-1], self.n), -10.0)
            o[..., self.idx] = 10.0
            return o

    real_lat_head = model.lat_head
    STAGE_B_W = (0.0, 0.0, 64.29715042415070)   # the Stage-B triple, verbatim

    def _plan(w=STAGE_B_W, target_speed=None, metric="ccos"):
        return model.plan(feats, v0=V0, nav_cmd=nav, plan_cfg=pc,
                          model_action_units="kappa", cost_metric=metric,
                          cost_weights=w, target_speed=target_speed)

    # ---- 1. THE GATE, end-to-end through plan() -------------------------- #
    per_token = {}
    for i, tok in enumerate(LAT):
        model.lat_head = _Forced(i)
        res = _plan()
        kap = res.controls[:, 1]
        per_token[tok] = {
            "sustained_kappa_mean": float(kap.mean()),
            "abs_kappa_max": float(kap.abs().max()),
            "goal_lat": (res.goal_action or {}).get("lat"),
            "source": str(res.source),
        }
    model.lat_head = real_lat_head
    R["plan_by_forced_token"] = per_token

    # the token the head emits must be the token the goal carries — if this
    # fails the forcing did not take and every number below is meaningless
    chk("forcing_reaches_goal_action",
        all(per_token[t]["goal_lat"] == t for t in LAT),
        {"decoded": {t: per_token[t]["goal_lat"] for t in LAT}})

    lk = per_token["LANE_KEEP"]["abs_kappa_max"]
    tl = per_token["TURN_L"]["sustained_kappa_mean"]
    tr = per_token["TURN_R"]["sustained_kappa_mean"]
    chk("gate_LANE_KEEP_plan_has_zero_curvature", lk < 1e-6,
        {"abs_kappa_max": lk, "expect": "~0"})
    # POSITIVE CONTROLS: the planner CAN steer, and in the right direction.
    chk("gate_CONTROL_TURN_L_plan_steers_LEFT", tl > 1e-3,
        {"sustained_kappa_mean": tl, "expect": f"> 0 (up to {GOAL_KAPPA_TURN})"})
    chk("gate_CONTROL_TURN_R_plan_steers_RIGHT", tr < -1e-3,
        {"sustained_kappa_mean": tr, "expect": f"< 0 (down to -{GOAL_KAPPA_TURN})"})
    zero_tok = sorted(t for t, v in per_token.items()
                      if v["abs_kappa_max"] < 1e-6)
    chk("gate_zero_curvature_plans_are_exactly_LANE_KEEP_and_ABORT_LC",
        zero_tok == ["ABORT_LC", "LANE_KEEP"], {"zero_curvature_plans": zero_tok})

    # ---- 2. IS W_VEND INERT UNDER THE SHIPPED ARM? ----------------------- #
    model.lat_head = _Forced(LAT.index("TURN_L"))   # a token that DOES steer,
    #   so an active weight would have something to change
    base = _plan(w=(0.0, 0.0, 0.0), target_speed=None).controls
    vend_rows = {}
    inert = True
    for wv in (1e-6, 1.0, 64.29715042415070, 1e6, 1e12):
        c = _plan(w=(0.0, 0.0, wv), target_speed=None).controls
        same = bool(torch.equal(base, c))
        vend_rows[f"{wv:g}"] = {"bit_identical_to_w_vend_0": same,
                                "max_abs_diff": float((c - base).abs().max())}
        inert = inert and same
    R["w_vend_sweep_target_speed_None"] = vend_rows
    chk("W_VEND_is_INERT_when_target_speed_is_None", inert,
        {"n_settings": len(vend_rows), "range": "1e-6 .. 1e12"})

    # POSITIVE CONTROL, same breath: supply a target_speed and the SAME sweep
    # must now CHANGE the plan. Without this, "inert" could just mean the probe
    # is blind.
    base_ts = _plan(w=(0.0, 0.0, 0.0), target_speed=0.0).controls
    ts_rows, ts_changed = {}, False
    for wv in (1.0, 64.29715042415070, 1e6):
        c = _plan(w=(0.0, 0.0, wv), target_speed=0.0).controls
        d = float((c - base_ts).abs().max())
        ts_rows[f"{wv:g}"] = {"max_abs_diff": d}
        ts_changed = ts_changed or d > 0
    R["w_vend_sweep_target_speed_0"] = ts_rows
    chk("CONTROL_W_VEND_IS_ACTIVE_when_target_speed_supplied", ts_changed,
        {"rows": ts_rows, "expect": "some max_abs_diff > 0"})

    # ---- 3. and W_JERK / W_KAPPA are ZERO in the Stage-B triple ---------- #
    # so the Stage-B cost is the GOAL TERM ALONE. Assert by equality against a
    # plan run with the goal term and literally nothing else.
    only_goal = _plan(w=(0.0, 0.0, 0.0), target_speed=None).controls
    stage_b = _plan(w=STAGE_B_W, target_speed=None).controls
    chk("StageB_cost_equals_GOAL_TERM_ALONE",
        bool(torch.equal(only_goal, stage_b)),
        {"max_abs_diff": float((stage_b - only_goal).abs().max()),
         "stage_b_weights": list(STAGE_B_W)})
    # POSITIVE CONTROL: a NON-zero W_KAPPA must change the plan, so the
    # equality above is a fact about the WEIGHTS and not about the probe.
    wk = _plan(w=(0.0, 1e3, 0.0), target_speed=None).controls
    chk("CONTROL_nonzero_W_KAPPA_does_change_the_plan",
        float((wk - only_goal).abs().max()) > 0,
        {"max_abs_diff": float((wk - only_goal).abs().max())})
    model.lat_head = real_lat_head

    R["PASS"] = not fails
    R["failed_checks"] = fails
    R["plan_cfg"] = {"n_samples": a.n_samples, "n_iters": a.n_iters,
                     "horizon": pc.horizon, "dt": pc.dt}
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1)
    for k, v in R["checks"].items():
        print(("PASS " if v["ok"] else "FAIL ") + k,
              {kk: vv for kk, vv in v.items() if kk != "ok"})
    print("\nper-token plan curvature:")
    for t, v in per_token.items():
        print(f"  {t:<15} mean_kappa={v['sustained_kappa_mean']:+.6f} "
              f"absmax={v['abs_kappa_max']:.6f}  source={v['source']}")
    print("\nOVERALL:", "PASS" if R["PASS"] else f"FAIL {fails}")
    return 0 if R["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
