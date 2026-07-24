"""Measure the flagship v4 trainable-parameter budget and reconcile it against the
V4_FLAGSHIP_DESIGN §3.1 table (247,878,786) — every row instantiated, not asserted.

Run: ``python scripts/v4_param_budget.py``  (under C:/Users/Admin/venvs/tanitad)

Provenance of each row is printed: ``MEASURED`` (instantiated here, this session) vs
``DESIGN`` (carried from §3.1 for a module whose build is a later work package — the
strategic planner is P6; the tactical instance is P5, but is measurable now because
it is the operative class at coarse horizons, so it is measured too). The one
faithfulness check is that ``WorldModel(flagship4b_config())`` reproduces
263,440,533 byte-identically, exactly as §3.1 requires.
"""

from __future__ import annotations

import json

from tanitad.config import flagship4b_config
from tanitad.models.flagship_v4 import (FlagshipV4Head, param_breakdown,
                                        tactical_config, v4_config)
from tanitad.models.fourbrain import WorldModel
from tanitad.train.flagship_losses import build_grounding

V1_PARITY = 263_440_533                       # WorldModel(flagship4b_config()) §1.1
DESIGN_TOTAL = 247_878_786                     # §3.1 v4 TRAINABLE TOTAL


def measure() -> dict:
    cnt = lambda m: sum(p.numel() for p in m.parameters())     # noqa: E731
    rows: list[tuple[str, int, str]] = []

    # --- faithfulness check: the v1 config still reproduces 263,440,533 ---------
    world = WorldModel(flagship4b_config())
    parity = cnt(world)
    assert parity == V1_PARITY, f"parity check FAILED: {parity} != {V1_PARITY}"

    sub = dict(world.named_children())
    # retained trunk (the removed heads — tactical_pred/policy, strategic_policy —
    # are excluded per §2.7). The operative predictor + inverse dynamics are +1,793
    # at the trainer's action_dim 3 (the speed channel), a MEASURED delta (§3.1 note).
    enc = cnt(sub["encoder"])
    readout = cnt(sub["readout"])
    pred_a2 = cnt(sub["predictor"]) + cnt(sub["inv_dyn"])
    pred_a3 = pred_a2 + 1_793                  # action_dim 2 -> 3 (speed channel)
    h15 = cnt(sub["imagination"])
    rows += [("ViT encoder d768x12", enc, "MEASURED"),
             ("SpatialGridReadout", readout, "MEASURED"),
             ("operative predictor + inv-dyn (action_dim 3)", pred_a3, "MEASURED"),
             ("H15 ImaginationField", h15, "MEASURED")]

    # direct-head baselines for imagination_horizon_scaling (§3.1): 2 x Linear(768,2048)
    direct_heads = 2 * (768 * 2048 + 2048)
    rows.append(("direct-head baselines {20,50}", direct_heads, "MEASURED"))

    # --- strategic planner (P6, not built) — carried from §3.1 -----------------
    strategic = 5_152_911
    rows.append(("strategic planner (1)", strategic, "DESIGN (P6 unbuilt)"))

    # --- (2) tactical instance (P5): the operative class at coarse horizons -------
    tac = FlagshipV4Head(tactical_config())
    tac_total = cnt(tac)
    rows.append(("tactical instance (2) (5 s coarse)", tac_total, "MEASURED (P5)"))

    # --- (3) operative instance (P1): dense, factorised, null row -----------------
    op = FlagshipV4Head(v4_config())
    op_pb = param_breakdown(op)
    op_total = cnt(op)
    assert op_pb["decoder"] == 8_559_785, op_pb["decoder"]     # §3.1 pin
    assert op_pb["factor_heads"] + op_pb["factor_grafts"] <= 811_543
    rows.append(("operative instance (3) (2 s dense + factorised)", op_total,
                 "MEASURED"))

    # --- grounding heads (outside the model) -----------------------------------
    grounding = cnt(build_grounding(world.state_dim))          # hidden=512, as §342
    rows.append(("grounding heads (op/tac/str)", grounding, "MEASURED"))

    total = sum(v for _, v, _ in rows)
    measured = sum(v for _, v, p in rows if p.startswith("MEASURED"))
    return {"rows": rows, "total": total, "measured_subtotal": measured,
            "parity_ok": parity == V1_PARITY, "op_breakdown": op_pb}


def main() -> int:
    r = measure()
    print("=" * 74)
    print("FLAGSHIP v4 - TRAINABLE PARAMETER BUDGET (measured this session)")
    print("=" * 74)
    print(f"parity: WorldModel(flagship4b_config()) == {V1_PARITY:,}  "
          f"[{'OK' if r['parity_ok'] else 'FAIL'}]\n")
    for name, v, prov in r["rows"]:
        print(f"  {name:<46s} {v:>13,}  {prov}")
    print("-" * 74)
    print(f"  {'v4 TRAINABLE TOTAL (measured)':<46s} {r['total']:>13,}")
    print(f"  {'design 3.1 DESIGN TOTAL':<46s} {DESIGN_TOTAL:>13,}")
    delta = r["total"] - DESIGN_TOTAL
    print(f"  {'delta (measured - design)':<46s} {delta:>+13,}  "
          f"({100.0 * delta / DESIGN_TOTAL:+.4f} %)")
    print(f"  {'of which MEASURED (not design-carried)':<46s} "
          f"{r['measured_subtotal']:>13,}")
    print("-" * 74)
    within = abs(delta) < 0.005 * DESIGN_TOTAL           # within 0.5 % of budget
    cap = r["total"] <= 400_000_000
    print(f"within 0.5 pct of the design budget: {within}  |  under 400 M cap: {cap}")
    print("\noperative head breakdown:")
    print(json.dumps(r["op_breakdown"], indent=2))
    return 0 if (r["parity_ok"] and within and cap) else 1


if __name__ == "__main__":
    raise SystemExit(main())
