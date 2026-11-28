#!/usr/bin/env python
"""⛔ THE SEED IS NOT THE GOAL — an EXACT, 0-GPU, arithmetic check.

`plan()` builds the imagined goal by rolling the tactical predictor under the
FULL 30-step canonical control, subsampled onto the tactical grid:

    refa_v1.py:1761   acts = augment_actions(ctrl, v0)[:, ::stride][:, :tac_steps]
                      # stride = tac_dt/op_dt = 3  ->  op indices [0,3,...,27]

and then seeds the SAME manoeuvre into the search TRUNCATED to the plan window:

    refa_v1.py:1932   seed = goal_action["controls"][:cfg.plan_steps]   # 10 of 30

which the cost then re-grids according to `plan(cost_time_grid=...)`:

    "dense"     (⭐ THE DEFAULT, and the behaviour every BANKED number was
                 produced under — the parameter did not exist before 4139203)
                 tactical step j consumes operative action j  -> [0..9]
    "tactical"  (the flag-gated repair landed in 4139203)
                 tactical step j consumes min(j*stride, H-1)  -> [0,3,6,9,9,...]

⇒ under EITHER grid the GOAL's tactical action sequence and the SEED's tactical
action sequence are DIFFERENT sequences for most manoeuvres — but they are
different in DIFFERENT WAYS, so this tool computes BOTH and never hard-codes one.
⚠️ It reads `COST_TIME_GRIDS` and `plan()`'s own default out of the shipped
source by `inspect`, because an earlier revision of this tool hard-coded the
non-default grid and reported its numbers as if they were the banked ones.

It prints, element by element, for every (lat, lon) token pair in the v7.0
vocabulary, the integrated heading each feed implies. It needs no model:
`canonical_controls` is deterministic and future-free.

If the two disagree, the canonical seed cannot reach `1 - cos = 0` against its
own goal even with a perfect world model — which is exactly what the banked cost
rows show (`turn_decomposition.py`: the seed's goal term beats cv on only 8/25
windows under convention A and 4/25 under B).

TIER: T0. EVIDENCE CLASS: MEASURED (arithmetic on the shipped source).
"""
from __future__ import annotations

import argparse
import json
import math


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import inspect

    import torch                                             # noqa: F401
    from tanitad.models.v6 import (tactical_lat_actions,
                                   tactical_lon_actions_v)
    from tanitad.refs import refa_v1 as R
    from tanitad.refs.refa_v1 import RefAV1Config, canonical_controls

    cfg = RefAV1Config()
    op_steps, op_dt = int(cfg.op_steps), float(cfg.op_dt)
    stride = int(round(cfg.tac_dt / cfg.op_dt))
    tac_steps = int(cfg.tac_steps)
    plan_steps = int(round(cfg.plan_horizon_s / cfg.op_dt))

    # ⭐ READ THE GRID SET AND THE DEFAULT OUT OF THE SOURCE — never hard-code.
    grids = tuple(getattr(R, "COST_TIME_GRIDS", ("dense",)))
    sig = inspect.signature(R.RefAV1.plan)
    p = sig.parameters.get("cost_time_grid")
    default_grid = (p.default if p is not None
                    and p.default is not inspect.Parameter.empty else "dense")

    goal_idx = list(range(0, op_steps, stride))[:tac_steps]

    def seed_indices(grid: str) -> list[int]:
        """The operative action each TACTICAL step of a candidate consumes."""
        if grid == "tactical":                    # refa_v1.py `tac_idx`
            return [min(j * stride, plan_steps - 1) for j in range(tac_steps)]
        if grid == "dense":                       # no regrid: step j <- action j
            return [min(j, plan_steps - 1) for j in range(tac_steps)]
        raise SystemExit(f"unhandled cost_time_grid {grid!r} — this tool must be "
                         f"extended before it can report on it")

    res = {
        "tool": "seed_goal_mismatch.py", "tier": "T0",
        "evidence_class": "MEASURED (arithmetic on shipped source)",
        "config": {"op_steps": op_steps, "op_dt": op_dt, "tac_dt": cfg.tac_dt,
                   "tac_steps": tac_steps, "stride": stride,
                   "plan_horizon_s": cfg.plan_horizon_s,
                   "plan_steps": plan_steps},
        "cost_time_grids_available": list(grids),
        "cost_time_grid_DEFAULT": default_grid,
        "default_note": ("'dense' is the DEFAULT and is the behaviour every "
                         "BANKED refav1 number was produced under — the "
                         "parameter itself did not exist before commit 4139203"),
        "goal_op_indices": goal_idx,
        "seed_op_indices_by_grid": {g: seed_indices(g) for g in grids},
        "source": {"goal": "refa_v1.py `_imagine_tactical_goal` "
                           "(acts[:, ::stride][:, :tac_steps])",
                   "seed": "refa_v1.py:1932 (controls[:cfg.plan_steps])",
                   "regrid": "refa_v1.py `tac_idx`, gated on cost_time_grid"},
        "v0_used_m_s": a.v0,
        "by_grid": {},
        "tokens": [],
    }

    lat_v = tactical_lat_actions("v7.0")
    lon_v = tactical_lon_actions_v("v7.0")
    for grid in grids:
        s_idx = seed_indices(grid)
        n_mismatch = 0
        n_kappa_mismatch = 0
        worst = None
        for lat in lat_v:
            for lon in lon_v:
                ctrl = canonical_controls(lat, lon, a.v0, op_steps, op_dt)
                k, acc = ctrl[:, 1].tolist(), ctrl[:, 0].tolist()
                k_goal = [k[i] for i in goal_idx]
                k_seed = [k[i] for i in s_idx]
                a_goal = [acc[i] for i in goal_idx]
                a_seed = [acc[i] for i in s_idx]
                # heading each feed implies on the tactical grid (unicycle, v
                # held at v0 — the DIFFERENCE is the quantity, not the value)
                psi_goal = sum(kk * a.v0 * cfg.tac_dt for kk in k_goal)
                psi_seed = sum(kk * a.v0 * cfg.tac_dt for kk in k_seed)
                mism = (k_goal != k_seed) or (a_goal != a_seed)
                n_mismatch += int(mism)
                n_kappa_mismatch += int(k_goal != k_seed)
                exc = math.degrees(psi_seed - psi_goal)
                if worst is None or abs(exc) > abs(worst[2]):
                    worst = (lat, lon, exc)
                res["tokens"].append({
                    "grid": grid, "lat": lat, "lon": lon, "mismatch": bool(mism),
                    "kappa_goal_tactical": [round(x, 6) for x in k_goal],
                    "kappa_seed_tactical": [round(x, 6) for x in k_seed],
                    "n_tactical_steps_differing_kappa":
                        int(sum(1 for x, y in zip(k_goal, k_seed) if x != y)),
                    "n_tactical_steps_differing_accel":
                        int(sum(1 for x, y in zip(a_goal, a_seed) if x != y)),
                    "accel_goal_tactical": [round(x, 6) for x in a_goal],
                    "accel_seed_tactical": [round(x, 6) for x in a_seed],
                    "heading_goal_rad": round(psi_goal, 6),
                    "heading_seed_rad": round(psi_seed, 6),
                    "heading_excess_deg": round(exc, 3),
                })
        agree = [(t["lat"], t["lon"]) for t in res["tokens"]
                 if t["grid"] == grid and not t["mismatch"]]
        res["by_grid"][grid] = {
            "is_default": grid == default_grid,
            "seed_op_indices": s_idx,
            "n_token_pairs": len(lat_v) * len(lon_v),
            "n_token_pairs_with_mismatch": n_mismatch,
            "n_token_pairs_with_KAPPA_mismatch": n_kappa_mismatch,
            "agreeing_pairs": agree,
            "worst_heading_excess_deg": {"lat": worst[0], "lon": worst[1],
                                         "excess_deg": round(worst[2], 3)},
        }

    d = res["by_grid"][default_grid]
    res["headline"] = (
        f"Under the DEFAULT grid '{default_grid}' (the one every banked number "
        f"was produced under), {d['n_token_pairs_with_mismatch']}/"
        f"{d['n_token_pairs']} (lat, lon) token pairs give a DIFFERENT tactical "
        f"action sequence to the goal and to the seed that chases it; the "
        f"agreeing pairs are {d['agreeing_pairs']}. The canonical seed therefore "
        f"cannot score 1-cos = 0 against its own goal even with a perfect world "
        f"model. Worst heading excess: "
        f"{d['worst_heading_excess_deg']['excess_deg']} deg on "
        f"{d['worst_heading_excess_deg']['lat']}.")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in
                      ("config", "cost_time_grids_available",
                       "cost_time_grid_DEFAULT", "goal_op_indices",
                       "seed_op_indices_by_grid", "by_grid", "headline")},
                     indent=1))
    for t in res["tokens"]:
        if t["lon"] == "CRUISE" and t["lat"] in ("TURN_L", "NUDGE_R",
                                                 "LANE_CHANGE_R"):
            print(f"  [{t['grid']}] {t['lat']}x{t['lon']}: "
                  f"psi_goal {t['heading_goal_rad']} psi_seed "
                  f"{t['heading_seed_rad']} excess {t['heading_excess_deg']} deg")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
