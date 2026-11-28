"""Evidence for BACKLOG R26/R27 — three probes, all cheap, none touching a pod.

  P1  the tiny-model cost numbers behind `test_A5` (units: consistent vs the
      shipped half-conversion), so RESULT.md quotes a run and not a docstring;
  P2  the index maps (`cost_time_grid`), derived from the BANKED checkpoint's
      own `config.json` rather than from the source defaults;
  P3  the model-free read of the banked decision dumps that decides whether the
      corrected horizon can move the banked winner at all.

Run (dev box, CPU only):
  PYTHONPATH=<wt>/stack;<wt>/colab;<wt>/taniteval  python raw/evidence.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v
from tanitad.refs.refa_v1 import RefAV1, RefAV1Config
from tanitad.refs.refa_v1_plan import PlanConfig

DUMP = os.environ.get("REFAV1_T1_DUMP",
                      r"C:\Users\Admin\refav1_eval_slice\t1_dump")
CKPT_CFG = os.environ.get(
    "REFAV1_CKPT_CFG", r"C:\Users\Admin\refav1_eval_slice\ckpt_ep2\config.json")
ULP = 5.9604645e-08


class _ConstHead(nn.Module):
    def __init__(self, n, idx):
        super().__init__()
        self.n, self.idx = n, idx

    def forward(self, x):
        out = torch.full((x.shape[0], self.n), -10.0, dtype=x.dtype)
        out[:, self.idx] = 10.0
        return out


def p1_tiny_model_costs():
    torch.manual_seed(0)
    cfg = RefAV1Config(
        tac_vocab_version="v7.0", d_enc=32, n_tokens=8, d_state=32,
        op_layers=1, op_heads=2, op_window=4, tac_layers=1, tac_queries=4,
        str_dim=16, str_layers=1,
        strategic_cfg=StrategicPolicyConfig(d_model=32, depth=1, n_heads=2,
                                            d_ctx=16, d_cmd=8),
        tactical_cfg=TacticalPolicyConfig(d_model=32, depth=1, n_heads=2,
                                          d_intent=16))
    m = RefAV1(cfg).eval()
    lat_v, lon_v = (tactical_lat_actions("v7.0"), tactical_lon_actions_v("v7.0"))
    m.lat_head = _ConstHead(len(lat_v), lat_v.index("TURN_R"))
    m.lon_head = _ConstHead(len(lon_v), lon_v.index("CRUISE"))
    feats = torch.randn(1, cfg.op_window, cfg.n_tokens, cfg.d_enc)

    field = m.encode(feats)
    last = m._last_state(field)
    brains = m._run_brains(field.mean(dim=-2), None)
    v0_t = torch.tensor([8.0])
    stride, z0 = m._stride(cfg.tac_dt), m._tac_field(last)

    def roll(ctrl, units):
        acts = m._model_actions(ctrl, v0_t, units)[:, ::stride][:, :cfg.tac_steps]
        return m.tactical.rollout(z0, acts, intent=brains["intent"],
                                  last_only=True)

    def cg(zk, goal):
        return float(1.0 - F.cosine_similarity(zk.flatten(1), goal.flatten(1),
                                               dim=-1))

    with torch.no_grad():
        goal_k, ga = m._imagine_tactical_goal(last, brains, v0_t, units="kappa")
        goal_s, _ = m._imagine_tactical_goal(last, brains, v0_t, units="steer")
        turn, straight = ga["controls"], torch.zeros_like(ga["controls"])
        rows = {
            "OFF_kappa_consistent": (roll(turn, "kappa"), roll(straight, "kappa"),
                                     goal_k),
            "ON_steer_SHIPPED_half": (roll(turn, "steer"), roll(straight, "steer"),
                                      goal_k),
            "ON_steer_FIXED_both": (roll(turn, "steer"), roll(straight, "steer"),
                                    goal_s),
        }
        out = {}
        for name, (zt, zs, g) in rows.items():
            c_turn, c_str = cg(zt, g), cg(zs, g)
            out[name] = {"c_goal_turn": c_turn, "c_goal_straight": c_str,
                         "advantage_turn_over_straight": c_str - c_turn,
                         "c_goal_turn_ULPs": c_turn / ULP}
    out["_note"] = ("tiny RANDOM model, seed 0 — a MECHANISM demonstration, NOT "
                    "a checkpoint measurement. `D-REFAV1-BOUNDARY-NULL` measured "
                    "the real effect on both banked checkpoints at exactly 0.")
    out["_goal_max_abs_kappa"] = float(ga["controls"][..., 1].abs().max())
    return out


def p2_index_maps():
    cfg_banked = json.load(open(CKPT_CFG))["cfg"]
    op_dt, tac_dt = cfg_banked["op_dt"], cfg_banked["tac_dt"]
    op_steps, tac_steps = cfg_banked["op_steps"], cfg_banked["tac_steps"]
    h = int(round(cfg_banked["plan_horizon_s"] / op_dt))
    s = max(1, int(round(tac_dt / op_dt)))
    return {
        "source": CKPT_CFG,
        "banked_cfg": {k: cfg_banked[k] for k in
                       ("op_dt", "op_steps", "tac_dt", "tac_steps", "a_dim",
                        "speed_channel", "plan_horizon_s", "plan_level",
                        "verify_on_operative")},
        "stride": s, "H": h,
        "candidate_dense_TODAY": list(range(h)),
        "candidate_tactical_FIXED": [min(j * s, h - 1) for j in range(tac_steps)],
        "goal_map": list(range(0, op_steps, s))[:tac_steps],
        "imagined_horizon_s_dense": h * tac_dt,
        "imagined_horizon_s_fixed": tac_steps * tac_dt,
        "plan_horizon_s": h * op_dt,
    }


def p3_banked_dumps():
    """Model-free: can the corrected horizon move the BANKED winner?

    With `speed_channel=false` (banked cfg) `_model_actions` is the identity on
    the controls, so `cost_time_grid="tactical"` is an EXACT no-op for any
    candidate that is CONSTANT over the plan horizon — the two index maps then
    select the same values. A zero-control winner (`cv` / `hold_v0`) is such a
    candidate."""
    files = sorted(glob.glob(os.path.join(DUMP, "decisions", "ep*.npz")))
    if not files:
        return {"error": f"no decision dumps under {DUMP}"}
    arms = ["cl", "cl_navshuf", "cl_oraclegoal"]
    acc = {a: {"n": 0, "zero": 0, "const": 0} for a in arms}
    costs = {a: [] for a in arms}
    for p in files:
        d = np.load(p)
        for a in arms:
            k = f"{a}_controls"
            if k not in d.files:
                continue
            c = d[k]                                  # [w, H, 2]
            acc[a]["n"] += c.shape[0]
            acc[a]["zero"] += int((np.abs(c).max(axis=(1, 2)) == 0.0).sum())
            same = (c == c[:, :1]).all(axis=(1, 2))
            acc[a]["const"] += int(same.sum())
            if f"plan_cost_{a}" in d.files:
                costs[a].extend(d[f"plan_cost_{a}"].tolist())
    out = {"n_episodes": len(files), "arms": {}}
    for a in arms:
        n = acc[a]["n"] or 1
        out["arms"][a] = {
            "n_windows": acc[a]["n"],
            "winner_all_zero": acc[a]["zero"],
            "winner_constant_over_H": acc[a]["const"],
            "frac_constant": acc[a]["const"] / n,
            "winning_cost_exactly_zero":
                int(sum(1 for x in costs[a] if x == 0.0)),
            "winning_cost_min": min(costs[a]) if costs[a] else None,
            "winning_cost_max": max(costs[a]) if costs[a] else None,
        }
    out["_derivation"] = (
        "speed_channel=false in the banked cfg => the regrid is the identity on "
        "a constant candidate; every constant winner's own cost is therefore "
        "bit-identical under cost_time_grid='tactical'. Whether it remains the "
        "ARGMIN needs the model and is UNVERIFIED here.")
    return out


def main():
    res = {"P1_tiny_model_costs": p1_tiny_model_costs(),
           "P2_index_maps": p2_index_maps(),
           "P3_banked_dumps": p3_banked_dumps()}
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "evidence.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    json.dump(res, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
