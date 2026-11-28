#!/usr/bin/env python3
"""Console summary of adam_sim.json (no new numbers)."""
import json
import sys
from collections import defaultdict

import numpy as np

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "adam_sim.json"))
print("controls:")
for rg, c in d["controls"].items():
    for k, v in c.items():
        print(" ", rg, k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
print()
for regime in ("P", "T"):
    for k in ("25", "50", "100", "250"):
        rows = [d["streams"][s]["modes"][regime]["at"][k] for s in d["streams"] if k in d["streams"][s]["modes"][regime]["at"]]
        if not rows:
            continue
        f = lambda key: np.array([r[key] for r in rows])  # noqa: E731
        print(regime, k, "S_C80 %.3f +-%.3f | S_all %.3f | cos_C80 %.3f | U %.3f | cosFull %.3f | e %.2f | nC80 %d" % (
            f("S_C80").mean(), f("S_C80").std(), f("S_all").mean(), f("cos_C80").mean(), f("U_full_norm_ratio").mean(), f("cos_full").mean(), f("elasticity_e").mean(), rows[0]["n_C80"]))
print()
s0 = d["streams"]["0"]
print("clip:", {rg: s0["modes"][rg]["clip"] for rg in ("P", "T")})
print("r total:", s0["r"]["total"])


def grp(n):
    if n.startswith("score_head."):
        return "score_head"
    if n.startswith("score_q_mlp."):
        return "score_q_mlp"
    if ".n1." in n or ".n2." in n or ".n3." in n:
        return "dec.layernorm"
    if ".self_attn." in n:
        return "dec.self_attn"
    if ".cross." in n:
        return "dec.cross_attn"
    if ".mlp." in n:
        return "dec.mlp"


g = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
for n, v in s0["r"]["per_tensor"].items():
    G = grp(n)
    g[G][0] += v["rms_norm_gB"] ** 2
    g[G][1] += v["rms_norm_gE"] ** 2
    g[G][2] += v["coord_share_EPO_dominates_lam1"] * v["numel"]
    g[G][3] += v["numel"]
for G, v in g.items():
    print(G, "r_group lam1 %.2f lam025 %.2f coord_dom_lam1 %.2f" % (np.sqrt(v[1] / v[0]), 0.25 * np.sqrt(v[1] / v[0]), v[2] / v[3]))
at = s0["modes"]["P"]["at"]["50"]
print("groups S_P50 stream0:", {k: (round(v["share_of_EPO_energy_at_lam1"], 3), round(v["S_group"], 3)) for k, v in at["groups"].items()})
