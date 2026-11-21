#!/usr/bin/env python3
"""REVIEW 7 -- where are the dirty fans? (fan NAVSIM-safe share < 0.5) by off-route goal, turn proxy, ego speed.
CPU, navsim venv python. Inputs: lane census (GT), proposal table (pick), goal census (route distance, goal), hooks (v0)."""
import json
import numpy as np
import gap_selection as GS

OUT = GS.PKG + "/raw/2026-10-04-gap-review/gap_dirty_fans.json"
GC = GS.PKG + "/raw/2026-10-01-goal-trigger/goal_census_navtest_full.json"
HOOKS = GS.ROOT + "/data/refe_navtest/score/refe_navtest_final/refe_navtest_final_hooks.json"
z = np.load(GS.PROPS, allow_pickle=True)
pos = {t: i for i, t in enumerate(z["token"])}
pick = z["pick"]; N = len(pick)
share = np.zeros(N); ps = np.zeros(N)
for line in open(GS.CENSUS):
    r = json.loads(line); i = pos[r["token"]]
    nc = np.asarray(r["nc"][1:65]); dac = np.asarray(r["dac"][1:65]); pd = np.asarray(r["pdms"][1:65])
    share[i] = np.mean(nc * dac == 1.0); ps[i] = pd[pick[i]]
gc = json.load(open(GC))["rows"]
d_route = np.zeros(N); gy = np.zeros(N); gx = np.zeros(N)
for t, row in gc.items():
    i = pos[t]; d_route[i] = row["ego_to_route_m"] if row["ego_to_route_m"] is not None else 999.0
    g = row["goal"] or [0, 0, 0, 0]; gx[i] = g[2]; gy[i] = g[3]
v0 = np.zeros(N)
for h in json.load(open(HOOKS))["pdm_score_calls"]:
    v0[pos[h["token"]]] = h["v0_mps"]
bearing = np.degrees(np.abs(np.arctan2(gy, np.maximum(gx, 1e-3))))
dirty = share < 0.5
def tab(name, groups):
    res = {}
    for lab, m in groups:
        if m.sum() == 0:
            continue
        res[lab] = {"tokens": int(m.sum()), "dirty_rate": float(dirty[m].mean()), "pick_PDMS_x100": float(ps[m].mean() * 100),
                    "zero_rate": float((ps[m] == 0).mean()), "fan_safe_share": float(share[m].mean())}
    return res
out = {"n": N, "dirty_rate_all": float(dirty.mean()),
       "by_route_distance": tab("route", [("on_route<=20m", d_route <= 20), ("off_route>20m", d_route > 20)]),
       "by_far_goal_bearing_deg": tab("bearing", [("<10", bearing < 10), ("10-30", (bearing >= 10) & (bearing < 30)),
                                                  ("30-60", (bearing >= 30) & (bearing < 60)), (">=60", bearing >= 60)]),
       "by_v0_mps": tab("v0", [("<1", v0 < 1), ("1-5", (v0 >= 1) & (v0 < 5)), ("5-10", (v0 >= 5) & (v0 < 10)), (">=10", v0 >= 10)]),
       "share_of_all_zero_picks_on_dirty_fans": float(((ps == 0) & dirty).sum() / max(1, (ps == 0).sum())),
       "cost_of_dirty_fans_vs_clean_level_points": float(dirty.mean() * (ps[share >= 0.95].mean() - ps[dirty].mean()) * 100)}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
