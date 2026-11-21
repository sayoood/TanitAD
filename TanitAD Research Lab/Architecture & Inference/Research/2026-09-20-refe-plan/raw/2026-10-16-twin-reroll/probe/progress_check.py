#!/usr/bin/env python3
"""Side-effect check for the goal-horizon lever (addendum 5): does a shorter goal horizon make the teacher SLOWER?
Per twin (log, token, rank): 4 s path length and end speed of the re-rolled trajectory, arm vs the route-held 12 s arm (rr_held), paired.
argv: <reroll_probe dir> <arm dirs...>   (each arm dir holds s*/targets_reroll.jsonl)  -> JSON on stdout."""
import glob, json, math, os, statistics as st, sys
R = sys.argv[1]
def load(arm):
    d = {}
    for p in glob.glob(os.path.join(R, arm, "s*", "targets_reroll.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            d[(r["log_name"], r["token"], r["rank"])] = r
    return d
def plen(t):
    pts = [(0.0, 0.0)] + [(p[0], p[1]) for p in t]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
def vend(t, dt=0.2):
    return math.dist(t[-1][:2], t[-2][:2]) / dt
base = load("rr_held")
out = {"base": "rr_held", "n_base": len(base)}
for arm in sys.argv[2:]:
    A = load(arm)
    ks = sorted(set(A) & set(base))
    rl = [plen(A[k]["traj"]) / max(plen(base[k]["traj"]), 1e-6) for k in ks if plen(base[k]["traj"]) > 1.0]
    dv = [vend(A[k]["traj"]) - vend(base[k]["traj"]) for k in ks]
    out[arm] = {"n_paired": len(ks), "n_moving": len(rl),
                "path_ratio_median": round(st.median(rl), 4) if rl else None,
                "path_ratio_p10_p90": [round(sorted(rl)[int(0.1 * (len(rl) - 1))], 4), round(sorted(rl)[int(0.9 * (len(rl) - 1))], 4)] if rl else None,
                "share_path_ratio_lt_0.9": round(sum(x < 0.9 for x in rl) / len(rl), 3) if rl else None,
                "end_speed_delta_median_mps": round(st.median(dv), 3) if dv else None,
                "path_m_median": round(st.median(plen(A[k]["traj"]) for k in ks), 2) if ks else None,
                "path_m_median_base": round(st.median(plen(base[k]["traj"]) for k in ks), 2) if ks else None}
print(json.dumps(out, indent=1))
