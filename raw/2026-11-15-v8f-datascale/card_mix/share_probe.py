"""Per-term labelled share of T-loss-bearing windows on the B set (corpus join md5 9ebc7a48), read-only, CPU.
Window list rule = v8f_mini_data.train_windows (t in [3, T-31], HORIZON 30, start = hist-1 = 3); T's label row = row(ci, t) (a_t = 0).
Exclusion rule = v8f_mini_train.build_data --exclude 540:600 (sorted by sha12 of the TRAIN manifest)."""
import hashlib
import json
import sys

import numpy as np

JOIN, MAN, CLIPS = sys.argv[1], sys.argv[2], sys.argv[3]
GOAL22 = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "YIELD", "STOP_POINT",
          "SPEED_BAND", "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR", "OVERTAKE_VEHICLE", "MERGE", "GAP_TARGET",
          "REACT_ON_ONCOMING", "TAKE_EXIT_L", "TAKE_EXIT_R", "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED",
          "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN", "LANE_CHANGE_L", "LANE_CHANGE_R")
sha12 = lambda c: hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]

z = np.load(JOIN, allow_pickle=True)
J = {k: z[k] for k in z.files}
man = json.load(open(MAN, encoding="utf-8"))
prov = man["vocab"]["GOAL_PROV"]
vlm = 0
for i, g in enumerate(GOAL22):
    if "vlm" in str(prov.get(g, "")).split("|"):
        vlm |= 1 << i
geo = ((1 << 22) - 1) & ~vlm
clips_all = [l.strip() for l in open(CLIPS, encoding="utf-8") if l.strip()]
excl = set(sorted(clips_all, key=sha12)[540:600])
bset = [c for c in clips_all if c not in excl]
train_sha = hashlib.sha256("\n".join(sorted(bset, key=sha12)).encode("utf-8")).hexdigest()
ci_of = {str(c): i for i, c in enumerate(J["clip"].tolist())}
rows = []
for c in bset:
    ci = ci_of[c]
    T = int(J["n"][ci])
    o = int(J["off"][ci])
    rows.extend(range(o + 3, o + max(3, T - 30)))
rows = np.asarray(rows, np.int64)
has = J["has"][rows]
gw = J["goal_w"][rows].astype(np.uint64)
pc = lambda m: (int(m.sum()), round(float(m.mean()), 4))
bits = lambda x: np.array([bin(int(v)).count("1") for v in x], np.int64)
n_vlm_el = bits(gw & np.uint64(vlm)) * has
n_geo_el = bits(gw & np.uint64(geo)) * has
cons = J["cons"][rows]
out = {
    "join": JOIN, "n_bset_clips": len(bset), "train_sha256": train_sha, "n_windows": int(len(rows)),
    "vlm_bits": [g for i, g in enumerate(GOAL22) if (vlm >> i) & 1], "geo_bits": [g for i, g in enumerate(GOAL22) if (geo >> i) & 1],
    "has": pc(has),
    "any_vlm_goal_w": pc(((gw & np.uint64(vlm)) != 0) & has),
    "any_geo_goal_w": pc(((gw & np.uint64(geo)) != 0) & has),
    "lat_valid": pc((J["lat_allowed"][rows] != 0) & has),
    "lon_valid": pc((J["lon_allowed"][rows] != 0) & has),
    "any_cons_finite": pc(np.isfinite(cons).any(1) & has),
    "goal_elements_per_window": {"vlm": round(float(n_vlm_el.mean()), 4), "geo": round(float(n_geo_el.mean()), 4)},
    "vlm_share_of_goal_denominator_B": round(float(n_vlm_el.sum() / max(1, (n_vlm_el + n_geo_el).sum())), 4),
}
print(json.dumps(out, indent=1))
