"""D3 check 4b -- are the clips with NO lane-line cells in the SAM3 GT explained by a stratum (night / road class / rig / country)?
If absence were a property of the ROADS it should not depend on lighting; a night/rig dependence would be a labeller dropout (a bias in the GT)."""
import gzip, json, csv, hashlib, sys, collections, numpy as np
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
SCR = sys.argv[1]
S = {}
for p in ("D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz", "D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz"):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l); S[sha(r["clip_id"])] = (r["strata"]["daynight_clock"], r["strata"]["road_class"], r["strata"]["country"], r["clip_id"])
cy = {}
for r in csv.DictReader(open(SCR + "/physicalai_front_wide_intrinsics.csv", encoding="utf-8")):
    cy[sha(r["clip_id"])] = float(r["cy"])
D = json.load(open("raw/c4_sam3_train_thor.json"))
rows = D["per_clip"]
out = {}
def tab(keyf, name):
    c = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        s = S.get(r["sha12"])
        if s is None: continue
        k = keyf(r, s); c[k][0] += 1; c[k][1] += int(r["lane_frames"] == 0); c[k][2] += int(r["lane_frames"] / r["T"] < 0.10); c[k][3] += r["lane_frames"] / r["T"]
    out[name] = {str(k): {"n": v[0], "pct_no_lane_line": round(100 * v[1] / v[0], 2), "pct_lt10pct_frames": round(100 * v[2] / v[0], 2), "mean_frac_frames_with_lane": round(v[3] / v[0], 3)} for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}
tab(lambda r, s: s[0], "by_daynight"); tab(lambda r, s: s[1], "by_road_class"); tab(lambda r, s: "rigB" if cy.get(r["sha12"], 0) >= 650 else "rigA", "by_rig")
tab(lambda r, s: (s[0], "rigB" if cy.get(r["sha12"], 0) >= 650 else "rigA"), "by_daynight_x_rig")
# countries: top/bottom 5 by no-lane rate (n>=60)
c = collections.defaultdict(lambda: [0, 0])
for r in rows:
    s = S.get(r["sha12"])
    if s: c[s[2]][0] += 1; c[s[2]][1] += int(r["lane_frames"] == 0)
cc = sorted([(k, v[0], round(100 * v[1] / v[0], 1)) for k, v in c.items() if v[0] >= 60], key=lambda x: -x[2])
out["by_country_no_lane_pct_top5_bottom5"] = {"top5": cc[:5], "bottom5": cc[-5:]}
# the worst-registered train clips (ego path roadlike fraction)
worst = sorted([r for r in rows if r["pts"] > 50], key=lambda r: r["road"])[:12]
out["worst_path_clips"] = [{"sha12": r["sha12"], "roadlike": r["road"], "pts": r["pts"], "strata": S.get(r["sha12"], ("?", "?", "?"))[:3], "rig": "B" if cy.get(r["sha12"], 0) >= 650 else "A", "corr_seen": r["corr_seen"]} for r in worst]
json.dump(out, open("raw/c4b_thin_by_stratum.json", "w"), indent=1)
print(json.dumps(out, indent=1))
