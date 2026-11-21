"""PART 5 -- single-arm episode-cluster CIs on the two contrasts the verdict
rests on. These groups are DISJOINT WINDOW SETS, so a PAIRED estimator does not
apply and is not used; each mean carries its own episode-cluster interval and
the reader compares overlap, which is the weaker but VALID statement."""
import json, os, numpy as np, sys
sys.path.insert(0, r"C:\Users\Admin\run_refcv4v\repo\taniteval")
from taniteval.ci import episode_cluster_bootstrap as ECB
O = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad\work\turncov_out"
d = np.load(os.path.join(O, "per_window.npz"))
eid = d["eid"]; hole = np.degrees(d["sup_eb"]) <= 30.0
out = {}
for tag, m in (("end_bearing_hole_windows", hole),
               ("all_other_windows", ~hole),
               ("gt_turn_ge30deg", np.degrees(d["dem_thx"]) >= 30.0),
               ("gt_turn_lt5deg", np.degrees(d["dem_thx"]) < 5.0)):
    row = {"n_windows": int(m.sum()), "n_episodes": int(len(np.unique(eid[m])))}
    for met in ("ade_oiv", "along_oiv", "lat_oiv"):
        r = ECB(d[met][m], eid[m], n_boot=2000, seed=0)
        row[met] = {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"],
                    "estimator": r["estimator"]}
    out[tag] = row
    print(f"{tag:<28} n {row['n_windows']:>5}/{row['n_episodes']:>3}ep  "
          f"ADE {row['ade_oiv']['mean']:.4f} [{row['ade_oiv']['lo']:.4f}, "
          f"{row['ade_oiv']['hi']:.4f}]  ALONG {row['along_oiv']['mean']:.4f} "
          f"[{row['along_oiv']['lo']:.4f}, {row['along_oiv']['hi']:.4f}]  "
          f"LAT {row['lat_oiv']['mean']:.4f} [{row['lat_oiv']['lo']:.4f}, "
          f"{row['lat_oiv']['hi']:.4f}]")
json.dump(out, open(os.path.join(O, "TURNCOV6S_PART5.json"), "w"), indent=1)
print("wrote TURNCOV6S_PART5.json")
