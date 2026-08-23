import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))


def cb(d, k):
    return d["cluster_bootstrap"]["model"][k]


rows = []
print("metric            40 ep (881 win)                     600 ep (13198 win)                 hw ratio")
for k in ("ade_0_2s", "fde@2s", "miss_rate@2m", "tms_openloop", "rmse",
          "ade@0.5s", "ade@1s", "ade@1.5s"):
    x, y = cb(a, k), cb(b, k)
    hx, hy = x["ci95"] / 2, y["ci95"] / 2
    rows.append({"metric": k,
                 "ep40": {"mean": x["mean"], "lo": x["lo"], "hi": x["hi"],
                          "half_width": round(hx, 4), "se": x["se"]},
                 "ep600": {"mean": y["mean"], "lo": y["lo"], "hi": y["hi"],
                           "half_width": round(hy, 4), "se": y["se"]},
                 "half_width_ratio": round(hx / hy, 3) if hy else None})
    print(f"{k:<16} {x['mean']:.4f} [{x['lo']:.4f},{x['hi']:.4f}] hw={hx:.4f}   "
          f"{y['mean']:.4f} [{y['lo']:.4f},{y['hi']:.4f}] hw={hy:.4f}   "
          f"x{hx / hy if hy else 0:.2f}")

print()
print("full_set ade_0_2s :", a["full_set"]["model"]["ade_0_2s"], "->",
      b["full_set"]["model"]["ade_0_2s"])
print("CV full_set       :", a["full_set"]["cv"]["ade_0_2s"], "->",
      b["full_set"]["cv"]["ade_0_2s"])
print("n_windows / eps   :", a["n_windows"], "/", a["n_episodes"], "->",
      b["n_windows"], "/", b["n_episodes"])
print("wall_s            :", a["wall_s"], "->", b["wall_s"])
print("beats_cv          :", a.get("beats_cv_ade_0_2s"), a.get("beats_cv_separated"),
      "->", b.get("beats_cv_ade_0_2s"), b.get("beats_cv_separated"))
print("val_parity        :", a["val_parity"]["episodes_loaded"], "->",
      b["val_parity"]["episodes_loaded"],
      "| uid_sha", a["val_parity"].get("episode_uid_sha256", "")[:16], "->",
      b["val_parity"].get("episode_uid_sha256", "")[:16])
print()
for tag, d in (("40 ", a), ("600", b)):
    dr = d.get("driving", {})
    if isinstance(dr, dict):
        keys = [k for k in dr if k in ("headline", "summary", "verdict",
                                       "win_lives", "tracks_speed")]
        print(tag, "driving keys:", sorted(dr.keys())[:18])
        for k in keys:
            print("    ", k, json.dumps(dr[k])[:400])

out = {"arm": "flagship-30k (v1 FINAL, step 29999)",
       "estimator": "episode_cluster_bootstrap B=2000, unit = val episode",
       "deployment_40": {"n_windows": a["n_windows"], "n_episodes": a["n_episodes"],
                         "wall_s": a["wall_s"]},
       "deployment_600": {"n_windows": b["n_windows"], "n_episodes": b["n_episodes"],
                          "wall_s": b["wall_s"]},
       "metrics": rows,
       "cv_floor_ade_0_2s": {"ep40": a["full_set"]["cv"]["ade_0_2s"],
                             "ep600": b["full_set"]["cv"]["ade_0_2s"]},
       "note": ("The 600-episode number is NOT a correction of the published "
                "40-episode 0.4271: it is a DIFFERENT deployment of the same "
                "corpus (a strict order-preserving superset). Both are valid; "
                "they must never be cross-quoted as one series.")}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print("\n->", sys.argv[3])
