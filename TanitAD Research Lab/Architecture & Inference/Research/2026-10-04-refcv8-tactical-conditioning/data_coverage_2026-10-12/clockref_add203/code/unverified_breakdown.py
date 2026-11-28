"""Which clips are unverified under references [ALLTRAIN, ADD203], and why. sha12-keyed (no raw UUIDs)."""
import hashlib, json, sys
sys.path.insert(0, "C:/Users/Admin/tanitad-data/add203/repo/stack")
from tanitad.data.v2_dataset import stable_episode_id
sha12 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
ADD = [l.strip() for l in open("raw/ADD_203_clip_ids.txt") if l.strip()]
OLD = [l.strip() for l in open("C:/Users/Admin/qland/landed_hold_datacov/raw/refcv6_train_clips.txt") if l.strip()]
assert len(ADD) == 203 and len(OLD) == 4369
sid2c = {stable_episode_id(c): c for c in ADD + OLD}
un = json.load(open("raw/clock_unverified_sids_from_add203_hold.json"))
un_old = [sid2c[s] for s in un["unclocked_old_train_sids"]]
un_add = [sid2c[s] for s in un["unclocked_add_sids"]]
ref = {r["clip"]: r for r in json.load(open("raw/q4c_grid_vs_egolog_ALLTRAIN.json"))["splits"]["train"]["per_clip"]}
pr = {r["clip"]: r for r in json.load(open("raw/probe_stationary.json"))}
rows = []
for grp, lst in (("old_fallback_clock", un_old), ("add_ts33_measured_unreferenced", un_add)):
    for c in lst:
        k = sha12(c); p = pr.get(k)
        why = ("in ALLTRAIN reference with resid %.3g ms but ABSENT from the clip-clock sidecar" % ref[k]["fit_resid_max_ms"]) if (k in ref and ref[k]["fit_resid_max_ms"] < 5) else (p["why_unfit"] if p else "unknown")
        rows.append({"clip": k, "group": grp, "why": why, "cache_path_m": (p or {}).get("cache_path_m"), "cache_v_max": (p or {}).get("cache_v_max"),
                     "n_rows_v_gt_0p5": (p or {}).get("n_gt_0p5"), "n_rows_v_gt_2": (p or {}).get("n_gt_2")})
out = {"n_old_fallback": len(un_old), "n_add_unreferenced": len(un_add), "n_total": len(rows),
       "cap_clips": 0.01 * 4572, "max_unverified_clips_under_cap": int(0.01 * 4572), "rows": rows}
json.dump(out, open("raw/unverified_55_breakdown.json", "w"), indent=1)
import collections
print(len(rows), collections.Counter((r["group"], r["why"][:40]) for r in rows))

# ---- state D: references = [ALLTRAIN, ADD203, ADD203_slowregime] -> 42 unverified
slow = {r["clip"] for r in json.load(open("raw/q4c_grid_vs_egolog_ADD203_slowregime.json"))["splits"]["train"]["per_clip"]}
rowsD = []
for r in rows:
    if r["group"] == "add_ts33_measured_unreferenced" and r["clip"] in slow:
        continue
    r2 = dict(r)
    if r["group"] == "add_ts33_measured_unreferenced":
        r2["why"] = ("static: cache path %.2f m, no kinematic content" % r["cache_path_m"]) if (r["cache_path_m"] is not None and r["cache_path_m"] < 0.1) \
            else ("below the instrument's 20-row floor: only %d rows above 0.5 m/s" % r["n_rows_v_gt_0p5"])
    rowsD.append(r2)
outD = {"n_total": len(rowsD), "n_old_fallback": sum(r["group"] == "old_fallback_clock" for r in rowsD),
        "n_add_unreferenced": sum(r["group"] != "old_fallback_clock" for r in rowsD),
        "max_unverified_clips_under_cap": int(0.01 * 4572), "margin_clips": int(0.01 * 4572) - len(rowsD), "rows": rowsD}
json.dump(outD, open("raw/unverified_42_breakdown.json", "w"), indent=1)
print("D:", outD["n_total"], outD["n_old_fallback"], outD["n_add_unreferenced"], "margin", outD["margin_clips"])
import collections
print(collections.Counter(r["why"].split(":")[0][:50] for r in rowsD if r["group"] != "old_fallback_clock"))
