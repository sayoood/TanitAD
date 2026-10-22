import csv, json, sys
label = sys.argv[1]
D = "E:/Projects/TanitAD/data/refe_navtest/score"
new = {r["token"]: r for r in csv.DictReader(open(f"{D}/{label}/{label}.csv", encoding="utf-8")) if r["token"] != "average"}
old = {r["token"]: r for r in csv.DictReader(open(f"{D}/refe_stage2_sft4_B_u400_navtest_full/refe_stage2_sft4_B_u400_navtest_full.csv", encoding="utf-8"))}
cols = ["valid", "score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance"]
bad = [(t, c, new[t][c], old[t][c]) for t in new for c in cols if new[t][c] != old[t][c]]
cnt = json.load(open(f"{D}/{label}/{label}.counts.json", encoding="utf-8"))
out = {"label": label, "n_tokens": len(new), "n_columns_compared": len(new) * len(cols), "mismatches": len(bad), "first": bad[:3],
       "status": cnt.get("status"), "harness_fix1": cnt.get("harness_fix1"), "csv_sha256": cnt.get("csv_sha256")}
print(json.dumps(out, indent=1))
json.dump(out, open(f"C:/Users/Admin/par18/p6gaps/raw/harness_ctl/{label}.compare.json", "w"), indent=1)
