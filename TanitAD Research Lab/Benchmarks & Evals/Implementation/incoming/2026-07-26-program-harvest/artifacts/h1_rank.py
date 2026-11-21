"""H1 ranking pass 2: dedupe, split by estimator validity, and surface the
decision-relevant nulls (headline verdicts), not incidental micro-strata.
"""
import json, os, re, collections

SP = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(SP, "h1_raw.json"), encoding="utf-8"))
usable = d["usable_ranked"]
allr = d["all"]

# --- 1. estimator validity -------------------------------------------------
def est_class(r):
    e = (r.get("estimator") or "").lower()
    if "paired_episode_cluster" in e:
        return "PAIRED_EPCLUSTER"
    if "episode_cluster" in e:
        return "EPCLUSTER"
    if "jack" in e or "overlapping" in e or "holdout" in e:
        return "DEPRECATED_JACK"
    if e == "":
        return "UNNAMED"
    return e.upper()

for r in usable:
    r["est_class"] = est_class(r)
    # hierarchy.py nodes with no estimator == the _jack family (per retraction log)
    if r["est_class"] == "UNNAMED" and "hierarchy" in r["file"]:
        r["est_class"] = "DEPRECATED_JACK(inferred: hierarchy.py used _jack)"

cnt = collections.Counter(r["est_class"] for r in usable)
print("=== estimator classes among separated:false nodes ===")
for k, v in cnt.most_common():
    print(f"  {v:5d}  {k}")

# --- 2. dedupe: taniteval/results/X.json and driving_X.json are the same numbers
def canon(r):
    f = r["file"]
    jp = r["json_path"]
    # strip the "driving." prefix difference between the two dump shapes
    jp = re.sub(r"^driving\.", "", jp)
    base = os.path.basename(f)
    base = re.sub(r"^driving_", "", base)
    return (base, jp, round(r["effect"], 6), round(r["half_width"], 6))

seen = {}
for r in usable:
    k = canon(r)
    if k not in seen:
        seen[k] = r
        r["dup_files"] = [r["file"]]
    else:
        seen[k]["dup_files"].append(r["file"])
dedup = list(seen.values())
print(f"\nafter dedupe: {len(dedup)} (from {len(usable)})")

# --- 3. power tiers --------------------------------------------------------
def ntier(r):
    ne = r.get("n_episodes")
    nw = r.get("n_windows")
    if ne is None:
        return "n_unknown"
    if ne >= 500:
        return "n>=500 (already powered)"
    if ne >= 36:
        return "n~40 (THE 600-EP TARGET)"
    if ne >= 10:
        return "n~12-24 (closed-loop suites)"
    return "n<10 (micro-stratum)"

for r in dedup:
    r["ntier"] = ntier(r)

# --- 4. output buckets -----------------------------------------------------
valid = [r for r in dedup if r["est_class"].startswith("PAIRED_EPCLUSTER")
         or r["est_class"] == "EPCLUSTER"]
deprecated = [r for r in dedup if "DEPRECATED" in r["est_class"]]
unnamed = [r for r in dedup if r["est_class"] == "UNNAMED"]

print(f"\nvalid-estimator nulls: {len(valid)}   deprecated-estimator nulls: {len(deprecated)}"
      f"   unnamed-estimator nulls: {len(unnamed)}")

valid.sort(key=lambda r: -r["proximity"])
n40 = [r for r in valid if r["ntier"] == "n~40 (THE 600-EP TARGET)"]
n12 = [r for r in valid if r["ntier"] == "n~12-24 (closed-loop suites)"]

print(f"\n  of which n~40: {len(n40)}   n~12-24: {len(n12)}")

def show(rows, title, k=40):
    print(f"\n\n########## {title} ##########")
    for r in rows[:k]:
        print(f"prox={r['proximity']:.3f} eff={r['effect']:+.4f} "
              f"CI[{r['lo'] if r['lo'] is not None else float('nan'):+.4f},"
              f"{r['hi'] if r['hi'] is not None else float('nan'):+.4f}] "
              f"hw={r['half_width']:.4f} nw={r['n_windows']} ne={r['n_episodes']}")
        print(f"   {r['file']}")
        print(f"   @ {r['json_path']}    dups={len(r['dup_files'])}")

show(n40, "A. VALID ESTIMATOR, n~40 -- RE-ADJUDICATE AT n=600", 45)
show(n12, "B. VALID ESTIMATOR, n=12-24 -- closed-loop suites", 30)

json.dump({"valid_n40": n40, "valid_n12": n12, "deprecated": deprecated,
           "unnamed": unnamed, "all_dedup": dedup},
          open(os.path.join(SP, "h1_ranked.json"), "w", encoding="utf-8"), indent=1)
print("\nwrote h1_ranked.json")
