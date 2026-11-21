"""H1 pass 3: project each n~40 null to n=600 using the MEASURED half-width
shrinkage (x2.8-3.9, mean 3.4 -- MODEL_REGISTRY 1.2a), and isolate HEADLINE /
PRIMARY / VERDICT-bearing nodes from auto-emitted panel strata.
"""
import json, os, re, collections

SP = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(SP, "h1_ranked.json"), encoding="utf-8"))
n40 = d["valid_n40"]
n12 = d["valid_n12"]
alld = d["all_dedup"]

SHRINK_LO, SHRINK_MEAN, SHRINK_HI = 2.8, 3.4, 3.9

def project(rows):
    for r in rows:
        p = r["proximity"]
        r["prox_at600_mean"] = p * SHRINK_MEAN
        r["prox_at600_lo"] = p * SHRINK_LO
        r["would_flip_conservative"] = p * SHRINK_LO > 1.0     # even at weakest shrinkage
        r["would_flip_mean"] = p * SHRINK_MEAN > 1.0
    return rows

project(n40); project(n12)

flip_cons = [r for r in n40 if r["would_flip_conservative"]]
flip_mean = [r for r in n40 if r["would_flip_mean"]]
print(f"n~40 valid-estimator nulls: {len(n40)}")
print(f"  would separate at n=600 under the MEAN shrinkage x3.4 : {len(flip_mean)} "
      f"({100*len(flip_mean)/len(n40):.1f}%)")
print(f"  would separate even under the WEAKEST shrinkage x2.8  : {len(flip_cons)} "
      f"({100*len(flip_cons)/len(n40):.1f}%)")

# --- headline / primary detection -----------------------------------------
HEADLINE_PAT = re.compile(
    r"headline|primary|verdict|PRIMARY|HEADLINE|^PAIRED|top_?level|"
    r"decision|gate|preregist|criteri|result\.|G[0-9]_|bar_|^points\.[0-9]+\.(?!.*by_)",
    re.I)
PANEL_PAT = re.compile(
    r"vs_floor_paired|by_curvature|kinematic_strata|longitudinal_regime|"
    r"by_horizon_paired|strata\.|conditions\.|per_|_grid\[", re.I)

for r in n40 + n12:
    jp = r["json_path"]
    r["is_panel_stratum"] = bool(PANEL_PAT.search(jp))
    r["is_headline"] = bool(HEADLINE_PAT.search(jp)) and not r["is_panel_stratum"]

hl40 = sorted([r for r in n40 if r["is_headline"]], key=lambda r: -r["proximity"])
hl12 = sorted([r for r in n12 if r["is_headline"]], key=lambda r: -r["proximity"])
nonpanel40 = sorted([r for r in n40 if not r["is_panel_stratum"]], key=lambda r: -r["proximity"])
nonpanel12 = sorted([r for r in n12 if not r["is_panel_stratum"]], key=lambda r: -r["proximity"])

print(f"\nheadline-shaped nulls: n40={len(hl40)}  n12={len(hl12)}")
print(f"non-panel nulls:       n40={len(nonpanel40)}  n12={len(nonpanel12)}")

# per-workstream (dir) rollup of the best non-panel null
byws = collections.defaultdict(list)
for r in nonpanel40 + nonpanel12:
    m = re.search(r"incoming/([^/]+)/", r["file"])
    ws = m.group(1) if m else ("taniteval/results" if r["file"].startswith("taniteval")
                               else os.path.dirname(r["file"]))
    byws[ws].append(r)

print("\n\n===== BEST NON-PANEL NULL PER WORKSTREAM (ranked) =====")
best = sorted(((ws, max(rs, key=lambda r: r["proximity"])) for ws, rs in byws.items()),
              key=lambda t: -t[1]["proximity"])
for ws, r in best:
    flag = "FLIP@2.8x" if r["would_flip_conservative"] else (
        "flip@3.4x" if r["would_flip_mean"] else "        ")
    print(f"{r['proximity']:.3f} -> {r['prox_at600_mean']:.2f}  {flag}  "
          f"eff={r['effect']:+.4f} hw={r['half_width']:.4f} ne={r['n_episodes']} nw={r['n_windows']}")
    print(f"      WS: {ws}")
    print(f"      {r['file']}")
    print(f"      @ {r['json_path']}  (n_nulls_in_ws={len(byws[ws])})")

json.dump({"n40": n40, "n12": n12, "headline40": hl40, "headline12": hl12,
           "nonpanel40": nonpanel40, "nonpanel12": nonpanel12,
           "flip_conservative": flip_cons,
           "counts": {"n40_total": len(n40), "n40_flip_mean": len(flip_mean),
                      "n40_flip_conservative": len(flip_cons),
                      "n12_total": len(n12)}},
          open(os.path.join(SP, "h1_projected.json"), "w", encoding="utf-8"), indent=1)
print("\nwrote h1_projected.json")
