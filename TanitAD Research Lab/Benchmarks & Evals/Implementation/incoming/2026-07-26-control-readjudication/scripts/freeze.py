# -*- coding: utf-8 -*-
"""FREEZE the control-null list. Classify every control-type node by family.
Run BEFORE any re-adjudication. Output: frozen_list.json"""
import json, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
rows = json.load(open("control_sweep_raw.json", encoding="utf-8"))
META = ("harvest_index.json","h2_unused_capabilities.json","h3_stranded.json","h4_contradictions.json")
rows = [r for r in rows if not any(m in r["file"] for m in META)]

def family(r):
    p, f = r["path"].lower(), r["file"].lower()
    if "paired_leak_" in p:            return "S3-LEAK (blind-baseline echo: does conditioning alone move a BLIND head?)"
    if "paired_clock_" in p:           return "S3-CLOCK (R3: does an observable-horizon channel alone move a BLIND head?)"
    if "blind_qwk_ci" in p:            return "S3-BLINDvsCHANCE (is the BLIND arm above chance at all?)"
    if "blind_vs_majority" in p:       return "S1-BLINDvsMAJORITY (does a goal-blind head beat the trivial majority?)"
    if "dead_shuffle" in p or "dead_noise" in p: return "DEAD-CONTROL (a perturbation that must have NO effect)"
    if "_shuf" in p:                   return "SHUFFLE (feature-shuffle negative control)"
    if "ce_control" in p:              return "CE-CONTROL (counterfactual-equal control arm)"
    if "vs_chance" in p or "random_at_rate" in p or "delta_vs_random" in p: return "CHANCE-BASELINE (does the head beat random at matched rate?)"
    return "UNCLASSIFIED"

# only nodes where a NULL is the DESIRED verdict
nulls = [r for r in rows if (r["sep"] is False) or
         (r["sep"] is None and r["lo"] is not None and r["hi"] is not None and r["lo"] <= 0 <= r["hi"])]
for r in nulls:
    r["family"] = family(r)
    # Minimum Detectable Effect: with a 95% CI, |effect| must EXCEED the half-width to separate.
    r["mde"] = r["hw"]
unc = [r for r in nulls if r["family"] == "UNCLASSIFIED"]
print(f"TOTAL control-type NULL nodes: {len(nulls)}")
print(f"UNCLASSIFIED (excluded, keyword false-positives): {len(unc)}")
for r in unc: print("   -", r["file"].split("incoming/")[-1], "::", r["path"])
frozen = [r for r in nulls if r["family"] != "UNCLASSIFIED"]
print(f"\nFROZEN LIST SIZE: {len(frozen)}")
from collections import Counter
for k, v in Counter(r["family"] for r in frozen).most_common():
    print(f"  {v:3d}  {k}")
frozen.sort(key=lambda r: (r["family"], -(r["prox"] or 0)))
json.dump(frozen, open("frozen_list.json","w",encoding="utf-8"), indent=1)
print("\n--- frozen rows ---")
for i, r in enumerate(frozen, 1):
    print(f"{i:2d} [{r['family'].split(' ')[0]}] prox={r['prox']:.3f} MDE={r['mde']:.4f} n_ep={r['n_ep']} | {r['effect']} [{r['lo']}, {r['hi']}]")
    print(f"    {r['file'].split('incoming/')[-1]} :: {r['path']}")
