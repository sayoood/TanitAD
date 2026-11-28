"""P4: where does the lane COUNT come from and how wrong is it?

`lanes_visible` is a VLM (B1) field, DEFINED at ph0_v2.py:140 as "lanes you
can count on the ego's carriageway (0 if unclear)" — i.e. it EXCLUDES the
oncoming carriageway by construction. Measure its distribution, its internal
consistency with `lane_ego`, and its rendering.
"""
import glob
import json
import os
from collections import Counter

SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
FUSED = os.path.join(SP, "s2_pull", "fused_aug120")
REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
PKG = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                   "Implementation", "incoming", "2026-08-16-s2-v1-labels")

files = sorted(glob.glob(os.path.join(FUSED, "*.json")))
files = [f for f in files if not os.path.basename(f).startswith("_")]
print(f"fused aug120 records: {len(files)}")

lv, le, rt, conf = Counter(), Counter(), Counter(), Counter()
viol_ego_ge = []       # lane_ego >= lanes_visible  (impossible: 0-based index)
zero_unclear = []
rows = {}
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    cid = d.get("clip_id") or os.path.splitext(os.path.basename(f))[0]
    sc = (d.get("semantics") or {}).get("scene") or {}
    n, e = sc.get("lanes_visible"), sc.get("lane_ego")
    rows[cid] = {"lanes_visible": n, "lane_ego": e,
                 "road_type": sc.get("road_type"), "domain": sc.get("domain"),
                 "conf": sc.get("conf"),
                 "scenario": d.get("scenario_description")}
    lv[n] += 1
    le[e] += 1
    rt[sc.get("road_type")] += 1
    conf[sc.get("conf")] += 1
    if isinstance(n, int) and isinstance(e, int) and n > 0 and e >= n:
        viol_ego_ge.append((cid, n, e))
    if n == 0:
        zero_unclear.append(cid)

print(f"\nlanes_visible distribution: {dict(sorted(lv.items(), key=lambda k: (k[0] is None, k[0])))}")
print(f"lane_ego     distribution: {dict(sorted(le.items(), key=lambda k: (k[0] is None, k[0])))}")
print(f"road_type    distribution: {dict(rt)}")
print(f"B1 conf      distribution: {dict(conf)}")
print(f"\nlanes_visible == 0 ('unclear' per the prompt): {len(zero_unclear)}"
      f"/{len(files)} = {100*len(zero_unclear)/len(files):.1f}%")
print(f"INTERNAL CONTRADICTION lane_ego >= lanes_visible (0-based index must "
      f"be < count): {len(viol_ego_ge)}/{len(files)} = "
      f"{100*len(viol_ego_ge)/len(files):.1f}%")
for cid, n, e in viol_ego_ge[:12]:
    print(f"   {cid[:8]}  lanes_visible={n} lane_ego={e}")

# --- the two clips the PI named -------------------------------------------- #
print("\n--- the clips the PI flagged on lane count ---")
verd = json.load(open(os.path.join(PKG, "review",
                                   "PI_VERDICTS_2026-08-16.json"),
                      encoding="utf-8"))["verdicts"]
for cid in ("03ba450b-121a-483b-aa48-6d9097c308de",
            "51fd1c9f-fc1d-4a04-94d4-2e6d50161944"):
    r = rows.get(cid, {})
    print(f"\n{cid[:8]}: lanes_visible={r.get('lanes_visible')} "
          f"lane_ego={r.get('lane_ego')} road_type={r.get('road_type')} "
          f"conf={r.get('conf')}")
    print(f"   rendered: {r.get('scenario')!r}")
    print(f"   PI note : {verd[cid]['note']!r}")

json.dump(rows, open(os.path.join(SP, "lane_count.json"), "w",
                     encoding="utf-8"), indent=1)

# --- how the census renders ------------------------------------------------- #
noag = sum(1 for r in rows.values()
           if "no agents" in (r.get("scenario") or ""))
print(f"\n--- 'no agents' rendered in scenario_description: {noag}/{len(files)}"
      f" = {100*noag/len(files):.1f}% of the fused aug120 corpus")
