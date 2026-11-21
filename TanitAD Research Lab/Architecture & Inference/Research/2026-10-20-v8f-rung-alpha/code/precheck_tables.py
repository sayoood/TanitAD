"""PRE-CHECK step 3: tabulate the S1-grid read of P10-pose (eval_*.json + analysis_*.json -> summary.json + tables.md).

usage: precheck_tables.py <dir holding eval_<H>_s<seed>.json / analysis_<H>_s0.json> <out dir>
"""
import json
import os
import sys

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)


def J(n):
    p = os.path.join(SRC, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fmt(r, dp=4):
    s = " SEP" if r.get("separated") else ""
    return f"{r['delta']:+.{dp}f} [{r['lo']:+.{dp}f}, {r['hi']:+.{dp}f}]{s}"


S = {"tier": "T1 (self-action OPEN loop; the model does not control the vehicle)",
     "estimator": "paired episode-cluster bootstrap, B 2000, seed 0 (taniteval/ci.py); episode variance only (models fixed)",
     "arms": "os = refcv7-50,400 (banked T1 dump); P10pose = pose-only P10, S1 recipe refit on the 4,572-clip train view "
             "(per horizon); ha0_ext_pose = alpha-0's prior; P10 = S1's steer P10 (control)"}
lines = ["| horizon | seed | pair (x - y) | ADE 0-H [95 % CI] | FDE [95 % CI] | n win / eps |", "|---|---|---|---|---|---|"]
PAIRS = ("os-P10pose", "os-ha0_ext_pose", "P10pose-ha0_ext_pose", "P10pose-P10", "P10pose-kd_x_pose", "os-P10", "os-ha0_ext",
         "ha0_ext_pose-ha0_ext", "P10pose-P10pose")
S["paired_direct"] = {}
for h in ("S2", "S6"):
    for sd in (0, 1):
        d = J(f"eval_{h}_s{sd}.json")
        if not d:
            continue
        S["paired_direct"][f"{h}_s{sd}"] = {k: {"ade": v["ade"], "fde": v["fde"]} for k, v in d["paired_x_minus_y"].items()}
        for pr in PAIRS:
            r = d["paired_x_minus_y"][pr]
            lines.append(f"| {h} | s{sd} | {pr} | {fmt(r['ade'])} | {fmt(r['fde'])} | {r['ade']['n_windows']} / {r['ade']['n_episodes']} |")
S["means"] = {h: J(f"eval_{h}_s0.json")["means"] for h in ("S2", "S6") if J(f"eval_{h}_s0.json")}
S["controls"] = {h: J(f"eval_{h}_s0.json")["controls"] for h in ("S2", "S6") if J(f"eval_{h}_s0.json")}
lv = ["| horizon | arm | mean ADE m | mean FDE m |", "|---|---|---|---|"]
for h, m in S["means"].items():
    for k, v in m.items():
        lv.append(f"| {h} | {k} | {v['ade']:.4f} | {v['fde']:.4f} |")
# ---- the four families (the battery's own tool, seed 0) --------------------------------------------------------
fam_lines = ["| horizon | pair (x - y) | family | metric | delta [95 % CI] | n |", "|---|---|---|---|---|---|"]
S["families"] = {}
for h in ("S2", "S6"):
    a = J(f"analysis_{h}_s0.json")
    if not a:
        continue
    fp = a["refcv3"]["families_paired"]
    S["families"][h] = {}
    for nm in ("paired_os_minus_P10pose", "paired_os_minus_ha0extpose", "paired_P10pose_minus_ha0extpose",
               "paired_P10pose_minus_P10", "paired_os_minus_P10"):
        if nm not in fp:
            continue
        rows = []
        for fam, dd in fp[nm]["families"].items():
            for mk, r in dd.items():
                if isinstance(r, dict) and "delta" in r:
                    rows.append({"family": fam, "metric": mk, "delta": r["delta"], "lo": r["lo"], "hi": r["hi"],
                                 "separated": r["separated"], "n_windows": r["n_windows"]})
                    fam_lines.append(f"| {h} | {fp[nm]['direction']} | {fam} | {mk} | {fmt(r)} | {r['n_windows']} |")
        S["families"][h][nm] = {"direction": fp[nm]["direction"], "rows": rows}
    dk = (a["refcv3"].get("distance_keeping") or {}).get("paired", {})
    for nm in ("paired_os_minus_P10pose", "paired_P10pose_minus_ha0extpose"):
        if nm in dk:
            S["families"][h][nm]["distance_keeping"] = {m: {k: r.get(k) for k in ("delta", "lo", "hi", "separated", "n_used")}
                                                        for m, r in dk[nm]["metrics"].items()}
# ---- the SHIPPED artifact (p10_pose_prior.pt: 30 knots, v8f_mini recipe, 4,572 train clips) read on the same grid ---
S["artifact"] = {"what": "the shipped frozen artifact itself, scored by the same grid (eval_<H>_s<seed>_art.json / analysis_<H>_s0_art.json)"}
art_lines = ["| horizon | seed | pair (x - y) | ADE 0-H [95 % CI] | FDE [95 % CI] |", "|---|---|---|---|---|"]
art_fam = ["| horizon | pair (x - y) | family | metric | delta [95 % CI] | n |", "|---|---|---|---|---|---|"]
for h in ("S2", "S6"):
    S["artifact"][h] = {"means": (J(f"eval_{h}_s0_art.json") or {}).get("means"), "paired": {}, "families": {}}
    for sd in (0, 1):
        d = J(f"eval_{h}_s{sd}_art.json")
        if not d:
            continue
        for pr in ("os-P10pose_artifact", "P10pose_artifact-ha0_ext_pose", "P10pose_artifact-P10pose"):
            r = d["paired_x_minus_y"][pr]
            S["artifact"][h]["paired"][f"{pr}_s{sd}"] = {"ade": r["ade"], "fde": r["fde"]}
            art_lines.append(f"| {h} | s{sd} | {pr} | {fmt(r['ade'])} | {fmt(r['fde'])} |")
    a = J(f"analysis_{h}_s0_art.json")
    if a:
        fp = a["refcv3"]["families_paired"]
        for nm in ("paired_os_minus_P10poseart", "paired_P10poseart_minus_ha0extpose"):
            rows = []
            for fam, dd in fp[nm]["families"].items():
                for mk, r in dd.items():
                    if isinstance(r, dict) and "delta" in r:
                        rows.append({"family": fam, "metric": mk, "delta": r["delta"], "lo": r["lo"], "hi": r["hi"],
                                     "separated": r["separated"], "n_windows": r["n_windows"]})
                        art_fam.append(f"| {h} | {fp[nm]['direction']} | {fam} | {mk} | {fmt(r)} | {r['n_windows']} |")
            S["artifact"][h]["families"][nm] = {"direction": fp[nm]["direction"], "rows": rows}
da = [J(f"eval_S2_s{s}_art.json")["paired_x_minus_y"]["os-P10pose_artifact"]["ade"] for s in (0, 1)]
S["premise_S2_artifact"] = {"statement": "os - (shipped artifact) ADE 0-2 s > 0 with the CI excluding 0 at BOTH inference seeds",
                            "s0": da[0], "s1": da[1], "holds": bool(all(r["delta"] > 0 and r["lo"] > 0 for r in da))}
S["strategic"] = "PAUSED (PI R6 ruling 2026-09-19, PI_DECISION_QUEUE ITEM 25) -- not a pass, not omitted"
# ---- the headline: does the pose-only P10 beat refcv7 at 2 s? ------------------------------------------------------
d2 = [J(f"eval_S2_s{s}.json")["paired_x_minus_y"]["os-P10pose"]["ade"] for s in (0, 1)]
S["premise_S2"] = {
    "statement": "os - P10pose ADE 0-2 s > 0 with the CI excluding 0 at BOTH inference seeds (P10-pose beats refcv7 at 2 s; "
                 "S1's steer P10 premise was os - P10 = +0.0276)",
    "s0": d2[0], "s1": d2[1], "holds": bool(all(r["delta"] > 0 and r["lo"] > 0 for r in d2))}
json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8").write(
    "\n".join(["## paired ADE / FDE (S1-recipe refit, per horizon)", *lines, "", "## levels", *lv, "",
                "## four families (seed 0, S1-recipe refit)", *fam_lines, "",
                "## the SHIPPED artifact -- paired ADE / FDE", *art_lines, "",
                "## the SHIPPED artifact -- four families (seed 0)", *art_fam]) + "\n")
print("PREMISE (P10-pose beats refcv7 at 2 s):", S["premise_S2"]["holds"], json.dumps({k: S["premise_S2"][k] for k in ("s0", "s1")}))
print("PREMISE (shipped artifact beats refcv7 at 2 s):", S["premise_S2_artifact"]["holds"],
      json.dumps({k: S["premise_S2_artifact"][k] for k in ("s0", "s1")}))
print("\n".join(lines[:8]))
