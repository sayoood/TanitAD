"""Verdicts of SPEC_sam3_speed_prereg.md from the artifacts of spd_chain1.sh (Thor). Nothing here tunes anything: bars are the
SPEC's literals.
  C-REPRO  spd0 raw frames == historical <c8>_v6sfraw (bit) and spd0 map == delivered render5_<c8>_v65rf (cell)
  EXACT    raw frames bit-identical to spd0 and composed map identical
  N1 map agreement over cells seen in either >= 0.990 · N2 IoU >= 0.95 per class with >= 500 cells in spd0
  N3 pi_checks roi deltas: fragments <= 2.0, crosswalk share <= 0.02, curb recall 1.0 m <= 0.01, edge precision 0.6 m <= 0.01
  N4 mean per-frame cls_CAM_FW agreement over labelled pixels >= 0.990
  C-REG    spdREG must fail at least one of N1-N4 on at least one clip
Usage: spd_eval.py <arm> [<arm> ...]   (spd0 is the reference) -> /home/nvidia/sam3map/spd_eval.json"""
import json, sys
from pathlib import Path
import numpy as np

SM = Path("/home/nvidia/sam3map")
CLIPS = ["73495082f98b", "4fbd97b6a4b7"]
# an argument "arm:base" declares an exact-class measure applied ON TOP of arm `base` (e.g. async CPU on the fp16 arm): its
# EXACT verdict is taken against `base`; the numerical bars N1-N4 are always against spd0
ARMS = [a.split(":")[0] for a in sys.argv[1:]]
BASE = {a.split(":")[0]: (a.split(":")[1] if ":" in a else "spd0") for a in sys.argv[1:]}
KEYS = ["tok", "T_world_rig", "cls_CAM_FW", "evid_CAM_FW", "xstripe_CAM_FW"] + [f"pts_{k}" for k in range(1, 8)] + [f"rng_{k}" for k in range(1, 8)]
N3 = {"fragments_per_1000m2": 2.0, "crosswalk_coloured_share_of_crossing_area": 0.02, "curb_recall_edge_within_1.0m": 0.01, "edge_precision_within_0.6m": 0.01}


def raw_compare(ref_dir, test_dir):
    files = sorted(ref_dir.glob("[0-9][0-9][0-9].npz"))
    ident, agree, missing = 0, [], 0
    for f in files:
        g = test_dir / f.name
        if not g.exists():
            missing += 1; continue
        a = np.load(f, allow_pickle=True); b = np.load(g, allow_pickle=True)
        same = all(k in a.files and k in b.files and a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in KEYS)
        ident += same
        ca, cb = a["cls_CAM_FW"], b["cls_CAM_FW"]; lab = (ca > 0) | (cb > 0)
        agree.append(1.0 - float(((ca != cb) & lab).sum()) / max(int(lab.sum()), 1))
    return {"frames": len(files), "missing": missing, "bit_identical": int(ident), "cls_agree_mean": float(np.mean(agree)) if agree else None,
            "cls_agree_min": float(np.min(agree)) if agree else None}


def map_compare(ref_npz, test_npz):
    a = np.load(ref_npz, allow_pickle=True); b = np.load(test_npz, allow_pickle=True)
    ca, cb = a["cls"], b["cls"]
    if ca.shape != cb.shape or not np.allclose(a["origin"], b["origin"]):
        return {"grid_mismatch": True, "shapes": [list(ca.shape), list(cb.shape)]}
    seen = (ca != 255) | (cb != 255)
    out = {"identical": bool(np.array_equal(ca, cb)), "cells_seen": int(seen.sum()), "cells_differ": int(((ca != cb) & seen).sum()),
           "agreement": float((ca[seen] == cb[seen]).mean()), "iou": {}, "ref_cells": {}}
    for k in range(0, 8):
        n = int((ca == k).sum())
        out["ref_cells"][k] = n
        if n >= 500:
            out["iou"][k] = float(((ca == k) & (cb == k)).sum() / max(int(((ca == k) | (cb == k)).sum()), 1))
    return out


def lidar(c8, tag):
    p = SM / f"sam3map_score_{c8}_{tag}c_gt_{tag}.json"
    if not p.exists():
        return None
    g = json.loads(p.read_text()).get("SAM3_GT_worldmap", {})
    return {k: v.get("value") for k, v in g.items() if isinstance(v, dict)}


rep = {"arms": {}, "controls": {}, "timing": {}}
# timing
for t in ARMS:
    p = SM / f"front_fast_{t}.json"
    if p.exists():
        d = json.loads(p.read_text())
        rep["timing"][t] = {"build_s": d["build_s"], "ready_s": d["ready_s"], "max_mem_gb": d.get("max_mem_gb"),
                            **{c8: {k: d["clips"][c8][k] for k in ("frames", "written", "wall_s", "s_per_frame", "sam3_s_mean", "cpu_s_mean", "load_wait_s_mean")}
                               for c8 in d["clips"]}}
# C-REPRO
rep["controls"]["C-REPRO"] = {}
for c8 in CLIPS:
    r = raw_compare(SM / f"{c8}_v6sfraw", SM / f"{c8}_spd0raw")
    mp = map_compare(SM / f"render5_{c8}_v65rf/worldmap.npz", SM / f"render5_{c8}_spd0r/worldmap.npz") if (SM / f"render5_{c8}_spd0r/worldmap.npz").exists() else None
    rep["controls"]["C-REPRO"][c8] = {"raw_vs_historical": r, "map_vs_delivered": mp,
                                       "pass": r["bit_identical"] == r["frames"] and r["missing"] == 0 and bool(mp and mp.get("identical"))}
pic = {c8: json.loads((SM / f"pi_checks_{c8}_spd.json").read_text()) if (SM / f"pi_checks_{c8}_spd.json").exists() else None for c8 in CLIPS}
for c8 in CLIPS:                                                               # parts 2, 3 ran pi_checks again with spd0 in the same run
  for p2 in (SM / f"pi_checks_{c8}_spd2.json", SM / f"pi_checks_{c8}_spd3.json", SM / f"pi_checks_{c8}_spd4.json", SM / f"pi_checks_{c8}_spd5.json"):
    if p2.exists():
        d2 = json.loads(p2.read_text())
        if pic[c8] is None:
            pic[c8] = d2
        else:
            same = pic[c8]["variants"]["spd0"]["roi"] == d2["variants"]["spd0"]["roi"]
            rep["controls"].setdefault("pi_checks_spd0_identical_across_runs", {})[c8] = bool(same)
            for k, v in d2["variants"].items():
                if k not in pic[c8]["variants"]:
                    pic[c8]["variants"][k] = v
for t in ARMS:
    if t == "spd0":
        continue
    rep["arms"][t] = {}
    for c8 in CLIPS:
        raw = raw_compare(SM / f"{c8}_spd0raw", SM / f"{c8}_{t}raw")
        mp = map_compare(SM / f"render5_{c8}_spd0r/worldmap.npz", SM / f"render5_{c8}_{t}r/worldmap.npz") if (SM / f"render5_{c8}_{t}r/worldmap.npz").exists() else {"missing": True}
        v = {"raw": raw, "map": mp, "exact": bool(raw["bit_identical"] == raw["frames"] and raw["missing"] == 0 and mp.get("identical"))}
        if BASE[t] != "spd0":                                                   # exact measure on top of another arm
            rb = raw_compare(SM / f"{c8}_{BASE[t]}raw", SM / f"{c8}_{t}raw")
            mb = map_compare(SM / f"render5_{c8}_{BASE[t]}r/worldmap.npz", SM / f"render5_{c8}_{t}r/worldmap.npz") if (SM / f"render5_{c8}_{t}r/worldmap.npz").exists() else {}
            v["vs_base"] = {"base": BASE[t], "raw": rb, "map_identical": mb.get("identical")}
            v["exact_vs_base"] = bool(rb["bit_identical"] == rb["frames"] and rb["missing"] == 0 and mb.get("identical"))
        bars = {}
        bars["N1"] = {"value": mp.get("agreement"), "pass": mp.get("agreement") is not None and mp["agreement"] >= 0.990}
        bars["N2"] = {"value": mp.get("iou"), "pass": bool(mp.get("iou")) and all(x >= 0.95 for x in mp["iou"].values())}
        if pic[c8] and t in pic[c8]["variants"]:
            ra, rt = pic[c8]["variants"]["spd0"]["roi"], pic[c8]["variants"][t]["roi"]
            d3 = {k: (rt.get(k) - ra.get(k)) if (rt.get(k) is not None and ra.get(k) is not None) else None for k in N3}
            bars["N3"] = {"value": d3, "pass": all(d3[k] is not None and abs(d3[k]) <= N3[k] for k in N3)}
        else:
            bars["N3"] = {"value": None, "pass": False, "note": "pi_checks missing"}
        bars["N4"] = {"value": raw["cls_agree_mean"], "pass": raw["cls_agree_mean"] is not None and raw["cls_agree_mean"] >= 0.990}
        v["bars"] = bars; v["numerical_pass"] = all(b["pass"] for b in bars.values())
        la, lt = lidar(c8, "spd0"), lidar(c8, t)
        v["lidar_delta"] = {k: round(lt[k] - la[k], 4) for k in la if lt and k in lt and la[k] is not None and lt[k] is not None} if la and lt else None
        rep["arms"][t][c8] = v
    rep["arms"][t]["verdict"] = ("EXACT" if all(rep["arms"][t][c]["exact"] for c in CLIPS) else
                                 "NUMERICAL-PASS" if all(rep["arms"][t][c]["numerical_pass"] for c in CLIPS) else "FAIL")
    if BASE[t] != "spd0":
        rep["arms"][t]["verdict_vs_base"] = f"EXACT vs {BASE[t]}" if all(rep["arms"][t][c].get("exact_vs_base") for c in CLIPS) else f"NOT EXACT vs {BASE[t]}"
if "spdREG" in rep["arms"]:
    rep["controls"]["C-REG"] = {"pass": rep["arms"]["spdREG"]["verdict"] == "FAIL"}
# delivered map vs its own pi_checks row (sanity that spd0 reads like map r)
for c8 in CLIPS:
    if pic[c8] and "delivered_r" in pic[c8]["variants"] and "spd0" in pic[c8]["variants"]:
        rep["controls"].setdefault("pi_checks_spd0_vs_delivered", {})[c8] = {k: (pic[c8]["variants"]["spd0"]["roi"].get(k), pic[c8]["variants"]["delivered_r"]["roi"].get(k)) for k in N3}
(SM / "spd_eval.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")

print("== timing (s/frame)")
for t, d in rep["timing"].items():
    print(f"  {t:8s} build {d['build_s']:.1f}s " + "  ".join(f"{c8}: {d[c8]['s_per_frame']:.3f} (sam3 {d[c8]['sam3_s_mean']:.3f} cpu {d[c8]['cpu_s_mean']:.3f})" for c8 in CLIPS if c8 in d))
print("== controls")
for c8, v in rep["controls"]["C-REPRO"].items():
    mp = v["map_vs_delivered"] or {}
    print(f"  C-REPRO {c8}: raw {v['raw_vs_historical']['bit_identical']}/{v['raw_vs_historical']['frames']} bit-identical, map identical {mp.get('identical')} "
          f"(differ {mp.get('cells_differ')}) -> {'PASS' if v['pass'] else 'FAIL'}")
if "C-REG" in rep["controls"]:
    print(f"  C-REG spdREG must fail: {'PASS' if rep['controls']['C-REG']['pass'] else 'FAIL'}")
print("== arms vs spd0")
for t, a in rep["arms"].items():
    print(f"  {t}: {a['verdict']}" + (f"  |  {a['verdict_vs_base']}" if "verdict_vs_base" in a else ""))
    for c8 in CLIPS:
        v = a[c8]; b = v["bars"]
        n3 = b["N3"]["value"] or {}
        print(f"    {c8}: raw bit-identical {v['raw']['bit_identical']}/{v['raw']['frames']} map identical {v['map'].get('identical')} differ {v['map'].get('cells_differ')} | "
              f"N1 {b['N1']['value'] if b['N1']['value'] is None else round(b['N1']['value'], 5)} {'ok' if b['N1']['pass'] else 'FAIL'} | "
              f"N2 {({k: round(x, 4) for k, x in (b['N2']['value'] or {}).items()})} {'ok' if b['N2']['pass'] else 'FAIL'} | "
              f"N3 {({k[:12]: (None if x is None else round(x, 4)) for k, x in n3.items()})} {'ok' if b['N3']['pass'] else 'FAIL'} | "
              f"N4 {b['N4']['value'] if b['N4']['value'] is None else round(b['N4']['value'], 5)} {'ok' if b['N4']['pass'] else 'FAIL'} | lidar {v['lidar_delta']}")
print("ZZSPDEVAL-DONEZZ")
