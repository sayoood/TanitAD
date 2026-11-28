"""The SPEC's bars (unchanged) on the validation clips (PI 2026-09-14: "check the fp16 on a third clip"). Reference = the exact arm
(tag <ref>, today's pipeline by the approved exactness of measures 1-4 and 7-8); each test arm vs the reference, per clip:
  N1 map agreement >= 0.990 over cells seen in either · N2 IoU >= 0.95 per class with >= 500 cells in the reference map
  N3 PI checks (roi) |d fragments| <= 2.0, |d crosswalk share| <= 0.02, |d curb recall 1.0 m| <= 0.01, |d edge precision 0.6 m| <= 0.01
     -- only where the clip has LiDAR (pi_checks_<c8>_<json tag>.json present); otherwise reported NOT COMPUTABLE (no LiDAR curbs)
  N4 mean per-frame raster agreement (cls_<CAM>) over labelled pixels >= 0.990
An arm passes when every computable bar passes on every clip.
Usage: spd_eval_clips.py <out json> <ref tag> <arm,arm> <cam>:<c8>[:<pi json tag>] [...]"""
import json, sys
from pathlib import Path
import numpy as np

SM = Path("/home/nvidia/sam3map")
out, ref, arms = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
specs = [a.split(":") for a in sys.argv[4:]]
N3 = {"fragments_per_1000m2": 2.0, "crosswalk_coloured_share_of_crossing_area": 0.02, "curb_recall_edge_within_1.0m": 0.01, "edge_precision_within_0.6m": 0.01}
rep = {"ref": ref, "arms": {a: {"clips": {}} for a in arms}}
for sp in specs:
    cam, c8 = sp[0], sp[1]; pij = sp[2] if len(sp) > 2 else None
    a_map = np.load(SM / f"render5_{c8}_{ref}r" / "worldmap.npz", allow_pickle=True)["cls"]
    pic = json.loads((SM / f"pi_checks_{c8}_{pij}.json").read_text()) if pij and (SM / f"pi_checks_{c8}_{pij}.json").exists() else None
    rfiles = sorted((SM / f"{c8}_{ref}raw").glob("[0-9][0-9][0-9].npz"))
    for arm in arms:
        b_map = np.load(SM / f"render5_{c8}_{arm}r" / "worldmap.npz", allow_pickle=True)["cls"]
        seen = (a_map != 255) | (b_map != 255)
        n1 = float((a_map[seen] == b_map[seen]).mean())
        iou = {int(k): float(((a_map == k) & (b_map == k)).sum() / max(int(((a_map == k) | (b_map == k)).sum()), 1)) for k in range(8) if int((a_map == k).sum()) >= 500}
        agree = []
        for f in rfiles:
            ca = np.load(f, allow_pickle=True)[f"cls_{cam}"]; cb = np.load(SM / f"{c8}_{arm}raw" / f.name, allow_pickle=True)[f"cls_{cam}"]
            lab = (ca > 0) | (cb > 0)
            agree.append(1.0 - float(((ca != cb) & lab).sum()) / max(int(lab.sum()), 1))
        bars = {"N1": {"value": n1, "pass": n1 >= 0.990}, "N2": {"value": iou, "pass": all(v >= 0.95 for v in iou.values())},
                "N4": {"value": float(np.mean(agree)), "min_frame": float(np.min(agree)), "pass": float(np.mean(agree)) >= 0.990}}
        if pic is not None and ref in pic["variants"] and arm in pic["variants"]:
            ra, rb = pic["variants"][ref]["roi"], pic["variants"][arm]["roi"]
            d3 = {k: (rb[k] - ra[k]) if (rb.get(k) is not None and ra.get(k) is not None) else None for k in N3}
            bars["N3"] = {"value": d3, "pass": all(d3[k] is not None and abs(d3[k]) <= N3[k] for k in N3)}
        else:
            bars["N3"] = {"value": None, "pass": None, "note": "not computable: no LiDAR on this clip" if not pij else "pi_checks missing"}
        ok = all(b["pass"] for b in bars.values() if b["pass"] is not None)
        if pij and bars["N3"]["pass"] is None:
            ok = False                                                          # a LiDAR clip whose PI checks are missing does not pass
        rep["arms"][arm]["clips"][c8] = {"cam": cam, "frames": len(rfiles), "cells_differ": int(((a_map != b_map) & seen).sum()), "bars": bars, "pass": ok}
for arm in arms:
    cl = rep["arms"][arm]["clips"]
    rep["arms"][arm]["verdict"] = "NUMERICAL-PASS" if all(v["pass"] for v in cl.values()) else "FAIL"
    rep["arms"][arm]["lowest_edge_iou"] = min((v["bars"]["N2"]["value"].get(5, 1.0) for v in cl.values()), default=None)
out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
for arm in arms:
    print(f"{arm}: {rep['arms'][arm]['verdict']}  (lowest edge IoU {rep['arms'][arm]['lowest_edge_iou']:.4f})")
    for c8, v in rep["arms"][arm]["clips"].items():
        b = v["bars"]
        n3 = "n/a (no LiDAR)" if b["N3"]["pass"] is None else ("ok" if b["N3"]["pass"] else f"FAIL {b['N3']['value']}")
        print(f"   {c8} {v['cam']}: N1 {b['N1']['value']:.5f} {'ok' if b['N1']['pass'] else 'FAIL'} | N2 {{{', '.join(f'{k}: {x:.3f}' for k, x in b['N2']['value'].items())}}} "
              f"{'ok' if b['N2']['pass'] else 'FAIL'} | N3 {n3} | N4 {b['N4']['value']:.5f} (min {b['N4']['min_frame']:.3f}) {'ok' if b['N4']['pass'] else 'FAIL'} | differ {v['cells_differ']}")
print("ZZSPDEVALCLIPS-DONEZZ")
