"""Score-threshold sweep of Qwen-Drive detections against OUR GT (v2 packing), per set.

The upstream visualizer's 0.25 was chosen on nuPlan/nuScenes. On a different domain the score
calibration moves, so the operating point is MEASURED here rather than inherited.

⛔ The threshold is SELECTED on the legacy 14 frames + the first two sequences (the FIT sets) and
REPORTED on the remaining sequences (the scored sets) -- a threshold chosen on the frames it is
scored on would manufacture its own improvement (CLAUDE.md probe rules).
Same matching as ours_metrics.py: class-agnostic BEV centre distance <= 2 m, GT and predictions <= 50 m.
"""
import json
from pathlib import Path
import numpy as np

Q = Path("/home/nvidia/qwendrive/v2")
THRS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
FIT = ["frames_legacy", "seq_4fbd97b6a4b7", "seq_73495082f98b"]
SCORED = ["seq_0d90d20036a3", "seq_6924358fafe0"]


def out_dir(s):
    return Q / ("out_legacy" if s == "frames_legacy" else f"out_{s}")


def match_count(pxy, ps, gxy):
    used = np.zeros(len(gxy), bool)
    for i in np.argsort(-ps):
        if not len(gxy):
            break
        d = np.hypot(*(gxy - pxy[i]).T); d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= 2.0:
            used[j] = True
    return int(used.sum())


def sweep(sets):
    acc = {t: [0, 0, 0] for t in THRS}          # tp, n_pred, n_gt
    nfr = 0
    for s in sets:
        fd = Q / s
        if not fd.is_dir():
            continue
        for d in sorted(p for p in fd.iterdir() if p.is_dir()):
            pp = out_dir(s) / f"{d.name}.npz"
            if not pp.exists():
                continue
            g, r = np.load(d / "gt.npz"), np.load(pp)
            gb = g["boxes"]; gb = gb[np.hypot(gb[:, 0], gb[:, 1]) <= 50.0]
            pb, ps = r["boxes"], r["scores"]
            inr = np.hypot(pb[:, 0], pb[:, 1]) <= 50.0
            nfr += 1
            for t in THRS:
                m = inr & (ps >= t)
                acc[t][0] += match_count(pb[m, :2], ps[m], gb[:, :2]); acc[t][1] += int(m.sum()); acc[t][2] += len(gb)
    rows = []
    for t in THRS:
        tp, npred, ngt = acc[t]
        P, R = tp / max(npred, 1), tp / max(ngt, 1)
        rows.append({"thr": t, "tp": tp, "n_pred": npred, "n_gt": ngt, "precision": round(P, 3),
                     "recall": round(R, 3), "f1": round(2 * P * R / max(P + R, 1e-9), 3)})
    return nfr, rows


res = {}
for name, sets in (("fit", FIT), ("scored", SCORED)):
    nfr, rows = sweep(sets)
    res[name] = {"sets": sets, "frames": nfr, "rows": rows}
best = max(res["fit"]["rows"], key=lambda r: r["f1"])
res["selected_on_fit"] = {"thr": best["thr"], "rule": "max F1 on the FIT sets only"}
sc = {r["thr"]: r for r in res["scored"]["rows"]}
res["scored_at_selected"] = sc.get(best["thr"])
res["scored_at_upstream_0.25"] = sc.get(0.25)
(Q / "pr_sweep.json").write_text(json.dumps(res, indent=1))
for name in ("fit", "scored"):
    print(f"== {name}: {res[name]['frames']} frames, sets {res[name]['sets']}")
    for r in res[name]["rows"]:
        print(f"  thr {r['thr']:.2f}  P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1']:.3f}  tp {r['tp']} / pred {r['n_pred']} / gt {r['n_gt']}")
print("SELECTED on fit:", res["selected_on_fit"], "| scored at selected:", res["scored_at_selected"], "| scored at 0.25:", res["scored_at_upstream_0.25"])
print("ZZPR-DONEZZ")
