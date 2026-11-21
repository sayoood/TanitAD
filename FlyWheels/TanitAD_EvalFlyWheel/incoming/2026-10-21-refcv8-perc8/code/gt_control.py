"""gt_control.py -- the control that must read a KNOWN value: GROUND TRUTH AS THE PREDICTION.
BOX: every window's slots are replaced by the window's own VIS-1 positives (logit +20, xy = gt_xy, cls = gt_cls) and scored by the
     programme's own `detection_metrics.summarise` -> every AP of every non-empty (threshold, class, band) cell must be exactly 1.0.
MAP: `map_hires_metrics.window_stats(gt, gt)` on a structured 1000 x 600 code grid (classes 0..7, 255 = not seen) -> pooled IoU must be
     exactly 1.0 in every (class, band) with ground truth; and the all-one-class prediction must NOT (negative control).
CPU only.   python gt_control.py --pkl <perc_det.pkl> --out <json> [--n-windows 300]
"""
import argparse, json, pickle, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-windows", type=int, default=300)
    a = ap.parse_args()
    from tanitad.eval import detection_metrics as dm
    from taniteval import map_hires_metrics as MH
    det = pickle.load(open(a.pkl, "rb"))
    recs = [r["box3d"][0] for r in det if r.get("box3d")]
    step = max(1, len(recs) // a.n_windows)
    sel = recs[::step][: a.n_windows]
    syn = []
    for pk in sel:
        p = np.asarray(pk["pos"], bool)
        n = int(p.sum())
        q = dict(pk)
        q["logit"] = np.full(n, 20.0, np.float32)
        q["xy"] = np.asarray(pk["gt_xy"])[p].astype(np.float32)
        q["cls"] = np.asarray(pk["gt_cls"])[p].astype(np.int16)
        q["cls_corr"] = q["cls"].copy()
        q["matched"] = np.ones(n, bool)
        q["exempt"] = np.zeros(n, bool)
        q["pair_err"] = np.zeros(n)
        syn.append(q)
    out = {"tool": "gt_control.py", "box": {}, "map": {}}
    s = dm.summarise(syn, "box3d")
    aps = {k: v for k, v in s.items() if "_det_ap" in k and v == v}
    bad = {k: v for k, v in aps.items() if abs(v - 1.0) > 1e-12}
    out["box"] = {"n_windows": len(syn), "n_pos": int(s["eval_box3d_n_pos"]), "n_ap_cells_defined": len(aps), "n_cells_not_1": len(bad),
                  "cells_not_1_examples": dict(list(bad.items())[:5]), "pass": bool(len(aps) > 0 and not bad),
                  "auroc_matched": s["eval_box3d_auroc_matched"], "rec@gate": s["eval_box3d_rec@gate"],
                  "prec@gate": s["eval_box3d_prec@gate"]}
    # MAP
    H, W = 1000, 600
    rng = np.random.default_rng(0)
    gt = np.full((H, W), 255, np.uint8)
    for r0 in range(0, H, 100):
        for c0 in range(0, W, 100):
            gt[r0:r0 + 100, c0:c0 + 100] = rng.integers(0, 8)
    gt[:, :30] = 255
    pred_gt = np.where(gt == 255, 0, gt).astype(np.uint8)          # a not-seen GT cell is masked by the scorer; the prediction there must still be a valid code
    tab = MH.WindowTable(arms=("gt", "const1"))
    tab.add("c0", gt, {"gt": pred_gt, "const1": np.full_like(gt, 1)})
    pooled_gt = MH.pooled({k: np.asarray(v) for k, v in tab.arrays("gt").items()})
    pooled_c1 = MH.pooled({k: np.asarray(v) for k, v in tab.arrays("const1").items()})
    iou_gt = pooled_gt["iou"]
    n_gt = tab.arrays("gt")["n_gt"].sum(0)
    ok = [(abs(float(iou_gt[k, b]) - 1.0) < 1e-12) for k in range(iou_gt.shape[0]) for b in range(iou_gt.shape[1]) if n_gt[k, b] > 0]
    iou_c = pooled_c1["iou"]
    neg = [float(iou_c[k, b]) for k in range(iou_c.shape[0]) for b in range(iou_c.shape[1]) if n_gt[k, b] > 0 and k != 1]
    out["map"] = {"n_cells_with_gt": len(ok), "n_cells_iou_1": int(sum(ok)), "pass_gt_as_prediction": bool(ok and all(ok)),
                  "negative_control_const_class1_other_classes_iou_max": max(neg) if neg else None,
                  "pass_negative_control": bool(neg and max(neg) < 1.0)}
    out["pass"] = bool(out["box"]["pass"] and out["map"]["pass_gt_as_prediction"] and out["map"]["pass_negative_control"])
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({"pass": out["pass"], "box": {k: out["box"][k] for k in ("n_windows", "n_pos", "n_ap_cells_defined", "n_cells_not_1", "pass")},
                      "map": out["map"]}))


if __name__ == "__main__":
    sys.exit(main())
