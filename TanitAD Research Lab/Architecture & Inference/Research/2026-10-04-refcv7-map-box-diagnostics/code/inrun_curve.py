"""The run's OWN in-run eval curve (metrics.jsonl eval rows, every 500 steps, the fixed 128-window subset):
per class pooled IoU under the declared rule and the raw argmax (all bands and 0-20 m), N_pred/N_gt, and the box
census / calibration keys. Read-only on the run dir. Evidence class of every value: MEASURED in-run
(artifact: <run>/metrics.jsonl), 128 windows -- a monitor, not the diagnostic sample.
"""
import json
import sys

CK = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")
BANDS = ("0_20", "20_40", "40_60", "60_80", "80_100")
BOX = ("conf_ratio", "n_conf", "n_pos", "tp@gate", "ap2m", "auroc_matched", "calib_pr_gate", "calib_prec",
       "calib_rec", "prec@gate", "rec@gate", "presence_frac_confident")


def main(path, out):
    rows = []
    for line in open(path):
        r = json.loads(line)
        if "eval_map_hires_n_windows" not in r and "eval_box3d_n_windows" not in r:
            continue
        o = {"step": r.get("step")}
        for c in CK:
            for rule, i, u in (("pc", "inter", "union"), ("raw", "interraw", "unionraw")):
                I = sum(r.get(f"eval_map_hires_{i}_{c}_{b}", 0.0) or 0.0 for b in BANDS)
                U = sum(r.get(f"eval_map_hires_{u}_{c}_{b}", 0.0) or 0.0 for b in BANDS)
                N = sum(r.get(f"eval_map_hires_n_{c}_{b}", 0.0) or 0.0 for b in BANDS)
                o[f"{c}_iou_{rule}"] = I / U if U else None
                o[f"{c}_pred_over_gt_{rule}"] = (U - N + I) / N if N else None
                i0, u0 = r.get(f"eval_map_hires_{i}_{c}_0_20"), r.get(f"eval_map_hires_{u}_{c}_0_20")
                o[f"{c}_iou_{rule}_0_20"] = (i0 / u0) if (i0 is not None and u0) else None
        for h in ("box3d", "agent"):
            for k in BOX:
                o[f"{h}_{k}"] = r.get(f"eval_{h}_{k}")
        rows.append(o)
    json.dump({"source": path, "n_rows": len(rows), "evidence": "MEASURED in-run (metrics.jsonl), 128 windows",
               "rows": rows}, open(out, "w"), indent=0)
    print(f"[curve] {len(rows)} eval rows -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
