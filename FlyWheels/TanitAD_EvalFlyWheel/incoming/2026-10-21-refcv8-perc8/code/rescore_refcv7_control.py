"""rescore_refcv7_control.py -- the control 'RE-SCORING refcv7's BANKED DUMP REPRODUCES ITS BANKED NUMBERS':
box: this instrument (perc8_rows + perc8_box_read, run with refcv7 as BOTH arms) vs `perc_score.json` (the refcv7 battery's own scorer);
map: the programme's `MH.summarize` on the banked WindowTable vs the banked `perc_score.json` map summary.
python rescore_refcv7_control.py --self-box <refcv7-vs-itself box read json> --banked <perc_score.json> --map-read <map_read.json> --out <json>"""
import argparse, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-box", required=True)
    ap.add_argument("--banked", required=True)
    ap.add_argument("--map-read", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = json.load(open(a.self_box))
    b = json.load(open(a.banked))
    bb = b["box"]["refcv7"]
    A = s["arms"]["a"]
    C = s["ap_auroc_cells"]
    cells = {}
    for t in ("0p5", "1", "2", "4"):
        cells[f"ap{t}_all"] = (C[f"ap{t}_all"][A]["mean"], bb[f"eval_box3d_det_ap{t}_all_all"])
        cells[f"map{t}"] = (C[f"map{t}"][A]["mean"], bb[f"eval_box3d_det_map{t}_all"])
    for i, bn in enumerate(("0_20", "20_40", "40_60")):
        cells[f"ap2_band{i}"] = (C[f"ap2_band{i}"][A]["mean"], bb[f"eval_box3d_det_ap2_all_{bn}"])
    for i, c in enumerate(s["classes"]):
        cells[f"ap2_cls{i}"] = (C[f"ap2_cls{i}"][A]["mean"], bb[f"eval_box3d_det_ap2_{c}_all"])
    cells["auroc_matched"] = (C["auroc_matched"][A]["mean"], bb["eval_box3d_auroc_matched"])
    cells["auroc_objectness"] = (C["auroc_objectness"][A]["mean"], bb["eval_box3d_auroc_objectness"])
    d = {k: abs(v[0] - v[1]) for k, v in cells.items()}
    out = {"box_cells": {k: {"mine": v[0], "banked": v[1], "abs_diff": d[k]} for k, v in cells.items()},
           "box_max_abs_diff": max(d.values()), "box_ap2m_mine": cells["ap2_all"][0], "box_ap2m_banked": cells["ap2_all"][1],
           "box_pass_1e-3": bool(max(d.values()) < 1e-3)}
    m = json.load(open(a.map_read))
    mb = b["map"]["summary"]["arms"]["refcv7"]
    ms = m["summary"]["arms"][m["arms"]["b"]]
    md = []
    for k, v in mb.items():
        if k.endswith("|iou") and v.get("mean") is not None and ms.get(k, {}).get("mean") is not None:
            md.append(abs(v["mean"] - ms[k]["mean"]))
    out["map_n_iou_cells"] = len(md)
    out["map_max_abs_diff_iou"] = max(md) if md else None
    out["map_pass_1e-3"] = bool(md and max(md) < 1e-3)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("box_ap2m_mine", "box_ap2m_banked", "box_max_abs_diff", "box_pass_1e-3", "map_n_iou_cells", "map_max_abs_diff_iou", "map_pass_1e-3")}))


if __name__ == "__main__":
    sys.exit(main())
