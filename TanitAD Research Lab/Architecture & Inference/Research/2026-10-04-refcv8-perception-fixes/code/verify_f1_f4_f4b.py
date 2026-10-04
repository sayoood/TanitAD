"""WP-C item 2: LOAD the three shipped opt-in files with the stack's own loaders and REPRODUCE the diagnostics' numbers from the banked
50,400-checkpoint outputs (CPU only; no clip id is read or written).

  F1   shipped class thresholds applied as ``score >= tau`` to the banked EVAL-DIAG score histogram -> EVAL IoU per class, vs RESULT.md M-c
  F4   shipped gates applied to the banked EVAL-DIAG packs (``eval_final.packs.pkl``)              -> conf_ratio / F1, vs raw/B_box.json
  F4b  shipped NMS + gate applied to the SAME diagnostics packs and to the route package's EVAL packs -> AP@2 m / F1 / boxes per object

Run:  python verify_f1_f4_f4b.py --diag-bin D:/refcv7_diag_bin/2026-10-04 --route-bin D:/refcv7_route_bin/2026-10-04 \
          --diag-pkg <2026-10-04-refcv7-map-box-diagnostics> --route-pkg <2026-10-04-refcv7-route-following> --out raw/f1_f4_f4b_reproduction.json
"""
import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from tanitad.eval import detection_metrics as det
from tanitad.eval import detection_nms as nms
from tanitad.models import map_head_hires as H


def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-bin", required=True)
    ap.add_argument("--route-bin", required=True)
    ap.add_argument("--diag-pkg", required=True)
    ap.add_argument("--route-pkg", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfgdir = Path(H.__file__).resolve().parent.parent / "configs"
    out = {}
    # ---------------------------------------------------------------------------------------- F1
    dm = _import(Path(a.diag_pkg) / "code" / "diag_metrics.py", "diag_metrics_v")
    th, st = H.load_class_thresholds(cfgdir / "refcv7_map_hires_class_thresholds_train.json")
    e = torch.load(Path(a.diag_bin) / "eval_final.acc.pt", map_location="cpu", weights_only=False)
    hist = e["hist"]["phat"]
    width = (dm.HIST_HI - dm.HIST_LO) / dm.HIST_NB
    m_c = json.loads((Path(a.diag_pkg) / "raw" / "M_c.json").read_text(encoding="utf-8"))
    f1 = {}
    for c, name in enumerate(H.CLASS_KEYS):
        h = hist[c].sum(dim=0).numpy()
        i = int(round((th[c].item() - dm.HIST_LO) / width))
        iou, pred, _inter, gt = dm.iou_at_index(h[0], h[1], i)
        f1[name] = {"tau_logit": round(float(th[c]), 4), "eval_iou_reproduced": round(iou, 4), "pred_over_gt": round(pred / gt, 4)}
    for name in H.CLASS_KEYS:
        ref = m_c["eval"]["thr_phat"][name]["all"]
        f1[name]["diagnostics_M_c_thr_phat_iou"] = round(ref["iou"], 4)
        f1[name]["diagnostics_pred_over_gt"] = round(ref["pred_over_gt"], 4)
        f1[name]["iou_abs_diff"] = abs(f1[name]["eval_iou_reproduced"] - f1[name]["diagnostics_M_c_thr_phat_iou"])
    out["F1_map_thresholds"] = {"loaded_sha256": st["sha256"], "classes": f1}
    # ---------------------------------------------------------------------------------------- F4
    gates, gst = det.load_head_gates(cfgdir / "refcv7_det_presence_gates_train.json")
    with open(Path(a.diag_bin) / "eval_final.packs.pkl", "rb") as fh:
        dpk = pickle.load(fh)
    b_box = json.loads((Path(a.diag_pkg) / "raw" / "B_box.json").read_text(encoding="utf-8"))
    f4 = {}
    for hd in det.HEADS:
        g = det.gated_census_keys(dpk[hd], hd, gates)
        t0 = det.train_row_keys(dpk[hd], hd)
        f4[hd] = {"gate": gates[hd], "declared_0p5": {"n_conf": t0[f"{hd}_n_conf"], "conf_ratio": round(t0[f"{hd}_conf_ratio"], 5)},
                  "at_gate": {k.split("gated_")[1]: (round(v, 5) if v == v else None) for k, v in g.items()}}
    out["F4_presence_gates"] = {"loaded_sha256": gst["sha256"], "heads": f4,
                                "diagnostics_B_box_eval_T3": {hd: {k: b_box[hd]["eval"]["transforms"]["T3"].get(k) for k in ("n_conf", "tp", "conf_ratio", "f1")}
                                                              for hd in det.HEADS}}
    # ---------------------------------------------------------------------------------------- F4b
    cfg, nst = nms.load_head_nms(cfgdir / "refcv7_det_nms_train.json")
    with open(Path(a.route_bin) / "eval_s0.packs.pkl", "rb") as fh:
        rpk = pickle.load(fh)
    box_nms = json.loads((Path(a.route_pkg) / "raw" / "box_nms.json").read_text(encoding="utf-8"))
    f4b = {}
    for label, src in (("route_eval_pass", rpk), ("diagnostics_eval_pass", dpk)):
        for hd in det.HEADS:
            P = src[hd]
            Q = nms.nms_packs(P, cfg[hd]["radius_m"], cfg[hd]["p_floor"])
            k = nms.nms_census_keys(P, hd, cfg)
            d0, d1 = nms.duplicate_stats(P, 0.2589), nms.duplicate_stats(Q, 0.2589)
            f4b.setdefault(label, {})[hd] = {
                "n_windows": len(P), "radius_m": cfg[hd]["radius_m"], "gate": round(cfg[hd]["gate"], 6),
                "ap2m_no_nms": round(nms.ap2m(P), 4), "ap2m_nms": round(k[f"eval_{hd}_nms_ap2m"], 4),
                "n_conf": k[f"eval_{hd}_nms_n_conf"], "tp": k[f"eval_{hd}_nms_tp"], "f1": round(k[f"eval_{hd}_nms_f1"], 4),
                "conf_ratio": round(k[f"eval_{hd}_nms_conf_ratio"], 4), "conf_ratio_alarm": k[f"eval_{hd}_nms_conf_ratio_alarm"],
                "boxes_per_object_at_0p2589": [round(d0["boxes_per_object"], 4), round(d1["boxes_per_object"], 4)],
                "conf_ratio_at_old_gate_0p2589": round(nms._census(Q, 0.2589)["conf_ratio"], 4)}
    out["F4b_nms"] = {"loaded_sha256": nst["sha256"], "reproduced": f4b,
                      "route_box_nms_json": {hd: {"AP2m_no_nms": box_nms[hd]["eval"]["no_NMS"]["AP2m"],
                                                  "AP2m_nms": [v["AP2m"] for kk, v in box_nms[hd]["eval"].items() if kk.startswith("centre")][0],
                                                  "gate_fit": [v["gate_fit"] for kk, v in box_nms[hd]["eval"].items() if kk.startswith("centre")][0]}
                                             for hd in det.HEADS}}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
