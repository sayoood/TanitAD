"""Summarise raw/measure_class_signal_38k.json into the task-3 tables (raw/measure_class_signal_38k_summary.json).

Pooling rules (declared):
  * loss share of class c      = sum_windows (L_c * n) / sum_windows (L * n)   (cell-weighted, = the pooled loss)
  * gradient share at a tensor = mean over windows of the per-window projection share (each window's
    shares sum to 1); also the cell-weighted mean. Both printed.
  * IoU                         = sum inter / sum union over windows (pooled), from the per-class argmax
                                  cell counts the probe records (n_cells_argmax_label / _pred, iou_argmax);
                                  recomputed from counts where available.
  * information gain of class c = CE_c(shuffled GT) - CE_c(true GT), each CE_c = sum (L_c n) / sum (mass_c n)
                                  (nats per unit of class-c label mass). > 0 means the head knows WHERE class c
                                  is in THIS scene beyond what any scene's labels would give it.
Groups: neutral (8 windows) and enriched (8 windows) separately, and all 16.
"""
import json
import math
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "raw" / "thor_2358"
C9 = ["seen-no-class", "drivable", "lane/road line", "crosswalk", "arrow/text", "non-drivable edge",
      "hatched", "sidewalk/verge", "not-seen"]
THIN = [2, 3, 4, 5, 6]


def pool(rows):
    out = {}
    tot_loss = sum(r["loss"] * r["n_cells"] for r in rows)
    for c in range(9):
        pc = [r["per_class"][c] for r in rows]
        lc = sum(p["loss_c"] * r["n_cells"] for p, r in zip(pc, rows))
        mass = sum(p["mass"] * r["n_cells"] for p, r in zip(pc, rows))
        ncell = sum(r["n_cells"] for r in rows)
        d = {"label_mass_share": mass / ncell,
             "loss_share": lc / tot_loss,
             "ce_per_unit_mass": (lc / mass) if mass > 0 else None}
        for key in ("grad_logits_proj_share", "grad_lift_out_proj_share", "grad_trunk_fmap_s16_proj_share",
                    "attrib_A_share", "own_pull_P_share"):
            v = [p.get(key) for p in pc if p.get(key) is not None]
            w = [r["n_cells"] for p, r in zip(pc, rows) if p.get(key) is not None]
            d[key + "_mean"] = sum(v) / len(v) if v else None
            d[key + "_cellw"] = sum(a * b for a, b in zip(v, w)) / sum(w) if v else None
        for key in ("grad_logits_norm", "grad_lift_out_norm", "grad_trunk_fmap_s16_norm",
                    "total_grad_on_logit_channel_c_norm"):
            v = [p[key] for p in pc]
            d[key + "_mean"] = sum(v) / len(v)
        nl = sum(p["n_cells_argmax_label"] for p in pc)
        npr = sum(p["n_cells_argmax_pred"] for p in pc)
        # pooled IoU from per-window IoU and counts: inter = iou*union; union = nl + npr - inter
        inter = 0.0
        union = 0.0
        for p in pc:
            if p["iou_argmax"] is None:
                continue
            u_w = None
            # iou = I / (L + P - I)  ->  I = iou (L + P) / (1 + iou)
            i_w = p["iou_argmax"] * (p["n_cells_argmax_label"] + p["n_cells_argmax_pred"]) / (1 + p["iou_argmax"])
            inter += i_w
            union += p["n_cells_argmax_label"] + p["n_cells_argmax_pred"] - i_w
        d["iou_pooled"] = inter / union if union > 0 else None
        d["n_label_argmax_cells"] = nl
        d["n_pred_argmax_cells"] = npr
        qv = [(p["mean_q_where_p_ge_0.5"], p["n_cells_p_ge_0.5"]) for p in pc if p["mean_q_where_p_ge_0.5"] is not None]
        d["mean_q_on_own_cells(p>=0.5)"] = (sum(a * b for a, b in qv) / sum(b for _, b in qv)) if qv else None
        d["n_cells_p_ge_0.5"] = sum(b for _, b in qv)
        q0 = [p["mean_q_where_p_0"] for p in pc if p["mean_q_where_p_0"] is not None]
        d["mean_q_where_absent(p=0)"] = sum(q0) / len(q0) if q0 else None
        d["max_q_any_window"] = max((p["max_q"] for p in pc if p["max_q"] is not None), default=None)
        out[C9[c]] = d
    thin = {k: sum(out[C9[c]][k] for c in THIN if out[C9[c]][k] is not None)
            for k in ("label_mass_share", "loss_share", "grad_logits_proj_share_cellw",
                      "grad_lift_out_proj_share_cellw", "grad_trunk_fmap_s16_proj_share_cellw",
                      "attrib_A_share_cellw", "own_pull_P_share_cellw")}
    return {"n_windows": len(rows), "n_cells": sum(r["n_cells"] for r in rows),
            "loss_cellw": tot_loss / sum(r["n_cells"] for r in rows), "per_class": out, "thin_sum": thin}


def main():
    rec = json.load(open(RAW / "measure_class_signal_38k.json", encoding="utf-8"))
    W = rec["windows"]
    S = rec["gt_shuffled"]
    res = {"controls": {}, "groups": {}}
    # controls
    ctl = {"rerun_logits_max_abs_diff_max": max(w["ctl_rerun_logits_max_abs_diff"] for w in W),
           "map_loss_row_abs_diff_max": max(w["ctl_map_loss_row_abs_diff"] for w in W),
           "uniform_loss_abs_err_max": max(w["ctl_constant"]["uniform"]["abs_err"] for w in W),
           "uniform_Lc_minus_mass_logC_max": max(w["ctl_constant"]["uniform"]["max_abs_Lc_minus_mass_logC"] for w in W),
           "uniform_grad_minus_analytic_max": max(w["ctl_constant"]["uniform"]["max_abs_grad_minus_analytic"] for w in W),
           "uniform_per_class_grad_norm_rel_err_max": max(w["ctl_constant"]["uniform"]["max_rel_err_per_class_grad_norm"] for w in W),
           "prior_loss_abs_err_max": max(w["ctl_constant"]["prior"]["abs_err"] for w in W),
           "gt_as_logits_minus_entropy_max": max(w["ctl_constant"]["gt_as_logits"]["abs_err"] for w in W)}
    res["controls"] = ctl
    for grp in ("neutral", "enriched", "all"):
        rows = [w for w in W if grp == "all" or w["group"] == grp]
        srows = [s for s in S if grp == "all" or s["group"] == grp]
        if not rows:
            continue
        g = pool(rows)
        gs = pool(srows) if srows else None
        if gs:
            g["gt_shuffled_loss_cellw"] = gs["loss_cellw"]
            for c in C9:
                a, b = g["per_class"][c]["ce_per_unit_mass"], gs["per_class"][c]["ce_per_unit_mass"]
                g["per_class"][c]["ce_per_unit_mass_shuffled"] = b
                g["per_class"][c]["info_gain_nats_per_unit_mass"] = (b - a) if (a is not None and b is not None) else None
        res["groups"][grp] = g
    json.dump(res, open(RAW / "measure_class_signal_38k_summary.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(res["controls"], indent=1))
    for grp, g in res["groups"].items():
        print(f"\n== {grp}: n_windows {g['n_windows']}, n_cells {g['n_cells']}, loss {g['loss_cellw']:.4f}, "
              f"shuffled {g.get('gt_shuffled_loss_cellw')}")
        print(f"{'class':18s} {'mass%':>7s} {'loss%':>7s} {'gZ%':>7s} {'gLift%':>7s} {'gTrunk%':>7s} {'A%':>7s} "
              f"{'IoU':>6s} {'q_own':>6s} {'q_abs':>6s} {'CE/m':>6s} {'CEsh/m':>6s} {'gain':>6s}")
        for c, d in g["per_class"].items():
            f = lambda v, s=100: ("   n/a" if v is None else f"{v * s:7.2f}")
            h = lambda v: ("   n/a" if v is None else f"{v:6.3f}")
            print(f"{c:18s} {f(d['label_mass_share'])} {f(d['loss_share'])} {f(d['grad_logits_proj_share_cellw'])} "
                  f"{f(d['grad_lift_out_proj_share_cellw'])} {f(d['grad_trunk_fmap_s16_proj_share_cellw'])} "
                  f"{f(d['attrib_A_share_cellw'])} {h(d['iou_pooled'])} {h(d['mean_q_on_own_cells(p>=0.5)'])} "
                  f"{h(d['mean_q_where_absent(p=0)'])} {h(d['ce_per_unit_mass'])} "
                  f"{h(d.get('ce_per_unit_mass_shuffled'))} {h(d.get('info_gain_nats_per_unit_mass'))}")
        print("thin sum:", {k: round(100 * v, 2) for k, v in g["thin_sum"].items()})


if __name__ == "__main__":
    main()
