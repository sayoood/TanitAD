"""RAW vs EMA in-run eval trajectories (T0, eval139 --eval-batches 8) for the E2 runs. Read-only.
Writes traj.json beside itself and prints tables."""
import json, sys, os

RUNS = {"K0": "thor/r9e2-K0/metrics.jsonl", "K2M1N20": "thor/r9e2-K2M1N20/metrics.jsonl",
        "K2DC": "thor/r9e2-K2DC/metrics.jsonl"}
BASE = os.path.dirname(os.path.abspath(__file__))


def rows(p):
    out = []
    for line in open(os.path.join(BASE, p), encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def iou(r, pre, ch, band):
    i = r.get("%smap_banded_inter_%s_%s" % (pre, ch, band))
    u = r.get("%smap_banded_union_%s_%s" % (pre, ch, band))
    if i is None or u in (None, 0):
        return None
    return i / u


def f(x, nd=4):
    return "  -   " if x is None else ("%.*f" % (nd, x))


res = {}
for arm, p in RUNS.items():
    R = rows(p)
    ev = [r for r in R if "eval_loss" in r or "ema_eval_loss" in r]
    tr = [r for r in R if "lr" in r and "eval_loss" not in r]
    res[arm] = {"eval": [], "train": []}
    print("\n######## %s  (%d eval rows, %d train rows)" % (arm, len(ev), len(tr)))
    banded = any("eval_map_banded_inter_drv_near" in r for r in ev)
    if banded:
        print(" step | IoU drv_near raw/ema | IoU sdw_near raw/ema | IoU edge_h_near raw/ema | IoU sdw_mid raw/ema | "
              "L drv_near raw/ema | L sdw_near raw/ema | L edge_near raw/ema | eval_loss raw/ema")
        for r in ev:
            s = r.get("step")
            d = {"step": s}
            for ch, band in (("drv", "near"), ("sdw", "near"), ("edge_h", "near"), ("sdw", "mid"), ("drv", "mid"),
                             ("edge_h", "mid"), ("lane", "near")):
                d["iou_%s_%s_raw" % (ch, band)] = iou(r, "eval_", ch, band)
                d["iou_%s_%s_ema" % (ch, band)] = iou(r, "ema_eval_", ch, band)
            for ch in ("drv", "sdw", "edge_h", "drv_sdf"):
                for band in ("near", "mid"):
                    d["L_%s_%s_raw" % (ch, band)] = r.get("eval_map_banded_loss_%s_%s" % (ch, band))
                    d["L_%s_%s_ema" % (ch, band)] = r.get("ema_eval_map_banded_loss_%s_%s" % (ch, band))
            d["eval_loss_raw"] = r.get("eval_loss")
            d["eval_loss_ema"] = r.get("ema_eval_loss")
            d["map_hires_raw"] = r.get("eval_map_hires")
            d["map_hires_ema"] = r.get("ema_eval_map_hires")
            res[arm]["eval"].append(d)
            print(" %5s | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s" % (
                s, f(d["iou_drv_near_raw"]), f(d["iou_drv_near_ema"]), f(d["iou_sdw_near_raw"]), f(d["iou_sdw_near_ema"]),
                f(d["iou_edge_h_near_raw"]), f(d["iou_edge_h_near_ema"]), f(d["iou_sdw_mid_raw"]), f(d["iou_sdw_mid_ema"]),
                f(d["L_drv_near_raw"]), f(d["L_drv_near_ema"]), f(d["L_sdw_near_raw"]), f(d["L_sdw_near_ema"]),
                f(d["L_edge_h_near_raw"]), f(d["L_edge_h_near_ema"]), f(d["eval_loss_raw"], 1), f(d["eval_loss_ema"], 1)))
    else:
        print(" step | IoU sidewalk_0_20 raw/ema | IoU drivable_0_20 raw/ema | IoU edge_0_20 raw/ema | map_hires loss raw/ema | eval_loss raw/ema")
        for r in ev:
            s = r.get("step")
            d = {"step": s}
            for c in ("sidewalk", "drivable", "edge"):
                for b in ("0_20", "20_40"):
                    d["iou_%s_%s_raw" % (c, b)] = r.get("eval_map_hires_iou_%s_%s" % (c, b))
                    d["iou_%s_%s_ema" % (c, b)] = r.get("ema_eval_map_hires_iou_%s_%s" % (c, b))
            d["map_hires_raw"] = r.get("eval_map_hires")
            d["map_hires_ema"] = r.get("ema_eval_map_hires")
            d["eval_loss_raw"] = r.get("eval_loss")
            d["eval_loss_ema"] = r.get("ema_eval_loss")
            res[arm]["eval"].append(d)
            print(" %5s | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s" % (
                s, f(d["iou_sidewalk_0_20_raw"]), f(d["iou_sidewalk_0_20_ema"]), f(d["iou_drivable_0_20_raw"]),
                f(d["iou_drivable_0_20_ema"]), f(d["iou_edge_0_20_raw"]), f(d["iou_edge_0_20_ema"]),
                f(d["map_hires_raw"]), f(d["map_hires_ema"]), f(d["eval_loss_raw"], 1), f(d["eval_loss_ema"], 1)))
    # training map loss, mean over each 500-step block
    print(" train blocks (mean over rows in (s-500, s]):  lr | map_hires(train) | L drv_near | L sdw_near | L edge_near | dc_share_read")
    for hi in range(500, 6001, 500):
        blk = [r for r in tr if hi - 500 < (r.get("step") or 0) <= hi]
        def m(k):
            v = [r[k] for r in blk if r.get(k) is not None]
            return sum(v) / len(v) if v else None
        dd = {"step": hi, "lr": m("lr"), "map_hires": m("map_hires"), "L_drv_near": m("map_banded_loss_drv_near"),
              "L_sdw_near": m("map_banded_loss_sdw_near"), "L_edge_h_near": m("map_banded_loss_edge_h_near"),
              "dc_share_read": m("dc_share_read"), "n": len(blk)}
        res[arm]["train"].append(dd)
        print("  %5d | %s | %s | %s | %s | %s | %s" % (hi, f(dd["lr"], 7), f(dd["map_hires"]), f(dd["L_drv_near"]),
                                                     f(dd["L_sdw_near"]), f(dd["L_edge_h_near"]), f(dd["dc_share_read"])))

json.dump(res, open(os.path.join(BASE, "traj.json"), "w"), indent=1)
