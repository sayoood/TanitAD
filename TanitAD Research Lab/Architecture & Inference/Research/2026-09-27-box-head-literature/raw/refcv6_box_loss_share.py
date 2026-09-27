"""refcv6-r101-s0: the detection losses' share of the joint loss, and the presence term's share of the box loss.

MEASURED from D:/refcv6_eval_kit/ckpt_final/metrics.jsonl (the run's own training rows, logged every 50 steps).
`box3d` is the weighted box3d total (verified: presence + cls + centre + size + yaw + 0.5*rates + 0.5*occ
+ z + h reproduces it on the last row to 4 dp). The AGENT head's total is not logged; its logged parts are
presence, cls, centre, size, yaw (rates/occ are in its total but never logged), so `agent_share_of_total`
is a LOWER bound. Medians per step band. Also: zero-drop check and the in-run eval keys.
"""
import json, re, statistics as st, sys
P = r"D:/refcv6_eval_kit/ckpt_final/metrics.jsonl"
rows, keys = [], set()
for line in open(P, encoding="utf-8"):
    try:
        d = json.loads(line)
    except Exception:
        continue
    keys.update(d.keys())
    if "loss" in d and "box3d" in d and "step" in d:
        rows.append(d)

def agent_lb(d):
    return sum(d.get(k, 0.0) or 0.0 for k in ("agent_presence", "agent_cls", "agent_centre", "agent_size", "agent_yaw"))

out = {"doc": __doc__.strip(), "n_train_rows": len(rows), "bands": {}}
for lo, hi in ((0, 1000), (1000, 5000), (5000, 15000), (15000, 25000), (25000, 34500), (34500, 38300)):
    R = [d for d in rows if lo < d["step"] <= hi]
    med = lambda f: round(st.median([f(d) for d in R]), 4)
    out["bands"][f"{lo}-{hi}"] = {
        "n_rows": len(R),
        "loss_total": med(lambda d: d["loss"]),
        "box3d_weighted_total": med(lambda d: d["box3d"]),
        "box3d_share_of_total": med(lambda d: d["box3d"] / d["loss"]),
        "agent_share_of_total_LOWER_BOUND": med(lambda d: agent_lb(d) / d["loss"]),
        "presence_share_of_box3d": med(lambda d: d["box3d_presence"] / d["box3d"]),
        "cls_share_of_box3d": med(lambda d: d["box3d_cls"] / d["box3d"]),
        "regression_share_of_box3d": med(lambda d: (d["box3d_centre"] + d["box3d_size"] + 0.5 * d["box3d_rates"]
                                                     + d["box3d_z"] + d["box3d_h"]) / d["box3d"]),
        "box3d_presence_nats": med(lambda d: d["box3d_presence"]),
        "box3d_centre_L1_m": med(lambda d: d["box3d_centre"]),
        "visible_targets_per_labelled_window": med(lambda d: d["box3d_n_target_visible"] / max(d["box3d_n_labelled"], 1)),
        "prefilter_targets_per_labelled_window": med(lambda d: d["box3d_n_target_prefilter"] / max(d["box3d_n_labelled"], 1)),
    }
out["query_budget"] = {"rows_with_box3d_drop": sum(1 for d in rows if d["box3d_n_dropped"] > 0),
                       "rows_with_agent_drop": sum(1 for d in rows if d.get("agent_n_dropped", 0) > 0),
                       "max_visible_targets_in_a_16_window_batch": max(d["box3d_n_target_visible"] for d in rows)}
_det = re.compile(r"(^|_)ap(\d|_|$)|prec|recall|auroc|confident|f1", re.I)
out["detection_metric_keys_logged"] = sorted(k for k in keys if _det.search(k) and "map" not in k.lower())
out["n_distinct_keys"] = len(keys)
json.dump(out, sys.stdout, indent=1)
