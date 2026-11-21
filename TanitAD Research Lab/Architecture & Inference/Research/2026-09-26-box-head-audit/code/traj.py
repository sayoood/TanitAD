#!/usr/bin/env python3
"""traj.py -- box-head audit task 4: how the box terms evolved in refcv6's own metrics.jsonl, and what was logged.

Reads the run's metrics.jsonl (4,621 rows; train rows every 50 steps carry the box keys, eval rows every 500).
Reports, per step band: box3d / agent presence, cls, centre, size, yaw, z, h; the box3d total's share of the
training loss; targets per labelled window; and the exchangeable-slot reference loss for the observed K
(analytic, NO_OBJECT_W = 0.1, 100 slots). Plus a census of every box-related key: which quality signals
(precision, recall, AP, confident-slot counts, calibration) were EVER logged.
"""
import json
import math
import sys

import numpy as np

W0, N = 0.1, 100


def exch_loss(K):
    """Mean per-slot weighted BCE when all 100 slots share p* = pi/(pi + w0(1-pi)), pi = K/N (the loss's
    reduction='mean' over B x N)."""
    pi = min(max(K / N, 1e-9), 1 - 1e-9)
    p = pi / (pi + W0 * (1 - pi))
    return -(pi * math.log(p) + W0 * (1 - pi) * math.log(1 - p))


def prior_loss(K, p=0.05):
    pi = K / N
    return -(pi * math.log(p) + W0 * (1 - pi) * math.log(1 - p))


def main():
    path = sys.argv[1]
    out = sys.argv[2]
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    keys = set()
    for r in rows:
        keys |= set(r)
    boxkeys = sorted(k for k in keys if any(s in k for s in ("box3d", "agent_", "presence", "slot", "prec", "recall",
                                                             "_ap", "conf", "calib")))
    tr = [r for r in rows if "box3d_presence" in r and "eval_loss" not in r]
    ev = [r for r in rows if "eval_box3d_presence" in r]
    res = {"n_rows": len(rows), "n_train_rows_with_box": len(tr), "n_eval_rows": len(ev),
           "box_related_keys": boxkeys,
           "quality_signals_logged": [k for k in boxkeys if any(s in k for s in ("prec", "recall", "_ap", "conf",
                                                                                  "calib", "tp", "fp"))],
           "loss_key_present": "loss" in keys}
    bands = ((0, 200), (200, 1000), (1000, 5000), (5000, 20000), (20000, 34500), (34500, 38300))
    tb = []
    for lo, hi in bands:
        sel = [r for r in tr if lo <= r["step"] < hi]
        if not sel:
            continue
        def med(k):
            v = [r[k] for r in sel if k in r and r[k] is not None]
            return float(np.median(v)) if v else None
        K = [r["box3d_n_target"] / max(r["box3d_n_labelled"], 1) for r in sel if r.get("box3d_n_labelled")]
        Km = float(np.median(K)) if K else None
        share = [r["box3d"] / r["loss"] for r in sel if r.get("loss") and "box3d" in r]
        ag_sum = [sum(r.get(f"agent_{t}", 0.0) for t in ("presence", "cls", "centre", "size", "yaw")) / r["loss"]
                  for r in sel if r.get("loss")]
        tb.append({"steps": f"{lo}-{hi}", "n_rows": len(sel),
                   **{k: med(k) for k in ("box3d_presence", "box3d_cls", "box3d_centre", "box3d_size", "box3d_yaw",
                                          "box3d_z", "box3d_h", "box3d_rates", "box3d", "agent_presence",
                                          "agent_cls", "agent_centre", "agent_size", "agent_yaw", "loss")},
                   "targets_per_labelled_window_median": Km,
                   "exchangeable_presence_loss_at_K": exch_loss(Km) if Km else None,
                   "prior_init_presence_loss_at_K": prior_loss(Km) if Km else None,
                   "box3d_share_of_loss_median": float(np.median(share)) if share else None,
                   "agent_logged_terms_share_of_loss_median": float(np.median(ag_sum)) if ag_sum else None})
    res["train_bands"] = tb
    first = [{"step": r["step"], "box3d_presence": r.get("box3d_presence"), "agent_presence": r.get("agent_presence"),
              "box3d_n_target": r.get("box3d_n_target"), "box3d_n_labelled": r.get("box3d_n_labelled"),
              "box3d_centre": r.get("box3d_centre")} for r in tr[:12]]
    res["first_train_rows"] = first
    res["eval_rows"] = [{"step": r["step"], "presence": r.get("eval_box3d_presence"), "cls": r.get("eval_box3d_cls"),
                         "centre": r.get("eval_box3d_centre"), "size": r.get("eval_box3d_size"),
                         "z": r.get("eval_box3d_z"), "agent_presence": r.get("eval_agent_presence"),
                         "agent_centre": r.get("eval_agent_centre"), "n_target": r.get("eval_box3d_n_target"),
                         "n_labelled": r.get("eval_box3d_n_labelled")} for r in ev]
    # prefilter vs visible (what the field cut removed, from the run's own counts)
    pre = sum(r.get("box3d_n_target_prefilter", 0) for r in tr)
    vis = sum(r.get("box3d_n_target_visible", 0) for r in tr)
    res["field_cut_from_train_rows"] = {"prefilter": pre, "visible": vis, "removed_frac": 1 - vis / pre if pre else None}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("eval_rows", "box_related_keys")}, indent=1)[:7000])
    print("eval presence trajectory:", [(e["step"], e["presence"]) for e in res["eval_rows"][::6]])


if __name__ == "__main__":
    main()
