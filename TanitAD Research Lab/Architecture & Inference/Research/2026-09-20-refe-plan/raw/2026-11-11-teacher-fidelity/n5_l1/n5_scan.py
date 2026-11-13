"""n5_scan.py - score the dual-labeller R labels of the RJ5 re-roll rows (finished batches only) with the builder scorer (tracked, W-TRANSP navtrain route),
and for every DAC = 1 row record its lateral-corridor geometry (for the CONSTRUCTION stress set). CPU only. Read-only on the RJ5 job's outputs.
Inputs : rows jsonl (finished-batch RJ5 rows, copied to scratch), labels dir (labels_s*of*.jsonl from the UNCHANGED p5c_dual_labels.py).
Outputs: scan.jsonl (one row per (log, token): base metrics, dual-label R vs cx.labels(t20)['R'] max deviation (C-DERIV, on every DAC = 0 row and on a
         sha1-fixed 5 % sample), and for DAC = 1: min clearance to the left / right boundary over steps 1..40, min interval width).
usage: python n5_scan.py ROWS LABELS_DIR OUT [--shard i/n]"""
import argparse
import glob
import hashlib
import json
import os

import n5_env  # noqa: F401
import numpy as np

import n4_geom as G
import n4_repair_l1 as L1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    rows = {(r["log_name"], r["token"]): r for r in (json.loads(x) for x in open(a.rows, encoding="utf-8"))}
    lab = {}
    for fp in sorted(glob.glob(os.path.join(a.labels, "labels_s*of*.jsonl"))):
        for ln in open(fp, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            lab[(r["log"], r["token"])] = r
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(x)["log"], json.loads(x)["token"]) for x in open(a.out, encoding="utf-8") if x.strip().endswith("}")}
    by = {}
    for k in sorted(rows):
        if k in lab and "p5r" in lab[k] and k not in done and int(hashlib.sha1(k[1].encode()).hexdigest(), 16) % sn == si:
            by.setdefault(k[0], []).append(k[1])
    with open(a.out, "a", encoding="utf-8") as fo:
        for lg, tk, cx in G.contexts(by):
            rec = {"log": lg, "token": tk}
            try:
                t20 = np.asarray(rows[(lg, tk)]["traj"], float)
                Rl = np.asarray(lab[(lg, tk)]["p5r"], float)
                res, sim = cx.score([Rl])
                b = res[0]
                rec["base"] = {k: float(b[k]) for k in ("dac", "nc", "ddc", "ttc", "c", "pdms")}
                rec["t20"] = t20.round(9).tolist()
                sample = int(hashlib.sha1(tk.encode()).hexdigest(), 16) % 20 == 0
                if b["dac"] == 0.0 or sample:
                    rec["cderiv_maxdev_m"] = float(np.abs(cx.labels(t20)["R"] - Rl).max())
                if b["dac"] == 1.0:
                    corners, heading = L1._corners_heading(cx, sim[0])
                    lo, hi, ne = L1.step_intervals(cx.drivable, corners, heading, L1.MARGIN)
                    if ne[1:].all():
                        rec["clear_left_m"] = float(np.min(hi[1:]))
                        rec["clear_right_m"] = float(np.min(-lo[1:]))
                        rec["width_min_m"] = float(np.min(hi[1:] - lo[1:]))
                        rec["clear_left_argmin_step"] = int(1 + np.argmin(hi[1:]))
                        rec["clear_right_argmin_step"] = int(1 + np.argmin(-lo[1:]))
                    else:
                        rec["tight_empty_at_margin"] = True
            except Exception as e:                                   # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            fo.write(json.dumps(rec) + "\n")
            fo.flush()


if __name__ == "__main__":
    main()
