#!/usr/bin/env python
"""margin_calib.py -- card PREREG_V8F_PROOF_ADD30 (MARGIN), section 3: the two constants of the vision-fed distance rule, per checkpoint,
from the TRAIN probe clips only.  It reads no evaluation file (control g4: every path is checked and listed with its sha256).

  python margin_calib.py --arms LS_s0,LS_s1,LSD_s0,LSD_s1 --out F:/TanitAD-scratch/v8proof/distrule/margin/CALIB.json

Per arm: the banked head read on the train probe clips (<in-dir>/<ARM>_train_lead.npz), the row layout of the states file it was computed from
(<states-dir>/<ARM>_train.npz: clips, clip_idx, t) and the labels of the train pack (<pack>: clips, clip_idx, t, gap, pres), rows matched by clip NAME and frame.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

P_GRID = (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
RECALL_BAR = 0.95
NEAR_M = 40.0
ROOT = "F:/TanitAD-scratch/v8proof"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def up_half(x):
    """Not below 0, rounded up to the next 0.5 m."""
    return float(math.ceil(max(0.0, float(x)) * 2.0 - 1e-9) / 2.0)


def calib_arm(arm, in_dir, states_dir, pack, files):
    lead_p = os.path.join(in_dir, arm + "_train_lead.npz")
    st_p = os.path.join(states_dir, arm + "_train.npz")
    for p in (lead_p, st_p):
        if "_eval" in os.path.basename(p):
            raise SystemExit("REFUSED: an evaluation file in the calibration: " + p)
        files[p.replace("\\", "/")] = sha256_file(p)
    L = np.load(lead_p, allow_pickle=True)
    S = np.load(st_p, allow_pickle=True)
    s_names = np.asarray(S["clips"]).astype(str)[np.asarray(S["clip_idx"]).astype(np.int64)]
    s_t = np.asarray(S["t"]).astype(np.int64)
    n = s_t.size
    if L["gap_m"].shape[0] != n:
        raise SystemExit("REFUSED: %s head read rows %d != state rows %d" % (arm, L["gap_m"].shape[0], n))
    idx = pack["index"]
    rows = np.array([idx.get((nm, int(t)), -1) for nm, t in zip(s_names, s_t)], np.int64)
    ok = (rows >= 0) & np.asarray(L["has_history"]).astype(bool)
    r = rows[ok]
    gap_l = pack["gap"][r]
    pres_l = pack["pres"][r] == 1
    p = 1.0 / (1.0 + np.exp(-np.asarray(L["presence_logit"], np.float64)[ok]))
    g_h = np.asarray(L["gap_m"], np.float64)[ok]
    near = pres_l & np.isfinite(gap_l) & (gap_l <= NEAR_M)
    absent = ~pres_l
    rec = {str(q): float((p[near] >= q).mean()) for q in P_GRID}
    fa = {str(q): float((p[absent] >= q).mean()) for q in P_GRID}
    p_star = next((q for q in P_GRID if rec[str(q)] >= RECALL_BAR), P_GRID[-1])
    m = near & np.isfinite(g_h)
    over = g_h[m] - gap_l[m]
    q50, q80, q95 = (float(np.percentile(over, q)) for q in (50, 80, 95))
    return {"n_state_rows": int(n), "n_matched_with_history": int(ok.sum()), "n_near_lead": int(near.sum()), "n_label_absent": int(absent.sum()),
            "n_over_read_rows": int(m.sum()), "recall_by_P": rec, "label_absent_called_present_by_P": fa, "recall_bar_met": bool(rec[str(p_star)] >= RECALL_BAR),
            "P_star": float(p_star), "over_read_percentiles_m": {"p50": q50, "p80": q80, "p95": q95, "mean": float(over.mean()), "rmse": float(np.sqrt((over ** 2).mean()))},
            "M50": up_half(q50), "M80": up_half(q80), "M95": up_half(q95)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="LS_s0,LS_s1,LSD_s0,LSD_s1")
    ap.add_argument("--in-dir", default=ROOT + "/spatial/leadsup/read/in")
    ap.add_argument("--states-dir", default="C:/Users/Admin/v8proof_c/s1states")
    ap.add_argument("--pack", default=ROOT + "/spatial/leadsup/read/train_pack.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit("REFUSED: %s exists (the constants are fixed once)" % a.out)
    if "eval" in os.path.basename(a.pack):
        raise SystemExit("REFUSED: the pack must be the train pack")
    files = {a.pack.replace("\\", "/"): sha256_file(a.pack)}
    P = np.load(a.pack, allow_pickle=True)
    names = np.asarray(P["clips"]).astype(str)[np.asarray(P["clip_idx"]).astype(np.int64)]
    t = np.asarray(P["t"]).astype(np.int64)
    pack = {"index": {(nm, int(tt)): i for i, (nm, tt) in enumerate(zip(names, t))}, "gap": np.asarray(P["gap"], np.float64), "pres": np.asarray(P["pres"], np.float64)}
    if len(pack["index"]) != t.size:
        raise SystemExit("REFUSED: the train pack has repeated (clip, frame) rows")
    out = {"card": "PREREG_V8F_PROOF_ADD30", "P_grid": list(P_GRID), "recall_bar": RECALL_BAR, "near_m": NEAR_M, "arms": {}, "files_sha256": files}
    for arm in a.arms.split(","):
        out["arms"][arm] = calib_arm(arm, a.in_dir, a.states_dir, pack, files)
        r = out["arms"][arm]
        print("%-7s near-lead rows %d | recall at 0.5 %.4f | P* %.2f (recall %.4f, label-absent called present %.4f; at 0.5: %.4f) | over-read p50 %+.2f p80 %+.2f p95 %+.2f m -> M50 %.1f M80 %.1f M95 %.1f" % (
            arm, r["n_near_lead"], r["recall_by_P"]["0.5"], r["P_star"], r["recall_by_P"][str(r["P_star"])], r["label_absent_called_present_by_P"][str(r["P_star"])],
            r["label_absent_called_present_by_P"]["0.5"], r["over_read_percentiles_m"]["p50"], r["over_read_percentiles_m"]["p80"], r["over_read_percentiles_m"]["p95"], r["M50"], r["M80"], r["M95"]))
    bad = [p for p in files if "_eval" in os.path.basename(p)]
    out["g4_only_train_files"] = bool(not bad)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("wrote", a.out, "g4", out["g4_only_train_files"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
