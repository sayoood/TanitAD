#!/usr/bin/env python
"""plumb_read.py -- controls p4, p5 and p7 of card PREREG_V8F_PROOF_ADD40 (note N1) from the logs of one checkpoint. Written by the lead. Reads no unsafe rate.

p4: the candidate actions of the logged future path (c = 1.0, definition logged) against the corpus' logged actions, full window set (imag_plan.p4_stats; bars of the card).
p5: at the first tick, the tool's imagined gap under the logged actions against the banked read of imag_dump.py for the same (clip, frame) and checkpoint.
    Matching control in the same breath: the head's PRESENT gap read of the tool (imag log, k = 0) against the banked present read on the matched rows.
p7: counts (windows, ticks, each (window, tick) once) and, with --cv-log, the active sets of cand-cv and cand-imag are the same set.
"""
import argparse
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imag_plan as IP  # noqa: E402


def stats(d):
    d = np.abs(np.asarray(d, np.float64))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n": 0}
    return {"n": int(d.size), "max": float(d.max()), "p99": float(np.percentile(d, 99)), "median": float(np.median(d)), "mean": float(d.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p45", required=True)
    ap.add_argument("--imag-log", required=True)
    ap.add_argument("--cv-log", default="")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = np.load(a.p45, allow_pickle=True)
    L = np.load(a.imag_log, allow_pickle=True)
    D = np.load(a.dump, allow_pickle=True)
    G = json.load(io.open(a.grid, encoding="utf-8"))
    order = list(G["order"])
    res = {"card": "PREREG_V8F_PROOF_ADD40 + note N1", "files": {k: os.path.basename(v) for k, v in vars(a).items() if v}}

    # p4
    p4 = IP.p4_stats(P["act_gt"], P["act_cache"])
    p4["PASS"] = bool(p4["corr_acc"] >= 0.99 and p4["corr_yaw"] >= 0.99 and p4["median_abs_diff_acc"] <= 0.1)
    acc_c, acc_l = np.asarray(P["act_gt"], np.float64)[..., 0], np.asarray(P["act_cache"], np.float64)[..., 0]
    ok = np.isfinite(acc_c).all(1) & np.isfinite(acc_l).all(1)
    p4["beside"] = {
        "corr_acc_window_mean": float(np.corrcoef(acc_c[ok].mean(1), acc_l[ok].mean(1))[0, 1]),
        "std_acc_logged": float(acc_l[ok].std()), "std_acc_candidate": float(acc_c[ok].std()),
        "rms_diff_acc": float(np.sqrt(np.mean((acc_c[ok] - acc_l[ok]) ** 2))),
        "p90_abs_diff_acc": float(np.percentile(np.abs(acc_c[ok] - acc_l[ok]), 90)),
        "note": "acceleration in the units of the log (see scale); window mean = mean over the 10 steps of a window",
    }
    res["p4"] = p4

    # p5: match (eid, frame)
    key_dump = {(str(e), int(t)): i for i, (e, t) in enumerate(zip(D["eid"], D["t"]))}
    ci, t0 = np.asarray(P["ci"]).astype(int), np.asarray(P["t0"]).astype(int)
    ia, ib = [], []
    for r, (c, t) in enumerate(zip(ci, t0)):
        j = key_dump.get((order[c], int(t)))
        if j is not None:
            ia.append(r)
            ib.append(j)
    ia, ib = np.asarray(ia, int), np.asarray(ib, int)
    p5 = {"n_dump_rows": int(len(D["t"])), "n_tool_windows": int(len(ci)), "n_matched": int(len(ia))}
    if len(ia):
        tool = np.asarray(P["imag_gap_logged"], np.float64)[ia]            # [n, 2] gaps at 1.0 s and 2.0 s
        bank = np.asarray(D["roll_logged"], np.float64)[ib][:, :, 1]       # [n, 2] gap channel
        p5["imag_T1.0"] = stats(tool[:, 0] - bank[:, 0])
        p5["imag_T2.0"] = stats(tool[:, 1] - bank[:, 1])
        p5["corr_T2.0"] = float(np.corrcoef(tool[:, 1], bank[:, 1])[0, 1])
        # matching control: present read of the tool at k = 0 against the banked present read
        k0 = np.asarray(L["k"]).astype(int) == 0
        lk = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(np.asarray(L["ci"])[k0], np.asarray(L["t0"])[k0]))}
        gs = np.asarray(L["g_src"], np.float64)[k0]
        rows = [lk.get((int(c), int(t))) for c, t in zip(ci[ia], t0[ia])]
        sel = np.asarray([r is not None for r in rows])
        gtool = np.asarray([gs[r] for r in rows if r is not None], np.float64)
        gbank = np.asarray(D["present"], np.float64)[ib][sel][:, 1]
        p5["present_control"] = stats(gtool - gbank)
        p5["bar_m"] = 1e-4
        p5["PASS_as_written"] = bool(max(p5["imag_T1.0"].get("max", np.inf), p5["imag_T2.0"].get("max", np.inf)) <= 1e-4)
    res["p5"] = p5

    # p7
    lci, lt0, lk_ = np.asarray(L["ci"]).astype(int), np.asarray(L["t0"]).astype(int), np.asarray(L["k"]).astype(int)
    trip = set(zip(lci.tolist(), lt0.tolist(), lk_.tolist()))
    wins = set(zip(lci.tolist(), lt0.tolist()))
    p7 = {"rows": int(len(lci)), "unique_window_ticks": len(trip), "windows": len(wins), "windows_grid": int(G["n_windows"]),
          "ticks_per_window": sorted(set(np.bincount(np.unique(np.stack([lci, lt0], 1), axis=0, return_inverse=True)[1].ravel()).tolist())),
          "tick_steps": sorted(set(lk_.tolist())), "active_rows": int(np.asarray(L["active"]).astype(bool).sum()),
          "windows_p45": int(len(ci))}
    p7["each_once"] = bool(p7["rows"] == p7["unique_window_ticks"])
    if a.cv_log:
        C = np.load(a.cv_log, allow_pickle=True)
        same_rows = bool(len(C["ci"]) == len(lci) and (np.asarray(C["ci"]) == L["ci"]).all() and (np.asarray(C["t0"]) == L["t0"]).all() and (np.asarray(C["k"]) == L["k"]).all())
        # the rolled track differs between modes after the first tick, so the active set is compared at k = 0 (same state) and reported for all ticks
        k0 = lk_ == 0
        p7["cv_rows_same_order"] = same_rows
        p7["active_equal_at_tick0"] = bool(same_rows and (np.asarray(C["active"])[k0] == np.asarray(L["active"])[k0]).all())
        p7["active_equal_all_ticks"] = bool(same_rows and (np.asarray(C["active"]) == np.asarray(L["active"])).all())
        p7["active_rows_cv"] = int(np.asarray(C["active"]).astype(bool).sum())
    p7["PASS_counts"] = bool(p7["windows"] == p7["windows_grid"] and p7["each_once"])
    res["p7"] = p7
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    print("p4", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p4.items() if k != "beside"}))
    print("p5", json.dumps({k: ({kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()} if isinstance(v, dict) else (round(v, 5) if isinstance(v, float) else v)) for k, v in p5.items()}))
    print("p7", json.dumps(p7))


if __name__ == "__main__":
    main()
