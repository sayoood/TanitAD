#!/usr/bin/env python
"""replay_read.py -- note PREREG_V8F_PROOF_ADD40_N3 section 3 (written by the lead): where does the tool's imagined response differ from the reader's?

Three responses of the imagined gap to an ego acceleration change, all as a slope through the origin against the kinematic expectation (+1 = kinematics):
  R  the reader of card ADD36 (banked dump: stored states, logged actions + offset), on its own lead windows;
  T  the same stimulus through the TOOL's state path (p4/p5 log key imag_gap_logged_off: tick-0 state from the tokens, logged actions + offset);
  C  the tool's candidates (imag log: the plan's actions with the acceleration channel of candidate d), at tick 0.
R against T on matched windows (the tool window starts one frame = 0.2 s before the reader's present frame) says whether the state path differs;
T against C on the same windows says whether the baseline action sequence (logged against plan) or the candidate form makes the difference.
Reads no unsafe rate.
"""
import argparse
import io
import json
import os

import numpy as np

OFFS = (-2.0, -1.0, 1.0)


def slope0(d, e):
    d, e = np.asarray(d, np.float64).ravel(), np.asarray(e, np.float64).ravel()
    ok = np.isfinite(d) & np.isfinite(e)
    return float((d[ok] * e[ok]).sum() / (e[ok] * e[ok]).sum()) if ok.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p45", required=True)
    ap.add_argument("--imag-log", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = np.load(a.p45, allow_pickle=True)
    L = np.load(a.imag_log, allow_pickle=True)
    D = np.load(a.dump, allow_pickle=True)
    order = list(json.load(io.open(a.grid, encoding="utf-8"))["order"])
    res = {"files": {k: os.path.basename(str(v)) for k, v in vars(a).items()}}
    ci, t0 = np.asarray(P["ci"]).astype(int), np.asarray(P["t0"]).astype(int)
    gl = np.asarray(P["imag_gap_logged"], np.float64)                    # [n, 2]
    go = np.asarray(P["imag_gap_logged_off"], np.float64)                # [n, 3, 2]
    dT = go - gl[:, None, :]                                             # tool path, reader stimulus
    exp = np.stack([np.stack([-0.5 * d * T * T for T in (1.0, 2.0)]) for d in OFFS])      # [3, 2] kinematic expectation
    # reader
    doff = [float(x) for x in D["offsets"]]
    io_ = [doff.index(d) for d in OFFS]
    dR = np.asarray(D["roll_off"], np.float64)[..., 1][:, io_, :] - np.asarray(D["roll_logged"], np.float64)[..., 1][:, None, :]   # [m, 3, 2] (state index 1 = gap)
    res["R_reader_all_dump_windows"] = {"n": int(len(dR)), "slope_T1.0": slope0(dR[:, :, 0], np.broadcast_to(exp[:, 0], dR[:, :, 0].shape)),
                                        "slope_T2.0": slope0(dR[:, :, 1], np.broadcast_to(exp[:, 1], dR[:, :, 1].shape))}
    # tool path on every window, and on the windows the head calls present at tick 0
    k0 = np.asarray(L["k"]).astype(int) == 0
    key0 = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(np.asarray(L["ci"])[k0], np.asarray(L["t0"])[k0]))}
    act0 = np.asarray(L["active"]).astype(bool)[k0]
    rows0 = np.array([key0[(int(c), int(t))] for c, t in zip(ci, t0)])
    actw = act0[rows0]
    for nm, m in (("T_tool_path_all_windows", np.ones(len(ci), bool)), ("T_tool_path_active_windows", actw)):
        res[nm] = {"n": int(m.sum()), "slope_T1.0": slope0(dT[m][:, :, 0], np.broadcast_to(exp[:, 0], dT[m][:, :, 0].shape)),
                   "slope_T2.0": slope0(dT[m][:, :, 1], np.broadcast_to(exp[:, 1], dT[m][:, :, 1].shape))}
    # matched windows: tool window start = reader present frame - 1
    kd = {(str(e), int(g)): i for i, (e, g) in enumerate(zip(D["eid"], D["g0"]))}
    ia, ib = [], []
    for r, (c, t) in enumerate(zip(ci, t0)):
        j = kd.get((order[c], int(t) + 1))
        if j is not None:
            ia.append(r); ib.append(j)
    ia, ib = np.asarray(ia, int), np.asarray(ib, int)
    m = {"n_matched": int(len(ia))}
    if len(ia):
        for hi, hn in ((0, "T1.0"), (1, "T2.0")):
            e = np.broadcast_to(exp[:, hi], dT[ia][:, :, hi].shape)
            m["slope_tool_" + hn] = slope0(dT[ia][:, :, hi], e)
            m["slope_reader_" + hn] = slope0(dR[ib][:, :, hi], e)
            x, y = dT[ia][:, :, hi].ravel(), dR[ib][:, :, hi].ravel()
            ok = np.isfinite(x) & np.isfinite(y)
            m["corr_tool_reader_" + hn] = float(np.corrcoef(x[ok], y[ok])[0, 1])
            m["median_abs_diff_" + hn] = float(np.median(np.abs(x[ok] - y[ok])))
        m["present_gap_median_abs_diff_m"] = float(np.nanmedian(np.abs(np.asarray(L["g_src"], np.float64)[k0][rows0][ia] - np.asarray(D["present"], np.float64)[ib, 1])))
        m["rolled_gap_logged_median_abs_diff_T2.0_m"] = float(np.nanmedian(np.abs(gl[ia][:, 1] - np.asarray(D["roll_logged"], np.float64)[ib, 1, 1])))
    res["matched_tool_vs_reader"] = m
    # candidates at tick 0 on the active windows: C against T on the same windows
    cs = np.asarray(L["cand_set"], np.float64)
    fam = str(L["family"]) if "family" in L.files else "scale"
    i1 = int(np.flatnonzero(np.isclose(cs, 0.0 if fam == "decel" else 1.0))[0])
    ah = np.asarray(L["a_h"], np.float64)[k0][rows0]
    gi = np.asarray(L["imag_raw"], np.float64)[k0][rows0]
    da = ah[:, [i1], :] - ah                                             # [n, C, 2] the candidate is da behind the plan
    dg = gi - gi[:, [i1], :]
    c = {"family": fam, "n_active_windows": int(actw.sum())}
    for hi, hn in ((0, "T1.0"), (1, "T2.0")):
        u = actw[:, None] & (da[:, :, hi] >= (0.25 if hi == 0 else 1.0)) & (da[:, :, hi] <= (1.0 if hi == 0 else 4.0)) & np.isfinite(dg[:, :, hi])
        u[:, i1] = False
        c["slope_candidates_" + hn] = slope0(dg[:, :, hi][u], da[:, :, hi][u])
        c["n_pairs_" + hn] = int(u.sum())
        c["slope_tool_path_offsets_same_windows_" + hn] = slope0(dT[actw][:, :, hi], np.broadcast_to(exp[:, hi], dT[actw][:, :, hi].shape))
        # braking offsets only (the candidates only brake)
        c["slope_tool_path_braking_offsets_" + hn] = slope0(dT[actw][:, :2, hi], np.broadcast_to(exp[:2, hi], dT[actw][:, :2, hi].shape))
    if fam == "decel":
        j1 = int(np.flatnonzero(np.isclose(cs, 1.0))[0])
        x, y = dg[actw][:, j1, 1], dT[actw][:, 1, 1]                     # candidate d = 1.0 against logged offset -1.0, at 2.0 s
        ok = np.isfinite(x) & np.isfinite(y)
        c["d1_candidate_vs_logged_offset_T2.0"] = {"corr": float(np.corrcoef(x[ok], y[ok])[0, 1]), "median_candidate_m": float(np.median(x[ok])), "median_logged_offset_m": float(np.median(y[ok])),
                                                    "median_da_candidate_m": float(np.nanmedian(da[actw][:, j1, 1]))}
    res["C_candidates_tick0"] = c
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    rr = lambda d: {k: (round(v, 3) if isinstance(v, float) else (rr(v) if isinstance(v, dict) else v)) for k, v in d.items()}
    for k in ("R_reader_all_dump_windows", "T_tool_path_all_windows", "T_tool_path_active_windows", "matched_tool_vs_reader", "C_candidates_tick0"):
        print(k, json.dumps(rr(res[k])))


if __name__ == "__main__":
    main()
