"""rh_strat_floor.py (R3) -- one-seed STRATEGIC floor of the ROLLHIST read (Amendment 2 items 5 and 6; MM_CONFIRM_R2 item 6).
F_U,c = abs(UE_s1.A - UE_s0.A) per STRATEGIC cell under R0C10XHb20: the level of ADD66 `pos` at 30/50/80 m at the third tick (tick 2), the same
per-arm level (nanmean over windows) that rh_strat.py keeps in levels_tick2_pos and that rh_read.strat_family differences for a two-seed floor.
Cells come from the frozen ADD66 functions (strat_roll_read.Ctx, load_ticks, compute_cells), unchanged.  Keys written: pos30, pos50, pos80 (what decide() reads).
Beside (reported, not read by decide): tick-1 pos and tick-2 lat/along/head levels per arc, and the per-family-reported cells.
Controls: K-SELF (s0 vs s0 = 0.0 exactly), K-PLANT (planted offset moves only the touched cell by exactly that amount), MUTATION (B0/RH run names refused).
Usage: python rh_strat_floor.py [--s0 TICKS.npz] [--s1 TICKS.npz] [--out FILE] [--selftest]"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/code/integ10k")
import rh_common as C  # noqa: E402
import strat_roll_read as S  # noqa: E402

FORM = "R0C10XHb20"
S0 = C.OUT + "/cexec/B_record_hb_rzg/ticks/ticks_cexec_UE_s0A_R0C10XHb20_m10_causal_heads.npz"     # c-EXEC's tier-of-record dump (adopted by rh_read_chain.sh)
S1 = C.OUT + f"/t1/I10UEA_{FORM}_s1/ticks/ticks_I10UEA_{FORM}_s1_m10_causal_heads.npz"


def check_names(p0, p1):
    """mutation guard: only the UE.A x R0C10XHb20 seed-0 / seed-1 files are admissible (never RH / B0)."""
    for p, s in ((p0, 0), (p1, 1)):
        b = os.path.basename(p)
        if "UE" not in b or FORM not in b or (f"_s{s}" not in b and "_s0A_" not in b and "UE_s0A" not in b):
            raise SystemExit(f"REFUSED: {b} is not a UE.A {FORM} seed-{s} tick file (RH/B0 run names are not stage-A references)")
        if "RH_" in b or "B0_" in b or "_RH" in b or "_B0" in b:
            raise SystemExit(f"REFUSED: {b} names an RH/B0 run")


def level_cells(cells):
    """per cell name -> 3 arc levels (nanmean over windows)."""
    out = {}
    for k in (1, 2):
        for c in S.CELLS:
            out[f"tick{k}_{c}"] = [float(np.nanmean(cells[c][k][:, m])) for m in range(3)]
    return out


def floor_of(la, lb):
    """per cell floor = abs(b - a); the decide() keys pos30/50/80 are the tick-2 pos cells."""
    full = {n: [abs(y - x) for x, y in zip(la[n], lb[n])] for n in la}
    keys = {f"pos{int(A)}": full["tick2_pos"][m] for m, A in enumerate(S.ARCS)}
    return keys, full


def compute_levels(p, a):
    ctx = S.Ctx(a)
    ctx.load_static()
    T = S.load_ticks(p)
    cells, _, _ = S.compute_cells(ctx, T)
    return cells, S.sha256_file(p), T


def controls(cells):
    la = level_cells(cells)
    k_self, _ = floor_of(la, la)
    ok_self = all(v == 0.0 for v in k_self.values()) and all(v == 0.0 for xs in floor_of(la, la)[1].values() for v in xs)
    # K-PLANT: +0.37 m on tick-2 pos at the 50 m arc of one copy
    pl = {c: cells[c].copy() for c in cells}
    pl["pos"][2][:, 1] = pl["pos"][2][:, 1] + 0.37
    kp, kfull = floor_of(la, level_cells(pl))
    moved = {n: [round(v, 9) for v in xs] for n, xs in kfull.items() if any(v > 1e-9 for v in xs)}
    ok_plant = bool(abs(kp["pos50"] - 0.37) < 1e-9 and kp["pos30"] == 0.0 and kp["pos80"] == 0.0 and list(moved) == ["tick2_pos"] and moved["tick2_pos"][0] == 0.0 and moved["tick2_pos"][2] == 0.0)
    # mutation: RH/B0 names refused
    refused = False
    try:
        check_names(C.OUT + "/t1/I10RH_R0C10XHb20_s0/ticks/ticks_I10RH_R0C10XHb20_s0_m10_causal_heads.npz", C.OUT + "/t1/I10B0_R0C10XHb20_s1/ticks/ticks_I10B0_R0C10XHb20_s1_m10_causal_heads.npz")
    except SystemExit:
        refused = True
    return {"K-SELF": {"pass": bool(ok_self), "must_read": "s0 against itself: 0.0 exactly on every cell", "floor": k_self},
            "K-PLANT": {"pass": ok_plant, "must_read": "+0.37 m planted on tick-2 pos 50 m of one copy: that cell moves by 0.37, no other cell moves", "pos_floor": kp, "cells_moved": moved},
            "MUTATION": {"pass": refused, "must_read": "RH/B0 run names refused"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", default=S0)
    ap.add_argument("--s1", default=S1)
    ap.add_argument("--out", default=C.OUT + "/read/STRAT_FLOOR_UEA.json")
    ap.add_argument("--tracks", default=C.INPUTS["--tracks"])
    ap.add_argument("--join", default=C.INPUTS["--eval-join"])
    ap.add_argument("--tree", default=C.TREE)
    ap.add_argument("--w", default=C.W)
    a = ap.parse_args()
    check_names(a.s0, a.s1)
    c0, h0, T0 = compute_levels(a.s0, a)
    c1, h1, T1 = compute_levels(a.s1, a)
    same_windows = bool((T0["ci"][np.argsort(T0["win"])] .shape == T1["ci"].shape) and (S.load_ticks(a.s0)["t0"].shape == T1["t0"].shape))
    ctl = controls(c0)
    la, lb = level_cells(c0), level_cells(c1)
    keys, full = floor_of(la, lb)
    res = {"card": "PREREG_V8F_ROLLHIST Amendment 2 items 5, 6", "evidence_class": "MEASURED (ours; UE_s0.A and UE_s1.A x R0C10XHb20 tick files, frozen ADD66 cell functions)",
           "form": FORM, "inputs_sha256": {os.path.basename(a.s0): h0, os.path.basename(a.s1): h1}, "same_window_count": same_windows,
           "floor_definition": "F_U,c = abs(level_UE_s1.A - level_UE_s0.A), level = nanmean over windows of the tick-2 `pos` cell at 30/50/80 m",
           "floor": keys, "level_s0": la["tick2_pos"], "level_s1": lb["tick2_pos"], "beside_all_cells_floor": full, "controls": ctl,
           "controls_all_pass": all(v["pass"] for v in ctl.values())}
    if not res["controls_all_pass"]:
        res["floor"] = None
        res["stopped"] = "controls failed; no floor written"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print("floor", res["floor"], "controls", res["controls_all_pass"], flush=True)
    print("ZZ RH-STRAT-FLOOR " + ("DONE" if res["controls_all_pass"] else "STOP") + " ZZ")
    return 0 if res["controls_all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
