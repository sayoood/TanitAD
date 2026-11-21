"""provisional_s1.py -- PROVISIONAL, NOT THE REGISTERED READ (PREREG R8-PICKER sealed e6ec30e7... before this ran; nothing here can move it).

Purpose (Rule Zero, while Thor is booked): push the post-hoc pickers through the SAME splice + SHIP-R8 lane + family read on the EXISTING
fanceil Thor s1 fan (inference seed 1, os-only, 99.4 % of fan cells equal to the bank's s1 draw), so the pipeline is exercised end to end on
real arms and the RETIME-on-top question gets an early, clearly-labelled answer.  Seed 0 of every provisional tree is the bank's OWN plan
(identity) -- so every s0 cell reads exactly 0 vs SHIP and only s1 is informative; the SHIP-R8 'overall' of a provisional tree is NOT a picker
verdict.  The registered read (both seeds, the model's own e9_keep mask) comes only from the Thor capture rolls R0 / R1.

Trees (picker archive):  pvship_pf  s1 = fan[sel_idx]                (SHIP_T)
                         pvbase_pf  s1 = fan[bank sel_idx_base]      (the fanceil 'base', the bank's s1 decisions)
                         pvmbrpx_pf s1 = the FROZEN cell sel_score,32,1.0 with the slot-level cap PROXY (exactly picker_panel_r8's
                                     mbr_chosenA_capproxy; the registered arm uses e9_keep instead, which this fan does not carry)
"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_picker_trees as BT  # noqa: E402

BANK = "C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf"
ARCH = "C:/Users/Admin/refcv8_pf/picker/archive"
FAN = "C:/Users/Admin/refcv8_pf/fanceil/runs/s1/dump/fan_extras.npz"
X8 = "C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code"
SLOT_T = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])


def vmax_plan(fan8):
    pts = np.concatenate([np.zeros(fan8.shape[:-2] + (1, 2)), fan8], -2)
    dt = np.diff(np.concatenate([[0.0], SLOT_T]))
    return (np.linalg.norm(np.diff(pts, axis=-2), axis=-1) / dt).max(-1)


def main():
    t0 = time.time()
    MB = BT.load_mbr(X8)
    z = np.load(FAN)
    ci, ws = z["clip_index"].astype(np.int64), z["ws"].astype(np.int64)
    fan = np.asarray(z["anchor_traj"], np.float32)
    n = len(ci)
    ar = np.arange(n)
    own = z["sel_idx"].astype(np.int64).reshape(-1)
    base = {}
    for f in sorted(glob.glob(os.path.join(BANK, "dump_s1", "decisions", "ep*.npz"))):
        d = np.load(f)
        c = int(np.asarray(np.load(os.path.join(BANK, "dump_s1", os.path.basename(f)))["clip_index"]).ravel()[0])
        for j, w in enumerate(d["ws"]):
            base[(c, int(w))] = int(d["sel_idx_base"][j])
    bsel = np.asarray([base[(int(c), int(w))] for c, w in zip(ci, ws)], np.int64)
    E = np.load(os.path.join(BANK, "dump_s1", "refcv6_extras.npz"))
    ib = BT.np.asarray([{(int(c), int(w)): i for i, (c, w) in enumerate(zip(E["clip_index"], E["ws"]))}[(int(c), int(w))] for c, w in zip(ci, ws)])
    vmax, vval = E["v_max_ms"][ib].astype(np.float64), E["v_max_valid"][ib] > 0.5
    fan64 = fan.astype(np.float64)
    ok = ~((vmax_plan(fan64) > vmax[:, None] + 0.5) & vval[:, None])
    ok = np.where(ok.any(1, keepdims=True), ok, True)
    s = np.asarray(z["sel_score"], np.float64)
    s = np.where(np.isfinite(s), s, -1e30)
    mbr = MB.mbr_pick(fan64[:, :, :4], np.where(ok, s, -1e30), 32, 1.0).astype(np.int64)
    bank0 = BT.bank_plans(BANK)[0]
    rec = {"status": "PROVISIONAL -- not the registered read", "fan": FAN, "fan_sha256": BT.sha256_file(FAN), "n_windows_s1": int(n),
           "own_vs_bank_sel_idx_nav_true_agree": float((own == E["nav_true.sel_idx"].astype(np.int64)[ib]).mean()),
           "agree_with_own": {"base": float((bsel == own).mean()), "mbrpx": float((mbr == own).mean())}, "trees": {}}
    for tree, sel in (("pvship_pf", own), ("pvbase_pf", bsel), ("pvmbrpx_pf", mbr)):
        plans = {0: bank0, 1: {"clip": ci, "ws": ws, "plan8": fan[ar, sel], "sel": sel}}
        rec["trees"][tree] = BT.splice_tree(BANK, os.path.join(ARCH, tree), plans, tree)["seeds"]["1"]["files"]
    rec["wall_s"] = round(time.time() - t0, 1)
    json.dump(rec, open("C:/Users/Admin/refcv8_pf/picker/PROVISIONAL_S1_BUILD.json", "w", encoding="utf-8"), indent=1, default=str)
    print("PROVISIONAL trees built", rec["agree_with_own"], rec["own_vs_bank_sel_idx_nav_true_agree"], "ZZPKPROVBUILTZZ")


if __name__ == "__main__":
    main()
