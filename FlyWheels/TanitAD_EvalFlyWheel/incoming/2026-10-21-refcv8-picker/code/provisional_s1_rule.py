"""provisional_s1_rule.py -- PROVISIONAL, NOT THE REGISTERED READ (PREREG R8-PICKER sealed e6ec30e7... before this ran; nothing here can move it).

Why (Rule Zero): on the fanceil Thor s1 fan the frozen MBR cell with the slot-level PROXY cap mask (pvmbrpx_pf) was separated WORSE than SHIP on
COMPLIANCE and four LATERAL cells at s1.  The registered MBR32 masks with the model's OWN e9_keep instead.  Is the compliance failure an artifact
of the proxy?  This tree rebuilds e9_keep on the same fan from the model's own code path and re-runs the frozen cell with it:

  reach_keep   : carried by the fanceil fan (the model's own, captured)
  ceil_keep    : RE-DERIVED by the producer's rule  planned_max_speed(fan, horizons 5..60 ticks, 0.1 s) <= fed ceiling, empty row -> all True
                 (MEASURED on the 2 x 8-window CPU smoke: equals the captured ceil_keep on 1.0 of cells, K1_SMOKE.json K6b 'beside')
  r8_emit_keep : ASSUMED all True (all True on every smoke window; at EMA-48,852 every candidate's emit flag is past its schedule)
  e9_keep      = isfinite(refc_v3.e9_rank(zeros, {reach, ceil, emit}))   -- the model's OWN function, with its dead-row fallbacks

Discriminating control (printed, not assumed): the CORE argmax re-built with these masks must reproduce the bank's s1 sel_idx_base on ~ the
fan's own agreement level, and must agree MORE than the same argmax WITHOUT the ceiling mask (otherwise the re-derived mask is inert).

Tree: pvmbrrule_pf   s1 = frozen cell (sel_score, K 32, T 1.0) masked with the re-derived e9_keep; s0 = the bank's own plan (identity).
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
import controls_pk as K  # noqa: E402

BANK = "C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf"
ARCH = "C:/Users/Admin/refcv8_pf/picker/archive"
FAN = "C:/Users/Admin/refcv8_pf/fanceil/runs/s1/dump/fan_extras.npz"
X8 = "C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code"
TREE = "pvmbrrule_pf"


def main():
    import torch
    from tanitad.refs import refc_v3 as RV3
    from tanitad.refs.refcv6_selection import planned_max_speed
    t0 = time.time()
    MB = BT.load_mbr(X8)
    z = np.load(FAN)
    ci, ws = z["clip_index"].astype(np.int64), z["ws"].astype(np.int64)
    fan = np.asarray(z["anchor_traj"], np.float32)
    n = len(ci)
    ar = np.arange(n)
    E = np.load(os.path.join(BANK, "dump_s1", "refcv6_extras.npz"))
    idx = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(E["clip_index"], E["ws"]))}
    ib = np.asarray([idx[(int(c), int(w))] for c, w in zip(ci, ws)])
    lim, linfo = K.fed_ceiling_eval(E["v_max_ms"][ib], E["v_max_valid"][ib])
    pm = planned_max_speed(torch.from_numpy(fan), horizons=K.HZ8, tick_s=0.1).numpy().astype(np.float64)
    ok = pm <= lim[:, None]
    ceil = ok | (~ok.any(1))[:, None]
    reach = np.asarray(z["reach_keep"]) > 0.5
    emit = np.ones_like(reach)
    out = {"reach_keep": torch.from_numpy(reach), "ceil_keep": torch.from_numpy(ceil), "r8_emit_keep": torch.from_numpy(emit)}
    s = np.asarray(z["sel_score"], np.float32)
    rank0, _ = RV3.e9_rank(torch.zeros_like(torch.from_numpy(s)), out)
    e9 = np.isfinite(rank0.numpy())
    # discriminating control: the CORE argmax with / without the re-derived ceiling vs the bank's own s1 sel_idx_base
    base = {}
    for f in sorted(glob.glob(os.path.join(BANK, "dump_s1", "decisions", "ep*.npz"))):
        d = np.load(f)
        c = int(np.asarray(np.load(os.path.join(BANK, "dump_s1", os.path.basename(f)))["clip_index"]).ravel()[0])
        for j, w in enumerate(d["ws"]):
            base[(c, int(w))] = int(d["sel_idx_base"][j])
    bsel = np.asarray([base[(int(c), int(w))] for c, w in zip(ci, ws)], np.int64)
    zc = {"sel_score": s, "reach_keep": reach, "ceil_keep": ceil, "r8_emit_keep": emit}
    zn = {"sel_score": s, "reach_keep": reach, "ceil_keep": np.ones_like(reach), "r8_emit_keep": emit}
    core_c = K.core_rank(zc).argmax(1)
    core_n = K.core_rank(zn).argmax(1)
    ctl = {"core_with_rederived_ceil_vs_bank_sel_idx_base": float((core_c == bsel).mean()),
           "core_WITHOUT_ceil_vs_bank_sel_idx_base": float((core_n == bsel).mean()),
           "n_windows_where_ceil_changes_core_argmax": int((core_c != core_n).sum()),
           "ceil_binding_windows": int((~ceil).any(1).sum()), "ceil_keep_cell_frac": float(ceil.mean()),
           "e9_keep_count_min": int(e9.sum(1).min()), "n_windows_e9_keep_lt_32": int((e9.sum(1) < 32).sum()), "fed_ceiling": linfo}
    ctl["PASS"] = bool(ctl["core_with_rederived_ceil_vs_bank_sel_idx_base"] >= 0.99
                       and ctl["core_with_rederived_ceil_vs_bank_sel_idx_base"] > ctl["core_WITHOUT_ceil_vs_bank_sel_idx_base"])
    print("[rule] control", ctl, flush=True)
    sc = np.where(np.isfinite(s), s, -1e30).astype(np.float64)
    mbr = MB.mbr_pick(fan.astype(np.float64)[:, :, :4], np.where(e9, sc, -1e30), 32, 1.0).astype(np.int64)
    own = z["sel_idx"].astype(np.int64).reshape(-1)
    px = np.load(os.path.join(ARCH, "pvmbrpx_pf", "dump_s1", "refcv6_extras.npz"))   # the proxy tree's s1 plan, for the agreement beside
    rec = {"status": "PROVISIONAL -- not the registered read", "fan": FAN, "fan_sha256": BT.sha256_file(FAN), "n_windows_s1": int(n), "control": ctl,
           "mbr_pick_outside_e9_keep_frac": float((~e9[ar, mbr]).mean()), "agree_with_own": float((mbr == own).mean()),
           "plan_equal_proxy_tree_frac": float((np.abs(fan[ar, mbr] - px["nav_true.traj"].astype(np.float32)[[{(int(c), int(w)): i for i, (c, w) in enumerate(zip(px['clip_index'], px['ws']))}[(int(c), int(w))] for c, w in zip(ci, ws)]]).max((1, 2)) == 0).mean())}
    if not ctl["PASS"]:
        rec["tree"] = None
        print("[rule] the re-derived mask FAILED its control -- no tree built", flush=True)
    else:
        bank0 = BT.bank_plans(BANK)[0]
        plans = {0: bank0, 1: {"clip": ci, "ws": ws, "plan8": fan[ar, mbr], "sel": mbr}}
        rec["tree"] = BT.splice_tree(BANK, os.path.join(ARCH, TREE), plans, TREE)["seeds"]["1"]["files"]
    rec["wall_s"] = round(time.time() - t0, 1)
    json.dump(rec, open("C:/Users/Admin/refcv8_pf/picker/PROVISIONAL_S1_RULE_BUILD.json", "w", encoding="utf-8"), indent=1, default=str)
    print("PROVISIONAL rule tree", {k: rec[k] for k in ("mbr_pick_outside_e9_keep_frac", "agree_with_own", "plan_equal_proxy_tree_frac")},
          "ZZPKPROVRULEBUILTZZ" if rec["tree"] else "ZZPKPROVRULEREFUSEDZZ", flush=True)


if __name__ == "__main__":
    import pk_peak
    pk_peak.arm("provisional_s1_rule")
    main()
