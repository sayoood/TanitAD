"""read1_eval: READ 1 of PREREG_WSP_GAIN.md section 3 (the mechanism check), T0-class, dev-box CPU. Scores eval-141 windows with v0 in [3, 8) m/s (586 windows, the p2b set).
Refuses unless the card section 6 carries a dated PIN entry (written before this script's first eval-141 score of an EDITED model).
Per (seed, alpha): the edited model's median dS/dv0 and dS/da0 (central FD, eps 0.05, the ego chain rebuilt) against the alpha-restored target (1 - alpha) g_C0 + alpha g_base (g_C0, g_base measured on the same
eval windows), tolerance 10 %. Second target (reported beside, not deciding): g_C0(eval) + alpha * dg(train).
usage: read1_eval.py   -> raw/READ1.json, raw/READ1_PERWINDOW.npz"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import torch
import wsp_common as W
import wsp_measure as WM
import wsp_edit as E

TOL = 0.10


def main():
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:7.0f}s]", *x, flush=True)
    card = open(W.CARD, "r", encoding="utf-8").read()
    assert "PIN " in card.split("## 6. Log (append-only, dated)")[1], "REFUSED: no dated PIN entry in card section 6"
    got, head = W.assert_card_frozen()
    assert head == W.CARD_SHA, f"REFUSED: card text above the log changed ({head})"
    fit = json.load(open(W.HERE + "/raw/FIT_RESULT.json"))
    T = W.load_tool(); M = T.M
    models = {}
    cfg0 = None
    for s in (0, 1):
        m, cfg, ck = W.load_model(M, W.CKPT_C0[s])
        models[f"s{s}_C0"] = (m, cfg)
        cfg0 = cfg0 or cfg
        for tag in ("05", "10"):
            p = fit["seeds"][str(s)]["arms"][tag]["ckpt"]
            assert W.sha256_file(p) == fit["seeds"][str(s)]["arms"][tag]["ckpt_sha256"], f"REFUSED: {p} changed since the pin"
            me, cfge, cke = W.load_model(M, p)
            models[f"s{s}_a{tag}"] = (me, cfge)
    rig = WM.Rig("eval", cfg0, nt=int(os.environ.get("NT", "8")))
    wins = rig.eval_windows()
    z = np.load(W.THEORY + "/wsp_read/P2B_PERWINDOW.npz", allow_pickle=True)
    idx = np.flatnonzero(z["band"] == 1)
    sub = [wins[i] for i in idx]
    log("eval-141 band windows", len(sub))
    r = WM.run_measure(rig, models, sub, bs=64, log=log)
    assert np.abs(r["kin3"][:, 0] - z["v0"][idx]).max() == 0.0 and (r["clip"] == z["clip"][idx]).all(), "window set differs from p2b's"
    gbm = (float(np.median(r["gb"][:, 0])), float(np.median(r["gb"][:, 1])))
    out = {"card_sha256": W.CARD_SHA, "n_windows": int(len(sub)), "tol": TOL, "base_median_gain_eval": {"dsdv0": gbm[0], "dsda0": gbm[1]}, "control_s_tot_vs_p2b_max_abs_C0_s0": float(np.abs(r["s0_C0.s_tot"] - z["val.s_tot"][idx]).max()), "cells": {}}
    all_ok = True
    for s in (0, 1):
        gC = (float(np.median(r[f"s{s}_C0.gv"])), float(np.median(r[f"s{s}_C0.ga"])))
        for tag, alpha in (("05", 0.5), ("10", 1.0)):
            nm = f"s{s}_a{tag}"
            gE = (float(np.median(r[f"{nm}.gv"])), float(np.median(r[f"{nm}.ga"])))
            ok, rows = E.read1_verdict(gE, gC, gbm, alpha, TOL)
            arm = fit["seeds"][str(s)]["arms"][tag]
            alt = {"dsdv0": gC[0] + alpha * arm["dg_v"], "dsda0": gC[1] + alpha * arm["dg_a"]}
            shift = float(np.mean(r[f"{nm}.s_tot"] - r[f"s{s}_C0.s_tot"]))
            out["cells"][nm] = {"seed": s, "alpha": alpha, "C0_median_gain": {"dsdv0": gC[0], "dsda0": gC[1]}, "edited_median_gain": {"dsdv0": gE[0], "dsda0": gE[1]}, "rows": rows,
                                "alt_target_train_dg": alt, "alt_rel_err": {"dsdv0": abs(gE[0] - alt["dsdv0"]) / abs(alt["dsdv0"]), "dsda0": abs(gE[1] - alt["dsda0"]) / abs(alt["dsda0"])},
                                "READ1_PASS": bool(ok), "mean_plan_speed_shift_eval_m_s": shift, "target_mean_shift_m_s": alpha * E.SHIFT,
                                "plan_speed_mean_C0": float(r[f"s{s}_C0.s_tot"].mean()), "plan_speed_mean_edited": float(r[f"{nm}.s_tot"].mean()), "mean_v0": float(r["kin3"][:, 0].mean())}
            all_ok = all_ok and ok
            log(nm, "PASS" if ok else "MISS", [(x["gain"], round(x["edited"], 4), round(x["target"], 4), round(x["rel_err"], 4)) for x in rows], f"shift {shift:+.3f} (target {alpha*E.SHIFT:+.3f})")
    out["READ1_ALL_CELLS_PASS"] = bool(all_ok)
    out["verdict"] = "MECH-OK" if all_ok else "MECH-FAIL"
    json.dump(out, open(W.HERE + "/raw/READ1.json", "w"), indent=1)
    np.savez(W.HERE + "/raw/READ1_PERWINDOW.npz", clip=r["clip"], t0=r["t0"], kin3=r["kin3"], **{k: v for k, v in r.items() if "." in k and not k.endswith("h2")})
    log("verdict", out["verdict"])


if __name__ == "__main__":
    main()
