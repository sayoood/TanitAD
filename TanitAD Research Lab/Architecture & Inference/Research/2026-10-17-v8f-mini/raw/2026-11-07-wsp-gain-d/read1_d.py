"""read1_d: READ 1 of PREREG_WSP_GAIN_D.md section 3 (the mechanism). T0-class (no rollout), dev-box CPU, zero training.
  The edited model's MEDIAN ds/dv0 and ds/da0 (central FD, eps 0.05, ego chain rebuilt: wsp_measure.py, the banked instrument) on the 20,000 TRAIN FIT windows must be within 10 % of g_data -- at both seeds, both gains (4 numbers).
  A miss is MECH-FAIL and nothing else is read. The eval-141 gains (3-8 m/s, the 586 P2b windows) are reported beside (never deciding).
Refuses unless the D card's section 6 carries a dated PIN entry and the edited checkpoints' sha256 equal raw/FIT_RESULT_D.json's.
usage:
  read1_d.py control                              128 TRAIN windows: C0 reproduces the banked shard values; the edit's plan-speed shift equals its analytic expectation (the instrument has a known-value check)
  read1_d.py train --shard I --nshards N [--nt 4] -> raw/READ1D_TRAIN_shard{I}.npz
  read1_d.py combine                              -> raw/READ1_D.json   (the verdict)
  read1_d.py eval                                 -> raw/READ1_D_EVAL.json   (beside; eval-141 band windows; needs the verdict file)"""
import os
import sys
import json
import time
import glob
import argparse

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import torch
import wsp_common as W
import wsp_measure as WM
import wsp_edit as E

TOL = 0.10


def gate():
    card = open(W.CARD, "r", encoding="utf-8").read()
    assert "PIN " in card.split("## 6. Log (append-only, dated)")[1], "REFUSED: no dated PIN entry in the D card's section 6"
    got, head = W.assert_card_frozen()
    assert head == W.CARD_SHA, f"REFUSED: the D card prefix changed ({head})"
    fit = json.load(open(W.HERE + "/raw/FIT_RESULT_D.json"))
    for s in ("0", "1"):
        p = fit["seeds"][s]["ckpt"]
        assert W.sha256_file(p) == fit["seeds"][s]["ckpt_sha256"], f"REFUSED: {p} changed since the pin"
    return fit


def load_edited(M, fit, which=("0", "1")):
    out, cfg0 = {}, None
    for s in which:
        m, cfg, ck = W.load_model(M, fit["seeds"][s]["ckpt"])
        out[s] = (m, cfg)
        cfg0 = cfg0 or cfg
    return out, cfg0


def shard_windows(rig, nshards, shard):
    wins = rig.train_windows(W.BANK + "/raw/train_window_ids.txt")
    assert len(wins) == 20000
    sh = np.array_split(np.arange(len(wins)), nshards)[shard]
    return [wins[i] for i in sh], sh


def cmd_control(a):
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:6.0f}s]", *x, flush=True)
    fit = gate()
    T = W.load_tool(); M = T.M
    ed, cfg = load_edited(M, fit)
    models = {}
    for s in (0, 1):
        m, c, _ = W.load_model(M, W.CKPT_C0[s])
        models[f"c{s}"] = (m, c)
        models[f"e{s}"] = ed[str(s)]
    # load every state dict BEFORE the train Rig: Rig("train") replaces torch.load by an mmap / weights_only=True loader for the clip store
    ckC = {s: torch.load(W.CKPT_C0[s], map_location="cpu", weights_only=False) for s in (0, 1)}
    ckE = {s: torch.load(fit["seeds"][str(s)]["ckpt"], map_location="cpu", weights_only=False) for s in (0, 1)}
    rig = WM.Rig("train", cfg, nt=a.nt)
    sub, sh = shard_windows(rig, 1, 0)
    sub, sh = sub[:a.n], sh[:a.n]
    r = WM.run_measure(rig, models, sub, bs=64, log=log)
    import fit_gdata as G
    R, _ = G.load_shards()
    out = {"n_windows": int(len(sub)), "seeds": {}}
    ok = True
    for s in (0, 1):
        ck = ckC[s]
        r_res = E.readout(ck["model"]["basis"].double().numpy(), ck["cfg"]["res_scale_m"])
        # (i) C0 measured now == the banked fit-time shard values
        d_gv = float(np.abs(r[f"c{s}.gv"] - R[f"s{s}.gv"][sh]).max()); d_ga = float(np.abs(r[f"c{s}.ga"] - R[f"s{s}.ga"][sh]).max()); d_st = float(np.abs(r[f"c{s}.s_tot"] - R[f"s{s}.s_tot"][sh]).max())
        # (ii) edited plan speed == C0 plan speed + the analytic edit effect on the float32 weights that ship
        Wd = (ckE[s]["model"][E.KEY_W][:cfg.nb].double() - ck["model"][E.KEY_W][:cfg.nb].double()).numpy()
        bd = (ckE[s]["model"][E.KEY_B][:cfg.nb].double() - ck["model"][E.KEY_B][:cfg.nb].double()).numpy()
        pred = (r[f"c{s}.h2"].astype(np.float64) @ Wd.T + bd) @ r_res
        d_shift = float(np.abs((r[f"e{s}.s_tot"] - r[f"c{s}.s_tot"]) - pred).max())
        out["seeds"][str(s)] = {"C0_vs_banked_shard_max_abs": {"gv": d_gv, "ga": d_ga, "s_tot": d_st}, "edited_minus_C0_plan_speed_vs_analytic_max_abs": d_shift,
                                "median_gains_C0": [float(np.median(r[f"c{s}.gv"])), float(np.median(r[f"c{s}.ga"]))], "median_gains_edited": [float(np.median(r[f"e{s}.gv"])), float(np.median(r[f"e{s}.ga"]))]}
        ok = ok and d_gv < 2e-3 and d_ga < 2e-3 and d_st < 1e-3 and d_shift < 1e-3
        log(f"s{s}: C0 vs shard max|d| gv {d_gv:.2e} ga {d_ga:.2e} s_tot {d_st:.2e} | edited-minus-C0 vs analytic {d_shift:.2e} | medians C0 {out['seeds'][str(s)]['median_gains_C0']} edited {out['seeds'][str(s)]['median_gains_edited']}")
    out["PASS"] = bool(ok)
    json.dump(out, open(W.HERE + "/raw/READ1D_CONTROL.json", "w"), indent=1)
    log("control", "PASS" if ok else "FAIL")
    return 0 if ok else 4


def cmd_train(a):
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:7.0f}s]", *x, flush=True)
    fit = gate()
    T = W.load_tool(); M = T.M
    ed, cfg = load_edited(M, fit)
    models = {f"e{s}": ed[str(s)] for s in (0, 1)}
    rig = WM.Rig("train", cfg, nt=a.nt)
    sub, sh = shard_windows(rig, a.nshards, a.shard)
    log("shard", a.shard, "of", a.nshards, "windows", len(sub))
    r = WM.run_measure(rig, models, sub, bs=64, log=log)
    r = {k: v for k, v in r.items() if not k.endswith(".h2") and not k.endswith(".traj30")}
    r["flat_idx"] = sh
    out = W.HERE + f"/raw/READ1D_TRAIN_shard{a.shard}.npz"
    np.savez(out, **r)
    log("saved", out)
    return 0


def cmd_combine(a):
    fit = gate()
    gd = json.load(open(W.HERE + "/raw/GDATA.json"))
    files = sorted(glob.glob(W.HERE + "/raw/READ1D_TRAIN_shard[0-9].npz"))
    parts = [dict(np.load(f, allow_pickle=True)) for f in files]
    idx = np.concatenate([p["flat_idx"] for p in parts])
    order = np.argsort(idx, kind="stable")
    assert np.array_equal(idx[order], np.arange(20000)), "the shards do not cover the 20,000 windows exactly once"
    R = {k: np.concatenate([p[k] for p in parts], 0)[order] for k in parts[0]}
    import fit_gdata as G
    B, _ = G.load_shards()
    s_gt = np.load(W.HERE + "/raw/s_gt_train20k.npy")
    tgt = (gd["g_data"]["dsdv0"], gd["g_data"]["dsda0"])
    out = {"card": "PREREG_WSP_GAIN_D.md", "tier": "T0-class mechanism check on the TRAIN fit windows (no rollout)", "n_windows": 20000, "tol": TOL, "g_data": {"dsdv0": tgt[0], "dsda0": tgt[1]},
           "shard_files": [os.path.basename(f) for f in files], "cells": {}}
    ok_all = True
    for s in (0, 1):
        e = (float(np.median(R[f"e{s}.gv"])), float(np.median(R[f"e{s}.ga"])))
        c = (float(np.median(B[f"s{s}.gv"])), float(np.median(B[f"s{s}.ga"])))
        rows = []
        for i, nm in enumerate(("dS/dv0", "dS/da0")):
            rel = abs(e[i] - tgt[i]) / abs(tgt[i])
            rows.append({"gain": nm, "edited": e[i], "g_data": tgt[i], "C0": c[i], "rel_err": float(rel), "within": bool(rel <= TOL)})
            ok_all = ok_all and rel <= TOL
        ds = R[f"e{s}.s_tot"].astype(np.float64) - B[f"s{s}.s_tot"].astype(np.float64)
        out["cells"][f"s{s}"] = {"rows": rows, "READ1_PASS": all(r["within"] for r in rows), "mean_plan_speed_shift_vs_C0": float(ds.mean()), "shift_target": fit["seeds"][str(s)]["shift_target_m_s"],
                                 "mean_edited_plan_speed": float(R[f"e{s}.s_tot"].mean()), "mean_realised_speed": float(s_gt.mean()), "mean_edited_minus_realised": float(R[f"e{s}.s_tot"].mean() - s_gt.mean())}
        print(f"s{s}:", [(r["gain"], round(r["edited"], 4), round(r["g_data"], 4), round(r["rel_err"], 4), r["within"]) for r in rows],
              f"mean shift {ds.mean():+.4f} (target {fit['seeds'][str(s)]['shift_target_m_s']:+.4f}) edited mean {R[f'e{s}.s_tot'].mean():.4f} vs realised {s_gt.mean():.4f}")
    out["READ1_ALL_CELLS_PASS"] = bool(ok_all)
    out["verdict"] = "MECH-OK" if ok_all else "MECH-FAIL"
    json.dump(out, open(W.HERE + "/raw/READ1_D.json", "w"), indent=1)
    print("verdict", out["verdict"])
    return 0


def cmd_eval(a):
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:7.0f}s]", *x, flush=True)
    fit = gate()
    assert os.path.exists(W.HERE + "/raw/READ1_D.json"), "REFUSED: the TRAIN-window verdict (READ1_D.json) does not exist yet"
    T = W.load_tool(); M = T.M
    ed, cfg = load_edited(M, fit)
    models = {}
    for s in (0, 1):
        m, c, _ = W.load_model(M, W.CKPT_C0[s])
        models[f"s{s}_C0"] = (m, c)
        models[f"s{s}_d"] = ed[str(s)]
    rig = WM.Rig("eval", cfg, nt=a.nt)
    wins = rig.eval_windows()
    z = np.load(W.THEORY + "/wsp_read/P2B_PERWINDOW.npz", allow_pickle=True)
    idx = np.flatnonzero(z["band"] == 1)
    sub = [wins[i] for i in idx]
    log("eval-141 band windows", len(sub))
    r = WM.run_measure(rig, models, sub, bs=64, log=log)
    assert np.abs(r["kin3"][:, 0] - z["v0"][idx]).max() == 0.0 and (r["clip"] == z["clip"][idx]).all(), "window set differs from p2b's"
    gd = json.load(open(W.HERE + "/raw/GDATA.json"))
    tgt = (gd["g_data"]["dsdv0"], gd["g_data"]["dsda0"])
    out = {"card": "PREREG_WSP_GAIN_D.md", "n_windows": int(len(sub)), "note": "BESIDE the TRAIN-window READ 1; reported, never deciding", "g_data_train": {"dsdv0": tgt[0], "dsda0": tgt[1]},
           "base_median_gain_eval": {"dsdv0": float(np.median(r["gb"][:, 0])), "dsda0": float(np.median(r["gb"][:, 1]))}, "control_s_tot_vs_p2b_max_abs_C0_s0": float(np.abs(r["s0_C0.s_tot"] - z["val.s_tot"][idx]).max()), "cells": {}}
    for s in (0, 1):
        gC = (float(np.median(r[f"s{s}_C0.gv"])), float(np.median(r[f"s{s}_C0.ga"])))
        gE = (float(np.median(r[f"s{s}_d.gv"])), float(np.median(r[f"s{s}_d.ga"])))
        out["cells"][f"s{s}"] = {"C0_median_gain": {"dsdv0": gC[0], "dsda0": gC[1]}, "edited_median_gain": {"dsdv0": gE[0], "dsda0": gE[1]},
                                 "rel_err_vs_g_data": {"dsdv0": abs(gE[0] - tgt[0]) / tgt[0], "dsda0": abs(gE[1] - tgt[1]) / tgt[1]},
                                 "mean_plan_speed_shift_eval_m_s": float(np.mean(r[f"s{s}_d.s_tot"] - r[f"s{s}_C0.s_tot"])), "plan_speed_mean_C0": float(r[f"s{s}_C0.s_tot"].mean()),
                                 "plan_speed_mean_edited": float(r[f"s{s}_d.s_tot"].mean()), "mean_v0": float(r["kin3"][:, 0].mean())}
        log(f"s{s}: C0 {gC} -> edited {gE}  (g_data {tgt})")
    json.dump(out, open(W.HERE + "/raw/READ1_D_EVAL.json", "w"), indent=1)
    np.savez(W.HERE + "/raw/READ1_D_EVAL_PERWINDOW.npz", clip=r["clip"], t0=r["t0"], kin3=r["kin3"], **{k: v for k, v in r.items() if "." in k and not k.endswith("h2")})
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["control", "train", "combine", "eval"])
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=3)
    ap.add_argument("--nt", type=int, default=4); ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()
    sys.exit({"control": cmd_control, "train": cmd_train, "combine": cmd_combine, "eval": cmd_eval}[a.cmd](a))
