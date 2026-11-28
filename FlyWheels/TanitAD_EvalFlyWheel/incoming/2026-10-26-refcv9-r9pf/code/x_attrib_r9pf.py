"""x_attrib_r9pf.py -- EXPLORATORY, NOT A BAR, NOT PRE-REGISTERED.  RULE ZERO attribution after the Phase-A bars: which term of R9PF loses to
H2CN_DEPLOY8 -- the regime DECISION or the stage-2 PROFILE magnitudes?  Post-hoc on eval139 with the registered heads; every arm here is a
DIAGNOSTIC that ranks the next lever and must be pre-registered before it can be claimed.  NEW (EvalFlyWheel 2026-10-26).

Arms (per fit seed 0/1/2, inference seeds 0/1, on the banked fb47a58 battery plan exactly as the bars):
  R9        the registered composition (K-REPRO: re-composed here == the a3 package, bit for bit)
  ORACLE    R9's profiles with the GT regime r* (the MD/FD metric's own kinematic class) -- INADMISSIBLE at inference; isolates the
            PROFILE term (if ORACLE still loses ADE / LON rows to H2cn, the magnitudes are the lever, not the decision)
  CRUISE    R9's cruise profile on every window (decision ablated)
  TAU       the prereg s1 EXPLORATORY calibration-tuned decel tau (x_tau_fit.py, OOF Youden J) on the registered head
  SOFT      the posterior-weighted profile sum_r p(r) dS[r] (no hard decision)
  HYB       H2CN_DEPLOY8's plan where R9 decides cruise, R9's plan where it decides decel / accel (R9's decision gating H2cn's magnitudes)
Cells: paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci, B 2000, seed 0) vs H2CN_DEPLOY8 on S6 compliant (MD on
GT-decel, FD on GT-cruise, ADE, the 14 four-family rows) and, for CRUISE / HYB, S2 compliant + exceedance.  ORACLE is S6-only (r* needs 5 s + 6 s).
-> result/R9PF_X_ATTRIB.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402
from a3_bars import PKG, TREE, LANDED, cell, fmt, sep_above, sep_below  # noqa: E402

LABEL = ("EXPLORATORY -- post-hoc on eval139 with the registered heads; NOT a bar, NOT pre-registered; attribution to rank the next lever only. "
         "ORACLE uses the GT regime (inadmissible at inference).")


def main() -> int:
    t0 = time.time()
    RC.check_prereg()
    RC.check_amendment1()
    RC.check_fix_pin()
    RC.setup_paths()
    sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
    import l2_common as L2
    import compliant_reread_r8 as CR
    import clamp_posthoc_r8 as CP
    import l1_fit_score as F
    st = L2.stack()
    PR, ra = st["PR"], st["ra"]
    METRICS = [(m[0], m[1], bool(m[4])) for m in PR.METRICS]
    rc = F.RT.tools_load()["rc"]
    V_cmp, _vrec = CP.load_vendored()
    R = RC.r9p()
    reg = {}
    for ln in (RC.WORK / "REGISTERED_R9PF_PA.sha256").read_text(encoding="utf-8-sig").splitlines():
        p = ln.strip().split(None, 1)
        if len(p) == 2:
            reg[Path(p[1].lstrip("*").strip()).as_posix()] = RC.hex64(p[0])
    out = {"label": LABEL, "tier": "T1 self-action OPEN LOOP on logged eval139 frames, not driving performance",
           "estimator": "paired episode-cluster bootstrap over EPISODES, B 2000 seed 0 -- blind to training and inference variance", "cells": {}, "controls": {}}
    for s in (0, 1):
        E = L2.load_eval_pf(s)
        I = np.load(PKG / f"EVAL_INPUTS_{TREE}_s{s}.npz", allow_pickle=False)
        H = np.load(LANDED / "H2CN_DEPLOY8" / f"DEPLOY8_POS_{TREE}_s{s}.npz", allow_pickle=False)
        eid6, comp6 = list(E["eid"]), np.asarray(E["compliant"], bool)
        m2 = I["S2_compliant"].astype(bool)
        eid2 = list(I["S2_eid"])
        em2 = [e for e, k in zip(eid2, m2) if k]
        em6 = [e for e, k in zip(eid6, comp6) if k]
        k2 = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(I["S2_clip"], I["S2_ws"]))}
        j6 = np.array([k2[(int(c), int(w))] for c, w in zip(I["S6_clip"], I["S6_ws"])])
        assert np.array_equal(I["S6_ws"], E["ws"]) and np.array_equal(I["S6_clip"], E["clip"])
        G6, G2 = E["g"].astype(np.float32), I["S2_G"].astype(np.float32)
        r_star, _dv = RC.regime_independent(I["S6_G"][:, 4], I["S6_G"][:, 5], I["S6_v0"])
        lim2 = I["S2_lim"].astype(np.float32)

        def comps(pos6, pos2=None):
            c = {"S6": PR.components(ra, np.asarray(pos6, np.float32), G6, float(E["dt"]))}
            if pos2 is not None:
                c["S2"] = PR.components(ra, np.asarray(pos2, np.float32), G2, float(I["S2_dt"]))
            return c

        def exceed(p8):
            pm = CR.pmax(V_cmp, np.asarray(p8, np.float32), [int(h) for h in rc["horizons"]], rc["tick_s"])
            return (np.isfinite(lim2) & ((pm - lim2).astype(np.float32) > 0)).astype(np.float64)

        cH = comps(H["S6_pos"], H["S2_pos"])
        FXp = np.load(LANDED / "FACTORED_DEPLOY8" / f"DEPLOY8_POS_{TREE}_s{s}.npz", allow_pickle=False)
        D8p = np.load(LANDED / "RETIME_DEPLOY8" / f"DEPLOY8_POS_{TREE}_s{s}.npz", allow_pickle=False)
        fdF, mdF = L2.lon_indicators(FXp["S6_pos"].astype(np.float32), E, comp6)[:2]
        cD, cHP = comps(D8p["S6_pos"], D8p["S2_pos"]), comps(I["S6_HP"], I["S2_HP"])
        fdH, mdH = L2.lon_indicators(H["S6_pos"].astype(np.float32), E, comp6)[:2]
        excH = exceed(H["S2_pos8"])
        for k in RC.FIT_SEEDS:
            tag = f"R9PF_fs{k}"
            hp = RC.HEADS / f"{tag}.pt"
            h, blob, _sh = RC.load_head(hp, reg.get(hp.as_posix()))
            pk = np.load(PKG / tag / f"R9_POS_{TREE}_s{s}.npz", allow_pickle=False)
            plan2 = I["S2_plan_battery"]
            o = RC.compose(h, blob, I["S2_Z"], I["S2_x9"], I["S2_prior8"], plan2, None)
            out["controls"][f"K-REPRO|{tag}|s{s}"] = {"PASS": bool(np.array_equal(o["traj"].astype(np.float32), pk["S2_pos8"])
                                                                   and np.array_equal(o["decision"], pk["S2_decision"]))}
            prof, s_p, dec2 = torch.from_numpy(o["profiles"]), torch.from_numpy(o["s_prior"]), o["decision"]

            def plan_for(r, rows):
                with torch.no_grad():
                    s_t = R.monotone_progress(s_p[rows] + R.select_profile(prof[rows], torch.from_numpy(np.asarray(r, np.int64))))
                    return R.retime(torch.from_numpy(np.ascontiguousarray(plan2[rows], np.float32)), s_t).to(torch.float32).numpy()
            all2 = np.arange(len(plan2))
            arms = {}
            arms["R9"] = {"p8_2": pk["S2_pos8"], "p8_6": pk["S2_pos8"][j6]}
            arms["ORACLE"] = {"p8_6": plan_for(r_star, j6)}
            cr = plan_for(np.ones(len(all2), np.int64), all2)
            arms["CRUISE"] = {"p8_2": cr, "p8_6": cr[j6]}
            taus = json.loads((RC.SCRATCH / "fits" / f"{tag}_tau.json").read_text(encoding="utf-8"))
            with torch.no_grad():
                lg = torch.from_numpy(o["regime_logits"])
                dec_tau = R.decide(lg, float(taus["tau"])).numpy()
                ds_soft = (lg.softmax(-1)[:, :, None] * prof).sum(1)
                soft = R.retime(torch.from_numpy(np.ascontiguousarray(plan2, np.float32)), R.monotone_progress(s_p + ds_soft)).to(torch.float32).numpy()
            tp = plan_for(dec_tau, all2)
            arms["TAU"] = {"p8_2": tp, "p8_6": tp[j6], "tau": float(taus["tau"])}
            arms["SOFT"] = {"p8_2": soft, "p8_6": soft[j6]}
            hyb2 = np.where((dec2 == 1)[:, None, None], H["S2_pos8"], pk["S2_pos8"]).astype(np.float32)
            hyb6 = np.where((dec2[j6] == 1)[:, None], H["S6_pos"].reshape(len(j6), -1), pk["S2_pos8"][j6][:, RC.SLOTS["S6"]].reshape(len(j6), -1))
            arms["HYB"] = {"p8_2": hyb2, "pos6_direct": hyb6.reshape(len(j6), -1, 2).astype(np.float32)}
            for a, d in arms.items():
                pos6 = d["pos6_direct"] if "pos6_direct" in d else d["p8_6"][:, RC.SLOTS["S6"]]
                pos2 = d["p8_2"][:, RC.SLOTS["S2"]] if "p8_2" in d else None
                cA = comps(pos6, pos2)
                fdA, mdA = L2.lon_indicators(np.asarray(pos6, np.float32), E, comp6)[:2]
                r_ = {"MD_rate": float(np.nanmean(mdA)), "FD_rate": float(np.nanmean(fdA)), "MD_vs_H8": cell(PR, mdA, mdH, eid6),
                      "FD_vs_H8": cell(PR, fdA, fdH, eid6),
                      "ADE_S6_vs_H8": cell(PR, cA["S6"]["ade_m"][comp6], cH["S6"]["ade_m"][comp6], em6),
                      "MD_vs_FX": cell(PR, mdA, mdF, eid6), "FD_vs_FX": cell(PR, fdA, fdF, eid6),
                      "ADE_S6_vs_D8": cell(PR, cA["S6"]["ade_m"][comp6], cD["S6"]["ade_m"][comp6], em6),
                      "ADE_S6_vs_HP": cell(PR, cA["S6"]["ade_m"][comp6], cHP["S6"]["ade_m"][comp6], em6)}
                if "tau" in d:
                    r_["tau"] = d["tau"]
                worse = []
                for sf, msk, em in (("S6", comp6, em6), ("S2", m2, em2)):
                    if sf not in cA:
                        continue
                    for fam, key, lib in METRICS:
                        if key in cA[sf]:
                            c = cell(PR, cA[sf][key][msk], cH[sf][key][msk], em)
                            if (lib and sep_above(c)) or (not lib and sep_below(c)):
                                worse.append(f"{sf}|{fam}|{key}: {fmt(c)}")
                if "p8_2" in d:
                    r_["ADE_S2_vs_H8"] = cell(PR, cA["S2"]["ade_m"][m2], cH["S2"]["ade_m"][m2], em2)
                    r_["ADE_S2_vs_D8"] = cell(PR, cA["S2"]["ade_m"][m2], cD["S2"]["ade_m"][m2], em2)
                    r_["ADE_S2_vs_HP"] = cell(PR, cA["S2"]["ade_m"][m2], cHP["S2"]["ade_m"][m2], em2)
                    ex = exceed(d["p8_2"])
                    r_["EXC_vs_H8"] = cell(PR, ex, excH, eid2)
                    if sep_above(r_["EXC_vs_H8"]):
                        worse.append(f"exceedance: {fmt(r_['EXC_vs_H8'])}")
                r_["rows_separated_worse_than_H8"] = worse
                r_["barlike_EXPLORATORY"] = {
                    "PA-1a": bool(sep_below(r_["MD_vs_H8"]) and not sep_above(r_["FD_vs_H8"])),
                    "PA-1b": bool(sep_below(r_["FD_vs_FX"]) and not sep_above(r_["MD_vs_FX"])),
                    "PA-2": (bool(sep_below(r_["ADE_S2_vs_D8"]) and sep_below(r_["ADE_S6_vs_D8"]) and sep_below(r_["ADE_S2_vs_HP"])
                                  and sep_below(r_["ADE_S6_vs_HP"])) if "ADE_S2_vs_D8" in r_ else None),
                    "PA-G": (not worse) if "p8_2" in d else None}
                r_["n_rows_worse"] = len(worse)
                r_["surfaces_scored"] = sorted(cA)
                out["cells"].setdefault(tag, {}).setdefault(a, {})[f"s{s}"] = r_
            out["cells"][tag].setdefault("decision_vs_r_star_S6_compliant", {})[f"s{s}"] = {
                "confusion_rows_rstar_cols_decision": [[int(((r_star == i) & (dec2[j6] == jj) & comp6).sum()) for jj in range(3)] for i in range(3)]}
        out["H8_rates"] = out.get("H8_rates", {}) | {f"s{s}": {"MD_rate": float(np.nanmean(mdH)), "FD_rate": float(np.nanmean(fdH))}}
        print(f"[xattr] seed {s} done ({time.time() - t0:.0f} s)", flush=True)
    out["controls"]["ALL_PASS"] = all(v["PASS"] for k, v in out["controls"].items() if k.startswith("K-"))
    out["script_sha256"] = RC.sha256_file(__file__)
    out["wall_s"] = round(time.time() - t0, 1)
    out["generated"] = time.strftime("%FT%T%z")
    RC.jdump(out, RC.RESULT / "R9PF_X_ATTRIB.json")
    print(f"[xattr] controls ALL_PASS={out['controls']['ALL_PASS']}  ZZXATTR{'OK' if out['controls']['ALL_PASS'] else 'FAIL'}ZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
