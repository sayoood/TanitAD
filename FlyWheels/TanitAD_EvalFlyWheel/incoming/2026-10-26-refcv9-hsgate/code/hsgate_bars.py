"""hsgate_bars.py -- PREREG_HSGATE (2331e1f3...) sections 2-3, PHASE 2: bars G1-G4 (+ G3 read from the SHIP-R8 verdict), the controls that are not in the
forward, and the beside rows.  CPU only; pure npz / json in, result/HSGATE_VERDICT.{json,md} out.  NEW file (refcv9-hsgate agent, 2026-10-26).

MD / FD EXACTLY as refcv9_l2c/code/h2cn_fdmd_diag.py / ship_r8_sub/bars_4c.py: E = l2_common.load_eval_pf(seed); comp = E['compliant'];
fd, md = l2_common.lon_indicators(S6_pos, E, comp); cells = pair_read_r8.paired_cell(new, ref, eid) (paired episode-cluster bootstrap B 2000, bootstrap
seed 0, over EPISODES, models fixed).  Every cell is S6 compliant and paired on the SAME draw (the battery tree's inference seed s).
  G1  MD(HSGATE8) - MD(H2CN_DEPLOY8) separated BELOW 0 (CI hi < 0)                                at BOTH inference seeds
  G2  FD(HSGATE8) - FD(H2CN_DEPLOY8) NOT separated above 0 (NOT CI lo > 0)                         at BOTH inference seeds
  G3  SHIP_R8_HSGATE8.json OVERALL == PASS
  G4  vs MIX_r (per window r * I_F + (1 - r) * I_H, r = HSGATE8's F-rate on the S6 compliant windows of that seed, deterministic):
      (MD(HS) - MD(MIX) separated below OR FD(HS) - FD(MIX) separated below) AND neither separated above      at BOTH inference seeds
Controls (a failed control VOIDS the bars):
  K-FWD      the --hsgate forward is ALL_PASS (K_REG_HSGATE, K_FIT_HSGATE, K_ID0, K_ID1, K_EVALFEAT, K_LANDED, K_GATE_SURF, K_TRACE, K_FEED + the A4 set)
  K-ALIGN    every arm package's S6 ws / clip == load_eval_pf's rows
  K-PKG-GATE the HSGATE8 package's S6 positions == FACTORED's where gate else H2CN's (bit for bit) -- the emitted positions ARE the gated pick
  K-DIAG     H2CN_DEPLOY8 - RETIME_DEPLOY8 FD / MD reproduce H2CN_FDMD_DIAG.txt (4 dp + n)
  K-ZERO     H2CN_DEPLOY8 - H2CN_DEPLOY8 reads delta 0, CI [0, 0]
  K-MIX      MIX with r := 0 equals I_H exactly and with r := 1 equals I_F exactly (analytic identity of the construction)
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hsgate_common as HG  # noqa: E402
from deviations_hsgate import DEVIATIONS  # noqa: E402

TREE = "step48852ema_pf"
OUT = HG.WORK / "out"
ARMS = OUT / "arms" / "h2cn_hsgate"
FWD = OUT / "forward" / f"FORWARD_H2CN_HSGATE_{TREE}.json"
VERDICT = OUT / "ship_r8" / "SHIP_R8_HSGATE8.json"
RESULT = HG.WORK / "result"
DIAG_TXT = Path("C:/Users/Admin/refcv8_pf/refcv9_l2c/result/H2CN_FDMD_DIAG.txt")
L2_DIR = Path("C:/Users/Admin/refcv8_pf/refcv9_l2/code")
FWD_4B = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub/out_4b/forward") / f"FORWARD_H2CN_CAP8_{TREE}.json"
READ_4D = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub_fs/out_4d/READ_4D.json")
ARM_LIST = ("HSGATE8", "H2CN_DEPLOY8", "FACTORED_DEPLOY8", "RETIME_DEPLOY8")
FAMILIES = {"ADE": ("ade_m", "fde_m"),
            "LONGITUDINAL": ("LON_speed_mae_mps", "LON_along_mae_m", "LON_accel_mae_mps2", "LON_along_abs_final_m", "LON_speed_within_0p5"),
            "LATERAL": ("LAT_cross_mae_m", "LAT_heading_mae_deg", "LAT_yaw_rate_mae_radps_valid", "LAT_curvature_mae_1pm_valid", "LAT_cross_abs_final_m"),
            "TACTICAL": ("TAC_traj_lat_correct", "TAC_traj_lon_correct")}
STRATEGIC_NA = ("n/a: eval139 carries no strategic decision / route-goal label this battery reads (SHIP-R8 clause 6 is N/A in the registered verdict); "
                "the gate does not touch the route input")


def cell(PR, a, b, eid) -> dict:
    c = PR.paired_cell(a, b, eid)
    keep = ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "mean_new", "mean_ref", "n_dropped_nonfinite", "status")
    return {k: (float(c[k]) if isinstance(c.get(k), (float, np.floating)) else c.get(k)) for k in keep if k in c}


def sep_below(c) -> bool:
    return c.get("hi") is not None and c["hi"] < 0


def sep_above(c) -> bool:
    return c.get("lo") is not None and c["lo"] > 0


def fmt(c) -> str:
    return (f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" SEP" if (c["hi"] < 0 or c["lo"] > 0) else "")
            + f" ({c['mean_new']:.4f} vs {c['mean_ref']:.4f}, n={c['n_windows']}/{c['n_episodes']} ep)")


def parse_diag() -> dict:
    out = {}
    pat = re.compile(r"^s(\d) (\S+)\s+(FD|MD): ([0-9.]+) vs RETIME ([0-9.]+)\s+delta ([+-][0-9.]+) \[([+-][0-9.]+), ([+-][0-9.]+)\] n=(\d+)")
    for line in DIAG_TXT.read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            out[(int(m.group(1)), m.group(2), m.group(3))] = {"mean_new": m.group(4), "mean_ref": m.group(5), "delta": m.group(6), "lo": m.group(7),
                                                               "hi": m.group(8), "n": int(m.group(9))}
    return out


def main() -> int:
    t0 = time.time()
    prereg = HG.check_prereg()
    sys.path.insert(0, str(L2_DIR))
    import l2_common as L2
    st_ = L2.stack()
    PR, ra = st_["PR"], st_["ra"]
    fd = json.loads(FWD.read_text(encoding="utf-8"))
    ver = json.loads(VERDICT.read_text(encoding="utf-8")) if VERDICT.exists() else None
    reg_g = HG.first_token(HG.WORK / "REGISTERED_HSGATE.sha256")
    gate_now = HG.sha256_file(HG.WORK / "HSGATE_FROZEN.json")
    ctl = {"K-FWD": bool((fd.get("controls") or {}).get("ALL_PASS") is True and fd.get("hsgate") and (fd["hsgate"].get("gate_sha256") == reg_g == gate_now)
                         and len(reg_g) == 64)}
    diag = parse_diag()
    per_seed, k_align, k_pkg, k_diag, k_zero, k_mix = {}, {}, {}, {}, {}, {}
    for s in (0, 1):
        E = L2.load_eval_pf(s)
        comp = np.asarray(E["compliant"], bool)
        eid = list(E["eid"])
        P, Z = {}, {}
        for arm in ARM_LIST:
            z = np.load(ARMS / arm / f"DEPLOY8_POS_{TREE}_s{s}.npz", allow_pickle=False)
            k_align[f"s{s}|{arm}"] = bool(np.array_equal(z["S6_ws"], E["ws"]) and np.array_equal(z["S6_clip"], E["clip"]))
            P[arm], Z[arm] = z["S6_pos"], z
        gate6 = np.asarray(Z["HSGATE8"]["S6_gate"], bool)
        k_pkg[f"s{s}"] = bool(np.array_equal(P["HSGATE8"], np.where(gate6[:, None, None], P["FACTORED_DEPLOY8"], P["H2CN_DEPLOY8"])))
        ind = {arm: L2.lon_indicators(P[arm], E, comp) for arm in ARM_LIST}
        fdI = {a: ind[a][0] for a in ARM_LIST}
        mdI = {a: ind[a][1] for a in ARM_LIST}
        r = float(gate6[comp].mean())
        mix = lambda I_F, I_H, rr: rr * I_F + (1.0 - rr) * I_H           # noqa: E731   (NaN outside the population on both sides)
        fd_mix = mix(fdI["FACTORED_DEPLOY8"], fdI["H2CN_DEPLOY8"], r)
        md_mix = mix(mdI["FACTORED_DEPLOY8"], mdI["H2CN_DEPLOY8"], r)
        k_mix[f"s{s}"] = bool(np.array_equal(mix(fdI["FACTORED_DEPLOY8"], fdI["H2CN_DEPLOY8"], 0.0), fdI["H2CN_DEPLOY8"], equal_nan=True)
                              and np.array_equal(mix(mdI["FACTORED_DEPLOY8"], mdI["H2CN_DEPLOY8"], 1.0), mdI["FACTORED_DEPLOY8"], equal_nan=True)
                              and np.array_equal(mix(mdI["FACTORED_DEPLOY8"], mdI["H2CN_DEPLOY8"], 0.0), mdI["H2CN_DEPLOY8"], equal_nan=True)
                              and np.array_equal(mix(fdI["FACTORED_DEPLOY8"], fdI["H2CN_DEPLOY8"], 1.0), fdI["FACTORED_DEPLOY8"], equal_nan=True))
        c = {"MD_HS_vs_H8": cell(PR, mdI["HSGATE8"], mdI["H2CN_DEPLOY8"], eid), "FD_HS_vs_H8": cell(PR, fdI["HSGATE8"], fdI["H2CN_DEPLOY8"], eid),
             "MD_HS_vs_MIX": cell(PR, mdI["HSGATE8"], md_mix, eid), "FD_HS_vs_MIX": cell(PR, fdI["HSGATE8"], fd_mix, eid),
             "MD_HS_vs_FX": cell(PR, mdI["HSGATE8"], mdI["FACTORED_DEPLOY8"], eid), "FD_HS_vs_FX": cell(PR, fdI["HSGATE8"], fdI["FACTORED_DEPLOY8"], eid),
             "MD_HS_vs_RETIME": cell(PR, mdI["HSGATE8"], mdI["RETIME_DEPLOY8"], eid), "FD_HS_vs_RETIME": cell(PR, fdI["HSGATE8"], fdI["RETIME_DEPLOY8"], eid),
             "MD_MIX_vs_H8": cell(PR, md_mix, mdI["H2CN_DEPLOY8"], eid), "FD_MIX_vs_H8": cell(PR, fd_mix, fdI["H2CN_DEPLOY8"], eid)}
        zc = cell(PR, mdI["H2CN_DEPLOY8"], mdI["H2CN_DEPLOY8"], eid)
        k_zero[f"s{s}"] = bool(zc["delta"] == 0.0 and zc["lo"] == 0.0 and zc["hi"] == 0.0)
        for k, j in (("FD", 0), ("MD", 1)):
            cc = cell(PR, ind["H2CN_DEPLOY8"][j], ind["RETIME_DEPLOY8"][j], eid)
            ref = diag.get((s, "H2CN_DEPLOY8", k))
            k_diag[f"s{s}|{k}"] = bool(ref is not None and f"{cc['mean_new']:.4f}" == ref["mean_new"] and f"{cc['mean_ref']:.4f}" == ref["mean_ref"]
                                      and f"{cc['delta']:+.4f}" == ref["delta"] and f"{cc['lo']:+.4f}" == ref["lo"] and f"{cc['hi']:+.4f}" == ref["hi"]
                                      and cc["n_windows"] == ref["n"])
        # ---- beside: ADE + the four families vs H2CN_DEPLOY8 and vs FACTORED_DEPLOY8 (S6 compliant)
        comps = {a: {k: np.asarray(v, np.float64) for k, v in PR.components(ra, np.asarray(P[a], np.float32), E["g"].astype(np.float32), E["dt"]).items()}
                 for a in ("HSGATE8", "H2CN_DEPLOY8", "FACTORED_DEPLOY8")}
        fam = {}
        for fname, keys in FAMILIES.items():
            fam[fname] = {}
            for key in keys:
                a_ = np.where(comp, comps["HSGATE8"][key], np.nan)
                fam[fname][key] = {"vs_H2CN_DEPLOY8": cell(PR, a_, np.where(comp, comps["H2CN_DEPLOY8"][key], np.nan), eid),
                                   "vs_FACTORED_DEPLOY8": cell(PR, a_, np.where(comp, comps["FACTORED_DEPLOY8"][key], np.nan), eid)}
        fam["TACTICAL"]["false_decel_FD"] = {"vs_H2CN_DEPLOY8": c["FD_HS_vs_H8"], "vs_FACTORED_DEPLOY8": c["FD_HS_vs_FX"]}
        fam["TACTICAL"]["missed_decel_MD"] = {"vs_H2CN_DEPLOY8": c["MD_HS_vs_H8"], "vs_FACTORED_DEPLOY8": c["MD_HS_vs_FX"]}
        fam["STRATEGIC"] = STRATEGIC_NA
        g1 = sep_below(c["MD_HS_vs_H8"])
        g2 = not sep_above(c["FD_HS_vs_H8"])
        g4 = bool((sep_below(c["MD_HS_vs_MIX"]) or sep_below(c["FD_HS_vs_MIX"])) and not sep_above(c["MD_HS_vs_MIX"]) and not sep_above(c["FD_HS_vs_MIX"]))
        gate2 = np.asarray(Z["HSGATE8"]["S2_gate"], bool)
        comp2 = np.asarray(Z["HSGATE8"]["S2_compliant"], bool)
        per_seed[str(s)] = {"F_rate_S6_compliant_r": r, "n_S6_compliant": int(comp.sum()), "n_F_S6_compliant": int(gate6[comp].sum()),
                            "F_rate_S6_all": float(gate6.mean()),
                            "beside_F_rate_S2_compliant": float(gate2[comp2].mean()), "beside_n_S2_compliant": int(comp2.sum()),
                            "beside_n_F_S2_compliant": int(gate2[comp2].sum()), "beside_F_rate_S2_all": float(gate2.mean()), "beside_n_S2_all": int(len(gate2)),
                            "cells": c, "families": fam, "G1": bool(g1), "G2": bool(g2), "G4": g4}
    ctl.update({"K-ALIGN": bool(all(k_align.values())), "K-PKG-GATE": bool(all(k_pkg.values())), "K-DIAG": bool(len(k_diag) == 4 and all(k_diag.values())),
                "K-ZERO": bool(all(k_zero.values())), "K-MIX": bool(all(k_mix.values()))})
    ctl_ok = bool(all(ctl.values()))
    g3 = bool(ver is not None and ver.get("overall") == "PASS")
    bars = {"G1": all(per_seed[s]["G1"] for s in ("0", "1")), "G2": all(per_seed[s]["G2"] for s in ("0", "1")), "G3": g3,
            "G4": all(per_seed[s]["G4"] for s in ("0", "1"))}
    if not ctl_ok:
        word = "VOID (a control failed: " + ", ".join(k for k, v in ctl.items() if not v) + ")"
        nxt = "fix the failing control; no bar is read"
    elif bars["G1"] and bars["G2"] and bars["G3"] and bars["G4"]:
        word, nxt = "PASS", ("HSGATE8 joins the shipped-stage choice for the PI (with the latency item: the self-sharpened decode at inference is UNMEASURED against the "
                             "100 ms budget); refcv9 design input: the progress head routes on the lon decision")
    elif bars["G1"] and bars["G2"] and bars["G3"]:
        word, nxt = "G1-G3 PASS, G4 FAIL: rate effect, not attributable", (
            "the gate learns a mixing RATE, not windows: these features cannot tell which progress is right; no further post-hoc progress stages on refcv8 -- the lever "
            "moves to refcv9 (a progress head trained with the lon decision as a training target on a clean split); the PI chooses between H2CN_DEPLOY8 / H2CN_CAP8 / "
            "FACTORED_DEPLOY8 on their four-family rows")
    elif not bars["G1"]:
        word, nxt = "FAIL (G1)", "G1 FAIL -> named; H2CN_DEPLOY8 stands"
    elif not bars["G2"]:
        word, nxt = "FAIL (G2)", ("operating-point trade: next = the same gate with its threshold chosen on FIT rows under an FD cap (FD_FIT(gate) <= FD_FIT(H2cn-OOF)), "
                                  "pre-registered first")
    else:
        word, nxt = "FAIL (G3)", "G3 FAIL -> the failing SHIP-R8 clause(s) named in SHIP_R8_HSGATE8.json; H2CN_DEPLOY8 stands"
    if ctl_ok and not bars["G4"] and bars["G1"] and bars["G2"] and not bars["G3"]:
        word += "; G4 also FAILS (rate effect, not attributable)"
    beside = {"F_rate_per_surface_seed": fd["hsgate"].get("F_rate"), "compliance_row": {}, "E1_E3": {}, "A4d_fit_seed_floor": None}
    for k_, v_ in ((fd.get("compliance_row") or {}).get("cells") or {}).items():
        beside["compliance_row"][k_] = {a: (v_["arms"].get(a) or {}).get("full_plan_0_6s", {}).get("compliant_set") for a in ("HSGATE8", "H2CN_DEPLOY8", "RETIME_DEPLOY8", "FACTORED_DEPLOY8")}
    if FWD_4B.exists():
        f4b = json.loads(FWD_4B.read_text(encoding="utf-8"))
        for k_, v_ in ((f4b.get("compliance_row") or {}).get("cells") or {}).items():
            beside["compliance_row"].setdefault(k_, {})["H2CN_CAP8"] = (v_["arms"].get("H2CN_CAP8") or {}).get("full_plan_0_6s", {}).get("compliant_set")
        beside["compliance_row_cap8_source"] = {"file": str(FWD_4B), "sha256": HG.sha256_file(FWD_4B)}
    if ver is not None:
        for c_ in ver["clauses"]:
            if str(c_["id"]) in ("2", "3"):
                beside["E1_E3"][f"HSGATE8|clause{c_['id']}"] = c_["evidence"].get("cells")
        for a_, b_ in (ver.get("beside") or {}).get("arms", {}).items():
            beside["E1_E3"][f"{a_}|E1"], beside["E1_E3"][f"{a_}|E3"] = b_.get("E1"), b_.get("E3")
    if READ_4D.exists():
        r4 = json.loads(READ_4D.read_text(encoding="utf-8"))
        beside["A4d_fit_seed_floor"] = {"file": str(READ_4D), "sha256": HG.sha256_file(READ_4D), "word": (r4.get("word") or {}).get("word"),
                                        "R3_vs_H2CN_DEPLOY8_seed0": r4.get("R3_vs_H2CN_DEPLOY8_seed0")}
    res = {"tool": "hsgate_bars.py", "prereg": prereg, "word": word, "next_step": nxt, "bars": bars, "per_seed": per_seed, "controls": ctl,
           "controls_detail": {"K-ALIGN": k_align, "K-PKG-GATE": k_pkg, "K-DIAG": k_diag, "K-ZERO": k_zero, "K-MIX": k_mix},
           "G3_verdict": {"file": str(VERDICT), "sha256": (HG.sha256_file(VERDICT) if VERDICT.exists() else None), "overall": (ver or {}).get("overall"),
                          "per_clause": (ver or {}).get("per_clause")},
           "beside": beside,
           "gate": {"file": str(HG.WORK / "HSGATE_FROZEN.json"), "sha256": gate_now, "registered": reg_g},
           "forward": {"file": str(FWD), "sha256": HG.sha256_file(FWD)},
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)",
           "estimator": "paired episode-cluster bootstrap B 2000, bootstrap seed 0, over EPISODES only (models fixed): answers 'another draw of episodes' only; "
                        "ONE refcv8 training seed; ONE gate fit; inference seeds 0 and 1 reported separately, never pooled",
           "disclosed_deviations": DEVIATIONS,
           "generated": time.strftime("%FT%T%z"), "wall_s": round(time.time() - t0, 1)}
    RESULT.mkdir(parents=True, exist_ok=True)
    HG.jdump(res, RESULT / "HSGATE_VERDICT.json")
    L = [f"# HSGATE_VERDICT -- {word}", "", f"tier: {res['tier']}", f"estimator: {res['estimator']}", "",
         "| bar | seed 0 | seed 1 | pass |", "|---|---|---|---|"]
    for b, key in (("G1 MD(HS)-MD(H2CN) sep below", "MD_HS_vs_H8"), ("G2 FD(HS)-FD(H2CN) not sep above", "FD_HS_vs_H8"),
                   ("G4a MD(HS)-MD(MIX_r)", "MD_HS_vs_MIX"), ("G4b FD(HS)-FD(MIX_r)", "FD_HS_vs_MIX")):
        L.append(f"| {b} | {fmt(per_seed['0']['cells'][key])} | {fmt(per_seed['1']['cells'][key])} | "
                 f"{bars['G1'] if key == 'MD_HS_vs_H8' else bars['G2'] if key == 'FD_HS_vs_H8' else bars['G4']} |")
    L += [f"| G3 SHIP-R8 OVERALL | {(ver or {}).get('overall')} | | {g3} |", "",
          f"F-rate (S6 compliant): s0 {per_seed['0']['F_rate_S6_compliant_r']:.4f} ({per_seed['0']['n_F_S6_compliant']}/{per_seed['0']['n_S6_compliant']}), "
          f"s1 {per_seed['1']['F_rate_S6_compliant_r']:.4f} ({per_seed['1']['n_F_S6_compliant']}/{per_seed['1']['n_S6_compliant']})", "",
          f"beside: S2 F-rate (compliant): s0 {per_seed['0']['beside_F_rate_S2_compliant']:.4f}, s1 {per_seed['1']['beside_F_rate_S2_compliant']:.4f}", "",
          f"controls: {json.dumps(ctl)}", "", f"NEXT: {nxt}", "", "## disclosed deviations", ""] + [f"* {d}" for d in DEVIATIONS] + [""]
    (RESULT / "HSGATE_VERDICT.md").write_text("\n".join(L), encoding="utf-8")
    chk = json.loads((RESULT / "HSGATE_VERDICT.json").read_text(encoding="utf-8"))
    assert chk["word"] == word and set(chk["bars"]) == {"G1", "G2", "G3", "G4"}
    print(f"[hsgate_bars] {word}  bars={bars} controls_ok={ctl_ok}  ZZHSGATEBARSZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
