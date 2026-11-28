"""mdfix_bars.py -- PREREG_HSGATE_MDFIX (545ff217...) s2-s3, PHASE B: bars G1-G4 (+ G3 read from SHIP_R8_MDFIX8.json), the controls that are not in the forward,
and the beside rows.  CPU only; npz / json in, result/MDFIX_VERDICT.{json,md} out.  NEW file (refcv9-hsgate agent, 2026-10-26), derived from hsgate_bars.py.

MD / FD exactly as hsgate_bars.py (l2_common.load_eval_pf(seed); comp = E['compliant']; l2_common.lon_indicators(S6_pos, E, comp)); paired episode-cluster
bootstrap B 2000, bootstrap seed 0, over EPISODES (models fixed); every cell S6 compliant, paired on the SAME draw (inference seed s).
  G1  MD(MDFIX8) - MD(H2CN_DEPLOY8) separated BELOW 0 on the 97.5 % interval (alpha 0.025)       at BOTH inference seeds   (95 % beside)
  G2  FD(MDFIX8) - FD(H2CN_DEPLOY8) NOT separated above 0 on the 95 % interval                    at BOTH inference seeds
  G3  SHIP_R8_MDFIX8.json OVERALL == PASS
  G4  PREREG_HSGATE s3 G4 with r = MDFIX8's own S6-compliant F-rate of that seed, 95 % intervals:
      (MD(MDF) - MD(MIX_r) separated below OR FD(MDF) - FD(MIX_r) separated below) AND neither separated above   at BOTH seeds
Controls (a failed control VOIDS the bars; no tolerance depends on the F-rate):
  K-FWD      the MDFIX forward is ALL_PASS, its gate sha256 == the EXACT REGISTERED_MDFIX line == the file now, and all four pins re-hash now
  K-ALIGN    every arm package's S6 ws / clip == load_eval_pf's rows (incl. the landed HSGATE8 package read beside)
  K-PKG-GATE MDFIX8's S6 positions == FACTORED's where gate else H2CN's (bit for bit)
  K-HS-LANDED the HSGATE8 packages read beside hash to their MANIFEST_HSGATE.sha256 lines
  K-DIAG     H2CN_DEPLOY8 - RETIME_DEPLOY8 FD / MD reproduce H2CN_FDMD_DIAG.txt (4 dp + n)
  K-ZERO     H2CN_DEPLOY8 - H2CN_DEPLOY8 reads delta 0, CI [0, 0]
  K-MIX      MIX with r := 0 / 1 equals I_H / I_F exactly
  K-CI975    the direct taniteval.ci call at alpha 0.05 reproduces pair_read_r8.paired_cell's interval exactly on the G1 cell, and the 97.5 % interval contains it
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
import mdfix_common as MC  # noqa: E402
from deviations_mdfix import DEVIATIONS, DISCLOSURES_S0  # noqa: E402

TREE = "step48852ema_pf"
OUT = HG.WORK / "out_mdfix"
ARMS = OUT / "arms" / "h2cn_mdfix"
HS_ARMS = HG.WORK / "out" / "arms" / "h2cn_hsgate"
HS_FWD = HG.WORK / "out" / "forward" / f"FORWARD_H2CN_HSGATE_{TREE}.json"
FWD = OUT / "forward" / f"FORWARD_H2CN_MDFIX_{TREE}.json"
VERDICT = OUT / "ship_r8" / "SHIP_R8_MDFIX8.json"
RESULT = HG.WORK / "result"
DIAG_TXT = Path("C:/Users/Admin/refcv8_pf/refcv9_l2c/result/H2CN_FDMD_DIAG.txt")
L2_DIR = Path("C:/Users/Admin/refcv8_pf/refcv9_l2/code")
FWD_4B = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub/out_4b/forward") / f"FORWARD_H2CN_CAP8_{TREE}.json"
READ_4D = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub_fs/out_4d/READ_4D.json")
MAN_HS = HG.WORK / "MANIFEST_HSGATE.sha256"
ARM_LIST = ("MDFIX8", "H2CN_DEPLOY8", "FACTORED_DEPLOY8", "RETIME_DEPLOY8", "HSGATE8")
FAMILIES = {"ADE": ("ade_m", "fde_m"),
            "LONGITUDINAL": ("LON_speed_mae_mps", "LON_along_mae_m", "LON_accel_mae_mps2", "LON_along_abs_final_m", "LON_speed_within_0p5"),
            "LATERAL": ("LAT_cross_mae_m", "LAT_heading_mae_deg", "LAT_yaw_rate_mae_radps_valid", "LAT_curvature_mae_1pm_valid", "LAT_cross_abs_final_m"),
            "TACTICAL": ("TAC_traj_lat_correct", "TAC_traj_lon_correct")}
STRATEGIC_NA = ("n/a: eval139 carries no strategic decision / route-goal label this battery reads (SHIP-R8 clause 6 is N/A in the registered verdict); "
                "the gate does not touch the route input")
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
ESTIMATOR = ("paired episode-cluster bootstrap B 2000, bootstrap seed 0, over EPISODES only (models fixed): answers 'another draw of episodes' only -- blind to "
             "training and inference variance; ONE refcv8 training seed; ONE gate fit; inference seeds 0 and 1 reported separately, never pooled")


def cell(PR, a, b, eid) -> dict:
    c = PR.paired_cell(a, b, eid)
    keep = ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "mean_new", "mean_ref", "n_dropped_nonfinite", "status")
    return {k: (float(c[k]) if isinstance(c.get(k), (float, np.floating)) else c.get(k)) for k in keep if k in c}


def cell_alpha(a, b, eid, alpha) -> dict:
    """the SAME estimator as pair_read_r8.paired_cell (finite-on-both-sides windows, taniteval.ci paired bootstrap, B 2000, seed 0) at a given alpha."""
    from taniteval import ci
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    r = ci.paired_episode_cluster_bootstrap(a[keep], b[keep], [x for x, k in zip(eid, keep) if k], n_boot=2000, seed=0, alpha=alpha)
    return {"delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]), "n_windows": int(r["n_windows"]),
            "n_episodes": int(r["n_episodes"]), "alpha": alpha, "mean_new": float(a[keep].mean()), "mean_ref": float(b[keep].mean())}


def sep_below(c) -> bool:
    return bool(c.get("separated") and c.get("hi") is not None and c["hi"] < 0)


def sep_above(c) -> bool:
    return bool(c.get("separated") and c.get("lo") is not None and c["lo"] > 0)


def fmt(c) -> str:
    return (f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" SEP" if c.get("separated") else "")
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


def manifest_line(man: Path, rel: str) -> str:
    for ln in man.read_text(encoding="utf-8").splitlines():
        p = ln.split(None, 1)
        if len(p) == 2 and p[1].lstrip("*").strip() == rel and len(p[0]) == 64:
            return p[0]
    return ""


def main() -> int:
    t0 = time.time()
    prereg = HG.check_prereg()
    prereg_m = MC.check_prereg_mdfix()
    pins = MC.check_registration()                                      # SystemExit unless every pin re-hashes now
    sys.path.insert(0, str(L2_DIR))
    import l2_common as L2
    st_ = L2.stack()
    PR, ra = st_["PR"], st_["ra"]
    fd = json.loads(FWD.read_text(encoding="utf-8"))
    ver = json.loads(VERDICT.read_text(encoding="utf-8")) if VERDICT.exists() else None
    reg_g = MC.registered_sha("MDFIX_FROZEN.json")
    gate_now = HG.sha256_file(HG.WORK / "MDFIX_FROZEN.json")
    ctl = {"K-FWD": bool((fd.get("controls") or {}).get("ALL_PASS") is True and fd.get("hsgate") and len(reg_g) == 64
                         and (fd["hsgate"].get("gate_sha256") == reg_g == gate_now) and pins.get("ALL_MATCH") is True
                         and fd["hsgate"].get("prereg_mdfix_sha256") == prereg_m["sha256"])}
    diag = parse_diag()
    per_seed, k_align, k_pkg, k_diag, k_zero, k_mix, k_ci, k_hs = {}, {}, {}, {}, {}, {}, {}, {}
    for s in (0, 1):
        E = L2.load_eval_pf(s)
        comp = np.asarray(E["compliant"], bool)
        eid = list(E["eid"])
        P, Z = {}, {}
        for arm in ARM_LIST:
            f = (HS_ARMS if arm == "HSGATE8" else ARMS) / arm / f"DEPLOY8_POS_{TREE}_s{s}.npz"
            z = np.load(f, allow_pickle=False)
            k_align[f"s{s}|{arm}"] = bool(np.array_equal(z["S6_ws"], E["ws"]) and np.array_equal(z["S6_clip"], E["clip"]))
            P[arm], Z[arm] = z["S6_pos"], z
            if arm == "HSGATE8":
                rel = f"out/arms/h2cn_hsgate/HSGATE8/DEPLOY8_POS_{TREE}_s{s}.npz"
                m_, now_ = manifest_line(MAN_HS, rel), HG.sha256_file(f)
                k_hs[f"s{s}"] = {"manifest": m_, "now": now_, "PASS": bool(len(m_) == 64 and m_ == now_)}
        gate6 = np.asarray(Z["MDFIX8"]["S6_gate"], bool)
        k_pkg[f"s{s}"] = bool(np.array_equal(P["MDFIX8"], np.where(gate6[:, None, None], P["FACTORED_DEPLOY8"], P["H2CN_DEPLOY8"])))
        ind = {arm: L2.lon_indicators(P[arm], E, comp) for arm in ARM_LIST}
        fdI = {a: ind[a][0] for a in ARM_LIST}
        mdI = {a: ind[a][1] for a in ARM_LIST}
        r = float(gate6[comp].mean())
        mix = lambda I_F, I_H, rr: rr * I_F + (1.0 - rr) * I_H           # noqa: E731
        fd_mix = mix(fdI["FACTORED_DEPLOY8"], fdI["H2CN_DEPLOY8"], r)
        md_mix = mix(mdI["FACTORED_DEPLOY8"], mdI["H2CN_DEPLOY8"], r)
        k_mix[f"s{s}"] = bool(np.array_equal(mix(fdI["FACTORED_DEPLOY8"], fdI["H2CN_DEPLOY8"], 0.0), fdI["H2CN_DEPLOY8"], equal_nan=True)
                              and np.array_equal(mix(mdI["FACTORED_DEPLOY8"], mdI["H2CN_DEPLOY8"], 1.0), mdI["FACTORED_DEPLOY8"], equal_nan=True)
                              and np.array_equal(mix(mdI["FACTORED_DEPLOY8"], mdI["H2CN_DEPLOY8"], 0.0), mdI["H2CN_DEPLOY8"], equal_nan=True)
                              and np.array_equal(mix(fdI["FACTORED_DEPLOY8"], fdI["H2CN_DEPLOY8"], 1.0), fdI["FACTORED_DEPLOY8"], equal_nan=True))
        c = {"MD_MDF_vs_H8": cell(PR, mdI["MDFIX8"], mdI["H2CN_DEPLOY8"], eid), "FD_MDF_vs_H8": cell(PR, fdI["MDFIX8"], fdI["H2CN_DEPLOY8"], eid),
             "MD_MDF_vs_MIX": cell(PR, mdI["MDFIX8"], md_mix, eid), "FD_MDF_vs_MIX": cell(PR, fdI["MDFIX8"], fd_mix, eid),
             "MD_MDF_vs_HS": cell(PR, mdI["MDFIX8"], mdI["HSGATE8"], eid), "FD_MDF_vs_HS": cell(PR, fdI["MDFIX8"], fdI["HSGATE8"], eid),
             "MD_MDF_vs_FX": cell(PR, mdI["MDFIX8"], mdI["FACTORED_DEPLOY8"], eid), "FD_MDF_vs_FX": cell(PR, fdI["MDFIX8"], fdI["FACTORED_DEPLOY8"], eid),
             "MD_MDF_vs_RETIME": cell(PR, mdI["MDFIX8"], mdI["RETIME_DEPLOY8"], eid), "FD_MDF_vs_RETIME": cell(PR, fdI["MDFIX8"], fdI["RETIME_DEPLOY8"], eid),
             "MD_MIX_vs_H8": cell(PR, md_mix, mdI["H2CN_DEPLOY8"], eid), "FD_MIX_vs_H8": cell(PR, fd_mix, fdI["H2CN_DEPLOY8"], eid)}
        g1_975 = cell_alpha(mdI["MDFIX8"], mdI["H2CN_DEPLOY8"], eid, 0.025)
        g1_95d = cell_alpha(mdI["MDFIX8"], mdI["H2CN_DEPLOY8"], eid, 0.05)
        b95 = c["MD_MDF_vs_H8"]
        k_ci[f"s{s}"] = bool(g1_95d["lo"] == b95["lo"] and g1_95d["hi"] == b95["hi"] and g1_95d["delta"] == b95["delta"] and g1_95d["n_windows"] == b95["n_windows"]
                             and g1_975["lo"] <= g1_95d["lo"] and g1_975["hi"] >= g1_95d["hi"])
        c["MD_MDF_vs_H8_975"] = g1_975
        zc = cell(PR, mdI["H2CN_DEPLOY8"], mdI["H2CN_DEPLOY8"], eid)
        k_zero[f"s{s}"] = bool(zc["delta"] == 0.0 and zc["lo"] == 0.0 and zc["hi"] == 0.0)
        for k, j in (("FD", 0), ("MD", 1)):
            cc = cell(PR, ind["H2CN_DEPLOY8"][j], ind["RETIME_DEPLOY8"][j], eid)
            ref = diag.get((s, "H2CN_DEPLOY8", k))
            k_diag[f"s{s}|{k}"] = bool(ref is not None and f"{cc['mean_new']:.4f}" == ref["mean_new"] and f"{cc['mean_ref']:.4f}" == ref["mean_ref"]
                                      and f"{cc['delta']:+.4f}" == ref["delta"] and f"{cc['lo']:+.4f}" == ref["lo"] and f"{cc['hi']:+.4f}" == ref["hi"]
                                      and cc["n_windows"] == ref["n"])
        comps = {a: {k: np.asarray(v, np.float64) for k, v in PR.components(ra, np.asarray(P[a], np.float32), E["g"].astype(np.float32), E["dt"]).items()}
                 for a in ("MDFIX8", "H2CN_DEPLOY8", "FACTORED_DEPLOY8", "HSGATE8")}
        fam = {}
        for fname, keys in FAMILIES.items():
            fam[fname] = {}
            for key in keys:
                a_ = np.where(comp, comps["MDFIX8"][key], np.nan)
                fam[fname][key] = {f"vs_{ref_}": cell(PR, a_, np.where(comp, comps[ref_][key], np.nan), eid)
                                   for ref_ in ("H2CN_DEPLOY8", "HSGATE8", "FACTORED_DEPLOY8")}
        fam["TACTICAL"]["false_decel_FD"] = {"vs_H2CN_DEPLOY8": c["FD_MDF_vs_H8"], "vs_HSGATE8": c["FD_MDF_vs_HS"], "vs_FACTORED_DEPLOY8": c["FD_MDF_vs_FX"]}
        fam["TACTICAL"]["missed_decel_MD"] = {"vs_H2CN_DEPLOY8": c["MD_MDF_vs_H8"], "vs_HSGATE8": c["MD_MDF_vs_HS"], "vs_FACTORED_DEPLOY8": c["MD_MDF_vs_FX"]}
        fam["STRATEGIC"] = STRATEGIC_NA
        g1 = sep_below(g1_975)
        g2 = not sep_above(c["FD_MDF_vs_H8"])
        g4 = bool((sep_below(c["MD_MDF_vs_MIX"]) or sep_below(c["FD_MDF_vs_MIX"])) and not sep_above(c["MD_MDF_vs_MIX"]) and not sep_above(c["FD_MDF_vs_MIX"]))
        gate2 = np.asarray(Z["MDFIX8"]["S2_gate"], bool)
        comp2 = np.asarray(Z["MDFIX8"]["S2_compliant"], bool)
        per_seed[str(s)] = {"F_rate_S6_compliant_r": r, "n_S6_compliant": int(comp.sum()), "n_F_S6_compliant": int(gate6[comp].sum()),
                            "F_rate_S6_all": float(gate6.mean()), "n_F_S6_all": int(gate6.sum()),
                            "beside_F_rate_S2_compliant": float(gate2[comp2].mean()), "beside_n_S2_compliant": int(comp2.sum()),
                            "beside_n_F_S2_compliant": int(gate2[comp2].sum()), "beside_F_rate_S2_all": float(gate2.mean()), "beside_n_S2_all": int(len(gate2)),
                            "beside_HSGATE8_F_rate_S6_compliant": float(np.asarray(Z["HSGATE8"]["S6_gate"], bool)[comp].mean()),
                            "beside_gated_windows_shared_with_HSGATE8_S6_compliant": int((gate6 & np.asarray(Z["HSGATE8"]["S6_gate"], bool))[comp].sum()),
                            "cells": c, "families": fam, "G1": bool(g1), "G1_beside_95": bool(sep_below(c["MD_MDF_vs_H8"])), "G2": bool(g2), "G4": g4}
    ctl.update({"K-ALIGN": bool(all(k_align.values())), "K-PKG-GATE": bool(all(k_pkg.values())), "K-HS-LANDED": bool(all(v["PASS"] for v in k_hs.values())),
                "K-DIAG": bool(len(k_diag) == 4 and all(k_diag.values())), "K-ZERO": bool(all(k_zero.values())), "K-MIX": bool(all(k_mix.values())),
                "K-CI975": bool(all(k_ci.values()))})
    ctl_ok = bool(all(ctl.values()))
    g3 = bool(ver is not None and ver.get("overall") == "PASS")
    bars = {"G1": all(per_seed[s]["G1"] for s in ("0", "1")), "G2": all(per_seed[s]["G2"] for s in ("0", "1")), "G3": g3,
            "G4": all(per_seed[s]["G4"] for s in ("0", "1"))}
    failing = [k for k in ("G1", "G2", "G3", "G4") if not bars[k]]
    end_line = ("the refcv8 post-hoc progress line ENDS; the lever is refcv9's trained lon decision (PREREG_LONHEAD_C, in flight); no third relabel of this gate "
                "family is admissible on eval139")
    if not ctl_ok:
        word, nxt = "VOID (a control failed: " + ", ".join(k for k, v in ctl.items() if not v) + ")", "fix the failing control; no bar is read"
    elif not failing:
        word, nxt = "PASS (second look, multiplicity-adjusted)", ("MDFIX8 joins the PI's shipped-stage choice, labelled 'second look, multiplicity-adjusted', "
                                                                   "with the latency item (the self-sharpened decode at inference is UNMEASURED against the budget)")
    else:
        word = "FAIL (" + ", ".join(failing) + ")"
        if not bars["G2"]:
            nxt = "s5 G2 FAIL: as UNDERPOWERED -- " + end_line + "; richer gate inputs are an architecture decision (PI / MM), not another relabel"
        elif not bars["G1"]:
            nxt = "s5 G1 FAIL (G2 PASS): the FIT-side MD gain does not transfer at this F-rate; H2CN_DEPLOY8 stands; " + end_line
        elif not bars["G4"]:
            nxt = "s5 G4 FAIL: rate effect, not attributable; H2CN_DEPLOY8 stands; " + end_line
        else:
            nxt = ("G3 FAIL (not separately enumerated in s5): the failing SHIP-R8 clause(s) are named in SHIP_R8_MDFIX8.json; H2CN_DEPLOY8 stands pending the "
                   "Master Mind")
    beside = {"F_rate_per_surface_seed": fd["hsgate"].get("F_rate"), "compliance_row": {}, "E1_E3": {}, "A4d_fit_seed_floor": None}
    for k_, v_ in ((fd.get("compliance_row") or {}).get("cells") or {}).items():
        beside["compliance_row"][k_] = {a: (v_["arms"].get(a) or {}).get("full_plan_0_6s", {}).get("compliant_set") for a in ("MDFIX8", "H2CN_DEPLOY8", "RETIME_DEPLOY8", "FACTORED_DEPLOY8")}
    for src, arm_ in ((FWD_4B, "H2CN_CAP8"), (HS_FWD, "HSGATE8")):
        if src.exists():
            fj = json.loads(src.read_text(encoding="utf-8"))
            for k_, v_ in ((fj.get("compliance_row") or {}).get("cells") or {}).items():
                beside["compliance_row"].setdefault(k_, {})[arm_] = (v_["arms"].get(arm_) or {}).get("full_plan_0_6s", {}).get("compliant_set")
            beside[f"compliance_row_{arm_}_source"] = {"file": str(src), "sha256": HG.sha256_file(src)}
    if ver is not None:
        for c_ in ver["clauses"]:
            if str(c_["id"]) in ("2", "3"):
                beside["E1_E3"][f"MDFIX8|clause{c_['id']}"] = c_["evidence"].get("cells")
        for a_, b_ in (ver.get("beside") or {}).get("arms", {}).items():
            beside["E1_E3"][f"{a_}|E1"], beside["E1_E3"][f"{a_}|E3"] = b_.get("E1"), b_.get("E3")
    if READ_4D.exists():
        r4 = json.loads(READ_4D.read_text(encoding="utf-8"))
        beside["A4d_fit_seed_floor"] = {"file": str(READ_4D), "sha256": HG.sha256_file(READ_4D), "word": (r4.get("word") or {}).get("word"),
                                        "R3_vs_H2CN_DEPLOY8_seed0": r4.get("R3_vs_H2CN_DEPLOY8_seed0")}
    res = {"tool": "mdfix_bars.py", "prereg": prereg_m, "prereg_parent": prereg, "prereg_s0_disclosures": DISCLOSURES_S0, "word": word, "next_step": nxt,
           "bars": bars, "bar_levels": {"G1": "97.5 % (95 % beside)", "G2": "95 %", "G4": "95 %"}, "per_seed": per_seed, "controls": ctl,
           "controls_detail": {"K-ALIGN": k_align, "K-PKG-GATE": k_pkg, "K-HS-LANDED": k_hs, "K-DIAG": k_diag, "K-ZERO": k_zero, "K-MIX": k_mix, "K-CI975": k_ci},
           "registration": pins,
           "G3_verdict": {"file": str(VERDICT), "sha256": (HG.sha256_file(VERDICT) if VERDICT.exists() else None), "overall": (ver or {}).get("overall"),
                          "per_clause": (ver or {}).get("per_clause")},
           "beside": beside, "gate": {"file": str(HG.WORK / "MDFIX_FROZEN.json"), "sha256": gate_now, "registered": reg_g, "tau": fd["hsgate"].get("tau")},
           "forward": {"file": str(FWD), "sha256": HG.sha256_file(FWD)}, "tier": TIER, "estimator": ESTIMATOR, "disclosed_deviations": DEVIATIONS,
           "generated": time.strftime("%FT%T%z"), "wall_s": round(time.time() - t0, 1)}
    RESULT.mkdir(parents=True, exist_ok=True)
    HG.jdump(res, RESULT / "MDFIX_VERDICT.json")
    ps = per_seed
    L = [f"# MDFIX_VERDICT -- {word}", "", "## prereg s0 disclosures", ""] + [f"* {d}" for d in DISCLOSURES_S0] + [
        "", f"tier: {TIER}", f"estimator: {ESTIMATOR}", "",
        "| bar | seed 0 | seed 1 | pass |", "|---|---|---|---|",
        f"| G1 MD(MDF)-MD(H2CN) sep below @97.5 % | {fmt(ps['0']['cells']['MD_MDF_vs_H8_975'])} | {fmt(ps['1']['cells']['MD_MDF_vs_H8_975'])} | {bars['G1']} |",
        f"| (beside) G1 cell @95 % | {fmt(ps['0']['cells']['MD_MDF_vs_H8'])} | {fmt(ps['1']['cells']['MD_MDF_vs_H8'])} | |",
        f"| G2 FD(MDF)-FD(H2CN) not sep above @95 % | {fmt(ps['0']['cells']['FD_MDF_vs_H8'])} | {fmt(ps['1']['cells']['FD_MDF_vs_H8'])} | {bars['G2']} |",
        f"| G4a MD(MDF)-MD(MIX_r) | {fmt(ps['0']['cells']['MD_MDF_vs_MIX'])} | {fmt(ps['1']['cells']['MD_MDF_vs_MIX'])} | {bars['G4']} |",
        f"| G4b FD(MDF)-FD(MIX_r) | {fmt(ps['0']['cells']['FD_MDF_vs_MIX'])} | {fmt(ps['1']['cells']['FD_MDF_vs_MIX'])} | {bars['G4']} |",
        f"| G3 SHIP-R8 OVERALL | {(ver or {}).get('overall')} | | {g3} |", "",
        f"F-rate r (S6 compliant): s0 {ps['0']['F_rate_S6_compliant_r']:.4f} ({ps['0']['n_F_S6_compliant']}/{ps['0']['n_S6_compliant']}), "
        f"s1 {ps['1']['F_rate_S6_compliant_r']:.4f} ({ps['1']['n_F_S6_compliant']}/{ps['1']['n_S6_compliant']}); beside S2 compliant "
        f"s0 {ps['0']['beside_F_rate_S2_compliant']:.4f}, s1 {ps['1']['beside_F_rate_S2_compliant']:.4f}; HSGATE8 S6 compliant s0 "
        f"{ps['0']['beside_HSGATE8_F_rate_S6_compliant']:.4f}, s1 {ps['1']['beside_HSGATE8_F_rate_S6_compliant']:.4f}", "",
        "| beside cell | seed 0 | seed 1 |", "|---|---|---|"]
    for key in ("MD_MDF_vs_HS", "FD_MDF_vs_HS", "MD_MDF_vs_FX", "FD_MDF_vs_FX", "MD_MDF_vs_RETIME", "FD_MDF_vs_RETIME", "MD_MIX_vs_H8", "FD_MIX_vs_H8"):
        L.append(f"| {key} | {fmt(ps['0']['cells'][key])} | {fmt(ps['1']['cells'][key])} |")
    L += ["", f"A4d fit-seed floor (beside every H2cn-relative cell): {json.dumps(beside['A4d_fit_seed_floor'])[:1500] if beside['A4d_fit_seed_floor'] else 'absent'}",
          "", f"controls: {json.dumps(ctl)}", "", f"NEXT: {nxt}", "", "## disclosed deviations", ""] + [f"* {d}" for d in DEVIATIONS] + [""]
    (RESULT / "MDFIX_VERDICT.md").write_text("\n".join(L), encoding="utf-8")
    chk = json.loads((RESULT / "MDFIX_VERDICT.json").read_text(encoding="utf-8"))
    assert chk["word"] == word and set(chk["bars"]) == {"G1", "G2", "G3", "G4"}
    print(f"[mdfix_bars] {word}  bars={bars} controls={ctl}  ZZMDFIXBARSZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
