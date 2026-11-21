#!/usr/bin/env python3
"""PREREG_R0E_D4_NC.md: does the constraint temperature that tracks the pick for TTC also track it for NC?  The NC label, A2R2's harness reused unchanged.  CPU numpy, no pod.
The banked R0 states scores_update{250,500,1000}_{A,B}.npz (3,000 sets / 75 logs) and the RL-val labels are on the dev box (see NT.DL / NT.X in so_a2r_neartie.py).

    PYTHONUTF8=1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe r0e_d4_nc_alignment.py [--pk <the refe-plan root>] [--out <json>]     ->  d4_nc_alignment.json
    --selftest   runs ONLY the planted-array control (no banked data)

The pre-registration is hashed in PREREG_R0E_D4_NC.sha256 and this script REFUSES to run on a PREREG_R0E_D4_NC.md whose sha256 differs (the committed outcomes are the file's, not the script's)."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--pk", default=None, help="the refe-plan root (default: two levels above this package)")
ap.add_argument("--out", default=None)
ap.add_argument("--selftest", action="store_true")
ARGS = ap.parse_args()
PK = os.path.abspath(ARGS.pk) if ARGS.pk else os.path.dirname(os.path.dirname(HERE))
os.environ.setdefault("SOCARVE_ARCH", PK)                                             # so_a2r_neartie imports attrib_rls1b from <ARCH>/raw/2026-10-14-rls1b-attribution
if os.path.isdir(os.path.join(PK, "raw", "2026-10-18-so-carve")):
    os.environ.setdefault("SOCARVE_DIR", os.path.join(PK, "raw", "2026-10-18-so-carve"))
sys.path.insert(0, os.path.join(PK, "raw", "2026-10-18-so-pickcon"))
import so_a2r2_tieaware as T2   # noqa: E402  (the harness: qmean / d_tie / selftest / TAU0 / TAUS; it imports so_a2r_neartie as NT)

NT = T2.NT
AT, SC = NT.AT, NT.SC
GRID, UP, ALIGN = NT.GRID, NT.UP, NT.ALIGN                    # (0.1, 0.05, 0.02, 0.01), (250, 500, 1000), 0.25
HEADLINE_T = (0.1, 0.02, 0.01)
TAU0, TAUS = T2.TAU0, T2.TAUS
PREREG = os.path.join(HERE, "PREREG_R0E_D4_NC.md")
PREREG_SHA = os.path.join(HERE, "PREREG_R0E_D4_NC.sha256")
TOL1 = 1e-9


def load_states():
    kv, tgv, dv, ckv, _ = AT.load_side(f"{NT.X}/pdm_val/pdm_val", f"{NT.X}/terms_val/terms_val")
    comps = AT.comps(tgv)
    st = {}
    for u in UP:
        lgA = AT.load_scores_aligned(f"{NT.DL}/r0/final/scores_update{u}_A.npz", kv, ckv)
        lgB = AT.load_scores_aligned(f"{NT.DL}/r0/final/scores_update{u}_B.npz", kv, ckv)
        st[u] = {"lgA": lgA, "lgB": lgB, "lA": NT.logscore(lgA), "lB": NT.logscore(lgB)}
    return kv, comps, st


def read_block(lA, lB, y, inv, nlog, VI75):
    """-> the per-state block for one label: L_T (all sets) per grid T, L_P1^tau per tau, the diffs vs the tau0 pick with their log-cluster CIs, the levels.  The A2R2 computation, label as a parameter."""
    def agg(d):
        return np.bincount(inv, weights=d, minlength=nlog), np.bincount(inv, minlength=nlog).astype(np.float64)

    def Lpt(s, c):
        return float(SC.Lof(s, c, np.arange(nlog)))
    blk = {"T": {}, "tau": {}, "diff_tau0": {}, "levels": {}}
    sT = {}
    for T in GRID:
        sT[T] = agg(NT.d_T(lA, lB, y, T))
        blk["T"][str(T)] = Lpt(*sT[T])
    sP0 = None
    for tau in TAUS:
        sP, cP = agg(T2.d_tie(lA, lB, y, tau))
        blk["tau"][str(tau)] = Lpt(sP, cP)
        if tau == TAU0:
            sP0 = (sP, cP)
    for T in GRID:
        s, c = sT[T]
        diffb = SC.Lof(s, c, VI75) - SC.Lof(sP0[0], sP0[1], VI75)
        blk["diff_tau0"][str(T)] = {"diff": blk["T"][str(T)] - blk["tau"][str(TAU0)], "ci": SC.pct(diffb)}
    ar = np.arange(len(y))
    for nm, l in (("A", lA), ("B", lB)):
        blk["levels"][nm] = {"E_pi_x100": {str(T): float(100.0 * (NT.pol_l(l, T) * y).sum(-1).mean()) for T in GRID},
                             "pick_argmax_x100": float(100.0 * y[ar, l.argmax(-1)].mean()), "pick_tie_tau0_x100": float(100.0 * T2.qmean(l, y, TAU0).mean())}
    return blk


def main():
    st = T2.selftest()
    print("planted control (A2R2 selftest):", st, flush=True)
    if ARGS.selftest:
        return 0 if st["PASS"] else 1
    h = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    want = open(PREREG_SHA, encoding="utf-8").read().split()[0]
    if h != want:
        print(f"ZZR0E_D4_REFUSED the pre-registration {PREREG} has sha256 {h}, not the registered {want}")
        return 4
    kv, comps, S = load_states()
    y_nc, y_ttc = comps["nc"], comps["ttc"]
    N = len(kv)
    logs = np.asarray([k[0] for k in kv]).astype(str)
    ul, inv = np.unique(logs, return_inverse=True)
    nlog = len(ul)
    VI75 = np.random.default_rng(SC.SEED).integers(0, nlog, size=(SC.NBOOT, nlog))

    def Lall(d):
        return float(SC.Lof(np.bincount(inv, weights=d, minlength=nlog), np.bincount(inv, minlength=nlog).astype(np.float64), np.arange(nlog)))
    banked = json.load(open(os.path.join(PK, "raw", "2026-10-18-so-pickcon", "a2r2_tieaware.json"), encoding="utf-8"))
    a2 = json.load(open(os.path.join(NT.SOC, "tempsweep_a2.json"), encoding="utf-8"))
    res = {"tool": "r0e_d4_nc_alignment/1", "prereg_sha256": h, "n_sets": N, "n_logs": nlog, "tau0": TAU0, "align_bar": ALIGN, "controls": {"planted_control": st}, "states": {}}
    ok = bool(st["PASS"])
    c1 = {}
    for u in UP:
        lA, lB = S[u]["lA"], S[u]["lB"]
        row = {"NC": read_block(lA, lB, y_nc, inv, nlog, VI75)}
        ttc = read_block(lA, lB, y_ttc, inv, nlog, VI75)                                 # control 1 only: the TTC label through THIS code path
        k0 = f"r0_u{u}"
        d_lt = max(abs(ttc["T"][str(T)] - banked["states"][k0]["T"][str(T)]["L_T_all"]) for T in GRID)
        d_df = max(abs(ttc["diff_tau0"][str(T)]["diff"] - banked["states"][k0]["tau"][str(TAU0)]["diff"][str(T)]["diff"]) for T in GRID)
        d_ci = max(max(abs(a - b) for a, b in zip(ttc["diff_tau0"][str(T)]["ci"], banked["states"][k0]["tau"][str(TAU0)]["diff"][str(T)]["ci"])) for T in GRID)
        d_a2 = abs(ttc["T"]["0.1"] - a2["states"][k0]["0.1"]["L_T"])
        c1[str(u)] = {"max_abs_L_T_vs_a2r2": d_lt, "max_abs_diff_tau0_vs_a2r2": d_df, "max_abs_ci_vs_a2r2": d_ci, "abs_L_T_0.1_vs_tempsweep_a2": d_a2}
        ok &= bool(d_lt <= TOL1 and d_df <= TOL1 and d_ci <= TOL1 and d_a2 <= TOL1)
        z_T = max(abs(Lall(NT.d_T(lA, lA, y_nc, T))) for T in GRID)
        z_P = max(abs(Lall(T2.d_tie(lA, lA, y_nc, t))) for t in TAUS)
        res["controls"][f"c2_zero_control_NC_u{u}"] = {"max_abs_L_T": z_T, "max_abs_L_P1_tau": z_P, "pass": bool(z_T == 0.0 and z_P == 0.0)}
        ok &= bool(z_T == 0.0 and z_P == 0.0)
        pA, pB = AT.picks(S[u]["lgA"]), AT.picks(S[u]["lgB"])
        n_tie = int((NT.margin(lA) == 0).sum() + (NT.margin(lB) == 0).sum())
        c3 = bool(np.array_equal(pA, lA.argmax(-1)) and np.array_equal(pB, lB.argmax(-1)) and n_tie == 0)
        res["controls"][f"c3_pick_route_u{u}"] = {"picks_equal_AT_picks": c3, "n_exact_top_tie": n_tie}
        ok &= c3
        row["n_exact_top_tie"] = n_tie
        row["NC_label"] = {"unique_values": sorted(float(v) for v in np.unique(np.round(y_nc, 6))), "share_of_sets_where_it_varies_across_proposals": float((y_nc.max(-1) > y_nc.min(-1)).mean()),
                           "mean": float(y_nc.mean())}
        row["TTC_label"] = {"share_of_sets_where_it_varies_across_proposals": float((y_ttc.max(-1) > y_ttc.min(-1)).mean())}
        res["states"][f"r0_u{u}"] = row
    res["controls"]["c1_machinery_reuse_TTC_label_through_this_code_path"] = {"per_state": c1, "tolerance": TOL1,
                                                                               "pass": bool(all(v["max_abs_L_T_vs_a2r2"] <= TOL1 and v["max_abs_diff_tau0_vs_a2r2"] <= TOL1 and v["max_abs_ci_vs_a2r2"] <= TOL1
                                                                                               and v["abs_L_T_0.1_vs_tempsweep_a2"] <= TOL1 for v in c1.values()))}
    # ------------------------------------------------------------------ the committed readings (PREREG section 4)
    M = {str(T): max(abs(res["states"][f"r0_u{u}"]["NC"]["diff_tau0"][str(T)]["diff"]) for u in UP) for T in GRID}
    V = max(abs(res["states"][f"r0_u{u}"]["NC"]["tau"][str(TAU0)]) for u in UP)
    res["M_max_abs_diff_over_states"] = M
    res["V_max_abs_pick_NC_loss_tau0"] = V
    res["null_predictor_abs_diff_per_state"] = {f"u{u}": abs(res["states"][f"r0_u{u}"]["NC"]["tau"][str(TAU0)]) for u in UP}
    res["aligned_NC"] = {str(T): bool(M[str(T)] <= ALIGN) for T in GRID}
    lv = {u: res["states"][f"r0_u{u}"]["NC"]["levels"] for u in UP}                       # descriptive (PREREG section 2: the levels of each arm): the per-arm LEVEL form of the same gap, E_pi[NC] - the tie-aware pick's NC, in points
    res["level_form_max_abs_E_pi_minus_pick_tau0_per_arm"] = {str(T): max(abs(lv[u][nm]["E_pi_x100"][str(T)] - lv[u][nm]["pick_tie_tau0_x100"]) for u in UP for nm in ("A", "B")) for T in GRID}
    res["controls_all_pass"] = bool(ok)
    if not ok:
        res["outcome"] = "INVALID"
    elif V <= ALIGN:
        res["outcome"] = "D4-NC-VACUOUS"
    else:
        res["outcome"] = "D4-NC-ALIGNED(0.01)" if res["aligned_NC"]["0.01"] else "D4-NC-NOT-ALIGNED(0.01)"
        a01, a02 = res["aligned_NC"]["0.01"], res["aligned_NC"]["0.02"]
        res["secondary"] = "NC-A" if (a01 and a02) else ("NC-B" if a01 else ("NC-C" if a02 else "NC-D"))
    out = ARGS.out or os.path.join(HERE, "d4_nc_alignment.json")
    json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print("controls_all_pass", ok, {k: (v.get("pass") if isinstance(v, dict) and "pass" in v else v) for k, v in res["controls"].items() if k != "planted_control"}, flush=True)
    for u in UP:
        nc = res["states"][f"r0_u{u}"]["NC"]
        print(f"u{u}: L_P1^tau0(NC) {nc['tau'][str(TAU0)]:+.4f} | L_T(NC) " + " ".join(f"T={T}: {nc['T'][str(T)]:+.4f}" for T in GRID))
        print("   diff_NC = L_T - L_P1^tau0: " + " | ".join(f"T={T}: {nc['diff_tau0'][str(T)]['diff']:+.4f} {[round(x, 3) for x in nc['diff_tau0'][str(T)]['ci']]}" for T in GRID))
    for T in HEADLINE_T:
        print(f"max |E_pi[NC] - pick NC| over states (contrast form, tie-aware tau0) at T={T}: {M[str(T)]:.4f}   aligned (<= {ALIGN}): {res['aligned_NC'][str(T)]}")
    print("descriptive, level form (per arm, A and B), max |E_pi[NC] - tie-aware pick NC| over states:", {k: round(v, 4) for k, v in res["level_form_max_abs_E_pi_minus_pick_tau0_per_arm"].items()})
    print("V = max |L_P1^tau0(NC)| =", round(V, 4), "| null-predictor |0 - L_P1|:", {k: round(v, 4) for k, v in res["null_predictor_abs_diff_per_state"].items()})
    print("OUTCOME", res["outcome"], res.get("secondary", ""))
    print("ZZR0E_D4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
