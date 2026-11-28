#!/usr/bin/env python3
"""PREREG_SOPICKCON_A2R_NEARTIE.md: control 3p (near-tie-robust) and the A2 tracking read on the near-tie-free sets S*.  CPU numpy, the SAME banked R0 states / loaders / estimator as
SO-CARVE's so_tempsweep_a2.py.   PYTHONUTF8=1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe so_a2r_neartie.py   ->  a2r_neartie.json (+ a printed verdict)
--selftest runs ONLY the planted-array control 4 (no banked data)."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOC = os.environ.get("SOCARVE_DIR") or next(p for p in (os.path.join(HERE, "..", "2026-10-18-so-carve"), "C:/Users/Admin/socarve") if os.path.isdir(p))
ARCH = os.environ.get("SOCARVE_ARCH", "C:/Users/Admin/AppData/Local/Temp/safehead/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
sys.path.insert(0, SOC)
sys.path.insert(0, os.path.join(ARCH, "raw", "2026-10-14-rls1b-attribution"))
import attrib_rls1b as AT   # noqa: E402
import so_carve_analysis as SC   # noqa: E402

TC, RHO, M_PROP = 1e-4, 5e-9, 64                         # PREREG s.2-3: control temperature, runner-up mass ceiling per arm, proposals per set
EPS_EQ = TC * np.log((M_PROP - 1) / RHO)                 # the equivalent top-2 log-score margin (reporting only)
GRID, UP = (0.1, 0.05, 0.02, 0.01), (250, 500, 1000)
COV_FLOOR, CTL_TOL, ALIGN = 0.50, 1e-6, 0.25
DL, X = "C:/Users/Admin/rls1b_dl", "C:/Users/Admin/rls1b_dl/data/x"


def logscore(lg):
    return np.log(AT.sel_v1_lane(AT.RV.sigmoid(lg)) + AT.EPS)


def runner_mass(l, Tc=TC):
    """r = sum over the NON-argmax proposals of exp((l_j - l_max) / Tc); an exact tie at the top gives r >= 1"""
    e = np.exp((l - l.max(-1, keepdims=True)) / Tc)
    e[np.arange(len(l)), l.argmax(-1)] = 0.0
    return e.sum(-1)


def margin(l):
    p = np.sort(l, -1)
    return p[:, -1] - p[:, -2]


def pol_l(l, T):
    return AT.softmax(l / T)


def d_pick(lA, lB, y):
    ar = np.arange(len(y))
    return 100.0 * (y[ar, lB.argmax(-1)] - y[ar, lA.argmax(-1)])


def d_T(lA, lB, y, T):
    return 100.0 * ((pol_l(lB, T) * y).sum(-1) - (pol_l(lA, T) * y).sum(-1))


def sel_mask(lA, lB):
    return (runner_mass(lA) <= RHO) & (runner_mass(lB) <= RHO)


def selftest():
    """control 4 (PREREG s.4): planted arrays, no banked data.  Three sets x 64 proposals; proposal 0 is unsafe (TTC label 0), proposal 1 safe (TTC label 1).
    set 0: an EXACT top-2 tie in both arms (argmax = proposal 0, the unsafe one; the tempered expectation splits 0.5 / 0.5);  set 1: a 1e-2 log margin (r = e^-100);  set 2: a clear top (margin 3)."""
    M = M_PROP
    base = np.full(M, -5.0)

    def one(top, second):
        l = base.copy()
        l[0], l[1] = top, second
        return l
    lA = np.stack([one(0.0, 0.0), one(0.0, -1e-2), one(0.0, -3.0)])
    lB = lA.copy()
    lB[1] = one(-1e-2, 0.0)                                                # B's pick on set 1 is the safe proposal 1 (margin 1e-2, retained); on set 0 B = A (the tie), on set 2 B = A
    y = np.zeros((3, M))
    y[:, 1] = 1.0
    S = sel_mask(lA, lB)
    # to make the tie bite in the DIFFERENCE, give B the safe proposal as its (arbitrary) argmax on the tie set: nudge by one ulp of the float64 log-score
    lB2 = lB.copy()
    lB2[0, 1] = np.nextafter(0.0, 1.0)                                    # B's top on set 0 is proposal 1 by ONE ulp: the pick flips, the tempered expectation does not move
    S2 = sel_mask(lA, lB2)
    err_all = np.abs(d_T(lA, lB2, y, TC) - d_pick(lA, lB2, y))
    ok = {"tie_excluded": bool(not S2[0]), "margin_1e-2_retained": bool(S2[1]), "clear_set_retained": bool(S2[2]), "unflipped_tie_is_also_excluded": bool(not S[0]),
          "all_sets_control_FAILS_on_the_tie": bool(err_all[0] > 1.0), "S_star_control_PASSES": bool(err_all[S2].max() <= CTL_TOL),
          "margins_reported": [float(m) for m in margin(lA)]}
    ok["PASS"] = bool(all(v for k, v in ok.items() if k != "margins_reported"))
    return ok


def main():
    st = selftest()
    print("control 4 (planted):", st)
    if "--selftest" in sys.argv:
        return 0 if st["PASS"] else 1
    kv, tgv, dv, ckv, _ = AT.load_side(f"{X}/pdm_val/pdm_val", f"{X}/terms_val/terms_val")
    y = AT.comps(tgv)["ttc"]
    N = len(kv)
    logs = np.asarray([k[0] for k in kv]).astype(str)
    ul, inv = np.unique(logs, return_inverse=True)
    nlog = len(ul)
    VI75 = np.random.default_rng(SC.SEED).integers(0, nlog, size=(SC.NBOOT, nlog))
    a2 = json.load(open(os.path.join(SOC, "tempsweep_a2.json"), encoding="utf-8"))

    def agg(d, mask):
        w = mask.astype(np.float64)
        return np.bincount(inv, weights=d * w, minlength=nlog), np.bincount(inv, weights=w, minlength=nlog)

    def Lpt(s, c):
        return float(SC.Lof(s, c, np.arange(nlog)))

    res = {"tool": "so_a2r_neartie/1", "prereg_sha256": open(os.path.join(HERE, "PREREG_SOPICKCON_A2R_NEARTIE.sha256")).read().split()[0], "TC": TC, "RHO": RHO, "eps_eq": float(EPS_EQ),
           "controls": {"planted_control4": st}, "states": {}, "n_sets": N, "n_logs": int(nlog)}
    ok = bool(st["PASS"])
    cov_ok = True
    for u in UP:
        lgA = AT.load_scores_aligned(f"{DL}/r0/final/scores_update{u}_A.npz", kv, ckv)
        lgB = AT.load_scores_aligned(f"{DL}/r0/final/scores_update{u}_B.npz", kv, ckv)
        assert lgA.shape[1] == M_PROP and lgB.shape[1] == M_PROP, (lgA.shape, lgB.shape)
        lA, lB = logscore(lgA), logscore(lgB)
        iB, iA = AT.picks(lgB), AT.picks(lgA)
        n_pick_ambiguous = int((lA.argmax(-1) != iA).sum() + (lB.argmax(-1) != iB).sum())
        S = sel_mask(lA, lB)
        allm = np.ones(N, bool)
        rA, rB, mA, mB = runner_mass(lA), runner_mass(lB), margin(lA), margin(lB)
        row = {"n_retained": int(S.sum()), "coverage": float(S.mean()), "n_logs_with_retained_sets": int(len(np.unique(inv[S]))), "n_excluded": int((~S).sum()),
               "n_excl_A": int((rA > RHO).sum()), "n_excl_B": int((rB > RHO).sum()), "n_pick_vs_argmax_logscore_ambiguous": n_pick_ambiguous,
               "n_margin_below_eps_eq_A": int((mA < EPS_EQ).sum()), "n_margin_below_eps_eq_B": int((mB < EPS_EQ).sum()),
               "n_exact_top_tie_A": int((mA == 0).sum()), "n_exact_top_tie_B": int((mB == 0).sum()),
               "n_excluded_not_explained_by_margin_lt_eps_eq": int(((~S) & (mA >= EPS_EQ) & (mB >= EPS_EQ)).sum()),
               "margin_quantiles_A": {str(q): float(np.quantile(mA, q)) for q in (0.001, 0.01, 0.05, 0.5)}, "margin_quantiles_B": {str(q): float(np.quantile(mB, q)) for q in (0.001, 0.01, 0.05, 0.5)}}
        dP = d_pick(lA, lB, y)
        sP_all, cP_all = agg(dP, allm)
        sP_S, cP_S = agg(dP, S)
        row["L_P1_all"], row["L_P1_S"] = Lpt(sP_all, cP_all), Lpt(sP_S, cP_S)
        d01 = d_T(lA, lB, y, 0.1)                                             # control 1: T = 0.1 over all sets reproduces tempsweep_a2.json
        s01, c01 = agg(d01, allm)
        c1 = abs(Lpt(s01, c01) - a2["states"][f"r0_u{u}"]["0.1"]["L_T"]) <= 1e-9
        res["controls"][f"c1_T0.1_all_sets_equals_tempsweep_a2_u{u}"] = bool(c1)
        ok &= bool(c1)
        z = all(not np.any(d_T(lA, lA, y, T)) for T in GRID + (TC,)) and not np.any(d_pick(lA, lA, y))          # control 2: B := A reads exactly 0.0
        res["controls"][f"c2_zero_control_u{u}"] = bool(z)
        ok &= bool(z)
        dc = d_T(lA, lB, y, TC)                                               # control 3p on S*, and the ORIGINAL control 3 on all sets for reference
        sC_S, cC_S = agg(dc, S)
        sC_all, cC_all = agg(dc, allm)
        row["ctl3_prime_diff_S"] = Lpt(sC_S, cC_S) - row["L_P1_S"]
        row["ctl3_orig_diff_all"] = Lpt(sC_all, cC_all) - row["L_P1_all"]
        c3 = abs(row["ctl3_prime_diff_S"]) <= CTL_TOL
        res["controls"][f"c3prime_Tc1e-4_on_S_star_u{u}"] = bool(c3)
        ok &= bool(c3)
        row["max_abs_per_set_err_on_S"] = float(np.abs(dc - dP)[S].max()) if S.any() else None
        row["max_abs_per_set_err_on_excluded"] = float(np.abs(dc - dP)[~S].max()) if (~S).any() else None
        row["disclosure_SEEN_post_hoc_colder_T_all_sets_diff"] = {}
        for T in (1e-6, 1e-8):
            s_, c_ = agg(d_T(lA, lB, y, T), allm)
            row["disclosure_SEEN_post_hoc_colder_T_all_sets_diff"][str(T)] = Lpt(s_, c_) - row["L_P1_all"]
        cov_ok &= bool(row["coverage"] >= COV_FLOOR)
        row["T"] = {}
        for T in GRID:
            d = d_T(lA, lB, y, T)
            sT, cT = agg(d, S)
            sTa, cTa = agg(d, allm)
            L_S, L_a = Lpt(sT, cT), Lpt(sTa, cTa)
            diffb = SC.Lof(sT, cT, VI75) - SC.Lof(sP_S, cP_S, VI75)
            row["T"][str(T)] = {"L_T_S": L_S, "L_P1_S": row["L_P1_S"], "diff_S": L_S - row["L_P1_S"], "ci_diff_S": SC.pct(diffb), "aligned_S": bool(abs(L_S - row["L_P1_S"]) <= ALIGN),
                                "L_T_all": L_a, "diff_all": L_a - row["L_P1_all"], "aligned_all": bool(abs(L_a - row["L_P1_all"]) <= ALIGN),
                                "diff_all_matches_tempsweep_a2": bool(abs((L_a - row["L_P1_all"]) - a2["states"][f"r0_u{u}"][str(T)]["diff"]) <= 1e-9)}
        res["states"][f"r0_u{u}"] = row
    al_S = {str(T): all(res["states"][f"r0_u{u}"]["T"][str(T)]["aligned_S"] for u in UP) for T in GRID}
    al_a = {str(T): all(res["states"][f"r0_u{u}"]["T"][str(T)]["aligned_all"] for u in UP) for T in GRID}
    res["ALIGNED_S"], res["ALIGNED_all"] = al_S, al_a
    res["controls_all_pass"], res["coverage_floor_met"] = bool(ok), bool(cov_ok)
    star = [T for T in GRID if al_S[str(T)]]
    if not cov_ok:
        out = "A2R-UNDERPOWERED"
    elif not ok:
        out = "A2R-CONTROL-FAILS"
    elif al_S["0.02"]:
        out = "A2R-CONFIRMED"
    elif star:
        out = f"A2R-SHIFTED(T'={max(star)})"
    else:
        out = "A2R-NONE"
    res["outcome"] = out
    json.dump(res, open(os.path.join(HERE, "a2r_neartie.json"), "w"), indent=1)
    print("controls", {k: v for k, v in res["controls"].items() if k != "planted_control4"})
    for u in UP:
        r = res["states"][f"r0_u{u}"]
        print(f"u{u}: retained {r['n_retained']}/{N} ({r['coverage']:.3f}), exact top ties A/B {r['n_exact_top_tie_A']}/{r['n_exact_top_tie_B']}, margin<eps_eq A/B {r['n_margin_below_eps_eq_A']}/{r['n_margin_below_eps_eq_B']}, "
              f"ctl3p diff on S* {r['ctl3_prime_diff_S']:.3e}, ctl3 diff on all {r['ctl3_orig_diff_all']:+.4f}, L_P1 all {r['L_P1_all']:+.3f} S* {r['L_P1_S']:+.3f}")
        print("   " + " | ".join(f"T={T}: S* d {r['T'][str(T)]['diff_S']:+.3f} (all {r['T'][str(T)]['diff_all']:+.3f})" for T in GRID))
    print("ALIGNED_S", al_S, "ALIGNED_all", al_a)
    print("OUTCOME", out)
    print("ZZPICKCON_A2R")
    return 0


if __name__ == "__main__":
    sys.exit(main())
