#!/usr/bin/env python3
"""PREREG_SOPICKCON_A2R2_TIEAWARE.md: L_T against the TIE-AWARE pick (the label averaged over the candidates within tau of the top), all 3,000 sets, R0 u250/u500/u1000.  CPU numpy.
PYTHONUTF8=1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe so_a2r2_tieaware.py  ->  a2r2_tieaware.json   (--selftest: the planted control 4 only)"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import so_a2r_neartie as NT   # noqa: E402  (logscore / pol_l / d_T / margin, the loaders and the constants; its main() is not run)

AT, SC = NT.AT, NT.SC
TAU0 = float(NT.EPS_EQ)                                  # fixed by A2R's derivation (T_c ln(63 / rho)), not chosen here
TAUS = (0.0, 1e-4, 1e-3, TAU0, 1e-2)
GRID, UP, ALIGN = NT.GRID, NT.UP, NT.ALIGN


def qmean(l, y, tau):
    """mean label over Q(tau) = {j : l_j >= l_max - tau}; tau = 0 is the ordinary pick when the top is not an exact tie"""
    w = (l >= l.max(-1, keepdims=True) - tau).astype(np.float64)
    return (w * y).sum(-1) / w.sum(-1)


def d_tie(lA, lB, y, tau):
    return 100.0 * (qmean(lB, y, tau) - qmean(lA, y, tau))


def selftest():
    M = NT.M_PROP
    base = np.full(M, -5.0)

    def one(top, second):
        l = base.copy()
        l[0], l[1] = top, second
        return l
    lB = np.stack([one(0.0, -1e-3), one(0.0, -3.0)])                     # set 0: a near-tie (gap 1e-3), argmax = proposal 0; set 1: clear (margin 3)
    lA = np.stack([one(0.0, -3.0), one(0.0, -3.0)])
    y = np.zeros((2, M)); y[:, 1] = 1.0                                    # proposal 0 unsafe (0), proposal 1 safe (1)
    out = {"tau0_is_the_argmax_label": bool(d_tie(lA, lB, y, 0.0)[0] == 0.0), "tau_ge_gap_reads_the_tie_mean_50": bool(abs(d_tie(lA, lB, y, 2e-3)[0] - 50.0) < 1e-9),
           "clear_set_same_at_every_tau": bool(all(d_tie(lA, lB, y, t)[1] == 0.0 for t in (0.0, 1e-3, 1e-2, 2.0))),
           "tau_below_gap_is_still_the_argmax": bool(d_tie(lA, lB, y, 5e-4)[0] == 0.0)}
    out["PASS"] = bool(all(out.values()))
    return out


def main():
    st = selftest()
    print("control 4 (planted):", st)
    if "--selftest" in sys.argv:
        return 0 if st["PASS"] else 1
    kv, tgv, dv, ckv, _ = AT.load_side(f"{NT.X}/pdm_val/pdm_val", f"{NT.X}/terms_val/terms_val")
    y = AT.comps(tgv)["ttc"]
    N = len(kv)
    logs = np.asarray([k[0] for k in kv]).astype(str)
    ul, inv = np.unique(logs, return_inverse=True)
    nlog = len(ul)
    VI75 = np.random.default_rng(SC.SEED).integers(0, nlog, size=(SC.NBOOT, nlog))
    a2 = json.load(open(os.path.join(NT.SOC, "tempsweep_a2.json"), encoding="utf-8"))

    def agg(d):
        return np.bincount(inv, weights=d, minlength=nlog), np.bincount(inv, minlength=nlog).astype(np.float64)

    def Lpt(s, c):
        return float(SC.Lof(s, c, np.arange(nlog)))
    res = {"tool": "so_a2r2_tieaware/1", "prereg_sha256": open(os.path.join(HERE, "PREREG_SOPICKCON_A2R2_TIEAWARE.sha256")).read().split()[0], "tau0": TAU0, "controls": {"planted_control4": st}, "states": {}}
    ok = bool(st["PASS"])
    for u in UP:
        lgA = AT.load_scores_aligned(f"{NT.DL}/r0/final/scores_update{u}_A.npz", kv, ckv)
        lgB = AT.load_scores_aligned(f"{NT.DL}/r0/final/scores_update{u}_B.npz", kv, ckv)
        lA, lB = NT.logscore(lgA), NT.logscore(lgB)
        row = {"n_exact_top_tie": int((NT.margin(lA) == 0).sum() + (NT.margin(lB) == 0).sum()), "tau": {}, "T": {}}
        sT_ = {}
        for T in GRID:
            d = NT.d_T(lA, lB, y, T)
            sT_[T] = agg(d)
            row["T"][str(T)] = {"L_T_all": Lpt(*sT_[T])}
        c1 = abs(row["T"]["0.1"]["L_T_all"] - a2["states"][f"r0_u{u}"]["0.1"]["L_T"]) <= 1e-9                       # control 1
        res["controls"][f"c1_T0.1_equals_tempsweep_a2_u{u}"] = bool(c1)
        ok &= bool(c1)
        z = all(not np.any(d_tie(lA, lA, y, t)) for t in TAUS) and all(not np.any(NT.d_T(lA, lA, y, T)) for T in GRID)          # control 2
        res["controls"][f"c2_zero_control_u{u}"] = bool(z)
        ok &= bool(z)
        L0 = Lpt(*agg(d_tie(lA, lB, y, 0.0)))                                                                          # control 3
        c3 = abs(L0 - a2["states"][f"r0_u{u}"]["0.1"]["L_P1"]) <= 1e-12 and row["n_exact_top_tie"] == 0
        res["controls"][f"c3_tau0_equals_L_P1_u{u}"] = bool(c3)
        ok &= bool(c3)
        for tau in TAUS:
            sP, cP = agg(d_tie(lA, lB, y, tau))
            Lt = Lpt(sP, cP)
            nq = {"A": float((lA >= lA.max(-1, keepdims=True) - tau).sum(-1).mean()), "B": float((lB >= lB.max(-1, keepdims=True) - tau).sum(-1).mean())}
            rt = {"L_P1_tau": Lt, "mean_candidates_in_Q": nq, "diff": {}}
            for T in GRID:
                s, c = sT_[T]
                diffb = SC.Lof(s, c, VI75) - SC.Lof(sP, cP, VI75)
                rt["diff"][str(T)] = {"diff": row["T"][str(T)]["L_T_all"] - Lt, "ci": SC.pct(diffb), "aligned": bool(abs(row["T"][str(T)]["L_T_all"] - Lt) <= ALIGN)}
            row["tau"][str(tau)] = rt
        res["states"][f"r0_u{u}"] = row
    key = str(TAU0)
    al = {str(T): all(res["states"][f"r0_u{u}"]["tau"][key]["diff"][str(T)]["aligned"] for u in NT.UP) for T in GRID}
    res["ALIGNED_tie_tau0"] = al
    res["sensitivity_alignment_by_tau"] = {str(tau): {str(T): all(res["states"][f"r0_u{u}"]["tau"][str(tau)]["diff"][str(T)]["aligned"] for u in UP) for T in GRID} for tau in TAUS}
    star = [T for T in GRID if al[str(T)]]
    res["controls_all_pass"] = bool(ok)
    if not ok:
        res["outcome"] = "INVALID"
    elif star and 0.02 in star:
        res["outcome"] = "A2R2-ALIGNED(T*=%s)" % max(star)
    elif star:
        res["outcome"] = "A2R2-SHIFTED(T'=%s)" % max(star)
    else:
        res["outcome"] = "A2R2-NONE"
    thin = {}
    for T in GRID:
        thin[str(T)] = max(abs(res["states"][f"r0_u{u}"]["tau"][key]["diff"][str(T)]["diff"]) for u in UP)
    res["max_abs_diff_over_states_tau0"] = thin
    json.dump(res, open(os.path.join(HERE, "a2r2_tieaware.json"), "w"), indent=1)
    print("controls_all_pass", ok, {k: v for k, v in res["controls"].items() if k != "planted_control4"})
    for u in UP:
        r = res["states"][f"r0_u{u}"]
        print(f"u{u}: L_P1^tau " + " | ".join(f"tau={t:g}: {r['tau'][str(t)]['L_P1_tau']:+.3f} (|Q| A/B {r['tau'][str(t)]['mean_candidates_in_Q']['A']:.1f}/{r['tau'][str(t)]['mean_candidates_in_Q']['B']:.1f})" for t in TAUS))
        print("   diff vs tau0: " + " | ".join(f"T={T}: {r['tau'][key]['diff'][str(T)]['diff']:+.3f} {r['tau'][key]['diff'][str(T)]['ci']}" for T in GRID))
    print("ALIGNED_tie_tau0", al, "max|diff|", thin)
    print("sensitivity (aligned at all three states) by tau:", json.dumps(res["sensitivity_alignment_by_tau"]))
    print("OUTCOME", res["outcome"])
    print("ZZPICKCON_A2R2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
