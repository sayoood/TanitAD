#!/usr/bin/env python3
"""PREREG_SOCARVE_A2_TEMPSWEEP.md: L_T (E_pi_T[TTC] loss) vs the pick's L_P1 on the full RL-val, R0 u250/u500/u1000.  CPU numpy.  -> tempsweep_a2.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ARCH = os.environ.get("SOCARVE_ARCH", "C:/Users/Admin/AppData/Local/Temp/safehead/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
sys.path.insert(0, os.path.join(ARCH, "raw", "2026-10-14-rls1b-attribution"))
import attrib_rls1b as AT   # noqa: E402
import so_carve_analysis as SC   # noqa: E402
DL, X = "C:/Users/Admin/rls1b_dl", "C:/Users/Admin/rls1b_dl/data/x"
kv, tgv, dv, ckv, _ = AT.load_side(f"{X}/pdm_val/pdm_val", f"{X}/terms_val/terms_val")
y = AT.comps(tgv)["ttc"]; N = len(kv); ar = np.arange(N)
logs = np.asarray([k[0] for k in kv]).astype(str); nlog = len(np.unique(logs))
VI75 = np.random.default_rng(SC.SEED).integers(0, nlog, size=(SC.NBOOT, nlog))
GRID, UP = (0.1, 0.05, 0.02, 0.01), (250, 500, 1000)
a1 = json.load(open(os.path.join(HERE, "fullval_a1.json"), encoding="utf-8"))


def pol(lg, T):
    return AT.softmax(np.log(AT.sel_v1_lane(AT.RV.sigmoid(lg)) + AT.EPS) / T)


def L_of(d):
    _, s, c = SC.aggregate(d, logs)
    return float(SC.Lof(s, c, np.arange(len(s)))), (s, c)


res = {"tool": "so_tempsweep_a2/1", "controls": {}, "states": {}}
ok = True
for u in UP:
    lgA = AT.load_scores_aligned(f"{DL}/r0/final/scores_update{u}_A.npz", kv, ckv)
    lgB = AT.load_scores_aligned(f"{DL}/r0/final/scores_update{u}_B.npz", kv, ckv)
    iB, iA = AT.picks(lgB), AT.picks(lgA)
    dP1 = 100.0 * (y[ar, iB] - y[ar, iA])
    LP1, (sP, cP) = L_of(dP1)
    rows = {}
    for T in GRID + (1e-4,):
        d = 100.0 * ((pol(lgB, T) * y).sum(-1) - (pol(lgA, T) * y).sum(-1))
        L, (s, c) = L_of(d)
        diff_b = SC.Lof(s, c, VI75) - SC.Lof(sP, cP, VI75)
        rows[str(T)] = {"L_T": L, "L_P1": LP1, "diff": L - LP1, "ci_diff": SC.pct(diff_b), "ci_L_T": SC.pct(SC.Lof(s, c, VI75)), "abs_diff_le_0.25": bool(abs(L - LP1) <= 0.25)}
        if T == 0.1:
            ok &= abs(L - a1["states"][f"r0_u{u}"]["P2_TTC"]["L_va_full"]) <= 1e-9
            res["controls"][f"T0.1_equals_A1_P2_u{u}"] = bool(abs(L - a1["states"][f"r0_u{u}"]["P2_TTC"]["L_va_full"]) <= 1e-9)
        if T == 1e-4:
            res["controls"][f"T1e-4_equals_P1_u{u}"] = bool(abs(L - LP1) <= 1e-6)
            res["controls"][f"T1e-4_n_picks_differing_from_argmax_u{u}"] = int((pol(lgB, T).argmax(-1) != iB).sum() + (pol(lgA, T).argmax(-1) != iA).sum())
            ok &= abs(L - LP1) <= 1e-6
    dz = [100.0 * ((pol(lgA, T) * y).sum(-1) - (pol(lgA, T) * y).sum(-1)) for T in GRID]
    res["controls"][f"zero_control_u{u}"] = bool(all(not d.any() for d in dz))
    ok &= res["controls"][f"zero_control_u{u}"]
    res["states"][f"r0_u{u}"] = rows
res["verdict_controls"] = "VALID" if ok else "INVALID"
al = {str(T): all(res["states"][f"r0_u{u}"][str(T)]["abs_diff_le_0.25"] for u in UP) for T in GRID}
res["PICK_ALIGNED"] = al
star = [T for T in GRID if al[str(T)]]
res["outcome"] = (f"A2-ALIGNED(T*={max(star)})" if star else "A2-NONE") if ok else "INVALID"
json.dump(res, open(os.path.join(HERE, "tempsweep_a2.json"), "w"), indent=1)
print(res["verdict_controls"], res["outcome"], res["PICK_ALIGNED"], res["controls"])
for u in UP:
    print(f"u{u}", " | ".join(f"T={T}: L_T {res['states'][f'r0_u{u}'][str(T)]['L_T']:+.3f} (d {res['states'][f'r0_u{u}'][str(T)]['diff']:+.3f})" for T in GRID), f"| L_P1 {res['states'][f'r0_u{u}']['0.1']['L_P1']:+.3f}")
print("ZZSOCARVE_A2")
