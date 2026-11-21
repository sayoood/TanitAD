#!/usr/bin/env python3
"""PREREG_SOCARVE_A1_FULLVAL.md: the constraint quantity g on the FULL RL-val (3,000 sets / 75 logs) from the banked LIVE score dumps of R0 (CPU, numpy, no forward pass).
    PYTHONUTF8=1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe so_fullval_a1.py   -> fullval_a1.json"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ARCH = os.environ.get("SOCARVE_ARCH", "C:/Users/Admin/AppData/Local/Temp/safehead/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
sys.path.insert(0, os.path.join(ARCH, "raw", "2026-10-14-rls1b-attribution"))
import attrib_rls1b as AT            # noqa: E402  (the RL-val loaders / picks / boot_delta that ab_vs_base.py and the R0d reads use)
import so_carve_analysis as SC       # noqa: E402

DL, X = "C:/Users/Admin/rls1b_dl", "C:/Users/Admin/rls1b_dl/data/x"
AB = os.path.join(ARCH, "raw", "2026-10-15-rls1-adam-dose", "ab_vs_base.json")
UP = (250, 500, 1000)
COMPS = ("TTC", "NC", "EP")

kv, tgv, dv, ckv, side = AT.load_side(os.path.join(X, "pdm_val", "pdm_val"), os.path.join(X, "terms_val", "terms_val"))
sv = AT.comps(tgv)
N = len(kv)
ar = np.arange(N)
logs_full = np.asarray([k[0] for k in kv]).astype(str)
print(f"RL-val joined: {N} sets, {len(np.unique(logs_full))} logs", flush=True)
res = {"tool": "so_fullval_a1/1", "n_sets": N, "n_logs": int(len(np.unique(logs_full))), "controls": {}, "states": {}}


def policy(logits):
    p = AT.RV.sigmoid(logits)
    return AT.softmax(np.log(AT.sel_v1_lane(p) + AT.EPS) / AT.T_POL)


def dsets(lgB, lgA):
    """-> {(quant, comp): per-set d (points)} on the full RL-val"""
    piB, piA = policy(lgB), policy(lgA)
    iB, iA = AT.picks(lgB), AT.picks(lgA)
    out = {}
    for c in COMPS + ("PDMS",):
        y = sv[c.lower()]
        out[("P2", c)] = 100.0 * ((piB * y).sum(-1) - (piA * y).sum(-1))
        out[("P1", c)] = 100.0 * (y[ar, iB] - y[ar, iA])
    return out


base = AT.load_scores_aligned(os.path.join(DL, "r0", "final", "scores_step0_base.npz"), kv, ckv)
# ---- train-pool side and the main read's in-sample carve, from the stage-1 arrays (the main read's own inputs)
Z = {u: dict(np.load(os.path.join(HERE, "out", f"persets_r0_u{u}.npz"), allow_pickle=False)) for u in UP}
logs_tr, logs_va_pool = Z[1000]["train__logs"], Z[1000]["val__logs"]
TI, _VI13, _POS = SC.make_resamples(len(np.unique(logs_tr)), len(np.unique(logs_va_pool)))
DR13 = SC.carve_draws(len(np.unique(logs_tr)), SC.M_PRIMARY)
VI75 = np.random.default_rng(SC.SEED).integers(0, len(np.unique(logs_full)), size=(SC.NBOOT, len(np.unique(logs_full))))
pool_idx = {}
kmap = {"|".join(str(x) for x in k): i for i, k in enumerate(kv)}
pool_keys = Z[1000]["val__keys"]
pool_idx = [kmap.get(str(k), -1) for k in pool_keys]
res["controls"]["pool_val_sets_found_in_full_val"] = f"{sum(i >= 0 for i in pool_idx)} of {len(pool_keys)}"
ab = json.load(open(AB, encoding="utf-8"))["rows"]
ctl_ok, ctl_rows = True, []
for u in UP:
    lgA = AT.load_scores_aligned(os.path.join(DL, "r0", "final", f"scores_update{u}_A.npz"), kv, ckv)
    lgB = AT.load_scores_aligned(os.path.join(DL, "r0", "final", f"scores_update{u}_B.npz"), kv, ckv)
    D = dsets(lgB, lgA)
    # control 1: P1 B - A reproduces ab_vs_base.json
    iB, iA = AT.picks(lgB), AT.picks(lgA)
    for c in COMPS + ("PDMS",):
        mine = AT.boot_delta(sv[c.lower()][ar, iB], sv[c.lower()][ar, iA], kv)
        ref = ab[f"r0_u{u}"]["B_minus_A"][c]
        dmax = max(abs(mine["mean"] - ref["mean"]), abs(mine["ci95"][0] - ref["ci95"][0]), abs(mine["ci95"][1] - ref["ci95"][1]))
        ctl_rows.append({"u": u, "comp": c, "mine": mine, "banked": ref, "max_absdiff": dmax})
        ctl_ok = ctl_ok and dmax <= 1e-6
    # control 2: zero control
    Dz = dsets(lgA, lgA)
    res["controls"][f"zero_control_u{u}"] = bool(all(not v.any() for v in Dz.values()))
    ul_f, _, _ = SC.aggregate(D[("P2", "TTC")], logs_full)
    st = {}
    for (q, c), d in D.items():
        if c == "PDMS":
            continue
        ul, s_v, c_v = SC.aggregate(d, logs_full)
        d_tr = (Z[u]["train__own__" + ("pi_comp" if q == "P2" else "pick_comp")][:, SC.COLS[c]] - Z[u]["train__A_own__" + ("pi_comp" if q == "P2" else "pick_comp")][:, SC.COLS[c]]) * 100.0
        _, s_t, c_t = SC.aggregate(d_tr, logs_tr)
        d_vp = (Z[u]["val__own__" + ("pi_comp" if q == "P2" else "pick_comp")][:, SC.COLS[c]] - Z[u]["val__A_own__" + ("pi_comp" if q == "P2" else "pick_comp")][:, SC.COLS[c]]) * 100.0
        _, s_vp, c_vp = SC.aggregate(d_vp, logs_va_pool)
        L_tr, L_va = float(SC.Lof(s_t, c_t, np.arange(len(s_t)))), float(SC.Lof(s_v, c_v, np.arange(len(s_v))))
        L_pool = float(SC.Lof(s_vp, c_vp, np.arange(len(s_vp))))
        Lc = float(SC.Lof(s_t, c_t, DR13).mean())
        Ltr_b, Lva_b = SC.Lof(s_t, c_t, TI), SC.Lof(s_v, c_v, VI75)
        G = Lva_b - Ltr_b
        ci_va, ci_G = SC.pct(Lva_b), SC.pct(G)
        lab, rho = SC.label_fn(L_tr, L_va, Lc, ci_va)
        gap = "RESOLVED-POSITIVE" if ci_G[0] > 0 else ("RESOLVED-NEGATIVE" if ci_G[1] < 0 else "UNRESOLVED")
        st[f"{q}_{c}"] = {"L_tr": L_tr, "ci_L_tr": SC.pct(Ltr_b), "L_va_full": L_va, "ci_L_va_full": ci_va, "L_va_pool62": L_pool, "pool_inside_full_ci": bool(ci_va[0] <= L_pool <= ci_va[1]),
                          "L_c_insample_carve": Lc, "G_tv_full": L_va - L_tr, "ci_G_tv_full": ci_G, "A1_GAP": gap, "bar_sized_gap_excluded": bool(ci_G[0] > -0.5 and ci_G[1] < 0.5),
                          "A1_LABEL": lab, "rho": rho, "F1_in_sample_carve": True}
        if (q, c) in (("P2", "TTC"), ("P1", "TTC")):
            # control 3: fidelity of the cached-context harness on the 62 matched sets
            m = [(j, i) for j, i in enumerate(pool_idx) if i >= 0]
            dj = np.asarray([d[i] for j, i in m]); dh = np.asarray([d_vp[j] for j, i in m])
            Lh = float(-dh.mean()); Ld = float(-dj.mean())
            st[f"{q}_{c}"]["fidelity_62"] = {"n": len(m), "max_abs_dset": float(np.abs(dj - dh).max()), "mean_abs_dset": float(np.abs(dj - dh).mean()), "L_pool_dump": Ld, "L_pool_harness": Lh,
                                             "abs_dL": abs(Ld - Lh), "tag": "FAITHFUL" if abs(Ld - Lh) <= 0.25 else "FIDELITY-GAP"}
    res["states"][f"r0_u{u}"] = st
res["controls"]["ab_vs_base_reproduction_all_rows_le_1e-6"] = bool(ctl_ok)
res["controls"]["ab_vs_base_rows"] = ctl_rows
res["verdict"] = "VALID" if (ctl_ok and all(v is True for k, v in res["controls"].items() if k.startswith("zero_control"))) else "INVALID"
json.dump(res, open(os.path.join(HERE, "fullval_a1.json"), "w"), indent=1)
print("controls:", {k: v for k, v in res["controls"].items() if k != "ab_vs_base_rows"}, "->", res["verdict"])
if res["verdict"] == "VALID":
    for u in UP:
        for k in ("P2_TTC", "P1_TTC", "P2_NC"):
            r = res["states"][f"r0_u{u}"][k]
            print(f"u{u} {k}: L_tr {r['L_tr']:+.3f} [{r['ci_L_tr'][0]:+.2f},{r['ci_L_tr'][1]:+.2f}] | L_va_full {r['L_va_full']:+.3f} [{r['ci_L_va_full'][0]:+.2f},{r['ci_L_va_full'][1]:+.2f}] "
                  f"| pool62 {r['L_va_pool62']:+.3f} in={r['pool_inside_full_ci']} | L_c {r['L_c_insample_carve']:+.3f} | G_tv {r['G_tv_full']:+.3f} [{r['ci_G_tv_full'][0]:+.2f},{r['ci_G_tv_full'][1]:+.2f}] "
                  f"{r['A1_GAP']} barexcl={r['bar_sized_gap_excluded']} | {r['A1_LABEL']} rho={r['rho']:.3f}" + (f" | fid {r['fidelity_62']['tag']} dL={r['fidelity_62']['abs_dL']:.3f}" if "fidelity_62" in r else ""))
print("ZZSOCARVE_A1", res["verdict"])
