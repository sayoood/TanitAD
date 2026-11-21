"""p5_analyze: the P5 read (THEORY_CRAWL_AMEND_3.md sections 3-4). lambda = 1 - Var(e_perp) / Var(a0_perp), e = a0 - a, perp = OLS residual on the model's other ego inputs.
PRIMARY (the verdict, decided by the card's rule): the 20,000-window seeded sample (raw/P5_WINDOWS.json), a0 = the MODEL's a0 (kin3[:, 1], ONE native frame = 0.1 s backward difference),
a = Savitzky-Golay order 2 over +-0.5 s at the present tick, perp on [1, v0, kappa0] (the two other kin3 entries the model reads), episode-cluster bootstrap B 2,000 seed 0 via the tip's taniteval ci.py
(stat_cluster_bootstrap). Everything else is a sensitivity and decides nothing. Output: raw/P5_RESULT.json
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import p5_lib as L

B, SEED = 2000, 0


def read_one(tab, rows, a0col, acol, kind, exact=False):
    a0, a, v0, kap = tab[a0col][rows], tab[acol][rows], tab["v0"][rows], tab["kappa0"][rows]
    eid = tab["clips"][tab["clip_idx"][rows]]
    parts = L.lam_parts(a0, a, v0, kap, kind)
    uniq, inv = np.unique(np.asarray([str(x) for x in eid]), return_inverse=True)
    G = L.gram_stats(a0, a, v0, kap, kind, inv, len(uniq))
    lam_g = L.lam_from_gram(G.sum(0))
    fast = L.fast_cluster_bootstrap(G, B, SEED)
    rec = {"point": parts["lam"], "point_from_gram": float(lam_g), "lo": fast["lo"], "hi": fast["hi"], "boot_sd": fast["sd"], "n_windows": parts["n"], "n_episodes": int(len(uniq)),
           "var_a0_perp": parts["var_a0_perp"], "var_e_perp": parts["var_e_perp"], "var_a_perp": parts["var_a_perp"], "corr_e_a_perp": parts["corr_e_a_perp"],
           "slope_a_on_a0_perp": parts["slope_a_on_a0_perp"], "estimator": "episode_cluster_bootstrap (fast Gram form, ci.py's resampling draws)", "n_boot": B, "seed": SEED}
    return rec


def main():
    t0 = time.time()
    L.assert_card_frozen()
    ci = L.load_ci()
    tab = {k: v for k, v in np.load(L.HERE + "/raw/window_table_band.npz").items()}
    gate = json.load(open(L.HERE + "/raw/P5_GATE.json"))
    win = json.load(open(L.HERE + "/raw/P5_WINDOWS.json"))
    rows = np.load(L.HERE + "/raw/p5_sample_rows.npy")
    ids_txt = open(L.HERE + "/raw/p5_window_ids.txt", "rb").read()
    assert L.sha256_bytes(ids_txt) == win["list_sha256"], "the window list changed"
    ok = np.isfinite(tab["a_pm0.3s"]) & np.isfinite(tab["a_pm0.5s"]) & np.isfinite(tab["a_pm0.8s"]) & np.isfinite(tab["a0_model"]) & np.isfinite(tab["kappa0"])
    allrows = np.flatnonzero(ok)
    out = {"card_sha256": L.CARD_SHA, "tip": L.TIP, "gate": gate["GATE"], "windows": win}

    # ---- PRIMARY via the tip's taniteval ci.py (stat_cluster_bootstrap), the stated estimator
    a0, a, v0, kap = tab["a0_model"][rows], tab["a_pm0.5s"][rows], tab["v0"][rows], tab["kappa0"][rows]
    eid = tab["clips"][tab["clip_idx"][rows]]
    items = list(range(len(rows)))

    def stat(it):
        ix = np.asarray(it)
        return L.lam_stat(a0[ix], a[ix], v0[ix], kap[ix], "lin")

    prim_ci = ci.stat_cluster_bootstrap(items, eid, stat, n_boot=B, seed=SEED)
    prim_fast = read_one(tab, rows, "a0_model", "a_pm0.5s", "lin")
    parts = L.lam_parts(a0, a, v0, kap, "lin")
    out["PRIMARY"] = {"definition": "a0 = model kin3[:,1] (0.1 s one-frame backward difference of the native 10 Hz speed); a = SG order 2 +-0.5 s at the present tick; perp on [1, v0, kappa0]; 20,000-window seeded sample",
                      "lambda_ci_py": prim_ci, "lambda_fast": prim_fast, "parts": parts,
                      "ci_py_vs_fast_max_abs_diff": float(max(abs(prim_ci["lo"] - prim_fast["lo"]), abs(prim_ci["hi"] - prim_fast["hi"]), abs(prim_ci["point"] - prim_fast["point"])))}
    lam, hi = prim_ci["point"], prim_ci["hi"]
    out["VERDICT"] = {**L.verdict(lam, hi), "lambda": lam, "lambda_ci": [prim_ci["lo"], prim_ci["hi"]], "gain_ratios": L.RATIO, "gain_total_ds_da0": L.GAIN_TOTAL, "base_ds_da0": L.GAIN_BASE,
                      "lambda_times_base": lam * L.GAIN_BASE,
                      "gate_passed": bool(gate["GATE"]["passes_with_the_model_s_own_definition"]), "n_episodes_ge_500": bool(prim_ci["n_episodes"] >= 500)}
    if not (out["VERDICT"]["gate_passed"] and out["VERDICT"]["n_episodes_ge_500"]):
        out["VERDICT"]["word"] = "NOT EVALUABLE"

    # ---- sensitivities (decide nothing)
    S = {}
    for nm in ("pm0.3s", "pm0.5s", "pm0.8s"):
        S["sg_" + nm] = read_one(tab, rows, "a0_model", "a_" + nm, "lin")
    S["design_p10_non_a0_features"] = read_one(tab, rows, "a0_model", "a_pm0.5s", "p10")
    S["MUTANT_no_residualisation"] = read_one(tab, rows, "a0_model", "a_pm0.5s", "none")
    S["all_valid_inband_windows"] = read_one(tab, allrows, "a0_model", "a_pm0.5s", "lin")
    S["card_prose_a0_0.2s_NOT_the_model_input"] = read_one(tab, rows, "a0_lag2", "a_pm0.5s", "lin")
    S["MUTANT_forward_difference_a0"] = read_one(tab, rows, "a0_fwd", "a_pm0.5s", "lin")
    for lo_, hi_ in ((3.0, 5.5), (5.5, 8.0)):
        r = rows[(tab["v0"][rows] >= lo_) & (tab["v0"][rows] < hi_)]
        S[f"v0_{lo_}_{hi_}"] = read_one(tab, r, "a0_model", "a_pm0.5s", "lin")
    # sample-seed spread: other 20,000-window draws from the same pool (point estimates only)
    sp = []
    for sd in range(1, 9):
        pk = np.sort(np.random.default_rng(sd).choice(allrows, size=20000, replace=False))
        sp.append(L.lam_stat(tab["a0_model"][pk], tab["a_pm0.5s"][pk], tab["v0"][pk], tab["kappa0"][pk], "lin"))
    S["sample_seed_spread_seeds_1_to_8_points"] = {"points": [float(x) for x in sp], "min": float(min(sp)), "max": float(max(sp)), "sd": float(np.std(sp))}
    for k, r in S.items():
        if isinstance(r, dict) and "point" in r:
            r["verdict_if_it_were_primary"] = L.verdict(r["point"], r["hi"])["word"]
    out["SENSITIVITIES_decide_nothing"] = S

    # ---- descriptive companions (decide nothing)
    e = tab["a0_model"][rows] - tab["a_pm0.5s"][rows]
    out["DESCRIPTIVE"] = {"a0_sd": float(a0.std()), "a_sd": float(a.std()), "e_sd": float(e.std()), "e_mean": float(e.mean()),
                          "corr_a0_a": float(np.corrcoef(a0, a)[0, 1]), "slope_of_a_on_a0_raw": float(np.cov(a0, a)[0, 1] / a0.var()),
                          "e_sd_over_a0_sd": float(e.std() / a0.std()), "implied_total_ds_da0_if_lambda_times_1.899": float(lam * L.GAIN_BASE),
                          "note": "slope_of_a_on_a0_raw is the best-linear-predictor weight of a on a0 (it equals lambda only if e is uncorrelated with a)."}
    out["wall_s"] = time.time() - t0
    json.dump(out, open(L.HERE + "/raw/P5_RESULT.json", "w"), indent=1)
    print(json.dumps({"PRIMARY": {k: out["PRIMARY"][k] for k in ("lambda_ci_py", "ci_py_vs_fast_max_abs_diff")}, "VERDICT": out["VERDICT"]}, indent=1))
    for k, r in S.items():
        if "point" in r:
            print(f"{k:50s} lam {r['point']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] n {r['n_windows']} -> {r['verdict_if_it_were_primary']}")
        else:
            print(k, r)


if __name__ == "__main__":
    main()
