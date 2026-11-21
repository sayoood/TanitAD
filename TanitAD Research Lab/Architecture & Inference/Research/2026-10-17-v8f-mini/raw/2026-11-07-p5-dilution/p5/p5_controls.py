"""p5_controls: the three controls of the P5 read, numbers for the report (test_p5.py asserts them). Output: raw/P5_CONTROLS.json
 1. a synthetic series with a known noise level returns its known lambda (population value from the noise coefficient vectors, no simulation);
 2. replacing a0 by a gives lambda = 1;
 3. deliberate-regression mutants change the number and are caught: drop the residualisation, forward-difference a0, 0.2 s a0, wrong SG width -- on the synthetic world (against the known value) and on the real sample.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import p5_lib as L

P = dict(sigma=0.06, alpha=0.5, sigma_u=0.8)


def main():
    W = L.synth_world(n_ep=20000, seed=0, **P)
    EXP = L.synth_expected_lambda(m=5, **P)
    out = {"synthetic_world": {**W["params"], "description": "one window per episode; true acceleration a_c = alpha (v_s - 5.5) + u constant; speed = v_s + a_c 0.1 (k - 15) + white noise sigma; kappa independent",
                               "expected": EXP}}

    def boot(r, kind="lin"):
        g = L.gram_stats(r["a0"], r["a"], W["v0"], W["kappa0"], kind, W["eid"], len(W["eid"]))
        return L.fast_cluster_bootstrap(g, 2000, 0)

    r = L.synth_read(W, "backward")
    b = boot(r)
    out["C1_known_noise"] = {"expected_lambda": EXP["lambda"], "observed": r["lam"], "ci": [b["lo"], b["hi"]], "boot_sd": b["sd"], "abs_error": abs(r["lam"] - EXP["lambda"]),
                             "expected_covered_by_95_ci": bool(b["lo"] <= EXP["lambda"] <= b["hi"]), "tolerance_used_in_test": 0.015}
    ra = L.synth_read(W, "a_itself")
    out["C2_a0_replaced_by_a"] = {"synthetic_lambda": ra["lam"], "real_sample_lambda": None}
    tab = np.load(L.HERE + "/raw/window_table_band.npz")
    rows = np.load(L.HERE + "/raw/p5_sample_rows.npy")
    ra_real = L.lam_stat(tab["a_pm0.5s"][rows], tab["a_pm0.5s"][rows], tab["v0"][rows], tab["kappa0"][rows], "lin")
    out["C2_a0_replaced_by_a"]["real_sample_lambda"] = ra_real
    mut = {}
    for nm, kw in (("no_residualisation", dict(a0_kind="backward", design_kind="none")), ("forward_difference_a0", dict(a0_kind="forward")), ("a0_0.2s", dict(a0_kind="lag2")),
                   ("sg_pm0.3s_instead_of_pm0.5s", dict(a0_kind="backward", m=3))):
        rr = L.synth_read(W, **kw)
        mut[nm] = {"synthetic_lambda": rr["lam"], "shift_vs_correct": rr["lam"] - r["lam"], "miss_vs_known": abs(rr["lam"] - EXP["lambda"]),
                   "caught_by_known_lambda_check": bool(abs(rr["lam"] - EXP["lambda"]) > 0.015)}
    gm = L.a0_backward(W["v_flat"], W["k0"], 1)
    mut["forward_difference_a0"]["caught_by_gate"] = bool(not L.gate_compare(gm, L.a0_forward(W["v_flat"], W["k0"]))["pass"])
    mut["a0_0.2s"]["caught_by_gate"] = bool(not L.gate_compare(gm, L.a0_backward(W["v_flat"], W["k0"], 2))["pass"])
    out["C3_mutants_synthetic"] = mut
    base = L.lam_stat(tab["a0_model"][rows], tab["a_pm0.5s"][rows], tab["v0"][rows], tab["kappa0"][rows], "lin")
    real = {"correct": base}
    real["no_residualisation"] = L.lam_stat(tab["a0_model"][rows], tab["a_pm0.5s"][rows], tab["v0"][rows], tab["kappa0"][rows], "none")
    real["forward_difference_a0"] = L.lam_stat(tab["a0_fwd"][rows], tab["a_pm0.5s"][rows], tab["v0"][rows], tab["kappa0"][rows], "lin")
    real["a0_0.2s"] = L.lam_stat(tab["a0_lag2"][rows], tab["a_pm0.5s"][rows], tab["v0"][rows], tab["kappa0"][rows], "lin")
    real["sg_pm0.3s_instead_of_pm0.5s"] = L.lam_stat(tab["a0_model"][rows], tab["a_pm0.3s"][rows], tab["v0"][rows], tab["kappa0"][rows], "lin")
    out["C3_mutants_real_sample"] = {k: {"lambda": v, "shift_vs_correct": v - base} for k, v in real.items()}
    out["C3_mutants_real_sample"]["gate_on_real_table_max_abs_vs_model_a0"] = {
        "reimplementation_lag1": L.gate_compare(tab["a0_model"], tab["a0_indep"]), "forward_mutant": L.gate_compare(tab["a0_model"], tab["a0_fwd"]), "lag2_prose": L.gate_compare(tab["a0_model"], tab["a0_lag2"])}
    out["verdict_unchanged_by_every_mutant_on_real_data"] = bool(all(not L.verdict(v, v + 0.002)["word"] == "HOLDS" for v in real.values()))
    json.dump(out, open(L.HERE + "/raw/P5_CONTROLS.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
