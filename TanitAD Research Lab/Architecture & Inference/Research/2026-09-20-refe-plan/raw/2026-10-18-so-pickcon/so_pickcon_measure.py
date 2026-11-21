#!/usr/bin/env python3
"""SO-PICKCON: the NUMBERS behind the reachability test's checks U(a), U(b), U(c) -- read from the test's OWN closures, not re-derived.
test_rl_pickcon.py proves each property as a boolean; this script runs the same closures (con_grads / hard) and prints the magnitudes the booleans rest on.
CPU only, no pod.  Run from anywhere:
  PYTHONUTF8=1 python so_pickcon_measure.py <path to the refe/ directory holding test_rl_pickcon.py + scorer_finetune.py>   ->  pickcon_measure.json next to this file
It does not edit anything.  Evidence class: MEASURED (this script's output) on the test's own tiny synthetic scorer (TE.Tiny, 6 candidates, seeded) -- a wiring measurement, NOT a driving result."""
import json, os, sys

REFE = os.path.abspath(sys.argv[1])
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickcon_measure.json")
sys.path.insert(0, REFE)
os.chdir(REFE)
import torch  # noqa: E402
import test_rl_pickcon as TP  # noqa: E402

CAP = {}


def capture(bad, name, fn):
    CAP[name] = fn                                  # keep the closure; do not run the check (the test itself is the proof, this is the magnitude)


TP.check = capture
src = open(os.path.join(REFE, "scorer_finetune.py"), encoding="utf-8").read()
TP.suite_u(src, "measure")


def cell(fn, name):
    return fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents


ua = next(f for k, f in CAP.items() if k.startswith("U_a_the_constraint_s_gradient"))
ub = next(f for k, f in CAP.items() if k.startswith("U_b_IDENTITY"))
uc = cell(next(f for k, f in CAP.items() if k.startswith("U_c_T_1e-06")), "uc")      # the check is a lambda around uc(T)
con_grads, maxdiff = cell(ua, "con_grads"), cell(ua, "maxdiff")
DEFAULT = cell(ub, "DEFAULT")
hard = cell(uc, "hard")


def gmax(r):
    return max(float(g.abs().max()) for g in r["gcon"])


res = {"tool": "so_pickcon_measure/1", "note": "tiny synthetic scorer of the test (6 candidates, seeded); a wiring magnitude, not a driving result", "a": {}, "b": {}, "c": {}}
r1, r2, r3 = con_grads(0.1), con_grads(0.02), con_grads(0.1)
den = gmax(r1)
res["a"] = {
    "max_abs_constraint_grad_T0.1": den, "max_abs_constraint_grad_T0.02": gmax(r2),
    "max_abs_diff_of_constraint_grads_0.1_vs_0.02": maxdiff(r1["gcon"], r2["gcon"]), "relative_diff_to_T0.1_max": maxdiff(r1["gcon"], r2["gcon"]) / den,
    "cB_ttc_T0.1": r1["parts"]["cB_ttc"], "cB_ttc_T0.02": r2["parts"]["cB_ttc"], "g_ttc_T0.1": r1["parts"]["g_ttc"], "g_ttc_T0.02": r2["parts"]["g_ttc"],
    "arm_A_grad_max_abs_diff_0.1_vs_0.02": maxdiff(r1["gA"], r2["gA"]),
    "twin_same_T_twice_max_abs_diff": maxdiff(r1["gcon"], r3["gcon"]),
}
d0, e1 = con_grads(DEFAULT), con_grads(0.1)
res["b"] = {
    "default_vs_explicit_0.1_max_abs_diff_of_total_gradient_B": maxdiff(d0["gB"], e1["gB"]), "default_vs_explicit_0.1_max_abs_diff_of_constraint_gradient": maxdiff(d0["gcon"], e1["gcon"]),
    "default_vs_explicit_0.1_max_abs_diff_of_gradient_A": maxdiff(d0["gA"], e1["gA"]), "loss_bits_equal": bool(d0["bits"] == e1["bits"]), "parts_equal": bool(d0["parts"] == e1["parts"]),
    "default_effective_T": d0["temp"], "explicit_T": e1["temp"],
}
z = con_grads(0.02, lam=0.0)
res["b"]["zero_control_lambda0_max_abs_constraint_grad"] = gmax(z)
d5, e5, e1b = con_grads(DEFAULT, epo_temp=0.05), con_grads(0.05, epo_temp=0.05), con_grads(0.1, epo_temp=0.05)
res["b"]["default_inherits_epo_temp_0.05"] = {"effective_T": d5["temp"], "diff_vs_explicit_0.05": maxdiff(d5["gcon"], e5["gcon"]), "diff_vs_explicit_0.1": maxdiff(d5["gcon"], e1b["gcon"])}
for T in (1e-6, 1e-8, 0.1):
    p, yA, yB, piB, ib = hard(T)
    onehot = torch.zeros_like(piB).scatter_(1, ib, 1.0)
    res["c"][f"T{T:g}"] = {"policy_minus_onehot_max_abs": float(piB.sub(onehot).abs().max()), "cA_ttc": p["cA_ttc"], "pickA_mean_ttc_label": float(yA.mean()), "cB_ttc": p["cB_ttc"], "pickB_mean_ttc_label": float(yB.mean()),
                           "g_ttc": p["g_ttc"], "g_minus_hard_pick_gap_exact": bool(p["g_ttc"] == float(yA.mean()) - float(yB.mean())), "p1gap_ttc": p["p1gap_ttc"], "wedge_g_plus_p1gap": p["g_ttc"] + p["p1gap_ttc"]}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
print("ZZPICKCON_MEASURE")
