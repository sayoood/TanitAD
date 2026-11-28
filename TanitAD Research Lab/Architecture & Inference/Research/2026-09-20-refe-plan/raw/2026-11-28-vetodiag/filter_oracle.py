"""FILTER-ORACLE (MM 2026-11-28): DESCRIPTIVE ceiling of a perfect per-proposal CHECKER applied to EVERY token, on the C1 navtest
harness-exact table (T1, banked). k = argmax C1 agg32 over proposals the checker passes (none pass -> keep the ship pick).
  F_dac : checker = the harness's own DAC (dac > 0)  -- realisable at inference only if an HD map is an admissible input.
  F_nc  : checker = the harness's NC  (nc > 0)       -- NOT realisable: NC scores the logged FUTURE agents (privileged).
  F_both: both.
Reuses vetodiag.py's loaders, agg32 and Boot (copied from eval/p5c/p5c_read.py). Not a result; decides only what to ask the PI."""
import json, sys
import numpy as np

sys.argv = [sys.argv[0]]
src = open("E:/Projects/TanitAD-artifacts/vetodiag/vetodiag.py", encoding="utf-8").read()
cut = src.index("if ok1 != N")
exec(compile(src[:cut], "vetodiag_head", "exec"))          # Boot, agg32, rows, tok, kship, S, T, ar, a, pd_ship, logs
assert ok1 == N and abs(mean_ship - 92.5548) <= 5e-5
B = Boot(logs)
TP = {m: np.array([rows[t]["tp"][m] for t in tok], np.float64) for m in ("nc", "dac")}   # tp = the PLANNED proposal scored untracked (what an inference-time checker sees)
res = {"checks": {"argmax_match": ok1, "mean_ship_pdms": mean_ship}, "tier": "T1 banked harness-exact table (navtest), DESCRIPTIVE ceiling"}
for name, okm in (("F_dac", T["dac"] > 0), ("F_nc", T["nc"] > 0), ("F_both", (T["dac"] > 0) & (T["nc"] > 0)),
                  ("F_dac_planned", TP["dac"] > 0)):    # realisable form: check the planned proposal, score the tracked execution
    am = np.where(okm, a, -np.inf)
    k = am.argmax(1)
    none = ~okm.any(1)
    k[none] = kship[none]
    sw = k != kship
    r = {"n_switched": int(sw.sum()), "n_no_passing": int(none.sum())}
    for m in ("pdms", "nc", "dac", "ttc", "ep", "c"):
        r[m] = B.ci((T[m][ar, k] - T[m][ar, kship]) * 100)
    r["n_switched_pdms_worse"] = int((sw & (T["pdms"][ar, k] < pd_ship)).sum())
    r["n_switched_pdms_better"] = int((sw & (T["pdms"][ar, k] > pd_ship)).sum())
    res[name] = r
    print(name, "switched", r["n_switched"], "better/worse", r["n_switched_pdms_better"], r["n_switched_pdms_worse"],
          "dPDMS %.3f [%.3f, %.3f]" % (r["pdms"]["point"], r["pdms"]["lo"], r["pdms"]["hi"]), "dEP %.3f" % r["ep"]["point"], "dDAC %.3f" % r["dac"]["point"])
json.dump(res, open("E:/Projects/TanitAD-artifacts/vetodiag/FILTER_ORACLE.json", "w"), indent=1)
