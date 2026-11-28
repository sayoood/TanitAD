"""r2_compare.py -- card R2 read of the GPU checks. Inputs (all selfcheck.json from the SAME code, same GPU, same real batch):
  N1D_a, N1D_b = two selfcheck-only runs of the N1D argv (w = 1) on the n1dw tree (the repeat);  ARMS = the N1DW arms' gate/selfcheck.json.
Per existing loss term t: repeat_diff = |N1D_a - N1D_b|; arm_diff = |N1DW - N1D_a|. PASS iff arm_diff <= repeat_diff (or both <= 1e-9 abs). The 'rb_window' term and total are excluded from the pass rule
(the total differs by design by w_traj*(w-1)*(ADE_rb/3+0.5 FDE_rb/6)); they are printed. usage: r2_compare.py OUT.json N1D_a.json N1D_b.json ARM=path ..."""
import json
import sys

out, a, b = sys.argv[1:4]
A = json.load(open(a))["live"]["terms"]
Bt = json.load(open(b))["live"]["terms"]
res = {"repeat": {k: abs(A[k] - Bt[k]) for k in A}, "arms": {}}
ok = True
for spec in sys.argv[4:]:
    name, path = spec.split("=", 1)
    T = json.load(open(path))["live"]["terms"]
    rows, bad = {}, []
    for k in A:
        if k not in T:
            bad.append(f"{k} missing")
            continue
        ad, rd = abs(T[k] - A[k]), abs(A[k] - Bt[k])
        rows[k] = {"arm_minus_N1D": ad, "repeat": rd, "ok": (ad <= rd) or (ad <= 1e-9 and rd <= 1e-9)}
        if k != "total" and not rows[k]["ok"]:
            bad.append(f"{k}: arm diff {ad:.3e} > repeat {rd:.3e}")
    extra = {k: T[k] for k in T if k not in A}
    res["arms"][name] = {"rows": rows, "extra_terms": extra, "total_N1DW": T.get("total"), "total_N1D": A.get("total"), "problems": bad}
    ok = ok and not bad
res["pass"] = ok
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: (v if k != "arms" else {n: x["problems"] for n, x in v.items()}) for k, v in res.items() if k != "repeat"}, indent=1))
sys.exit(0 if ok else 1)
