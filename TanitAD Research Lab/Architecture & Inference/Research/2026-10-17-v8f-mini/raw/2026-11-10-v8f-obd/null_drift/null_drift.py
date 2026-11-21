import json, sys, math
a = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
b = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
A = {r["step"]: r for r in a if "step" in r}; B = {r["step"]: r for r in b if "step" in r}
steps = sorted(set(A) & set(B)); keys = ["grad_norm", "ckpt", "goal", "lat", "lon", "cons", "ade", "fde"]
print("n_steps_a", len(A), "n_steps_b", len(B), "matched", len(steps), "first", steps[0], "last", steps[-1])
exact = [s for s in steps if all(A[s].get(k) == B[s].get(k) for k in keys)]
print("steps with all 8 keys bit-equal:", len(exact))
for s in [steps[0], 100, 150, 500, 1000, 3000, 6000, steps[-1]]:
    if s in A and s in B:
        print(s, " ".join(f"{k}={A[s][k]:.6g}/{B[s][k]:.6g}" for k in ["grad_norm", "ade", "lon"]))
for k in ["ade", "lon", "grad_norm"]:
    rel = [abs(A[s][k] - B[s][k]) / max(abs(B[s][k]), 1e-9) for s in steps if k in A[s] and k in B[s]]
    q = sorted(rel); print(k, "rel diff median %.3g p90 %.3g max %.3g" % (q[len(q)//2], q[int(.9*len(q))], q[-1]))
la, lb = a[-1], b[-1]; print("last rows:", {k: la.get(k) for k in ("step", "ade", "fde")}, {k: lb.get(k) for k in ("step", "ade", "fde")})
