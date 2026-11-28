cd /workspace/v8proof || exit 1
# pod_z12.sh : controls z1 and z2 of card PREREG_V8F_PROOF_ADD33 (NOCLOSE), from the rule logs.
#   z1 : no plan moved ahead (lift 0) in any rule log of the card
#   z2 : the closing speed the rule used is exactly 0 at every tick, and the gap the rule used at the first tick equals that of the partner run exactly
#        (partner: <BASE>RH for <BASE>RZ, <BASE>QC for <BASE>QZ)
# CPU only, reads files, starts nothing else.
CUDA_VISIBLE_DEVICES=-1 nice -n 19 /workspace/venv/bin/python - <<'PY'
import numpy as np, os
root = "runs/distrule/t1"
def log(a):
    p = os.path.join(root, a, "ticks", "rule_log_%s.npz" % a)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None
def key0(z):
    m = z["tick"] == 0
    return {(int(c), int(t)): i for i, (c, t, mm) in enumerate(zip(z["ci"], z["t0"], m)) if mm}
tot, bad, n = 0, 0, 0
for b in ("LS", "LSD"):
    for s in ("s0", "s1"):
        for code, partner in (("RZ", "RH"), ("QZ", "QC")):
            a, pa = "%s%s_%s" % (b, code, s), "%s%s_%s" % (b, partner, s)
            z, zp = log(a), log(pa)
            if z is None:
                print("Z12 %-10s NO RULE LOG" % a); bad += 1; continue
            n += 1
            lift = int(z["lift"].sum()); tot += lift
            cmax = float(np.nanmax(np.abs(z["c_src"]))) if z["c_src"].size else float("nan")
            line = "Z12 %-10s ticks=%d lift_sum=%d max_abs closing used=%.3g present=%.4f changed=%.4f sat_exec_sum=%d" % (a, z["ci"].size, lift, cmax, z["present"].mean(), z["changed"].mean(), int(z["sat_exec"].sum()))
            if zp is None:
                print(line + " | partner %s has no log" % pa); bad += 1; continue
            ka, kp = key0(z), key0(zp)
            com = sorted(set(ka) & set(kp))
            ia = np.array([ka[k] for k in com]); ip = np.array([kp[k] for k in com])
            fin = np.isfinite(z["g_src"][ia]) & np.isfinite(zp["g_src"][ip])
            d = float(np.max(np.abs(z["g_src"][ia][fin] - zp["g_src"][ip][fin]))) if fin.any() else float("nan")
            same_p = bool(np.array_equal(z["present"][ia].astype(bool), zp["present"][ip].astype(bool)))
            ok = bool(cmax == 0.0 and fin.any() and d == 0.0)
            bad += 0 if ok else 1
            print(line + " | first tick vs %s: n=%d max_abs gap diff=%.3g presence equal=%s z2 %s" % (pa, int(fin.sum()), d, same_p, "PASS" if ok else "FAIL"))
print("Z12 TOTAL: logs %d, lift over all %d (z1 %s), runs failing z2 or missing %d" % (n, tot, "PASS" if tot == 0 else "FAIL", bad))
PY
