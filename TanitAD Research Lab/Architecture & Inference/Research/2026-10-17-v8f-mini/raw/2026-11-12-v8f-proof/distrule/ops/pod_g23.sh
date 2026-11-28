cd /workspace/v8proof || exit 1
# pod_g23.sh : controls g2 and g3 of card PREREG_V8F_PROOF_ADD30 (with note N1) and q2 and q3 of card PREREG_V8F_PROOF_ADD31, from the rule logs.
#   g2 / q2 : no plan moved ahead (lift 0) in any rule log of the two cards
#   g3      : at the first tick, gap used in a margin run == max(gap of the head-fed run - M, 0) within 1e-5 m (note N1)
#   q3      : at the first tick, gap used in a QC run == the map applied to the gap of the head-fed run, within 1e-4 m
# Also a reported table per run: share of ticks present, active, changed, and the saturated executed steps. CPU only, reads files, starts nothing else.
CUDA_VISIBLE_DEVICES=-1 nice -n 19 /workspace/venv/bin/python - <<'PY'
import numpy as np, os, sys
sys.path.insert(0, "distrule")
import qcal_map as QM
root = "runs/distrule/t1"
def log(a):
    p = os.path.join(root, a, "ticks", "rule_log_%s.npz" % a)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None
def key0(z):
    m = z["tick"] == 0
    return {(int(c), int(t)): i for i, (c, t, mm) in enumerate(zip(z["ci"], z["t0"], m)) if mm}
arms = {}
for fn, kind in (("distrule/margin_arms.txt", "m"), ("distrule/qcal_arms.txt", "q")):
    for ln in open(fn, encoding="utf-8"):
        w = ln.split()
        if len(w) == 5 and not ln.startswith("#"):
            arms[w[0]] = (kind, w[2], float(w[3]), w[4])
arms["LSM0_s0"] = ("m", "LS_s0", 0.5, "0")
arms["LSQ0_s0"] = ("q", "LS_s0", 0.5, "qcal/identity.json")
tot_lift, n_logs, g3_bad, q3_bad = 0, 0, 0, 0
for a in sorted(arms):
    kind, base, P, x = arms[a]
    z = log(a)
    if z is None:
        print("G23 %-10s NO RULE LOG" % a); continue
    n_logs += 1
    lift = int(z["lift"].sum()); tot_lift += lift
    b, s = base.split("_")
    H = log("%sRH_%s" % (b, s))
    line = "G23 %-10s P %.2f %s ticks=%d lift_sum=%d present=%.4f active=%.4f changed=%.4f sat_exec_sum=%d" % (
        a, P, ("M %s" % x) if kind == "m" else ("map %s" % x), z["ci"].size, lift, z["present"].mean(), z["active"].mean(), z["changed"].mean(), int(z["sat_exec"].sum()))
    if H is None:
        print(line + " | no head-fed log"); continue
    ka, kh = key0(z), key0(H)
    com = sorted(set(ka) & set(kh))
    ia = np.array([ka[k] for k in com]); ih = np.array([kh[k] for k in com])
    both = z["present"][ia].astype(bool) & H["present"][ih].astype(bool) & np.isfinite(z["g_src"][ia]) & np.isfinite(H["g_src"][ih])
    gh = H["g_src"][ih][both].astype(np.float64); ga = z["g_src"][ia][both].astype(np.float64)
    if kind == "m":
        M = float(x)
        exp = np.maximum(gh - M, 0.0) if M != 0.0 else gh
        d = float(np.max(np.abs(ga - exp))) if both.any() else float("nan")
        ok = bool(both.any() and d <= 1e-5); g3_bad += 0 if ok else 1
        print(line + " | g3 n=%d max_abs=%.3g %s" % (int(both.sum()), d, "PASS" if ok else "FAIL"))
    else:
        xs, ys = QM.load_map(os.path.join("distrule", x))
        exp = QM.apply_map(gh, xs, ys)
        d = float(np.max(np.abs(ga - exp))) if both.any() else float("nan")
        ok = bool(both.any() and d <= 1e-4); q3_bad += 0 if ok else 1
        print(line + " | q3 n=%d max_abs=%.3g mean shift %.2f m %s" % (int(both.sum()), d, float(np.mean(gh - ga)), "PASS" if ok else "FAIL"))
print("G23 TOTAL: logs %d, lift over all %d (g2/q2 %s), g3 failing runs %d, q3 failing runs %d" % (n_logs, tot_lift, "PASS" if tot_lift == 0 else "FAIL", g3_bad, q3_bad))
PY
