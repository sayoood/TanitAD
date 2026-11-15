cd /workspace/v8proof || exit 1
# pod_m3.sh : control m3 of card PREREG_V8F_PROOF_ADD29 (no plan moved ahead in any mixed-source rule log) and a reported table (names nothing):
# how the head parts differ from the label parts at the ticks of the rollouts (presence misses and false alarms, gap and closing-speed error).
# CPU only, reads files, starts nothing else.
CUDA_VISIBLE_DEVICES=-1 nice -n 19 /workspace/venv/bin/python - <<'PY'
import numpy as np, os
root = "runs/distrule/t1"
def log(a):
    p = os.path.join(root, a, "ticks", "rule_log_%s.npz" % a)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None
arms = sorted(a for a in os.listdir(root) if any(a.split("_")[0].endswith(x) for x in ("XP", "XG", "XC", "XZ", "XM1", "XM2")))
tot = 0
for a in arms:
    z = log(a)
    if z is None:
        print("M3 %-10s NO RULE LOG" % a); continue
    lift = int(z["lift"].sum()); tot += lift
    print("M3 %-10s ticks=%d lift_sum=%d changed=%.4f active=%.4f present=%.4f sat_exec_sum=%d src_bad=%d" % (
        a, z["ci"].size, lift, z["changed"].mean(), z["active"].mean(), z["present"].mean(), int(z["sat_exec"].sum()), int(z["src_bad"].sum())))
print("M3 TOTAL lift over %d mixed-source logs: %d" % (len(arms), tot))
def key(z):
    return {(int(c), int(t), int(k)): i for i, (c, t, k) in enumerate(zip(z["ci"], z["t0"], z["tick"]))}
for base in ("LS", "LSD"):
    for s in ("0", "1"):
        L = log("%sRL_s%s" % (base, s)); H = log("%sRH_s%s" % (base, s))
        if L is None or H is None:
            print("PART %s_s%s: label-fed or head-fed log missing" % (base, s)); continue
        kl, kh = key(L), key(H)
        com = sorted(set(kl) & set(kh))
        il = np.array([kl[k] for k in com]); ih = np.array([kh[k] for k in com])
        pl = L["present"][il].astype(bool); ph = H["present"][ih].astype(bool)
        both = pl & ph
        dg = H["g_src"][ih][both] - L["g_src"][il][both]; dc = H["c_src"][ih][both] - L["c_src"][il][both]
        fin = np.isfinite(dg) & np.isfinite(dc)
        print("PART %-4s s%s ticks=%d label_present=%.4f head_present=%.4f miss(label yes, head no)=%.4f of label-present, false_alarm(head yes, label no)=%.4f of label-absent | both n=%d gap err mean %+.2f m rmse %.2f m, p10 %+.2f p90 %+.2f | closing err mean %+.2f rmse %.2f m/s" % (
            base, s, len(com), pl.mean(), ph.mean(), float((pl & ~ph).sum()) / max(1, int(pl.sum())), float((ph & ~pl).sum()) / max(1, int((~pl).sum())), int(fin.sum()),
            float(np.mean(dg[fin])), float(np.sqrt(np.mean(dg[fin] ** 2))), float(np.percentile(dg[fin], 10)), float(np.percentile(dg[fin], 90)),
            float(np.mean(dc[fin])), float(np.sqrt(np.mean(dc[fin] ** 2)))))
        act = L["active"][il].astype(bool)
        if act.any():
            print("PART %-4s s%s at ticks where the label-fed rule is active (n=%d): head_present=%.4f; gap err mean %+.2f m rmse %.2f m (n=%d)" % (
                base, s, int(act.sum()), ph[act].mean(), float(np.nanmean((H["g_src"][ih] - L["g_src"][il])[act & both])), float(np.sqrt(np.nanmean(((H["g_src"][ih] - L["g_src"][il])[act & both]) ** 2))), int((act & both).sum())))
PY
