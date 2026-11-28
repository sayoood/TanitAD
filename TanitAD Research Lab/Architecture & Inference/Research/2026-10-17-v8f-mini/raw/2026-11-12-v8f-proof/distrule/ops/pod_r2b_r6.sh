cd /workspace/v8proof || exit 1
CUDA_VISIBLE_DEVICES=-1 nice -n 19 /workspace/venv/bin/python - <<'PY'
import numpy as np, os, glob
root = "runs/distrule/t1"
arms = sorted(a for a in os.listdir(root) if os.path.exists(os.path.join(root, a, "ticks", "rule_log_%s.npz" % a)))
tot_lift = 0
for a in arms:
    z = np.load(os.path.join(root, a, "ticks", "rule_log_%s.npz" % a), allow_pickle=True)
    lift = int(z["lift"].sum()); tot_lift += lift if str(z["mode"]) != "push" else 0
    print("R2B %-10s mode=%-5s floor=%s ticks=%d lift_sum=%d changed=%.4f active=%.4f present=%.4f sat_exec_sum=%d src_bad=%d" % (
        a, str(z["mode"]), str(z["floor"]), z["ci"].size, lift, z["changed"].mean(), z["active"].mean(), z["present"].mean(), int(z["sat_exec"].sum()), int(z["src_bad"].sum())))
print("R2B TOTAL lift over all arms except push:", tot_lift)
base = {"LSRH": "LS", "LSHRH": "LSH", "LSDRH": "LSD"}
for a in arms:
    pre, s = a.rsplit("_", 1)
    if pre not in base: continue
    b = base[pre] + "_" + s
    z = np.load(os.path.join(root, a, "ticks", "rule_log_%s.npz" % a), allow_pickle=True)
    st = np.load("probe/states/%s_eval.npz" % b, allow_pickle=True)
    ld = np.load("probe/states/%s_eval_lead.npz" % b, allow_pickle=True)
    keys = st.files
    ck = [k for k in keys if k in ("clip_idx", "clip", "ci")]; tk = [k for k in keys if k in ("t", "frame", "ti")]
    if not ck or not tk:
        print("R6 %s: state keys %s" % (a, keys)); continue
    ci, t = st[ck[0]].astype(np.int64), st[tk[0]].astype(np.int64)
    idx = {(int(c), int(x)): i for i, (c, x) in enumerate(zip(ci, t))}
    rows = np.array([idx.get((int(c), int(j)), -1) for c, j in zip(z["ci"], z["j"])])
    ok = rows >= 0
    hh = np.zeros(rows.size, bool); hh[ok] = ld["has_history"][rows[ok]]
    for tick in (0, 1, 2):
        m = ok & hh & (z["tick"] == tick)
        if not m.any(): print("R6 %s tick %d: no rows" % (a, tick)); continue
        r = rows[m]
        dg = np.abs(z["g_src"][m] - ld["gap_m"][r]); dc = np.abs(z["c_src"][m] - ld["closing_ms"][r])
        p = 1.0 / (1.0 + np.exp(-ld["presence_logit"][r].astype(np.float64))); dp = np.abs(z["p_src"][m] - p)
        print("R6 %-9s vs %-6s tick %d n=%d max_abs gap=%.3g closing=%.3g presence=%.3g | share within 1e-4: gap %.4f closing %.4f" % (
            a, b, tick, int(m.sum()), np.nanmax(dg), np.nanmax(dc), np.nanmax(dp), float((dg <= 1e-4).mean()), float((dc <= 1e-4).mean())))
PY
