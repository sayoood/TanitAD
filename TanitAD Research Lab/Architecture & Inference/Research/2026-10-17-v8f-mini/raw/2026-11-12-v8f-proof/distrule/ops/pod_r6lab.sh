cd /workspace/v8proof || exit 1
CUDA_VISIBLE_DEVICES=-1 nice -n 19 /workspace/venv/bin/python - <<'PY'
import numpy as np
def r2(y, p):
    m = np.isfinite(y) & np.isfinite(p); y, p = y[m], p[m]
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(), int(m.sum()), float(np.sqrt(((y - p) ** 2).mean())), float((p - y).mean())
for arm, base in (("LS_s0", "LSR"), ("LSD_s0", "LSDR"), ("LSH_s1", "LSHR")):
    s = arm.split("_")[1]
    H = np.load("runs/distrule/t1/%sH_%s/ticks/rule_log_%sH_%s.npz" % (base, s, base, s), allow_pickle=True)
    L = np.load("runs/distrule/t1/%sL_%s/ticks/rule_log_%sL_%s.npz" % (base, s, base, s), allow_pickle=True)
    same = bool((H["ci"] == L["ci"]).all() and (H["j"] == L["j"]).all() and (H["tick"] == L["tick"]).all())
    st = np.load("probe/states/%s_eval.npz" % arm, allow_pickle=True); ld = np.load("probe/states/%s_eval_lead.npz" % arm, allow_pickle=True)
    idx = {(int(c), int(x)): i for i, (c, x) in enumerate(zip(st["clip_idx"], st["t"]))}
    rows = np.array([idx.get((int(c), int(j)), -1) for c, j in zip(H["ci"], H["j"])])
    t0gap = np.where(rows >= 0, ld["gap_m"][np.maximum(rows, 0)], np.nan); t0gap[(rows >= 0) & ~ld["has_history"][np.maximum(rows, 0)]] = np.nan
    t0p = 1 / (1 + np.exp(-ld["presence_logit"][np.maximum(rows, 0)].astype(float)))
    for tick in (0, 1, 2):
        m = L["present"] & (H["tick"] == tick)
        a = r2(L["g_src"][m], H["g_src"][m]); b = r2(L["g_src"][m], t0gap[m])
        mm = (H["tick"] == tick)
        print("LAB %s tick %d same_ticks=%s | rollout head vs label gap: R2 %+.3f n=%d rmse %.2f bias %+.2f | banked T0 head vs label gap: R2 %+.3f n=%d rmse %.2f bias %+.2f | presence prob mean on label-present / absent: rollout %.3f / %.3f, banked %.3f / %.3f" % (
            arm, tick, same, a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
            H["p_src"][mm & L["present"]].mean(), H["p_src"][mm & ~L["present"]].mean(), np.nanmean(t0p[mm & L["present"]]), np.nanmean(t0p[mm & ~L["present"]])))
PY
