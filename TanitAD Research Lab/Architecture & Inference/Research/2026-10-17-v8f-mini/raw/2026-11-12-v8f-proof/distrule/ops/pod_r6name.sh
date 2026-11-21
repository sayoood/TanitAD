cd /workspace/v8proof || exit 1
CUDA_VISIBLE_DEVICES=-1 nice -n 19 /workspace/venv/bin/python - <<'PY'
import numpy as np, json
J = np.load("hierrep/pins/join_eval141.npz", allow_pickle=True)
print("JOIN KEYS", [(k, J[k].shape, str(J[k].dtype)) for k in J.files][:24])
names_join = None
for k in J.files:
    a = J[k]
    if a.dtype.kind in "US" and a.ndim == 1 and a.size == 141: names_join = a; print("join names key", k, a[:2])
W = json.load(open("hierrep/pins/eval_grid_windows.json"))
print("WINDOWS type", type(W).__name__, (list(W.keys())[:12] if isinstance(W, dict) else (len(W), W[0])))
st = np.load("probe/states/LS_s0_eval.npz", allow_pickle=True)
print("state clips[:2]", st["clips"][:2])
if names_join is not None:
    order_same = bool((names_join == st["clips"]).all()); print("clip order equal:", order_same)
    pos = {str(n): i for i, n in enumerate(st["clips"])}
    remap = np.array([pos.get(str(n), -1) for n in names_join])
    for arm, tag in (("LS_s0", "LSRH_s0"), ("LSD_s1", "LSDRH_s1"), ("LSH_s0", "LSHRH_s0")):
        z = np.load("runs/distrule/t1/%s/ticks/rule_log_%s.npz" % (tag, tag), allow_pickle=True)
        s2 = np.load("probe/states/%s_eval.npz" % arm, allow_pickle=True); ld = np.load("probe/states/%s_eval_lead.npz" % arm, allow_pickle=True)
        idx = {(int(c), int(x)): i for i, (c, x) in enumerate(zip(s2["clip_idx"], s2["t"]))}
        rows = np.array([idx.get((int(remap[c]), int(j)), -1) for c, j in zip(z["ci"], z["j"])])
        for tick in (0, 1, 2):
            m = (rows >= 0) & (z["tick"] == tick); m[m] &= ld["has_history"][rows[m]]
            r = rows[m]; dg = np.abs(z["g_src"][m] - ld["gap_m"][r]); dc = np.abs(z["c_src"][m] - ld["closing_ms"][r])
            dp = np.abs(z["p_src"][m] - 1 / (1 + np.exp(-ld["presence_logit"][r].astype(float))))
            print("R6N %-9s tick %d n=%d max_abs gap %.3g m closing %.3g m/s presence %.3g | within 1e-4: gap %.4f closing %.4f" % (tag, tick, int(m.sum()), dg.max(), dc.max(), dp.max(), (dg <= 1e-4).mean(), (dc <= 1e-4).mean()))
PY
