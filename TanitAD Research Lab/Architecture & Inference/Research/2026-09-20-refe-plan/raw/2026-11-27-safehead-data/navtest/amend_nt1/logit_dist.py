"""MM 2026-11-27: distribution of |s @ W_C1.T + b_C1 - logits| over the navtest dump (CTRL-LOGIT failed at max 1.144e-5 vs its 1e-5 bar)."""
import glob, json, sys
import numpy as np
h = np.load("E:/Projects/TanitAD-artifacts/safehead/run1/sh_s0/head_step00000.npz")
W, b = h["init_weight"].astype(np.float32), h["init_bias"].astype(np.float32)
qs, mx, n, over, absl = [], 0.0, 0, 0, []
hist = np.zeros(8, np.int64); edges = [0, 1e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 1e-4, np.inf]
for f in sorted(glob.glob("F:/safehead_cache/navtest/*.npz")):
    z = np.load(f)
    s, L = z["s"].astype(np.float32), z["logits"].astype(np.float32)
    d = np.abs(s @ W.T + b - L).ravel()
    hist += np.histogram(d, bins=edges)[0]
    mx = max(mx, float(d.max())); n += d.size; over += int((d > 1e-5).sum())
    absl.append(float(np.abs(L).max()))
    qs.append(np.quantile(d, 0.99))
out = {"n_values": n, "max": mx, "n_over_1e-5": over, "share_over_1e-5": over / n, "p99_max_over_files": float(max(qs)),
       "hist_edges": [str(e) for e in edges], "hist": hist.tolist(), "max_abs_logit": max(absl), "files": len(qs),
       "float32_eps_at_max_logit": float(np.spacing(np.float32(max(absl))))}
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out))
