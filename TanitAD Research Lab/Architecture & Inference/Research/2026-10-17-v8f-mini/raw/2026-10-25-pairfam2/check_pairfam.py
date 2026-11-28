# Independent re-derivation of one strategic cell (ck_pos_err at 50 m, X - B) with a hand-written episode-cluster bootstrap.
import numpy as np
E = "C:/Users/Admin/box2_ship/logs/v8f_mixo/eval/"
A = np.load(E + "eval_M2b-mixo_s0.npz", allow_pickle=True); B = np.load(E + "eval_M2b_s0.npz", allow_pickle=True)
print(sorted(A.files)[:40])
ea, eb = A["eid"], B["eid"]; assert (ea == eb).all()
ca, cb = np.asarray(A["ck_pos_err"], float), np.asarray(B["ck_pos_err"], float)
print("ck shape", ca.shape)
j = 1  # 50 m column
m = np.isfinite(ca[:, j]) & np.isfinite(cb[:, j]); d = ca[m, j] - cb[m, j]; e = ea[m]
print("n", m.sum(), "episodes", len(np.unique(e)), "delta", round(d.mean(), 4), "a", round(ca[m, j].mean(), 4), "b", round(cb[m, j].mean(), 4))
u = np.unique(e); idx = {k: np.where(e == k)[0] for k in u}; rng = np.random.default_rng(12345); bs = []
for _ in range(2000):
    s = rng.choice(u, len(u), replace=True); ii = np.concatenate([idx[k] for k in s]); bs.append(d[ii].mean())
print("my 95% CI", np.round(np.percentile(bs, [2.5, 97.5]), 4))
