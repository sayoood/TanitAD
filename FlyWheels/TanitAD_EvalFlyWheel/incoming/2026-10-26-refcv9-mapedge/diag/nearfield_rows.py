import numpy as np, os
R = "F:/refcv9_mapsel/rast_s0"
fs = sorted(os.listdir(R))[::6]
nb = 16                                   # 0.5 m bins over 0-8 m
P = np.zeros(nb); TP = np.zeros(nb); S = np.zeros(nb); G = np.zeros(nb); M = []
for f in fs:
    z = np.load(os.path.join(R, f))
    if "gt" not in z.files:
        continue
    gt, psw = z["gt"][:80, 270:330], z["psw"][:80, 270:330].astype(np.float32)
    seen = gt != 255
    m = []
    for i in range(nb):
        sl = slice(i * 5, i * 5 + 5)
        p = (psw[sl] >= 128) & seen[sl]
        P[i] += p.sum(); TP[i] += (p & (gt[sl] == 7)).sum(); S[i] += seen[sl].sum(); G[i] += ((gt[sl] == 7) & seen[sl]).sum()
        m.append(psw[sl].mean())
    M.append(m)
M = np.array(M)
print("x_m   flag_rate  precision  GT_sw_rate  mean_psw  std_across_windows")
for i in range(nb):
    print(f"{i * 0.5:4.1f}-{i * 0.5 + 0.5:<4.1f} {P[i] / S[i]:.4f}    {TP[i] / max(P[i], 1):.3f}      {G[i] / S[i]:.4f}     {M[:, i].mean() / 255:.3f}    {M[:, i].std() / 255:.3f}")
