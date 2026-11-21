import glob, numpy as np
W = "/workspace/v8proof/integ10k/t1"
def load(arm, k):
    g, cl, v0 = [], [], []
    for f in sorted(glob.glob("%s/%s_s%d/dump_%s_s%d_m10_causal_heads/ep*.npz" % (W, arm, k, arm, k))):
        z = np.load(f); g.append(z["g"]); cl.append(z["cl"]); v0.append(z["v0"])
    return np.concatenate(g), np.concatenate(cl), np.concatenate(v0)
def speeds(tr):
    p = np.concatenate([np.zeros((len(tr), 1, 2)), tr], 1)
    return np.linalg.norm(np.diff(p, axis=1), axis=-1) / 0.2
for k in (0, 1):
    g, c10, v0 = load("I10R0", k); _, c5, _ = load("I10R0M5", k)
    sg, s10, s5 = speeds(g), speeds(c10), speeds(c5)
    idx = [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 19, 20, 24, 25, 29]
    print("seed", k, "n", len(g), "mean v0 %.2f" % v0.mean())
    print(" step     ", " ".join("%5d" % (i + 1) for i in idx))
    print(" logged   ", " ".join("%5.2f" % sg[:, i].mean() for i in idx))
    print(" m10      ", " ".join("%5.2f" % s10[:, i].mean() for i in idx))
    print(" m5       ", " ".join("%5.2f" % s5[:, i].mean() for i in idx))
    print(" m5 median", " ".join("%5.2f" % np.median(s5[:, i]) for i in idx))
    d = s5[:, -1] - sg[:, -1]
    print(" m5 last-step speed minus logged: mean %.2f median %.2f share>5 m/s %.3f share<-5 %.3f ; m10: mean %.2f" % (d.mean(), np.median(d), (d > 5).mean(), (d < -5).mean(), (s10[:, -1] - sg[:, -1]).mean()))
    print(" identical first 5 steps m5 vs m10: max abs pos diff %.2e" % np.abs(c5[:, :5] - c10[:, :5]).max())
