import numpy as np, os, sys
R = "F:/refcv9_mapsel/rast_s0"
fs = sorted(os.listdir(R))[::6]
bands = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 60)]
c0, c1 = 300 - 30, 300 + 30            # |y| < 3 m
nogt = 0
acc = {b: np.zeros(8) for b in bands}
for f in fs:
    z = np.load(os.path.join(R, f))
    if "gt" not in z.files:
        nogt += 1; continue
    gt, psw, code = z["gt"], z["psw"], z["code"]
    for (a, b) in bands:
        g = gt[a * 10:b * 10, c0:c1]; p = psw[a * 10:b * 10, c0:c1] >= 128; cd = code[a * 10:b * 10, c0:c1]
        seen = g != 255
        acc[(a, b)] += [g.size, (~seen).sum(), (p & seen).sum(), (p & ~seen).sum(), ((g == 7) & seen).sum(), seen.sum(), (cd == 7).sum(), ((cd == 7) & ~seen).sum()]
print("windows sampled", len(fs), "without gt", nogt)
print("band_m | GT unseen | own psw>=.5 on GT-seen | own psw>=.5 on GT-unseen | GT sidewalk on seen | share of own flags on unseen")
for k, v in acc.items():
    n, un, ps, pu, gs, se, oc, ocu = v
    print(f"{k[0]:>2}-{k[1]:<3} | {un / n:.3f} | {ps / max(se, 1):.4f} | {pu / max(un, 1):.4f} | {gs / max(se, 1):.4f} | {pu / max(ps + pu, 1):.3f}")
