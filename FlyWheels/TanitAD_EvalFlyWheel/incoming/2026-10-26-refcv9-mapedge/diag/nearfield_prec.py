import numpy as np, os
from scipy.ndimage import binary_dilation, distance_transform_edt
R = "F:/refcv9_mapsel/rast_s0"
fs = sorted(os.listdir(R))[::6]
bands = [(0, 5), (5, 10), (10, 20), (20, 40)]
c0, c1 = 270, 330
acc = {b: np.zeros(6) for b in bands}
gtcls = {b: np.zeros(9) for b in bands}
n = 0
for f in fs:
    z = np.load(os.path.join(R, f))
    if "gt" not in z.files:
        continue
    n += 1
    gt, psw = z["gt"], z["psw"]
    sw = gt == 7
    near_sw = binary_dilation(sw, iterations=5)          # within 0.5 m of a GT sidewalk cell
    for (a, b) in bands:
        sl = (slice(a * 10, b * 10), slice(c0, c1))
        p, g, seen, ns = psw[sl] >= 128, sw[sl], gt[sl] != 255, near_sw[sl]
        p = p & seen
        acc[(a, b)] += [p.sum(), (p & g).sum(), (p & ns).sum(), g.sum(), (g & p).sum(), seen.sum()]
        cls = gt[sl][p]
        for c in range(8):
            gtcls[(a, b)][c] += (cls == c).sum()
print("windows", n, "corridor |y|<3 m, own psw>=0.5 on GT-seen cells")
print("band | own flag rate | precision exact | precision within 0.5 m | recall | GT class of own flags (0 nocls,1 drv,2 lane,3 xwalk,4 arrow,5 edge,6 hatch,7 sw)")
for k, v in acc.items():
    P, TP, TPn, G_, TPr, S_ = v
    d = gtcls[k]; d = d[:8] / max(d[:8].sum(), 1)
    print(f"{k[0]:>2}-{k[1]:<3}| {P / S_:.4f} | {TP / max(P, 1):.3f} | {TPn / max(P, 1):.3f} | {TPr / max(G_, 1):.3f} | " + " ".join(f"{x:.2f}" for x in d))
