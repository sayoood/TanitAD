import sys, numpy as np
sys.path.insert(0, "C:/Users/Admin/freept/raw/2026-10-17-free-point-navtest/code")
import fp_picks as P
AT = P.load_attrib()
c3 = np.load("C:/fp/PK/raw/2026-10-11-c3-selection-headroom/table.npz")
lg = np.asarray(c3["logits"], np.float64)
base = np.asarray(AT.picks(lg))
print("C3 picks vs stored planner picks agree:", float((base == c3["pick"]).mean()))
rng = np.random.default_rng(1)
for sd in (3e-4, 1e-3, 3e-3):
    fl = []
    for r in range(20):
        n = lg + rng.normal(scale=sd, size=lg.shape)
        fl.append(float((np.asarray(AT.picks(n)) != base).mean()))
    print(f"logit noise sd {sd:g}: flip rate mean {np.mean(fl):.4f} max {np.max(fl):.4f}")

# margins of flips: sel-score difference between the NEW pick and the ORIGINAL pick, evaluated on the ORIGINAL logits
def sel(lgts):
    p = 0.5 * (1.0 + np.tanh(0.5 * lgts))
    return p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12 * p[..., 6]
s0 = sel(lg)
ar = np.arange(len(lg))
for sd in (1e-3, 3e-3):
    m = []
    for r in range(20):
        n = lg + rng.normal(scale=sd, size=lg.shape)
        pn = np.asarray(AT.picks(n))
        fl = pn != base
        m += list(s0[ar, base][fl] - s0[ar, pn][fl])        # how much worse the flipped pick is, on the original logits
    m = np.array(m)
    print(f"sd {sd:g}: flips {len(m)}, margin quantiles 50/90/99/max: {np.quantile(m,[.5,.9,.99]).round(4)} {m.max():.4f}")
